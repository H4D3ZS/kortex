//! Routing heat and physical memory pinning.
//!
//! Borrowed from Colibri's expert-streaming design, minus the streaming.
//! Colibri tracks which MoE experts a workload keeps routing to and
//! prefetches those; the same observation applies to a code catalog —
//! an agent working on a task hits a small, stable set of chunks over
//! and over, and those are exactly the pages that must never take a
//! disk fault.
//!
//! Two pieces:
//!
//! * [`HeatMap`] — per-chunk access counters with exponential decay, so
//!   heat reflects the *current* task and not everything since boot.
//! * [`lock_region`] — `VirtualLock` on Windows, `mlock` on Unix, to
//!   keep the hottest payload pages resident in physical RAM.
//!
//! Pinning is best-effort by design. Both syscalls are quota-limited
//! (Windows caps the process working set; Unix caps `RLIMIT_MEMLOCK`,
//! commonly at a few megabytes), so a failure is normal operation on an
//! untuned machine, not an error. A [`PinReport`] records what happened
//! and the caller carries on — an unpinned catalog is slower on a cold
//! page, never incorrect.

use std::collections::HashMap;

/// Per-chunk access frequency with exponential decay.
///
/// Decay is applied lazily: rather than walking every counter on each
/// query, the weight of a *new* hit grows over time and scores are read
/// back relative to that scale. This keeps `record` O(1) regardless of
/// how many chunks have ever been touched.
#[derive(Debug, Clone)]
pub struct HeatMap {
    counts: HashMap<u64, f64>,
    /// Weight applied to the next recorded hit. Grows by `1/decay` each
    /// generation, which is equivalent to decaying every stored count.
    current_weight: f64,
    decay: f64,
    queries: u64,
}

impl Default for HeatMap {
    fn default() -> Self {
        Self::new(0.98)
    }
}

impl HeatMap {
    /// Create a heat map. `decay` is the factor applied to accumulated
    /// heat each generation — 0.98 keeps roughly a 50-query half-life,
    /// long enough to cover a coherent task and short enough to forget
    /// the previous one.
    ///
    /// `decay` is clamped to `(0, 1]`; a value outside that range would
    /// make the weight scale diverge or collapse.
    pub fn new(decay: f64) -> Self {
        Self {
            counts: HashMap::new(),
            current_weight: 1.0,
            decay: decay.clamp(1e-6, 1.0),
            queries: 0,
        }
    }

    /// Record that a query retrieved these chunks, then advance one
    /// generation so older heat is worth proportionally less.
    pub fn record_query(&mut self, chunk_ids: impl IntoIterator<Item = u64>) {
        for id in chunk_ids {
            *self.counts.entry(id).or_insert(0.0) += self.current_weight;
        }
        self.queries += 1;

        // Advancing the weight is the dual of decaying every count.
        self.current_weight /= self.decay;

        // f64 tops out around 1e308; renormalize well before that so a
        // long-running daemon cannot drift into infinity.
        if self.current_weight > 1e150 {
            let scale = self.current_weight;
            for v in self.counts.values_mut() {
                *v /= scale;
            }
            self.current_weight = 1.0;
        }
    }

    /// Number of queries recorded.
    pub fn queries(&self) -> u64 {
        self.queries
    }

    /// Number of distinct chunks with nonzero heat.
    pub fn tracked_chunks(&self) -> usize {
        self.counts.len()
    }

    /// Heat of one chunk, normalized so the hottest chunk is 1.0.
    pub fn heat(&self, chunk_id: u64) -> f64 {
        let max = self.max_heat();
        if max <= 0.0 {
            return 0.0;
        }
        self.counts.get(&chunk_id).copied().unwrap_or(0.0) / max
    }

    fn max_heat(&self) -> f64 {
        self.counts.values().copied().fold(0.0, f64::max)
    }

    /// The `n` hottest chunk ids, hottest first, paired with heat
    /// normalized to the hottest chunk.
    ///
    /// Ties break on chunk id so the ordering is deterministic — a
    /// nondeterministic pin set would make the pinning behaviour
    /// untestable.
    pub fn hottest(&self, n: usize) -> Vec<(u64, f64)> {
        let max = self.max_heat();
        if max <= 0.0 {
            return Vec::new();
        }
        let mut all: Vec<(u64, f64)> = self
            .counts
            .iter()
            .map(|(id, c)| (*id, c / max))
            .collect();
        all.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap().then(a.0.cmp(&b.0)));
        all.truncate(n);
        all
    }

    /// Drop chunks whose normalized heat has fallen below `floor`, so
    /// the map does not grow without bound across a long session.
    pub fn prune(&mut self, floor: f64) {
        let max = self.max_heat();
        if max <= 0.0 {
            return;
        }
        self.counts.retain(|_, c| *c / max >= floor);
    }
}

/// What a pinning pass achieved.
#[derive(Debug, Clone, Default)]
pub struct PinReport {
    /// Chunks the caller asked to pin.
    pub requested: usize,
    /// Chunks whose pages are now locked.
    pub pinned: usize,
    /// Bytes locked.
    pub pinned_bytes: usize,
    /// Why pinning stopped or was refused. Empty on full success.
    pub failures: Vec<String>,
}

impl PinReport {
    /// True when every requested chunk was locked.
    pub fn complete(&self) -> bool {
        self.requested == self.pinned && self.failures.is_empty()
    }
}

/// Lock a byte range into physical RAM.
///
/// Returns `Ok(bytes_locked)` — which may exceed `len`, since both
/// syscalls operate at page granularity — or a description of why the
/// OS refused.
///
/// # Safety
///
/// `ptr` must point to `len` valid mapped bytes that outlive the lock.
/// The intended caller passes a subslice of a live `Mmap`.
pub unsafe fn lock_region(ptr: *const u8, len: usize) -> Result<usize, String> {
    if len == 0 {
        return Ok(0);
    }
    // Both APIs round to page boundaries. Align the base down and
    // extend the length to match, or the tail page of the range is
    // silently left unlocked.
    let page = page_size();
    let addr = ptr as usize;
    let aligned = addr & !(page - 1);
    let span = (addr - aligned) + len;
    let span = (span + page - 1) & !(page - 1);

    platform_lock(aligned as *const u8, span).map(|_| span)
}

#[cfg(windows)]
unsafe fn platform_lock(ptr: *const u8, len: usize) -> Result<(), String> {
    // Declared directly rather than pulling in the `windows` crate:
    // two symbols do not justify the dependency in a crate this small.
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn VirtualLock(addr: *const core::ffi::c_void, size: usize) -> i32;
        fn GetLastError() -> u32;
    }

    if VirtualLock(ptr as *const core::ffi::c_void, len) != 0 {
        return Ok(());
    }
    let code = GetLastError();
    // ERROR_WORKING_SET_QUOTA (1453): the process working set is too
    // small. Raising it needs SetProcessWorkingSetSize and, for large
    // amounts, the SeLockMemoryPrivilege — a deployment decision, not
    // something a library should do behind the caller's back.
    let hint = if code == 1453 {
        " (working-set quota exceeded; raise it with SetProcessWorkingSetSize \
          or pin fewer bytes)"
    } else {
        ""
    };
    Err(format!("VirtualLock failed: Windows error {code}{hint}"))
}

#[cfg(unix)]
unsafe fn platform_lock(ptr: *const u8, len: usize) -> Result<(), String> {
    if libc::mlock(ptr as *const libc::c_void, len) == 0 {
        return Ok(());
    }
    let err = std::io::Error::last_os_error();
    let hint = if err.raw_os_error() == Some(libc::ENOMEM) {
        " (RLIMIT_MEMLOCK exceeded; raise it with `ulimit -l` or pin fewer bytes)"
    } else {
        ""
    };
    Err(format!("mlock failed: {err}{hint}"))
}

#[cfg(not(any(windows, unix)))]
unsafe fn platform_lock(_ptr: *const u8, _len: usize) -> Result<(), String> {
    Err("memory pinning is not supported on this platform".to_string())
}

#[cfg(windows)]
fn page_size() -> usize {
    // 4KiB on every Windows target that matters; the value only affects
    // alignment slack, and over-aligning is harmless.
    4096
}

#[cfg(unix)]
fn page_size() -> usize {
    let v = unsafe { libc::sysconf(libc::_SC_PAGESIZE) };
    if v > 0 {
        v as usize
    } else {
        4096
    }
}

#[cfg(not(any(windows, unix)))]
fn page_size() -> usize {
    4096
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn heat_ranks_frequently_retrieved_chunks_first() {
        let mut h = HeatMap::new(0.98);
        for _ in 0..10 {
            h.record_query([1, 2]);
        }
        for _ in 0..3 {
            h.record_query([3]);
        }
        let hottest = h.hottest(3);
        assert_eq!(hottest[0].1, 1.0, "hottest chunk normalizes to 1.0");
        let ids: Vec<u64> = hottest.iter().map(|(id, _)| *id).collect();
        assert!(ids[..2].contains(&1) && ids[..2].contains(&2), "got {ids:?}");
        assert_eq!(ids[2], 3);
    }

    #[test]
    fn recent_access_outweighs_stale_access() {
        let mut h = HeatMap::new(0.5); // aggressive decay
        // Chunk 1 is hit many times, then abandoned.
        for _ in 0..5 {
            h.record_query([1]);
        }
        // Chunk 2 is hit fewer times, but recently.
        for _ in 0..3 {
            h.record_query([2]);
        }
        assert!(
            h.heat(2) > h.heat(1),
            "recent chunk 2 ({}) should outrank stale chunk 1 ({})",
            h.heat(2),
            h.heat(1)
        );
    }

    #[test]
    fn empty_map_reports_no_heat() {
        let h = HeatMap::default();
        assert!(h.hottest(5).is_empty());
        assert_eq!(h.heat(7), 0.0);
        assert_eq!(h.tracked_chunks(), 0);
    }

    #[test]
    fn hottest_is_deterministic_across_equal_heat() {
        let mut a = HeatMap::new(0.98);
        let mut b = HeatMap::new(0.98);
        for m in [&mut a, &mut b] {
            m.record_query([9, 4, 7]);
        }
        assert_eq!(a.hottest(3), b.hottest(3));
        // All equal heat, so ordering falls back to ascending id.
        assert_eq!(
            a.hottest(3).iter().map(|(i, _)| *i).collect::<Vec<_>>(),
            vec![4, 7, 9]
        );
    }

    #[test]
    fn prune_drops_cold_chunks_but_keeps_hot_ones() {
        let mut h = HeatMap::new(1.0); // no decay, so counts are raw
        for _ in 0..20 {
            h.record_query([1]);
        }
        h.record_query([2]);
        h.prune(0.5);
        assert!(h.heat(1) > 0.0);
        assert_eq!(h.heat(2), 0.0);
        assert_eq!(h.tracked_chunks(), 1);
    }

    #[test]
    fn weight_renormalization_keeps_heat_finite() {
        // Enough generations at a steep decay to blow past 1e150 and
        // trip the renormalization branch.
        let mut h = HeatMap::new(0.1);
        for i in 0..400u64 {
            h.record_query([i % 8]);
        }
        for id in 0..8u64 {
            assert!(h.heat(id).is_finite(), "heat for {id} went non-finite");
        }
        assert!(h.hottest(8).iter().all(|(_, v)| v.is_finite()));
    }

    #[test]
    fn locking_a_live_allocation_succeeds_or_reports_a_quota_error() {
        let buf = vec![7u8; 64 * 1024];
        let result = unsafe { lock_region(buf.as_ptr(), buf.len()) };
        match result {
            Ok(bytes) => assert!(bytes >= buf.len()),
            // A refusal is legitimate on an untuned machine; what must
            // not happen is a crash or a silent success.
            Err(msg) => assert!(
                msg.contains("VirtualLock") || msg.contains("mlock") || msg.contains("supported"),
                "unexpected error text: {msg}"
            ),
        }
    }

    #[test]
    fn locking_zero_bytes_is_a_no_op() {
        let buf = [0u8; 8];
        assert_eq!(unsafe { lock_region(buf.as_ptr(), 0) }.unwrap(), 0);
    }
}
