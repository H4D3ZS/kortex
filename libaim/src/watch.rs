//! Watching a workspace and feeding edits into the delta layer.
//!
//! [`start_watcher`] spawns a background thread that turns filesystem
//! events into [`LiveCatalog::ingest_file`] calls, so retrieval tracks
//! the tree without anyone calling into the delta layer by hand.
//!
//! # Never trust the event kind
//!
//! The instinct is to match on `Create` / `Modify` / `Remove` and act
//! accordingly. That breaks immediately on real editors, because most of
//! them save atomically: write `foo.rs.tmp`, then rename it over
//! `foo.rs`. Depending on the platform and the editor that surfaces as
//! `Remove` then `Create`, as two `Modify`s, or as a `Rename` — and
//! acting on the `Remove` drops a file that still exists, silently
//! removing it from retrieval until the next full rebuild.
//!
//! So events are used only as a hint that *something happened to this
//! path*. After the debounce window closes, the path is re-checked
//! against the filesystem, and that observation decides: readable file →
//! re-ingest, gone → remove. The event kind is never consulted.
//!
//! # Debouncing
//!
//! A single save can emit several events, and a `git checkout` can emit
//! thousands. Paths are collected into a pending set and only flushed
//! once they have been quiet for [`WatchConfig::debounce`], which
//! collapses a burst into one re-ingest per file.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{mpsc, Arc, RwLock};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use notify::{RecursiveMode, Watcher};

use crate::chunk::{has_indexable_extension, ChunkConfig, DEFAULT_IGNORED_DIRS};
use crate::delta::LiveCatalog;
use crate::embed::HashEmbedder;
use crate::error::AimError;

/// Watcher tuning.
#[derive(Debug, Clone)]
pub struct WatchConfig {
    /// How long a path must be quiet before it is re-ingested.
    pub debounce: Duration,
    /// Skip files larger than this, matching the indexer's limit.
    pub max_file_bytes: u64,
    /// Directory names to ignore, on top of [`DEFAULT_IGNORED_DIRS`].
    pub extra_ignored_dirs: Vec<String>,
    /// Chunking settings, which must match the ones the catalog was
    /// built with or delta chunks will not line up with base ones.
    pub chunk: ChunkConfig,
}

impl Default for WatchConfig {
    fn default() -> Self {
        Self {
            // Long enough to swallow an editor's write-temp-then-rename
            // and a fast sequence of saves, short enough that context is
            // current by the time someone finishes typing a question.
            debounce: Duration::from_millis(500),
            max_file_bytes: 2 * 1024 * 1024,
            extra_ignored_dirs: Vec::new(),
            chunk: ChunkConfig::default(),
        }
    }
}

/// Counters describing what the watcher has done.
#[derive(Debug, Default)]
pub struct WatchStats {
    /// Files re-chunked and installed into the delta layer.
    pub ingested: AtomicU64,
    /// Files observed as deleted and shadowed.
    pub removed: AtomicU64,
    /// Events discarded by the path filter.
    pub ignored: AtomicU64,
    /// Files that existed but could not be read or chunked.
    pub failed: AtomicU64,
}

impl WatchStats {
    /// Snapshot as `(ingested, removed, ignored, failed)`.
    pub fn snapshot(&self) -> (u64, u64, u64, u64) {
        (
            self.ingested.load(Ordering::Relaxed),
            self.removed.load(Ordering::Relaxed),
            self.ignored.load(Ordering::Relaxed),
            self.failed.load(Ordering::Relaxed),
        )
    }
}

/// Owns the watcher thread. Dropping it stops the thread.
pub struct WatcherHandle {
    stop: Arc<AtomicBool>,
    thread: Option<JoinHandle<()>>,
    stats: Arc<WatchStats>,
    // Held so the OS watch registration outlives the handle. Dropping
    // the watcher unregisters it and the event channel closes.
    _watcher: Box<dyn Watcher + Send>,
}

impl WatcherHandle {
    /// Counters for what the watcher has processed.
    pub fn stats(&self) -> &WatchStats {
        &self.stats
    }

    /// Stop the thread and wait for it to finish.
    pub fn stop(mut self) {
        self.shutdown();
    }

    fn shutdown(&mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

impl Drop for WatcherHandle {
    fn drop(&mut self) {
        self.shutdown();
    }
}

/// True when a path should be indexed: right extension, no ignored
/// directory component, not inside a dotted directory.
pub fn is_watchable(root: &Path, path: &Path, extra_ignored: &[String]) -> bool {
    if !has_indexable_extension(path) {
        return false;
    }
    let relative = path.strip_prefix(root).unwrap_or(path);

    for component in relative.components() {
        let std::path::Component::Normal(name) = component else {
            continue;
        };
        let name = name.to_string_lossy();
        // The final component is the file itself; a dotted *file*
        // (.eslintrc) is fine, a dotted *directory* is not.
        let is_dir_component = Path::new(name.as_ref()) != relative.file_name().map(Path::new).unwrap_or(Path::new(""));
        if DEFAULT_IGNORED_DIRS.contains(&name.as_ref())
            || extra_ignored.iter().any(|d| d == name.as_ref())
            || (is_dir_component && name.starts_with('.'))
        {
            return false;
        }
    }
    true
}

/// Workspace-relative path with forward slashes, matching the indexer.
fn relative_path(root: &Path, path: &Path) -> Option<String> {
    let relative = path.strip_prefix(root).ok()?;
    Some(relative.to_string_lossy().replace('\\', "/"))
}

/// Collects touched paths and releases them once they go quiet.
///
/// Separated from the IO so the timing rules can be tested without
/// provoking real filesystem events, which are inherently racy.
#[derive(Debug, Default)]
pub struct Debouncer {
    pending: HashMap<PathBuf, Instant>,
}

impl Debouncer {
    /// Create an empty debouncer.
    pub fn new() -> Self {
        Self::default()
    }

    /// Note that `path` was touched at `now`, resetting its timer.
    pub fn touch(&mut self, path: PathBuf, now: Instant) {
        self.pending.insert(path, now);
    }

    /// Number of paths waiting.
    pub fn pending(&self) -> usize {
        self.pending.len()
    }

    /// Remove and return every path quiet for at least `debounce`.
    ///
    /// Sorted so a burst is processed in a stable order, which keeps
    /// logs and tests reproducible.
    pub fn take_ready(&mut self, now: Instant, debounce: Duration) -> Vec<PathBuf> {
        let ready: Vec<PathBuf> = self
            .pending
            .iter()
            .filter(|(_, last)| now.duration_since(**last) >= debounce)
            .map(|(p, _)| p.clone())
            .collect();
        for path in &ready {
            self.pending.remove(path);
        }
        let mut ready = ready;
        ready.sort();
        ready
    }
}

/// Re-check one path against the filesystem and apply the result.
///
/// The event that queued this path is deliberately not consulted — see
/// the module docs on atomic saves.
fn apply_path(
    root: &Path,
    path: &Path,
    catalog: &RwLock<LiveCatalog>,
    embedder: &HashEmbedder,
    cfg: &WatchConfig,
    stats: &WatchStats,
) {
    let Some(relative) = relative_path(root, path) else {
        stats.ignored.fetch_add(1, Ordering::Relaxed);
        return;
    };

    let metadata = std::fs::metadata(path);
    let exists_as_file = metadata.as_ref().map(|m| m.is_file()).unwrap_or(false);

    if !exists_as_file {
        // Gone, or replaced by a directory. Either way its chunks are
        // no longer valid.
        if let Ok(mut guard) = catalog.write() {
            guard.delta_mut().remove_file(&relative);
            stats.removed.fetch_add(1, Ordering::Relaxed);
        }
        return;
    }

    if metadata.map(|m| m.len()).unwrap_or(0) > cfg.max_file_bytes {
        stats.ignored.fetch_add(1, Ordering::Relaxed);
        return;
    }

    // Read before taking the write lock: a slow disk must not block
    // in-flight queries.
    let Ok(contents) = std::fs::read_to_string(path) else {
        // Unreadable or not UTF-8. Leaving the base chunks in place is
        // the safer failure: stale context beats none, and a transient
        // lock during a save is the common cause.
        stats.failed.fetch_add(1, Ordering::Relaxed);
        return;
    };

    match catalog.write() {
        Ok(mut guard) => match guard.ingest_file(&relative, &contents, embedder, &cfg.chunk) {
            Ok(_) => {
                stats.ingested.fetch_add(1, Ordering::Relaxed);
            }
            Err(_) => {
                stats.failed.fetch_add(1, Ordering::Relaxed);
            }
        },
        Err(_) => {
            stats.failed.fetch_add(1, Ordering::Relaxed);
        }
    }
}

/// Watch `root` and keep `catalog`'s delta layer current.
///
/// The returned handle owns the thread; drop it to stop watching.
pub fn start_watcher(
    root: impl Into<PathBuf>,
    catalog: Arc<RwLock<LiveCatalog>>,
    cfg: WatchConfig,
) -> Result<WatcherHandle, AimError> {
    let root: PathBuf = root.into();
    let root = root
        .canonicalize()
        .map_err(|e| AimError::io("resolve watch root", &root, e))?;

    // Cloned once: the embedder is stateless apart from dimension and
    // the IDF table, and re-deriving it per event would copy the table
    // on every save.
    let embedder = {
        let guard = catalog
            .read()
            .map_err(|_| AimError::Index("catalog lock poisoned".into()))?;
        guard.base().query_embedder()
    };

    let (tx, rx) = mpsc::channel::<notify::Result<notify::Event>>();
    let mut watcher = notify::recommended_watcher(move |res| {
        // A closed receiver means the handle was dropped; nothing to do.
        let _ = tx.send(res);
    })
    .map_err(|e| AimError::Index(format!("create filesystem watcher: {e}")))?;

    watcher
        .watch(&root, RecursiveMode::Recursive)
        .map_err(|e| AimError::Index(format!("watch {}: {e}", root.display())))?;

    let stop = Arc::new(AtomicBool::new(false));
    let stats = Arc::new(WatchStats::default());

    let thread = {
        let stop = stop.clone();
        let stats = stats.clone();
        let root = root.clone();
        std::thread::Builder::new()
            .name("aim-watch".to_string())
            .spawn(move || {
                let mut debouncer = Debouncer::new();
                // Poll interval bounds how long a flush can be late.
                let tick = cfg.debounce.min(Duration::from_millis(100));

                while !stop.load(Ordering::Relaxed) {
                    // Drain whatever has arrived without blocking past
                    // the tick, so shutdown stays responsive.
                    match rx.recv_timeout(tick) {
                        Ok(Ok(event)) => {
                            for path in event.paths {
                                if is_watchable(&root, &path, &cfg.extra_ignored_dirs) {
                                    debouncer.touch(path, Instant::now());
                                } else {
                                    stats.ignored.fetch_add(1, Ordering::Relaxed);
                                }
                            }
                        }
                        Ok(Err(_)) => {}
                        Err(mpsc::RecvTimeoutError::Timeout) => {}
                        // The watcher was dropped; no further events can
                        // arrive, so flush what is pending and finish.
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    }

                    for path in debouncer.take_ready(Instant::now(), cfg.debounce) {
                        apply_path(&root, &path, &catalog, &embedder, &cfg, &stats);
                    }
                }

                // Final flush so an edit made just before shutdown is
                // not silently lost.
                for path in debouncer.take_ready(
                    Instant::now() + cfg.debounce,
                    cfg.debounce,
                ) {
                    apply_path(&root, &path, &catalog, &embedder, &cfg, &stats);
                }
            })
            .map_err(|e| AimError::Index(format!("spawn watcher thread: {e}")))?
    };

    Ok(WatcherHandle {
        stop,
        thread: Some(thread),
        stats,
        _watcher: Box::new(watcher),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn debouncer_holds_a_path_until_it_goes_quiet() {
        let mut d = Debouncer::new();
        let start = Instant::now();
        let debounce = Duration::from_millis(500);

        d.touch(PathBuf::from("a.rs"), start);
        assert!(d.take_ready(start, debounce).is_empty(), "released too early");
        assert_eq!(d.pending(), 1);

        let ready = d.take_ready(start + Duration::from_millis(600), debounce);
        assert_eq!(ready, vec![PathBuf::from("a.rs")]);
        assert_eq!(d.pending(), 0);
    }

    #[test]
    fn a_burst_of_saves_collapses_to_one_ingest() {
        // Editors emit several events per save; a git checkout emits
        // thousands. Each must cost one re-ingest per file, not one per
        // event.
        let mut d = Debouncer::new();
        let start = Instant::now();
        let debounce = Duration::from_millis(500);

        for i in 0..20 {
            d.touch(PathBuf::from("a.rs"), start + Duration::from_millis(i * 10));
        }
        assert_eq!(d.pending(), 1);

        let last_touch = start + Duration::from_millis(190);
        assert!(d.take_ready(last_touch + Duration::from_millis(400), debounce).is_empty());
        assert_eq!(
            d.take_ready(last_touch + Duration::from_millis(600), debounce),
            vec![PathBuf::from("a.rs")]
        );
    }

    #[test]
    fn continued_editing_keeps_resetting_the_timer() {
        let mut d = Debouncer::new();
        let mut now = Instant::now();
        let debounce = Duration::from_millis(500);

        for _ in 0..10 {
            d.touch(PathBuf::from("a.rs"), now);
            now += Duration::from_millis(100);
            assert!(
                d.take_ready(now, debounce).is_empty(),
                "flushed while still being edited"
            );
        }
        assert_eq!(
            d.take_ready(now + Duration::from_millis(600), debounce).len(),
            1
        );
    }

    #[test]
    fn ready_paths_come_back_in_a_stable_order() {
        let mut d = Debouncer::new();
        let start = Instant::now();
        for name in ["z.rs", "a.rs", "m.rs"] {
            d.touch(PathBuf::from(name), start);
        }
        let ready = d.take_ready(start + Duration::from_secs(1), Duration::from_millis(500));
        assert_eq!(
            ready,
            vec![
                PathBuf::from("a.rs"),
                PathBuf::from("m.rs"),
                PathBuf::from("z.rs")
            ]
        );
    }

    #[test]
    fn independent_paths_flush_independently() {
        let mut d = Debouncer::new();
        let start = Instant::now();
        let debounce = Duration::from_millis(500);

        d.touch(PathBuf::from("old.rs"), start);
        d.touch(PathBuf::from("new.rs"), start + Duration::from_millis(400));

        let ready = d.take_ready(start + Duration::from_millis(600), debounce);
        assert_eq!(ready, vec![PathBuf::from("old.rs")]);
        assert_eq!(d.pending(), 1);
    }

    #[test]
    fn watchable_accepts_source_and_rejects_noise() {
        let root = Path::new("/ws");
        assert!(is_watchable(root, Path::new("/ws/src/lib.rs"), &[]));
        assert!(is_watchable(root, Path::new("/ws/hw/misc/apple_mbox.c"), &[]));

        // Build output, VCS metadata, dotted directories, binaries.
        assert!(!is_watchable(root, Path::new("/ws/target/debug/build.rs"), &[]));
        assert!(!is_watchable(root, Path::new("/ws/.git/COMMIT_EDITMSG.md"), &[]));
        assert!(!is_watchable(root, Path::new("/ws/node_modules/x/a.js"), &[]));
        assert!(!is_watchable(root, Path::new("/ws/logo.png"), &[]));
        assert!(!is_watchable(root, Path::new("/ws/.aim/catalog.json"), &[]));
    }

    #[test]
    fn watchable_honours_extra_ignored_dirs() {
        let root = Path::new("/ws");
        let extra = vec!["generated".to_string()];
        assert!(!is_watchable(root, Path::new("/ws/generated/api.rs"), &extra));
        assert!(is_watchable(root, Path::new("/ws/src/api.rs"), &extra));
    }

    #[test]
    fn relative_paths_use_forward_slashes() {
        let root = Path::new("/ws");
        assert_eq!(
            relative_path(root, Path::new("/ws/src/hw/mbox.c")).unwrap(),
            "src/hw/mbox.c"
        );
        // A path outside the root has no workspace-relative form.
        assert!(relative_path(root, Path::new("/elsewhere/x.rs")).is_none());
    }
}
