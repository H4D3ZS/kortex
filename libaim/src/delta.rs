//! Keeping retrieval current between full rebuilds.
//!
//! # Why an overlay instead of mutating the catalog
//!
//! The obvious design is "call `IdMapIndex::remove` on the stale chunks
//! and add the new ones". That works for the index — it is in memory and
//! `remove` is O(1) — but the index is only one of four things a chunk
//! lives in. Its text is in a packed zstd payload, its path is in a
//! string heap, and its extent is in a fixed-stride chunk table, all
//! inside a memory-mapped container whose section offsets are baked into
//! the header. Inserting one chunk shifts every offset after it.
//!
//! On this workspace that container is 179 MB. Rewriting it on every
//! file save is not a tuning problem, it is the wrong shape.
//!
//! So the base catalog stays immutable and edits accumulate in a
//! [`DeltaLayer`] held in memory: re-chunked text and vectors for
//! touched files, plus a shadow set naming the paths whose base chunks
//! are now stale. [`LiveCatalog`] searches both and merges. This is the
//! same split an LSM tree makes, for the same reason — the base is large
//! and cold, the delta is small and hot.
//!
//! Brute-force cosine over the delta is deliberate. A coding session
//! touches tens of files, so the delta holds hundreds of vectors at
//! most; scoring 500 × 1536 floats takes microseconds, which is far
//! below the millisecond-scale quantized search it rides alongside.
//! Quantizing the delta would add TQ+ calibration drift for no
//! measurable gain.
//!
//! The delta is not persisted. It is rebuilt by re-reading changed files
//! at startup or discarded by a full `aim-index build`, and treating it
//! as a cache means a crash can never leave a half-written catalog.

use std::collections::{HashMap, HashSet};

use crate::catalog::{Catalog, Hit, PageFaultResult, RetrievalConfig};
use crate::chunk::SourceChunk;
use crate::embed::{cosine, normalize};
use crate::error::AimError;

/// A chunk living in the delta layer, held uncompressed and unquantized.
#[derive(Debug, Clone)]
pub struct LiveChunk {
    pub path: String,
    pub line_start: u32,
    pub line_end: u32,
    pub text: String,
    /// L2-normalized embedding, in the base catalog's vector space.
    pub vector: Vec<f32>,
    pub token_estimate: u32,
}

/// Edits layered over an immutable base catalog.
#[derive(Debug, Default)]
pub struct DeltaLayer {
    dim: usize,
    /// Current chunks for each edited path.
    live: HashMap<String, Vec<LiveChunk>>,
    /// Paths whose base-catalog chunks must be ignored. A deleted file
    /// is shadowed with no live chunks.
    shadowed: HashSet<String>,
}

impl DeltaLayer {
    /// Create an empty layer for `dim`-dimensional vectors.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            live: HashMap::new(),
            shadowed: HashSet::new(),
        }
    }

    /// Vector dimension this layer expects.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Replace the chunks for one path.
    ///
    /// Shadows the path unconditionally, including when `chunks` is
    /// empty: a file edited down to whitespace still has stale chunks in
    /// the base that must stop being retrieved.
    pub fn update_file(&mut self, path: &str, chunks: Vec<LiveChunk>) -> Result<(), AimError> {
        for chunk in &chunks {
            if chunk.vector.len() != self.dim {
                return Err(AimError::DimMismatch {
                    catalog: self.dim,
                    got: chunk.vector.len(),
                });
            }
        }
        let path = normalize_path(path);
        self.shadowed.insert(path.clone());
        if chunks.is_empty() {
            self.live.remove(&path);
        } else {
            self.live.insert(path, chunks);
        }
        Ok(())
    }

    /// Mark a path deleted: its base chunks stop being retrieved and it
    /// contributes nothing new.
    pub fn remove_file(&mut self, path: &str) {
        let path = normalize_path(path);
        self.live.remove(&path);
        self.shadowed.insert(path);
    }

    /// Handle a rename as a delete of `from` plus an update of `to`.
    pub fn rename_file(
        &mut self,
        from: &str,
        to: &str,
        chunks: Vec<LiveChunk>,
    ) -> Result<(), AimError> {
        self.remove_file(from);
        self.update_file(to, chunks)
    }

    /// True when this path's base-catalog chunks must be ignored.
    pub fn is_shadowed(&self, path: &str) -> bool {
        self.shadowed.contains(&normalize_path(path))
    }

    /// Number of live chunks across all edited files.
    pub fn len(&self) -> usize {
        self.live.values().map(|v| v.len()).sum()
    }

    /// True when nothing has been edited.
    pub fn is_empty(&self) -> bool {
        self.live.is_empty() && self.shadowed.is_empty()
    }

    /// Paths currently shadowed.
    pub fn shadowed_paths(&self) -> impl Iterator<Item = &str> {
        self.shadowed.iter().map(|s| s.as_str())
    }

    /// Forget every edit.
    pub fn clear(&mut self) {
        self.live.clear();
        self.shadowed.clear();
    }

    /// Top-`k` live chunks by cosine similarity, best first.
    pub fn search(&self, query: &[f32], k: usize) -> Vec<(f32, &LiveChunk)> {
        if k == 0 || query.len() != self.dim {
            return Vec::new();
        }
        let mut scored: Vec<(f32, &LiveChunk)> = self
            .live
            .values()
            .flatten()
            .map(|c| (cosine(query, &c.vector), c))
            .filter(|(s, _)| s.is_finite())
            .collect();

        // Ties break on path and line so results are deterministic;
        // HashMap iteration order is not.
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap()
                .then_with(|| a.1.path.cmp(&b.1.path))
                .then_with(|| a.1.line_start.cmp(&b.1.line_start))
        });
        scored.truncate(k);
        scored
    }
}

/// Normalize separators so a path from a file watcher matches one stored
/// by the indexer.
fn normalize_path(path: &str) -> String {
    path.replace('\\', "/")
}

/// A base catalog plus its delta layer, queried as one.
pub struct LiveCatalog {
    base: Catalog,
    delta: DeltaLayer,
}

impl LiveCatalog {
    /// Wrap a catalog with an empty delta layer.
    pub fn new(base: Catalog) -> Self {
        let dim = base.dim();
        Self {
            base,
            delta: DeltaLayer::new(dim),
        }
    }

    /// The immutable base catalog.
    pub fn base(&self) -> &Catalog {
        &self.base
    }

    /// The delta layer, for applying edits.
    pub fn delta_mut(&mut self) -> &mut DeltaLayer {
        &mut self.delta
    }

    /// The delta layer.
    pub fn delta(&self) -> &DeltaLayer {
        &self.delta
    }

    /// Embedding dimension.
    pub fn dim(&self) -> usize {
        self.base.dim()
    }

    /// Chunk the given text and install it as the current state of
    /// `path`, embedding with `embedder`.
    ///
    /// This is the whole ingest path for one edited file.
    pub fn ingest_file(
        &mut self,
        path: &str,
        contents: &str,
        embedder: &crate::embed::HashEmbedder,
        cfg: &crate::chunk::ChunkConfig,
    ) -> Result<usize, AimError> {
        let chunks = crate::chunk::chunk_source(path, contents, cfg);
        let live: Result<Vec<LiveChunk>, AimError> = chunks
            .into_iter()
            .map(|c: SourceChunk| {
                let vector = embedder.embed_with_path(&c.path, &c.text)?;
                Ok(LiveChunk {
                    token_estimate: c.token_estimate(),
                    path: c.path,
                    line_start: c.line_start,
                    line_end: c.line_end,
                    text: c.text,
                    vector,
                })
            })
            .collect();

        let live = live?;
        let count = live.len();
        self.delta.update_file(path, live)?;
        Ok(count)
    }

    /// The gist, adjusted for the delta.
    ///
    /// The base gist is the L2-normalized mean of every chunk vector, so
    /// it updates by removing the shadowed contributions and adding the
    /// live ones. That is arithmetic on a mean — there is no holographic
    /// binding here to invert.
    ///
    /// Approximate on one point: the number of base chunks each shadowed
    /// path contributed is known, but their summed vectors are not
    /// recoverable from the normalized mean alone. The live vectors are
    /// therefore blended in proportionally rather than exactly. The gist
    /// is only a coarse "is this query about this workspace" signal, so
    /// the approximation costs nothing that matters; ranking never
    /// consults it.
    pub fn gist(&self) -> Vec<f32> {
        let base = self.base.gist();
        if self.delta.is_empty() {
            return base.to_vec();
        }

        let base_n = self.base.len().max(1) as f32;
        let mut acc: Vec<f32> = base.iter().map(|x| x * base_n).collect();

        for chunk in self.delta.live.values().flatten() {
            for (a, v) in acc.iter_mut().zip(&chunk.vector) {
                *a += *v;
            }
        }
        normalize(&mut acc);
        acc
    }

    /// Retrieve, merging base and delta results.
    ///
    /// Base hits from shadowed paths are discarded, so an edited file is
    /// never served from the stale catalog — that is the whole point,
    /// since a stale hit carries line numbers that no longer exist.
    pub fn page_fault(
        &self,
        query: &[f32],
        cfg: &RetrievalConfig,
    ) -> Result<PageFaultResult, AimError> {
        self.page_fault_inner(query, cfg, None)
    }

    /// Retrieval restricted to paths containing one of `substrings`.
    ///
    /// Applies to both tiers: the base is searched through turbovec's
    /// allowlist, and delta chunks are filtered by the same substrings.
    /// Scoping only the base would let an edited file outside the scope
    /// leak into a request that named a specific file.
    pub fn page_fault_scoped(
        &self,
        query: &[f32],
        cfg: &RetrievalConfig,
        path_substrings: &[&str],
    ) -> Result<PageFaultResult, AimError> {
        let matches_scope = |path: &str| {
            path_substrings
                .iter()
                .any(|s| !s.is_empty() && path.contains(s))
        };
        let base_ids = self.base.chunks_matching_paths(path_substrings);
        let delta_matches = self
            .delta
            .live
            .keys()
            .any(|p| matches_scope(p));

        // Nothing anywhere matched the hint, so it was probably wrong.
        // A corpus-wide search beats returning nothing.
        if base_ids.is_empty() && !delta_matches {
            return self.page_fault(query, cfg);
        }
        self.page_fault_inner(query, cfg, Some(path_substrings))
    }

    /// Pin the hottest base chunks into physical RAM. Delta chunks are
    /// already resident, so only the base needs locking.
    pub fn pin_chunks(&self, chunk_ids: &[u64], budget_bytes: usize) -> crate::heat::PinReport {
        self.base.pin_chunks(chunk_ids, budget_bytes)
    }

    fn page_fault_inner(
        &self,
        query: &[f32],
        cfg: &RetrievalConfig,
        scope: Option<&[&str]>,
    ) -> Result<PageFaultResult, AimError> {
        if query.len() != self.dim() {
            return Err(AimError::DimMismatch {
                catalog: self.dim(),
                got: query.len(),
            });
        }

        // Over-fetch from the base: shadowed hits are dropped after
        // scoring, and without headroom a heavily-edited session could
        // filter the candidate list down to nothing.
        let base_k = cfg.candidates.saturating_mul(2).max(cfg.candidates);
        let in_scope = |path: &str| match scope {
            None => true,
            Some(subs) => subs.iter().any(|s| !s.is_empty() && path.contains(s)),
        };

        let mut candidates: Vec<(f32, Option<u64>, Option<&LiveChunk>)> = Vec::new();

        let base_hits = match scope {
            Some(subs) => {
                let allowed = self.base.chunks_matching_paths(subs);
                self.base.search_scoped(query, base_k, &allowed)?
            }
            None => self.base.search(query, base_k)?,
        };
        for (score, id) in base_hits {
            let path = self.base.chunk_path(id)?;
            if self.delta.is_shadowed(path) {
                continue;
            }
            candidates.push((score, Some(id), None));
        }
        for (score, chunk) in self.delta.search(query, cfg.candidates) {
            if !in_scope(&chunk.path) {
                continue;
            }
            candidates.push((score, None, Some(chunk)));
        }

        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
        let best_score = candidates.first().map(|(s, _, _)| *s).unwrap_or(0.0);
        let cutoff = cfg
            .fault_threshold
            .max(best_score * cfg.relative_floor.clamp(0.0, 1.0));

        let mut hits: Vec<Hit> = Vec::new();
        let mut spent = 0u32;
        let mut dropped = 0usize;
        let mut seen: Vec<(String, u32, u32)> = Vec::new();

        for (score, base_id, live) in candidates {
            if score < cutoff {
                break;
            }
            if hits.len() >= cfg.max_chunks {
                dropped += 1;
                continue;
            }

            let hit = match (base_id, live) {
                (Some(id), _) => {
                    let (line_start, line_end, token_estimate) = self
                        .base
                        .chunk_extent(id)
                        .ok_or(AimError::UnknownChunk(id))?;
                    Hit {
                        chunk_id: id,
                        path: self.base.chunk_path(id)?.to_string(),
                        line_start,
                        line_end,
                        score,
                        token_estimate,
                        text: self.base.inflate(id)?,
                    }
                }
                (None, Some(c)) => Hit {
                    // Delta chunks have no base id. u64::MAX marks them
                    // so a caller cannot mistake one for a catalog id and
                    // try to inflate it from the container.
                    chunk_id: u64::MAX,
                    path: c.path.clone(),
                    line_start: c.line_start,
                    line_end: c.line_end,
                    score,
                    token_estimate: c.token_estimate,
                    text: c.text.clone(),
                },
                (None, None) => continue,
            };

            if spent + hit.token_estimate > cfg.token_budget {
                dropped += 1;
                continue;
            }
            if seen
                .iter()
                .any(|(p, s, e)| *p == hit.path && hit.line_start <= *e && hit.line_end >= *s)
            {
                continue;
            }

            seen.push((hit.path.clone(), hit.line_start, hit.line_end));
            spent += hit.token_estimate;
            hits.push(hit);
        }

        let gist = self.gist();
        Ok(PageFaultResult {
            hits,
            best_score,
            cutoff,
            gist_score: cosine(query, &gist),
            dropped_to_budget: dropped,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::embed::{Embedder, HashEmbedder};

    fn chunk(path: &str, line_start: u32, vector: Vec<f32>) -> LiveChunk {
        LiveChunk {
            path: path.to_string(),
            line_start,
            line_end: line_start + 10,
            text: format!("contents of {path} at {line_start}"),
            vector,
            token_estimate: 10,
        }
    }

    fn unit(dim: usize, hot: usize) -> Vec<f32> {
        let mut v = vec![0.0; dim];
        v[hot % dim] = 1.0;
        v
    }

    #[test]
    fn updating_a_file_shadows_its_base_chunks() {
        let mut d = DeltaLayer::new(4);
        assert!(!d.is_shadowed("src/lib.rs"));
        d.update_file("src/lib.rs", vec![chunk("src/lib.rs", 1, unit(4, 0))])
            .unwrap();
        assert!(d.is_shadowed("src/lib.rs"));
        assert_eq!(d.len(), 1);
    }

    #[test]
    fn a_file_edited_to_empty_still_shadows_the_base() {
        // Otherwise the stale base chunks keep being retrieved for a file
        // whose content is gone.
        let mut d = DeltaLayer::new(4);
        d.update_file("src/lib.rs", vec![]).unwrap();
        assert!(d.is_shadowed("src/lib.rs"));
        assert_eq!(d.len(), 0);
    }

    #[test]
    fn deleting_a_file_shadows_it_and_drops_its_chunks() {
        let mut d = DeltaLayer::new(4);
        d.update_file("a.rs", vec![chunk("a.rs", 1, unit(4, 0))]).unwrap();
        d.remove_file("a.rs");
        assert!(d.is_shadowed("a.rs"));
        assert_eq!(d.len(), 0);
    }

    #[test]
    fn renaming_shadows_the_old_path_and_installs_the_new() {
        let mut d = DeltaLayer::new(4);
        d.rename_file("old.rs", "new.rs", vec![chunk("new.rs", 1, unit(4, 1))])
            .unwrap();
        assert!(d.is_shadowed("old.rs"));
        assert!(d.is_shadowed("new.rs"));
        assert_eq!(d.len(), 1);
    }

    #[test]
    fn updating_twice_replaces_rather_than_accumulates() {
        // A file saved ten times must not leave ten copies in the layer.
        let mut d = DeltaLayer::new(4);
        for i in 0..10 {
            d.update_file("a.rs", vec![chunk("a.rs", i, unit(4, 0))]).unwrap();
        }
        assert_eq!(d.len(), 1);
    }

    #[test]
    fn windows_and_unix_paths_refer_to_the_same_file() {
        let mut d = DeltaLayer::new(4);
        d.update_file(r"src\hw\mbox.c", vec![chunk("src/hw/mbox.c", 1, unit(4, 0))])
            .unwrap();
        assert!(d.is_shadowed("src/hw/mbox.c"));
        assert!(d.is_shadowed(r"src\hw\mbox.c"));
    }

    #[test]
    fn wrong_dimension_is_rejected() {
        let mut d = DeltaLayer::new(4);
        let err = d.update_file("a.rs", vec![chunk("a.rs", 1, vec![0.0; 8])]);
        assert!(matches!(err, Err(AimError::DimMismatch { .. })));
        // And the layer must not have been half-updated.
        assert!(!d.is_shadowed("a.rs"));
    }

    #[test]
    fn search_ranks_by_similarity_and_respects_k() {
        let mut d = DeltaLayer::new(8);
        d.update_file("a.rs", vec![chunk("a.rs", 1, unit(8, 0))]).unwrap();
        d.update_file("b.rs", vec![chunk("b.rs", 1, unit(8, 1))]).unwrap();
        d.update_file("c.rs", vec![chunk("c.rs", 1, unit(8, 2))]).unwrap();

        let hits = d.search(&unit(8, 1), 2);
        assert_eq!(hits.len(), 2);
        assert_eq!(hits[0].1.path, "b.rs");
        assert!(hits[0].0 > hits[1].0);
    }

    #[test]
    fn search_is_deterministic_across_equal_scores() {
        let mut a = DeltaLayer::new(8);
        let mut b = DeltaLayer::new(8);
        for d in [&mut a, &mut b] {
            for name in ["z.rs", "m.rs", "a.rs"] {
                d.update_file(name, vec![chunk(name, 1, unit(8, 0))]).unwrap();
            }
        }
        let order = |d: &DeltaLayer| -> Vec<String> {
            d.search(&unit(8, 0), 3)
                .into_iter()
                .map(|(_, c)| c.path.clone())
                .collect()
        };
        assert_eq!(order(&a), order(&b));
        assert_eq!(order(&a), vec!["a.rs", "m.rs", "z.rs"]);
    }

    #[test]
    fn search_with_wrong_dimension_returns_nothing_rather_than_panicking() {
        let mut d = DeltaLayer::new(8);
        d.update_file("a.rs", vec![chunk("a.rs", 1, unit(8, 0))]).unwrap();
        assert!(d.search(&vec![0.0; 4], 5).is_empty());
        assert!(d.search(&unit(8, 0), 0).is_empty());
    }

    #[test]
    fn clear_forgets_every_edit() {
        let mut d = DeltaLayer::new(4);
        d.update_file("a.rs", vec![chunk("a.rs", 1, unit(4, 0))]).unwrap();
        d.remove_file("b.rs");
        d.clear();
        assert!(d.is_empty());
        assert!(!d.is_shadowed("a.rs"));
        assert!(!d.is_shadowed("b.rs"));
    }

    #[test]
    fn live_chunks_embed_in_the_same_space_as_the_indexer() {
        // The delta must use `embed_with_path`, matching what the
        // indexer did, or delta scores are not comparable to base ones
        // and merging them ranks nonsense.
        let e = HashEmbedder::new(256);
        let with_path = e.embed_with_path("src/apple_mbox.c", "void irq(void) {}").unwrap();
        let without = e.embed("void irq(void) {}").unwrap();
        assert_ne!(with_path, without);
    }
}
