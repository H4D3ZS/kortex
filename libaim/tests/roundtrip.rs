//! End-to-end tests: build a catalog from files on disk, reopen it, and
//! retrieve through the real search path.
//!
//! The unit tests cover each stage in isolation. These exist to catch
//! the failures that only appear when the stages are wired together —
//! an offset computed one way at write time and read another way, an id
//! that does not survive the turbovec round trip, a payload slice that
//! is correct in memory and wrong through the mmap.

use std::path::{Path, PathBuf};

use libaim::{
    index_workspace, Catalog, ChunkConfig, Embedder, HashEmbedder, IndexOptions, RetrievalConfig,
};

/// Embedding dimension used by these tests.
///
/// Not the 1536 default: turbovec builds a `dim × dim` random rotation
/// at index time, so the cost is quadratic in `dim` and dominates an
/// unoptimized test binary. 512 keeps retrieval quality on this fixture
/// identical while cutting the suite from minutes to seconds.
const TEST_DIM: usize = 512;

/// A temp directory that cleans itself up.
struct TempDir(PathBuf);

impl TempDir {
    fn new(tag: &str) -> Self {
        let mut p = std::env::temp_dir();
        // Nanosecond clock plus the test name: `cargo test` runs these
        // concurrently, and a shared path would make them flaky.
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        p.push(format!("libaim-test-{tag}-{stamp}"));
        std::fs::create_dir_all(&p).unwrap();
        TempDir(p)
    }

    fn path(&self) -> &Path {
        &self.0
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

/// Write a small but realistic workspace: a few C and Rust files whose
/// contents are distinguishable by topic.
fn write_fixture(root: &Path) {
    let files: &[(&str, &str)] = &[
        (
            "hw/misc/apple_mbox.c",
            r#"
#include "hw/misc/apple_mbox.h"

/*
 * AGX mailbox. The IRQ is level-triggered, so a handler that drains
 * fewer messages than arrived will starve the endpoint.
 */
static void apple_mbox_irq_update(AppleMboxState *s)
{
    bool pending = !fifo_is_empty(&s->inbox);
    qemu_set_irq(s->irq, pending);
}

static void apple_mbox_drain_inbox(AppleMboxState *s)
{
    while (!fifo_is_empty(&s->inbox)) {
        AppleMboxMessage msg = fifo_pop(&s->inbox);
        apple_mbox_dispatch(s, &msg);
    }
    apple_mbox_irq_update(s);
}
"#,
        ),
        (
            "hw/char/serial_pl011.c",
            r#"
#include "hw/char/pl011.h"

static void pl011_write_fifo(PL011State *s, uint32_t value)
{
    s->read_fifo[s->read_pos] = value;
    s->read_count++;
    pl011_update(s);
}

static uint64_t pl011_read(void *opaque, hwaddr offset, unsigned size)
{
    PL011State *s = opaque;
    return s->read_fifo[s->read_pos];
}
"#,
        ),
        (
            "src/phonycode/shim.rs",
            r#"
//! Intercepts xcodebuild invocations and rewrites them to clang.

pub fn translate_xcodebuild(args: &[String], sdk_path: &str) -> Vec<String> {
    let mut out = vec!["clang".to_string()];
    out.push("-target".to_string());
    out.push("arm64-apple-ios".to_string());
    out.push("-isysroot".to_string());
    out.push(sdk_path.to_string());
    out.extend(args.iter().cloned());
    out
}
"#,
        ),
        (
            "docs/CHANGELOG.md",
            r#"
# Changelog

## 0.3.0
- Updated installation instructions in the README.
- Bumped the documentation theme.
- Fixed a broken link in the contributing guide.
"#,
        ),
    ];

    for (rel, body) in files {
        let path = root.join(rel);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(&path, body.trim_start()).unwrap();
    }

    // Must be excluded by the extension filter and the ignore list.
    std::fs::create_dir_all(root.join("target/debug")).unwrap();
    std::fs::write(root.join("target/debug/build.rs"), "fn main() {}").unwrap();
    std::fs::write(root.join("logo.png"), [0x89u8, 0x50, 0x4e, 0x47]).unwrap();
}

/// Build a catalog over the fixture and return `(tempdir, catalog dir)`.
fn build_fixture(tag: &str) -> (TempDir, PathBuf) {
    let tmp = TempDir::new(tag);
    write_fixture(tmp.path());

    let embedder = HashEmbedder::new(TEST_DIM);
    let opts = IndexOptions::default();
    let stats = index_workspace(tmp.path(), &embedder, &opts, |_, _| {}).unwrap();

    assert!(stats.meta.chunk_count > 0, "catalog has no chunks");
    let catalog_dir = tmp.path().join(".aim");
    (tmp, catalog_dir)
}

#[test]
fn catalog_round_trips_through_disk() {
    let (_tmp, dir) = build_fixture("roundtrip");

    let catalog = Catalog::open(&dir).unwrap();
    assert!(catalog.len() > 0);
    assert_eq!(catalog.dim(), TEST_DIM);
    catalog.verify_integrity().unwrap();

    let embedder = HashEmbedder::new(catalog.dim());
    catalog.check_embedder(&embedder.id()).unwrap();
}

#[test]
fn indexer_skips_ignored_dirs_and_non_source_files() {
    let (_tmp, dir) = build_fixture("filters");
    let catalog = Catalog::open(&dir).unwrap();

    let paths: Vec<String> = catalog.paths().map(|p| p.to_string()).collect();
    assert!(
        !paths.iter().any(|p| p.contains("target/")),
        "target/ was indexed: {paths:?}"
    );
    assert!(
        !paths.iter().any(|p| p.ends_with(".png")),
        "a binary was indexed: {paths:?}"
    );
    assert!(
        paths.iter().any(|p| p.contains("apple_mbox.c")),
        "expected source file missing: {paths:?}"
    );
}

#[test]
fn paths_use_forward_slashes_regardless_of_host() {
    let (_tmp, dir) = build_fixture("slashes");
    let catalog = Catalog::open(&dir).unwrap();
    for p in catalog.paths() {
        assert!(!p.contains('\\'), "path kept a backslash: {p}");
    }
}

#[test]
fn retrieval_returns_the_topically_correct_file() {
    let (_tmp, dir) = build_fixture("retrieve");
    let catalog = Catalog::open(&dir).unwrap();
    let embedder = HashEmbedder::new(catalog.dim());

    let query = embedder
        .embed("fix the AGX mailbox IRQ starvation in apple_mbox.c")
        .unwrap();

    let hits = catalog.search(&query, 4).unwrap();
    assert!(!hits.is_empty(), "search returned nothing");

    let top_path = catalog.chunk_path(hits[0].1).unwrap();
    assert!(
        top_path.contains("apple_mbox.c"),
        "top hit was {top_path} (scores: {:?})",
        hits.iter().map(|(s, _)| *s).collect::<Vec<_>>()
    );

    // Scores must be ordered best-first — the budget loop in
    // `page_fault` breaks on the first sub-threshold score and would
    // silently drop good hits if this did not hold.
    for w in hits.windows(2) {
        assert!(w[0].0 >= w[1].0, "scores not descending: {hits:?}");
    }
}

#[test]
fn inflated_text_matches_the_original_source() {
    let (tmp, dir) = build_fixture("inflate");
    let catalog = Catalog::open(&dir).unwrap();

    for chunk_id in 0..catalog.len() as u64 {
        let path = catalog.chunk_path(chunk_id).unwrap().to_string();
        let text = catalog.inflate(chunk_id).unwrap();

        let on_disk = std::fs::read_to_string(tmp.path().join(&path)).unwrap();
        let lines: Vec<&str> = on_disk.lines().collect();

        // The chunk must be a verbatim contiguous run of the real file,
        // which is what makes the `path:line` citations trustworthy.
        assert!(
            on_disk.contains(&text),
            "chunk {chunk_id} of {path} is not a verbatim substring of the source"
        );

        // And the path index must agree that this chunk belongs here.
        assert!(
            catalog.chunks_matching_paths(&[&path]).contains(&chunk_id),
            "chunk {chunk_id} is missing from the path index for {path}"
        );

        assert!(!lines.is_empty());
    }
}

#[test]
fn page_fault_reports_line_ranges_that_match_the_source() {
    let (tmp, dir) = build_fixture("lines");
    let catalog = Catalog::open(&dir).unwrap();
    let embedder = HashEmbedder::new(catalog.dim());

    let query = embedder.embed("apple_mbox drain inbox fifo").unwrap();
    let cfg = RetrievalConfig {
        fault_threshold: 0.0, // take whatever comes back
        ..Default::default()
    };
    let fault = catalog.page_fault(&query, &cfg).unwrap();
    assert!(fault.faulted(), "nothing retrieved at threshold 0");

    for hit in &fault.hits {
        let on_disk = std::fs::read_to_string(tmp.path().join(&hit.path)).unwrap();
        let lines: Vec<&str> = on_disk.lines().collect();

        assert!(hit.line_start >= 1, "line numbers are 1-based");
        assert!(
            hit.line_end as usize <= lines.len(),
            "chunk claims line {} but {} has {} lines",
            hit.line_end,
            hit.path,
            lines.len()
        );

        // The chunk's first line must be the file's line_start.
        let expected_first = lines[hit.line_start as usize - 1];
        let actual_first = hit.text.lines().next().unwrap_or("");
        assert_eq!(
            expected_first, actual_first,
            "chunk {} of {} starts at the wrong line",
            hit.chunk_id, hit.path
        );
    }
}

#[test]
fn threshold_gates_injection() {
    let (_tmp, dir) = build_fixture("threshold");
    let catalog = Catalog::open(&dir).unwrap();
    let embedder = HashEmbedder::new(catalog.dim());
    let query = embedder.embed("apple mbox irq").unwrap();

    let permissive = catalog
        .page_fault(
            &query,
            &RetrievalConfig {
                fault_threshold: 0.0,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(permissive.faulted());

    // A cosine above 1.0 is unreachable, so nothing may qualify.
    let impossible = catalog
        .page_fault(
            &query,
            &RetrievalConfig {
                fault_threshold: 1.01,
                ..Default::default()
            },
        )
        .unwrap();
    assert!(
        !impossible.faulted(),
        "hits cleared an impossible threshold: {:?}",
        impossible.hits.iter().map(|h| h.score).collect::<Vec<_>>()
    );
    assert!(impossible.render_context().is_empty());
}

#[test]
fn one_config_serves_both_short_and_long_queries() {
    // The regression this guards: absolute cosine magnitude scales with
    // query length, so a fixed threshold tuned on a long query silently
    // retrieves nothing for a short one. The relative gate must make a
    // single default work for both.
    let (_tmp, dir) = build_fixture("scalefree");
    let catalog = Catalog::open(&dir).unwrap();
    let embedder = HashEmbedder::new(catalog.dim());
    let cfg = RetrievalConfig::default();

    let short = embedder.embed("mailbox irq").unwrap();
    let long = embedder
        .embed("fix the AGX mailbox IRQ starvation bug in the apple_mbox drain path")
        .unwrap();

    let short_fault = catalog.page_fault(&short, &cfg).unwrap();
    let long_fault = catalog.page_fault(&long, &cfg).unwrap();

    assert!(
        short_fault.faulted(),
        "short query retrieved nothing (best {:.4}, cutoff {:.4})",
        short_fault.best_score,
        short_fault.cutoff
    );
    assert!(
        long_fault.faulted(),
        "long query retrieved nothing (best {:.4}, cutoff {:.4})",
        long_fault.best_score,
        long_fault.cutoff
    );

    // The cutoff must track the score scale rather than being constant.
    assert!(
        long_fault.cutoff > short_fault.cutoff,
        "cutoff did not adapt: short {:.4}, long {:.4}",
        short_fault.cutoff,
        long_fault.cutoff
    );

    // Nothing below the reported cutoff may be injected.
    for f in [&short_fault, &long_fault] {
        for hit in &f.hits {
            assert!(
                hit.score >= f.cutoff,
                "injected {:.4} below cutoff {:.4}",
                hit.score,
                f.cutoff
            );
        }
    }
}

#[test]
fn relative_floor_of_one_keeps_only_the_best_hit() {
    let (_tmp, dir) = build_fixture("relone");
    let catalog = Catalog::open(&dir).unwrap();
    let embedder = HashEmbedder::new(catalog.dim());
    let query = embedder.embed("static void update").unwrap();

    let fault = catalog
        .page_fault(
            &query,
            &RetrievalConfig {
                fault_threshold: 0.0,
                relative_floor: 1.0,
                candidates: 64,
                max_chunks: 20,
                token_budget: u32::MAX,
            },
        )
        .unwrap();

    for hit in &fault.hits {
        assert!(
            (hit.score - fault.best_score).abs() < 1e-6,
            "kept {:.6} when only ties with {:.6} should survive",
            hit.score,
            fault.best_score
        );
    }
}

#[test]
fn idf_table_is_stored_and_applied_at_query_time() {
    let (_tmp, dir) = build_fixture("idf");
    let catalog = Catalog::open(&dir).unwrap();

    let idf = catalog.idf();
    assert_eq!(idf.len(), catalog.dim());
    assert!(idf.iter().all(|w| w.is_finite() && *w >= 0.0));
    assert!(
        idf.iter().any(|w| *w > 0.0),
        "every IDF weight is zero, so all vectors collapse"
    );

    // The catalog's embedder must carry the weights; a bare one must not.
    assert!(catalog.query_embedder().has_idf());
    assert!(!HashEmbedder::new(catalog.dim()).has_idf());
}

#[test]
fn common_words_score_below_distinctive_ones() {
    // The bug this guards is subtle and severe: without IDF, a query of
    // ubiquitous words concentrates its mass on a few features and
    // out-scores a precise technical query. On the real 243k-chunk
    // workspace "hello" scored 0.40 while "implement zstd decompression
    // for chunk payloads" scored 0.14 — retrieval was anti-correlated
    // with relevance.
    let (_tmp, dir) = build_fixture("idfrank");
    let catalog = Catalog::open(&dir).unwrap();
    let embedder = catalog.query_embedder();

    let common = embedder.embed("the s a void").unwrap();
    let distinctive = embedder.embed("apple_mbox_drain_inbox").unwrap();

    let common_best = catalog
        .search(&common, 1)
        .unwrap()
        .first()
        .map(|(s, _)| *s)
        .unwrap_or(0.0);
    let distinctive_best = catalog
        .search(&distinctive, 1)
        .unwrap()
        .first()
        .map(|(s, _)| *s)
        .unwrap_or(0.0);

    assert!(
        distinctive_best > common_best,
        "distinctive query scored {distinctive_best:.4} but common words scored \
         {common_best:.4} — IDF is not doing its job"
    );
}

#[test]
fn token_budget_is_never_exceeded() {
    let (_tmp, dir) = build_fixture("budget");
    let catalog = Catalog::open(&dir).unwrap();
    let embedder = HashEmbedder::new(catalog.dim());
    let query = embedder.embed("static void update fifo state").unwrap();

    let cfg = RetrievalConfig {
        fault_threshold: 0.0,
        relative_floor: 0.0,
        candidates: 64,
        max_chunks: 100,
        token_budget: 150,
    };
    let fault = catalog.page_fault(&query, &cfg).unwrap();
    assert!(
        fault.injected_tokens() <= cfg.token_budget,
        "injected {} tokens against a {} budget",
        fault.injected_tokens(),
        cfg.token_budget
    );
}

#[test]
fn max_chunks_is_respected() {
    let (_tmp, dir) = build_fixture("maxchunks");
    let catalog = Catalog::open(&dir).unwrap();
    let embedder = HashEmbedder::new(catalog.dim());
    let query = embedder.embed("static void").unwrap();

    let cfg = RetrievalConfig {
        fault_threshold: 0.0,
        relative_floor: 0.0,
        candidates: 64,
        max_chunks: 2,
        token_budget: u32::MAX,
    };
    let fault = catalog.page_fault(&query, &cfg).unwrap();
    assert!(fault.hits.len() <= 2, "got {} hits", fault.hits.len());
}

#[test]
fn scoped_retrieval_stays_inside_the_named_file() {
    let (_tmp, dir) = build_fixture("scoped");
    let catalog = Catalog::open(&dir).unwrap();
    let embedder = HashEmbedder::new(catalog.dim());

    // A query whose words appear in several files.
    let query = embedder.embed("static void update state").unwrap();
    let cfg = RetrievalConfig {
        fault_threshold: 0.0,
        ..Default::default()
    };

    let fault = catalog
        .page_fault_scoped(&query, &cfg, &["serial_pl011"])
        .unwrap();
    assert!(fault.faulted(), "scoped retrieval returned nothing");
    for hit in &fault.hits {
        assert!(
            hit.path.contains("serial_pl011"),
            "scope leaked to {}",
            hit.path
        );
    }
}

#[test]
fn scoped_retrieval_falls_back_when_the_scope_matches_nothing() {
    let (_tmp, dir) = build_fixture("scopemiss");
    let catalog = Catalog::open(&dir).unwrap();
    let embedder = HashEmbedder::new(catalog.dim());
    let query = embedder.embed("mailbox irq").unwrap();
    let cfg = RetrievalConfig {
        fault_threshold: 0.0,
        ..Default::default()
    };

    let fault = catalog
        .page_fault_scoped(&query, &cfg, &["no_such_file_anywhere"])
        .unwrap();
    // Falling back beats returning nothing when a path hint is wrong.
    assert!(fault.faulted(), "fallback did not happen");
}

#[test]
fn scoped_search_ignores_unknown_ids_instead_of_panicking() {
    let (_tmp, dir) = build_fixture("badids");
    let catalog = Catalog::open(&dir).unwrap();
    let embedder = HashEmbedder::new(catalog.dim());
    let query = embedder.embed("mailbox").unwrap();

    // turbovec's allowlist path panics on an unknown id; the wrapper
    // must filter them out.
    let hits = catalog
        .search_scoped(&query, 4, &[u64::MAX, 999_999, 0])
        .unwrap();
    assert!(hits.iter().all(|(_, id)| *id == 0));

    // An entirely unknown allowlist yields no hits, not a panic.
    assert!(catalog
        .search_scoped(&query, 4, &[u64::MAX])
        .unwrap()
        .is_empty());
    assert!(catalog.search_scoped(&query, 4, &[]).unwrap().is_empty());
}

#[test]
fn overlapping_chunks_are_not_injected_twice() {
    let (_tmp, dir) = build_fixture("dedupe");
    let catalog = Catalog::open(&dir).unwrap();
    let embedder = HashEmbedder::new(catalog.dim());
    let query = embedder.embed("apple mbox inbox fifo dispatch irq").unwrap();

    let cfg = RetrievalConfig {
        fault_threshold: 0.0,
        relative_floor: 0.0,
        candidates: 64,
        max_chunks: 20,
        token_budget: u32::MAX,
    };
    let fault = catalog.page_fault(&query, &cfg).unwrap();

    for (i, a) in fault.hits.iter().enumerate() {
        for b in &fault.hits[i + 1..] {
            let overlaps =
                a.path == b.path && a.line_start <= b.line_end && a.line_end >= b.line_start;
            assert!(
                !overlaps,
                "{}:{}-{} overlaps {}:{}-{}",
                a.path, a.line_start, a.line_end, b.path, b.line_start, b.line_end
            );
        }
    }
}

#[test]
fn rendered_context_cites_every_injected_chunk() {
    let (_tmp, dir) = build_fixture("render");
    let catalog = Catalog::open(&dir).unwrap();
    let embedder = HashEmbedder::new(catalog.dim());
    let query = embedder.embed("translate xcodebuild to clang").unwrap();

    let cfg = RetrievalConfig {
        fault_threshold: 0.0,
        ..Default::default()
    };
    let fault = catalog.page_fault(&query, &cfg).unwrap();
    let rendered = fault.render_context();

    for hit in &fault.hits {
        let citation = format!("{}:{}-{}", hit.path, hit.line_start, hit.line_end);
        assert!(
            rendered.contains(&citation),
            "context is missing citation {citation}"
        );
        // The text itself must be present, or the citation is a lie.
        let first_line = hit.text.lines().next().unwrap_or("");
        if !first_line.trim().is_empty() {
            assert!(rendered.contains(first_line));
        }
    }
}

#[test]
fn dimension_mismatch_is_rejected() {
    let (_tmp, dir) = build_fixture("dim");
    let catalog = Catalog::open(&dir).unwrap();

    let wrong = vec![0.1f32; catalog.dim() + 1];
    assert!(catalog.search(&wrong, 4).is_err());
    assert!(catalog
        .page_fault(&wrong, &RetrievalConfig::default())
        .is_err());
}

#[test]
fn a_different_embedder_is_rejected_rather_than_silently_wrong() {
    let (_tmp, dir) = build_fixture("embedder");
    let catalog = Catalog::open(&dir).unwrap();

    // Same family, different dimension — so a different vector space.
    let other = HashEmbedder::new(TEST_DIM * 2);
    assert_ne!(other.id(), HashEmbedder::new(TEST_DIM).id());
    let err = catalog.check_embedder(&other.id()).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("meaningless"), "unhelpful error: {msg}");
}

#[test]
fn unknown_chunk_id_is_an_error_not_a_panic() {
    let (_tmp, dir) = build_fixture("unknown");
    let catalog = Catalog::open(&dir).unwrap();
    assert!(catalog.inflate(u64::MAX).is_err());
    assert!(catalog.chunk_path(u64::MAX).is_err());
}

#[test]
fn corrupt_container_is_detected_not_mmapped_blindly() {
    let (_tmp, dir) = build_fixture("corrupt");

    // Flip a byte deep in the payload section and confirm the integrity
    // check notices. A silent corruption here would surface as garbled
    // code injected into a prompt.
    let container = dir.join(libaim::CONTAINER_FILE);
    let mut bytes = std::fs::read(&container).unwrap();
    let last = bytes.len() - 1;
    bytes[last] ^= 0xff;
    std::fs::write(&container, &bytes).unwrap();

    let catalog = Catalog::open(&dir).unwrap();
    assert!(
        catalog.verify_integrity().is_err(),
        "corruption went undetected"
    );
}

#[test]
fn truncated_container_fails_to_open() {
    let (_tmp, dir) = build_fixture("truncated");
    let container = dir.join(libaim::CONTAINER_FILE);
    let bytes = std::fs::read(&container).unwrap();
    std::fs::write(&container, &bytes[..bytes.len() / 2]).unwrap();

    assert!(
        Catalog::open(&dir).is_err(),
        "a half-length catalog opened successfully"
    );
}

#[test]
fn reindexing_replaces_the_catalog_atomically() {
    let tmp = TempDir::new("reindex");
    write_fixture(tmp.path());
    let embedder = HashEmbedder::new(TEST_DIM);
    let opts = IndexOptions::default();

    let first = index_workspace(tmp.path(), &embedder, &opts, |_, _| {}).unwrap();
    // Add a file, then re-index over the top of the existing catalog.
    std::fs::write(
        tmp.path().join("hw/misc/extra_device.c"),
        "static void extra_device_reset(void) { /* zero the registers */ }\n",
    )
    .unwrap();
    let second = index_workspace(tmp.path(), &embedder, &opts, |_, _| {}).unwrap();

    assert!(second.meta.chunk_count > first.meta.chunk_count);
    // No staging directory left behind.
    assert!(!tmp.path().join(".aim-staging").exists());

    let catalog = Catalog::open(tmp.path().join(".aim")).unwrap();
    catalog.verify_integrity().unwrap();
    assert!(catalog.paths().any(|p| p.contains("extra_device.c")));
}

#[test]
fn empty_workspace_is_a_clear_error() {
    let tmp = TempDir::new("emptyws");
    let embedder = HashEmbedder::new(256);
    let err = index_workspace(tmp.path(), &embedder, &IndexOptions::default(), |_, _| {})
        .unwrap_err();
    assert!(matches!(err, libaim::AimError::EmptyCatalog), "{err:?}");
}

#[test]
fn gist_is_a_unit_vector_readable_from_the_mmap() {
    let (_tmp, dir) = build_fixture("gist");
    let catalog = Catalog::open(&dir).unwrap();

    let gist = catalog.gist();
    assert_eq!(gist.len(), catalog.dim());
    let norm = gist.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!(
        (norm - 1.0).abs() < 1e-4,
        "gist is not unit length: {norm}"
    );
    assert!(gist.iter().all(|x| x.is_finite()));
}

#[test]
fn heat_tracked_pinning_reports_honestly() {
    let (_tmp, dir) = build_fixture("pin");
    let catalog = Catalog::open(&dir).unwrap();
    let embedder = HashEmbedder::new(catalog.dim());

    let mut heat = libaim::HeatMap::default();
    let cfg = RetrievalConfig {
        fault_threshold: 0.0,
        ..Default::default()
    };
    for q in ["apple mbox irq", "apple mbox drain", "pl011 fifo"] {
        let v = embedder.embed(q).unwrap();
        let fault = catalog.page_fault(&v, &cfg).unwrap();
        heat.record_query(fault.hits.iter().map(|h| h.chunk_id));
    }
    assert!(heat.tracked_chunks() > 0);

    let hottest: Vec<u64> = heat.hottest(4).into_iter().map(|(id, _)| id).collect();
    let report = catalog.pin_chunks(&hottest, 1024 * 1024);
    assert_eq!(report.requested, hottest.len());
    // Pinning may be refused by OS quota; what must hold is that the
    // report is internally consistent either way.
    assert!(report.pinned <= report.requested);
    if report.pinned == 0 {
        assert!(!report.failures.is_empty(), "silent pinning failure");
    }
}

#[test]
fn pinning_respects_its_byte_budget() {
    let (_tmp, dir) = build_fixture("pinbudget");
    let catalog = Catalog::open(&dir).unwrap();
    let all: Vec<u64> = (0..catalog.len() as u64).collect();

    // Zero budget must pin nothing at all.
    let report = catalog.pin_chunks(&all, 0);
    assert_eq!(report.pinned, 0);
    assert_eq!(report.pinned_bytes, 0);
}

#[test]
fn payload_compresses_the_chunk_text() {
    let (_tmp, dir) = build_fixture("compression");
    let catalog = Catalog::open(&dir).unwrap();
    let m = catalog.meta();

    assert!(m.source_bytes > 0);
    assert!(
        m.payload_bytes < m.source_bytes,
        "payload ({} B) did not compress chunk text ({} B)",
        m.payload_bytes,
        m.source_bytes
    );

    // The container is deliberately *not* compared to the source here.
    // Fixed overhead — a `dim * 4` gist plus a 128-byte header plus 64
    // bytes per chunk — exceeds the source on a fixture this small, and
    // only amortizes on a real workspace. Asserting otherwise would
    // encode a claim that is false at small scale.
    let fixed = (catalog.dim() * 4 + 128 + catalog.len() * 64) as u64;
    assert!(
        m.container_bytes <= m.payload_bytes + fixed + 4096,
        "container ({} B) is larger than payload ({} B) plus fixed overhead ({} B)",
        m.container_bytes,
        m.payload_bytes,
        fixed
    );
}

// ---------------------------------------------------------------------
// Delta layer: keeping retrieval current between full rebuilds
// ---------------------------------------------------------------------

#[test]
fn editing_a_file_stops_the_stale_version_being_retrieved() {
    // The defect this closes: after an edit, the catalog still holds the
    // old chunks, so retrieval hands the model text and line numbers
    // that no longer exist — which reads exactly like a hallucination.
    let (tmp, dir) = build_fixture("delta_stale");
    let catalog = Catalog::open(&dir).unwrap();
    let embedder = catalog.query_embedder();
    let mut live = libaim::LiveCatalog::new(catalog);

    let query = embedder.embed("apple_mbox drain inbox fifo").unwrap();
    let cfg = RetrievalConfig {
        fault_threshold: 0.0,
        relative_floor: 0.0,
        ..Default::default()
    };

    // Before the edit, the base catalog serves the original text.
    let before = live.page_fault(&query, &cfg).unwrap();
    assert!(
        before.hits.iter().any(|h| h.text.contains("apple_mbox_drain_inbox")),
        "fixture text missing from base retrieval"
    );

    // Rewrite the file, removing the function entirely.
    let path = "hw/misc/apple_mbox.c";
    let rewritten = "/* rewritten: the drain helper was removed */\n\
                     static void apple_mbox_reset(AppleMboxState *s) { s->count = 0; }\n";
    std::fs::write(tmp.path().join(path), rewritten).unwrap();
    live.ingest_file(path, rewritten, &embedder, &ChunkConfig::default())
        .unwrap();

    let after = live.page_fault(&query, &cfg).unwrap();
    for hit in &after.hits {
        assert!(
            !hit.text.contains("apple_mbox_drain_inbox"),
            "stale text from {} still retrieved after the edit",
            hit.path
        );
    }
    assert!(
        after.hits.iter().any(|h| h.path == path && h.text.contains("apple_mbox_reset")),
        "the edited file's new content was not retrieved: {:?}",
        after.hits.iter().map(|h| (&h.path, h.line_start)).collect::<Vec<_>>()
    );
}

#[test]
fn delta_line_numbers_match_the_edited_file_on_disk() {
    let (tmp, dir) = build_fixture("delta_lines");
    let catalog = Catalog::open(&dir).unwrap();
    let embedder = catalog.query_embedder();
    let mut live = libaim::LiveCatalog::new(catalog);

    let path = "hw/misc/apple_mbox.c";
    let rewritten = (1..=40)
        .map(|i| format!("int marker_{i}(void) {{ return {i}; }}"))
        .collect::<Vec<_>>()
        .join("\n");
    std::fs::write(tmp.path().join(path), &rewritten).unwrap();
    live.ingest_file(path, &rewritten, &embedder, &ChunkConfig::default())
        .unwrap();

    let query = embedder.embed("marker_17 return value").unwrap();
    let cfg = RetrievalConfig {
        fault_threshold: 0.0,
        relative_floor: 0.0,
        ..Default::default()
    };
    let fault = live.page_fault(&query, &cfg).unwrap();

    let on_disk: Vec<&str> = rewritten.lines().collect();
    for hit in fault.hits.iter().filter(|h| h.path == path) {
        // Citations must point at the file as it exists now.
        assert!(hit.line_end as usize <= on_disk.len());
        let expected = on_disk[hit.line_start as usize - 1];
        let actual = hit.text.lines().next().unwrap_or("");
        assert_eq!(expected, actual, "delta hit cites the wrong line");
    }
}

#[test]
fn deleting_a_file_removes_it_from_retrieval() {
    let (_tmp, dir) = build_fixture("delta_delete");
    let catalog = Catalog::open(&dir).unwrap();
    let embedder = catalog.query_embedder();
    let mut live = libaim::LiveCatalog::new(catalog);

    let cfg = RetrievalConfig {
        fault_threshold: 0.0,
        relative_floor: 0.0,
        ..Default::default()
    };
    let query = embedder.embed("pl011 write fifo read count").unwrap();

    assert!(
        live.page_fault(&query, &cfg)
            .unwrap()
            .hits
            .iter()
            .any(|h| h.path.contains("serial_pl011")),
        "fixture file not retrievable before deletion"
    );

    live.delta_mut().remove_file("hw/char/serial_pl011.c");

    assert!(
        !live.page_fault(&query, &cfg)
            .unwrap()
            .hits
            .iter()
            .any(|h| h.path.contains("serial_pl011")),
        "deleted file still retrieved"
    );
}

#[test]
fn unedited_files_still_come_from_the_base_catalog() {
    // The delta must not shadow anything it was not asked to.
    let (_tmp, dir) = build_fixture("delta_isolation");
    let catalog = Catalog::open(&dir).unwrap();
    let embedder = catalog.query_embedder();
    let mut live = libaim::LiveCatalog::new(catalog);

    live.ingest_file(
        "hw/misc/apple_mbox.c",
        "static void unrelated(void) {}\n",
        &embedder,
        &ChunkConfig::default(),
    )
    .unwrap();

    let cfg = RetrievalConfig {
        fault_threshold: 0.0,
        relative_floor: 0.0,
        ..Default::default()
    };
    let query = embedder.embed("translate xcodebuild clang arm64 apple ios").unwrap();
    let fault = live.page_fault(&query, &cfg).unwrap();

    assert!(
        fault.hits.iter().any(|h| h.path.contains("shim.rs")),
        "an untouched file stopped being retrievable: {:?}",
        fault.hits.iter().map(|h| &h.path).collect::<Vec<_>>()
    );
}

#[test]
fn gist_stays_a_unit_vector_after_edits() {
    let (_tmp, dir) = build_fixture("delta_gist");
    let catalog = Catalog::open(&dir).unwrap();
    let embedder = catalog.query_embedder();
    let mut live = libaim::LiveCatalog::new(catalog);

    live.ingest_file(
        "hw/misc/apple_mbox.c",
        "static void replaced(void) { int x = 1; }\n",
        &embedder,
        &ChunkConfig::default(),
    )
    .unwrap();

    let gist = live.gist();
    assert_eq!(gist.len(), live.dim());
    let norm = gist.iter().map(|x| x * x).sum::<f32>().sqrt();
    assert!((norm - 1.0).abs() < 1e-4, "gist norm drifted to {norm}");
    assert!(gist.iter().all(|x| x.is_finite()));
}

#[test]
fn watcher_picks_up_an_edit_and_refreshes_retrieval() {
    use std::sync::{Arc, RwLock};
    use std::time::{Duration, Instant};

    let (tmp, dir) = build_fixture("watch_edit");
    let catalog = Catalog::open(&dir).unwrap();
    let embedder = catalog.query_embedder();
    let live = Arc::new(RwLock::new(libaim::LiveCatalog::new(catalog)));

    let cfg = libaim::WatchConfig {
        debounce: Duration::from_millis(150),
        ..Default::default()
    };
    let handle = libaim::start_watcher(tmp.path(), live.clone(), cfg).unwrap();

    // Replace the mailbox file with content containing a distinctive
    // marker that cannot appear in the base catalog.
    let path = "hw/misc/apple_mbox.c";
    let rewritten = "static void watcher_marker_fn(void) { /* freshly written */ }\n";
    std::fs::write(tmp.path().join(path), rewritten).unwrap();

    // Filesystem events are inherently asynchronous, so poll rather than
    // sleeping a fixed amount.
    let deadline = Instant::now() + Duration::from_secs(20);
    loop {
        if handle.stats().snapshot().0 > 0 {
            break;
        }
        assert!(
            Instant::now() < deadline,
            "watcher never ingested the edit (stats: {:?})",
            handle.stats().snapshot()
        );
        std::thread::sleep(Duration::from_millis(50));
    }

    let query = embedder.embed("watcher_marker_fn freshly written").unwrap();
    let retrieval = RetrievalConfig {
        fault_threshold: 0.0,
        relative_floor: 0.0,
        ..Default::default()
    };
    let fault = live.read().unwrap().page_fault(&query, &retrieval).unwrap();

    assert!(
        fault.hits.iter().any(|h| h.text.contains("watcher_marker_fn")),
        "new content not retrievable after the watcher ran: {:?}",
        fault.hits.iter().map(|h| (&h.path, h.line_start)).collect::<Vec<_>>()
    );
    assert!(
        !fault.hits.iter().any(|h| h.text.contains("apple_mbox_drain_inbox")),
        "stale content still retrievable after the watcher ran"
    );

    handle.stop();
}

#[test]
fn watcher_ignores_build_output_and_binaries() {
    use std::sync::{Arc, RwLock};
    use std::time::Duration;

    let (tmp, dir) = build_fixture("watch_ignore");
    let catalog = Catalog::open(&dir).unwrap();
    let live = Arc::new(RwLock::new(libaim::LiveCatalog::new(catalog)));

    let handle = libaim::start_watcher(
        tmp.path(),
        live.clone(),
        libaim::WatchConfig {
            debounce: Duration::from_millis(100),
            ..Default::default()
        },
    )
    .unwrap();

    // None of these may reach the delta layer.
    std::fs::write(tmp.path().join("target/debug/build.rs"), "fn main() { }").unwrap();
    std::fs::write(tmp.path().join("logo.png"), [0u8, 1, 2, 3]).unwrap();
    std::fs::create_dir_all(tmp.path().join("node_modules/pkg")).unwrap();
    std::fs::write(tmp.path().join("node_modules/pkg/index.js"), "module.exports={}").unwrap();

    std::thread::sleep(Duration::from_millis(900));

    assert_eq!(
        handle.stats().snapshot().0,
        0,
        "watcher ingested a file it should have filtered"
    );
    assert!(live.read().unwrap().delta().is_empty());

    handle.stop();
}
