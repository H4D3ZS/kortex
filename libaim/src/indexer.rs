//! Walking a workspace and building a catalog from it.

use std::path::{Path, PathBuf};
use std::time::Instant;

use rayon::prelude::*;

use crate::catalog::{CatalogBuilder, CatalogMeta};
use crate::chunk::{chunk_source, has_indexable_extension, ChunkConfig, DEFAULT_IGNORED_DIRS};
use crate::embed::{Embedder, HashEmbedder};
use crate::error::AimError;
use crate::format::DEFAULT_BIT_WIDTH;

/// Knobs for [`index_workspace`].
#[derive(Debug, Clone)]
pub struct IndexOptions {
    /// Where to write the catalog. Defaults to `<root>/.aim`.
    pub out_dir: Option<PathBuf>,
    pub chunk: ChunkConfig,
    pub bit_width: usize,
    /// Skip files larger than this. Generated and vendored blobs
    /// contribute chunks that are never retrieved but cost index space.
    pub max_file_bytes: u64,
    /// Extra directory names to skip, on top of
    /// [`DEFAULT_IGNORED_DIRS`].
    pub extra_ignored_dirs: Vec<String>,
}

impl Default for IndexOptions {
    fn default() -> Self {
        Self {
            out_dir: None,
            chunk: ChunkConfig::default(),
            bit_width: DEFAULT_BIT_WIDTH,
            max_file_bytes: 2 * 1024 * 1024,
            extra_ignored_dirs: Vec::new(),
        }
    }
}

/// What an indexing run did.
#[derive(Debug, Clone)]
pub struct IndexStats {
    pub meta: CatalogMeta,
    pub files_scanned: usize,
    pub files_indexed: usize,
    pub files_skipped_too_large: usize,
    pub files_unreadable: usize,
    pub elapsed_secs: f64,
    /// Container bytes divided by indexed source bytes.
    pub compression_ratio: f64,
}

/// Index every source file under `root` into a `.aim` catalog.
///
/// `progress` is called with `(files_done, chunks_so_far)` roughly once
/// per file so a CLI can report movement on a large tree.
pub fn index_workspace<E: Embedder + ?Sized>(
    root: impl AsRef<Path>,
    embedder: &E,
    opts: &IndexOptions,
    mut progress: impl FnMut(usize, usize),
) -> Result<IndexStats, AimError> {
    let started = Instant::now();
    let root = root.as_ref();
    let root = root
        .canonicalize()
        .map_err(|e| AimError::io("resolve workspace root", root, e))?;
    let out_dir = opts
        .out_dir
        .clone()
        .unwrap_or_else(|| root.join(".aim"));

    let files = collect_files(&root, opts);
    let mut stats_scanned = files.len();
    let mut skipped_large = 0usize;
    let mut unreadable = 0usize;

    let mut builder =
        CatalogBuilder::new(&root, embedder.dim(), embedder.id()).with_bit_width(opts.bit_width);

    let mut indexed = 0usize;
    for path in &files {
        let Ok(meta) = std::fs::metadata(path) else {
            unreadable += 1;
            continue;
        };
        if meta.len() > opts.max_file_bytes {
            skipped_large += 1;
            continue;
        }
        // Non-UTF-8 files are skipped rather than lossily converted:
        // mojibake in the index produces hits whose text is wrong.
        let Ok(contents) = std::fs::read_to_string(path) else {
            unreadable += 1;
            continue;
        };

        let rel = path
            .strip_prefix(&root)
            .unwrap_or(path)
            .to_string_lossy()
            .to_string();
        let chunks = chunk_source(&rel, &contents, &opts.chunk);
        if chunks.is_empty() {
            continue;
        }

        // Embedding is the bottleneck and every chunk is independent, so
        // fan out across the 24 threads rather than the disk walk.
        let vectors: Result<Vec<Vec<f32>>, AimError> = chunks
            .par_iter()
            .map(|c| embed_chunk(embedder, c))
            .collect();
        let vectors = vectors?;

        for (chunk, vector) in chunks.into_iter().zip(vectors) {
            builder.push(chunk, &vector)?;
        }
        builder.note_file();
        indexed += 1;
        progress(indexed, builder.len());
    }

    if builder.is_empty() {
        return Err(AimError::EmptyCatalog);
    }

    // Write to a sibling directory and swap, so a failed or interrupted
    // run cannot leave a half-written catalog where readers will mmap it.
    let staging = out_dir.with_extension("aim-staging");
    if staging.exists() {
        std::fs::remove_dir_all(&staging)
            .map_err(|e| AimError::io("clear staging directory", &staging, e))?;
    }
    let meta = builder.write(&staging)?;

    if out_dir.exists() {
        std::fs::remove_dir_all(&out_dir)
            .map_err(|e| AimError::io("remove previous catalog", &out_dir, e))?;
    }
    std::fs::rename(&staging, &out_dir)
        .map_err(|e| AimError::io("move catalog into place", &out_dir, e))?;

    stats_scanned = stats_scanned.max(indexed);
    let ratio = if meta.source_bytes > 0 {
        meta.container_bytes as f64 / meta.source_bytes as f64
    } else {
        0.0
    };

    Ok(IndexStats {
        meta,
        files_scanned: stats_scanned,
        files_indexed: indexed,
        files_skipped_too_large: skipped_large,
        files_unreadable: unreadable,
        elapsed_secs: started.elapsed().as_secs_f64(),
        compression_ratio: ratio,
    })
}

/// Embed a chunk, using the path-weighted variant when available.
fn embed_chunk<E: Embedder + ?Sized>(
    embedder: &E,
    chunk: &crate::chunk::SourceChunk,
) -> Result<Vec<f32>, AimError> {
    // `HashEmbedder` can weight path tokens, which materially improves
    // hits for queries that name a file. Dense backends have no such
    // hook, so the path is prepended to the text instead — same
    // information, expressed the only way a general embedder can use it.
    if let Some(h) = as_hash_embedder(embedder) {
        h.embed_with_path(&chunk.path, &chunk.text)
    } else {
        embedder.embed(&format!("{}\n{}", chunk.path, chunk.text))
    }
}

/// Downcast to [`HashEmbedder`] by comparing [`Embedder::id`].
///
/// A real `Any` downcast would need `Embedder: 'static + Any`, which
/// would force that bound on every implementor including borrowed ones.
/// Reconstructing the embedder is sound because `HashEmbedder` is
/// stateless apart from its dimension, which the id encodes.
fn as_hash_embedder<E: Embedder + ?Sized>(embedder: &E) -> Option<HashEmbedder> {
    let id = embedder.id();
    let expected = HashEmbedder::new(embedder.dim()).id();
    (id == expected).then(|| HashEmbedder::new(embedder.dim()))
}

/// Recursively collect indexable files, skipping ignored directories.
fn collect_files(root: &Path, opts: &IndexOptions) -> Vec<PathBuf> {
    let ignored: Vec<&str> = DEFAULT_IGNORED_DIRS
        .iter()
        .copied()
        .chain(opts.extra_ignored_dirs.iter().map(|s| s.as_str()))
        .collect();

    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];

    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let Ok(file_type) = entry.file_type() else {
                continue;
            };

            // Not following symlinks: a link pointing at a parent turns
            // the walk into an infinite loop, and one pointing outside
            // the workspace would index files the user did not offer.
            if file_type.is_symlink() {
                continue;
            }

            if file_type.is_dir() {
                let name = entry.file_name().to_string_lossy().to_string();
                if ignored.contains(&name.as_str()) || name.starts_with('.') {
                    continue;
                }
                stack.push(path);
            } else if file_type.is_file() && has_indexable_extension(&path) {
                out.push(path);
            }
        }
    }

    // Deterministic order keeps chunk ids stable between runs over an
    // unchanged tree, which is what makes a heat map or a pinned set
    // still meaningful after a re-index.
    out.sort();
    out
}
