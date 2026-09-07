//! `libaim` — the `.aim` catalog: a compressed, searchable index of a
//! workspace that resolves a query to the handful of source excerpts
//! that actually matter.
//!
//! # What this replaces
//!
//! The original `.aim` path dumped a fixed blob of workspace metadata
//! into every prompt. That is not retrieval — it costs the same tokens
//! on every request and gets less useful as the workspace grows. This
//! crate does the real thing: embed the query, search a quantized index,
//! and inflate only the chunks that clear a similarity threshold.
//!
//! # Pipeline
//!
//! ```text
//!  index time                          query time
//!  ──────────                          ──────────
//!  walk workspace                      embed query          (embed)
//!    │                                   │
//!  chunk files          (chunk)         search index         (turbovec)
//!    │                                   │
//!  embed chunks         (embed)         gate on threshold    (catalog)
//!    │                                   │
//!  quantize → .tvim     (turbovec)      inflate survivors    (zstd)
//!  compress → .aim      (catalog)        │
//!                                       render prompt block
//! ```
//!
//! Compression is turbovec's TurboQuant: each 1536-d f32 embedding
//! (6KB) becomes 4 bits per coordinate (768 bytes), a 8× reduction with
//! near-optimal distortion and no codebook training. Chunk *text* is
//! zstd-compressed separately and only decompressed on a hit.
//!
//! # Example
//!
//! ```no_run
//! use libaim::{Catalog, HashEmbedder, Embedder, RetrievalConfig};
//!
//! let catalog = Catalog::open(".aim")?;
//! let embedder = HashEmbedder::new(catalog.dim());
//! catalog.check_embedder(&embedder.id())?;
//!
//! let query = embedder.embed("fix the mailbox IRQ starvation")?;
//! let fault = catalog.page_fault(&query, &RetrievalConfig::default())?;
//! if fault.faulted() {
//!     print!("{}", fault.render_context());
//! }
//! # Ok::<(), libaim::AimError>(())
//! ```

pub mod catalog;
pub mod chunk;
pub mod delta;
pub mod embed;
#[cfg(feature = "http-embed")]
pub mod embed_http;
pub mod error;
pub mod ffi;
pub mod format;
pub mod gate;
pub mod heat;
pub mod indexer;
pub mod watch;

pub mod ivf;
pub mod ternary;
pub use catalog::{
    Catalog, CatalogBuilder, CatalogMeta, Hit, PageFaultResult, RetrievalConfig, CONTAINER_FILE,
    INDEX_FILE, META_FILE,
};
pub use chunk::{chunk_source, ChunkConfig, SourceChunk};
pub use delta::{DeltaLayer, LiveCatalog, LiveChunk};
pub use embed::{cosine, Embedder, HashEmbedder};
pub use error::AimError;
pub use format::{Header, DEFAULT_BIT_WIDTH, DEFAULT_DIM};
pub use gate::{GateDecision, QueryGate, SkipReason};
pub use heat::{HeatMap, PinReport};
pub use indexer::{index_workspace, IndexOptions, IndexStats};
pub use watch::{start_watcher, Debouncer, WatchConfig, WatchStats, WatcherHandle};

#[cfg(feature = "http-embed")]
pub use embed_http::{
    EmbedProtocol, HttpEmbedder, LEMONADE_DEFAULT_URL, OLLAMA_DEFAULT_URL,
};
