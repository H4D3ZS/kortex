//! Building and querying a `.aim` catalog.
//!
//! A catalog is three files in one directory:
//!
//! | file           | contents                                          |
//! |----------------|---------------------------------------------------|
//! | `catalog.aim`  | container: chunk table, path heap, gist, payload   |
//! | `catalog.tvim` | turbovec `IdMapIndex` — one vector per chunk       |
//! | `catalog.json` | which embedder built it, and with what settings    |
//!
//! The read path is the part that has to be fast, so it is built around
//! a single memory map: [`Catalog::open`] mmaps the container, decodes
//! the fixed-size chunk table once, and after that resolving a search
//! hit to source text is a slice into the map plus one zstd frame.
//!
//! `catalog.json` exists because the most damaging failure mode in a
//! retrieval system is silent: query with a different embedder than the
//! one that built the index and every score is meaningless noise, but
//! nothing errors. [`Catalog::check_embedder`] turns that into a loud
//! error.

use std::collections::HashMap;
use std::fs::File;
use std::io::Write;
use std::path::{Path, PathBuf};

use memmap2::Mmap;
use turbovec::IdMapIndex;

use crate::chunk::SourceChunk;
use crate::embed::{cosine, normalize};
use crate::error::AimError;
use crate::format::{
    align_up, ChunkRecord, Header, CHUNK_RECORD_SIZE, DEFAULT_BIT_WIDTH, FLAG_PAYLOAD_ZSTD,
    FORMAT_VERSION, HEADER_SIZE,
};

/// zstd level. 9 sits near the knee of the ratio/speed curve for source
/// text; compression happens once at index time, so paying a little
/// more here buys a smaller resident payload forever.
const ZSTD_LEVEL: i32 = 9;

/// Container file name within a catalog directory.
pub const CONTAINER_FILE: &str = "catalog.aim";
/// turbovec index file name within a catalog directory.
pub const INDEX_FILE: &str = "catalog.tvim";

/// IVF coarse-index sidecar (optional). See [`crate::ivf`].
pub const IVF_FILE: &str = "catalog.ivf";
/// Below this many chunks, IVF is skipped: the SIMD scan is already instant
/// and partitioning would only cost recall.
pub const IVF_MIN_CHUNKS: usize = 4096;
/// Metadata file name within a catalog directory.
pub const META_FILE: &str = "catalog.json";

/// Sidecar metadata describing how a catalog was built.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CatalogMeta {
    /// [`crate::embed::Embedder::id`] of the embedder that built this
    /// catalog. Must match at query time.
    pub embedder_id: String,
    /// Absolute path of the workspace root the paths are relative to.
    pub root: String,
    pub dim: usize,
    pub bit_width: usize,
    pub chunk_count: usize,
    pub file_count: usize,
    /// Total uncompressed bytes of chunk text indexed. Exceeds the size
    /// of the source files, because chunks overlap.
    pub source_bytes: u64,
    /// Size of the compressed payload section. Compare against
    /// `source_bytes` for the true compression ratio — `container_bytes`
    /// also carries fixed overhead (a `dim * 4` gist, a 128-byte header,
    /// 64 bytes per chunk) that only amortizes on a real workspace.
    pub payload_bytes: u64,
    /// Total bytes the container occupies on disk.
    pub container_bytes: u64,
    pub built_unix_secs: u64,
}

/// Accumulates chunks and their embeddings, then writes a catalog.
///
/// Vectors are handed to turbovec in a single `add_with_ids` call at
/// write time rather than incrementally, because turbovec freezes its
/// TQ+ per-coordinate calibration on the first add — one batch means the
/// calibration sees the whole corpus.
pub struct CatalogBuilder {
    dim: usize,
    bit_width: usize,
    embedder_id: String,
    root: PathBuf,
    chunks: Vec<SourceChunk>,
    vectors: Vec<f32>,
    file_count: usize,
}

impl CatalogBuilder {
    /// Start a catalog for `root`, embedding with `embedder_id` vectors
    /// of length `dim`.
    pub fn new(root: impl Into<PathBuf>, dim: usize, embedder_id: String) -> Self {
        Self {
            dim,
            bit_width: DEFAULT_BIT_WIDTH,
            embedder_id,
            root: root.into(),
            chunks: Vec::new(),
            vectors: Vec::new(),
            file_count: 0,
        }
    }

    /// Override the turbovec quantizer bit width (2–4).
    pub fn with_bit_width(mut self, bit_width: usize) -> Self {
        self.bit_width = bit_width;
        self
    }

    /// Record that one more source file was visited. Metadata only.
    pub fn note_file(&mut self) {
        self.file_count += 1;
    }

    /// Add a chunk and its embedding.
    ///
    /// The vector must already be L2-normalized — the fault threshold is
    /// interpreted as a cosine similarity, which only holds for unit
    /// vectors.
    pub fn push(&mut self, chunk: SourceChunk, vector: &[f32]) -> Result<(), AimError> {
        if vector.len() != self.dim {
            return Err(AimError::DimMismatch {
                catalog: self.dim,
                got: vector.len(),
            });
        }
        self.chunks.push(chunk);
        self.vectors.extend_from_slice(vector);
        Ok(())
    }

    /// Number of chunks accumulated so far.
    pub fn len(&self) -> usize {
        self.chunks.len()
    }

    /// True when no chunks have been added.
    pub fn is_empty(&self) -> bool {
        self.chunks.is_empty()
    }

    /// Write the container, index and metadata into `dir`.
    pub fn write(mut self, dir: impl AsRef<Path>) -> Result<CatalogMeta, AimError> {
        let dir = dir.as_ref();
        if self.chunks.is_empty() {
            return Err(AimError::EmptyCatalog);
        }
        std::fs::create_dir_all(dir).map_err(|e| AimError::io("create catalog directory", dir, e))?;

        // ---- Sections -------------------------------------------------
        // Built in memory in the same order they land on disk, so the
        // running `offset` below is the single source of truth for where
        // each one starts.
        let mut strings: Vec<u8> = Vec::new();
        let mut payload: Vec<u8> = Vec::new();
        let mut records: Vec<ChunkRecord> = Vec::with_capacity(self.chunks.len());
        let mut ids: Vec<u64> = Vec::with_capacity(self.chunks.len());
        let mut source_bytes = 0u64;

        // Interning paths keeps the heap at one copy per file rather than
        // one per chunk; with ~10 chunks/file that is most of the heap.
        let mut path_offsets: HashMap<&str, (u32, u32)> = HashMap::new();

        for (i, chunk) in self.chunks.iter().enumerate() {
            let (path_off, path_len) = match path_offsets.get(chunk.path.as_str()) {
                Some(&hit) => hit,
                None => {
                    let entry = (strings.len() as u32, chunk.path.len() as u32);
                    strings.extend_from_slice(chunk.path.as_bytes());
                    path_offsets.insert(chunk.path.as_str(), entry);
                    entry
                }
            };

            let raw = chunk.text.as_bytes();
            source_bytes += raw.len() as u64;
            let compressed = zstd::bulk::compress(raw, ZSTD_LEVEL)
                .map_err(|e| AimError::io("compress chunk", dir.join(CONTAINER_FILE), e))?;

            let id = i as u64;
            records.push(ChunkRecord {
                id,
                path_off,
                path_len,
                payload_off: payload.len() as u64,
                payload_len: compressed.len() as u64,
                uncompressed_len: raw.len() as u32,
                line_start: chunk.line_start,
                line_end: chunk.line_end,
                token_estimate: chunk.token_estimate(),
                hash_prefix: hash_prefix(raw),
            });
            ids.push(id);
            payload.extend_from_slice(&compressed);
        }

        // Inverse document frequency, computed over the corpus we just
        // collected, then folded into every vector.
        //
        // Without this, retrieval is actively misleading: a one-word
        // query like "hello" concentrates all its mass on a single
        // feature, so whichever chunk contains that word scores higher
        // (measured 0.40 on a 243k-chunk corpus) than a precise
        // technical query does against the file it names (0.14). IDF
        // pushes ubiquitous features toward zero weight, so a query made
        // only of common words produces a near-zero vector and retrieves
        // nothing — which is the correct answer for "hello".
        let idf = compute_idf(&self.vectors, self.dim);
        apply_idf(&mut self.vectors, self.dim, &idf);

        // L1 limbic gist: the corpus centroid. Cheap global signal for
        // "is this query about this workspace at all", without touching
        // the index.
        let gist = centroid(&self.vectors, self.dim);

        // ---- Offsets --------------------------------------------------
        let mut offset = HEADER_SIZE;
        let chunk_table_off = offset;
        let chunk_table_len = records.len() * CHUNK_RECORD_SIZE;
        offset = align_up(chunk_table_off + chunk_table_len);

        let strings_off = offset;
        offset = align_up(strings_off + strings.len());

        let gist_off = offset;
        let gist_len = self.dim * 4;
        offset = align_up(gist_off + gist_len);

        let idf_off = offset;
        let idf_len = self.dim * 4;
        offset = align_up(idf_off + idf_len);

        let payload_off = offset;
        let total_len = payload_off + payload.len();

        // ---- Body -----------------------------------------------------
        // Assembled first so its hash can go into the header.
        let mut body = vec![0u8; total_len - HEADER_SIZE];
        let put = |body: &mut Vec<u8>, at: usize, bytes: &[u8]| {
            let at = at - HEADER_SIZE;
            body[at..at + bytes.len()].copy_from_slice(bytes);
        };
        for (i, rec) in records.iter().enumerate() {
            put(&mut body, chunk_table_off + i * CHUNK_RECORD_SIZE, &rec.encode());
        }
        put(&mut body, strings_off, &strings);
        let gist_bytes: Vec<u8> = gist.iter().flat_map(|f| f.to_le_bytes()).collect();
        put(&mut body, gist_off, &gist_bytes);
        let idf_bytes: Vec<u8> = idf.iter().flat_map(|f| f.to_le_bytes()).collect();
        put(&mut body, idf_off, &idf_bytes);
        put(&mut body, payload_off, &payload);

        let header = Header {
            format_version: FORMAT_VERSION,
            flags: FLAG_PAYLOAD_ZSTD,
            dim: self.dim as u32,
            bit_width: self.bit_width as u32,
            chunk_count: records.len() as u64,
            chunk_table_off: chunk_table_off as u64,
            chunk_table_len: chunk_table_len as u64,
            strings_off: strings_off as u64,
            strings_len: strings.len() as u64,
            gist_off: gist_off as u64,
            gist_len: gist_len as u64,
            idf_off: idf_off as u64,
            idf_len: idf_len as u64,
            payload_off: payload_off as u64,
            payload_len: payload.len() as u64,
            built_unix_secs: Header::now_unix_secs(),
            content_hash: hash_prefix(&body),
        };

        let container_path = dir.join(CONTAINER_FILE);
        {
            let mut f = File::create(&container_path)
                .map_err(|e| AimError::io("create container", &container_path, e))?;
            f.write_all(&header.encode())
                .and_then(|_| f.write_all(&body))
                .and_then(|_| f.flush())
                .map_err(|e| AimError::io("write container", &container_path, e))?;
        }

        // ---- turbovec index -------------------------------------------
        let mut index = IdMapIndex::new(self.dim, self.bit_width)
            .map_err(|e| AimError::Index(format!("{e:?}")))?;
        index
            .add_with_ids(&self.vectors, &ids)
            .map_err(|e| AimError::Index(format!("{e:?}")))?;
        let index_path = dir.join(INDEX_FILE);
        index
            .write(&index_path)
            .map_err(|e| AimError::io("write turbovec index", &index_path, e))?;

        // ---- IVF coarse index (sidecar) -------------------------------
        // Only for large corpora — below IVF_MIN_CHUNKS the SIMD scan is
        // already instant. sqrt(N) partitions is the standard IVF-flat choice.
        // self.vectors are the IDF-weighted, normalized vectors that were
        // indexed, so the IVF partitions live in the same space queries land in.
        if records.len() >= IVF_MIN_CHUNKS {
            let n_parts = (records.len() as f64).sqrt().round() as usize;
            if let Some(ivf) = crate::ivf::IvfIndex::build(&self.vectors, &ids, self.dim, n_parts) {
                let ivf_path = dir.join(IVF_FILE);
                ivf.write(&ivf_path)
                    .map_err(|e| AimError::io("write IVF sidecar", &ivf_path, e))?;
            }
        }

        let meta = CatalogMeta {
            embedder_id: self.embedder_id,
            root: self.root.to_string_lossy().to_string(),
            dim: self.dim,
            bit_width: self.bit_width,
            chunk_count: records.len(),
            file_count: self.file_count,
            source_bytes,
            payload_bytes: payload.len() as u64,
            container_bytes: total_len as u64,
            built_unix_secs: header.built_unix_secs,
        };
        let meta_path = dir.join(META_FILE);
        let meta_json = serde_json::to_vec_pretty(&meta)
            .map_err(|e| AimError::Index(format!("serialize metadata: {e}")))?;
        std::fs::write(&meta_path, meta_json)
            .map_err(|e| AimError::io("write metadata", &meta_path, e))?;

        Ok(meta)
    }
}

/// How aggressively to retrieve.
#[derive(Debug, Clone, Copy)]
pub struct RetrievalConfig {
    /// Absolute cosine floor. Chunks below this never qualify, however
    /// good they look relative to the rest.
    ///
    /// Keep this low. It exists to reject "the corpus contains nothing
    /// about this", not to rank. The real selectivity comes from
    /// [`Self::relative_floor`].
    ///
    /// Note this is a *cosine*, not the 0.85 attention-activation
    /// threshold from the design notes — different quantity, different
    /// scale.
    pub fault_threshold: f32,

    /// Fraction of the best score a chunk must reach to qualify.
    ///
    /// This is the load-bearing gate, because absolute cosine magnitude
    /// tracks *query length*, not relevance: on a 243k-chunk corpus the
    /// same relevant chunk scores ~0.08 against a three-word query and
    /// ~0.33 against a nine-word one. A fixed absolute threshold tuned
    /// for one shape silently retrieves nothing for the other. Gating on
    /// a fraction of the best score is scale-free, so one setting works
    /// for both.
    ///
    /// 1.0 keeps only ties with the best hit; 0.0 disables the relative
    /// gate and falls back to the absolute floor alone.
    pub relative_floor: f32,
    /// How many candidates to pull from the index before budgeting.
    pub candidates: usize,
    /// Hard cap on chunks injected into one prompt.
    pub max_chunks: usize,
    /// Hard cap on estimated tokens injected into one prompt. This is
    /// the number that keeps retrieval from re-inflating the context it
    /// exists to shrink.
    pub token_budget: u32,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            fault_threshold: 0.05,
            relative_floor: 0.6,
            candidates: 48,
            max_chunks: 8,
            token_budget: 6000,
        }
    }
}

/// One retrieved chunk, with its source text inflated.
#[derive(Debug, Clone)]
pub struct Hit {
    pub chunk_id: u64,
    pub path: String,
    pub line_start: u32,
    pub line_end: u32,
    /// Cosine similarity against the query, as estimated by turbovec.
    pub score: f32,
    pub token_estimate: u32,
    pub text: String,
}

/// Outcome of a retrieval attempt.
#[derive(Debug, Clone)]
pub struct PageFaultResult {
    /// Chunks that cleared the threshold and fit the budget, best first.
    pub hits: Vec<Hit>,
    /// Best score seen, threshold or not. Useful for tuning and for
    /// logging why nothing was injected.
    pub best_score: f32,
    /// The score a chunk actually had to beat, after combining the
    /// absolute floor with the relative one. Reported so a "why did
    /// nothing come back" question has a direct answer.
    pub cutoff: f32,
    /// Cosine of the query against the corpus centroid.
    pub gist_score: f32,
    /// Candidates that cleared the threshold but were dropped by the
    /// chunk or token budget.
    pub dropped_to_budget: usize,
}

impl PageFaultResult {
    /// True when at least one chunk is worth injecting.
    pub fn faulted(&self) -> bool {
        !self.hits.is_empty()
    }

    /// Sum of the token estimates actually being injected.
    pub fn injected_tokens(&self) -> u32 {
        self.hits.iter().map(|h| h.token_estimate).sum()
    }

    /// Render the hits as a prompt block with path and line citations.
    ///
    /// Citations are not decoration: without them the model cannot tell
    /// the injected excerpt from the user's own pasted code, and cannot
    /// refer to a location in its answer.
    pub fn render_context(&self) -> String {
        if self.hits.is_empty() {
            return String::new();
        }
        let mut out = String::with_capacity(self.injected_tokens() as usize * 4);
        out.push_str(
            "[KORTEX-AIM] Retrieved workspace context. These excerpts were selected by \
             similarity to the request; line numbers are exact. Cite them as path:line when \
             you refer to them, and say so if the excerpt you need is not here.\n",
        );
        for h in &self.hits {
            out.push_str(&format!(
                "\n--- {}:{}-{} (similarity {:.3}) ---\n{}\n",
                h.path, h.line_start, h.line_end, h.score, h.text
            ));
        }
        out.push_str("\n[KORTEX-AIM] End of retrieved context.\n");
        out
    }
}

/// A memory-mapped catalog, ready to query.
pub struct Catalog {
    mmap: Mmap,
    header: Header,
    records: Vec<ChunkRecord>,
    id_to_idx: HashMap<u64, usize>,
    /// Chunk ids grouped by source path, for scoped search.
    path_to_chunks: HashMap<String, Vec<u64>>,
    index: IdMapIndex,
    meta: CatalogMeta,
    /// Optional IVF coarse index (sidecar `catalog.ivf`), memory-mapped. When
    /// present and the corpus is large, search probes only the nearest
    /// partitions (sublinear, disk-resident). Absent → full SIMD scan, so old
    /// catalogs keep working unchanged.
    ivf: Option<crate::ivf::IvfMmap>,
}

impl Catalog {
    /// Open the catalog in `dir`, mapping the container and loading the
    /// turbovec index.
    pub fn open(dir: impl AsRef<Path>) -> Result<Self, AimError> {
        let dir = dir.as_ref();
        let container_path = dir.join(CONTAINER_FILE);

        let file = File::open(&container_path)
            .map_err(|e| AimError::io("open container", &container_path, e))?;
        // Safety: the catalog is treated as immutable once written. A
        // concurrent writer truncating the file could fault on access,
        // which is why the indexer writes to a temp directory and
        // renames rather than editing in place.
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| AimError::io("memory-map container", &container_path, e))?;

        let header = Header::decode(&mmap, mmap.len())?;

        let table = &mmap[header.chunk_table_off as usize
            ..(header.chunk_table_off + header.chunk_table_len) as usize];
        let mut records = Vec::with_capacity(header.chunk_count as usize);
        let mut id_to_idx = HashMap::with_capacity(header.chunk_count as usize);
        for i in 0..header.chunk_count as usize {
            let rec = ChunkRecord::decode(&table[i * CHUNK_RECORD_SIZE..])?;
            // Validate the record's slices now, once, so the hot
            // resolve path can index without re-checking.
            let path_end = rec.path_off as u64 + rec.path_len as u64;
            if path_end > header.strings_len {
                return Err(AimError::Corrupt("chunk path range exceeds the path heap"));
            }
            let payload_end = rec.payload_off + rec.payload_len;
            if payload_end > header.payload_len {
                return Err(AimError::Corrupt(
                    "chunk payload range exceeds the payload section",
                ));
            }
            id_to_idx.insert(rec.id, i);
            records.push(rec);
        }

        let index_path = dir.join(INDEX_FILE);
        let index = IdMapIndex::load(&index_path)
            .map_err(|e| AimError::io("load turbovec index", &index_path, e))?;
        if index.len() != records.len() {
            return Err(AimError::Corrupt(
                "turbovec index vector count does not match the container's chunk count",
            ));
        }

        let meta_path = dir.join(META_FILE);
        let meta_bytes = std::fs::read(&meta_path)
            .map_err(|e| AimError::io("read metadata", &meta_path, e))?;
        let meta: CatalogMeta = serde_json::from_slice(&meta_bytes)
            .map_err(|e| AimError::Index(format!("parse {META_FILE}: {e}")))?;

        // Pay turbovec's one-time lazy init now, so the first real query
        // does not eat the rotation-matrix and codebook build.
        index.prepare();

        // Load the IVF sidecar if present; a missing/corrupt one is non-fatal
        // (fall back to full scan) so a catalog is never unusable because of it.
        let ivf = {
            let p = dir.join(IVF_FILE);
            if p.exists() {
                match crate::ivf::IvfMmap::open(&p) {
                    Ok(v) => Some(v),
                    Err(e) => {
                        eprintln!("[libaim] ignoring unreadable {IVF_FILE}: {e}");
                        None
                    }
                }
            } else {
                None
            }
        };

        let mut catalog = Self {
            mmap,
            header,
            records,
            id_to_idx,
            path_to_chunks: HashMap::new(),
            index,
            meta,
            ivf,
        };

        // Built after construction so it can reuse `chunk_path`, which
        // needs the header and records already in place.
        let ids: Vec<u64> = catalog.records.iter().map(|r| r.id).collect();
        for id in ids {
            let path = catalog.chunk_path(id)?.to_string();
            catalog.path_to_chunks.entry(path).or_default().push(id);
        }

        Ok(catalog)
    }

    /// Metadata this catalog was built with.
    pub fn meta(&self) -> &CatalogMeta {
        &self.meta
    }

    /// Embedding dimension.
    pub fn dim(&self) -> usize {
        self.header.dim as usize
    }

    /// Number of indexed chunks.
    pub fn len(&self) -> usize {
        self.records.len()
    }

    /// True when the catalog holds no chunks.
    pub fn is_empty(&self) -> bool {
        self.records.is_empty()
    }

    /// Fail loudly when the query-time embedder is not the one that
    /// built this catalog.
    ///
    /// Without this the mismatch is silent: scores stay in range, hits
    /// come back, and every one of them is noise.
    pub fn check_embedder(&self, embedder_id: &str) -> Result<(), AimError> {
        if self.meta.embedder_id != embedder_id {
            return Err(AimError::Index(format!(
                "catalog was built with embedder `{}` but the query used `{}`; \
                 scores would be meaningless. Re-run `aim-index` with the same embedder.",
                self.meta.embedder_id, embedder_id
            )));
        }
        Ok(())
    }

    /// The L1 limbic gist: the corpus centroid, as an aligned slice
    /// straight out of the memory map.
    pub fn gist(&self) -> &[f32] {
        let off = self.header.gist_off as usize;
        let len = self.header.dim as usize;
        let bytes = &self.mmap[off..off + len * 4];
        // Safety: `Header::decode` verified `gist_off` is 16-byte
        // aligned and that `gist_len == dim * 4`. The map's base address
        // is page-aligned, so `off` being 16-aligned makes the pointer
        // 4-aligned for f32.
        debug_assert_eq!(bytes.as_ptr().align_offset(std::mem::align_of::<f32>()), 0);
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, len) }
    }

    /// Per-slot inverse document frequency weights, as an aligned slice
    /// straight out of the memory map.
    ///
    /// A query-time embedder must apply these, or its vectors live in a
    /// different space than the indexed ones. [`Self::query_embedder`]
    /// wires that up correctly.
    pub fn idf(&self) -> &[f32] {
        let off = self.header.idf_off as usize;
        let len = self.header.dim as usize;
        let bytes = &self.mmap[off..off + len * 4];
        // Safety: `Header::decode` verified `idf_off` is 16-byte aligned
        // and that `idf_len == dim * 4`, so the pointer is f32-aligned.
        debug_assert_eq!(bytes.as_ptr().align_offset(std::mem::align_of::<f32>()), 0);
        unsafe { std::slice::from_raw_parts(bytes.as_ptr() as *const f32, len) }
    }

    /// An embedder configured to produce vectors in this catalog's
    /// space — right dimension, right IDF weights.
    ///
    /// Always prefer this over constructing a `HashEmbedder` by hand: an
    /// embedder without the catalog's IDF table scores against a
    /// different vector space, and nothing about the resulting numbers
    /// looks wrong.
    pub fn query_embedder(&self) -> crate::embed::HashEmbedder {
        crate::embed::HashEmbedder::with_idf(self.dim(), self.idf().to_vec())
    }

    /// A query-time embedder matching whichever backend built this
    /// catalog, chosen from the recorded `embedder_id`.
    ///
    /// Hash catalogs get the IDF table (as [`Self::query_embedder`]);
    /// dense (`http`) catalogs are reconnected to their embedding server
    /// (URL from `KORTEX_EMBED_SERVER`, else the protocol default). This
    /// is what the proxy MUST use: a catalog built with dense embeddings
    /// is meaningless to a `HashEmbedder`, and every score would be noise.
    pub fn query_embedder_dyn(
        &self,
    ) -> Result<Box<dyn crate::embed::Embedder>, AimError> {
        let id = self.meta.embedder_id.clone();
        if id.starts_with("hash-") {
            return Ok(Box::new(self.query_embedder()));
        }
        // Dense backends record ids of the form "{tag}:{model}-d{dim}".
        #[cfg(feature = "http-embed")]
        {
            if let Some((tag, model)) = parse_dense_embedder_id(&id) {
                let protocol = match tag {
                    "lemonade" => crate::embed_http::EmbedProtocol::Lemonade,
                    "openai" => crate::embed_http::EmbedProtocol::OpenAi,
                    "ollama" => crate::embed_http::EmbedProtocol::Ollama,
                    other => {
                        return Err(AimError::Index(format!(
                            "catalog embedder tag `{other}` is not a known dense backend"
                        )))
                    }
                };
                let url = std::env::var("KORTEX_EMBED_SERVER")
                    .unwrap_or_else(|_| protocol.default_url().to_string());
                let emb =
                    crate::embed_http::HttpEmbedder::connect(protocol, &url, model)?;
                if emb.dim() != self.dim() {
                    return Err(AimError::DimMismatch {
                        catalog: self.dim(),
                        got: emb.dim(),
                    });
                }
                return Ok(Box::new(emb));
            }
        }
        Err(AimError::Index(format!(
            "catalog was built with embedder `{id}`, which needs dense-embedding \
             support — rebuild aim-proxy with `--features http-embed`"
        )))
    }

    /// Line range and token estimate for a chunk, or `None` if the id
    /// is not in this catalog.
    ///
    /// Exposed so an overlay ([`crate::delta::LiveCatalog`]) can build a
    /// [`Hit`] for a base chunk without reaching into private state.
    pub fn chunk_extent(&self, chunk_id: u64) -> Option<(u32, u32, u32)> {
        self.record(chunk_id)
            .ok()
            .map(|r| (r.line_start, r.line_end, r.token_estimate))
    }

    /// Workspace-relative path of a chunk.
    pub fn chunk_path(&self, chunk_id: u64) -> Result<&str, AimError> {
        let rec = self.record(chunk_id)?;
        let start = self.header.strings_off as usize + rec.path_off as usize;
        let bytes = &self.mmap[start..start + rec.path_len as usize];
        std::str::from_utf8(bytes).map_err(|e| AimError::InvalidUtf8 {
            id: chunk_id,
            offset: e.valid_up_to(),
        })
    }

    /// Inflate one chunk: decompress its payload and return the source
    /// text. This is the JIT step — nothing is decompressed until a
    /// search hit asks for it.
    pub fn inflate(&self, chunk_id: u64) -> Result<String, AimError> {
        let rec = self.record(chunk_id)?;
        let start = self.header.payload_off as usize + rec.payload_off as usize;
        let raw = &self.mmap[start..start + rec.payload_len as usize];

        let bytes = if self.header.payload_is_zstd() {
            // Bounding by the recorded length means a corrupt or hostile
            // frame cannot allocate without limit.
            zstd::bulk::decompress(raw, rec.uncompressed_len as usize).map_err(|e| {
                AimError::io(
                    "decompress chunk",
                    PathBuf::from(format!("chunk:{chunk_id}")),
                    e,
                )
            })?
        } else {
            raw.to_vec()
        };

        if bytes.len() != rec.uncompressed_len as usize {
            return Err(AimError::PayloadLengthMismatch {
                id: chunk_id,
                expected: rec.uncompressed_len as usize,
                got: bytes.len(),
            });
        }

        String::from_utf8(bytes).map_err(|e| AimError::InvalidUtf8 {
            id: chunk_id,
            offset: e.utf8_error().valid_up_to(),
        })
    }

    /// Search the index and return `(score, chunk_id)` pairs, best
    /// first. No inflation happens here.
    pub fn search(&self, query: &[f32], k: usize) -> Result<Vec<(f32, u64)>, AimError> {
        if query.len() != self.dim() {
            return Err(AimError::DimMismatch {
                catalog: self.dim(),
                got: query.len(),
            });
        }
        if self.records.is_empty() || k == 0 {
            return Ok(Vec::new());
        }
        // IVF fast path: probe the nearest partitions and restrict the SIMD
        // scan to their ids (sublinear). n_probe = sqrt(n_partitions) balances
        // recall vs. work; override with KORTEX_IVF_NPROBE. Falls through to a
        // full scan when there is no IVF or the probe yields nothing.
        if let Some(ivf) = &self.ivf {
            let n_probe = std::env::var("KORTEX_IVF_NPROBE")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or_else(|| (ivf.n_partitions() as f64).sqrt().ceil() as usize);
            let allow = ivf.probe(query, n_probe);
            if !allow.is_empty() {
                let (scores, ids) = self
                    .index
                    .search_with_allowlist(query, k.min(allow.len()), Some(&allow))
                    .map_err(|e| AimError::Index(e.to_string()))?;
                return Ok(scores
                    .into_iter()
                    .zip(ids)
                    .filter(|(_, id)| self.id_to_idx.contains_key(id))
                    .collect());
            }
        }
        let (scores, ids) = self.index.search(query, k.min(self.records.len()));
        Ok(scores
            .into_iter()
            .zip(ids)
            // turbovec pads the result set when k exceeds what a query
            // can fill; those slots carry ids that are not in the map.
            .filter(|(s, id)| s.is_finite() && self.id_to_idx.contains_key(id))
            .collect())
    }

    /// Paths present in the catalog.
    pub fn paths(&self) -> impl Iterator<Item = &str> {
        self.path_to_chunks.keys().map(|s| s.as_str())
    }

    /// Chunk ids belonging to any path containing one of `substrings`.
    ///
    /// Substring rather than exact match because a query mentions
    /// `apple_mbox.c` while the catalog stores `hw/misc/apple_mbox.c`.
    pub fn chunks_matching_paths(&self, substrings: &[&str]) -> Vec<u64> {
        let mut ids = Vec::new();
        for (path, chunk_ids) in &self.path_to_chunks {
            if substrings.iter().any(|s| !s.is_empty() && path.contains(s)) {
                ids.extend_from_slice(chunk_ids);
            }
        }
        ids.sort_unstable();
        ids
    }

    /// Search restricted to `allowed` chunk ids.
    ///
    /// turbovec's allowlist path *panics* on an empty list or an unknown
    /// id, so both are filtered out here rather than trusted — a caller
    /// deriving ids from user input must not be able to abort the
    /// process.
    pub fn search_scoped(
        &self,
        query: &[f32],
        k: usize,
        allowed: &[u64],
    ) -> Result<Vec<(f32, u64)>, AimError> {
        if query.len() != self.dim() {
            return Err(AimError::DimMismatch {
                catalog: self.dim(),
                got: query.len(),
            });
        }
        let allowed: Vec<u64> = allowed
            .iter()
            .copied()
            .filter(|id| self.id_to_idx.contains_key(id))
            .collect();
        if allowed.is_empty() || k == 0 {
            return Ok(Vec::new());
        }

        let k = k.min(allowed.len());
        let (scores, ids) = self
            .index
            .search_with_allowlist(query, k, Some(&allowed))
            .map_err(|e| AimError::Index(e.to_string()))?;
        Ok(scores
            .into_iter()
            .zip(ids)
            .filter(|(s, id)| s.is_finite() && self.id_to_idx.contains_key(id))
            .collect())
    }

    /// Lock the payload pages of `chunk_ids` into physical RAM, spending
    /// at most `budget_bytes`.
    ///
    /// This is the Colibri "routing heat" idea applied to a code
    /// catalog: the chunks an agent keeps retrieving are the ones that
    /// must never take a disk fault. Best-effort — see
    /// [`crate::heat::lock_region`] for why a refusal is not an error.
    pub fn pin_chunks(&self, chunk_ids: &[u64], budget_bytes: usize) -> crate::heat::PinReport {
        let mut report = crate::heat::PinReport {
            requested: chunk_ids.len(),
            ..Default::default()
        };

        for &id in chunk_ids {
            let Ok(rec) = self.record(id) else {
                report.failures.push(format!("chunk {id} is not in this catalog"));
                continue;
            };
            let len = rec.payload_len as usize;
            if report.pinned_bytes + len > budget_bytes {
                break;
            }
            let start = self.header.payload_off as usize + rec.payload_off as usize;

            // Safety: `start .. start + len` was bounds-checked against
            // the payload section in `open`, and `self.mmap` outlives
            // this borrow.
            let locked = unsafe { crate::heat::lock_region(self.mmap[start..].as_ptr(), len) };
            match locked {
                Ok(bytes) => {
                    report.pinned += 1;
                    report.pinned_bytes += bytes;
                }
                Err(msg) => {
                    // One quota error means every later call fails the
                    // same way; recording it once and stopping beats
                    // thousands of identical strings.
                    report.failures.push(msg);
                    break;
                }
            }
        }

        report
    }

    /// Full retrieval: search, gate on the threshold, spend the budget
    /// best-first, and inflate only what survives.
    pub fn page_fault(
        &self,
        query: &[f32],
        cfg: &RetrievalConfig,
    ) -> Result<PageFaultResult, AimError> {
        self.page_fault_inner(query, cfg, None)
    }

    /// Every chunk that *defines* `symbol` (declaration keyword before it,
    /// or an assignment/typed field). The structural half of Stage 2/3,
    /// exposed for the agentic `find_definition` tool. Exact, no embedding.
    pub fn find_definitions(&self, symbol: &str) -> Result<Vec<Hit>, AimError> {
        let sym = symbol.to_lowercase();
        let mut hits = Vec::new();
        for rec in &self.records {
            let text = self.inflate(rec.id)?;
            if chunk_defines(&text, &sym) {
                hits.push(Hit {
                    chunk_id: rec.id,
                    path: self.chunk_path(rec.id)?.to_string(),
                    line_start: rec.line_start,
                    line_end: rec.line_end,
                    score: 1.0,
                    token_estimate: rec.token_estimate,
                    text,
                });
            }
        }
        Ok(hits)
    }

    /// Hybrid retrieval: fuse the dense/semantic ranking with a lexical
    /// (exact-token) ranking via reciprocal-rank fusion. This is the fix
    /// for the case a pure-semantic search misses an exact-symbol query
    /// ("what is KVTYPE set to") that lexical matching nails.
    ///
    /// `lex` is a lexical embedder (a `HashEmbedder`); every chunk's text
    /// is scored against it — cheap at kortex catalog sizes (precompute a
    /// lexical index for very large corpora; that is Stage 1.5). A chunk is
    /// kept only if EITHER channel clears the absolute floor, so an
    /// off-topic query still yields no hits and the gate is preserved.
    pub fn hybrid_fault(
        &self,
        query_text: &str,
        sem_vec: &[f32],
        cfg: &RetrievalConfig,
    ) -> Result<PageFaultResult, AimError> {
        use std::collections::{HashMap, HashSet};

        // Channel 1 — semantic (the existing dense search).
        let sem = self.search(sem_vec, cfg.candidates)?;
        let sem_score: HashMap<u64, f32> = sem.iter().map(|(s, id)| (*id, *s)).collect();
        let sem_ranking: Vec<u64> = sem.iter().map(|(_, id)| *id).collect();
        // Same scale-free semantic gate as page_fault: a fraction of the
        // best score, never below the absolute floor. Keeps the dense
        // channel from admitting weakly-similar chunks on an off-topic query.
        let sem_best = sem.first().map(|(s, _)| *s).unwrap_or(0.0);
        let sem_cutoff = cfg
            .fault_threshold
            .max(sem_best * cfg.relative_floor.clamp(0.0, 1.0));

        // Channel 2 — lexical BM25 over every chunk's text. BM25's IDF term
        // heavily rewards RARE query tokens (an identifier like `KVTYPE`
        // that occurs in only one chunk) — exactly what a pure-cosine
        // lexical channel washes out. Inflating every chunk is cheap at
        // kortex sizes; precompute for very large corpora (Stage 1.5).
        let mut q_terms = bm25_tokenize(query_text);
        q_terms.sort();
        q_terms.dedup();
        // Stage-2 structural signal: which chunks DEFINE a query symbol
        // (vs merely mention it). Only identifier-ish terms (len >= 4) are
        // treated as symbols, so stopwords/keywords ("for", "type", "the")
        // do not trigger spurious definition boosts.
        let sym_terms: Vec<&str> =
            q_terms.iter().filter(|t| t.len() >= 4).map(|s| s.as_str()).collect();
        let mut def_ids: HashSet<u64> = HashSet::new();
        let mut chunk_toks: Vec<(u64, Vec<String>)> =
            Vec::with_capacity(self.records.len());
        for rec in &self.records {
            let text = self.inflate(rec.id)?;
            if !sym_terms.is_empty() && sym_terms.iter().any(|s| chunk_defines(&text, s)) {
                def_ids.insert(rec.id);
            }
            chunk_toks.push((rec.id, bm25_tokenize(&text)));
        }
        let n_docs = chunk_toks.len().max(1) as f64;
        let avgdl = (chunk_toks.iter().map(|(_, t)| t.len()).sum::<usize>() as f64
            / n_docs)
            .max(1.0);
        let df: HashMap<&str, f64> = q_terms
            .iter()
            .map(|term| {
                let d = chunk_toks
                    .iter()
                    .filter(|(_, t)| t.iter().any(|w| w == term))
                    .count() as f64;
                (term.as_str(), d)
            })
            .collect();
        let (k1, b) = (1.5f64, 0.75f64);
        let mut lex_scored: Vec<(f32, u64)> = chunk_toks
            .iter()
            .map(|(id, toks)| {
                let dl = toks.len() as f64;
                let mut score = 0.0f64;
                for term in &q_terms {
                    let tf = toks.iter().filter(|w| *w == term).count() as f64;
                    if tf == 0.0 {
                        continue;
                    }
                    let d = df[term.as_str()];
                    let idf = ((n_docs - d + 0.5) / (d + 0.5) + 1.0).ln();
                    score += idf * (tf * (k1 + 1.0))
                        / (tf + k1 * (1.0 - b + b * dl / avgdl));
                }
                (score as f32, *id)
            })
            .collect();
        lex_scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
        });
        let lex_score: HashMap<u64, f32> =
            lex_scored.iter().map(|(s, id)| (*id, *s)).collect();
        let lex_ranking: Vec<u64> =
            lex_scored.iter().take(cfg.candidates).map(|(_, id)| *id).collect();

        // Fuse by NORMALIZED weighted sum. Semantic cosine is already in
        // [0,1]; lexical BM25 is normalized by the channel's own max so a
        // STRONG exact match (an identifier only one chunk contains) can
        // surface to the top even when semantic missed it entirely — the
        // whole point of hybrid. Plain rank-fusion (RRF) buries a
        // lexical-only hit because it over-rewards cross-channel agreement.
        // The lexical channel is trusted only when it actually matched
        // (max well above stopword noise), so an off-topic query where no
        // term hits cannot be amplified into false positives.
        let lex_max = lex_scored.first().map(|(s, _)| *s).unwrap_or(0.0);
        let lex_active = lex_max > 0.5;
        const LEX_WEIGHT: f32 = 1.5;
        // A lexical-only chunk (semantic missed it) is admitted only when
        // its RAW BM25 clears this bar — i.e. it matched a RARE query term
        // (an identifier like `KVTYPE` ~4.0), not one incidental common
        // word (e.g. "recipe" ~1.7 in an off-topic query). Heuristic and
        // mildly corpus-scale sensitive; Stage 2's symbol graph replaces it
        // with exact structure. Semantic relevance still admits on its own.
        const LEX_GATE_BM25: f32 = 2.5;
        // Structural boost: a chunk that DEFINES a queried symbol beats one
        // that merely uses it (e.g. `KVTYPE=...` at the definition site over
        // `-ctk "$KVTYPE"` usages). Large enough to reorder within a file.
        const DEF_BOOST: f32 = 1.0;
        let lex_norm = |id: u64| -> f32 {
            if lex_active {
                lex_score.get(&id).copied().unwrap_or(0.0) / lex_max
            } else {
                0.0
            }
        };
        let mut ids: Vec<u64> = sem_ranking.clone();
        for id in &lex_ranking {
            if !ids.contains(id) {
                ids.push(*id);
            }
        }
        let mut fused: Vec<(u64, f32)> = ids
            .iter()
            .map(|id| {
                let s = sem_score.get(id).copied().unwrap_or(0.0);
                let d = if def_ids.contains(id) { DEF_BOOST } else { 0.0 };
                (*id, s + LEX_WEIGHT * lex_norm(*id) + d)
            })
            .collect();
        fused.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });

        let gist_score = cosine(sem_vec, self.gist());
        let mut best_score = 0.0f32;
        let mut hits = Vec::new();
        let mut spent = 0u32;
        let mut dropped = 0usize;
        let mut seen_ranges: Vec<(String, u32, u32)> = Vec::new();

        for (id, fscore) in fused {
            // Gate: admit on genuine SEMANTIC relevance, OR a STRONG lexical
            // match (raw BM25 over the bar = a rare identifier). An
            // off-topic query sharing one incidental common word clears
            // neither and injects nothing.
            let sem_s = sem_score.get(&id).copied().unwrap_or(0.0);
            let lex_raw = lex_score.get(&id).copied().unwrap_or(0.0);
            // A structural definition of a queried symbol is always relevant.
            if sem_s < sem_cutoff && lex_raw < LEX_GATE_BM25 && !def_ids.contains(&id) {
                continue;
            }
            best_score = best_score.max(fscore);
            if hits.len() >= cfg.max_chunks {
                dropped += 1;
                continue;
            }
            let rec = self.record(id)?;
            if spent + rec.token_estimate > cfg.token_budget {
                dropped += 1;
                continue;
            }
            let path = self.chunk_path(id)?.to_string();
            if seen_ranges
                .iter()
                .any(|(p, s, e)| *p == path && rec.line_start <= *e && rec.line_end >= *s)
            {
                continue;
            }
            seen_ranges.push((path.clone(), rec.line_start, rec.line_end));
            spent += rec.token_estimate;
            hits.push(Hit {
                chunk_id: id,
                path,
                line_start: rec.line_start,
                line_end: rec.line_end,
                score: fscore as f32,
                token_estimate: rec.token_estimate,
                text: self.inflate(id)?,
            });
        }

        Ok(PageFaultResult {
            hits,
            best_score,
            cutoff: cfg.fault_threshold,
            gist_score,
            dropped_to_budget: dropped,
        })
    }

    /// Retrieval restricted to chunks from paths matching `substrings`.
    ///
    /// When a request names a file, scoping to it skips the SIMD cost of
    /// scanning the rest of the corpus and stops a lexically similar
    /// chunk from another file outranking the right one.
    pub fn page_fault_scoped(
        &self,
        query: &[f32],
        cfg: &RetrievalConfig,
        path_substrings: &[&str],
    ) -> Result<PageFaultResult, AimError> {
        let allowed = self.chunks_matching_paths(path_substrings);
        if allowed.is_empty() {
            // Nothing matched the hint; a corpus-wide search beats
            // returning nothing.
            return self.page_fault(query, cfg);
        }
        self.page_fault_inner(query, cfg, Some(&allowed))
    }

    fn page_fault_inner(
        &self,
        query: &[f32],
        cfg: &RetrievalConfig,
        allowed: Option<&[u64]>,
    ) -> Result<PageFaultResult, AimError> {
        let gist_score = cosine(query, self.gist());
        let candidates = match allowed {
            Some(ids) => self.search_scoped(query, cfg.candidates, ids)?,
            None => self.search(query, cfg.candidates)?,
        };
        let best_score = candidates.first().map(|(s, _)| *s).unwrap_or(0.0);
        // Scale-free gate: a fraction of the best score, but never below
        // the absolute floor. `relative_floor` is clamped because a
        // negative value would admit everything and a value above 1
        // would reject even the best hit.
        let cutoff = cfg
            .fault_threshold
            .max(best_score * cfg.relative_floor.clamp(0.0, 1.0));

        let mut hits = Vec::new();
        let mut spent = 0u32;
        let mut dropped = 0usize;
        let mut seen_ranges: Vec<(String, u32, u32)> = Vec::new();

        for (score, id) in candidates {
            if score < cutoff {
                // Candidates are sorted, so nothing further can qualify.
                break;
            }
            if hits.len() >= cfg.max_chunks {
                dropped += 1;
                continue;
            }

            let rec = self.record(id)?;
            if spent + rec.token_estimate > cfg.token_budget {
                dropped += 1;
                continue;
            }

            let path = self.chunk_path(id)?.to_string();
            // Chunks overlap by design, so adjacent windows of one file
            // can both hit. Injecting both wastes budget on duplicated
            // lines.
            if seen_ranges.iter().any(|(p, s, e)| {
                *p == path && rec.line_start <= *e && rec.line_end >= *s
            }) {
                continue;
            }
            seen_ranges.push((path.clone(), rec.line_start, rec.line_end));

            spent += rec.token_estimate;
            hits.push(Hit {
                chunk_id: id,
                path,
                line_start: rec.line_start,
                line_end: rec.line_end,
                score,
                token_estimate: rec.token_estimate,
                text: self.inflate(id)?,
            });
        }

        Ok(PageFaultResult {
            hits,
            best_score,
            cutoff,
            gist_score,
            dropped_to_budget: dropped,
        })
    }

    /// Verify the body hash in the header. Not on the open path — it
    /// reads the whole file, which defeats the point of the mmap — but
    /// worth running after a copy or a suspected bad disk.
    pub fn verify_integrity(&self) -> Result<(), AimError> {
        let actual = hash_prefix(&self.mmap[HEADER_SIZE..]);
        if actual != self.header.content_hash {
            return Err(AimError::Corrupt(
                "content hash mismatch: the catalog body does not match its header",
            ));
        }
        Ok(())
    }

    fn record(&self, chunk_id: u64) -> Result<&ChunkRecord, AimError> {
        self.id_to_idx
            .get(&chunk_id)
            .map(|&i| &self.records[i])
            .ok_or(AimError::UnknownChunk(chunk_id))
    }
}

/// First 16 bytes of the blake3 digest.
pub(crate) fn hash_prefix(bytes: &[u8]) -> [u8; 16] {
    let full = blake3::hash(bytes);
    let mut out = [0u8; 16];
    out.copy_from_slice(&full.as_bytes()[..16]);
    out
}

/// Per-slot inverse document frequency over a flat `n × dim` array.
///
/// "Document frequency" is the number of chunks with a nonzero value in
/// that slot. Uses the standard smoothed form `ln(1 + n/df)`, which is
/// positive for every slot and tends to zero for a feature present in
/// every chunk.
fn compute_idf(flat: &[f32], dim: usize) -> Vec<f32> {
    let mut idf = vec![0f32; dim];
    if dim == 0 || flat.is_empty() {
        return idf;
    }
    let n = (flat.len() / dim) as f32;

    let mut df = vec![0u32; dim];
    for row in flat.chunks_exact(dim) {
        for (slot, value) in row.iter().enumerate() {
            if *value != 0.0 {
                df[slot] += 1;
            }
        }
    }
    for (slot, count) in df.iter().enumerate() {
        // A slot no chunk uses gets the maximum weight rather than a
        // division by zero. It can never match anything anyway, so the
        // value only matters for staying finite.
        idf[slot] = (1.0 + n / (*count).max(1) as f32).ln();
    }
    idf
}

/// Scale every vector by `idf` element-wise and renormalize.
///
/// Renormalizing after weighting is what keeps scores comparable to a
/// cosine. Applying IDF before or after the original normalization is
/// equivalent, since `normalize(w * (v/|v|)) == normalize(w * v)`.
fn apply_idf(flat: &mut [f32], dim: usize, idf: &[f32]) {
    if dim == 0 || idf.len() != dim {
        return;
    }
    for row in flat.chunks_exact_mut(dim) {
        for (value, weight) in row.iter_mut().zip(idf) {
            *value *= *weight;
        }
        normalize(row);
    }
}

/// L2-normalized mean of a flat `n × dim` vector array.
fn centroid(flat: &[f32], dim: usize) -> Vec<f32> {
    let mut acc = vec![0f32; dim];
    if dim == 0 || flat.is_empty() {
        return acc;
    }
    let n = flat.len() / dim;
    for row in flat.chunks_exact(dim) {
        for (a, x) in acc.iter_mut().zip(row) {
            *a += *x;
        }
    }
    for a in acc.iter_mut() {
        *a /= n as f32;
    }
    normalize(&mut acc);
    acc
}

/// Parse a dense-embedder id of the form `"{tag}:{model}-d{dim}"` into
/// `(tag, model)`. Returns `None` for ids that are not `tag:...` — in
/// particular the lexical `hash-*` ids, which carry no colon. The
/// `-d{dim}` suffix is stripped from the model using the LAST `-d`, so a
/// model name that itself contains `-d` (e.g. `text-embedding-3-large`,
/// `e5-dense`) survives intact.
///
/// Only compiled where used: the dense-embed reconnect path (behind
/// `http-embed`) and the tests. Keeps lexical-only builds warning-free.
#[cfg(any(feature = "http-embed", test))]
fn parse_dense_embedder_id(id: &str) -> Option<(&str, &str)> {
    let (tag, rest) = id.split_once(':')?;
    let model = rest.rsplit_once("-d").map(|(m, _)| m).unwrap_or(rest);
    Some((tag, model))
}

/// Lowercase alphanumeric tokens (splitting on non-word chars and `_`),
/// for the BM25 lexical channel. Deliberately simple + self-consistent
/// between query and chunk — BM25 only needs matching tokenization, not
/// the embedder's subword scheme.
fn bm25_tokenize(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric() && c != '_')
        .flat_map(|w| w.split('_'))
        .filter(|w| !w.is_empty())
        .map(|w| w.to_lowercase())
        .collect()
}

#[inline]
fn is_word_byte(b: u8) -> bool {
    b.is_ascii_alphanumeric() || b == b'_'
}

/// True if `line_lower` (already lowercased) *defines* the symbol `sym`
/// (also lowercase): a declaration keyword immediately before it, or an
/// assignment/field (`sym =` / `sym:`) immediately after. Language-agnostic
/// and deliberately conservative — this is the Stage-2 def/usage signal
/// that BM25 cannot give (a definition vs one of many usages), without the
/// weight of full tree-sitter grammars. Matches only at word boundaries.
fn line_defines(line_lower: &str, sym: &str) -> bool {
    const DEF_KW: &[&str] = &[
        "fn", "def", "struct", "enum", "trait", "class", "type", "interface",
        "function", "const", "let", "var", "static", "impl", "mod", "macro",
    ];
    if sym.is_empty() {
        return false;
    }
    let bytes = line_lower.as_bytes();
    let mut from = 0usize;
    while let Some(rel) = line_lower[from..].find(sym) {
        let idx = from + rel;
        let end = idx + sym.len();
        let bound_before = idx == 0 || !is_word_byte(bytes[idx - 1]);
        let bound_after = end >= bytes.len() || !is_word_byte(bytes[end]);
        if bound_before && bound_after {
            // preceding word a declaration keyword?
            let prev = line_lower[..idx]
                .trim_end()
                .rsplit(|c: char| !c.is_alphanumeric() && c != '_')
                .next()
                .unwrap_or("");
            if DEF_KW.contains(&prev) {
                return true;
            }
            // immediately assigned or a typed field?
            let after = line_lower[end..].trim_start();
            if after.starts_with('=') && !after.starts_with("==") {
                return true;
            }
            if after.starts_with(':') && !after.starts_with("::") {
                return true;
            }
        }
        from = end;
    }
    false
}

/// True if any line of `text` defines `sym` (case-insensitive on both).
fn chunk_defines(text: &str, sym_lower: &str) -> bool {
    text.lines().any(|l| line_defines(&l.to_lowercase(), sym_lower))
}

#[cfg(test)]
mod defines_tests {
    use super::{chunk_defines, line_defines};

    #[test]
    fn detects_assignments_and_declarations() {
        assert!(line_defines("kvtype=\"${kvtype:-q4_0}\"", "kvtype")); // shell assign
        assert!(line_defines("    let dim: usize = 5;", "dim")); // rust let + kw
        assert!(line_defines("def process_orders(orders):", "process_orders")); // python def
        assert!(line_defines("const lex_gate_bm25: f32 = 2.5;", "lex_gate_bm25"));
    }

    #[test]
    fn ignores_usages_and_equality() {
        assert!(!line_defines("total = kvtype + 1", "kvtype")); // usage on RHS
        assert!(!line_defines("if kvtype == q4", "kvtype")); // equality, not assign
        assert!(!line_defines("self::kvtype", "kvtype")); // path, not field
        assert!(!line_defines("kvtypes_map = {}", "kvtype")); // not a word boundary
    }

    #[test]
    fn chunk_scans_all_lines_case_insensitively() {
        let text = "line one\n  KVTYPE = \"q4_0\"\nlast line";
        assert!(chunk_defines(text, "kvtype"));
        assert!(!chunk_defines("just a mention of kvtype here", "kvtype"));
    }
}

#[cfg(test)]
mod bm25_tokenize_tests {
    use super::bm25_tokenize as t;

    #[test]
    fn splits_on_non_word_and_underscore_and_lowercases() {
        assert_eq!(t("KV_cache q4_0"), vec!["kv", "cache", "q4", "0"]);
        assert_eq!(t("KVTYPE=\"${KVTYPE:-q4_0}\""), vec!["kvtype", "kvtype", "q4", "0"]);
    }
}

#[cfg(test)]
mod parse_id_tests {
    use super::parse_dense_embedder_id as p;

    #[test]
    fn strips_dim_suffix_even_with_hyphenated_model_names() {
        assert_eq!(p("lemonade:Qwen3-Embedding-0.6B-GGUF-d1024"),
                   Some(("lemonade", "Qwen3-Embedding-0.6B-GGUF")));
        assert_eq!(p("openai:text-embedding-3-large-d3072"),
                   Some(("openai", "text-embedding-3-large")));
        assert_eq!(p("ollama:e5-dense-d768"), Some(("ollama", "e5-dense")));
    }

    #[test]
    fn lexical_hash_ids_are_not_dense() {
        assert_eq!(p("hash-v1-d1536"), None);
        assert_eq!(p("hash-v2-d256"), None);
    }
}
