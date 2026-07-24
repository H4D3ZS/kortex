//! On-disk layout for the `.aim` catalog container (format version 3).
//!
//! A catalog is two files that live side by side:
//!
//! * `catalog.aim`  — this container: header, chunk table, path heap,
//!   L1 gist vector, and the compressed chunk payload blob.
//! * `catalog.tvim` — the turbovec [`IdMapIndex`](turbovec::IdMapIndex)
//!   holding one quantized embedding per chunk, keyed by chunk id.
//!
//! The index is a sibling rather than an embedded section because
//! turbovec owns its own versioned format and only exposes path-based
//! read/write. Keeping it separate means we never copy index bytes out
//! of the mmap into a temp file just to hand turbovec a path.
//!
//! # Container layout
//!
//! ```text
//! 0                                                             160
//! ├──────────────────────── Header ─────────────────────────────┤
//! │ magic "AIMCAT03" │ version │ flags │ dim │ bit_width │ ...   │
//! ├─────────────────────────────────────────────────────────────┤
//! │ Chunk table   — chunk_count × 64-byte ChunkRecord           │
//! │ Path heap     — UTF-8 bytes, sliced by ChunkRecord.path_*   │
//! │ Gist          — `dim` × f32, the L1 limbic vector           │
//! │ IDF           — `dim` × f32, per-slot inverse doc frequency │
//! │ Payload       — concatenated per-chunk compressed blobs     │
//! └─────────────────────────────────────────────────────────────┘
//! ```
//!
//! All integers are little-endian. Every section begins on a 16-byte
//! boundary so the gist and IDF tables can be read as aligned `f32`
//! slices directly out of the memory map.

use std::time::{SystemTime, UNIX_EPOCH};

use crate::error::AimError;

/// Container magic. The trailing digits are the format version so a
/// mismatched file is rejected by magic rather than by version check.
pub const MAGIC: &[u8; 8] = b"AIMCAT03";

/// Current container format version.
pub const FORMAT_VERSION: u32 = 3;

/// Size of the fixed header in bytes.
pub const HEADER_SIZE: usize = 160;

/// Size of a single chunk table record in bytes.
pub const CHUNK_RECORD_SIZE: usize = 64;

/// Alignment applied to the start of every section.
pub const SECTION_ALIGN: usize = 16;

/// Default embedding dimension. Matches the 6KB (1536 × f32) limbic
/// vector the kernel keeps resident.
pub const DEFAULT_DIM: usize = 1536;

/// Default turbovec bit width. 4 bits/coord is the accuracy-favouring
/// end of TurboQuant's 2–4 bit range: 1536 dims compress to 768 bytes.
pub const DEFAULT_BIT_WIDTH: usize = 4;

/// Flag bit: chunk payloads are zstd-compressed.
pub const FLAG_PAYLOAD_ZSTD: u32 = 1 << 0;

/// Round `n` up to the next multiple of [`SECTION_ALIGN`].
#[inline]
pub fn align_up(n: usize) -> usize {
    (n + SECTION_ALIGN - 1) & !(SECTION_ALIGN - 1)
}

/// Fixed-size container header.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Header {
    pub format_version: u32,
    pub flags: u32,
    /// Embedding dimension for both the gist and every chunk vector.
    pub dim: u32,
    /// turbovec quantizer bit width used to build the sibling index.
    pub bit_width: u32,
    pub chunk_count: u64,
    pub chunk_table_off: u64,
    pub chunk_table_len: u64,
    pub strings_off: u64,
    pub strings_len: u64,
    pub gist_off: u64,
    pub gist_len: u64,
    /// Per-slot inverse document frequency weights, `dim` × f32.
    pub idf_off: u64,
    pub idf_len: u64,
    pub payload_off: u64,
    pub payload_len: u64,
    /// Build time, seconds since the Unix epoch.
    pub built_unix_secs: u64,
    /// blake3 prefix over every byte after the header. Detects
    /// truncation and bit-rot without hashing on the read path unless
    /// the caller explicitly asks to verify.
    pub content_hash: [u8; 16],
}

impl Header {
    /// True when chunk payloads are zstd-compressed.
    #[inline]
    pub fn payload_is_zstd(&self) -> bool {
        self.flags & FLAG_PAYLOAD_ZSTD != 0
    }

    /// Seconds since the Unix epoch, for stamping a freshly built header.
    pub fn now_unix_secs() -> u64 {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0)
    }

    /// Serialize into exactly [`HEADER_SIZE`] bytes.
    pub fn encode(&self) -> [u8; HEADER_SIZE] {
        let mut out = [0u8; HEADER_SIZE];
        let mut w = Cursor::new(&mut out);
        w.bytes(MAGIC);
        w.u32(self.format_version);
        w.u32(self.flags);
        w.u32(self.dim);
        w.u32(self.bit_width);
        w.u64(self.chunk_count);
        w.u64(self.chunk_table_off);
        w.u64(self.chunk_table_len);
        w.u64(self.strings_off);
        w.u64(self.strings_len);
        w.u64(self.gist_off);
        w.u64(self.gist_len);
        w.u64(self.idf_off);
        w.u64(self.idf_len);
        w.u64(self.payload_off);
        w.u64(self.payload_len);
        w.u64(self.built_unix_secs);
        w.bytes(&self.content_hash);
        out
    }

    /// Parse a header and check every section fits inside `total_len`.
    pub fn decode(buf: &[u8], total_len: usize) -> Result<Self, AimError> {
        if buf.len() < HEADER_SIZE {
            return Err(AimError::Truncated {
                what: "header",
                need: HEADER_SIZE,
                have: buf.len(),
            });
        }
        if &buf[..8] != MAGIC {
            return Err(AimError::BadMagic {
                found: buf[..8].to_vec(),
            });
        }

        let mut r = Reader::new(&buf[8..HEADER_SIZE]);
        let header = Header {
            format_version: r.u32(),
            flags: r.u32(),
            dim: r.u32(),
            bit_width: r.u32(),
            chunk_count: r.u64(),
            chunk_table_off: r.u64(),
            chunk_table_len: r.u64(),
            strings_off: r.u64(),
            strings_len: r.u64(),
            gist_off: r.u64(),
            gist_len: r.u64(),
            idf_off: r.u64(),
            idf_len: r.u64(),
            payload_off: r.u64(),
            payload_len: r.u64(),
            built_unix_secs: r.u64(),
            content_hash: r.array16(),
        };

        if header.format_version != FORMAT_VERSION {
            return Err(AimError::UnsupportedVersion {
                found: header.format_version,
                supported: FORMAT_VERSION,
            });
        }
        if header.dim == 0 {
            return Err(AimError::Corrupt("header declares dim = 0"));
        }
        // The gist section must hold exactly `dim` f32 values, otherwise
        // the aligned reinterpret in `Catalog::gist` would read garbage.
        if header.gist_len != header.dim as u64 * 4 {
            return Err(AimError::Corrupt(
                "gist section length does not match dim × 4",
            ));
        }
        if header.idf_len != header.dim as u64 * 4 {
            return Err(AimError::Corrupt(
                "idf section length does not match dim × 4",
            ));
        }
        if header.chunk_table_len != header.chunk_count * CHUNK_RECORD_SIZE as u64 {
            return Err(AimError::Corrupt(
                "chunk table length does not match chunk_count × 64",
            ));
        }

        for (name, off, len) in [
            ("chunk_table", header.chunk_table_off, header.chunk_table_len),
            ("strings", header.strings_off, header.strings_len),
            ("gist", header.gist_off, header.gist_len),
            ("idf", header.idf_off, header.idf_len),
            ("payload", header.payload_off, header.payload_len),
        ] {
            let end = off.checked_add(len).ok_or(AimError::Corrupt(
                "section offset + length overflows u64",
            ))?;
            if end > total_len as u64 {
                return Err(AimError::SectionOutOfBounds {
                    section: name,
                    end,
                    file_len: total_len as u64,
                });
            }
        }

        // Every section must start aligned, or the zero-copy gist read
        // and record slicing below are unsound.
        if header.gist_off as usize % SECTION_ALIGN != 0 {
            return Err(AimError::Corrupt("gist section is not 16-byte aligned"));
        }
        if header.idf_off as usize % SECTION_ALIGN != 0 {
            return Err(AimError::Corrupt("idf section is not 16-byte aligned"));
        }

        Ok(header)
    }
}

/// One entry in the chunk table: where a chunk's bytes live and what
/// region of which file they came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ChunkRecord {
    /// Stable chunk id. Also the turbovec external id for this chunk's
    /// embedding, which is what makes index hits resolvable to bytes.
    pub id: u64,
    /// Byte range of this chunk's path within the path heap.
    pub path_off: u32,
    pub path_len: u32,
    /// Byte range of this chunk's blob within the payload section.
    pub payload_off: u64,
    pub payload_len: u64,
    /// Length after decompression. Sizes the output buffer and bounds
    /// the decompressor so a corrupt frame cannot balloon memory.
    pub uncompressed_len: u32,
    /// 1-based inclusive source line range, for citation in the prompt.
    pub line_start: u32,
    pub line_end: u32,
    /// Rough token count, used to spend the retrieval budget without
    /// running a tokenizer at query time.
    pub token_estimate: u32,
    /// blake3 prefix of the uncompressed bytes. Lets an incremental
    /// re-index skip chunks whose content is unchanged.
    pub hash_prefix: [u8; 16],
}

impl ChunkRecord {
    /// Serialize into exactly [`CHUNK_RECORD_SIZE`] bytes.
    pub fn encode(&self) -> [u8; CHUNK_RECORD_SIZE] {
        let mut out = [0u8; CHUNK_RECORD_SIZE];
        let mut w = Cursor::new(&mut out);
        w.u64(self.id);
        w.u32(self.path_off);
        w.u32(self.path_len);
        w.u64(self.payload_off);
        w.u64(self.payload_len);
        w.u32(self.uncompressed_len);
        w.u32(self.line_start);
        w.u32(self.line_end);
        w.u32(self.token_estimate);
        w.bytes(&self.hash_prefix);
        out
    }

    /// Parse a record from exactly [`CHUNK_RECORD_SIZE`] bytes.
    pub fn decode(buf: &[u8]) -> Result<Self, AimError> {
        if buf.len() < CHUNK_RECORD_SIZE {
            return Err(AimError::Truncated {
                what: "chunk record",
                need: CHUNK_RECORD_SIZE,
                have: buf.len(),
            });
        }
        let mut r = Reader::new(buf);
        Ok(ChunkRecord {
            id: r.u64(),
            path_off: r.u32(),
            path_len: r.u32(),
            payload_off: r.u64(),
            payload_len: r.u64(),
            uncompressed_len: r.u32(),
            line_start: r.u32(),
            line_end: r.u32(),
            token_estimate: r.u32(),
            hash_prefix: r.array16(),
        })
    }
}

/// Minimal fixed-width writer over a byte buffer.
///
/// Panics on overflow by construction: callers size the destination to
/// the exact encoded length, so a panic here means an encode function
/// and its `*_SIZE` constant disagree — a bug, not bad input.
struct Cursor<'a> {
    buf: &'a mut [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(buf: &'a mut [u8]) -> Self {
        Cursor { buf, pos: 0 }
    }
    fn bytes(&mut self, v: &[u8]) {
        self.buf[self.pos..self.pos + v.len()].copy_from_slice(v);
        self.pos += v.len();
    }
    fn u32(&mut self, v: u32) {
        self.bytes(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }
}

/// Minimal fixed-width reader. Length is validated by the caller before
/// construction, so the indexing here cannot go out of bounds.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(buf: &'a [u8]) -> Self {
        Reader { buf, pos: 0 }
    }
    fn u32(&mut self) -> u32 {
        let v = u32::from_le_bytes(self.buf[self.pos..self.pos + 4].try_into().unwrap());
        self.pos += 4;
        v
    }
    fn u64(&mut self) -> u64 {
        let v = u64::from_le_bytes(self.buf[self.pos..self.pos + 8].try_into().unwrap());
        self.pos += 8;
        v
    }
    fn array16(&mut self) -> [u8; 16] {
        let v: [u8; 16] = self.buf[self.pos..self.pos + 16].try_into().unwrap();
        self.pos += 16;
        v
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_header() -> Header {
        Header {
            format_version: FORMAT_VERSION,
            flags: FLAG_PAYLOAD_ZSTD,
            dim: 8,
            bit_width: 4,
            chunk_count: 2,
            chunk_table_off: 128,
            chunk_table_len: 128,
            strings_off: 256,
            strings_len: 16,
            gist_off: 272,
            gist_len: 32,
            idf_off: 304,
            idf_len: 32,
            payload_off: 336,
            payload_len: 64,
            built_unix_secs: 1_700_000_000,
            content_hash: [7u8; 16],
        }
    }

    #[test]
    fn header_roundtrips() {
        let h = sample_header();
        let bytes = h.encode();
        assert_eq!(&bytes[..8], MAGIC);
        let decoded = Header::decode(&bytes, 400).unwrap();
        assert_eq!(h, decoded);
    }

    #[test]
    fn chunk_record_roundtrips() {
        let rec = ChunkRecord {
            id: 42,
            path_off: 3,
            path_len: 11,
            payload_off: 64,
            payload_len: 128,
            uncompressed_len: 900,
            line_start: 10,
            line_end: 70,
            token_estimate: 220,
            hash_prefix: [9u8; 16],
        };
        let decoded = ChunkRecord::decode(&rec.encode()).unwrap();
        assert_eq!(rec, decoded);
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = sample_header().encode();
        bytes[0] = b'X';
        assert!(matches!(
            Header::decode(&bytes, 400),
            Err(AimError::BadMagic { .. })
        ));
    }

    #[test]
    fn rejects_section_past_end_of_file() {
        let h = sample_header();
        // File is one byte short of the payload section's end.
        let err = Header::decode(&h.encode(), 399).unwrap_err();
        assert!(matches!(err, AimError::SectionOutOfBounds { .. }), "{err:?}");
    }

    #[test]
    fn rejects_gist_len_that_disagrees_with_dim() {
        let mut h = sample_header();
        h.gist_len = 16; // dim is 8, so this must be 32
        assert!(matches!(
            Header::decode(&h.encode(), 400),
            Err(AimError::Corrupt(_))
        ));
    }

    #[test]
    fn rejects_chunk_table_len_that_disagrees_with_count() {
        let mut h = sample_header();
        h.chunk_count = 3; // table_len says 2 records
        assert!(matches!(
            Header::decode(&h.encode(), 400),
            Err(AimError::Corrupt(_))
        ));
    }

    #[test]
    fn align_up_rounds_to_16() {
        assert_eq!(align_up(0), 0);
        assert_eq!(align_up(1), 16);
        assert_eq!(align_up(16), 16);
        assert_eq!(align_up(17), 32);
    }
}
