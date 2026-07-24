//! Error type for catalog construction, loading, and retrieval.

use std::path::PathBuf;

/// Everything that can go wrong reading or building a `.aim` catalog.
#[derive(Debug, thiserror::Error)]
pub enum AimError {
    #[error("not a .aim catalog: expected magic {expected:?}, found {found:?}", expected = crate::format::MAGIC)]
    BadMagic { found: Vec<u8> },

    #[error(
        "catalog format version {found} is not supported (this build reads version {supported}); \
         rebuild the catalog with `aim-index`"
    )]
    UnsupportedVersion { found: u32, supported: u32 },

    #[error("catalog truncated while reading {what}: need {need} bytes, have {have}")]
    Truncated {
        what: &'static str,
        need: usize,
        have: usize,
    },

    #[error("catalog corrupt: {0}")]
    Corrupt(&'static str),

    #[error(
        "catalog section `{section}` ends at byte {end} but the file is only {file_len} bytes"
    )]
    SectionOutOfBounds {
        section: &'static str,
        end: u64,
        file_len: u64,
    },

    #[error("chunk id {0} is not present in this catalog")]
    UnknownChunk(u64),

    #[error(
        "embedding dimension mismatch: catalog was built with dim {catalog}, \
         got a {got}-dimensional vector"
    )]
    DimMismatch { catalog: usize, got: usize },

    #[error("chunk {id} decompressed to {got} bytes but the record declares {expected}")]
    PayloadLengthMismatch { id: u64, expected: usize, got: usize },

    #[error("chunk {id} contains invalid UTF-8 at byte {offset}")]
    InvalidUtf8 { id: u64, offset: usize },

    #[error("cannot build a catalog with zero chunks")]
    EmptyCatalog,

    #[error("turbovec index rejected the vectors: {0}")]
    Index(String),

    #[error("failed to {action} `{path}`")]
    Io {
        action: &'static str,
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
}

impl AimError {
    /// Attach a path and an action verb to an [`std::io::Error`].
    pub(crate) fn io(action: &'static str, path: impl Into<PathBuf>, source: std::io::Error) -> Self {
        AimError::Io {
            action,
            path: path.into(),
            source,
        }
    }
}
