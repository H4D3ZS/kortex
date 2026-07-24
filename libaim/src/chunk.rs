//! Splitting source files into retrievable chunks.
//!
//! Chunks are overlapping line windows. This is deliberately
//! language-agnostic: a syntax-aware splitter would need a grammar per
//! language, and the failure mode when a grammar is missing or the file
//! does not parse is worse than a slightly ragged window boundary.
//!
//! The overlap matters. A function whose signature lands on the last
//! line of one window would otherwise be split from its body, and
//! neither half would score well against a query naming the function.

/// One chunk of a source file, before it is embedded and written.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceChunk {
    /// Workspace-relative path, always with `/` separators so a catalog
    /// built on Windows resolves identically on macOS.
    pub path: String,
    /// 1-based inclusive line range this chunk covers.
    pub line_start: u32,
    pub line_end: u32,
    /// The chunk's text.
    pub text: String,
}

impl SourceChunk {
    /// Rough token count: source code runs about 3.2 bytes per token for
    /// BPE tokenizers. Used only to spend a retrieval budget, so a
    /// cheap estimate beats loading a real tokenizer.
    pub fn token_estimate(&self) -> u32 {
        ((self.text.len() as f32 / 3.2).ceil() as u32).max(1)
    }
}

/// How to split files into chunks.
#[derive(Debug, Clone, Copy)]
pub struct ChunkConfig {
    /// Target window height in lines.
    pub window_lines: usize,
    /// Lines of overlap between consecutive windows.
    pub overlap_lines: usize,
    /// Hard cap on a chunk's byte length. A single minified or generated
    /// line can be megabytes; without this cap one such line would
    /// dominate the payload and blow the prompt budget when inflated.
    pub max_chunk_bytes: usize,
}

impl Default for ChunkConfig {
    fn default() -> Self {
        Self {
            window_lines: 60,
            overlap_lines: 12,
            max_chunk_bytes: 8 * 1024,
        }
    }
}

impl ChunkConfig {
    /// Distance between the start of one window and the next.
    fn stride(&self) -> usize {
        self.window_lines.saturating_sub(self.overlap_lines).max(1)
    }
}

/// Split one file's contents into overlapping line windows.
///
/// Returns an empty vec for whitespace-only input: an all-zero embedding
/// carries no signal and would only waste an index slot.
pub fn chunk_source(path: &str, contents: &str, cfg: &ChunkConfig) -> Vec<SourceChunk> {
    if contents.trim().is_empty() {
        return Vec::new();
    }

    let path = path.replace('\\', "/");
    let lines: Vec<&str> = contents.lines().collect();
    let stride = cfg.stride();
    let mut out = Vec::new();
    let mut start = 0usize;

    while start < lines.len() {
        let end = (start + cfg.window_lines).min(lines.len());

        // Grow the window line by line until the byte cap is hit, so a
        // single oversized line still produces one (truncated) chunk
        // rather than being dropped entirely.
        let mut text = String::new();
        let mut last_line = start;
        for (i, line) in lines[start..end].iter().enumerate() {
            if !text.is_empty() && text.len() + line.len() + 1 > cfg.max_chunk_bytes {
                break;
            }
            if !text.is_empty() {
                text.push('\n');
            }
            if line.len() > cfg.max_chunk_bytes {
                text.push_str(truncate_on_char_boundary(line, cfg.max_chunk_bytes));
            } else {
                text.push_str(line);
            }
            last_line = start + i;
        }

        if !text.trim().is_empty() {
            out.push(SourceChunk {
                path: path.clone(),
                line_start: start as u32 + 1,
                line_end: last_line as u32 + 1,
                text,
            });
        }

        // Advance past what was actually consumed. When the byte cap cut
        // the window short, stepping by `stride` would skip the lines
        // that did not fit.
        let consumed = last_line + 1 - start;
        start += stride.min(consumed.max(1));
        if consumed >= lines.len() - start.min(lines.len() - 1) && last_line + 1 >= lines.len() {
            break;
        }
    }

    out
}

/// Truncate to at most `max` bytes without splitting a UTF-8 sequence.
fn truncate_on_char_boundary(s: &str, max: usize) -> &str {
    if s.len() <= max {
        return s;
    }
    let mut end = max;
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[..end]
}

/// File extensions worth indexing. Restricting by extension keeps
/// binaries, lockfiles and build artifacts out of the catalog, which is
/// what keeps a 30k-file workspace's index small.
pub const DEFAULT_EXTENSIONS: &[&str] = &[
    "rs", "c", "h", "cpp", "cc", "hpp", "m", "mm", "swift", "go", "py", "ts", "tsx", "js", "jsx",
    "java", "kt", "cs", "rb", "php", "lua", "sh", "ps1", "sql", "toml", "yaml", "yml", "json",
    "md", "proto", "hip", "cu", "metal", "asm", "s",
];

/// Directory names never worth walking into.
pub const DEFAULT_IGNORED_DIRS: &[&str] = &[
    ".git", "target", "node_modules", "dist", "build", ".venv", "venv", "__pycache__", ".next",
    ".cache", "vendor", ".aim", "out", "Pods", ".idea", ".vscode-test",
];

/// True when a path's extension is in [`DEFAULT_EXTENSIONS`].
pub fn has_indexable_extension(path: &std::path::Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| {
            let e = e.to_ascii_lowercase();
            DEFAULT_EXTENSIONS.contains(&e.as_str())
        })
        .unwrap_or(false)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn numbered(n: usize) -> String {
        (1..=n).map(|i| format!("line {i}")).collect::<Vec<_>>().join("\n")
    }

    #[test]
    fn empty_and_whitespace_files_produce_no_chunks() {
        let cfg = ChunkConfig::default();
        assert!(chunk_source("a.rs", "", &cfg).is_empty());
        assert!(chunk_source("a.rs", "   \n\n  \t\n", &cfg).is_empty());
    }

    #[test]
    fn short_file_is_one_chunk_covering_every_line() {
        let cfg = ChunkConfig::default();
        let chunks = chunk_source("a.rs", &numbered(10), &cfg);
        assert_eq!(chunks.len(), 1);
        assert_eq!((chunks[0].line_start, chunks[0].line_end), (1, 10));
    }

    #[test]
    fn windows_overlap_by_the_configured_amount() {
        let cfg = ChunkConfig {
            window_lines: 10,
            overlap_lines: 3,
            max_chunk_bytes: 8192,
        };
        let chunks = chunk_source("a.rs", &numbered(30), &cfg);
        assert!(chunks.len() >= 3, "got {} chunks", chunks.len());
        // stride = 7, so window 2 starts at line 8 while window 1 ended
        // at line 10 — three lines of overlap.
        assert_eq!(chunks[0].line_start, 1);
        assert_eq!(chunks[1].line_start, 8);
        assert!(chunks[1].line_start <= chunks[0].line_end);
    }

    #[test]
    fn every_line_of_the_file_appears_in_some_chunk() {
        let cfg = ChunkConfig {
            window_lines: 10,
            overlap_lines: 3,
            max_chunk_bytes: 8192,
        };
        let n = 97;
        let chunks = chunk_source("a.rs", &numbered(n), &cfg);
        let mut covered = vec![false; n + 1];
        for c in &chunks {
            for line in c.line_start..=c.line_end {
                covered[line as usize] = true;
            }
        }
        let missing: Vec<usize> = (1..=n).filter(|i| !covered[*i]).collect();
        assert!(missing.is_empty(), "lines not covered: {missing:?}");
    }

    #[test]
    fn oversized_single_line_is_truncated_not_dropped() {
        let cfg = ChunkConfig {
            window_lines: 60,
            overlap_lines: 12,
            max_chunk_bytes: 100,
        };
        let huge = "x".repeat(10_000);
        let chunks = chunk_source("min.js", &huge, &cfg);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].text.len() <= 100);
    }

    #[test]
    fn byte_cap_does_not_skip_lines() {
        // Each line is ~50 bytes; a 120-byte cap fits about 2 lines,
        // far fewer than the 60-line window. The walker must not jump
        // a full stride and lose the lines that did not fit.
        let cfg = ChunkConfig {
            window_lines: 60,
            overlap_lines: 0,
            max_chunk_bytes: 120,
        };
        let body = (1..=20)
            .map(|i| format!("{:0>45}{i}", "a"))
            .collect::<Vec<_>>()
            .join("\n");
        let chunks = chunk_source("a.rs", &body, &cfg);
        let mut covered = vec![false; 21];
        for c in &chunks {
            for line in c.line_start..=c.line_end {
                covered[line as usize] = true;
            }
        }
        let missing: Vec<usize> = (1..=20).filter(|i| !covered[*i]).collect();
        assert!(missing.is_empty(), "lines not covered: {missing:?}");
    }

    #[test]
    fn truncation_respects_utf8_boundaries() {
        let cfg = ChunkConfig {
            window_lines: 60,
            overlap_lines: 12,
            // Cap lands mid-way through a 3-byte character.
            max_chunk_bytes: 10,
        };
        let chunks = chunk_source("a.rs", &"→".repeat(50), &cfg);
        assert_eq!(chunks.len(), 1);
        // Reaching here at all proves no panic on a non-boundary slice.
        assert!(chunks[0].text.chars().all(|c| c == '→'));
    }

    #[test]
    fn backslash_paths_are_normalized() {
        let cfg = ChunkConfig::default();
        let chunks = chunk_source("src\\hw\\mbox.c", "int main(void){}", &cfg);
        assert_eq!(chunks[0].path, "src/hw/mbox.c");
    }

    #[test]
    fn token_estimate_is_nonzero_and_scales_with_length() {
        let short = SourceChunk {
            path: "a.rs".into(),
            line_start: 1,
            line_end: 1,
            text: "x".into(),
        };
        let long = SourceChunk {
            text: "x".repeat(3200),
            ..short.clone()
        };
        assert!(short.token_estimate() >= 1);
        assert!(long.token_estimate() > short.token_estimate());
    }

    #[test]
    fn extension_filter_accepts_source_and_rejects_binaries() {
        use std::path::Path;
        assert!(has_indexable_extension(Path::new("a/b/c.rs")));
        assert!(has_indexable_extension(Path::new("k.HIP")));
        assert!(!has_indexable_extension(Path::new("a.png")));
        assert!(!has_indexable_extension(Path::new("Cargo.lock")));
    }
}
