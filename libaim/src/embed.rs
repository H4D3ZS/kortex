//! Embedding backends.
//!
//! Retrieval only works if the corpus vectors and the query vector come
//! out of the *same* function, so both the indexer and the proxy go
//! through the [`Embedder`] trait and record which backend built a
//! catalog.
//!
//! Two backends ship here:
//!
//! * [`HashEmbedder`] — a deterministic signed feature-hash over code
//!   tokens. No model, no network, no GPU. It is a *lexical* embedder:
//!   it matches on shared identifiers, paths and n-grams, not on
//!   paraphrase. That is a genuine limitation, and it is also why it is
//!   the default — it is hermetic, sub-microsecond, and strong on the
//!   query shape agents actually send ("fix the IRQ starvation in
//!   apple_mbox.c" shares literal tokens with the chunk that defines it).
//! * `HttpEmbedder` (in [`crate::embed_http`], behind the `http-embed`
//!   feature) — real dense embeddings from a local inference server:
//!   Lemonade, any OpenAI-compatible endpoint, or Ollama. Use it when
//!   queries are phrased in prose that shares no tokens with the code.
//!
//! Both emit L2-normalized vectors. turbovec scores inner products, so
//! normalizing makes a score exactly a cosine similarity, which is what
//! lets the fault threshold be interpreted as a similarity floor.

use crate::error::AimError;

/// Produces embeddings for text. Implementations must be deterministic
/// and must return L2-normalized vectors of exactly [`Embedder::dim`].
pub trait Embedder: Send + Sync {
    /// Dimension of every vector this embedder returns.
    fn dim(&self) -> usize;

    /// Short stable name recorded in catalog metadata, so a query-time
    /// mismatch against the catalog can be reported instead of silently
    /// returning nonsense scores.
    fn id(&self) -> String;

    /// Embed one text into a `dim`-length L2-normalized vector.
    fn embed(&self, text: &str) -> Result<Vec<f32>, AimError>;

    /// Embed a batch. The default is a loop; network-backed backends
    /// override this to amortize round trips.
    fn embed_batch(&self, texts: &[String]) -> Result<Vec<f32>, AimError> {
        let mut out = Vec::with_capacity(texts.len() * self.dim());
        for t in texts {
            out.extend_from_slice(&self.embed(t)?);
        }
        Ok(out)
    }
}

/// FNV-1a. Chosen over [`std::collections::hash_map::DefaultHasher`]
/// because that one is explicitly not stable across Rust releases — a
/// toolchain bump would silently invalidate every catalog on disk.
#[inline]
fn fnv1a(bytes: &[u8], seed: u64) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325 ^ seed;
    for &b in bytes {
        h ^= b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    h
}

/// Split an identifier into lowercase subwords, breaking on
/// non-alphanumerics, `camelCase` humps, and letter/digit boundaries.
///
/// `parse_HTTPResponse2` yields `["parse", "http", "response", "2"]`.
fn subwords(token: &str, out: &mut Vec<String>) {
    let chars: Vec<char> = token.chars().collect();
    let mut start = 0usize;
    for i in 1..=chars.len() {
        let split_here = if i == chars.len() {
            true
        } else {
            let (prev, cur) = (chars[i - 1], chars[i]);
            // lower→upper is a camel hump; a run of capitals followed by
            // a lowercase means the last capital starts a new word
            // (HTTPResponse → HTTP | Response).
            let camel = prev.is_lowercase() && cur.is_uppercase();
            let acronym_end = prev.is_uppercase()
                && cur.is_lowercase()
                && i >= 2
                && chars[i - 2].is_uppercase();
            let digit_edge = prev.is_alphabetic() != cur.is_alphabetic();
            camel || digit_edge || acronym_end
        };
        if split_here {
            let (s, e) = if i < chars.len()
                && chars[i - 1].is_uppercase()
                && chars[i].is_lowercase()
                && i >= 2
                && chars[i - 2].is_uppercase()
            {
                // Emit up to the acronym boundary, restart on the capital.
                (start, i - 1)
            } else {
                (start, i)
            };
            if e > s {
                let w: String = chars[s..e].iter().flat_map(|c| c.to_lowercase()).collect();
                if !w.is_empty() {
                    out.push(w);
                }
            }
            start = e;
        }
    }
}

/// Tokenize source text or a query into lowercase subword tokens.
fn tokenize(text: &str) -> Vec<String> {
    let mut out = Vec::new();
    for raw in text.split(|c: char| !c.is_alphanumeric() && c != '_') {
        if raw.is_empty() {
            continue;
        }
        for part in raw.split('_').filter(|p| !p.is_empty()) {
            subwords(part, &mut out);
        }
    }
    out
}

/// Deterministic signed feature-hash embedder over code tokens.
///
/// Features are token unigrams plus adjacent bigrams. Each feature is
/// hashed to a dimension and to a sign, accumulated with sublinear term
/// frequency (`1 + ln tf`), then L2-normalized. Signed hashing keeps
/// collisions unbiased in expectation rather than always additive.
#[derive(Debug, Clone)]
pub struct HashEmbedder {
    dim: usize,
    /// Extra weight applied to tokens from the chunk's file path.
    path_weight: f32,
    /// Per-slot inverse document frequency, when embedding against an
    /// existing catalog. Empty means unweighted — correct only while
    /// *building* a catalog, since IDF is not known until the whole
    /// corpus has been seen.
    idf: Vec<f32>,
}

impl HashEmbedder {
    /// Version tag baked into [`Embedder::id`]. Bump this whenever the
    /// feature extraction changes, so stale catalogs are detected
    /// instead of silently scoring against a different vector space.
    pub const ALGO_VERSION: u32 = 1;

    /// Create an unweighted embedder producing `dim`-dimensional
    /// vectors.
    ///
    /// Used during indexing, before document frequencies are known. To
    /// query an existing catalog use [`crate::Catalog::query_embedder`],
    /// which supplies that catalog's IDF table.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            path_weight: 2.0,
            idf: Vec::new(),
        }
    }

    /// Create an embedder that applies `idf` to every vector.
    ///
    /// `idf` must have exactly `dim` entries; a mismatched table is
    /// ignored rather than silently applied to the wrong slots.
    pub fn with_idf(dim: usize, idf: Vec<f32>) -> Self {
        let idf = if idf.len() == dim { idf } else { Vec::new() };
        Self {
            dim,
            path_weight: 2.0,
            idf,
        }
    }

    /// True when this embedder carries an IDF table.
    pub fn has_idf(&self) -> bool {
        !self.idf.is_empty()
    }

    /// Apply the IDF table, if present. Called before normalization so
    /// the result is still a unit vector.
    fn weight(&self, acc: &mut [f32]) {
        if self.idf.len() == acc.len() {
            for (value, w) in acc.iter_mut().zip(&self.idf) {
                *value *= *w;
            }
        }
    }

    /// Embed chunk text together with its path. Path tokens are weighted
    /// up because a query naming a file should strongly prefer chunks
    /// from that file.
    pub fn embed_with_path(&self, path: &str, text: &str) -> Result<Vec<f32>, AimError> {
        let mut acc = vec![0f32; self.dim];
        self.accumulate(&tokenize(path), self.path_weight, &mut acc);
        self.accumulate(&tokenize(text), 1.0, &mut acc);
        self.weight(&mut acc);
        normalize(&mut acc);
        Ok(acc)
    }

    /// Hash `tokens` into `acc`, scaling every contribution by `weight`.
    fn accumulate(&self, tokens: &[String], weight: f32, acc: &mut [f32]) {
        // Count features first so term frequency can be dampened; raw tf
        // lets one repeated identifier dominate a whole chunk.
        let mut counts: std::collections::HashMap<u64, f32> = std::collections::HashMap::new();

        for (i, tok) in tokens.iter().enumerate() {
            *counts.entry(fnv1a(tok.as_bytes(), 0)).or_insert(0.0) += 1.0;
            if i + 1 < tokens.len() {
                let mut bigram = tok.clone();
                bigram.push(' ');
                bigram.push_str(&tokens[i + 1]);
                // Distinct seed keeps the bigram space from aliasing
                // onto the unigram space.
                *counts.entry(fnv1a(bigram.as_bytes(), 0x9e37_79b9)).or_insert(0.0) += 1.0;
            }
        }

        for (feature, tf) in counts {
            let value = weight * (1.0 + tf.ln());
            let slot = (feature % self.dim as u64) as usize;
            // Use a high bit for the sign so it is independent of the
            // low bits that chose the slot.
            let sign = if feature & 0x8000_0000_0000_0000 != 0 {
                -1.0
            } else {
                1.0
            };
            acc[slot] += sign * value;
        }
    }
}

impl Embedder for HashEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn id(&self) -> String {
        format!("hash-v{}-d{}", Self::ALGO_VERSION, self.dim)
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, AimError> {
        let mut acc = vec![0f32; self.dim];
        self.accumulate(&tokenize(text), 1.0, &mut acc);
        self.weight(&mut acc);
        normalize(&mut acc);
        Ok(acc)
    }
}

/// L2-normalize in place. An all-zero vector is left as zeros: turbovec
/// rejects non-finite input, and a zero vector simply scores 0 against
/// every query, which is the correct behaviour for empty text.
pub fn normalize(v: &mut [f32]) {
    let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 && norm.is_finite() {
        for x in v.iter_mut() {
            *x /= norm;
        }
    } else {
        v.fill(0.0);
    }
}

/// Cosine similarity between two equal-length L2-normalized vectors.
/// Used to score the exact rerank pass after quantized search.
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn subwords_splits_camel_snake_and_digits() {
        let got = tokenize("parse_HTTPResponse2 applembox");
        assert_eq!(
            got,
            vec!["parse", "http", "response", "2", "applembox"]
        );
    }

    #[test]
    fn tokenize_handles_snake_case_paths() {
        let got = tokenize("src/hw/misc/apple_mbox.c");
        assert_eq!(got, vec!["src", "hw", "misc", "apple", "mbox", "c"]);
    }

    #[test]
    fn embeddings_are_unit_length() {
        let e = HashEmbedder::new(256);
        let v = e.embed("fn handle_irq(mbox: &Mailbox) -> Result<()>").unwrap();
        let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm was {norm}");
    }

    #[test]
    fn embedding_is_deterministic() {
        let e = HashEmbedder::new(128);
        assert_eq!(e.embed("static void apple_mbox_irq(void)").unwrap(),
                   e.embed("static void apple_mbox_irq(void)").unwrap());
    }

    #[test]
    fn empty_text_yields_zero_vector_not_nan() {
        let e = HashEmbedder::new(64);
        let v = e.embed("").unwrap();
        assert!(v.iter().all(|x| *x == 0.0));
        assert!(v.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn related_code_scores_above_unrelated_code() {
        let e = HashEmbedder::new(1536);
        let query = e.embed("fix the AGX mailbox IRQ starvation in apple_mbox.c").unwrap();

        let relevant = e
            .embed_with_path(
                "src/hw/misc/apple_mbox.c",
                "static void apple_mbox_irq_starvation(AppleMboxState *s) { \
                 qemu_irq_raise(s->irq); }",
            )
            .unwrap();
        let irrelevant = e
            .embed_with_path(
                "docs/CHANGELOG.md",
                "## 1.2.0 - Updated the README with new installation instructions.",
            )
            .unwrap();

        let hit = cosine(&query, &relevant);
        let miss = cosine(&query, &irrelevant);
        assert!(
            hit > miss,
            "expected relevant chunk to win: hit={hit} miss={miss}"
        );
    }

    #[test]
    fn path_tokens_are_weighted_into_the_vector() {
        let e = HashEmbedder::new(1536);
        // Identical body text, different paths: naming the path in the
        // query must break the tie.
        let query = e.embed("apple_mbox").unwrap();
        let same_path = e.embed_with_path("hw/apple_mbox.c", "int x = 1;").unwrap();
        let other_path = e.embed_with_path("hw/serial_pl011.c", "int x = 1;").unwrap();
        assert!(cosine(&query, &same_path) > cosine(&query, &other_path));
    }
}
