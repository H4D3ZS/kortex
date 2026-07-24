//! Deciding whether a request is worth retrieving for at all.
//!
//! # Why this is not a similarity threshold
//!
//! The obvious design is "retrieve, and inject if the top score clears a
//! bar". Measured against a real 243k-chunk workspace, that does not
//! work — and the failure is not a tuning problem:
//!
//! | query                                               | top cosine |
//! |-----------------------------------------------------|-----------|
//! | `thanks`                                            | 0.44      |
//! | `hello`                                             | 0.38      |
//! | `fix the AGX mailbox IRQ starvation in apple_mbox.c` | 0.32      |
//! | `implement zstd decompression for chunk payloads`   | 0.16      |
//!
//! Score is *anti-correlated* with relevance. Two effects cause it, and
//! they compound:
//!
//! 1. A one-word query puts all its mass on one feature, so any chunk
//!    containing that word scores near-perfectly. A long query spreads
//!    mass across many features that no single chunk contains.
//! 2. IDF makes this worse, not better. `hello` and `thanks` are *rare*
//!    in source code, so IDF weights them **up**.
//!
//! Cosine measures "does this chunk contain these tokens". It cannot
//! measure "is this request about the codebase". So the gate looks at
//! the query instead: a request needs enough distinct, non-conversational
//! words before searching for code is a sensible thing to do.
//!
//! IDF is still applied — it genuinely improves *ranking* among
//! qualifying queries — it is just not a gate.

/// Words that carry no retrieval signal: English function words plus the
/// conversational filler that surrounds agent chat.
///
/// Deliberately short. Every entry costs recall if it also appears as an
/// identifier, so this covers only words that are near-useless for
/// locating code.
const STOPWORDS: &[&str] = &[
    // articles, conjunctions, prepositions
    "a", "an", "the", "and", "or", "but", "if", "then", "else", "of", "to", "in", "on", "at",
    "by", "for", "with", "from", "into", "over", "under", "as", "is", "are", "was", "were", "be",
    "been", "being", "am", "do", "does", "did", "have", "has", "had", "will", "would", "shall",
    "should", "can", "could", "may", "might", "must", "not", "no", "yes", "this", "that", "these",
    "those", "it", "its", "i", "you", "we", "they", "he", "she", "me", "my", "your", "our",
    "their", "what", "which", "who", "when", "where", "how", "why", "all", "any", "some", "more",
    "most", "very", "just", "only", "also", "so", "than", "too", "up", "out", "about",
    // conversational filler
    "hi", "hello", "hey", "thanks", "thank", "thx", "please", "ok", "okay", "sure", "yeah",
    "yep", "nope", "cool", "nice", "great", "good", "bad", "sorry", "oops", "wow", "hmm", "sounds",
    "let", "lets", "now", "again", "still", "here", "there",
];

/// Query-side gate configuration.
#[derive(Debug, Clone, Copy)]
pub struct QueryGate {
    /// Minimum distinct non-stopword tokens before retrieval runs.
    ///
    /// Three is the smallest value that rejects the observed failure
    /// cases (`hello`, `thanks`, `ok sounds good`, `what is 2+2`) while
    /// admitting terse but real requests (`apple_mbox irq starvation`).
    pub min_content_tokens: usize,
    /// A token at least this long counts as content even if it is a
    /// stopword-shaped word, because long tokens in a code request are
    /// almost always identifiers.
    pub long_token_len: usize,
}

impl Default for QueryGate {
    fn default() -> Self {
        Self {
            min_content_tokens: 3,
            long_token_len: 12,
        }
    }
}

/// Why a query was or was not admitted.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GateDecision {
    /// Worth retrieving for. Carries the distinct content-token count.
    Retrieve { content_tokens: usize },
    /// Not worth retrieving for.
    Skip { reason: SkipReason },
}

/// Why retrieval was skipped.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SkipReason {
    /// The query had no usable text.
    Empty,
    /// Too few distinct content tokens.
    TooFewContentTokens { found: usize, needed: usize },
}

impl GateDecision {
    /// True when retrieval should run.
    pub fn should_retrieve(&self) -> bool {
        matches!(self, GateDecision::Retrieve { .. })
    }

    /// A short human-readable explanation, for logs.
    pub fn describe(&self) -> String {
        match self {
            GateDecision::Retrieve { content_tokens } => {
                format!("{content_tokens} content tokens")
            }
            GateDecision::Skip {
                reason: SkipReason::Empty,
            } => "query is empty".to_string(),
            GateDecision::Skip {
                reason: SkipReason::TooFewContentTokens { found, needed },
            } => format!("only {found} content tokens, need {needed}"),
        }
    }
}

/// True when `token` is a stopword.
pub fn is_stopword(token: &str) -> bool {
    STOPWORDS.contains(&token)
}

/// True when a lowercase token is unmistakably code: a path, a
/// snake_case identifier, or a filename with a known source extension.
///
/// A bare dot is not enough — "e.g." and "1.5" both contain one — so a
/// dotted token only qualifies when its suffix is an extension we index.
pub fn looks_like_code(token: &str) -> bool {
    if token.contains('/') || token.contains('_') {
        return true;
    }
    token
        .rsplit_once('.')
        .map(|(stem, ext)| {
            !stem.is_empty() && crate::chunk::DEFAULT_EXTENSIONS.contains(&ext)
        })
        .unwrap_or(false)
}

impl QueryGate {
    /// Decide whether `query` warrants a catalog search.
    pub fn evaluate(&self, query: &str) -> GateDecision {
        if query.trim().is_empty() {
            return GateDecision::Skip {
                reason: SkipReason::Empty,
            };
        }

        let mut content: Vec<String> = Vec::new();
        for raw in query.split(|c: char| !c.is_alphanumeric() && c != '_' && c != '.' && c != '/') {
            let token = raw.trim_matches(|c: char| c == '.' || c == '/');
            if token.is_empty() {
                continue;
            }
            let lower = token.to_ascii_lowercase();

            // A filename or a snake_case/path identifier is by itself
            // conclusive evidence that this is a code request, so it
            // short-circuits the token count. Without this, the terse
            // requests agents actually send — "apple_mbox.c" — would be
            // rejected for being one token long.
            if looks_like_code(&lower) {
                return GateDecision::Retrieve {
                    content_tokens: content.len().max(1),
                };
            }

            // Not code, so any remaining dots are prose punctuation
            // ("e.g.") or a decimal point ("1.5"). Split on them and
            // judge the pieces, or those two would each count as a
            // content token and drag junk past the gate.
            for part in lower.split('.').filter(|p| !p.is_empty()) {
                // Single letters and bare numbers carry no retrieval
                // signal — this is why "what is 2+2" must not search.
                if part.len() < 2 || part.chars().all(|c| c.is_ascii_digit()) {
                    continue;
                }
                if part.len() < self.long_token_len && is_stopword(part) {
                    continue;
                }
                if !content.iter().any(|c| c == part) {
                    content.push(part.to_string());
                }
            }
        }

        if content.len() < self.min_content_tokens {
            return GateDecision::Skip {
                reason: SkipReason::TooFewContentTokens {
                    found: content.len(),
                    needed: self.min_content_tokens,
                },
            };
        }
        GateDecision::Retrieve {
            content_tokens: content.len(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn gate() -> QueryGate {
        QueryGate::default()
    }

    /// Every one of these was measured scoring *above* a real technical
    /// query on the 243k-chunk workspace. They must not retrieve.
    #[test]
    fn rejects_conversational_filler() {
        for q in [
            "hello",
            "thanks",
            "thank you",
            "ok sounds good",
            "what is 2+2",
            "hi",
            "yes please",
            "cool, nice",
            "ok",
        ] {
            assert!(
                !gate().evaluate(q).should_retrieve(),
                "{q:?} should not trigger retrieval ({})",
                gate().evaluate(q).describe()
            );
        }
    }

    #[test]
    fn admits_real_coding_requests() {
        for q in [
            "fix the AGX mailbox IRQ starvation in apple_mbox.c",
            "how does the tauri command registry dispatch IPC handlers",
            "implement zstd decompression for chunk payloads",
            "turbovec IdMapIndex allowlist search",
            "apple_mbox irq starvation",
            "why does pl011_write_fifo drop bytes",
        ] {
            assert!(
                gate().evaluate(q).should_retrieve(),
                "{q:?} should trigger retrieval ({})",
                gate().evaluate(q).describe()
            );
        }
    }

    #[test]
    fn a_single_filename_is_enough_signal() {
        // One token, but unmistakably a code request. The filename
        // escape hatch is what keeps terse requests working.
        assert!(gate()
            .evaluate("src/hw/misc/apple_mbox.c")
            .should_retrieve());
        assert!(gate().evaluate("apple_mbox.c").should_retrieve());
    }

    #[test]
    fn empty_query_is_skipped_with_a_clear_reason() {
        assert_eq!(
            gate().evaluate("   \n\t "),
            GateDecision::Skip {
                reason: SkipReason::Empty
            }
        );
        assert_eq!(gate().evaluate("").describe(), "query is empty");
    }

    #[test]
    fn a_dotted_non_filename_is_not_treated_as_code() {
        // "e.g." and version numbers must not smuggle a query past the
        // gate, or every conversational message retrieves.
        assert!(!looks_like_code("e.g"));
        assert!(!looks_like_code("1.5"));
        assert!(!gate().evaluate("e.g. version 1.5").should_retrieve());
        assert!(looks_like_code("apple_mbox.c"));
        assert!(looks_like_code("src/main"));
    }

    #[test]
    fn bare_numbers_do_not_count_as_content() {
        assert!(!gate().evaluate("1 2 3 4 5").should_retrieve());
    }

    #[test]
    fn repeated_words_count_once() {
        // "mailbox mailbox mailbox" is one distinct token, not three.
        assert!(!gate().evaluate("mailbox mailbox mailbox").should_retrieve());
    }

    #[test]
    fn long_tokens_bypass_the_stopword_list() {
        // A long identifier that happens to start with a stopword-ish
        // word must still count.
        let g = QueryGate {
            min_content_tokens: 1,
            long_token_len: 6,
        };
        assert!(g.evaluate("thereIsSomethingHere").should_retrieve());
    }

    #[test]
    fn decision_explains_itself_for_logs() {
        let d = gate().evaluate("hello");
        assert!(d.describe().contains("content tokens"), "{}", d.describe());
        let d = gate().evaluate("fix apple_mbox irq starvation");
        assert!(d.describe().contains("content tokens"));
    }

    #[test]
    fn threshold_is_configurable() {
        let strict = QueryGate {
            min_content_tokens: 10,
            ..Default::default()
        };
        assert!(!strict.evaluate("fix the mailbox irq bug").should_retrieve());
    }
}
