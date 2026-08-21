//! Query-time retrieval for the proxy.
//!
//! What changed here matters more than the code: the proxy used to read
//! a fixed `.aim` blob and staple a summary of it onto *every* prompt.
//! That costs the same tokens on every request and gets less useful as
//! the workspace grows. This module does retrieval instead — embed the
//! request, search the catalog, and inject only the chunks that clear
//! the gate.
//!
//! # Staying off the reactor
//!
//! Editors fire completion, lint and chat requests concurrently. Search
//! plus zstd inflation is CPU work measured in tens of milliseconds, and
//! running it on the async reactor would stall every other in-flight
//! socket. So retrieval runs on the blocking pool under a wall-clock
//! budget, and if it overruns, the request goes upstream un-augmented.
//! A slow catalog degrades to a plain proxy; it never hangs the editor.

use std::path::PathBuf;
use std::sync::{Arc, Mutex, RwLock};
use std::time::{Duration, Instant};

use libaim::{
    Catalog, Embedder, HashEmbedder, HeatMap, LiveCatalog, QueryGate, RetrievalConfig,
    WatchConfig, WatcherHandle,
};
use serde_json::Value;

/// Environment variable holding the catalog directory.
pub const ENV_CATALOG: &str = "KORTEX_AIM_CATALOG";
/// Environment variable holding the retrieval budget in milliseconds.
pub const ENV_BUDGET_MS: &str = "KORTEX_RETRIEVAL_BUDGET_MS";
/// Environment variable overriding the relative gate.
pub const ENV_RELATIVE: &str = "KORTEX_RETRIEVAL_RELATIVE";
/// Environment variable overriding the injected token budget.
pub const ENV_TOKEN_BUDGET: &str = "KORTEX_RETRIEVAL_TOKEN_BUDGET";
/// Environment variable overriding the minimum content-token gate.
pub const ENV_MIN_TOKENS: &str = "KORTEX_RETRIEVAL_MIN_TOKENS";
/// Set to `0` or `false` to disable the filesystem watcher.
pub const ENV_WATCH: &str = "KORTEX_WATCH";
/// Workspace root to watch. Defaults to the catalog's recorded root.
pub const ENV_WORKSPACE: &str = "KORTEX_WORKSPACE";

/// Default wall-clock budget for one retrieval.
const DEFAULT_BUDGET_MS: u64 = 100;

/// Shared retrieval state.
pub struct RetrievalEngine {
    catalog: Option<Arc<RwLock<LiveCatalog>>>,
    /// Whichever backend built the loaded catalog (lexical hash OR dense
    /// http). Arc so a `spawn_blocking` retrieval can hold its own handle.
    embedder: Arc<dyn Embedder>,
    /// Fuse a lexical BM25 channel with the dense one (hybrid retrieval).
    /// On only for dense catalogs — BM25 adds nothing to an already-lexical
    /// hash catalog. Fixes exact-symbol queries the dense channel misses.
    hybrid: bool,
    cfg: RetrievalConfig,
    gate: QueryGate,
    budget: Duration,
    heat: Arc<Mutex<HeatMap>>,
    /// Held to keep the watcher thread alive for the process lifetime.
    /// Dropping it stops watching.
    _watcher: Option<WatcherHandle>,
}

impl RetrievalEngine {
    /// Load the catalog named by [`ENV_CATALOG`], or the first of a few
    /// conventional locations.
    ///
    /// A missing or unreadable catalog is not fatal: the engine reports
    /// why once at startup and then behaves as a pass-through, which is
    /// the right failure mode for something sitting in front of an
    /// editor.
    pub fn load() -> Self {
        let mut cfg = RetrievalConfig::default();
        if let Some(v) = env_parse::<f32>(ENV_RELATIVE) {
            cfg.relative_floor = v;
        }
        if let Some(v) = env_parse::<u32>(ENV_TOKEN_BUDGET) {
            cfg.token_budget = v;
        }
        let budget = Duration::from_millis(
            env_parse::<u64>(ENV_BUDGET_MS).unwrap_or(DEFAULT_BUDGET_MS),
        );
        let mut gate = QueryGate::default();
        if let Some(v) = env_parse::<usize>(ENV_MIN_TOKENS) {
            gate.min_content_tokens = v;
        }

        let mut engine = Self {
            catalog: None,
            embedder: Arc::new(HashEmbedder::new(libaim::DEFAULT_DIM)),
            hybrid: false,
            cfg,
            gate,
            budget,
            heat: Arc::new(Mutex::new(HeatMap::default())),
            _watcher: None,
        };

        let Some(dir) = Self::locate_catalog() else {
            eprintln!(
                "[aim-proxy] no .aim catalog found. Set {ENV_CATALOG}, or build one with \
                 `aim-index build <workspace>`. Running as a plain pass-through proxy."
            );
            return engine;
        };

        match Catalog::open(&dir) {
            Ok(catalog) => {
                // Build the embedder that matches whichever backend built
                // this catalog (lexical hash OR dense http). A mismatch
                // makes every score noise, so bail to pass-through instead
                // of injecting unrelated code. For hash catalogs this also
                // carries the catalog's IDF weights; for http it reconnects
                // to the embedding server recorded in the catalog id.
                let embedder = match catalog.query_embedder_dyn() {
                    Ok(e) => e,
                    Err(e) => {
                        eprintln!("[aim-proxy] catalog at {} unusable: {e}", dir.display());
                        return engine;
                    }
                };
                eprintln!(
                    "[aim-proxy] catalog loaded: {} chunks, {} dims, embedder `{}`, from {}",
                    catalog.len(),
                    catalog.dim(),
                    embedder.id(),
                    dir.display()
                );

                // The workspace the catalog's paths are relative to.
                // Without it the watcher cannot form matching relative
                // paths, and every edit would shadow nothing.
                let workspace = std::env::var(ENV_WORKSPACE)
                    .map(PathBuf::from)
                    .unwrap_or_else(|_| PathBuf::from(&catalog.meta().root));

                let live = Arc::new(RwLock::new(LiveCatalog::new(catalog)));
                // Hybrid (dense + lexical BM25) for dense catalogs; disable
                // with KORTEX_HYBRID=0. Hash catalogs gain nothing from it.
                engine.hybrid = !embedder.id().starts_with("hash-")
                    && std::env::var("KORTEX_HYBRID")
                        .map(|v| v != "0" && v != "false")
                        .unwrap_or(true);
                if engine.hybrid {
                    eprintln!("[aim-proxy] hybrid retrieval on (dense + lexical BM25)");
                }
                engine.embedder = Arc::from(embedder);

                if watching_enabled() {
                    match libaim::start_watcher(
                        &workspace,
                        live.clone(),
                        WatchConfig::default(),
                    ) {
                        Ok(handle) => {
                            eprintln!(
                                "[aim-proxy] watching {} — edits refresh retrieval automatically",
                                workspace.display()
                            );
                            engine._watcher = Some(handle);
                        }
                        Err(e) => {
                            // Retrieval still works, it just goes stale
                            // on edits. Worth saying out loud.
                            eprintln!(
                                "[aim-proxy] could not watch {}: {e}. Retrieval will serve the                                  catalog as built until the next `aim-index build`.",
                                workspace.display()
                            );
                        }
                    }
                }

                engine.catalog = Some(live);
            }
            Err(e) => {
                eprintln!(
                    "[aim-proxy] could not open catalog at {}: {e}",
                    dir.display()
                );
            }
        }
        engine
    }

    /// True when a usable catalog is loaded.
    pub fn is_active(&self) -> bool {
        self.catalog.is_some()
    }

    /// Find a catalog directory to use.
    fn locate_catalog() -> Option<PathBuf> {
        if let Ok(p) = std::env::var(ENV_CATALOG) {
            let p = PathBuf::from(p);
            return p.join(libaim::CONTAINER_FILE).exists().then_some(p);
        }
        // Walk up from the working directory: the proxy is usually
        // launched from somewhere inside the workspace it serves.
        let mut dir = std::env::current_dir().ok()?;
        loop {
            let candidate = dir.join(".aim");
            if candidate.join(libaim::CONTAINER_FILE).exists() {
                return Some(candidate);
            }
            if !dir.pop() {
                return None;
            }
        }
    }

    /// Retrieve context for `query`, or `None` if nothing qualified,
    /// no catalog is loaded, or the budget was exceeded.
    pub async fn context_for(&self, query: &str) -> Option<String> {
        let catalog = self.catalog.clone()?;

        // Decide from the query itself, before touching the index.
        // Similarity cannot make this call — see `libaim::gate`.
        let decision = self.gate.evaluate(query);
        if !decision.should_retrieve() {
            println!("[aim-proxy] skipped retrieval: {}", decision.describe());
            return None;
        }

        let embedder = self.embedder.clone();
        let hybrid = self.hybrid;
        let cfg = self.cfg;
        let heat = self.heat.clone();
        let query = query.to_string();
        let started = Instant::now();

        // The CPU work goes to the blocking pool; the reactor thread
        // only awaits the join handle.
        let work = tokio::task::spawn_blocking(move || {
            let vector = embedder.embed(&query).ok()?;
            let hints = path_hints(&query);
            let hint_refs: Vec<&str> = hints.iter().map(|s| s.as_str()).collect();

            // Read lock: concurrent requests retrieve in parallel and
            // only a watcher ingest blocks them, briefly.
            let guard = catalog.read().ok()?;
            let fault = if hybrid && hint_refs.is_empty() {
                // hybrid runs on the base catalog (dense + lexical BM25);
                // live edits still flow through the page_fault paths below.
                guard.base().hybrid_fault(&query, &vector, &cfg)
            } else if hint_refs.is_empty() {
                guard.page_fault(&vector, &cfg)
            } else {
                guard.page_fault_scoped(&vector, &cfg, &hint_refs)
            }
            .ok()?;

            if !fault.faulted() {
                return None;
            }
            if let Ok(mut h) = heat.lock() {
                h.record_query(fault.hits.iter().map(|x| x.chunk_id));
            }
            Some((
                fault.render_context(),
                fault.hits.len(),
                fault.injected_tokens(),
                fault.best_score,
            ))
        });

        match tokio::time::timeout(self.budget, work).await {
            Ok(Ok(Some((context, hits, tokens, best)))) => {
                println!(
                    "[aim-proxy] retrieved {hits} chunks (~{tokens} tokens, best {best:.3}) \
                     in {:.1} ms",
                    started.elapsed().as_secs_f64() * 1000.0
                );
                Some(context)
            }
            Ok(Ok(None)) => None,
            Ok(Err(e)) => {
                eprintln!("[aim-proxy] retrieval task failed: {e}");
                None
            }
            Err(_) => {
                // Budget exceeded. The task keeps running to completion
                // on the blocking pool — aborting mid-search would not
                // return the time — but this request goes upstream
                // without context rather than making the editor wait.
                eprintln!(
                    "[aim-proxy] retrieval exceeded its {} ms budget; forwarding un-augmented",
                    self.budget.as_millis()
                );
                None
            }
        }
    }

    /// Pin the hottest chunks into physical RAM.
    ///
    /// Colibri's routing-heat idea: the chunks this session keeps
    /// retrieving are the ones that must never take a disk fault.
    pub fn pin_hot(&self, budget_bytes: usize) {
        let Some(catalog) = &self.catalog else { return };
        let hottest: Vec<u64> = match self.heat.lock() {
            Ok(h) => h.hottest(512).into_iter().map(|(id, _)| id).collect(),
            Err(_) => return,
        };
        if hottest.is_empty() {
            return;
        }
        let Ok(guard) = catalog.read() else { return };
        let report = guard.pin_chunks(&hottest, budget_bytes);
        println!(
            "[aim-proxy] pinned {}/{} hot chunks ({} KiB resident)",
            report.pinned,
            report.requested,
            report.pinned_bytes / 1024
        );
        for failure in report.failures.iter().take(1) {
            eprintln!("[aim-proxy] pinning stopped: {failure}");
        }
    }
}

/// Pull the text to retrieve against out of a request body.
///
/// Only the last user turn is used. Earlier turns already had their
/// context injected, and including them makes the query vector drift
/// toward whatever the conversation was about ten messages ago.
pub fn extract_query(payload: &Value) -> Option<String> {
    // Ollama /api/generate
    if let Some(p) = payload.get("prompt").and_then(|p| p.as_str()) {
        return Some(p.to_string());
    }

    let messages = payload.get("messages")?.as_array()?;
    for message in messages.iter().rev() {
        if message.get("role").and_then(|r| r.as_str()) != Some("user") {
            continue;
        }
        let content = message.get("content")?;

        // Plain string content.
        if let Some(s) = content.as_str() {
            return Some(s.to_string());
        }
        // Anthropic / OpenAI structured content blocks.
        if let Some(blocks) = content.as_array() {
            let text: Vec<&str> = blocks
                .iter()
                .filter_map(|b| b.get("text").and_then(|t| t.as_str()))
                .collect();
            if !text.is_empty() {
                return Some(text.join("\n"));
            }
        }
    }
    None
}

/// Extract file-path-looking tokens from a query, for scoped search.
///
/// A request naming a file is the strongest signal available, and
/// scoping to it is both more accurate and roughly 13x faster than a
/// corpus-wide scan, because turbovec skips whole 32-vector blocks that
/// contain no allowed slot.
pub fn path_hints(query: &str) -> Vec<String> {
    let mut hints = Vec::new();
    for raw in query.split(|c: char| c.is_whitespace() || "()[]{}<>,;:\"'`".contains(c)) {
        let token = raw.trim_matches(|c: char| c == '.' || c == '/');
        if token.len() < 3 || token.len() > 200 {
            continue;
        }

        let has_dir = token.contains('/') || token.contains('\\');
        let has_known_ext = token
            .rsplit_once('.')
            .map(|(_, ext)| {
                !ext.is_empty()
                    && libaim::chunk::DEFAULT_EXTENSIONS.contains(&ext.to_ascii_lowercase().as_str())
            })
            .unwrap_or(false);

        if has_known_ext || has_dir {
            hints.push(token.replace('\\', "/"));
        }
    }
    hints.sort();
    hints.dedup();
    hints
}

/// Whether the filesystem watcher should run.
fn watching_enabled() -> bool {
    match std::env::var(ENV_WATCH) {
        Ok(v) => !matches!(v.trim().to_ascii_lowercase().as_str(), "0" | "false" | "no" | "off"),
        Err(_) => true,
    }
}

/// Read and parse an environment variable, ignoring unparseable values.
fn env_parse<T: std::str::FromStr>(name: &str) -> Option<T> {
    std::env::var(name).ok()?.parse().ok()
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn extracts_prompt_from_ollama_generate() {
        let p = json!({ "model": "x", "prompt": "explain the mailbox" });
        assert_eq!(extract_query(&p).unwrap(), "explain the mailbox");
    }

    #[test]
    fn extracts_last_user_message_not_the_first() {
        let p = json!({ "messages": [
            { "role": "user", "content": "first question" },
            { "role": "assistant", "content": "an answer" },
            { "role": "user", "content": "second question" }
        ]});
        assert_eq!(extract_query(&p).unwrap(), "second question");
    }

    #[test]
    fn ignores_system_and_assistant_turns() {
        let p = json!({ "messages": [
            { "role": "user", "content": "the real question" },
            { "role": "assistant", "content": "chatter" },
            { "role": "system", "content": "instructions" }
        ]});
        assert_eq!(extract_query(&p).unwrap(), "the real question");
    }

    #[test]
    fn extracts_structured_content_blocks() {
        let p = json!({ "messages": [
            { "role": "user", "content": [
                { "type": "text", "text": "line one" },
                { "type": "image", "source": {} },
                { "type": "text", "text": "line two" }
            ]}
        ]});
        assert_eq!(extract_query(&p).unwrap(), "line one\nline two");
    }

    #[test]
    fn returns_none_when_there_is_no_user_turn() {
        assert!(extract_query(&json!({ "messages": [] })).is_none());
        assert!(extract_query(&json!({ "model": "x" })).is_none());
        assert!(extract_query(&json!({ "messages": [
            { "role": "assistant", "content": "hi" }
        ]}))
        .is_none());
    }

    #[test]
    fn finds_filenames_in_a_query() {
        let hints = path_hints("fix the IRQ starvation in apple_mbox.c please");
        assert_eq!(hints, vec!["apple_mbox.c"]);
    }

    #[test]
    fn finds_directory_style_paths() {
        let hints = path_hints("look at src/hw/misc/apple_mbox.c and hw/char/pl011.c");
        assert!(hints.contains(&"src/hw/misc/apple_mbox.c".to_string()));
        assert!(hints.contains(&"hw/char/pl011.c".to_string()));
    }

    #[test]
    fn normalizes_windows_separators_in_hints() {
        let hints = path_hints(r"open src\hw\mbox.c");
        assert_eq!(hints, vec!["src/hw/mbox.c"]);
    }

    #[test]
    fn ignores_prose_and_unknown_extensions() {
        // "e.g." and version numbers must not be mistaken for paths, or
        // every query gets scoped to nothing and falls back anyway.
        let hints = path_hints("e.g. version 1.2 of the thing costs 3.50 dollars");
        assert!(hints.is_empty(), "got {hints:?}");
    }

    #[test]
    fn strips_surrounding_punctuation() {
        let hints = path_hints("see (apple_mbox.c), and \"pl011.c\";");
        assert!(hints.contains(&"apple_mbox.c".to_string()), "got {hints:?}");
        assert!(hints.contains(&"pl011.c".to_string()), "got {hints:?}");
    }

    #[test]
    fn hints_are_deduplicated() {
        let hints = path_hints("apple_mbox.c calls apple_mbox.c twice");
        assert_eq!(hints, vec!["apple_mbox.c"]);
    }
}
