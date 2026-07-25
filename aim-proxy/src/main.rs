//! `aim-proxy` — an inference proxy that injects retrieved workspace
//! context into prompts on their way to a model.
//!
//! Listens on `127.0.0.1:1536` and speaks three request shapes: Ollama
//! (`/api/generate`, `/api/chat`), OpenAI (`/v1/chat/completions`), and
//! Anthropic (`/v1/messages`). Anything else is passed straight through.
//!
//! For each request it pulls out the last user turn, retrieves the
//! workspace chunks that match it, and prepends them as a system block.
//! When retrieval finds nothing — or takes too long, or no catalog is
//! loaded — the request is forwarded unchanged. Augmentation is an
//! optimization, never a dependency.
//!
//! The server itself lives in [`aim_proxy::server`] so the IDE can host the
//! same router in-process; this binary is a thin wrapper.
//!
//! # Upstreams
//!
//! Every upstream is an environment variable, because "local first"
//! should not mean "recompile to change the port":
//!
//! | variable                     | default                   |
//! |------------------------------|---------------------------|
//! | `KORTEX_UPSTREAM_OLLAMA`     | `http://127.0.0.1:11434`  |
//! | `KORTEX_UPSTREAM_OPENAI`     | `http://localhost:13305`  |
//! | `KORTEX_UPSTREAM_ANTHROPIC`  | `https://api.anthropic.com` |
//!
//! The OpenAI-compatible route defaults to **Lemonade Server** rather
//! than `api.openai.com`: this stack is built for local inference on AMD
//! hardware, and Lemonade serves the OpenAI schema on :13305.

use std::net::SocketAddr;

use aim_proxy::server::{build_router, AppState};

#[tokio::main]
async fn main() {
    let state = AppState::from_env();

    let addr = SocketAddr::from(([127, 0, 0, 1], 1536));
    println!("aim-proxy listening on http://{addr}");
    println!(
        "  retrieval   {}",
        if state.retrieval.is_active() {
            "active"
        } else {
            "inactive (pass-through only)"
        }
    );
    println!("  ollama    -> {}", state.ollama_url);
    println!("  openai    -> {}", state.openai_url);
    println!("  anthropic -> {}", state.anthropic_url);

    let app = build_router(state);
    let listener = tokio::net::TcpListener::bind(addr)
        .await
        .expect("failed to bind 127.0.0.1:1536");
    axum::serve(listener, app)
        .await
        .expect("proxy server terminated unexpectedly");
}
