//! The aim-proxy server: state, request handlers, and router construction.
//!
//! Extracted from `main.rs` so the IDE can host this router **in-process**
//! (see `vscodium-rust/src-tauri/src/kortex_retrieval/`) instead of only as a
//! standalone binary. Behaviour is unchanged: each intercepted request has the
//! last user turn pulled out, matching workspace chunks retrieved, and those
//! prepended as a leading system block; anything that finds nothing (or is too
//! slow, or has no catalog) is forwarded untouched.

use axum::{
    body::Body,
    extract::{Request, State},
    http::Method,
    response::{IntoResponse, Response},
    routing::{any, post},
    Router,
};
use reqwest::Client;
use serde_json::{json, Value};
use std::sync::Arc;
use tower_http::cors::{Any, CorsLayer};

use crate::api_manifest;
use crate::retrieval::{extract_query, RetrievalEngine};

/// Marker prefixing every injected block, so a re-proxied request can be
/// recognised and not augmented twice.
pub const GIST_MARKER: &str = "[KORTEX-AIM]";

#[derive(Clone)]
pub struct AppState {
    pub http_client: Client,
    pub ollama_url: String,
    pub openai_url: String,
    pub anthropic_url: String,
    pub retrieval: Arc<RetrievalEngine>,
}

impl AppState {
    /// Build state from the `KORTEX_UPSTREAM_*` env vars (Lemonade on :13305 is
    /// the OpenAI-dialect default; Ollama on :11434 the native default), loading
    /// the `.aim` catalog via [`RetrievalEngine::load`].
    pub fn from_env() -> Self {
        Self {
            http_client: Client::new(),
            ollama_url: env_or("KORTEX_UPSTREAM_OLLAMA", "http://127.0.0.1:11434"),
            openai_url: env_or("KORTEX_UPSTREAM_OPENAI", "http://localhost:13305"),
            anthropic_url: env_or("KORTEX_UPSTREAM_ANTHROPIC", "https://api.anthropic.com"),
            retrieval: Arc::new(RetrievalEngine::load()),
        }
    }
}

/// Which upstream and request dialect a handler is working in.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Dialect {
    Ollama,
    OpenAi,
    Anthropic,
}

pub fn env_or(name: &str, default: &str) -> String {
    std::env::var(name)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
        .trim_end_matches('/')
        .to_string()
}

/// Build the aim-proxy axum router around a ready [`AppState`]. Callers bind a
/// listener and `axum::serve` it (optionally with graceful shutdown).
pub fn build_router(state: AppState) -> Router {
    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers(Any);

    Router::new()
        .route("/api/generate", post(intercept_ollama))
        .route("/api/chat", post(intercept_ollama))
        .route("/api/manifest", post(api_manifest::handle_manifest))
        .route("/v1/messages", post(intercept_anthropic))
        .route("/v1/chat/completions", post(intercept_openai))
        .route("/*path", any(pass_through))
        .layer(cors)
        .with_state(state)
}

/// Retrieve context for a payload, or `None` if nothing should be added.
/// Uses catch_unwind to prevent corrupted catalog from injecting garbage.
async fn context_for_payload(state: &AppState, payload: &Value) -> Option<String> {
    let query = extract_query(payload)?;
    // A request that already carries an injected block came from another
    // proxy hop; augmenting again would duplicate the context and burn
    // the budget twice.
    if query.contains(GIST_MARKER) {
        return None;
    }
    // Catch panics from retrieval — if the catalog is corrupted or the
    // mmap index has wrong offsets, we pass through unchanged instead of
    // injecting garbage that breaks all models.
    let state_clone = state.clone();
    let query_clone = query.clone();
    match tokio::task::spawn_blocking(move || {
        // Run in a separate thread so panics don't crash the proxy.
        let rt = tokio::runtime::Handle::current();
        rt.block_on(state_clone.retrieval.context_for(&query_clone))
    }).await {
        Ok(result) => result,
        Err(e) => {
            eprintln!("[aim-proxy] retrieval panicked (passing through): {e}");
            None
        }
    }
}

async fn intercept_ollama(State(state): State<AppState>, req: Request<Body>) -> Response {
    augment_and_forward(state, req, Dialect::Ollama).await
}

async fn intercept_openai(State(state): State<AppState>, req: Request<Body>) -> Response {
    augment_and_forward(state, req, Dialect::OpenAi).await
}

async fn intercept_anthropic(State(state): State<AppState>, req: Request<Body>) -> Response {
    augment_and_forward(state, req, Dialect::Anthropic).await
}

/// Read the body, inject retrieved context if any, and forward.
async fn augment_and_forward(state: AppState, req: Request<Body>, dialect: Dialect) -> Response {
    let (parts, body) = req.into_parts();

    let Ok(bytes) = axum::body::to_bytes(body, 64 * 1024 * 1024).await else {
        return error_response(413, "request body too large");
    };

    // A body that is not JSON is not something to rewrite. Forward it
    // untouched rather than failing the request.
    let Ok(mut payload) = serde_json::from_slice::<Value>(&bytes) else {
        return forward(&state, dialect, &parts, bytes.to_vec()).await;
    };

    if let Some(context) = context_for_payload(&state, &payload).await {
        inject(&mut payload, dialect, &context);
    }

    let body = match serde_json::to_vec(&payload) {
        Ok(b) => b,
        // Reserializing what we just parsed cannot normally fail; if it
        // somehow does, send the original bytes rather than a 500.
        Err(_) => bytes.to_vec(),
    };
    forward(&state, dialect, &parts, body).await
}

/// Insert `context` into a payload as a leading system message.
///
/// Always at index 0. Prompt caches key on a prefix, so putting the
/// injected block first keeps the cached region stable across turns
/// instead of invalidating it on every request.
fn inject(payload: &mut Value, dialect: Dialect, context: &str) {
    match dialect {
        Dialect::Anthropic => {
            // Anthropic takes a top-level `system`, not a system message.
            let block = json!({
                "type": "text",
                "text": context,
                "cache_control": { "type": "ephemeral" }
            });
            match payload.get_mut("system") {
                Some(Value::String(existing)) => {
                    let merged = format!("{context}\n\n{existing}");
                    payload["system"] = json!([{ "type": "text", "text": merged }]);
                }
                Some(Value::Array(arr)) => arr.insert(0, block),
                _ => payload["system"] = json!([block]),
            }
        }
        Dialect::Ollama | Dialect::OpenAi => {
            if let Some(messages) = payload.get_mut("messages").and_then(|m| m.as_array_mut()) {
                messages.insert(0, json!({ "role": "system", "content": context }));
            } else if let Some(prompt) = payload.get("prompt").and_then(|p| p.as_str()) {
                // Ollama /api/generate has no message list.
                payload["prompt"] = json!(format!("{context}\n\n{prompt}"));
            }
        }
    }
}

/// Forward a request upstream and stream the response back.
async fn forward(
    state: &AppState,
    dialect: Dialect,
    parts: &axum::http::request::Parts,
    body: Vec<u8>,
) -> Response {
    let base = match dialect {
        Dialect::Ollama => &state.ollama_url,
        Dialect::OpenAi => &state.openai_url,
        Dialect::Anthropic => &state.anthropic_url,
    };
    let path = parts
        .uri
        .path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or("/");
    let url = format!("{base}{path}");

    let mut request = state.http_client.post(&url).body(body);
    for (name, value) in parts.headers.iter() {
        // `host` would point at the proxy, and `content-length` no
        // longer matches after injection.
        if matches!(name.as_str(), "host" | "content-length") {
            continue;
        }
        if let (Ok(k), Ok(v)) = (
            reqwest::header::HeaderName::try_from(name.as_str()),
            reqwest::header::HeaderValue::from_bytes(value.as_bytes()),
        ) {
            request = request.header(k, v);
        }
    }

    match request.send().await {
        Ok(resp) => relay(resp),
        Err(e) => {
            eprintln!("[aim-proxy] forwarding to {url} failed: {e}");
            error_response(502, &format!("upstream request failed: {e}"))
        }
    }
}

/// Turn a `reqwest` response into an axum streaming response.
fn relay(resp: reqwest::Response) -> Response {
    let status = axum::http::StatusCode::from_u16(resp.status().as_u16())
        .unwrap_or(axum::http::StatusCode::BAD_GATEWAY);
    let mut builder = Response::builder().status(status);

    for (name, value) in resp.headers() {
        // These describe the upstream's framing, which no longer applies
        // once the body is re-streamed.
        if matches!(
            name.as_str(),
            "content-length" | "transfer-encoding" | "content-encoding"
        ) {
            continue;
        }
        if let (Ok(k), Ok(v)) = (
            axum::http::HeaderName::try_from(name.as_str()),
            axum::http::HeaderValue::from_bytes(value.as_bytes()),
        ) {
            builder = builder.header(k, v);
        }
    }

    builder
        .body(Body::from_stream(resp.bytes_stream()))
        .unwrap_or_else(|e| error_response(500, &format!("could not relay response: {e}")))
}

fn error_response(status: u16, message: &str) -> Response {
    Response::builder()
        .status(status)
        .body(Body::from(message.to_string()))
        .expect("static error response is always valid")
}

/// Anything not explicitly intercepted goes to Ollama untouched —
/// `/api/tags`, `/api/show`, health checks, and so on.
async fn pass_through(State(state): State<AppState>, req: Request<Body>) -> Response {
    let (parts, body) = req.into_parts();
    let path = parts
        .uri
        .path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or("/");
    let url = format!("{}{}", state.ollama_url, path);

    let body_bytes = if parts.method == Method::GET || parts.method == Method::HEAD {
        None
    } else {
        axum::body::to_bytes(body, 64 * 1024 * 1024).await.ok()
    };

    let method: reqwest::Method = parts
        .method
        .as_str()
        .parse()
        .unwrap_or(reqwest::Method::GET);
    let mut request = state.http_client.request(method, &url);

    for (name, value) in parts.headers.iter() {
        if matches!(name.as_str(), "host" | "content-length") {
            continue;
        }
        if let (Ok(k), Ok(v)) = (
            reqwest::header::HeaderName::try_from(name.as_str()),
            reqwest::header::HeaderValue::from_bytes(value.as_bytes()),
        ) {
            request = request.header(k, v);
        }
    }
    if let Some(b) = body_bytes {
        request = request.body(b);
    }

    match request.send().await {
        Ok(resp) => relay(resp),
        Err(e) => {
            eprintln!("[aim-proxy] pass-through to {url} failed: {e}");
            error_response(502, &format!("upstream request failed: {e}"))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn injects_a_system_message_at_index_zero_for_openai() {
        let mut p = json!({ "messages": [{ "role": "user", "content": "hi" }] });
        inject(&mut p, Dialect::OpenAi, "CONTEXT");
        let messages = p["messages"].as_array().unwrap();
        assert_eq!(messages[0]["role"], "system");
        assert_eq!(messages[0]["content"], "CONTEXT");
        // The user's own turn must survive untouched.
        assert_eq!(messages[1]["content"], "hi");
    }

    #[test]
    fn injects_into_an_ollama_generate_prompt() {
        let mut p = json!({ "prompt": "explain this" });
        inject(&mut p, Dialect::Ollama, "CONTEXT");
        let prompt = p["prompt"].as_str().unwrap();
        assert!(prompt.starts_with("CONTEXT"));
        assert!(prompt.contains("explain this"));
    }
}
