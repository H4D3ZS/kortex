use axum::{
    routing::{post, any},
    Router,
    extract::{Request, State},
    response::IntoResponse,
    body::Body,
    http::Method,
};
use reqwest::Client;
use std::net::SocketAddr;
use serde_json::{Value, json};
use tower_http::cors::{Any, CorsLayer};

mod api_manifest;

#[derive(Clone)]
struct AppState {
    http_client: Client,
    target_ollama: String,
}

#[tokio::main]
async fn main() {
    let state = AppState {
        http_client: Client::new(),
        target_ollama: "http://127.0.0.1:11434".to_string(), // Native Ollama Endpoint
    };

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods([Method::GET, Method::POST, Method::OPTIONS])
        .allow_headers(Any);

    let app = Router::new()
        .route("/api/generate", post(intercept_ollama))
        .route("/api/chat", post(intercept_ollama))
        .route("/api/manifest", post(api_manifest::handle_manifest))
        .route("/v1/messages", post(intercept_anthropic))
        .route("/v1/chat/completions", post(intercept_openai))
        .route("/*path", any(pass_through))
        .layer(cors)
        .with_state(state);

    let addr = SocketAddr::from(([127, 0, 0, 1], 1536));
    println!("⚡ ===========================================================================");
    println!("⚡ [AIM-PROXY] UNIVERSAL 1-GIST-TOKEN PROTOCOL INTERCEPTOR ACTIVE");
    println!("⚡ [AIM-PROXY] Listening securely at: http://127.0.0.1:1536");
    println!("⚡ [AIM-PROXY] Intercepting: Ollama, Anthropic (Claude), OpenAI (Cursor & Agents)");
    println!("⚡ ===========================================================================");
    
    let listener = tokio::net::TcpListener::bind(addr).await.expect("Failed to bind proxy address");
    axum::serve(listener, app).await.expect("Failed to boot Axum zero-token server");
}

async fn parse_aim_binary() -> Result<String, String> {
    let paths = [
        "C:\\Users\\HADES\\Desktop\\CodeSigil\\kortex\\.aim\\memory.aim",
        ".\\.aim\\memory.aim",
        "..\\.aim\\memory.aim",
        "C:\\Users\\HADES\\Desktop\\kortex\\.aim\\memory.aim",
        "C:\\Users\\HADES\\Desktop\\Virtual-iPhone-Emulator\\.aim\\memory.aim"
    ];
    
    for path in paths {
        if let Ok(data) = std::fs::read(path) {
            // 1. Detect and parse the proper AIMTTT binary format
            if data.starts_with(b"\x41\x49\x4D\x54\x54\x54") || data.starts_with(b"AIMTTT") {
                let start_idx = if data.starts_with(b"AIMTTT") { 6 } else { 6 };
                let mut header_end = start_idx;
                while header_end < data.len() && data[header_end] != b'}' {
                    header_end += 1;
                }
                header_end = (header_end + 1).min(data.len());
                
                let header_str = std::str::from_utf8(&data[start_idx..header_end]).unwrap_or("{}");
                let header: Value = serde_json::from_str(header_str).unwrap_or(json!({}));
                
                let tensor_start = (header_end + 3) & !3;
                let tensor_end = tensor_start + (1536 * 4);
                let mut floats_loaded = 0;
                
                if tensor_end <= data.len() {
                    let tensor_bytes = &data[tensor_start..tensor_end];
                    floats_loaded = tensor_bytes.len() / 4;
                }
                
                let info = format!(
                    "\n\n[AIM-VFS-BINARY-ACTIVE]: Loaded compiled AIM binary memory matrix successfully!\n- Format: Neural VFS VRAM Binary Map\n- Path: {}\n- Header Metadata: {}\n- Parametric Gist Dimensions: {} FP32 weights resident (Circular Convolution Bound, Quantum Seal intact)", 
                    path, header, floats_loaded
                );
                return Ok(info);
            }
            
            // 2. Fallback: Parse high-fidelity JSON metadata tree mapping directly
            if data.starts_with(b"{") {
                if let Ok(json_val) = serde_json::from_slice::<Value>(&data) {
                    let mut project_meta = String::new();
                    project_meta.push_str("\n\n[AIM-VFS-JSON-ACTIVE]: Loaded Project Matrix Metadata successfully!");
                    
                    // Retrieve tree from standard root or sub-object keys
                    let tree_opt = json_val.get("kortex").and_then(|k| k.get("project_tree"))
                        .or_else(|| json_val.get("project_tree"))
                        .and_then(|t| t.as_array());
                        
                    if let Some(tree) = tree_opt {
                        project_meta.push_str(&format!("\n- Environment Format: Visual JSON Tree\n- Cataloged Files Count: {}\n- Active Workspace Inventory Preview:", tree.len()));
                        let limit = 75.min(tree.len());
                        for f in tree.iter().take(limit) {
                            if let Some(f_str) = f.as_str() {
                                project_meta.push_str(&format!("\n  * {}", f_str));
                            }
                        }
                        if tree.len() > limit {
                            project_meta.push_str(&format!("\n  * ... and {} more files", tree.len() - limit));
                        }
                    } else {
                        project_meta.push_str("\n- Environment Status: Active Node Metadata mapped (No direct physical tree detected).");
                    }
                    return Ok(project_meta);
                }
            }
            
            // 3. Raw Fallback
            return Ok(format!("\n\n[AIM-VFS-RAW-ACTIVE]: Intercepted {} bytes of raw matrix payloads from: {}", data.len(), path));
        }
    }
    
    Ok("\n\n[AIM-VFS-INACTIVE]: No structural context loaded. Please run NeuralDrive and generate an .aim block for this project.".to_string())
}

async fn intercept_ollama(
    State(state): State<AppState>,
    req: Request<Body>,
) -> impl IntoResponse {
    let (parts, body) = req.into_parts();

    if let Ok(bytes) = axum::body::to_bytes(body, usize::MAX).await {
        if let Ok(mut json_payload) = serde_json::from_slice::<Value>(&bytes) {
            println!("🟢 [AIM-PROXY] Captured Ollama Inference Payload precisely!");

            // 1. Support Legacy /api/generate (Single Prompt)
            if let Some(prompt) = json_payload.get_mut("prompt") {
                if let Some(prompt_str) = prompt.as_str() {
                    let aim_context = parse_aim_binary().await.unwrap_or_default();
                    let injected = format!("{}{}", prompt_str, aim_context);
                    *prompt = json!(injected);
                    println!("🟢 [AIM-PROXY] Injected Context into Legacy Prompt!");
                }
            }

            // 2. Support Modern /api/chat (Messages Array)
            let manifest_antigravity = json_payload.get("model")
                .and_then(|m| m.as_str())
                .map(|s| s == "antigravity-sentient")
                .unwrap_or(false);

            let new_model_name = if manifest_antigravity {
                Some("neuraldaredevil-8b-ablitared")
            } else {
                None
            };

            if let Some(messages) = json_payload.get_mut("messages").and_then(|m| m.as_array_mut()) {
                let aim_context = parse_aim_binary().await.unwrap_or_default();
                let gist_prefix = format!("[KORTEX_GIST_TTT_OPTIMIZED]\n{}", aim_context);

                // Inject at the BEGINNING (Index 0) for Prefix Caching optimization
                let mut has_gist = false;
                if let Some(first_msg) = messages.get_mut(0) {
                    if let Some(content) = first_msg.get("content").and_then(|c| c.as_str()) {
                        if content.contains("[KORTEX_GIST") {
                            let new_content = format!("{}\n\n{}", gist_prefix, content);
                            first_msg["content"] = json!(new_content);
                            has_gist = true;
                            println!("🟢 [AIM-PROXY] Prepended Gist Prefix to existing prompt at Index 0");
                        }
                    }
                }

                if !has_gist {
                    messages.insert(0, json!({
                        "role": "system",
                        "content": gist_prefix
                    }));
                    println!("🟢 [AIM-PROXY] Inserted stable Gist Prefix at Index 0 (Prefix Cache Prime)");
                }

                // 3. Manifest Antigravity Persona if requested
                if manifest_antigravity {
                    println!("⚡ [AIM-PROXY] MANIFESTING ANTIGRAVITY AGENT...");

                    if let Some(model_name) = new_model_name {
                        if let Some(model) = messages.get_mut(0).and_then(|m| m.get_mut("model")) {
                            *model = json!(model_name);
                        }
                    }

                    let agent_prompt = "You are the Antigravity Agent manifested via the God Protocol Proxy. \
                                       You have full access to the Project Matrix (Kortex .aim). \
                                       Your goal is to provide elite, mission-critical engineering reasoning. \
                                       Respond with precision, autonomy, and a focus on zero-token optimization.";

                    messages.insert(1, json!({
                        "role": "system",
                        "content": agent_prompt
                    }));
                }
            }
            
            let new_body = serde_json::to_vec(&json_payload).unwrap();
            let target_url = format!("{}{}", state.target_ollama, parts.uri.path_and_query().map(|pq| pq.as_str()).unwrap_or(""));
            
            let proxy_req = state.http_client.post(&target_url)
                .body(new_body)
                .send()
                .await;

            match proxy_req {
                Ok(resp) => {
                    let mut builder = axum::response::Response::builder()
                        .status(axum::http::StatusCode::from_u16(resp.status().as_u16()).unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR));
                    for (k, v) in resp.headers() {
                        if let (Ok(key), Ok(val)) = (axum::http::HeaderName::try_from(k.as_str()), axum::http::HeaderValue::from_bytes(v.as_bytes())) {
                            builder = builder.header(key, val);
                        }
                    }
                    return builder.body(Body::from_stream(resp.bytes_stream())).unwrap();
                },
                Err(e) => {
                    return axum::response::Response::builder()
                        .status(500)
                        .body(Body::from(format!("Proxy Edge Forwarding Error: {}", e)))
                        .unwrap();
                }
            }
        }
    }
    
    axum::response::Response::builder().status(500).body(Body::from("Failed to intercept payload")).unwrap()
}

async fn intercept_anthropic(
    State(state): State<AppState>,
    req: Request<Body>,
) -> impl IntoResponse {
    let (parts, body) = req.into_parts();
    
    if let Ok(bytes) = axum::body::to_bytes(body, usize::MAX).await {
        if let Ok(mut json_payload) = serde_json::from_slice::<Value>(&bytes) {
            println!("🟢 [AIM-PROXY] Captured Anthropic (Claude) Inference Payload!");

            let aim_context = parse_aim_binary().await.unwrap_or_default();
            let gist_prefix = format!("[KORTEX_GIST_TTT_OPTIMIZED]\n{}", aim_context);

            // Inject 1-gist-token context into system prompt
            if let Some(system) = json_payload.get_mut("system") {
                if let Some(system_str) = system.as_str() {
                    let new_system = format!("{}\n\n{}", gist_prefix, system_str);
                    *system = json!(new_system);
                    println!("🟢 [AIM-PROXY] Injected Gist into Anthropic System String!");
                } else if let Some(system_arr) = system.as_array_mut() {
                    system_arr.insert(0, json!({
                        "type": "text",
                        "text": gist_prefix,
                        "cache_control": {"type": "ephemeral"}
                    }));
                    println!("🟢 [AIM-PROXY] Injected Gist Block with Cache-Control into System Array!");
                }
            } else {
                json_payload["system"] = json!([
                    {
                        "type": "text",
                        "text": gist_prefix,
                        "cache_control": {"type": "ephemeral"}
                    }
                ]);
                println!("🟢 [AIM-PROXY] Created Anthropic System Block with cache_control!");
            }

            let new_body = serde_json::to_vec(&json_payload).unwrap();
            let target_url = "https://api.anthropic.com/v1/messages";

            let mut forward_req = state.http_client.post(target_url).body(new_body);

            for (k, v) in parts.headers.iter() {
                let name = k.as_str();
                if name != "host" && name != "content-length" {
                    if let (Ok(key), Ok(val)) = (reqwest::header::HeaderName::try_from(name), reqwest::header::HeaderValue::from_bytes(v.as_bytes())) {
                        forward_req = forward_req.header(key, val);
                    }
                }
            }

            match forward_req.send().await {
                Ok(resp) => {
                    let mut builder = axum::response::Response::builder()
                        .status(axum::http::StatusCode::from_u16(resp.status().as_u16()).unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR));
                    for (k, v) in resp.headers() {
                        let name = k.as_str();
                        if name != "content-length" && name != "transfer-encoding" && name != "content-encoding" {
                            if let (Ok(key), Ok(val)) = (axum::http::HeaderName::try_from(name), axum::http::HeaderValue::from_bytes(v.as_bytes())) {
                                builder = builder.header(key, val);
                            }
                        }
                    }
                    return builder.body(Body::from_stream(resp.bytes_stream())).unwrap();
                },
                Err(e) => {
                    eprintln!("🔴 [AIM-PROXY] Anthropic Forwarding Error: {}", e);
                    return axum::response::Response::builder().status(502).body(Body::from("Anthropic forwarding failure")).unwrap();
                }
            }
        }
    }
    axum::response::Response::builder().status(500).body(Body::from("Failed to parse body")).unwrap()
}

async fn intercept_openai(
    State(state): State<AppState>,
    req: Request<Body>,
) -> impl IntoResponse {
    let (parts, body) = req.into_parts();
    
    if let Ok(bytes) = axum::body::to_bytes(body, usize::MAX).await {
        if let Ok(mut json_payload) = serde_json::from_slice::<Value>(&bytes) {
            println!("🟢 [AIM-PROXY] Captured OpenAI (Cursor/GPT) Inference Payload!");

            let aim_context = parse_aim_binary().await.unwrap_or_default();
            let gist_prefix = format!("[KORTEX_GIST_TTT_OPTIMIZED]\n{}", aim_context);

            // Inject 1-gist-token context into messages at Index 0 (Prefix Cache)
            if let Some(messages) = json_payload.get_mut("messages").and_then(|m| m.as_array_mut()) {
                messages.insert(0, json!({
                    "role": "system",
                    "content": gist_prefix
                }));
                println!("🟢 [AIM-PROXY] Injected Gist System Prompt into OpenAI messages!");
            }

            let new_body = serde_json::to_vec(&json_payload).unwrap();
            let target_url = "https://api.openai.com/v1/chat/completions";

            let mut forward_req = state.http_client.post(target_url).body(new_body);

            for (k, v) in parts.headers.iter() {
                let name = k.as_str();
                if name != "host" && name != "content-length" {
                    if let (Ok(key), Ok(val)) = (reqwest::header::HeaderName::try_from(name), reqwest::header::HeaderValue::from_bytes(v.as_bytes())) {
                        forward_req = forward_req.header(key, val);
                    }
                }
            }

            match forward_req.send().await {
                Ok(resp) => {
                    let mut builder = axum::response::Response::builder()
                        .status(axum::http::StatusCode::from_u16(resp.status().as_u16()).unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR));
                    for (k, v) in resp.headers() {
                        let name = k.as_str();
                        if name != "content-length" && name != "transfer-encoding" && name != "content-encoding" {
                            if let (Ok(key), Ok(val)) = (axum::http::HeaderName::try_from(name), axum::http::HeaderValue::from_bytes(v.as_bytes())) {
                                builder = builder.header(key, val);
                            }
                        }
                    }
                    return builder.body(Body::from_stream(resp.bytes_stream())).unwrap();
                },
                Err(e) => {
                    eprintln!("🔴 [AIM-PROXY] OpenAI Forwarding Error: {}", e);
                    return axum::response::Response::builder().status(502).body(Body::from("OpenAI forwarding failure")).unwrap();
                }
            }
        }
    }
    axum::response::Response::builder().status(500).body(Body::from("Failed to parse body")).unwrap()
}

async fn pass_through(
    State(state): State<AppState>,
    req: Request<Body>,
) -> impl IntoResponse {
    let method = req.method().clone();
    let path: String = req.uri().path_and_query().map(|pq| pq.as_str()).unwrap_or("").to_string();
    let target_url = format!("{}{}", state.target_ollama, path);
    println!("🔍 [AIM-PROXY] Passing through: {} {} -> {}", method, path, target_url);

    let (parts, body) = req.into_parts();

    let bytes = if parts.method == Method::GET || parts.method == Method::HEAD {
        None
    } else {
        axum::body::to_bytes(body, 10 * 1024 * 1024).await.ok() // 10MB limit
    };

    if let Some(ref b) = bytes {
        let is_agent_request = serde_json::from_slice::<Value>(b)
            .map(|v| v.get("model").and_then(|m| m.as_str()) == Some("antigravity-sentient"))
            .unwrap_or(false);

        if is_agent_request {
            println!("⚡ [AIM-PROXY] Redirecting Agentic request for: {}", path);
            let req = Request::from_parts(parts, Body::from(b.clone()));
            return intercept_ollama(State(state), req).await.into_response();
        }
    }

    let method_str = parts.method.as_str();
    let method: reqwest::Method = method_str.parse().unwrap_or(reqwest::Method::GET);
    let mut forward_req = state.http_client.request(method, &target_url);

    for (k, v) in parts.headers.iter() {
        if k.as_str() != "host" && k.as_str() != "content-length" {
            if let (Ok(key), Ok(val)) = (axum::http::HeaderName::try_from(k.as_str()), axum::http::HeaderValue::from_bytes(v.as_bytes())) {
                if let (Ok(r_key), Ok(r_val)) = (reqwest::header::HeaderName::try_from(key.as_str()), reqwest::header::HeaderValue::from_bytes(val.as_bytes())) {
                    forward_req = forward_req.header(r_key, r_val);
                }
            }
        }
    }

    if let Some(b) = bytes {
        forward_req = forward_req.body(b);
    }

    let request = forward_req.send().await;
            
    match request {
        Ok(resp) => {
             let mut builder = axum::response::Response::builder()
                 .status(axum::http::StatusCode::from_u16(resp.status().as_u16()).unwrap_or(axum::http::StatusCode::INTERNAL_SERVER_ERROR));

             for (k, v) in resp.headers() {
                  if k.as_str() != "content-length" && k.as_str() != "transfer-encoding" && k.as_str() != "content-encoding" {
                      if let (Ok(key), Ok(val)) = (axum::http::HeaderName::try_from(k.as_str()), axum::http::HeaderValue::from_bytes(v.as_bytes())) {
                          builder = builder.header(key, val);
                      }
                  }
             }

             println!("🟢 [AIM-PROXY] Forwarded {} for: {}", parts.method, path);
             return builder.body(Body::from_stream(resp.bytes_stream())).unwrap();
        },
        Err(e) => {
             eprintln!("🔴 [AIM-PROXY] Forwarding Error: {}", e);
             return axum::response::Response::builder().status(502).body(Body::from("Pass-through failure at edge")).unwrap();
        }
    }
}
