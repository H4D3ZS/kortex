//! Kortex MCP server (Stage 3): expose retrieval as TOOLS the model pulls,
//! instead of blindly pushing top-k into the prompt. Speaks the Model
//! Context Protocol over stdio (newline-delimited JSON-RPC 2.0), so Claude
//! Code and any MCP client can call:
//!
//!   * `kortex_search`          — hybrid (dense + BM25 + structural) retrieval
//!   * `kortex_find_definition` — every chunk that DEFINES a symbol (exact)
//!
//! Load the catalog from `--catalog DIR` or `KORTEX_AIM_CATALOG` (default
//! `.aim`). A dense catalog also needs its embedding server up (Lemonade);
//! `KORTEX_EMBED_SERVER` overrides the URL.
//!
//! This is deliberately a thin, dependency-light server: the value is in
//! libaim's retrieval, not in the transport.

use std::io::{self, BufRead, Write};
use std::path::Path;

use libaim::{Catalog, Embedder, RetrievalConfig};
use serde_json::{json, Value};

const PROTOCOL_VERSION: &str = "2024-11-05";

fn main() {
    let mut dir = std::env::var("KORTEX_AIM_CATALOG").unwrap_or_else(|_| ".aim".to_string());
    let argv: Vec<String> = std::env::args().collect();
    if let Some(i) = argv.iter().position(|a| a == "--catalog") {
        if let Some(v) = argv.get(i + 1) {
            dir = v.clone();
        }
    }

    let catalog = match Catalog::open(Path::new(&dir)) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[aim-mcp] cannot open catalog `{dir}`: {e}");
            std::process::exit(1);
        }
    };
    let embedder = match catalog.query_embedder_dyn() {
        Ok(e) => e,
        Err(e) => {
            eprintln!("[aim-mcp] embedder unavailable: {e}");
            std::process::exit(1);
        }
    };
    let hybrid = !embedder.id().starts_with("hash-");
    eprintln!(
        "[aim-mcp] ready: {} chunks, embedder `{}`, hybrid {}",
        catalog.len(),
        embedder.id(),
        hybrid
    );

    let stdin = io::stdin();
    let mut out = io::stdout();
    for line in stdin.lock().lines() {
        let line = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        if line.trim().is_empty() {
            continue;
        }
        let req: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue, // malformed line: ignore, per stdio transport
        };
        let id = req.get("id").cloned();
        let method = req.get("method").and_then(Value::as_str).unwrap_or("");
        let params = req.get("params");

        let outcome = dispatch(method, params, &catalog, embedder.as_ref(), hybrid);

        // Notifications (no id) never get a response.
        let Some(id) = id else { continue };
        let msg = match outcome {
            Ok(result) => json!({"jsonrpc": "2.0", "id": id, "result": result}),
            Err((code, message)) => {
                json!({"jsonrpc": "2.0", "id": id, "error": {"code": code, "message": message}})
            }
        };
        if writeln!(out, "{msg}").is_err() {
            break;
        }
        let _ = out.flush();
    }
}

fn dispatch(
    method: &str,
    params: Option<&Value>,
    catalog: &Catalog,
    embedder: &dyn Embedder,
    hybrid: bool,
) -> Result<Value, (i64, String)> {
    match method {
        "initialize" => Ok(json!({
            "protocolVersion": PROTOCOL_VERSION,
            "capabilities": { "tools": {} },
            "serverInfo": { "name": "kortex", "version": env!("CARGO_PKG_VERSION") }
        })),
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({ "tools": tool_specs() })),
        "tools/call" => tools_call(params, catalog, embedder, hybrid),
        // Unknown request -> JSON-RPC "method not found".
        other => Err((-32601, format!("method not found: {other}"))),
    }
}

fn tool_specs() -> Value {
    json!([
        {
            "name": "kortex_search",
            "description": "Search this codebase's kortex index (hybrid: dense semantic + BM25 \
                            lexical + structural definition boost). Returns the most relevant \
                            chunks with file path and line range. Use for 'where/how is X done' \
                            and to pull context on demand instead of guessing.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": { "type": "string", "description": "natural-language or code query" },
                    "k": { "type": "integer", "description": "max chunks to return (default 6)" }
                },
                "required": ["query"]
            }
        },
        {
            "name": "kortex_find_definition",
            "description": "Find where a symbol is DEFINED (declaration or assignment), not merely \
                            used. Exact, no embedding. Use to jump to the definition of a function, \
                            struct, const, or variable.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "symbol": { "type": "string", "description": "identifier to locate" }
                },
                "required": ["symbol"]
            }
        }
    ])
}

fn tools_call(
    params: Option<&Value>,
    catalog: &Catalog,
    embedder: &dyn Embedder,
    hybrid: bool,
) -> Result<Value, (i64, String)> {
    let params = params.ok_or((-32602, "missing params".into()))?;
    let name = params
        .get("name")
        .and_then(Value::as_str)
        .ok_or((-32602, "missing tool name".into()))?;
    let args = params.get("arguments").cloned().unwrap_or(json!({}));

    let text = match name {
        "kortex_search" => {
            let query = args
                .get("query")
                .and_then(Value::as_str)
                .ok_or((-32602, "kortex_search needs `query`".into()))?;
            let k = args.get("k").and_then(Value::as_u64).unwrap_or(6) as usize;
            do_search(catalog, embedder, hybrid, query, k)
        }
        "kortex_find_definition" => {
            let symbol = args
                .get("symbol")
                .and_then(Value::as_str)
                .ok_or((-32602, "kortex_find_definition needs `symbol`".into()))?;
            do_find_definition(catalog, symbol)
        }
        other => return Err((-32602, format!("unknown tool: {other}"))),
    };

    Ok(json!({ "content": [ { "type": "text", "text": text } ] }))
}

fn do_search(
    catalog: &Catalog,
    embedder: &dyn Embedder,
    hybrid: bool,
    query: &str,
    k: usize,
) -> String {
    let cfg = RetrievalConfig { max_chunks: k.max(1), ..Default::default() };
    let vector = match embedder.embed(query) {
        Ok(v) => v,
        Err(e) => return format!("embedding failed: {e}"),
    };
    let fault = if hybrid {
        catalog.hybrid_fault(query, &vector, &cfg)
    } else {
        catalog.page_fault(&vector, &cfg)
    };
    match fault {
        Ok(f) if f.faulted() => render_hits(&f.hits),
        Ok(_) => "no relevant chunks found.".into(),
        Err(e) => format!("search failed: {e}"),
    }
}

fn do_find_definition(catalog: &Catalog, symbol: &str) -> String {
    match catalog.find_definitions(symbol) {
        Ok(hits) if !hits.is_empty() => {
            format!("Definitions of `{symbol}`:\n\n{}", render_hits(&hits))
        }
        Ok(_) => format!("no definition of `{symbol}` found in the index."),
        Err(e) => format!("lookup failed: {e}"),
    }
}

fn render_hits(hits: &[libaim::Hit]) -> String {
    let mut s = String::new();
    for h in hits {
        s.push_str(&format!(
            "── {}:{}-{}  (score {:.3})\n{}\n\n",
            h.path, h.line_start, h.line_end, h.score, h.text
        ));
    }
    s.trim_end().to_string()
}
