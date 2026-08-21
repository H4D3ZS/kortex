//! `aim-index` — build and inspect `.aim` catalogs.
//!
//! ```text
//! aim-index build <workspace> [--out DIR] [--bits 2|3|4] [--dim N]
//!                             [--model NAME] [--server URL] [--backend NAME]
//! aim-index query <catalog>   <query...> [--relative F] [--k N] [--scope SUBSTR]
//! aim-index stats <catalog>
//! ```
//!
//! `query` exists to check the gate against a real corpus. Retrieval is
//! gated on a fraction of the best score (`--relative`) rather than a
//! fixed cosine, because absolute score magnitude tracks query length
//! rather than relevance.

use std::process::ExitCode;

use libaim::{
    Catalog, Embedder, HashEmbedder, IndexOptions, RetrievalConfig, DEFAULT_BIT_WIDTH, DEFAULT_DIM,
};

const USAGE: &str = "\
aim-index — build and inspect .aim catalogs

USAGE:
    aim-index build <workspace> [OPTIONS]
    aim-index query <catalog> <query text...> [OPTIONS]
    aim-index stats <catalog>

BUILD OPTIONS:
    --out DIR          Catalog output directory     [default: <workspace>/.aim]
    --dim N            Embedding dimension          [default: 1536]
    --bits N           Quantizer bit width (2-4)    [default: 4]
    --max-file-bytes N Skip files larger than this  [default: 2097152]
    --ignore NAME      Skip a directory name (repeatable), on top of
                       the .git/target/node_modules/... defaults

  Dense embeddings (instead of the built-in hash embedder):
    --model NAME       Embedding model to request from the server
    --server URL       Server base URL. Defaults to Lemonade on :13305,
                       or :11434 when --backend ollama is given.
    --backend NAME     lemonade | openai | ollama | auto   [default: auto]

  `auto` probes Lemonade, then OpenAI-compatible, then Ollama, and uses
  whichever answers first.

QUERY OPTIONS:
    --threshold F      Absolute cosine floor        [default: 0.05]
    --relative F       Fraction of the best score a hit must reach
                       (the load-bearing gate)      [default: 0.6]
    --k N              Max chunks to inject         [default: 8]
    --scope SUBSTR     Restrict to paths containing SUBSTR (repeatable)
    --show-text        Print the inflated chunk text
";

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() || args[0] == "-h" || args[0] == "--help" {
        print!("{USAGE}");
        return ExitCode::SUCCESS;
    }

    let result = match args[0].as_str() {
        "build" => cmd_build(&args[1..]),
        "query" => cmd_query(&args[1..]),
        "stats" => cmd_stats(&args[1..]),
        other => Err(format!("unknown subcommand `{other}`\n\n{USAGE}")),
    };

    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(msg) => {
            eprintln!("error: {msg}");
            ExitCode::FAILURE
        }
    }
}

/// Minimal flag parser. Returns the positional arguments and leaves
/// flags in a lookup map.
struct Args {
    positional: Vec<String>,
    flags: Vec<(String, String)>,
}

impl Args {
    fn parse(argv: &[String], valueless: &[&str]) -> Result<Self, String> {
        let mut positional = Vec::new();
        let mut flags = Vec::new();
        let mut i = 0;
        while i < argv.len() {
            let a = &argv[i];
            if let Some(name) = a.strip_prefix("--") {
                if valueless.contains(&name) {
                    flags.push((name.to_string(), "true".to_string()));
                    i += 1;
                } else {
                    let value = argv
                        .get(i + 1)
                        .ok_or_else(|| format!("flag --{name} needs a value"))?;
                    flags.push((name.to_string(), value.clone()));
                    i += 2;
                }
            } else {
                positional.push(a.clone());
                i += 1;
            }
        }
        Ok(Args { positional, flags })
    }

    fn get(&self, name: &str) -> Option<&str> {
        self.flags
            .iter()
            .find(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
    }

    fn all(&self, name: &str) -> Vec<&str> {
        self.flags
            .iter()
            .filter(|(k, _)| k == name)
            .map(|(_, v)| v.as_str())
            .collect()
    }

    fn has(&self, name: &str) -> bool {
        self.get(name).is_some()
    }

    fn parse_or<T: std::str::FromStr>(&self, name: &str, default: T) -> Result<T, String> {
        match self.get(name) {
            None => Ok(default),
            Some(v) => v
                .parse()
                .map_err(|_| format!("could not parse --{name} value `{v}`")),
        }
    }
}

fn cmd_build(argv: &[String]) -> Result<(), String> {
    let args = Args::parse(argv, &[])?;
    let root = args
        .positional
        .first()
        .ok_or("build needs a workspace path")?;

    let dim: usize = args.parse_or("dim", DEFAULT_DIM)?;
    let bits: usize = args.parse_or("bits", DEFAULT_BIT_WIDTH)?;
    if !(2..=4).contains(&bits) {
        return Err(format!("--bits must be 2, 3 or 4 (got {bits})"));
    }

    let mut opts = IndexOptions {
        bit_width: bits,
        ..Default::default()
    };
    opts.out_dir = args.get("out").map(Into::into);
    opts.max_file_bytes = args.parse_or("max-file-bytes", opts.max_file_bytes)?;
    // Repeatable --ignore NAME: skip project-specific noise dirs (e.g. a
    // vendored submodule) on top of DEFAULT_IGNORED_DIRS. Without this a
    // large vendored tree dominates the catalog and drowns real results.
    opts.extra_ignored_dirs = args
        .all("ignore")
        .into_iter()
        .map(str::to_string)
        .collect();

    // Chosen here rather than inside the library so the catalog records
    // exactly which embedder produced it.
    let backend = args.get("backend").unwrap_or("auto");
    // `--ollama MODEL` was the previous spelling; keep it working.
    let model = args.get("model").or_else(|| args.get("ollama"));
    let server = args.get("server").or_else(|| args.get("ollama-url"));

    let embedder: Box<dyn Embedder> = match model {
        None => Box::new(HashEmbedder::new(dim)),
        Some(model) => build_http_embedder(backend, server, model)?,
    };

    eprintln!(
        "indexing {root} with embedder `{}` ({} dims, {bits}-bit quantization)",
        embedder.id(),
        embedder.dim()
    );

    let stats = libaim::index_workspace(root, embedder.as_ref(), &opts, |files, chunks| {
        if files % 25 == 0 {
            eprint!("\r  {files} files, {chunks} chunks");
        }
    })
    .map_err(|e| e.to_string())?;
    eprintln!("\r  {} files, {} chunks", stats.files_indexed, stats.meta.chunk_count);

    let m = &stats.meta;
    println!("catalog written to {}", m.root);
    println!("  chunks            {}", m.chunk_count);
    println!("  files indexed     {}", stats.files_indexed);
    if stats.files_skipped_too_large > 0 {
        println!("  skipped (large)   {}", stats.files_skipped_too_large);
    }
    if stats.files_unreadable > 0 {
        println!("  skipped (unread)  {}", stats.files_unreadable);
    }
    println!("  chunk text        {}", human(m.source_bytes));
    println!(
        "  payload (zstd)    {}  ({:.1}% of chunk text)",
        human(m.payload_bytes),
        if m.source_bytes > 0 {
            m.payload_bytes as f64 / m.source_bytes as f64 * 100.0
        } else {
            0.0
        }
    );
    println!(
        "  container total   {}  ({:.1}% of chunk text)",
        human(m.container_bytes),
        stats.compression_ratio * 100.0
    );
    println!(
        "  index bytes       {}  ({} dims x {bits} bits x {} chunks)",
        human((m.chunk_count * m.dim * bits / 8) as u64),
        m.dim,
        m.chunk_count
    );
    println!("  elapsed           {:.2}s", stats.elapsed_secs);
    Ok(())
}

#[cfg(feature = "http-embed")]
fn build_http_embedder(
    backend: &str,
    server: Option<&str>,
    model: &str,
) -> Result<Box<dyn Embedder>, String> {
    use libaim::{EmbedProtocol, HttpEmbedder};

    let protocol = match backend {
        "auto" => None,
        "lemonade" => Some(EmbedProtocol::Lemonade),
        "openai" => Some(EmbedProtocol::OpenAi),
        "ollama" => Some(EmbedProtocol::Ollama),
        other => {
            return Err(format!(
                "unknown --backend `{other}` (expected lemonade, openai, ollama or auto)"
            ))
        }
    };

    // Without an explicit --server, default to the port the chosen
    // backend actually listens on.
    let url = server
        .unwrap_or_else(|| protocol.unwrap_or(EmbedProtocol::Lemonade).default_url())
        .to_string();

    let embedder = match protocol {
        Some(p) => HttpEmbedder::connect(p, &url, model),
        None => HttpEmbedder::autodetect(&url, model),
    }
    .map_err(|e| e.to_string())?;

    eprintln!(
        "using {} at {} with model `{model}`: {} dimensions",
        embedder.protocol().tag(),
        embedder.base_url(),
        embedder.dim()
    );
    Ok(Box::new(embedder))
}

#[cfg(not(feature = "http-embed"))]
fn build_http_embedder(
    _backend: &str,
    _server: Option<&str>,
    _model: &str,
) -> Result<Box<dyn Embedder>, String> {
    Err("this build has no dense-embedding support; rebuild with \
         `--features http-embed`"
        .to_string())
}

fn cmd_query(argv: &[String]) -> Result<(), String> {
    let args = Args::parse(argv, &["show-text", "no-hybrid"])?;
    let dir = args
        .positional
        .first()
        .ok_or("query needs a catalog directory")?;
    let query = args.positional[1..].join(" ");
    if query.trim().is_empty() {
        return Err("query text is empty".into());
    }

    let catalog = Catalog::open(dir).map_err(|e| e.to_string())?;
    // Matches whichever backend built the catalog (lexical hash OR dense
    // http); carries the catalog's IDF for hash, reconnects the server
    // for http. Using a plain HashEmbedder on a dense catalog would score
    // pure noise.
    let embedder = catalog.query_embedder_dyn().map_err(|e| e.to_string())?;

    let cfg = RetrievalConfig {
        fault_threshold: args.parse_or("threshold", 0.05f32)?,
        relative_floor: args.parse_or("relative", 0.6f32)?,
        max_chunks: args.parse_or("k", 8usize)?,
        ..Default::default()
    };

    let vector = embedder.embed(&query).map_err(|e| e.to_string())?;
    let scopes = args.all("scope");
    // Hybrid (dense semantic + lexical exact, fused by RRF) is the default
    // for dense catalogs; `--no-hybrid` forces pure semantic. Lexical adds
    // nothing to an already-lexical catalog, so hash catalogs stay pure.
    let hybrid = !args.has("no-hybrid")
        && scopes.is_empty()
        && !catalog.meta().embedder_id.starts_with("hash-");
    let started = std::time::Instant::now();
    let fault = if hybrid {
        catalog.hybrid_fault(&query, &vector, &cfg)
    } else if scopes.is_empty() {
        catalog.page_fault(&vector, &cfg)
    } else {
        catalog.page_fault_scoped(&vector, &cfg, &scopes)
    }
    .map_err(|e| e.to_string())?;
    let elapsed = started.elapsed();

    println!("query      {query:?}");
    println!("catalog    {} chunks, {} dims", catalog.len(), catalog.dim());
    if !scopes.is_empty() {
        println!("scoped to  {scopes:?}");
    }
    println!("retrieval  {:.3} ms", elapsed.as_secs_f64() * 1000.0);
    println!(
        "faulted    {}  (best {:.4}, cutoff {:.4}, gist {:.4})",
        fault.faulted(),
        fault.best_score,
        fault.cutoff,
        fault.gist_score
    );
    println!(
        "injected   {} chunks, ~{} tokens{}",
        fault.hits.len(),
        fault.injected_tokens(),
        if fault.dropped_to_budget > 0 {
            format!(" ({} dropped to budget)", fault.dropped_to_budget)
        } else {
            String::new()
        }
    );

    for (i, h) in fault.hits.iter().enumerate() {
        println!(
            "\n  {}. {:.4}  {}:{}-{}  (~{} tokens)",
            i + 1,
            h.score,
            h.path,
            h.line_start,
            h.line_end,
            h.token_estimate
        );
        if args.has("show-text") {
            for line in h.text.lines() {
                println!("     | {line}");
            }
        }
    }

    if !fault.faulted() {
        println!(
            "\nNothing cleared the threshold. Best score was {:.4}; try \
             --threshold {:.2} to see what is just below it.",
            fault.best_score,
            (fault.best_score - 0.05).max(0.0)
        );
    }
    Ok(())
}

fn cmd_stats(argv: &[String]) -> Result<(), String> {
    let args = Args::parse(argv, &[])?;
    let dir = args
        .positional
        .first()
        .ok_or("stats needs a catalog directory")?;
    let catalog = Catalog::open(dir).map_err(|e| e.to_string())?;
    catalog.verify_integrity().map_err(|e| e.to_string())?;

    let m = catalog.meta();
    println!("catalog          {dir}");
    println!("  integrity      verified (blake3 body hash matches header)");
    println!("  embedder       {}", m.embedder_id);
    println!("  root           {}", m.root);
    println!("  chunks         {}", m.chunk_count);
    println!("  files          {}", m.file_count);
    println!("  dim / bits     {} / {}", m.dim, m.bit_width);
    println!("  source bytes   {}", human(m.source_bytes));
    println!("  container      {}", human(m.container_bytes));

    let paths: Vec<&str> = catalog.paths().collect();
    println!("  distinct paths {}", paths.len());
    Ok(())
}

fn human(bytes: u64) -> String {
    const UNITS: [&str; 4] = ["B", "KiB", "MiB", "GiB"];
    let mut v = bytes as f64;
    let mut u = 0;
    while v >= 1024.0 && u < UNITS.len() - 1 {
        v /= 1024.0;
        u += 1;
    }
    if u == 0 {
        format!("{bytes} B")
    } else {
        format!("{v:.2} {}", UNITS[u])
    }
}
