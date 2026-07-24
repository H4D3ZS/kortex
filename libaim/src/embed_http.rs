//! Dense embeddings from a local inference server.
//!
//! Three wire protocols, one implementation, because the only real
//! differences are the URL path, the request key, and where the vector
//! sits in the response:
//!
//! | protocol   | path                   | request key | response         |
//! |------------|------------------------|-------------|------------------|
//! | Lemonade   | `/api/v1/embeddings`   | `input`     | `data[].embedding` |
//! | OpenAI     | `/v1/embeddings`       | `input`     | `data[].embedding` |
//! | Ollama     | `/api/embeddings`      | `prompt`    | `embedding`        |
//!
//! [`HttpEmbedder::autodetect`] tries them in that order, so an AMD box
//! running Lemonade Server is found first and an Ollama box still works
//! with no configuration.
//!
//! Everything here is blocking on purpose. The indexer is a synchronous
//! batch job, and the proxy calls this from inside `spawn_blocking` so a
//! slow model never occupies the async reactor.

use std::time::Duration;

use crate::embed::{normalize, Embedder};
use crate::error::AimError;

/// Default Lemonade Server address.
///
/// Lemonade also emulates Ollama's API on this same port, so a machine
/// running only Lemonade satisfies both the `Lemonade` and `Ollama`
/// protocols here.
pub const LEMONADE_DEFAULT_URL: &str = "http://localhost:13305";
/// Default Ollama address.
pub const OLLAMA_DEFAULT_URL: &str = "http://localhost:11434";

/// Which embeddings wire format a server speaks.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EmbedProtocol {
    /// AMD Lemonade Server: OpenAI-compatible, under `/api/v1`.
    Lemonade,
    /// Any other OpenAI-compatible server, under `/v1`.
    OpenAi,
    /// Ollama's native single-prompt embeddings endpoint.
    Ollama,
}

impl EmbedProtocol {
    /// Probe order. Lemonade first: on the AMD hardware this stack
    /// targets it is the one with GPU and NPU acceleration.
    pub const PROBE_ORDER: [EmbedProtocol; 3] =
        [EmbedProtocol::Lemonade, EmbedProtocol::OpenAi, EmbedProtocol::Ollama];

    /// Path appended to the base URL.
    pub fn path(&self) -> &'static str {
        match self {
            EmbedProtocol::Lemonade => "/api/v1/embeddings",
            EmbedProtocol::OpenAi => "/v1/embeddings",
            EmbedProtocol::Ollama => "/api/embeddings",
        }
    }

    /// Short name, used in [`Embedder::id`].
    pub fn tag(&self) -> &'static str {
        match self {
            EmbedProtocol::Lemonade => "lemonade",
            EmbedProtocol::OpenAi => "openai",
            EmbedProtocol::Ollama => "ollama",
        }
    }

    /// True when the server accepts an array of inputs in one request.
    pub fn supports_batching(&self) -> bool {
        // Ollama's `/api/embeddings` takes a single `prompt` string. Its
        // newer `/api/embed` does batch, but is not present on the older
        // builds this has to keep working against.
        !matches!(self, EmbedProtocol::Ollama)
    }

    /// Conventional default base URL for this protocol.
    pub fn default_url(&self) -> &'static str {
        match self {
            EmbedProtocol::Lemonade | EmbedProtocol::OpenAi => LEMONADE_DEFAULT_URL,
            EmbedProtocol::Ollama => OLLAMA_DEFAULT_URL,
        }
    }
}

/// Dense embeddings over HTTP.
pub struct HttpEmbedder {
    base_url: String,
    model: String,
    protocol: EmbedProtocol,
    dim: usize,
    client: reqwest::blocking::Client,
}

impl HttpEmbedder {
    /// Largest batch sent in one request. Keeps a single request from
    /// growing unbounded on a big workspace.
    const MAX_BATCH: usize = 64;

    /// Connect using an explicit protocol, probing the model's true
    /// output dimension.
    ///
    /// The dimension is measured rather than configured: a model swap
    /// then surfaces as a loud mismatch at build time instead of an
    /// index full of vectors from two different spaces.
    pub fn connect(
        protocol: EmbedProtocol,
        base_url: &str,
        model: &str,
    ) -> Result<Self, AimError> {
        let client = reqwest::blocking::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .map_err(|e| AimError::Index(format!("build http client: {e}")))?;

        let mut me = Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            protocol,
            dim: 0,
            client,
        };

        let probe = me.request(&["dimension probe".to_string()])?;
        let first = probe
            .first()
            .ok_or_else(|| AimError::Index("server returned no embeddings".into()))?;
        if first.is_empty() {
            return Err(AimError::Index(
                "server returned a zero-length embedding".into(),
            ));
        }
        me.dim = first.len();
        Ok(me)
    }

    /// Connect to a Lemonade Server.
    pub fn lemonade(base_url: &str, model: &str) -> Result<Self, AimError> {
        Self::connect(EmbedProtocol::Lemonade, base_url, model)
    }

    /// Connect to an Ollama server.
    pub fn ollama(base_url: &str, model: &str) -> Result<Self, AimError> {
        Self::connect(EmbedProtocol::Ollama, base_url, model)
    }

    /// Try each protocol in [`EmbedProtocol::PROBE_ORDER`] against
    /// `base_url` and keep the first that answers.
    ///
    /// On total failure the error lists what every protocol said, since
    /// "connection refused" on all three and "model not found" on one
    /// call for completely different fixes.
    pub fn autodetect(base_url: &str, model: &str) -> Result<Self, AimError> {
        let mut errors = Vec::new();
        for protocol in EmbedProtocol::PROBE_ORDER {
            match Self::connect(protocol, base_url, model) {
                Ok(e) => return Ok(e),
                Err(err) => errors.push(format!("  {}: {err}", protocol.tag())),
            }
        }
        Err(AimError::Index(format!(
            "no embeddings endpoint answered at {base_url} for model `{model}`:\n{}",
            errors.join("\n")
        )))
    }

    /// Which protocol this embedder settled on.
    pub fn protocol(&self) -> EmbedProtocol {
        self.protocol
    }

    /// Base URL in use.
    pub fn base_url(&self) -> &str {
        &self.base_url
    }

    /// Reported embedding dimension.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// POST one batch and return the raw (un-normalized) vectors.
    fn request(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, AimError> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let url = format!("{}{}", self.base_url, self.protocol.path());

        let body = match self.protocol {
            EmbedProtocol::Ollama => {
                // Single-prompt only; `embed_batch` never hands this
                // protocol more than one text.
                serde_json::json!({ "model": self.model, "prompt": texts[0] })
            }
            _ => serde_json::json!({ "model": self.model, "input": texts }),
        };

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .map_err(|e| AimError::Index(format!("POST {url}: {e}")))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let detail = resp.text().unwrap_or_default();
            let truncated: String = detail.chars().take(300).collect();
            return Err(AimError::Index(format!(
                "{url} returned {status} for model `{}`: {truncated}",
                self.model
            )));
        }

        let json: serde_json::Value = resp
            .json()
            .map_err(|e| AimError::Index(format!("decode response from {url}: {e}")))?;

        parse_embeddings(&json, self.protocol).ok_or_else(|| {
            AimError::Index(format!(
                "response from {url} did not contain embeddings in {} format",
                self.protocol.tag()
            ))
        })
    }
}

/// Pull vectors out of a response body according to `protocol`.
///
/// Both shapes are accepted regardless of the configured protocol: some
/// OpenAI-compatible servers answer on `/api/v1` and some Ollama builds
/// answer OpenAI-shaped. Being permissive on read costs nothing and
/// removes a class of confusing misconfiguration.
fn parse_embeddings(json: &serde_json::Value, protocol: EmbedProtocol) -> Option<Vec<Vec<f32>>> {
    let as_vec = |v: &serde_json::Value| -> Option<Vec<f32>> {
        Some(
            v.as_array()?
                .iter()
                .map(|x| x.as_f64().unwrap_or(0.0) as f32)
                .collect(),
        )
    };

    // OpenAI shape: { data: [ { index, embedding: [...] } ] }
    if let Some(data) = json.get("data").and_then(|d| d.as_array()) {
        let mut rows: Vec<(usize, Vec<f32>)> = Vec::with_capacity(data.len());
        for (fallback_idx, item) in data.iter().enumerate() {
            let vector = as_vec(item.get("embedding")?)?;
            let idx = item
                .get("index")
                .and_then(|i| i.as_u64())
                .map(|i| i as usize)
                .unwrap_or(fallback_idx);
            rows.push((idx, vector));
        }
        // The spec does not promise `data` is ordered, and a shuffled
        // batch would silently pair every chunk with another chunk's
        // vector — the worst kind of bug, because nothing errors.
        rows.sort_by_key(|(i, _)| *i);
        return Some(rows.into_iter().map(|(_, v)| v).collect());
    }

    // Ollama shape: { embedding: [...] }
    if let Some(v) = json.get("embedding").and_then(as_vec) {
        return Some(vec![v]);
    }

    // Ollama /api/embed shape: { embeddings: [[...]] }
    if let Some(rows) = json.get("embeddings").and_then(|e| e.as_array()) {
        return rows.iter().map(as_vec).collect();
    }

    let _ = protocol;
    None
}

impl Embedder for HttpEmbedder {
    fn dim(&self) -> usize {
        self.dim
    }

    fn id(&self) -> String {
        format!("{}:{}-d{}", self.protocol.tag(), self.model, self.dim)
    }

    fn embed(&self, text: &str) -> Result<Vec<f32>, AimError> {
        let mut rows = self.request(&[text.to_string()])?;
        let mut v = rows
            .pop()
            .ok_or_else(|| AimError::Index("server returned no embedding".into()))?;
        if v.len() != self.dim {
            return Err(AimError::DimMismatch {
                catalog: self.dim,
                got: v.len(),
            });
        }
        // Servers do not promise unit vectors, and the cosine threshold
        // downstream is only a cosine if they are.
        normalize(&mut v);
        Ok(v)
    }

    fn embed_batch(&self, texts: &[String]) -> Result<Vec<f32>, AimError> {
        let mut out = Vec::with_capacity(texts.len() * self.dim);
        let batch = if self.protocol.supports_batching() {
            Self::MAX_BATCH
        } else {
            1
        };

        for group in texts.chunks(batch) {
            let rows = self.request(group)?;
            if rows.len() != group.len() {
                return Err(AimError::Index(format!(
                    "asked for {} embeddings but the server returned {}",
                    group.len(),
                    rows.len()
                )));
            }
            for mut v in rows {
                if v.len() != self.dim {
                    return Err(AimError::DimMismatch {
                        catalog: self.dim,
                        got: v.len(),
                    });
                }
                normalize(&mut v);
                out.extend_from_slice(&v);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_paths_are_distinct_and_prioritize_lemonade() {
        assert_eq!(EmbedProtocol::PROBE_ORDER[0], EmbedProtocol::Lemonade);
        assert_eq!(EmbedProtocol::Lemonade.path(), "/api/v1/embeddings");
        assert_eq!(EmbedProtocol::OpenAi.path(), "/v1/embeddings");
        assert_eq!(EmbedProtocol::Ollama.path(), "/api/embeddings");
    }

    #[test]
    fn ollama_is_not_batched_but_openai_shapes_are() {
        assert!(!EmbedProtocol::Ollama.supports_batching());
        assert!(EmbedProtocol::Lemonade.supports_batching());
        assert!(EmbedProtocol::OpenAi.supports_batching());
    }

    #[test]
    fn parses_openai_shaped_response() {
        let json = serde_json::json!({
            "data": [
                { "index": 0, "embedding": [1.0, 0.0] },
                { "index": 1, "embedding": [0.0, 2.0] }
            ]
        });
        let rows = parse_embeddings(&json, EmbedProtocol::Lemonade).unwrap();
        assert_eq!(rows, vec![vec![1.0, 0.0], vec![0.0, 2.0]]);
    }

    #[test]
    fn reorders_openai_response_by_index() {
        // A server returning the batch out of order must not cause
        // vectors to be paired with the wrong chunk.
        let json = serde_json::json!({
            "data": [
                { "index": 2, "embedding": [3.0] },
                { "index": 0, "embedding": [1.0] },
                { "index": 1, "embedding": [2.0] }
            ]
        });
        let rows = parse_embeddings(&json, EmbedProtocol::OpenAi).unwrap();
        assert_eq!(rows, vec![vec![1.0], vec![2.0], vec![3.0]]);
    }

    #[test]
    fn parses_ollama_shaped_response() {
        let json = serde_json::json!({ "embedding": [0.5, 0.5] });
        let rows = parse_embeddings(&json, EmbedProtocol::Ollama).unwrap();
        assert_eq!(rows, vec![vec![0.5, 0.5]]);
    }

    #[test]
    fn parses_ollama_batch_embed_response() {
        let json = serde_json::json!({ "embeddings": [[1.0], [2.0]] });
        let rows = parse_embeddings(&json, EmbedProtocol::Ollama).unwrap();
        assert_eq!(rows, vec![vec![1.0], vec![2.0]]);
    }

    #[test]
    fn either_shape_parses_under_either_protocol() {
        // Misconfiguration tolerance: an OpenAI-shaped body must parse
        // even when the caller thinks it is talking to Ollama.
        let openai = serde_json::json!({ "data": [{ "index": 0, "embedding": [1.0] }] });
        assert!(parse_embeddings(&openai, EmbedProtocol::Ollama).is_some());
        let ollama = serde_json::json!({ "embedding": [1.0] });
        assert!(parse_embeddings(&ollama, EmbedProtocol::Lemonade).is_some());
    }

    #[test]
    fn rejects_a_response_with_no_embeddings() {
        let json = serde_json::json!({ "error": "model not found" });
        assert!(parse_embeddings(&json, EmbedProtocol::Lemonade).is_none());
    }

    #[test]
    fn embedder_id_records_protocol_model_and_dim() {
        // The id is what `check_embedder` compares, so it must
        // distinguish two servers that differ in any of the three.
        assert_ne!(
            format!("{}:{}-d{}", EmbedProtocol::Lemonade.tag(), "nomic", 768),
            format!("{}:{}-d{}", EmbedProtocol::Ollama.tag(), "nomic", 768)
        );
    }
}
