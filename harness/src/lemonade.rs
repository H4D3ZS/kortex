//! A [`PatchSource`] backed by a local Lemonade server.
//!
//! This closes the correction cycle: [`CorrectionLoop`] compiles a
//! patch, and when it fails this module asks the model for a corrected
//! one, parses the reply into a [`CodePatch`], and hands it back.
//!
//! Lemonade speaks the OpenAI schema at `/v1/chat/completions` on port
//! 13305 by default, so this also works against any OpenAI-compatible
//! server — llama.cpp's `llama-server`, vLLM, or Ollama's compatibility
//! endpoint — by pointing [`LemonadeAgent::new`] elsewhere.
//!
//! # Why this is blocking, not async
//!
//! The verification step shells out to `cargo check` and blocks for
//! seconds. Making the HTTP call async would not make the compiler
//! faster; it would just spread one inherently sequential cycle across
//! two concurrency models. The right boundary is a single
//! `spawn_blocking` around the whole cycle:
//!
//! ```no_run
//! # use hades_harness::verify::{CorrectionLoop, CodePatch};
//! # use hades_harness::lemonade::LemonadeAgent;
//! # async fn example() -> anyhow::Result<()> {
//! let outcome = tokio::task::spawn_blocking(move || {
//!     let mut agent = LemonadeAgent::new("http://localhost:13305", "qwen2.5-coder:3b")?;
//!     CorrectionLoop::new("/path/to/workspace")
//!         .run(CodePatch::new("src/lib.rs", "fn main() {}"), &mut agent)
//! })
//! .await??;
//! # Ok(())
//! # }
//! ```
//!
//! # Treating model output as untrusted
//!
//! A patch is a file path plus file contents produced by a language
//! model. Both are untrusted input. [`parse_patch`] rejects absolute
//! paths and any path that escapes the workspace root, because
//! `{"path": "../../../.ssh/authorized_keys"}` is a plausible thing for
//! a confused — or steered — model to emit, and the loop's whole job is
//! to write files.

use std::path::{Component, Path, PathBuf};
use std::time::Duration;

use anyhow::{anyhow, Result};
use serde_json::Value;

use crate::verify::{CodePatch, CorrectionRequest, PatchSource};

/// Default Lemonade address.
pub const LEMONADE_DEFAULT_URL: &str = "http://localhost:13305";

/// System prompt establishing the reply contract.
///
/// Kept short deliberately: a 3B-class model has a small context, and
/// every token here is one not available for the code it has to fix.
const SYSTEM_PROMPT: &str = "\
You are a code-fixing agent. You receive a compiler error and must repair the file.

Reply with a single JSON object and nothing else:
{\"path\": \"<file path relative to the workspace root>\", \"content\": \"<the complete corrected file>\"}

Rules:
- `content` must be the entire file, not a diff or a fragment.
- Fix the cause of the error. Do not delete the failing code to silence the compiler.
- No prose, no markdown, no explanation outside the JSON object.";

/// Talks to a Lemonade (or any OpenAI-compatible) server.
pub struct LemonadeAgent {
    base_url: String,
    model: String,
    client: reqwest::blocking::Client,
    workspace_root: Option<PathBuf>,
    temperature: f32,
    /// Full exchange history, for logging and trajectory memory.
    transcript: Vec<(String, String)>,
}

impl LemonadeAgent {
    /// Connect to `base_url` and use `model` for corrections.
    pub fn new(base_url: &str, model: &str) -> Result<Self> {
        let client = reqwest::blocking::Client::builder()
            // Generous: a small model on CPU can take a while for a
            // whole-file emission, and timing out mid-generation wastes
            // the attempt entirely.
            .timeout(Duration::from_secs(300))
            .build()?;

        Ok(Self {
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
            client,
            workspace_root: None,
            temperature: 0.2,
            transcript: Vec::new(),
        })
    }

    /// Connect to the default Lemonade address.
    pub fn local(model: &str) -> Result<Self> {
        Self::new(LEMONADE_DEFAULT_URL, model)
    }

    /// Confine proposed paths to this root. Strongly recommended: without
    /// it, only absolute paths and `..` escapes are rejected, and a
    /// symlink inside the tree could still redirect a write.
    pub fn with_workspace_root(mut self, root: impl Into<PathBuf>) -> Self {
        self.workspace_root = Some(root.into());
        self
    }

    /// Sampling temperature. Low by default — this is a repair task with
    /// a mostly-determined answer, not a creative one.
    pub fn with_temperature(mut self, temperature: f32) -> Self {
        self.temperature = temperature;
        self
    }

    /// Every (prompt, reply) pair so far.
    pub fn transcript(&self) -> &[(String, String)] {
        &self.transcript
    }

    /// Send one completion request and return the assistant text.
    pub fn complete(&self, user_prompt: &str) -> Result<String> {
        let url = format!("{}/v1/chat/completions", self.base_url);
        let body = serde_json::json!({
            "model": self.model,
            "temperature": self.temperature,
            "messages": [
                { "role": "system", "content": SYSTEM_PROMPT },
                { "role": "user", "content": user_prompt }
            ]
        });

        let resp = self
            .client
            .post(&url)
            .json(&body)
            .send()
            .map_err(|e| anyhow!("POST {url}: {e}"))?;

        if !resp.status().is_success() {
            let status = resp.status();
            let detail = resp.text().unwrap_or_default();
            let detail: String = detail.chars().take(400).collect();
            return Err(anyhow!(
                "{url} returned {status} for model `{}`: {detail}",
                self.model
            ));
        }

        let json: Value = resp
            .json()
            .map_err(|e| anyhow!("decode response from {url}: {e}"))?;

        extract_message_text(&json)
            .ok_or_else(|| anyhow!("no assistant message in response from {url}"))
    }
}

impl PatchSource for LemonadeAgent {
    fn correct(&mut self, request: &CorrectionRequest) -> Result<CodePatch> {
        let prompt = format!(
            "{}\n\nCurrent contents of {}:\n```\n{}\n```",
            request.render_prompt(5),
            request.patch.relative_path.to_string_lossy().replace('\\', "/"),
            request.patch.content
        );

        let reply = self.complete(&prompt)?;
        self.transcript.push((prompt, reply.clone()));
        parse_patch(&reply, self.workspace_root.as_deref())
    }
}

/// Pull the assistant text out of an OpenAI-shaped response.
///
/// Handles both `content` and a tool call's `arguments`, since a model
/// with tool-calling enabled may put the patch in either place.
pub fn extract_message_text(json: &Value) -> Option<String> {
    let message = json
        .get("choices")?
        .as_array()?
        .first()?
        .get("message")?;

    if let Some(content) = message.get("content").and_then(|c| c.as_str()) {
        if !content.trim().is_empty() {
            return Some(content.to_string());
        }
    }

    // Tool-calling models put the payload in `arguments`, a JSON string.
    message
        .get("tool_calls")?
        .as_array()?
        .first()?
        .get("function")?
        .get("arguments")?
        .as_str()
        .map(|s| s.to_string())
}

/// Parse a model reply into a [`CodePatch`].
///
/// Accepts a bare JSON object, one inside a ``` fence, or one embedded
/// in prose — small models emit all three regardless of instructions,
/// and failing the attempt over a stray "Here's the fix:" wastes a whole
/// compile cycle.
pub fn parse_patch(reply: &str, workspace_root: Option<&Path>) -> Result<CodePatch> {
    let candidate = extract_json_object(reply)
        .ok_or_else(|| anyhow!("no JSON object found in model reply: {}", preview(reply)))?;

    let value: Value = serde_json::from_str(&candidate)
        .map_err(|e| anyhow!("model reply is not valid JSON ({e}): {}", preview(&candidate)))?;

    // Accept a few key spellings. Constraining a small model to exactly
    // one is not reliable, and the alternative is discarding an
    // otherwise-correct patch.
    let path = first_str(&value, &["path", "file", "file_path", "relative_path"])
        .ok_or_else(|| anyhow!("model reply has no `path` field: {}", preview(&candidate)))?;
    let content = first_str(&value, &["content", "code", "contents", "new_content"])
        .ok_or_else(|| anyhow!("model reply has no `content` field: {}", preview(&candidate)))?;

    let relative = validate_path(&path, workspace_root)?;
    Ok(CodePatch::new(relative, content))
}

/// Reject a proposed path that is absolute or escapes the workspace.
///
/// Model output is untrusted, and this function guards a code path whose
/// entire purpose is writing files.
fn validate_path(path: &str, workspace_root: Option<&Path>) -> Result<PathBuf> {
    let normalized = path.replace('\\', "/");
    let candidate = Path::new(&normalized);

    if candidate.is_absolute() {
        return Err(anyhow!(
            "model proposed an absolute path `{path}`; only workspace-relative paths are allowed"
        ));
    }

    let mut depth = 0i32;
    for component in candidate.components() {
        match component {
            Component::ParentDir => {
                depth -= 1;
                if depth < 0 {
                    return Err(anyhow!(
                        "model proposed a path escaping the workspace: `{path}`"
                    ));
                }
            }
            Component::Normal(_) => depth += 1,
            Component::CurDir => {}
            // A Windows drive or root prefix on a supposedly relative
            // path is malformed by definition.
            Component::Prefix(_) | Component::RootDir => {
                return Err(anyhow!("model proposed a rooted path `{path}`"));
            }
        }
    }

    if let Some(root) = workspace_root {
        // Compare against the resolved root when it exists. `canonicalize`
        // on the joined path would fail for a file being created, so only
        // the parent is resolved.
        let joined = root.join(candidate);
        if let (Ok(real_root), Some(parent)) = (root.canonicalize(), joined.parent()) {
            if let Ok(real_parent) = parent.canonicalize() {
                if !real_parent.starts_with(&real_root) {
                    return Err(anyhow!(
                        "model proposed `{path}`, which resolves outside the workspace \
                         (possibly through a symlink)"
                    ));
                }
            }
        }
    }

    Ok(candidate.to_path_buf())
}

/// First present string field among `keys`.
fn first_str(value: &Value, keys: &[&str]) -> Option<String> {
    keys.iter()
        .find_map(|k| value.get(*k).and_then(|v| v.as_str()))
        .map(|s| s.to_string())
}

/// Find the first balanced `{...}` object, ignoring braces inside string
/// literals so that code containing `{` in the `content` field does not
/// truncate the match.
fn extract_json_object(text: &str) -> Option<String> {
    let bytes: Vec<char> = text.chars().collect();
    let start = bytes.iter().position(|c| *c == '{')?;

    let mut depth = 0usize;
    let mut in_string = false;
    let mut escaped = false;

    for (i, ch) in bytes.iter().enumerate().skip(start) {
        if in_string {
            if escaped {
                escaped = false;
            } else if *ch == '\\' {
                escaped = true;
            } else if *ch == '"' {
                in_string = false;
            }
            continue;
        }
        match ch {
            '"' => in_string = true,
            '{' => depth += 1,
            '}' => {
                depth -= 1;
                if depth == 0 {
                    return Some(bytes[start..=i].iter().collect());
                }
            }
            _ => {}
        }
    }
    None
}

/// Short excerpt for error messages.
fn preview(text: &str) -> String {
    let trimmed = text.trim();
    let short: String = trimmed.chars().take(160).collect();
    if trimmed.chars().count() > 160 {
        format!("{short}…")
    } else {
        short
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    #[test]
    fn parses_a_bare_json_object() {
        let patch = parse_patch(r#"{"path":"src/lib.rs","content":"fn main() {}"}"#, None).unwrap();
        assert_eq!(patch.relative_path, PathBuf::from("src/lib.rs"));
        assert_eq!(patch.content, "fn main() {}");
    }

    #[test]
    fn parses_json_inside_a_markdown_fence() {
        // Small models emit fences no matter what the system prompt says.
        let reply = "Here is the fix:\n```json\n{\"path\":\"a.rs\",\"content\":\"x\"}\n```\nHope that helps!";
        let patch = parse_patch(reply, None).unwrap();
        assert_eq!(patch.relative_path, PathBuf::from("a.rs"));
    }

    #[test]
    fn braces_inside_content_do_not_truncate_the_object() {
        // The naive "find first { and last }" approach breaks here, and
        // code content almost always contains braces.
        let reply = r#"{"path":"src/lib.rs","content":"fn main() { let x = {1}; }"}"#;
        let patch = parse_patch(reply, None).unwrap();
        assert_eq!(patch.content, "fn main() { let x = {1}; }");
    }

    #[test]
    fn escaped_quotes_in_content_are_handled() {
        let reply = r#"{"path":"a.rs","content":"println!(\"hi {}\", x);"}"#;
        let patch = parse_patch(reply, None).unwrap();
        assert_eq!(patch.content, r#"println!("hi {}", x);"#);
    }

    #[test]
    fn accepts_alternate_key_spellings() {
        for reply in [
            r#"{"file":"a.rs","code":"x"}"#,
            r#"{"file_path":"a.rs","contents":"x"}"#,
            r#"{"relative_path":"a.rs","new_content":"x"}"#,
        ] {
            let patch = parse_patch(reply, None).unwrap();
            assert_eq!(patch.relative_path, PathBuf::from("a.rs"), "{reply}");
            assert_eq!(patch.content, "x");
        }
    }

    #[test]
    fn rejects_absolute_paths() {
        #[cfg(windows)]
        let evil = r#"{"path":"C:\\Windows\\System32\\drivers\\etc\\hosts","content":"x"}"#;
        #[cfg(not(windows))]
        let evil = r#"{"path":"/etc/passwd","content":"x"}"#;

        let err = parse_patch(evil, None).unwrap_err().to_string();
        assert!(
            err.contains("absolute") || err.contains("rooted"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn rejects_paths_that_escape_the_workspace() {
        // The loop writes files, so this is the security boundary.
        let evil = r#"{"path":"../../../.ssh/authorized_keys","content":"ssh-rsa AAA"}"#;
        let err = parse_patch(evil, None).unwrap_err().to_string();
        assert!(err.contains("escaping"), "unexpected error: {err}");
    }

    #[test]
    fn allows_dotdot_that_stays_inside_the_tree() {
        // `src/foo/../lib.rs` is still `src/lib.rs` — legitimate.
        let patch = parse_patch(r#"{"path":"src/foo/../lib.rs","content":"x"}"#, None).unwrap();
        assert!(patch.relative_path.to_string_lossy().contains("lib.rs"));
    }

    #[test]
    fn normalizes_windows_separators() {
        let patch = parse_patch(r#"{"path":"src\\hw\\mbox.c","content":"x"}"#, None).unwrap();
        let shown = patch.relative_path.to_string_lossy().replace('\\', "/");
        assert_eq!(shown, "src/hw/mbox.c");
    }

    #[test]
    fn reports_a_helpful_error_when_there_is_no_json() {
        let err = parse_patch("I cannot help with that.", None)
            .unwrap_err()
            .to_string();
        assert!(err.contains("no JSON object"), "{err}");
        assert!(err.contains("I cannot help"), "error should quote the reply: {err}");
    }

    #[test]
    fn reports_missing_fields_distinctly() {
        let err = parse_patch(r#"{"path":"a.rs"}"#, None).unwrap_err().to_string();
        assert!(err.contains("`content`"), "{err}");
        let err = parse_patch(r#"{"content":"x"}"#, None).unwrap_err().to_string();
        assert!(err.contains("`path`"), "{err}");
    }

    #[test]
    fn extracts_content_from_an_openai_response() {
        let body = json!({
            "choices": [{ "message": { "role": "assistant", "content": "{\"path\":\"a.rs\"}" } }]
        });
        assert_eq!(
            extract_message_text(&body).unwrap(),
            "{\"path\":\"a.rs\"}"
        );
    }

    #[test]
    fn falls_back_to_tool_call_arguments() {
        // A tool-calling model leaves `content` empty and puts the
        // payload in `arguments`.
        let body = json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "function": {
                            "name": "write_file",
                            "arguments": "{\"path\":\"a.rs\",\"content\":\"x\"}"
                        }
                    }]
                }
            }]
        });
        let text = extract_message_text(&body).unwrap();
        let patch = parse_patch(&text, None).unwrap();
        assert_eq!(patch.relative_path, PathBuf::from("a.rs"));
    }

    #[test]
    fn missing_choices_is_none_not_a_panic() {
        assert!(extract_message_text(&json!({})).is_none());
        assert!(extract_message_text(&json!({ "choices": [] })).is_none());
    }

    #[test]
    fn workspace_root_confines_writes() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();

        let ok = parse_patch(
            r#"{"path":"src/lib.rs","content":"x"}"#,
            Some(dir.path()),
        );
        assert!(ok.is_ok(), "{:?}", ok.err());

        let escaped = parse_patch(
            r#"{"path":"../outside.rs","content":"x"}"#,
            Some(dir.path()),
        );
        assert!(escaped.is_err());
    }
}
