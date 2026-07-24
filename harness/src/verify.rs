//! The compiler-in-the-loop correction cycle.
//!
//! A small model's characteristic failure is confident, plausible, wrong
//! code. The fix is not a better prompt — it is refusing to accept any
//! patch the compiler has not seen. This module applies a proposed patch
//! to a shadow copy of the workspace, compiles it, and on failure hands
//! the exact diagnostics back to whatever produced the patch, up to a
//! retry limit. Only a patch that compiles is written to the real tree.
//!
//! # What this does and does not prove
//!
//! A successful run proves the workspace **compiles**. It does not prove
//! the change is correct: code can compile and still be wrong, and a
//! model can satisfy the compiler by deleting the failing call. That is
//! why the success variant is named [`Outcome::Compiles`] rather than
//! "verified" or "correct".
//!
//! Claims of "0% hallucination" from a loop like this are false. What it
//! genuinely removes is the large class of errors a type checker can
//! see: missing imports, wrong arity, invented method names, borrow
//! violations. On a 3B-class model that is most of them — which is worth
//! a lot, and is not the same as correctness.
//!
//! Pair it with tests ([`Verifier::Command`]) when you want evidence
//! about behaviour rather than types.

use std::path::{Path, PathBuf};
use std::process::Command;

use anyhow::Result;

use crate::{Diagnostic, ShadowVFS, SymbolicVerifier};

/// A proposed edit: the full new contents of one file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CodePatch {
    /// Path relative to the workspace root.
    pub relative_path: PathBuf,
    /// Complete new file contents.
    pub content: String,
}

impl CodePatch {
    /// Create a patch replacing `relative_path` with `content`.
    pub fn new(relative_path: impl Into<PathBuf>, content: impl Into<String>) -> Self {
        Self {
            relative_path: relative_path.into(),
            content: content.into(),
        }
    }
}

/// What the loop asks for after a failed attempt.
#[derive(Debug, Clone)]
pub struct CorrectionRequest {
    /// 1-based index of the attempt that just failed.
    pub failed_attempt: usize,
    /// How many attempts remain after this one.
    pub attempts_remaining: usize,
    /// The patch that failed.
    pub patch: CodePatch,
    /// Every diagnostic the verifier produced.
    pub diagnostics: Vec<Diagnostic>,
}

impl CorrectionRequest {
    /// Diagnostics at `level == "error"`. Warnings never block a patch.
    pub fn errors(&self) -> Vec<&Diagnostic> {
        self.diagnostics
            .iter()
            .filter(|d| d.level == "error")
            .collect()
    }

    /// Render the failure as prompt text, capped at `max_errors`.
    ///
    /// The cap matters: a single bad edit can produce hundreds of
    /// cascading errors, and pasting all of them into a small model's
    /// context evicts the code it needs to do the fix. The first few
    /// errors are almost always the root cause; the rest are echoes.
    pub fn render_prompt(&self, max_errors: usize) -> String {
        let errors = self.errors();
        let shown = errors.len().min(max_errors.max(1));

        let mut out = format!(
            "Your patch to {} failed to compile (attempt {} of {}).\n\n",
            self.relative_path_display(),
            self.failed_attempt,
            self.failed_attempt + self.attempts_remaining
        );

        for diag in errors.iter().take(shown) {
            match &diag.span {
                Some(span) => out.push_str(&format!(
                    "error at {}:{}:{}\n  {}\n",
                    span.file_name, span.line_start, span.column_start, diag.message
                )),
                None => out.push_str(&format!("error\n  {}\n", diag.message)),
            }
        }

        if errors.len() > shown {
            out.push_str(&format!(
                "\n... and {} further errors, most likely cascading from the ones above.\n",
                errors.len() - shown
            ));
        }

        out.push_str(
            "\nFix the cause of the first error and output the complete corrected file. \
             Do not delete the failing code to silence the compiler.",
        );
        out
    }

    fn relative_path_display(&self) -> String {
        self.patch.relative_path.to_string_lossy().replace('\\', "/")
    }
}

/// Produces patches: a model, a scripted sequence, or a test fake.
pub trait PatchSource {
    /// Produce a corrected patch given the failure that just occurred.
    fn correct(&mut self, request: &CorrectionRequest) -> Result<CodePatch>;
}

/// How to decide whether a shadow workspace is healthy.
pub enum Verifier {
    /// `cargo check --message-format=json`, parsed into diagnostics.
    CargoCheck,
    /// An arbitrary command run in the shadow root — `cargo test`,
    /// `clang`, `flutter analyze`. A non-zero exit is a failure and
    /// stderr becomes a single diagnostic.
    ///
    /// Use this when you want evidence about *behaviour*. `CargoCheck`
    /// only ever proves the code type-checks.
    Command {
        program: String,
        args: Vec<String>,
    },
}

impl Verifier {
    /// Run against an already-populated shadow root.
    fn run(&self, shadow_root: &Path, original_root: &Path) -> Result<Vec<Diagnostic>> {
        match self {
            Verifier::CargoCheck => SymbolicVerifier::verify_cargo(shadow_root, original_root),
            Verifier::Command { program, args } => {
                let output = Command::new(program)
                    .args(args)
                    .current_dir(shadow_root)
                    .output()?;

                if output.status.success() {
                    return Ok(Vec::new());
                }
                // Prefer stderr, fall back to stdout: test harnesses vary
                // in which stream carries the failure detail.
                let mut detail = String::from_utf8_lossy(&output.stderr).trim().to_string();
                if detail.is_empty() {
                    detail = String::from_utf8_lossy(&output.stdout).trim().to_string();
                }
                if detail.is_empty() {
                    detail = format!("`{program}` exited with {}", output.status);
                }
                Ok(vec![Diagnostic {
                    message: detail,
                    level: "error".to_string(),
                    span: None,
                }])
            }
        }
    }
}

/// How a correction cycle ended.
#[derive(Debug)]
pub enum Outcome {
    /// A patch compiled. Named for what was actually proven — the
    /// workspace builds — not for correctness, which no compiler can
    /// establish.
    Compiles {
        patch: CodePatch,
        /// Attempts used, 1 meaning the first proposal was accepted.
        attempts: usize,
        /// Non-blocking warnings from the successful build.
        warnings: Vec<Diagnostic>,
        /// Whether the patch was written to the real workspace.
        committed: bool,
    },
    /// Every attempt failed to compile. The real workspace is untouched.
    Exhausted {
        attempts: usize,
        last_patch: CodePatch,
        diagnostics: Vec<Diagnostic>,
    },
}

impl Outcome {
    /// True when a patch compiled.
    pub fn compiles(&self) -> bool {
        matches!(self, Outcome::Compiles { .. })
    }

    /// Attempts consumed.
    pub fn attempts(&self) -> usize {
        match self {
            Outcome::Compiles { attempts, .. } | Outcome::Exhausted { attempts, .. } => *attempts,
        }
    }
}

/// Settings for [`CorrectionLoop`].
#[derive(Debug, Clone)]
pub struct LoopConfig {
    /// Total attempts including the initial proposal. Must be >= 1.
    pub max_attempts: usize,
    /// Errors included in a correction prompt.
    pub max_errors_in_prompt: usize,
    /// Write the patch to the real workspace once it compiles.
    ///
    /// Off by default. Writing to a user's tree is not something a
    /// library should do because a config field happened to default to
    /// true.
    pub commit_on_success: bool,
}

impl Default for LoopConfig {
    fn default() -> Self {
        Self {
            max_attempts: 3,
            max_errors_in_prompt: 5,
            commit_on_success: false,
        }
    }
}

/// Applies patches to a shadow workspace, compiles, and retries.
pub struct CorrectionLoop {
    workspace_root: PathBuf,
    verifier: Verifier,
    config: LoopConfig,
}

impl CorrectionLoop {
    /// Create a loop over `workspace_root` using `cargo check`.
    pub fn new(workspace_root: impl Into<PathBuf>) -> Self {
        Self {
            workspace_root: workspace_root.into(),
            verifier: Verifier::CargoCheck,
            config: LoopConfig::default(),
        }
    }

    /// Use a different verifier — `cargo test`, `clang`, and so on.
    pub fn with_verifier(mut self, verifier: Verifier) -> Self {
        self.verifier = verifier;
        self
    }

    /// Override the loop settings.
    pub fn with_config(mut self, config: LoopConfig) -> Self {
        self.config = config;
        self
    }

    /// Run the cycle: apply, compile, and on failure ask `source` for a
    /// correction, until a patch compiles or attempts run out.
    ///
    /// The real workspace is only touched after a patch compiles, and
    /// only when `commit_on_success` is set. A failed cycle leaves the
    /// tree byte-for-byte unchanged.
    pub fn run<S: PatchSource>(
        &self,
        initial: CodePatch,
        source: &mut S,
    ) -> Result<Outcome> {
        let max_attempts = self.config.max_attempts.max(1);
        let mut patch = initial;

        for attempt in 1..=max_attempts {
            // A fresh shadow per attempt. Reusing one would let a
            // discarded edit from a previous attempt linger and make the
            // next verification describe a tree nobody proposed.
            let shadow = ShadowVFS::new(&self.workspace_root)?;
            shadow.apply_patch(&patch.relative_path, &patch.content)?;

            let diagnostics = self.verifier.run(&shadow.mount_path, &self.workspace_root)?;
            let errors: Vec<Diagnostic> = diagnostics
                .iter()
                .filter(|d| d.level == "error")
                .cloned()
                .collect();

            if errors.is_empty() {
                let committed = if self.config.commit_on_success {
                    self.commit(&patch)?;
                    true
                } else {
                    false
                };
                return Ok(Outcome::Compiles {
                    patch,
                    attempts: attempt,
                    warnings: diagnostics
                        .into_iter()
                        .filter(|d| d.level != "error")
                        .collect(),
                    committed,
                });
            }

            if attempt == max_attempts {
                return Ok(Outcome::Exhausted {
                    attempts: attempt,
                    last_patch: patch,
                    diagnostics,
                });
            }

            let request = CorrectionRequest {
                failed_attempt: attempt,
                attempts_remaining: max_attempts - attempt,
                patch: patch.clone(),
                diagnostics,
            };
            patch = source.correct(&request)?;
        }

        // `max_attempts >= 1` and the loop returns on its final
        // iteration, so this is unreachable.
        unreachable!("correction loop exited without returning an outcome")
    }

    /// Write a patch into the real workspace.
    fn commit(&self, patch: &CodePatch) -> Result<()> {
        let target = self.workspace_root.join(&patch.relative_path);
        if let Some(parent) = target.parent() {
            std::fs::create_dir_all(parent)?;
        }
        std::fs::write(&target, &patch.content)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Span;

    fn diag(level: &str, message: &str, line: Option<usize>) -> Diagnostic {
        Diagnostic {
            message: message.to_string(),
            level: level.to_string(),
            span: line.map(|line_start| Span {
                file_name: "src/lib.rs".to_string(),
                line_start,
                column_start: 5,
            }),
        }
    }

    /// Replays a fixed sequence of corrections and counts calls.
    struct Scripted {
        patches: Vec<CodePatch>,
        calls: usize,
        last_prompt: String,
    }

    impl Scripted {
        fn new(patches: Vec<CodePatch>) -> Self {
            Self {
                patches,
                calls: 0,
                last_prompt: String::new(),
            }
        }
    }

    impl PatchSource for Scripted {
        fn correct(&mut self, request: &CorrectionRequest) -> Result<CodePatch> {
            self.last_prompt = request.render_prompt(5);
            let patch = self
                .patches
                .get(self.calls)
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("scripted source ran out of patches"))?;
            self.calls += 1;
            Ok(patch)
        }
    }

    fn temp_workspace() -> tempfile::TempDir {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("src")).unwrap();
        std::fs::write(dir.path().join("src/lib.rs"), "// original\n").unwrap();
        dir
    }

    /// Succeeds or fails based on a queue of scripted verdicts, so the
    /// loop's control flow can be tested without invoking a compiler.
    enum Scriptable {
        Verdicts(std::cell::RefCell<Vec<Vec<Diagnostic>>>),
    }

    impl Scriptable {
        fn run(&self) -> Vec<Diagnostic> {
            match self {
                Scriptable::Verdicts(q) => {
                    let mut q = q.borrow_mut();
                    if q.is_empty() {
                        Vec::new()
                    } else {
                        q.remove(0)
                    }
                }
            }
        }
    }

    #[test]
    fn correction_prompt_includes_file_line_and_message() {
        let request = CorrectionRequest {
            failed_attempt: 1,
            attempts_remaining: 2,
            patch: CodePatch::new("src/lib.rs", "fn main() {}"),
            diagnostics: vec![diag("error", "cannot find value `x`", Some(42))],
        };
        let prompt = request.render_prompt(5);
        assert!(prompt.contains("src/lib.rs:42:5"), "{prompt}");
        assert!(prompt.contains("cannot find value `x`"), "{prompt}");
        assert!(prompt.contains("attempt 1 of 3"), "{prompt}");
    }

    #[test]
    fn correction_prompt_caps_cascading_errors() {
        // One bad edit can produce hundreds of errors; pasting them all
        // into a small context window evicts the code needed to fix it.
        let diagnostics: Vec<Diagnostic> = (1..=50)
            .map(|i| diag("error", &format!("error number {i}"), Some(i)))
            .collect();
        let request = CorrectionRequest {
            failed_attempt: 1,
            attempts_remaining: 1,
            patch: CodePatch::new("src/lib.rs", ""),
            diagnostics,
        };
        let prompt = request.render_prompt(3);
        assert!(prompt.contains("error number 1"));
        assert!(prompt.contains("error number 3"));
        assert!(!prompt.contains("error number 4"), "{prompt}");
        assert!(prompt.contains("47 further errors"), "{prompt}");
    }

    #[test]
    fn warnings_are_not_treated_as_errors() {
        let request = CorrectionRequest {
            failed_attempt: 1,
            attempts_remaining: 1,
            patch: CodePatch::new("src/lib.rs", ""),
            diagnostics: vec![
                diag("warning", "unused variable", Some(3)),
                diag("error", "type mismatch", Some(7)),
            ],
        };
        assert_eq!(request.errors().len(), 1);
        assert_eq!(request.errors()[0].message, "type mismatch");
    }

    #[test]
    fn prompt_survives_a_diagnostic_with_no_span() {
        let request = CorrectionRequest {
            failed_attempt: 1,
            attempts_remaining: 1,
            patch: CodePatch::new("src/lib.rs", ""),
            diagnostics: vec![diag("error", "linking failed", None)],
        };
        let prompt = request.render_prompt(5);
        assert!(prompt.contains("linking failed"), "{prompt}");
    }

    #[test]
    fn prompt_forbids_deleting_code_to_satisfy_the_compiler() {
        // The cheapest way to make a build pass is to remove the failing
        // call, which a small model will happily do unless told not to.
        let request = CorrectionRequest {
            failed_attempt: 1,
            attempts_remaining: 1,
            patch: CodePatch::new("src/lib.rs", ""),
            diagnostics: vec![diag("error", "boom", Some(1))],
        };
        assert!(request.render_prompt(5).contains("Do not delete"));
    }

    #[test]
    fn windows_paths_render_with_forward_slashes() {
        let request = CorrectionRequest {
            failed_attempt: 1,
            attempts_remaining: 0,
            patch: CodePatch::new(PathBuf::from("src").join("hw").join("mbox.c"), ""),
            diagnostics: vec![diag("error", "boom", Some(1))],
        };
        assert!(request.render_prompt(1).contains("src/hw/mbox.c"));
    }

    #[test]
    fn command_verifier_reports_failure_from_stderr() {
        let dir = temp_workspace();
        let verifier = Verifier::Command {
            #[cfg(windows)]
            program: "cmd".to_string(),
            #[cfg(windows)]
            args: vec!["/C".into(), "echo boom 1>&2 && exit 1".into()],
            #[cfg(not(windows))]
            program: "sh".to_string(),
            #[cfg(not(windows))]
            args: vec!["-c".into(), "echo boom >&2; exit 1".into()],
        };
        let diagnostics = verifier.run(dir.path(), dir.path()).unwrap();
        assert_eq!(diagnostics.len(), 1);
        assert_eq!(diagnostics[0].level, "error");
        assert!(diagnostics[0].message.contains("boom"), "{diagnostics:?}");
    }

    #[test]
    fn command_verifier_reports_success_as_no_diagnostics() {
        let dir = temp_workspace();
        let verifier = Verifier::Command {
            #[cfg(windows)]
            program: "cmd".to_string(),
            #[cfg(windows)]
            args: vec!["/C".into(), "exit 0".into()],
            #[cfg(not(windows))]
            program: "sh".to_string(),
            #[cfg(not(windows))]
            args: vec!["-c".into(), "exit 0".into()],
        };
        assert!(verifier.run(dir.path(), dir.path()).unwrap().is_empty());
    }

    #[test]
    fn a_passing_patch_is_accepted_on_the_first_attempt() {
        let dir = temp_workspace();
        let loop_ = CorrectionLoop::new(dir.path()).with_verifier(Verifier::Command {
            #[cfg(windows)]
            program: "cmd".to_string(),
            #[cfg(windows)]
            args: vec!["/C".into(), "exit 0".into()],
            #[cfg(not(windows))]
            program: "sh".to_string(),
            #[cfg(not(windows))]
            args: vec!["-c".into(), "exit 0".into()],
        });

        let mut source = Scripted::new(vec![]);
        let outcome = loop_
            .run(CodePatch::new("src/lib.rs", "fn good() {}"), &mut source)
            .unwrap();

        assert!(outcome.compiles());
        assert_eq!(outcome.attempts(), 1);
        assert_eq!(source.calls, 0, "no correction should have been requested");
    }

    #[test]
    fn a_failing_patch_exhausts_attempts_and_asks_for_corrections() {
        let dir = temp_workspace();
        let loop_ = CorrectionLoop::new(dir.path())
            .with_verifier(Verifier::Command {
                #[cfg(windows)]
                program: "cmd".to_string(),
                #[cfg(windows)]
                args: vec!["/C".into(), "echo nope 1>&2 && exit 1".into()],
                #[cfg(not(windows))]
                program: "sh".to_string(),
                #[cfg(not(windows))]
                args: vec!["-c".into(), "echo nope >&2; exit 1".into()],
            })
            .with_config(LoopConfig {
                max_attempts: 3,
                ..Default::default()
            });

        let mut source = Scripted::new(vec![
            CodePatch::new("src/lib.rs", "attempt 2"),
            CodePatch::new("src/lib.rs", "attempt 3"),
        ]);
        let outcome = loop_
            .run(CodePatch::new("src/lib.rs", "attempt 1"), &mut source)
            .unwrap();

        assert!(!outcome.compiles());
        assert_eq!(outcome.attempts(), 3);
        // Corrections are requested after attempts 1 and 2, but not
        // after the last one — asking for a patch nobody will try is
        // a wasted model call.
        assert_eq!(source.calls, 2);
        assert!(source.last_prompt.contains("nope"), "{}", source.last_prompt);
    }

    #[test]
    fn a_failed_cycle_leaves_the_real_workspace_untouched() {
        let dir = temp_workspace();
        let before = std::fs::read_to_string(dir.path().join("src/lib.rs")).unwrap();

        let loop_ = CorrectionLoop::new(dir.path())
            .with_verifier(Verifier::Command {
                #[cfg(windows)]
                program: "cmd".to_string(),
                #[cfg(windows)]
                args: vec!["/C".into(), "exit 1".into()],
                #[cfg(not(windows))]
                program: "sh".to_string(),
                #[cfg(not(windows))]
                args: vec!["-c".into(), "exit 1".into()],
            })
            .with_config(LoopConfig {
                max_attempts: 1,
                commit_on_success: true,
                ..Default::default()
            });

        let mut source = Scripted::new(vec![]);
        let outcome = loop_
            .run(CodePatch::new("src/lib.rs", "DESTRUCTIVE"), &mut source)
            .unwrap();

        assert!(!outcome.compiles());
        let after = std::fs::read_to_string(dir.path().join("src/lib.rs")).unwrap();
        assert_eq!(before, after, "a failed cycle modified the real workspace");
    }

    #[test]
    fn commit_is_off_unless_requested() {
        let dir = temp_workspace();
        let ok = Verifier::Command {
            #[cfg(windows)]
            program: "cmd".to_string(),
            #[cfg(windows)]
            args: vec!["/C".into(), "exit 0".into()],
            #[cfg(not(windows))]
            program: "sh".to_string(),
            #[cfg(not(windows))]
            args: vec!["-c".into(), "exit 0".into()],
        };

        let mut source = Scripted::new(vec![]);
        let outcome = CorrectionLoop::new(dir.path())
            .with_verifier(ok)
            .run(CodePatch::new("src/lib.rs", "NEW"), &mut source)
            .unwrap();

        assert!(matches!(outcome, Outcome::Compiles { committed: false, .. }));
        let on_disk = std::fs::read_to_string(dir.path().join("src/lib.rs")).unwrap();
        assert_eq!(on_disk, "// original\n", "workspace written without opt-in");
    }

    #[test]
    fn commit_on_success_writes_the_patch() {
        let dir = temp_workspace();
        let ok = Verifier::Command {
            #[cfg(windows)]
            program: "cmd".to_string(),
            #[cfg(windows)]
            args: vec!["/C".into(), "exit 0".into()],
            #[cfg(not(windows))]
            program: "sh".to_string(),
            #[cfg(not(windows))]
            args: vec!["-c".into(), "exit 0".into()],
        };

        let mut source = Scripted::new(vec![]);
        let outcome = CorrectionLoop::new(dir.path())
            .with_verifier(ok)
            .with_config(LoopConfig {
                commit_on_success: true,
                ..Default::default()
            })
            .run(CodePatch::new("src/lib.rs", "COMMITTED"), &mut source)
            .unwrap();

        assert!(matches!(outcome, Outcome::Compiles { committed: true, .. }));
        let on_disk = std::fs::read_to_string(dir.path().join("src/lib.rs")).unwrap();
        assert_eq!(on_disk, "COMMITTED");
    }

    #[test]
    fn max_attempts_of_zero_is_clamped_to_one() {
        let dir = temp_workspace();
        let ok = Verifier::Command {
            #[cfg(windows)]
            program: "cmd".to_string(),
            #[cfg(windows)]
            args: vec!["/C".into(), "exit 0".into()],
            #[cfg(not(windows))]
            program: "sh".to_string(),
            #[cfg(not(windows))]
            args: vec!["-c".into(), "exit 0".into()],
        };
        let mut source = Scripted::new(vec![]);
        let outcome = CorrectionLoop::new(dir.path())
            .with_verifier(ok)
            .with_config(LoopConfig {
                max_attempts: 0,
                ..Default::default()
            })
            .run(CodePatch::new("src/lib.rs", "x"), &mut source)
            .unwrap();
        assert_eq!(outcome.attempts(), 1);
    }

    #[test]
    fn scriptable_verdicts_helper_drains_in_order() {
        let s = Scriptable::Verdicts(std::cell::RefCell::new(vec![
            vec![diag("error", "first", None)],
            vec![],
        ]));
        assert_eq!(s.run().len(), 1);
        assert!(s.run().is_empty());
    }
}
