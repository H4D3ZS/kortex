//! Retrieval-accuracy benchmark for the HRR path-key superposition scheme
//! (daemon::neural_math). This is the harness that produces the numbers in
//! kortex/README.md's "Semantic Retrieval Accuracy" table — run it yourself:
//!
//!     cargo run --release --example hrr_benchmark -p daemon
//!
//! WHAT IT MEASURES
//! For each scale k, it superposes k synthetic (path, content-vector) pairs
//! into one 1536-dim global vector via circular_convolution + addition — the
//! exact construction in neuraldrive/src-tauri/src/lib.rs's chunk-indexing
//! loop, minus the TTT recency blend (0.85/0.15), which trades off *older*
//! bindings for *newer* ones and would only make retrieval of any given item
//! look better or worse depending on when it was written relative to the
//! query — a separate design axis from raw superposition capacity. Testing
//! pure superposition isolates the question this benchmark exists to answer.
//!
//! Paths are generated with shared directory prefixes on purpose
//! (src/module_7/component_3/file_12.rs, .../file_13.rs, ...) — the exact
//! shape of paths that shared a correlated key under the old sin-based
//! scheme. Content vectors are independent random unit vectors: a stand-in
//! for embeddings of unrelated file content, which is optimistic (real
//! embeddings of genuinely similar code are *more* correlated than random,
//! which would only hurt retrieval further — this benchmark is a best case,
//! not a worst case).
//!
//! "Retrieved correctly" = the unbound candidate's cosine similarity to the
//! item's true content vector exceeds 0.85 — the same "Page Fault Trigger"
//! threshold the README's own architecture diagram already specifies.
//!
//! For large k, testing retrieval on every single item is unnecessary to
//! get a stable accuracy estimate; each scale samples up to 300 items for
//! the retrieval test (construction still superposes all k).

use daemon::neural_math::{circular_convolution, circular_correlation, path_key, VECTOR_DIM};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// A random unit vector in R^1536, standing in for a content embedding.
fn random_unit_vector(rng: &mut StdRng) -> [f32; VECTOR_DIM] {
    let mut v = [0.0f32; VECTOR_DIM];
    let mut i = 0;
    while i < VECTOR_DIM {
        let u1: f32 = rng.gen_range(f32::EPSILON..1.0);
        let u2: f32 = rng.gen_range(0.0..1.0);
        let r = (-2.0 * u1.ln()).sqrt();
        let theta = std::f32::consts::TAU * u2;
        v[i] = r * theta.cos();
        if i + 1 < VECTOR_DIM {
            v[i + 1] = r * theta.sin();
        }
        i += 2;
    }
    let mag: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag > 1e-6 {
        for x in v.iter_mut() {
            *x /= mag;
        }
    }
    v
}

fn cosine_similarity(a: &[f32; VECTOR_DIM], b: &[f32; VECTOR_DIM]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-9 || nb < 1e-9 { 0.0 } else { dot / (na * nb) }
}

/// Synthetic path with real directory-collision structure: many files
/// share a long common prefix, the exact case that broke the old scheme.
fn synth_path(i: usize) -> String {
    let module = i / 400;
    let component = (i / 20) % 20;
    format!("src/module_{module}/component_{component}/file_{i}.rs")
}

const SIM_THRESHOLD: f32 = 0.85;
const MAX_RETRIEVAL_SAMPLES: usize = 300;

fn run_scale(k: usize, seed: u64) -> (f32, f32, f32) {
    let mut content_rng = StdRng::seed_from_u64(seed);

    let paths: Vec<String> = (0..k).map(synth_path).collect();
    let contents: Vec<[f32; VECTOR_DIM]> = (0..k).map(|_| random_unit_vector(&mut content_rng)).collect();

    let mut global = [0.0f32; VECTOR_DIM];
    for (path, content) in paths.iter().zip(contents.iter()) {
        let key = path_key(path);
        let bound = circular_convolution(&key, content);
        for i in 0..VECTOR_DIM {
            global[i] += bound[i];
        }
    }

    // Sample which items to test retrieval on (deterministic, evenly spread).
    let sample_n = k.min(MAX_RETRIEVAL_SAMPLES);
    let stride = (k / sample_n).max(1);

    let mut hits = 0usize;
    let mut sims = Vec::with_capacity(sample_n);
    let mut tested = 0usize;
    let mut idx = 0;
    while idx < k && tested < sample_n {
        let key = path_key(&paths[idx]);
        let candidate = circular_correlation(&key, &global);
        let sim = cosine_similarity(&candidate, &contents[idx]);
        sims.push(sim);
        if sim > SIM_THRESHOLD {
            hits += 1;
        }
        tested += 1;
        idx += stride;
    }

    let accuracy = hits as f32 / tested as f32 * 100.0;
    let mean_sim = sims.iter().sum::<f32>() / sims.len() as f32;
    let min_sim = sims.iter().cloned().fold(f32::INFINITY, f32::min);
    (accuracy, mean_sim, min_sim)
}

fn main() {
    println!("HRR superposition retrieval benchmark — daemon::neural_math (post path_key fix)");
    println!("Similarity threshold (README's Page-Fault trigger): {SIM_THRESHOLD}");
    println!("VECTOR_DIM = {VECTOR_DIM}\n");
    println!("| Chunks (k) | Retrieval Accuracy (%) | Mean cos-sim | Min cos-sim |");
    println!("|:---|:---:|:---:|:---:|");

    // Fine-grained small-k points to find where accuracy actually breaks
    // down, plus the original table's scale points, plus 30,000 -- the
    // exact number the README's SNR sentence claims works.
    for &k in &[1usize, 2, 5, 10, 20, 50, 100, 200, 2_000, 10_000, 20_000, 30_000] {
        let (accuracy, mean_sim, min_sim) = run_scale(k, 0xC0FFEE ^ k as u64);
        println!("| {k} | {accuracy:.1}% | {mean_sim:.3} | {min_sim:.3} |");
    }
}
