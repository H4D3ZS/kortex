/// Neural Math Engine for Holographic Reduced Representations (HRR) and Surprise Analysis
/// Part of the .aim Sentient Singularity (Phase 6)
// Removed unused PI constant

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use rustfft::{num_complex::Complex32, FftPlanner};
use sha3::{Digest, Sha3_256};

pub const VECTOR_DIM: usize = 1536;

/// Deterministic HRR path key: SHA3-256(path) seeds the construction of a
/// *unitary* HRR vector (see `unitary_vector` below).
///
/// This replaces an earlier `sin(byte * sin(i))` scheme. That scheme derived
/// each key directly from the path's own bytes at each index, so two paths
/// sharing a prefix (`src/foo/a.rs`, `src/foo/b.rs`) produced strongly
/// correlated keys — exactly the files most likely to be retrieved together,
/// and exactly where HRR unbinding needs keys to be close to orthogonal.
/// Hashing first destroys that correlation: a one-bit path difference flips
/// roughly half the hash's output bits (avalanche effect), so the seeded
/// construction for any two distinct paths is independent regardless of how
/// similar the paths look as strings.
pub fn path_key(path: &str) -> [f32; VECTOR_DIM] {
    let digest = Sha3_256::digest(path.as_bytes());
    let seed: [u8; 32] = digest.into();
    unitary_vector(seed)
}

/// Construct a *unitary* HRR vector: every DFT bin has magnitude exactly 1
/// and a random phase (subject to the conjugate symmetry a real signal's
/// spectrum must have). This is the textbook HRR key construction (Plate
/// 1995; the Fourier-domain "FHRR" variant that Schlegel et al. 2022 —
/// already cited in this project's whitepaper — describes) — NOT the same
/// thing as a random Gaussian vector normalized to unit L2 norm. A random
/// Gaussian vector's spectrum has magnitudes that vary bin to bin
/// (Rayleigh-distributed), so correlating with it as a key does not cleanly
/// invert convolution even for a single bound item with zero interference
/// from any other item — verified empirically (`examples/hrr_sanity.rs`):
/// a Gaussian key round-trips a single bind/unbind at cos-sim ~0.70, not
/// ~1.0. A flat unit-magnitude spectrum is what makes bind/unbind exact
/// inverses: convolution multiplies spectra (`Bound[k] = Key[k]*Content[k]`);
/// correlating with the same key multiplies again by the key's magnitude
/// *squared*, which is exactly 1 at every bin for a unitary key — recovering
/// `Content[k]` exactly, independent of what Content's own spectrum looks
/// like. This is what makes the superposition capacity argument in the
/// whitepaper's SNR formula apply at all: it assumes clean per-item
/// recovery with noise coming only from *other* superposed items, which
/// requires unitary keys to begin with.
fn unitary_vector(seed: [u8; 32]) -> [f32; VECTOR_DIM] {
    let mut rng = StdRng::from_seed(seed);
    let d = VECTOR_DIM;
    let half = d / 2; // VECTOR_DIM is even, so a real Nyquist bin exists.

    // Conjugate-symmetric, unit-magnitude spectrum: DC (k=0) and Nyquist
    // (k=half) bins are real-valued (magnitude 1, random sign) since a real
    // time-domain signal's spectrum must be conjugate-symmetric; every other
    // bin gets a random phase, mirrored at d-k as its complex conjugate.
    let mut spectrum = vec![Complex32::new(0.0, 0.0); d];
    spectrum[0] = Complex32::new(if rng.gen_bool(0.5) { 1.0 } else { -1.0 }, 0.0);
    spectrum[half] = Complex32::new(if rng.gen_bool(0.5) { 1.0 } else { -1.0 }, 0.0);
    for k in 1..half {
        let theta = rng.gen_range(0.0..std::f32::consts::TAU);
        let c = Complex32::new(theta.cos(), theta.sin());
        spectrum[k] = c;
        spectrum[d - k] = c.conj();
    }

    // VECTOR_DIM = 1536 = 2^9 * 3 factors cleanly, so rustfft's mixed-radix
    // planner is fast here (this replaced an O(d^2) trig-heavy direct
    // inverse-DFT loop that made key generation ~18ms/call — fine for a
    // one-off sanity check, but a real cost multiplied by every chunk
    // indexed in production, and by every item at every scale in the
    // benchmark below).
    let mut planner = FftPlanner::<f32>::new();
    planner.plan_fft_inverse(d).process(&mut spectrum);

    // rustfft's inverse transform is unnormalized by convention (Parseval
    // energy scales by d); divide by d for the true real inverse DFT value.
    let mut v = [0.0f32; VECTOR_DIM];
    let scale = 1.0 / d as f32;
    for (i, c) in spectrum.iter().enumerate() {
        v[i] = c.re * scale;
    }

    // Parseval's theorem already guarantees ~unit L2 norm for a
    // unit-magnitude spectrum; this only cleans up floating-point drift.
    let mag: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag > 1e-6 {
        for x in v.iter_mut() {
            *x /= mag;
        }
    }
    v
}

/// Performs Circular Convolution using the Fast Fourier Transform (FFT) equivalent.
/// This "smears" features of vector B into vector A to create a holographic bind.
pub fn circular_convolution(a: &[f32; VECTOR_DIM], b: &[f32; VECTOR_DIM]) -> [f32; VECTOR_DIM] {
    let mut result = [0.0; VECTOR_DIM];
    
    // Naive Circular Convolution (O(N^2)) - To be optimized with FFT in production
    for i in 0..VECTOR_DIM {
        let mut sum = 0.0;
        for j in 0..VECTOR_DIM {
            let k = (i + VECTOR_DIM - j) % VECTOR_DIM;
            sum += a[j] * b[k];
        }
        result[i] = sum;
    }
    
    result
}

/// Calculate the "Surprise" (Loss) of new information compared to the existing memory.
/// MIRAS Framework: Loss = ||V_new - V_old||^2
pub fn calculate_surprise(current: &[f32], incoming: &[f32; VECTOR_DIM]) -> f32 {
    let mut sum_sq_diff = 0.0;
    for i in 0..VECTOR_DIM {
        let diff = incoming[i] - current[i];
        sum_sq_diff += diff * diff;
    }
    sum_sq_diff.sqrt()
}

/// Circular Correlation (Inverse of Convolution) used for Unbinding / Retrieval
pub fn circular_correlation(a: &[f32; VECTOR_DIM], b: &[f32; VECTOR_DIM]) -> [f32; VECTOR_DIM] {
    let mut result = [0.0; VECTOR_DIM];
    for i in 0..VECTOR_DIM {
        let mut sum = 0.0;
        for j in 0..VECTOR_DIM {
            let k = (i + j) % VECTOR_DIM;
            sum += a[j] * b[k];
        }
        result[i] = sum;
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cosine_similarity(a: &[f32; VECTOR_DIM], b: &[f32; VECTOR_DIM]) -> f32 {
        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
        dot / (na * nb)
    }

    #[test]
    fn path_key_is_unit_length() {
        let k = path_key("src/lib.rs");
        let mag: f32 = k.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((mag - 1.0).abs() < 1e-3, "expected unit length, got {mag}");
    }

    #[test]
    fn single_item_bind_unbind_round_trips_cleanly() {
        // A unitary key must recover a single bound item almost exactly --
        // the property the old sin-based / plain-normalized-Gaussian keys
        // lacked (measured ~0.70 cos-sim instead of ~1.0; see
        // examples/hrr_sanity.rs for the standalone repro).
        let key = path_key("src/a.rs");
        let content = path_key("some/other/content/seed.rs"); // any unit vector will do here
        let bound = circular_convolution(&key, &content);
        let recovered = circular_correlation(&key, &bound);
        let sim = cosine_similarity(&recovered, &content);
        assert!(sim > 0.999, "expected near-exact single-item recovery, got cos-sim {sim}");
    }

    #[test]
    fn similar_paths_produce_near_orthogonal_keys() {
        // The crosstalk bug: the old scheme derived the key from the path's
        // own bytes at each index, so `src/foo/a.rs` and `src/foo/b.rs`
        // (identical for all but the last few bytes) produced strongly
        // correlated keys -- interference concentrated exactly among the
        // files most likely to be retrieved together. Hashing first must
        // make these close to independent regardless of shared prefixes.
        let a = path_key("src/foo/component/a.rs");
        let b = path_key("src/foo/component/b.rs");
        let sim = cosine_similarity(&a, &b).abs();
        assert!(sim < 0.15, "expected near-orthogonal keys for sibling paths, got |cos-sim| {sim}");
    }
}
