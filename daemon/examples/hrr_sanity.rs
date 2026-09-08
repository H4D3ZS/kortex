//! Minimal sanity check: does bind -> unbind even round-trip for ONE item,
//! with no superposition at all? If this doesn't recover a high cosine
//! similarity, the bug is in circular_convolution/circular_correlation
//! themselves, not in superposition capacity.
use daemon::neural_math::{circular_convolution, circular_correlation, path_key, VECTOR_DIM};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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
    for x in v.iter_mut() { *x /= mag; }
    v
}

fn cosine_similarity(a: &[f32; VECTOR_DIM], b: &[f32; VECTOR_DIM]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    dot / (na * nb)
}

fn main() {
    let mut rng = StdRng::seed_from_u64(1);
    let key = path_key("src/a.rs");
    let content = random_unit_vector(&mut rng);

    let key_mag: f32 = key.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("key magnitude (should be ~1.0): {key_mag}");

    let bound = circular_convolution(&key, &content);
    let bound_mag: f32 = bound.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("bound magnitude: {bound_mag}");

    let recovered = circular_correlation(&key, &bound);
    let recovered_mag: f32 = recovered.iter().map(|x| x * x).sum::<f32>().sqrt();
    println!("recovered magnitude: {recovered_mag}");

    let sim = cosine_similarity(&recovered, &content);
    println!("cos-sim(recovered, original content) = {sim}  (should be close to 1.0 for a correct single round-trip)");

    // Also try correlation(bound, key) -- order swapped, in case the
    // convolution/correlation index convention is asymmetric.
    let recovered2 = circular_correlation(&bound, &key);
    let sim2 = cosine_similarity(&recovered2, &content);
    println!("cos-sim with args swapped = {sim2}");
}
