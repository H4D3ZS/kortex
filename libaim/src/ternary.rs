//! Ternary sign codes for coarse-to-fine retrieval.
//!
//! A vector quantised to {-1, 0, +1} scores with **no multiplies** — the dot
//! product becomes popcounts over packed sign bitmasks (the Rhind-papyrus /
//! BitNet move). It is ~2.7x faster and 16x smaller than the f32 kernel, but
//! sign quantisation is lossy, so it is used ONLY as a coarse pre-filter: the
//! ternary scan narrows a large candidate set, then the exact f32 cosine
//! reranks the survivors. Final scores are always exact f32.
//!
//! See `tools/ternary-dot/` for the measured recall/speed characterisation
//! (top-50 holds ~0.998 of the exact top-10 on realistic anisotropic
//! embeddings; the worst-case isotropic distribution fails, which is why this
//! is a filter and not a scorer).

/// Packed ternary code: one bit per dimension in each of `pos`/`neg`.
#[derive(Debug, Clone)]
pub struct TernaryVec {
    pos: Box<[u64]>,
    neg: Box<[u64]>,
    dim: usize,
}

/// Threshold below which a coordinate is quantised to 0. Unit vectors have
/// per-coordinate RMS ~ `1/sqrt(dim)`, so the cutoff is a multiple of that;
/// `alpha` ~ 0.3 keeps the signal-carrying dimensions and zeros the noise.
pub fn tau_for(dim: usize, alpha: f32) -> f32 {
    if dim == 0 { 0.0 } else { alpha / (dim as f32).sqrt() }
}

impl TernaryVec {
    /// Quantise an f32 vector: `+1` where `v_i > tau`, `-1` where `v_i < -tau`,
    /// else `0`.
    pub fn from_f32(v: &[f32], tau: f32) -> Self {
        let dim = v.len();
        let words = dim.div_ceil(64);
        let mut pos = vec![0u64; words];
        let mut neg = vec![0u64; words];
        for (i, &x) in v.iter().enumerate() {
            if x > tau {
                pos[i >> 6] |= 1u64 << (i & 63);
            } else if x < -tau {
                neg[i >> 6] |= 1u64 << (i & 63);
            }
        }
        Self { pos: pos.into_boxed_slice(), neg: neg.into_boxed_slice(), dim }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Ternary dot: (agreements - disagreements) over nonzero positions. Pure
    /// bit ops — no multiplies. Larger is more similar, same ordering sense as
    /// cosine on unit vectors.
    #[inline]
    pub fn dot(&self, other: &TernaryVec) -> i32 {
        let n = self.pos.len().min(other.pos.len());
        let (mut agree, mut disagree) = (0u32, 0u32);
        for w in 0..n {
            agree += (self.pos[w] & other.pos[w]).count_ones()
                + (self.neg[w] & other.neg[w]).count_ones();
            disagree += (self.pos[w] & other.neg[w]).count_ones()
                + (self.neg[w] & other.pos[w]).count_ones();
        }
        agree as i32 - disagree as i32
    }

    /// Number of nonzero (kept) dimensions — diagnostic.
    pub fn nnz(&self) -> u32 {
        self.pos.iter().map(|w| w.count_ones()).sum::<u32>()
            + self.neg.iter().map(|w| w.count_ones()).sum::<u32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encodes_signs_with_threshold() {
        let v = [0.5, -0.5, 0.01, -0.01, 0.0];
        let t = TernaryVec::from_f32(&v, 0.1);
        // only the two large-magnitude coords survive
        assert_eq!(t.nnz(), 2);
    }

    #[test]
    fn dot_counts_agreements_minus_disagreements() {
        // a = [+, -, +, 0], b = [+, +, -, 0]
        let a = TernaryVec::from_f32(&[1.0, -1.0, 1.0, 0.0], 0.5);
        let b = TernaryVec::from_f32(&[1.0, 1.0, -1.0, 0.0], 0.5);
        // pos0 agree (+1); pos1 a=-,b=+ disagree (-1); pos2 a=+,b=- disagree (-1)
        assert_eq!(a.dot(&b), 1 - 2);
    }

    #[test]
    fn identical_vectors_score_their_nnz() {
        let v = [0.6, -0.7, 0.0, 0.8, -0.9];
        let t = TernaryVec::from_f32(&v, 0.1);
        assert_eq!(t.dot(&t), t.nnz() as i32); // all agreements
    }

    #[test]
    fn ternary_ranking_tracks_cosine_on_anisotropic_data() {
        // Coarse-to-fine premise: on realistic (anisotropic) unit vectors, the
        // ternary top-N must contain the exact nearest neighbour. Build a query,
        // its near twin, and distractors; the twin must win under ternary too.
        let dim = 256;
        let mut s = 1234u64;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let scale: Vec<f32> = (0..dim).map(|d| 1.0 / ((d as f32 + 1.0).powf(0.7))).collect();
        let norm = |v: &mut Vec<f32>| {
            let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if n > 0.0 { for x in v.iter_mut() { *x /= n; } }
        };
        let mut q: Vec<f32> = (0..dim).map(|d| rnd() * scale[d]).collect();
        norm(&mut q);
        let mut twin = q.clone();
        for (i, x) in twin.iter_mut().enumerate() { *x += 0.1 * rnd() * scale[i]; }
        norm(&mut twin);
        let tau = tau_for(dim, 0.3);
        let (qt, tt) = (TernaryVec::from_f32(&q, tau), TernaryVec::from_f32(&twin, tau));
        let twin_score = qt.dot(&tt);
        // 40 distractors must all score below the twin under ternary
        let mut worst_beats = 0;
        for _ in 0..40 {
            let mut d: Vec<f32> = (0..dim).map(|k| rnd() * scale[k]).collect();
            norm(&mut d);
            if qt.dot(&TernaryVec::from_f32(&d, tau)) >= twin_score {
                worst_beats += 1;
            }
        }
        assert_eq!(worst_beats, 0, "a distractor out-scored the near twin under ternary");
    }
}
