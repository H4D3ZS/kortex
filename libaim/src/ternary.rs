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

// ─────────────────────────────────────────────────────────────────────────────
// Ternary-weight GEMV — the arithmetic root (BitNet b1.58).
//
// The retrieval kernel above quantises *both* sides to signs. The bigger prize
// is quantising the *weights* of a linear layer to {-1, 0, +1} while the
// activations stay f32: then `y = W·x` has **no weight×activation multiply** —
// each weight only decides whether to add, subtract, or skip an activation, and
// a single per-row scale (absmean) is applied at the end. That is the actual
// unit-cost-of-compute change the whole "solve compute cost" thesis points at:
// a MAC (multiply-accumulate, the dominant FLOP and the thing GPUs are sized
// around) becomes an add/subtract.
//
// This is the deployable kernel groundwork. It runs and benchmarks now; using
// it inside the LLM's matmuls requires a **ternary-trained** model (BitNet-class
// or QAT) — post-hoc rounding a normal model's weights to ternary collapses
// quality (the same lesson as 2.5-bit Escha). The kernel is model-agnostic;
// what's gated is the weights.
// ─────────────────────────────────────────────────────────────────────────────

/// A row-major linear layer whose weights are ternary ({-1,0,+1}) with a
/// per-row f32 scale — the BitNet b1.58 representation. ~16x smaller than f32
/// (2 bits/weight packed) and its GEMV is multiply-free but for one scale per
/// output.
#[derive(Debug, Clone)]
pub struct TernaryMatrix {
    rows: usize,
    cols: usize,
    words_per_row: usize,
    /// `rows * words_per_row` packed +1 positions.
    pos: Box<[u64]>,
    /// `rows * words_per_row` packed -1 positions.
    neg: Box<[u64]>,
    /// Per-row dequant scale (absmean of the original row).
    scale: Box<[f32]>,
}

impl TernaryMatrix {
    /// Quantise a dense row-major `rows × cols` f32 weight matrix. Per BitNet
    /// b1.58: `scale = mean(|w|)` per row, then `w → round(w/scale)` clamped to
    /// {-1,0,+1} — i.e. `+1` for `w > scale/2`, `-1` for `w < -scale/2`, else 0.
    pub fn from_f32_rows(weights: &[f32], rows: usize, cols: usize) -> Self {
        assert_eq!(weights.len(), rows * cols, "weights must be rows*cols");
        let words_per_row = cols.div_ceil(64);
        let mut pos = vec![0u64; rows * words_per_row];
        let mut neg = vec![0u64; rows * words_per_row];
        let mut scale = vec![0f32; rows];
        for r in 0..rows {
            let row = &weights[r * cols..(r + 1) * cols];
            let absmean = row.iter().map(|w| w.abs()).sum::<f32>() / cols.max(1) as f32;
            scale[r] = absmean;
            let tau = absmean * 0.5;
            let base = r * words_per_row;
            for (c, &w) in row.iter().enumerate() {
                if w > tau {
                    pos[base + (c >> 6)] |= 1u64 << (c & 63);
                } else if w < -tau {
                    neg[base + (c >> 6)] |= 1u64 << (c & 63);
                }
            }
        }
        Self {
            rows,
            cols,
            words_per_row,
            pos: pos.into_boxed_slice(),
            neg: neg.into_boxed_slice(),
            scale: scale.into_boxed_slice(),
        }
    }

    pub fn rows(&self) -> usize {
        self.rows
    }
    pub fn cols(&self) -> usize {
        self.cols
    }

    /// `y = W · x`, multiply-free: for each output row, add the activations at
    /// `+1` weights, subtract those at `-1`, then apply the one row scale.
    pub fn matvec(&self, x: &[f32], y: &mut [f32]) {
        assert_eq!(x.len(), self.cols, "x len must be cols");
        assert_eq!(y.len(), self.rows, "y len must be rows");
        for r in 0..self.rows {
            let base = r * self.words_per_row;
            let mut acc = 0f32;
            for w in 0..self.words_per_row {
                let col0 = w * 64;
                let mut p = self.pos[base + w];
                while p != 0 {
                    let b = p.trailing_zeros() as usize;
                    acc += x[col0 + b];
                    p &= p - 1;
                }
                let mut n = self.neg[base + w];
                while n != 0 {
                    let b = n.trailing_zeros() as usize;
                    acc -= x[col0 + b];
                    n &= n - 1;
                }
            }
            y[r] = acc * self.scale[r];
        }
    }

    /// Convenience allocating form of [`matvec`].
    pub fn matvec_vec(&self, x: &[f32]) -> Vec<f32> {
        let mut y = vec![0f32; self.rows];
        self.matvec(x, &mut y);
        y
    }

    /// Bytes held by the packed weights (excludes the small per-row scale). The
    /// f32 original is `rows*cols*4`; this is `rows*words_per_row*16`.
    pub fn packed_bytes(&self) -> usize {
        self.pos.len() * 8 + self.neg.len() * 8
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

    // ── Ternary-weight GEMV ──────────────────────────────────────────────────

    /// The multiply-free matvec must equal the *dequantized ternary* reference
    /// exactly in intent: y_r = scale_r * Σ sign(w_rc)·x_c. We compute the
    /// reference the same summation order and allow only float-rounding slack.
    fn ternary_ref(weights: &[f32], rows: usize, cols: usize, x: &[f32]) -> Vec<f32> {
        let mut y = vec![0f32; rows];
        for r in 0..rows {
            let row = &weights[r * cols..(r + 1) * cols];
            let absmean = row.iter().map(|w| w.abs()).sum::<f32>() / cols as f32;
            let tau = absmean * 0.5;
            let mut acc = 0f32;
            // positives first, then negatives — matches matvec's word/bit order
            for (c, &w) in row.iter().enumerate() {
                if w > tau {
                    acc += x[c];
                }
            }
            for (c, &w) in row.iter().enumerate() {
                if w < -tau {
                    acc -= x[c];
                }
            }
            y[r] = acc * absmean;
        }
        y
    }

    #[test]
    fn gemv_matches_reference() {
        let (rows, cols) = (5, 130); // >2 words per row, non-multiple of 64
        let mut s = 99u64;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        let weights: Vec<f32> = (0..rows * cols).map(|_| rnd()).collect();
        let x: Vec<f32> = (0..cols).map(|_| rnd()).collect();

        let m = TernaryMatrix::from_f32_rows(&weights, rows, cols);
        let y = m.matvec_vec(&x);
        let yref = ternary_ref(&weights, rows, cols, &x);
        assert_eq!(y.len(), rows);
        for (a, b) in y.iter().zip(&yref) {
            assert!((a - b).abs() <= 1e-4 * (1.0 + b.abs()), "got {a}, want {b}");
        }
    }

    #[test]
    fn gemv_is_multiply_free_shape_and_savings() {
        let (rows, cols) = (8, 256);
        let weights = vec![0.7f32; rows * cols]; // all +1 after quant
        let m = TernaryMatrix::from_f32_rows(&weights, rows, cols);
        let x = vec![2.0f32; cols];
        let y = m.matvec_vec(&x);
        // every weight is +1, scale = 0.7 → y_r = 0.7 * (256 * 2.0)
        for v in &y {
            assert!((v - 0.7 * 512.0).abs() < 1e-2);
        }
        // packed is far smaller than the f32 original
        let f32_bytes = rows * cols * 4;
        assert!(m.packed_bytes() * 6 < f32_bytes, "expected big memory win");
    }

    #[test]
    fn gemv_zero_row_yields_zero() {
        let (rows, cols) = (2, 64);
        let mut weights = vec![0f32; rows * cols];
        // row 1 nonzero, row 0 all zero
        for c in 0..cols {
            weights[cols + c] = if c % 2 == 0 { 1.0 } else { -1.0 };
        }
        let m = TernaryMatrix::from_f32_rows(&weights, rows, cols);
        let x = vec![1.0f32; cols];
        let y = m.matvec_vec(&x);
        assert_eq!(y[0], 0.0, "all-zero weight row → zero output");
    }

    #[test]
    fn gemv_approximates_full_precision_on_structured_weights() {
        // On a low-rank/structured weight, ternary keeps the dominant direction,
        // so the output correlates strongly with the exact f32 matvec.
        let (rows, cols) = (16, 128);
        let mut s = 7u64;
        let mut rnd = || {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
        };
        // rank-1-ish: w[r][c] = a[r]*b[c] + small noise
        let a: Vec<f32> = (0..rows).map(|_| rnd()).collect();
        let b: Vec<f32> = (0..cols).map(|_| rnd()).collect();
        let mut weights = vec![0f32; rows * cols];
        for r in 0..rows {
            for c in 0..cols {
                weights[r * cols + c] = a[r] * b[c] + 0.05 * rnd();
            }
        }
        let x: Vec<f32> = (0..cols).map(|_| rnd()).collect();

        // exact f32 matvec
        let mut exact = vec![0f32; rows];
        for r in 0..rows {
            let mut acc = 0f32;
            for c in 0..cols {
                acc += weights[r * cols + c] * x[c];
            }
            exact[r] = acc;
        }
        let approx = TernaryMatrix::from_f32_rows(&weights, rows, cols).matvec_vec(&x);

        // Pearson correlation between exact and ternary outputs must be high.
        let mean = |v: &[f32]| v.iter().sum::<f32>() / v.len() as f32;
        let (me, ma) = (mean(&exact), mean(&approx));
        let mut num = 0f32;
        let (mut de, mut da) = (0f32, 0f32);
        for r in 0..rows {
            let (ce, ca) = (exact[r] - me, approx[r] - ma);
            num += ce * ca;
            de += ce * ce;
            da += ca * ca;
        }
        let corr = num / (de.sqrt() * da.sqrt() + 1e-9);
        assert!(corr > 0.9, "ternary GEMV should track full precision, corr={corr}");
    }
}
