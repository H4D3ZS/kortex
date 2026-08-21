//! IVF (inverted file) coarse index — sublinear retrieval at scale.
//!
//! turbovec's scan is exact SIMD brute force: great to ~millions, O(N) beyond.
//! IVF clusters the vectors into `sqrt(N)`-ish partitions; a query probes only
//! the nearest few partitions, so the scan touches a small fraction of the
//! corpus. It reuses turbovec's existing `search_with_allowlist` — the probed
//! partitions' ids become the allowlist — so nothing in the SIMD scan or the
//! `.tvim` format changes. This is what turns kortex from "millions" into
//! "billions, sublinear". (True trillion-scale further needs disk-resident
//! vectors — DiskANN-class — since IVF still keeps all vectors in the index.)
//!
//! Stored as a sidecar `catalog.ivf`; absent → callers fall back to full scan,
//! so old catalogs keep working.

use std::io::{Read, Write};
use std::path::Path;

const MAGIC: &[u8; 4] = b"IVF1";

/// A trained coarse quantizer: centroids + the chunk ids in each partition.
#[derive(Debug, Clone)]
pub struct IvfIndex {
    dim: usize,
    /// `n_parts * dim` row-major centroids (L2-normalized).
    centroids: Vec<f32>,
    /// partition → external chunk ids assigned to it.
    partitions: Vec<Vec<u64>>,
}

impl IvfIndex {
    pub fn n_partitions(&self) -> usize {
        self.partitions.len()
    }

    /// Build an IVF over `vectors` (`n * dim`, row-major) with external `ids`.
    /// `n_partitions` clusters via Lloyd's k-means (deterministic init, no RNG,
    /// so a rebuild of the same corpus is byte-identical). Returns `None` when
    /// the corpus is too small to be worth partitioning (caller full-scans).
    pub fn build(vectors: &[f32], ids: &[u64], dim: usize, n_partitions: usize) -> Option<Self> {
        let n = if dim == 0 { 0 } else { vectors.len() / dim };
        if n != ids.len() || n < 256 || n_partitions < 2 || n_partitions >= n {
            return None;
        }
        let k = n_partitions;
        // Deterministic init: evenly-strided seeds across the corpus.
        let mut centroids = vec![0f32; k * dim];
        let stride = (n / k).max(1);
        for c in 0..k {
            let src = (c * stride) % n;
            centroids[c * dim..(c + 1) * dim]
                .copy_from_slice(&vectors[src * dim..(src + 1) * dim]);
        }
        let mut assign = vec![0usize; n];
        for _iter in 0..12 {
            // Assign each vector to its nearest centroid (max dot product;
            // vectors ~unit-norm so dot ≈ cosine).
            let mut changed = false;
            for i in 0..n {
                let v = &vectors[i * dim..(i + 1) * dim];
                let mut best = 0usize;
                let mut best_s = f32::NEG_INFINITY;
                for c in 0..k {
                    let s = dot(v, &centroids[c * dim..(c + 1) * dim]);
                    if s > best_s {
                        best_s = s;
                        best = c;
                    }
                }
                if assign[i] != best {
                    assign[i] = best;
                    changed = true;
                }
            }
            // Recompute centroids as the (normalized) mean of members.
            let mut sums = vec![0f32; k * dim];
            let mut counts = vec![0u32; k];
            for i in 0..n {
                let c = assign[i];
                counts[c] += 1;
                let v = &vectors[i * dim..(i + 1) * dim];
                let acc = &mut sums[c * dim..(c + 1) * dim];
                for (a, x) in acc.iter_mut().zip(v) {
                    *a += *x;
                }
            }
            for c in 0..k {
                if counts[c] == 0 {
                    continue; // keep the old centroid for an empty cluster
                }
                let acc = &mut sums[c * dim..(c + 1) * dim];
                for a in acc.iter_mut() {
                    *a /= counts[c] as f32;
                }
                normalize(acc);
                centroids[c * dim..(c + 1) * dim].copy_from_slice(acc);
            }
            if !changed {
                break;
            }
        }
        let mut partitions = vec![Vec::new(); k];
        for i in 0..n {
            partitions[assign[i]].push(ids[i]);
        }
        Some(Self { dim, centroids, partitions })
    }

    /// Return the concatenated chunk ids of the `n_probe` partitions whose
    /// centroids are nearest `query` — the allowlist to hand turbovec.
    pub fn probe(&self, query: &[f32], n_probe: usize) -> Vec<u64> {
        let k = self.partitions.len();
        let n_probe = n_probe.clamp(1, k);
        let mut scored: Vec<(f32, usize)> = (0..k)
            .map(|c| (dot(query, &self.centroids[c * self.dim..(c + 1) * self.dim]), c))
            .collect();
        // Partial order: nearest n_probe by descending centroid similarity.
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        let mut out = Vec::new();
        for &(_, c) in scored.iter().take(n_probe) {
            out.extend_from_slice(&self.partitions[c]);
        }
        out
    }

    pub fn write(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(MAGIC)?;
        f.write_all(&(self.dim as u32).to_le_bytes())?;
        f.write_all(&(self.partitions.len() as u32).to_le_bytes())?;
        for x in &self.centroids {
            f.write_all(&x.to_le_bytes())?;
        }
        for part in &self.partitions {
            f.write_all(&(part.len() as u32).to_le_bytes())?;
            for &id in part {
                f.write_all(&id.to_le_bytes())?;
            }
        }
        f.flush()
    }

    pub fn read(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let mut f = std::io::BufReader::new(std::fs::File::open(path)?);
        let mut m = [0u8; 4];
        f.read_exact(&mut m)?;
        if &m != MAGIC {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad IVF magic"));
        }
        let dim = read_u32(&mut f)? as usize;
        let k = read_u32(&mut f)? as usize;
        let mut centroids = vec![0f32; k * dim];
        let mut buf = [0u8; 4];
        for x in centroids.iter_mut() {
            f.read_exact(&mut buf)?;
            *x = f32::from_le_bytes(buf);
        }
        let mut partitions = Vec::with_capacity(k);
        for _ in 0..k {
            let cnt = read_u32(&mut f)? as usize;
            let mut ids = Vec::with_capacity(cnt);
            let mut idb = [0u8; 8];
            for _ in 0..cnt {
                f.read_exact(&mut idb)?;
                ids.push(u64::from_le_bytes(idb));
            }
            partitions.push(ids);
        }
        Ok(Self { dim, centroids, partitions })
    }
}

#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 && n.is_finite() {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

fn read_u32(f: &mut impl Read) -> std::io::Result<u32> {
    let mut b = [0u8; 4];
    f.read_exact(&mut b)?;
    Ok(u32::from_le_bytes(b))
}

#[cfg(test)]
mod tests {
    use super::*;

    // Deterministic pseudo-random unit vectors (no rng dep; splitmix64).
    fn synth(n: usize, dim: usize, seed: u64) -> (Vec<f32>, Vec<u64>) {
        let mut s = seed;
        let mut next = || {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
            ((z ^ (z >> 31)) as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0
        };
        let mut v = vec![0f32; n * dim];
        for x in v.iter_mut() {
            *x = next();
        }
        for i in 0..n {
            normalize(&mut v[i * dim..(i + 1) * dim]);
        }
        (v, (0..n as u64).collect())
    }

    // Clustered unit vectors — mimics real embeddings (similar items group),
    // which is exactly the structure IVF exploits. Purely-random vectors have
    // no clusters, so IVF cannot and should not help there.
    fn synth_clustered(n: usize, dim: usize, n_clusters: usize, seed: u64) -> (Vec<f32>, Vec<u64>) {
        let (centers, _) = synth(n_clusters, dim, seed);
        let mut s = seed ^ 0xABCDEF;
        let mut jitter = || {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let mut z = s;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
            ((z ^ (z >> 31)) as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0
        };
        let mut v = vec![0f32; n * dim];
        for i in 0..n {
            let c = i % n_clusters;
            for d in 0..dim {
                // center + small jitter -> tight clusters
                v[i * dim + d] = centers[c * dim + d] + 0.15 * jitter();
            }
            normalize(&mut v[i * dim..(i + 1) * dim]);
        }
        (v, (0..n as u64).collect())
    }

    fn brute_nn(vectors: &[f32], dim: usize, q: &[f32]) -> u64 {
        (0..vectors.len() / dim)
            .map(|i| (dot(q, &vectors[i * dim..(i + 1) * dim]), i as u64))
            .max_by(|a, b| a.0.partial_cmp(&b.0).unwrap())
            .unwrap()
            .1
    }

    #[test]
    fn probe_finds_true_nn_and_scans_a_fraction() {
        // Clustered data like real embeddings. Queries are near cluster centers.
        let (dim, n, nc) = (64, 8000, 80);
        let (vecs, ids) = synth_clustered(n, dim, nc, 42);
        let ivf = IvfIndex::build(&vecs, &ids, dim, 90).expect("built");
        let (mut hit, trials) = (0, 300);
        let mut scanned_total = 0usize;
        for t in 0..trials {
            // A query = a real member with light jitter (a realistic near-dup query).
            let base = (t * 7) % n;
            let mut q = vecs[base * dim..(base + 1) * dim].to_vec();
            let mut s = 555 + t as u64;
            for x in q.iter_mut() {
                s = s.wrapping_add(0x9E3779B97F4A7C15);
                *x += 0.05 * (((s >> 33) as f64 / u32::MAX as f64) as f32 * 2.0 - 1.0);
            }
            normalize(&mut q);
            let allow = ivf.probe(&q, 8); // probe 8 of ~90 partitions
            scanned_total += allow.len();
            let truth = brute_nn(&vecs, dim, &q);
            if allow.contains(&truth) {
                hit += 1;
            }
        }
        let recall = hit as f64 / trials as f64;
        let avg_scanned = scanned_total as f64 / trials as f64;
        // High recall on clustered data, scanning a small fraction of N.
        assert!(recall > 0.90, "recall too low: {recall}");
        assert!(avg_scanned < (n as f64) * 0.30, "not sublinear enough: {avg_scanned}/{n}");
    }

    #[test]
    fn too_small_returns_none() {
        let (v, ids) = synth(100, 32, 1);
        assert!(IvfIndex::build(&v, &ids, 32, 16).is_none());
    }

    #[test]
    fn roundtrip_serialization() {
        let (v, ids) = synth(1000, 48, 7);
        let ivf = IvfIndex::build(&v, &ids, 48, 30).unwrap();
        let tmp = std::env::temp_dir().join("kortex_ivf_test.ivf");
        ivf.write(&tmp).unwrap();
        let back = IvfIndex::read(&tmp).unwrap();
        assert_eq!(back.n_partitions(), ivf.n_partitions());
        let (q, _) = synth(1, 48, 123);
        assert_eq!(ivf.probe(&q, 5), back.probe(&q, 5));
        let _ = std::fs::remove_file(&tmp);
    }
}
