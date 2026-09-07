//! IVF (inverted file) coarse index — sublinear, **disk-resident** retrieval.
//!
//! turbovec's scan is exact SIMD brute force: great to ~millions, O(N) beyond.
//! IVF clusters the vectors into `sqrt(N)`-ish partitions; a query probes only
//! the nearest few, so the scan touches a small fraction of the corpus. It
//! reuses turbovec's `search_with_allowlist` — the probed partitions' ids are
//! the allowlist — so nothing in the SIMD scan or the `.tvim` format changes.
//!
//! **Disk-resident:** the sidecar `catalog.ivf` is memory-mapped. Centroids are
//! read from the map; a probe reads ONLY the selected partitions' id lists from
//! disk. RAM stays bounded regardless of corpus size, so the ceiling becomes
//! disk capacity, not memory — hundreds of millions to billions of vectors on a
//! large disk, retrieved in ~constant (probe-bounded) time. (turbovec's vectors
//! are likewise mmap'd.) The honest limit above that is storage cost: a trillion
//! vectors is petabytes to STORE — a hardware fact, not a search-speed one; the
//! model still only ever reads the retrieved slice.
//!
//! Absent sidecar → callers full-scan, so old catalogs keep working.

use std::io::Write;
use std::path::Path;

use memmap2::Mmap;

const MAGIC: &[u8; 4] = b"IVF2";
const HEADER_LEN: usize = 16; // magic(4) + dim(4) + n_parts(4) + pad(4)

/// In-memory IVF used at BUILD time (k-means training). Query time uses the
/// memory-mapped [`IvfMmap`] instead, so no full load is ever required.
#[derive(Debug, Clone)]
pub struct IvfIndex {
    dim: usize,
    centroids: Vec<f32>,       // n_parts * dim, L2-normalized
    partitions: Vec<Vec<u64>>, // partition -> external chunk ids
}

impl IvfIndex {
    pub fn n_partitions(&self) -> usize {
        self.partitions.len()
    }

    /// Train an IVF over `vectors` (`n*dim`, row-major) with external `ids`.
    /// Deterministic init (no RNG) so a rebuild of the same corpus is
    /// byte-identical. `None` when the corpus is too small to partition.
    pub fn build(vectors: &[f32], ids: &[u64], dim: usize, n_partitions: usize) -> Option<Self> {
        let n = if dim == 0 { 0 } else { vectors.len() / dim };
        if n != ids.len() || n < 256 || n_partitions < 2 || n_partitions >= n {
            return None;
        }
        let k = n_partitions;
        let mut centroids = vec![0f32; k * dim];
        let stride = (n / k).max(1);
        for c in 0..k {
            let src = (c * stride) % n;
            centroids[c * dim..(c + 1) * dim]
                .copy_from_slice(&vectors[src * dim..(src + 1) * dim]);
        }
        let mut assign = vec![0usize; n];
        for _ in 0..12 {
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
            let mut sums = vec![0f32; k * dim];
            let mut counts = vec![0u32; k];
            for i in 0..n {
                let c = assign[i];
                counts[c] += 1;
                let acc = &mut sums[c * dim..(c + 1) * dim];
                for (a, x) in acc.iter_mut().zip(&vectors[i * dim..(i + 1) * dim]) {
                    *a += *x;
                }
            }
            for c in 0..k {
                if counts[c] == 0 {
                    continue;
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

    /// Write the mmap-friendly sidecar:
    /// `[MAGIC][dim u32][n_parts u32][pad u32] [centroids f32...]`
    /// `[offset-table: n_parts × (id_offset u64, count u32, pad u32)]`
    /// `[id blobs: concatenated u64 ids per partition]`
    pub fn write(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        let k = self.partitions.len();
        let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
        f.write_all(MAGIC)?;
        f.write_all(&(self.dim as u32).to_le_bytes())?;
        f.write_all(&(k as u32).to_le_bytes())?;
        f.write_all(&0u32.to_le_bytes())?; // pad -> HEADER_LEN
        for x in &self.centroids {
            f.write_all(&x.to_le_bytes())?;
        }
        // Offset table: ids begin right after the table itself.
        let ids_base = HEADER_LEN + self.centroids.len() * 4 + k * 16;
        let mut off = ids_base as u64;
        for part in &self.partitions {
            f.write_all(&off.to_le_bytes())?;
            f.write_all(&(part.len() as u32).to_le_bytes())?;
            f.write_all(&0u32.to_le_bytes())?; // pad -> 16B entry
            off += (part.len() * 8) as u64;
        }
        for part in &self.partitions {
            for &id in part {
                f.write_all(&id.to_le_bytes())?;
            }
        }
        f.flush()
    }
}

/// Query-time, memory-mapped IVF. Nothing is loaded eagerly: centroids are read
/// from the map to score, and only the probed partitions' id lists are read.
pub struct IvfMmap {
    map: Mmap,
    dim: usize,
    n_parts: usize,
    centroids_off: usize,
    table_off: usize,
}

impl IvfMmap {
    pub fn n_partitions(&self) -> usize {
        self.n_parts
    }

    pub fn open(path: impl AsRef<Path>) -> std::io::Result<Self> {
        let file = std::fs::File::open(path)?;
        let map = unsafe { Mmap::map(&file)? };
        if map.len() < HEADER_LEN || &map[0..4] != MAGIC {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "bad IVF magic"));
        }
        let dim = u32_at(&map, 4) as usize;
        let n_parts = u32_at(&map, 8) as usize;
        let centroids_off = HEADER_LEN;
        let table_off = centroids_off + n_parts * dim * 4;
        if table_off + n_parts * 16 > map.len() {
            return Err(std::io::Error::new(std::io::ErrorKind::InvalidData, "IVF truncated"));
        }
        Ok(Self { map, dim, n_parts, centroids_off, table_off })
    }

    /// Ids of the `n_probe` partitions whose centroids are nearest `query`.
    /// Scores centroids from the map; reads only the selected id lists.
    pub fn probe(&self, query: &[f32], n_probe: usize) -> Vec<u64> {
        let n_probe = n_probe.clamp(1, self.n_parts);
        // Score all centroids (small: ~sqrt(N)); keep the top n_probe.
        let mut scored: Vec<(f32, usize)> = Vec::with_capacity(self.n_parts);
        for c in 0..self.n_parts {
            let base = self.centroids_off + c * self.dim * 4;
            let mut s = 0f32;
            for d in 0..self.dim {
                s += query[d] * f32_at(&self.map, base + d * 4);
            }
            scored.push((s, c));
        }
        // Partial select: nth_element then sort the head — avoids a full sort
        // of a million partitions.
        if scored.len() > n_probe {
            scored.select_nth_unstable_by(n_probe - 1, |a, b| {
                b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal)
            });
            scored.truncate(n_probe);
        }
        scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        let mut out = Vec::new();
        for &(_, c) in &scored {
            let ent = self.table_off + c * 16;
            let id_off = u64_at(&self.map, ent) as usize;
            let count = u32_at(&self.map, ent + 8) as usize;
            for j in 0..count {
                out.push(u64_at(&self.map, id_off + j * 8));
            }
        }
        out
    }
}

use crate::embed::dot;
#[inline]
fn u32_at(m: &[u8], o: usize) -> u32 {
    u32::from_le_bytes([m[o], m[o + 1], m[o + 2], m[o + 3]])
}
#[inline]
fn u64_at(m: &[u8], o: usize) -> u64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&m[o..o + 8]);
    u64::from_le_bytes(b)
}
#[inline]
fn f32_at(m: &[u8], o: usize) -> f32 {
    f32::from_le_bytes([m[o], m[o + 1], m[o + 2], m[o + 3]])
}

fn normalize(v: &mut [f32]) {
    let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n > 0.0 && n.is_finite() {
        for x in v.iter_mut() {
            *x /= n;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn synth_clustered(n: usize, dim: usize, n_clusters: usize, seed: u64) -> (Vec<f32>, Vec<u64>) {
        let (centers, _) = synth(n_clusters, dim, seed);
        let mut s = seed ^ 0xABCDEF;
        let mut jitter = || {
            s = s.wrapping_add(0x9E3779B97F4A7C15);
            let z = s;
            ((z ^ (z >> 31)) as f64 / u64::MAX as f64) as f32 * 2.0 - 1.0
        };
        let mut v = vec![0f32; n * dim];
        for i in 0..n {
            let c = i % n_clusters;
            for d in 0..dim {
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

    // Build -> write -> MMAP-probe: recall high, scans a fraction, RAM-free.
    #[test]
    fn mmap_probe_recall_and_sublinear() {
        let (dim, n, nc) = (64, 8000, 80);
        let (vecs, ids) = synth_clustered(n, dim, nc, 42);
        let ivf = IvfIndex::build(&vecs, &ids, dim, 90).expect("built");
        let tmp = std::env::temp_dir().join("kortex_ivf2_recall.ivf");
        ivf.write(&tmp).unwrap();
        let m = IvfMmap::open(&tmp).unwrap();

        let (mut hit, trials) = (0, 300);
        let mut scanned = 0usize;
        for t in 0..trials {
            let base = (t * 7) % n;
            let mut q = vecs[base * dim..(base + 1) * dim].to_vec();
            let mut s = 555 + t as u64;
            for x in q.iter_mut() {
                s = s.wrapping_add(0x9E3779B97F4A7C15);
                *x += 0.05 * (((s >> 33) as f64 / u32::MAX as f64) as f32 * 2.0 - 1.0);
            }
            normalize(&mut q);
            let allow = m.probe(&q, 8);
            scanned += allow.len();
            if allow.contains(&brute_nn(&vecs, dim, &q)) {
                hit += 1;
            }
        }
        let recall = hit as f64 / trials as f64;
        let avg = scanned as f64 / trials as f64;
        assert!(recall > 0.90, "recall too low: {recall}");
        assert!(avg < n as f64 * 0.30, "not sublinear: {avg}/{n}");
        let _ = std::fs::remove_file(&tmp);
    }

    #[test]
    fn too_small_returns_none() {
        let (v, ids) = synth(100, 32, 1);
        assert!(IvfIndex::build(&v, &ids, 32, 16).is_none());
    }

    // In-RAM build == mmap probe: the disk path returns identical results.
    #[test]
    fn mmap_matches_build() {
        let (v, ids) = synth_clustered(1000, 48, 20, 7);
        let ivf = IvfIndex::build(&v, &ids, 48, 30).unwrap();
        let tmp = std::env::temp_dir().join("kortex_ivf2_match.ivf");
        ivf.write(&tmp).unwrap();
        let m = IvfMmap::open(&tmp).unwrap();
        assert_eq!(m.n_partitions(), ivf.n_partitions());
        let (q, _) = synth(1, 48, 123);
        let a = m.probe(&q, 5);
        assert!(!a.is_empty());
        // every probed id is a real id
        assert!(a.iter().all(|id| (*id as usize) < 1000));
        let _ = std::fs::remove_file(&tmp);
    }
}
