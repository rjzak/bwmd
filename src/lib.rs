// SPDX-License-Identifier: Apache-2.0

#![doc = include_str!("../readme.md")]
#![deny(clippy::all)]
#![deny(clippy::cargo)]
#![deny(clippy::pedantic)]
#![deny(missing_docs)]
#![forbid(unsafe_code)]

use std::collections::{HashMap, HashSet};

use base64::{Engine as _, engine::general_purpose};

const ALPHABET_SIZE: usize = 256;
const VEC_LEN: usize = ALPHABET_SIZE * ALPHABET_SIZE;

/// Store a sparse vector of pairs of non-zero items and cooresponding index.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct SparseVector {
    /// Map of indices to values from the original vector.
    pub pairs: HashMap<u32, f32>,

    /// Size of the original vector.
    pub size: u32,
}

impl SparseVector {
    /// Reconstruct a sparse vector from its base64-encoded representation.
    ///
    /// # Errors
    ///
    /// Returns an error if the string isn't valid base64 or if the decoded bytes aren't the expected length.
    pub fn from_b64<T: AsRef<[u8]>>(b64: T) -> Result<Self, base64::DecodeError> {
        let bytes = general_purpose::STANDARD.decode(b64)?;
        if bytes.len() % 4 != 0 {
            return Err(base64::DecodeError::InvalidLength(bytes.len()));
        }

        let mut pairs = HashMap::new();
        for chunk in bytes[0..bytes.len() - 4].chunks_exact(8) {
            let i = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
            let v = f32::from_be_bytes([chunk[4], chunk[5], chunk[6], chunk[7]]);
            pairs.insert(i, v);
        }

        let size = u32::from_be_bytes([
            bytes[bytes.len() - 4],
            bytes[bytes.len() - 3],
            bytes[bytes.len() - 2],
            bytes[bytes.len() - 1],
        ]);
        Ok(Self { pairs, size })
    }

    /// Create a new sparse vector from a dense vector.
    #[inline]
    #[must_use]
    #[allow(clippy::cast_possible_truncation)]
    pub fn from_dense(dense: &[f32]) -> Self {
        let mut pairs = HashMap::new();
        for (i, v) in dense.iter().enumerate() {
            if *v != 0.0 {
                pairs.insert(i as u32, *v);
            }
        }

        Self {
            pairs,
            size: dense.len() as u32,
        }
    }

    /// Get a dense vector representation from this sparse vector.
    #[inline]
    #[must_use]
    pub fn to_dense(&self) -> Vec<f32> {
        let mut dense = vec![0.0f32; self.size as usize];
        for (i, v) in &self.pairs {
            dense[*i as usize] = *v;
        }
        dense
    }

    /// Calculate the distance between two sparse vectors.
    #[must_use]
    pub fn distance(&self, other: &Self) -> f32 {
        if self.size != other.size {
            return 1.0;
        }

        let indices = self
            .pairs
            .keys()
            .copied()
            .chain(other.pairs.keys().copied())
            .collect::<HashSet<u32>>();
        indices
            .iter()
            .map(|i| {
                let x = self.pairs.get(i).unwrap_or(&0.0);
                let y = other.pairs.get(i).unwrap_or(&0.0);
                (x - y) * (x - y)
            })
            .sum::<f32>()
            .sqrt()
    }

    /// Get a base64-encoded representation of this sparse vector
    #[must_use]
    pub fn to_b64(&self) -> String {
        let mut bytes = Vec::with_capacity(4 + self.pairs.len() * 8);
        for (i, v) in &self.pairs {
            bytes.extend_from_slice(&i.to_be_bytes());
            bytes.extend_from_slice(&v.to_be_bytes());
        }
        bytes.extend_from_slice(&self.size.to_be_bytes());
        general_purpose::STANDARD.encode(bytes)
    }
}

/// Builds the suffix array of `s` using prefix doubling with radix sorting
/// (O(n log n)).
///
/// Returns a vector SA where SA[i] is the start of the i-th
/// lexicographically smallest suffix. Ranks are 1-based so that index 0 can
/// model end-of-string: shorter suffixes that are a prefix of a longer one
/// sort first. Because every suffix of a string is distinct, the suffix array
/// is unique, so this is byte-for-byte equivalent to a comparison-based build.
#[inline]
fn build_suffix_array(s: &[u8]) -> Vec<usize> {
    let n = s.len();
    if n == 0 {
        return vec![];
    }
    if n == 1 {
        return vec![0];
    }

    // Counting-sort buckets must span every rank value. After the first round
    // ranks are bounded by `n`; the initial byte-derived ranks reach 256.
    let cap = n.max(ALPHABET_SIZE) + 2;

    // 1-based ranks: 0 is reserved as the "past end of string" sentinel.
    let mut rank: Vec<u32> = s.iter().map(|&b| u32::from(b) + 1).collect();
    let mut next_rank = vec![0u32; n];
    let mut sa: Vec<usize> = (0..n).collect();
    let mut sa_by_low = vec![0usize; n];
    let mut count = vec![0usize; cap];

    let mut k = 1usize;
    loop {
        // Radix sort the suffixes by the pair (rank[i], rank[i + k]) using two
        // stable counting-sort passes: first by the low key, then by the high.

        // Pass 1: stable sort by the low key rank[i + k] (0 when i + k >= n).
        count.fill(0);
        for i in 0..n {
            let low = if i + k < n { rank[i + k] as usize } else { 0 };
            count[low + 1] += 1;
        }
        for i in 1..cap {
            count[i] += count[i - 1];
        }
        for i in 0..n {
            let low = if i + k < n { rank[i + k] as usize } else { 0 };
            sa_by_low[count[low]] = i;
            count[low] += 1;
        }

        // Pass 2: stable sort by the high key rank[i], preserving low-key order.
        count.fill(0);
        for &r in &rank {
            count[r as usize + 1] += 1;
        }
        for i in 1..cap {
            count[i] += count[i - 1];
        }
        for &i in &sa_by_low {
            let high = rank[i] as usize;
            sa[count[high]] = i;
            count[high] += 1;
        }

        // Recompute ranks from the new ordering.
        let key = |i: usize| (rank[i], if i + k < n { rank[i + k] } else { 0 });
        next_rank[sa[0]] = 1;
        let mut r = 1u32;
        for w in 1..n {
            if key(sa[w]) != key(sa[w - 1]) {
                r += 1;
            }
            next_rank[sa[w]] = r;
        }
        std::mem::swap(&mut rank, &mut next_rank);

        if r as usize == n || k >= n {
            break;
        }
        k *= 2;
    }

    sa
}

/// Computes the BWMD feature vector for a byte sequence.
///
/// Builds a suffix array of `data`, derives BWT characters from it, counts
/// bigram transitions between consecutive BWT characters (skipping the first
/// two positions to match the reference implementation), normalizes counts to
/// probabilities, and applies the element-wise transform `sqrt(p) / sqrt(2)`.
///
/// Returns a 65,536-element `f32` vector (256 × 256 transition matrix,
/// row-major: index `prev * 256 + cur`). Inputs shorter than 3 bytes yield
/// an all-zero vector.
#[must_use]
#[allow(clippy::cast_precision_loss)]
pub fn vectorize(data: &[u8]) -> Vec<f32> {
    let l = data.len();
    if l < 3 {
        return vec![0.0f32; VEC_LEN];
    }

    let sa = build_suffix_array(data);
    let mut counts = vec![0.0f32; VEC_LEN];
    let mut prev_val = 0usize;

    for (pos, &sa_val) in sa.iter().enumerate() {
        // BWT character: the byte immediately before suffix sa_val
        let bwt_idx = if sa_val == 0 { l - 1 } else { sa_val - 1 };
        let cur_val = data[bwt_idx] as usize;

        // Skip pos 0 (no prev_val yet) and pos 1 (mirrors reference behaviour,
        // which avoids the sentinel-adjacent transition in the BWT).
        if pos > 1 {
            counts[prev_val * ALPHABET_SIZE + cur_val] += 1.0;
        }
        prev_val = cur_val;
    }

    // Normalize to transition probabilities, then apply the Hellinger-like
    // transform sqrt(p) / sqrt(2), in a single pass.
    let norm = (l - 1) as f32;
    let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
    for c in &mut counts {
        *c = (*c / norm).sqrt() * inv_sqrt2;
    }

    counts
}

/// Sparse vector version of the [`vectorize`] function.
#[must_use]
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::for_kv_map
)]
pub fn vectorize_sparse(data: &[u8]) -> SparseVector {
    let l = data.len();
    let mut counts = HashMap::new();
    if l < 3 {
        return SparseVector {
            pairs: counts,
            size: VEC_LEN as u32,
        };
    }

    let sa = build_suffix_array(data);
    let mut prev_val = 0usize;

    for (pos, &sa_val) in sa.iter().enumerate() {
        // BWT character: the byte immediately before suffix sa_val
        let bwt_idx = if sa_val == 0 { l - 1 } else { sa_val - 1 };
        let cur_val = data[bwt_idx] as usize;

        // Skip pos 0 (no prev_val yet) and pos 1 (mirrors reference behaviour,
        // which avoids the sentinel-adjacent transition in the BWT).
        if pos > 1 {
            let index = (prev_val * ALPHABET_SIZE + cur_val) as u32;
            if let Some(value) = counts.get_mut(&index) {
                *value += 1.0;
            } else {
                counts.insert(index, 1.0);
            }
        }
        prev_val = cur_val;
    }

    // Normalize to transition probabilities, then apply the Hellinger-like
    // transform sqrt(p) / sqrt(2), in a single pass.
    let norm = (l - 1) as f32;
    let inv_sqrt2 = std::f32::consts::FRAC_1_SQRT_2;
    for (_, c) in &mut counts {
        *c = (*c / norm).sqrt() * inv_sqrt2;
    }

    SparseVector {
        pairs: counts,
        size: VEC_LEN as u32,
    }
}

/// Returns the BWMD between two byte sequences: the Euclidean distance
/// between their `vectorize` feature vectors. Range is [0.0, 1.0]
#[inline]
#[must_use]
pub fn distance(a: &[u8], b: &[u8]) -> f32 {
    let va = vectorize(a);
    let vb = vectorize(b);

    va.iter()
        .zip(vb.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f32>()
        .sqrt()
}

#[cfg(test)]
mod tests;
