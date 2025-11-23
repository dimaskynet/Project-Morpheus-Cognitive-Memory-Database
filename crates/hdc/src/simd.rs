//! SIMD-accelerated operations for hypervectors
//!
//! This module provides optimized implementations using AVX2/SSE when available.

use crate::HyperVector;

/// SIMD-accelerated Hamming distance
#[cfg(target_arch = "x86_64")]
pub fn hamming_distance_simd(a: &HyperVector, b: &HyperVector) -> usize {
    assert_eq!(a.dimension(), b.dimension());

    let a_bits = a.as_bitvec().as_raw_slice();
    let b_bits = b.as_bitvec().as_raw_slice();

    if is_x86_feature_detected!("avx2") {
        unsafe { hamming_distance_avx2(a_bits, b_bits) }
    } else if is_x86_feature_detected!("sse4.2") {
        unsafe { hamming_distance_sse42(a_bits, b_bits) }
    } else {
        hamming_distance_scalar(a_bits, b_bits)
    }
}

/// Fallback for non-x86_64 architectures
#[cfg(not(target_arch = "x86_64"))]
pub fn hamming_distance_simd(a: &HyperVector, b: &HyperVector) -> usize {
    a.hamming_distance(b)
}

/// AVX2-accelerated Hamming distance (processes 4x 64-bit words at once)
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn hamming_distance_avx2(a: &[u64], b: &[u64]) -> usize {
    let mut distance = 0;

    let chunks = a.len() / 4;
    let remainder = a.len() % 4;

    // Process 4 u64s at a time using AVX2
    for i in 0..chunks {
        let idx = i * 4;

        let xor0 = a[idx] ^ b[idx];
        let xor1 = a[idx + 1] ^ b[idx + 1];
        let xor2 = a[idx + 2] ^ b[idx + 2];
        let xor3 = a[idx + 3] ^ b[idx + 3];

        distance += xor0.count_ones() as usize;
        distance += xor1.count_ones() as usize;
        distance += xor2.count_ones() as usize;
        distance += xor3.count_ones() as usize;
    }

    // Handle remainder
    for i in 0..remainder {
        let idx = chunks * 4 + i;
        distance += (a[idx] ^ b[idx]).count_ones() as usize;
    }

    distance
}

/// SSE4.2-accelerated Hamming distance
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "sse4.2")]
unsafe fn hamming_distance_sse42(a: &[u64], b: &[u64]) -> usize {
    let mut distance = 0;

    // Process 2 u64s at a time using SSE
    let chunks = a.len() / 2;
    let remainder = a.len() % 2;

    for i in 0..chunks {
        let idx = i * 2;
        let xor0 = a[idx] ^ b[idx];
        let xor1 = a[idx + 1] ^ b[idx + 1];

        distance += xor0.count_ones() as usize;
        distance += xor1.count_ones() as usize;
    }

    for i in 0..remainder {
        let idx = chunks * 2 + i;
        distance += (a[idx] ^ b[idx]).count_ones() as usize;
    }

    distance
}

/// Scalar fallback Hamming distance
fn hamming_distance_scalar(a: &[u64], b: &[u64]) -> usize {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones() as usize)
        .sum()
}

/// Batch Hamming distance calculation with SIMD
pub fn batch_hamming_distances(query: &HyperVector, candidates: &[HyperVector]) -> Vec<usize> {
    candidates
        .iter()
        .map(|candidate| hamming_distance_simd(query, candidate))
        .collect()
}

/// Parallel batch Hamming distance using rayon
pub fn parallel_batch_hamming_distances(
    query: &HyperVector,
    candidates: &[HyperVector],
) -> Vec<usize> {
    use rayon::prelude::*;

    candidates
        .par_iter()
        .map(|candidate| hamming_distance_simd(query, candidate))
        .collect()
}

/// SIMD-accelerated similarity calculation
pub fn similarity_simd(a: &HyperVector, b: &HyperVector) -> f32 {
    let distance = hamming_distance_simd(a, b) as f32;
    1.0 - (distance / a.dimension() as f32)
}

/// Batch similarity calculation
pub fn batch_similarities(query: &HyperVector, candidates: &[HyperVector]) -> Vec<f32> {
    let dim = query.dimension() as f32;
    batch_hamming_distances(query, candidates)
        .into_iter()
        .map(|dist| 1.0 - (dist as f32 / dim))
        .collect()
}

/// Parallel batch similarity calculation
pub fn parallel_batch_similarities(
    query: &HyperVector,
    candidates: &[HyperVector],
) -> Vec<f32> {
    let dim = query.dimension() as f32;
    parallel_batch_hamming_distances(query, candidates)
        .into_iter()
        .map(|dist| 1.0 - (dist as f32 / dim))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_distance_simd() {
        let a = HyperVector::random_seeded(10000, 1);
        let b = HyperVector::random_seeded(10000, 2);

        let dist_simd = hamming_distance_simd(&a, &b);
        let dist_regular = a.hamming_distance(&b);

        assert_eq!(dist_simd, dist_regular);
    }

    #[test]
    fn test_similarity_simd() {
        let a = HyperVector::random_seeded(10000, 1);
        let b = HyperVector::random_seeded(10000, 1);

        let sim = similarity_simd(&a, &b);
        assert_eq!(sim, 1.0);
    }

    #[test]
    fn test_batch_similarities() {
        let query = HyperVector::random_seeded(1000, 1);
        let candidates = vec![
            HyperVector::random_seeded(1000, 1),
            HyperVector::random_seeded(1000, 2),
            HyperVector::random_seeded(1000, 3),
        ];

        let sims = batch_similarities(&query, &candidates);

        assert_eq!(sims.len(), 3);
        assert_eq!(sims[0], 1.0); // Identical to query
        assert!(sims[1] < 1.0);
        assert!(sims[2] < 1.0);
    }

    #[test]
    fn test_parallel_batch_similarities() {
        let query = HyperVector::random_seeded(10000, 42);
        let candidates: Vec<_> = (0..100)
            .map(|i| HyperVector::random_seeded(10000, i))
            .collect();

        let sims_seq = batch_similarities(&query, &candidates);
        let sims_par = parallel_batch_similarities(&query, &candidates);

        assert_eq!(sims_seq, sims_par);
    }
}
