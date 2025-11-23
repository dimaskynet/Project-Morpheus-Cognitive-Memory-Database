//! Similarity metrics for hypervectors

use crate::HyperVector;

/// Cosine similarity between two hypervectors (normalized Hamming)
///
/// Returns value in range [0, 1] where 1 is identical
pub fn cosine_similarity(a: &HyperVector, b: &HyperVector) -> f32 {
    a.similarity(b)
}

/// Hamming distance (number of differing bits)
pub fn hamming_distance(a: &HyperVector, b: &HyperVector) -> usize {
    a.hamming_distance(b)
}

/// Jaccard similarity (intersection over union)
pub fn jaccard_similarity(a: &HyperVector, b: &HyperVector) -> f32 {
    let a_bits = a.as_bitvec();
    let b_bits = b.as_bitvec();

    let intersection = (a_bits.clone() & b_bits.clone()).count_ones();
    let union = (a_bits.clone() | b_bits.clone()).count_ones();

    if union == 0 {
        return 0.0;
    }

    intersection as f32 / union as f32
}

/// Dot product similarity (count of matching 1s)
pub fn dot_product(a: &HyperVector, b: &HyperVector) -> usize {
    let a_bits = a.as_bitvec();
    let b_bits = b.as_bitvec();

    (a_bits.clone() & b_bits.clone()).count_ones()
}

/// Normalized dot product similarity
pub fn normalized_dot_product(a: &HyperVector, b: &HyperVector) -> f32 {
    let dot = dot_product(a, b) as f32;
    let a_norm = a.count_ones() as f32;
    let b_norm = b.count_ones() as f32;

    if a_norm == 0.0 || b_norm == 0.0 {
        return 0.0;
    }

    dot / (a_norm * b_norm).sqrt()
}

/// Find k most similar vectors from a set
pub fn top_k_similar<'a>(
    query: &HyperVector,
    candidates: &'a [HyperVector],
    k: usize,
) -> Vec<(usize, &'a HyperVector, f32)> {
    let mut similarities: Vec<(usize, &HyperVector, f32)> = candidates
        .iter()
        .enumerate()
        .map(|(idx, hv)| (idx, hv, query.similarity(hv)))
        .collect();

    similarities.sort_by(|a, b| b.2.partial_cmp(&a.2).unwrap());
    similarities.truncate(k);

    similarities
}

/// Threshold-based search: find all vectors above similarity threshold
pub fn threshold_search<'a>(
    query: &HyperVector,
    candidates: &'a [HyperVector],
    threshold: f32,
) -> Vec<(usize, &'a HyperVector, f32)> {
    candidates
        .iter()
        .enumerate()
        .filter_map(|(idx, hv)| {
            let sim = query.similarity(hv);
            if sim >= threshold {
                Some((idx, hv, sim))
            } else {
                None
            }
        })
        .collect()
}

/// Distance-based clustering helper
pub struct SimilarityMatrix {
    size: usize,
    similarities: Vec<f32>,
}

impl SimilarityMatrix {
    /// Compute pairwise similarities for a set of vectors
    pub fn compute(vectors: &[HyperVector]) -> Self {
        let size = vectors.len();
        let mut similarities = Vec::with_capacity(size * size);

        for i in 0..size {
            for j in 0..size {
                if i == j {
                    similarities.push(1.0);
                } else {
                    similarities.push(vectors[i].similarity(&vectors[j]));
                }
            }
        }

        Self { size, similarities }
    }

    /// Get similarity between vectors i and j
    pub fn get(&self, i: usize, j: usize) -> f32 {
        self.similarities[i * self.size + j]
    }

    /// Find the most similar pair
    pub fn most_similar_pair(&self) -> Option<(usize, usize, f32)> {
        let mut best = None;
        let mut best_sim = 0.0;

        for i in 0..self.size {
            for j in (i + 1)..self.size {
                let sim = self.get(i, j);
                if sim > best_sim {
                    best_sim = sim;
                    best = Some((i, j, sim));
                }
            }
        }

        best
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine_similarity() {
        let a = HyperVector::random_seeded(1000, 1);
        let b = HyperVector::random_seeded(1000, 1);

        assert_eq!(cosine_similarity(&a, &b), 1.0);

        let c = HyperVector::random_seeded(1000, 2);
        let sim = cosine_similarity(&a, &c);
        assert!(sim > 0.3 && sim < 0.7); // Random vectors
    }

    #[test]
    fn test_jaccard_similarity() {
        let ones = HyperVector::ones(100);
        let zeros = HyperVector::zeros(100);

        assert_eq!(jaccard_similarity(&ones, &ones), 1.0);
        assert_eq!(jaccard_similarity(&ones, &zeros), 0.0);
    }

    #[test]
    fn test_top_k_similar() {
        let query = HyperVector::random_seeded(1000, 1);
        let candidates = vec![
            HyperVector::random_seeded(1000, 1), // Identical
            HyperVector::random_seeded(1000, 2),
            HyperVector::random_seeded(1000, 3),
        ];

        let top = top_k_similar(&query, &candidates, 1);

        assert_eq!(top.len(), 1);
        assert_eq!(top[0].0, 0); // First candidate is identical
        assert_eq!(top[0].2, 1.0);
    }

    #[test]
    fn test_threshold_search() {
        let query = HyperVector::random_seeded(1000, 1);
        let candidates = vec![
            HyperVector::random_seeded(1000, 1), // Identical
            HyperVector::random_seeded(1000, 2),
        ];

        let results = threshold_search(&query, &candidates, 0.9);

        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
    }

    #[test]
    fn test_similarity_matrix() {
        let v1 = HyperVector::random_seeded(100, 1);
        let v2 = HyperVector::random_seeded(100, 2);
        let v3 = HyperVector::random_seeded(100, 1); // Same as v1

        let matrix = SimilarityMatrix::compute(&[v1, v2, v3]);

        assert_eq!(matrix.get(0, 0), 1.0);
        assert_eq!(matrix.get(0, 2), 1.0); // v1 == v3
        assert!(matrix.get(0, 1) < 0.8);   // v1 != v2
    }
}
