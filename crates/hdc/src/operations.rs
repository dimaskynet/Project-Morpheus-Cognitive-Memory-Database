//! Core HDC operations: bind, bundle, permute
//!
//! These operations form the algebra of hyperdimensional computing:
//! - BIND: XOR - creates dissimilar vector from similar inputs (A XOR B)
//! - BUNDLE: Majority - creates similar vector from inputs (A + B + C)
//! - PERMUTE: Rotation - creates dissimilar but reversible transformation

use crate::HyperVector;

/// Bind two hypervectors together using XOR
///
/// Binding creates a new vector that is dissimilar to both inputs.
/// Key property: A � B � B H A (unbinding)
pub fn bind(a: &HyperVector, b: &HyperVector) -> HyperVector {
    a.xor(b)
}

/// Bind multiple hypervectors together
///
/// Equivalent to chaining XOR operations: A � B � C � ...
pub fn bind_multiple(vectors: &[HyperVector]) -> HyperVector {
    if vectors.is_empty() {
        panic!("Cannot bind empty vector set");
    }

    let mut result = vectors[0].clone();
    for v in &vectors[1..] {
        result = result.xor(v);
    }

    result
}

/// Bundle multiple hypervectors using majority vote
///
/// Bundling creates a new vector that is similar to all inputs.
/// This is the HDC equivalent of addition/superposition.
pub fn bundle(vectors: &[HyperVector]) -> HyperVector {
    HyperVector::majority(vectors)
}

/// Permute a hypervector by rotating bits
///
/// Permutation creates a dissimilar vector that can be reversed.
/// Used to encode sequential/positional information.
pub fn permute(v: &HyperVector, positions: isize) -> HyperVector {
    v.rotate(positions)
}

/// Create a sequence representation by permuting and bundling
///
/// Encodes order information: [A, B, C] becomes bundle(A, permute(B, 1), permute(C, 2))
pub fn encode_sequence(vectors: &[HyperVector]) -> HyperVector {
    if vectors.is_empty() {
        panic!("Cannot encode empty sequence");
    }

    let permuted: Vec<HyperVector> = vectors
        .iter()
        .enumerate()
        .map(|(i, v)| permute(v, i as isize))
        .collect();

    bundle(&permuted)
}

/// Unbind a value from a bound vector
///
/// Given C = A � B, retrieve B by computing C � A
pub fn unbind(bound: &HyperVector, key: &HyperVector) -> HyperVector {
    bind(bound, key)
}

/// Clean up a noisy hypervector using a codebook
///
/// Finds the most similar vector in the codebook (cleanup memory)
pub fn cleanup<'a>(noisy: &HyperVector, codebook: &'a [HyperVector]) -> &'a HyperVector {
    if codebook.is_empty() {
        panic!("Codebook cannot be empty");
    }

    codebook
        .iter()
        .max_by(|a, b| {
            let sim_a = noisy.similarity(a);
            let sim_b = noisy.similarity(b);
            sim_a.partial_cmp(&sim_b).unwrap()
        })
        .unwrap()
}

/// Resonator network for iterative cleanup
///
/// Uses consensus-based approach to resolve noisy/ambiguous vectors
pub struct Resonator {
    codebook: Vec<HyperVector>,
    max_iterations: usize,
    convergence_threshold: f32,
}

impl Resonator {
    pub fn new(codebook: Vec<HyperVector>) -> Self {
        Self {
            codebook,
            max_iterations: 10,
            convergence_threshold: 0.95,
        }
    }

    /// Iteratively clean up a vector until convergence
    pub fn resonate(&self, input: &HyperVector) -> HyperVector {
        let mut current = input.clone();
        let mut prev_similarity = 0.0;

        for _ in 0..self.max_iterations {
            let cleaned = cleanup(&current, &self.codebook).clone();
            let similarity = current.similarity(&cleaned);

            if similarity >= self.convergence_threshold {
                return cleaned;
            }

            if (similarity - prev_similarity).abs() < 0.01 {
                // Converged or oscillating
                return cleaned;
            }

            current = cleaned;
            prev_similarity = similarity;
        }

        current
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bind_unbind() {
        let key = HyperVector::random_seeded(1000, 1);
        let value = HyperVector::random_seeded(1000, 2);

        let bound = bind(&key, &value);
        let unbound = unbind(&bound, &key);

        // Should recover original value (with some noise)
        assert!(unbound.similarity(&value) > 0.95);
    }

    #[test]
    fn test_bundle_similarity() {
        let v1 = HyperVector::random_seeded(1000, 1);
        let v2 = HyperVector::random_seeded(1000, 2);
        let v3 = HyperVector::random_seeded(1000, 3);

        let bundled = bundle(&[v1.clone(), v2.clone(), v3.clone()]);

        // Bundled should be similar to all inputs
        assert!(bundled.similarity(&v1) > 0.5);
        assert!(bundled.similarity(&v2) > 0.5);
        assert!(bundled.similarity(&v3) > 0.5);
    }

    #[test]
    fn test_permute_reversible() {
        let v = HyperVector::random_seeded(1000, 42);
        let p = permute(&v, 100);
        let back = permute(&p, -100);

        assert_eq!(v, back);
    }

    #[test]
    fn test_sequence_encoding() {
        let a = HyperVector::random_seeded(1000, 1);
        let b = HyperVector::random_seeded(1000, 2);

        let seq_ab = encode_sequence(&[a.clone(), b.clone()]);
        let seq_ba = encode_sequence(&[b.clone(), a.clone()]);

        // Order matters - different sequences should be dissimilar
        assert!(seq_ab.similarity(&seq_ba) < 0.7);
    }

    #[test]
    fn test_cleanup() {
        let v1 = HyperVector::random_seeded(1000, 1);
        let v2 = HyperVector::random_seeded(1000, 2);
        let v3 = HyperVector::random_seeded(1000, 3);

        let codebook = vec![v1.clone(), v2.clone(), v3.clone()];

        // Add noise to v1
        let mut noisy = v1.clone();
        for i in 0..50 {
            noisy.flip(i);
        }

        let cleaned = cleanup(&noisy, &codebook);

        // Should recover v1
        assert_eq!(cleaned, &v1);
    }

    #[test]
    fn test_resonator() {
        let v1 = HyperVector::random_seeded(1000, 1);
        let v2 = HyperVector::random_seeded(1000, 2);

        let codebook = vec![v1.clone(), v2.clone()];
        let resonator = Resonator::new(codebook);

        let mut noisy = v1.clone();
        for i in 0..100 {
            noisy.flip(i);
        }

        let result = resonator.resonate(&noisy);

        // Should converge to v1
        assert!(result.similarity(&v1) > 0.9);
    }
}
