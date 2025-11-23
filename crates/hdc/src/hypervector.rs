//! High-dimensional binary vectors for cognitive computing
//!
//! HyperVectors are sparse binary vectors (10,000+ dimensions) that can
//! represent and compose concepts using simple bitwise operations.

use bitvec::prelude::*;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;
use serde::{Deserialize, Deserializer, Serialize, Serializer};
use std::fmt;

/// A high-dimensional binary vector
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct HyperVector {
    /// The actual bit storage
    bits: BitVec<u64, Lsb0>,
    /// Dimension of the vector
    dimension: usize,
}

impl HyperVector {
    /// Create a new random hypervector
    pub fn random(dimension: usize) -> Self {
        let mut rng = rand::thread_rng();
        let mut bits = BitVec::with_capacity(dimension);

        for _ in 0..dimension {
            bits.push(rng.gen_bool(0.5));
        }

        Self { bits, dimension }
    }

    /// Create a random hypervector with seed for deterministic generation
    pub fn random_seeded(dimension: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut bits = BitVec::with_capacity(dimension);

        for _ in 0..dimension {
            bits.push(rng.gen_bool(0.5));
        }

        Self { bits, dimension }
    }

    /// Create a zero hypervector
    pub fn zeros(dimension: usize) -> Self {
        Self {
            bits: BitVec::repeat(false, dimension),
            dimension,
        }
    }

    /// Create an all-ones hypervector
    pub fn ones(dimension: usize) -> Self {
        Self {
            bits: BitVec::repeat(true, dimension),
            dimension,
        }
    }

    /// Get the dimension
    pub fn dimension(&self) -> usize {
        self.dimension
    }

    /// Get a specific bit
    pub fn get(&self, index: usize) -> Option<bool> {
        self.bits.get(index).map(|b| *b)
    }

    /// Set a specific bit
    pub fn set(&mut self, index: usize, value: bool) {
        if let Some(mut bit) = self.bits.get_mut(index) {
            bit.set(value);
        }
    }

    /// Flip a specific bit
    pub fn flip(&mut self, index: usize) {
        if let Some(mut bit) = self.bits.get_mut(index) {
            let val = *bit;
            bit.set(!val);
        }
    }

    /// Count the number of 1s (Hamming weight)
    pub fn count_ones(&self) -> usize {
        self.bits.count_ones()
    }

    /// Hamming distance to another vector
    pub fn hamming_distance(&self, other: &HyperVector) -> usize {
        assert_eq!(self.dimension, other.dimension, "Dimensions must match");

        (self.bits.clone() ^ other.bits.clone()).count_ones()
    }

    /// Normalized Hamming similarity (0.0 to 1.0)
    pub fn similarity(&self, other: &HyperVector) -> f32 {
        let distance = self.hamming_distance(other) as f32;
        1.0 - (distance / self.dimension as f32)
    }

    /// XOR with another vector (binding operation)
    pub fn xor(&self, other: &HyperVector) -> HyperVector {
        assert_eq!(self.dimension, other.dimension, "Dimensions must match");

        HyperVector {
            bits: self.bits.clone() ^ other.bits.clone(),
            dimension: self.dimension,
        }
    }

    /// Majority vote across multiple vectors (bundling operation)
    pub fn majority(vectors: &[HyperVector]) -> HyperVector {
        if vectors.is_empty() {
            panic!("Cannot compute majority of empty vector set");
        }

        let dimension = vectors[0].dimension;
        for v in vectors {
            assert_eq!(v.dimension, dimension, "All vectors must have same dimension");
        }

        let mut result = HyperVector::zeros(dimension);

        // Count votes for each bit position
        for i in 0..dimension {
            let ones_count = vectors.iter()
                .filter(|v| v.get(i).unwrap_or(false))
                .count();

            // Set bit if majority are 1s (with random tiebreaker)
            if ones_count > vectors.len() / 2 {
                result.set(i, true);
            } else if ones_count == vectors.len() / 2 {
                // Random tiebreaker for even split
                result.set(i, rand::thread_rng().gen_bool(0.5));
            }
        }

        result
    }

    /// Circular shift (permutation operation)
    pub fn rotate(&self, positions: isize) -> HyperVector {
        let mut result = self.clone();

        if positions == 0 {
            return result;
        }

        let shift = positions.rem_euclid(self.dimension as isize) as usize;

        if shift > 0 {
            result.bits.rotate_right(shift);
        }

        result
    }

    /// Create from raw bytes
    pub fn from_bytes(bytes: &[u8], dimension: usize) -> Self {
        let mut bits = BitVec::<u64, Lsb0>::with_capacity(dimension);

        // Read bits from bytes
        for byte in bytes {
            for bit_idx in 0..8 {
                if bits.len() >= dimension {
                    break;
                }
                bits.push((byte >> bit_idx) & 1 == 1);
            }
            if bits.len() >= dimension {
                break;
            }
        }

        // Pad with zeros if needed
        while bits.len() < dimension {
            bits.push(false);
        }

        Self {
            bits,
            dimension,
        }
    }

    /// Convert to raw bytes
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity((self.dimension + 7) / 8);

        for chunk in self.bits.chunks(8) {
            let mut byte = 0u8;
            for (i, bit) in chunk.iter().enumerate() {
                if *bit {
                    byte |= 1 << i;
                }
            }
            bytes.push(byte);
        }

        bytes
    }

    /// Get inner bitvec for advanced operations
    pub fn as_bitvec(&self) -> &BitVec<u64, Lsb0> {
        &self.bits
    }

    /// Get mutable inner bitvec for advanced operations
    pub fn as_bitvec_mut(&mut self) -> &mut BitVec<u64, Lsb0> {
        &mut self.bits
    }
}

impl Serialize for HyperVector {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("HyperVector", 2)?;
        state.serialize_field("bytes", &self.to_bytes())?;
        state.serialize_field("dimension", &self.dimension)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for HyperVector {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        use serde::de::{self, MapAccess, Visitor};
        use std::fmt;

        struct HyperVectorVisitor;

        impl<'de> Visitor<'de> for HyperVectorVisitor {
            type Value = HyperVector;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct HyperVector")
            }

            fn visit_map<V>(self, mut map: V) -> Result<HyperVector, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut bytes = None;
                let mut dimension = None;

                while let Some(key) = map.next_key::<String>()? {
                    match key.as_str() {
                        "bytes" => {
                            bytes = Some(map.next_value()?);
                        }
                        "dimension" => {
                            dimension = Some(map.next_value()?);
                        }
                        _ => {
                            let _: de::IgnoredAny = map.next_value()?;
                        }
                    }
                }

                let bytes: Vec<u8> = bytes.ok_or_else(|| de::Error::missing_field("bytes"))?;
                let dimension = dimension.ok_or_else(|| de::Error::missing_field("dimension"))?;

                Ok(HyperVector::from_bytes(&bytes, dimension))
            }
        }

        const FIELDS: &[&str] = &["bytes", "dimension"];
        deserializer.deserialize_struct("HyperVector", FIELDS, HyperVectorVisitor)
    }
}

impl fmt::Display for HyperVector {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "HyperVector[{}D, {}% ones]",
            self.dimension,
            (self.count_ones() as f32 / self.dimension as f32 * 100.0) as u32
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_creation() {
        let hv = HyperVector::random(1000);
        assert_eq!(hv.dimension(), 1000);

        // Should be approximately 50% ones
        let ones_ratio = hv.count_ones() as f32 / 1000.0;
        assert!(ones_ratio > 0.4 && ones_ratio < 0.6);
    }

    #[test]
    fn test_deterministic() {
        let hv1 = HyperVector::random_seeded(100, 42);
        let hv2 = HyperVector::random_seeded(100, 42);
        assert_eq!(hv1, hv2);
    }

    #[test]
    fn test_xor_binding() {
        let hv1 = HyperVector::ones(100);
        let hv2 = HyperVector::zeros(100);
        let bound = hv1.xor(&hv2);

        // XOR of all-ones and all-zeros should be all-ones
        assert_eq!(bound.count_ones(), 100);
    }

    #[test]
    fn test_majority_bundling() {
        let hv1 = HyperVector::ones(100);
        let hv2 = HyperVector::ones(100);
        let hv3 = HyperVector::zeros(100);

        let bundled = HyperVector::majority(&[hv1, hv2, hv3]);

        // Majority should be ones (2 out of 3)
        assert_eq!(bundled.count_ones(), 100);
    }

    #[test]
    fn test_rotation() {
        let mut hv = HyperVector::zeros(100);
        hv.set(0, true);
        hv.set(1, true);

        let rotated = hv.rotate(1);
        assert!(rotated.get(1).unwrap());
        assert!(rotated.get(2).unwrap());

        let rotated_back = rotated.rotate(-1);
        assert_eq!(hv, rotated_back);
    }

    #[test]
    fn test_similarity() {
        let hv1 = HyperVector::random_seeded(1000, 42);
        let hv2 = HyperVector::random_seeded(1000, 42);
        let hv3 = HyperVector::random_seeded(1000, 99);

        assert_eq!(hv1.similarity(&hv2), 1.0);

        let sim = hv1.similarity(&hv3);
        assert!(sim > 0.4 && sim < 0.6); // Random vectors ~50% similar
    }
}