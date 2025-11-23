//! Encoders for converting data into hypervectors
//!
//! Different data types require different encoding strategies:
//! - Scalars: Level hypervectors with thresholding
//! - Symbols: Random codebook lookup
//! - Sequences: Permute-and-bundle
//! - Spatial: N-grams with permutation

use crate::{HyperVector, operations::{bind, bundle, permute}};
use std::collections::HashMap;

/// Encoder for scalar values into hypervectors
pub struct ScalarEncoder {
    dimension: usize,
    min_val: f32,
    max_val: f32,
    levels: Vec<HyperVector>,
}

impl ScalarEncoder {
    /// Create a new scalar encoder with specified number of levels
    pub fn new(dimension: usize, min_val: f32, max_val: f32, num_levels: usize) -> Self {
        // Generate random hypervectors for each level
        let levels: Vec<HyperVector> = (0..num_levels)
            .map(|i| HyperVector::random_seeded(dimension, i as u64))
            .collect();

        Self {
            dimension,
            min_val,
            max_val,
            levels,
        }
    }

    /// Encode a scalar value into a hypervector
    pub fn encode(&self, value: f32) -> HyperVector {
        // Clamp to range
        let clamped = value.clamp(self.min_val, self.max_val);

        // Normalize to [0, 1]
        let normalized = (clamped - self.min_val) / (self.max_val - self.min_val);

        // Map to level index
        let level_idx = (normalized * (self.levels.len() - 1) as f32) as usize;

        self.levels[level_idx].clone()
    }

    /// Encode with interpolation between levels for smoother representation
    pub fn encode_interpolated(&self, value: f32) -> HyperVector {
        let clamped = value.clamp(self.min_val, self.max_val);
        let normalized = (clamped - self.min_val) / (self.max_val - self.min_val);

        let float_idx = normalized * (self.levels.len() - 1) as f32;
        let lower_idx = float_idx.floor() as usize;
        let upper_idx = (float_idx.ceil() as usize).min(self.levels.len() - 1);

        if lower_idx == upper_idx {
            return self.levels[lower_idx].clone();
        }

        // Bundle the two adjacent levels
        bundle(&[self.levels[lower_idx].clone(), self.levels[upper_idx].clone()])
    }
}

/// Encoder for symbolic/categorical data
pub struct SymbolEncoder {
    dimension: usize,
    codebook: HashMap<String, HyperVector>,
    seed_counter: u64,
}

impl SymbolEncoder {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            codebook: HashMap::new(),
            seed_counter: 0,
        }
    }

    /// Encode a symbol, creating new hypervector if not seen before
    pub fn encode(&mut self, symbol: &str) -> HyperVector {
        if let Some(hv) = self.codebook.get(symbol) {
            return hv.clone();
        }

        // Create new random hypervector for this symbol
        let hv = HyperVector::random_seeded(self.dimension, self.seed_counter);
        self.seed_counter += 1;
        self.codebook.insert(symbol.to_string(), hv.clone());

        hv
    }

    /// Pre-populate codebook with known symbols
    pub fn add_symbols(&mut self, symbols: &[&str]) {
        for symbol in symbols {
            if !self.codebook.contains_key(*symbol) {
                let hv = HyperVector::random_seeded(self.dimension, self.seed_counter);
                self.seed_counter += 1;
                self.codebook.insert(symbol.to_string(), hv);
            }
        }
    }

    /// Get the codebook for cleanup operations
    pub fn get_codebook(&self) -> Vec<HyperVector> {
        self.codebook.values().cloned().collect()
    }

    /// Decode by finding closest symbol
    pub fn decode(&self, hv: &HyperVector) -> Option<String> {
        self.codebook
            .iter()
            .max_by(|(_, a), (_, b)| {
                let sim_a = hv.similarity(a);
                let sim_b = hv.similarity(b);
                sim_a.partial_cmp(&sim_b).unwrap()
            })
            .map(|(sym, _)| sym.clone())
    }
}

/// Encoder for sequences (text, time series, etc.)
pub struct SequenceEncoder {
    dimension: usize,
    symbol_encoder: SymbolEncoder,
}

impl SequenceEncoder {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            symbol_encoder: SymbolEncoder::new(dimension),
        }
    }

    /// Encode a sequence using permutation-based position encoding
    pub fn encode(&mut self, sequence: &[&str]) -> HyperVector {
        if sequence.is_empty() {
            return HyperVector::zeros(self.dimension);
        }

        let encoded: Vec<HyperVector> = sequence
            .iter()
            .enumerate()
            .map(|(pos, symbol)| {
                let symbol_hv = self.symbol_encoder.encode(symbol);
                permute(&symbol_hv, pos as isize)
            })
            .collect();

        bundle(&encoded)
    }

    /// Encode using n-grams for better context
    pub fn encode_ngrams(&mut self, sequence: &[&str], n: usize) -> HyperVector {
        if sequence.len() < n {
            return self.encode(sequence);
        }

        let mut ngram_hvs = Vec::new();

        for i in 0..=sequence.len() - n {
            let ngram = &sequence[i..i + n];

            // Bind n symbols together
            let ngram_symbols: Vec<HyperVector> = ngram
                .iter()
                .map(|s| self.symbol_encoder.encode(s))
                .collect();

            let mut ngram_hv = ngram_symbols[0].clone();
            for hv in &ngram_symbols[1..] {
                ngram_hv = bind(&ngram_hv, hv);
            }

            // Position-encode the n-gram
            ngram_hvs.push(permute(&ngram_hv, i as isize));
        }

        bundle(&ngram_hvs)
    }
}

/// Encoder for key-value associations
pub struct MapEncoder {
    dimension: usize,
    key_encoder: SymbolEncoder,
}

impl MapEncoder {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            key_encoder: SymbolEncoder::new(dimension),
        }
    }

    /// Encode a key-value pair
    pub fn encode_pair(&mut self, key: &str, value: &HyperVector) -> HyperVector {
        let key_hv = self.key_encoder.encode(key);
        bind(&key_hv, value)
    }

    /// Encode a map of key-value pairs
    pub fn encode_map(&mut self, pairs: &[(&str, HyperVector)]) -> HyperVector {
        let bound_pairs: Vec<HyperVector> = pairs
            .iter()
            .map(|(k, v)| self.encode_pair(k, v))
            .collect();

        bundle(&bound_pairs)
    }

    /// Query a map hypervector with a key
    pub fn query(&mut self, map_hv: &HyperVector, key: &str) -> HyperVector {
        let key_hv = self.key_encoder.encode(key);
        bind(map_hv, &key_hv) // Unbind to retrieve value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scalar_encoder() {
        let encoder = ScalarEncoder::new(1000, 0.0, 100.0, 10);

        let hv1 = encoder.encode(10.0);
        let hv2 = encoder.encode(10.5);
        let hv3 = encoder.encode(90.0);

        // Similar values should produce similar vectors
        assert!(hv1.similarity(&hv2) > 0.8);

        // Distant values should be dissimilar
        assert!(hv1.similarity(&hv3) < 0.7);
    }

    #[test]
    fn test_symbol_encoder() {
        let mut encoder = SymbolEncoder::new(1000);

        let cat = encoder.encode("cat");
        let dog = encoder.encode("dog");
        let cat2 = encoder.encode("cat");

        // Same symbol should give identical vector
        assert_eq!(cat, cat2);

        // Different symbols should be dissimilar
        assert!(cat.similarity(&dog) < 0.7);
    }

    #[test]
    fn test_symbol_decode() {
        let mut encoder = SymbolEncoder::new(1000);

        encoder.add_symbols(&["red", "green", "blue"]);

        let red_hv = encoder.encode("red");
        let decoded = encoder.decode(&red_hv).unwrap();

        assert_eq!(decoded, "red");
    }

    #[test]
    fn test_sequence_encoder() {
        let mut encoder = SequenceEncoder::new(1000);

        let seq1 = encoder.encode(&["the", "cat", "sat"]);
        let seq2 = encoder.encode(&["cat", "the", "sat"]);

        // Different order should produce different vectors
        assert!(seq1.similarity(&seq2) < 0.9);
    }

    #[test]
    fn test_map_encoder() {
        let mut encoder = MapEncoder::new(1000);

        let name_hv = HyperVector::random_seeded(1000, 1);
        let age_hv = HyperVector::random_seeded(1000, 2);

        let map = encoder.encode_map(&[
            ("name", name_hv.clone()),
            ("age", age_hv.clone()),
        ]);

        let retrieved_name = encoder.query(&map, "name");

        // Should retrieve similar vector
        assert!(retrieved_name.similarity(&name_hv) > 0.6);
    }
}
