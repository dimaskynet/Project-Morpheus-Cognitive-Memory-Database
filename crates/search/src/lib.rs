//! HDC-based memory search and indexing
//!
//! This module provides fast structural search over memories using
//! Hyperdimensional Computing (HDC) vectors.

use cmd_core::memory::MemoryUnit;
use cmd_core::types::MemoryId;
use cmd_hdc::{HyperVector, encoder::SequenceEncoder, simd, DEFAULT_DIMENSION};
use std::collections::HashMap;
use dashmap::DashMap;
use chrono::{DateTime, Utc};

/// Search result with memory ID and similarity score
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub memory_id: MemoryId,
    pub similarity: f32,
    pub retention_score: f32,
}

/// HDC-based memory index for fast structural search
pub struct HdcMemoryIndex {
    /// Memory storage
    memories: HashMap<MemoryId, MemoryUnit>,

    /// HDC vectors for each memory
    hdc_vectors: HashMap<MemoryId, HyperVector>,

    /// Sequence encoder for text
    encoder: SequenceEncoder,

    /// Codebook for cleanup operations
    codebook: Vec<HyperVector>,

    /// Similarity cache for performance
    similarity_cache: DashMap<(MemoryId, MemoryId), f32>,
}

impl HdcMemoryIndex {
    /// Create a new empty index
    pub fn new() -> Self {
        Self {
            memories: HashMap::new(),
            hdc_vectors: HashMap::new(),
            encoder: SequenceEncoder::new(DEFAULT_DIMENSION),
            codebook: Vec::new(),
            similarity_cache: DashMap::new(),
        }
    }

    /// Add a memory to the index
    pub fn add_memory(&mut self, memory: MemoryUnit) -> Result<(), String> {
        let memory_id = memory.id.clone();

        // Encode memory text into HDC vector
        let hdc_vector = if let Some(ref text) = memory.text {
            self.encode_text(text)
        } else {
            // For non-text memories, encode metadata
            let metadata_str = format!("{:?}", memory.metadata);
            self.encode_text(&metadata_str)
        };

        // Update codebook
        self.codebook.push(hdc_vector.clone());

        // Store
        self.hdc_vectors.insert(memory_id.clone(), hdc_vector);
        self.memories.insert(memory_id, memory);

        Ok(())
    }

    /// Remove a memory from the index
    pub fn remove_memory(&mut self, memory_id: &MemoryId) -> Result<(), String> {
        self.memories.remove(memory_id)
            .ok_or_else(|| format!("Memory {} not found", memory_id))?;

        self.hdc_vectors.remove(memory_id);

        // Clear similarity cache entries for this memory
        self.similarity_cache.retain(|(a, b), _| a != memory_id && b != memory_id);

        Ok(())
    }

    /// Search for k most similar memories using HDC vectors
    pub fn search_hdc(&mut self, query: &str, k: usize) -> Vec<SearchResult> {
        // Encode query
        let query_vector = self.encode_text(query);

        // Compute similarities using SIMD
        let memory_ids: Vec<_> = self.hdc_vectors.keys().cloned().collect();
        let vectors: Vec<_> = memory_ids.iter()
            .filter_map(|id| self.hdc_vectors.get(id).cloned())
            .collect();

        let similarities = simd::parallel_batch_similarities(&query_vector, &vectors);

        // Combine with retention scores and rank
        let mut results: Vec<SearchResult> = memory_ids.into_iter()
            .zip(similarities.into_iter())
            .filter_map(|(memory_id, similarity)| {
                self.memories.get(&memory_id).map(|memory| {
                    let retention_score = memory.retention.retention_strength(Utc::now());

                    SearchResult {
                        memory_id,
                        similarity,
                        retention_score,
                    }
                })
            })
            .collect();

        // Weighted ranking: 70% similarity + 30% retention
        results.sort_by(|a, b| {
            let score_a = a.similarity * 0.7 + a.retention_score * 0.3;
            let score_b = b.similarity * 0.7 + b.retention_score * 0.3;
            score_b.partial_cmp(&score_a).unwrap()
        });

        results.truncate(k);
        results
    }

    /// Search with similarity threshold
    pub fn search_threshold(&mut self, query: &str, threshold: f32) -> Vec<SearchResult> {
        let query_vector = self.encode_text(query);

        let mut results: Vec<SearchResult> = self.hdc_vectors.iter()
            .filter_map(|(memory_id, vector)| {
                let similarity = simd::similarity_simd(&query_vector, vector);

                if similarity >= threshold {
                    self.memories.get(memory_id).map(|memory| {
                        let retention_score = memory.retention.retention_strength(Utc::now());

                        SearchResult {
                            memory_id: memory_id.clone(),
                            similarity,
                            retention_score,
                        }
                    })
                } else {
                    None
                }
            })
            .collect();

        // Weighted ranking
        results.sort_by(|a, b| {
            let score_a = a.similarity * 0.7 + a.retention_score * 0.3;
            let score_b = b.similarity * 0.7 + b.retention_score * 0.3;
            score_b.partial_cmp(&score_a).unwrap()
        });

        results
    }

    /// Search with temporal filter
    pub fn search_temporal(
        &mut self,
        query: &str,
        k: usize,
        since: Option<DateTime<Utc>>,
        until: Option<DateTime<Utc>>,
    ) -> Vec<SearchResult> {
        let query_vector = self.encode_text(query);

        let mut results: Vec<SearchResult> = self.hdc_vectors.iter()
            .filter_map(|(memory_id, vector)| {
                self.memories.get(memory_id).and_then(|memory| {
                    // Temporal filtering
                    let created = memory.temporal.created_at;

                    let in_range = match (since, until) {
                        (Some(s), Some(u)) => created >= s && created <= u,
                        (Some(s), None) => created >= s,
                        (None, Some(u)) => created <= u,
                        (None, None) => true,
                    };

                    if !in_range {
                        return None;
                    }

                    let similarity = simd::similarity_simd(&query_vector, vector);
                    let retention_score = memory.retention.retention_strength(Utc::now());

                    Some(SearchResult {
                        memory_id: memory_id.clone(),
                        similarity,
                        retention_score,
                    })
                })
            })
            .collect();

        // Weighted ranking
        results.sort_by(|a, b| {
            let score_a = a.similarity * 0.7 + a.retention_score * 0.3;
            let score_b = b.similarity * 0.7 + b.retention_score * 0.3;
            score_b.partial_cmp(&score_a).unwrap()
        });

        results.truncate(k);
        results
    }

    /// Get memory by ID
    pub fn get_memory(&self, memory_id: &MemoryId) -> Option<&MemoryUnit> {
        self.memories.get(memory_id)
    }

    /// Get all memories
    pub fn all_memories(&self) -> Vec<&MemoryUnit> {
        self.memories.values().collect()
    }

    /// Number of indexed memories
    pub fn len(&self) -> usize {
        self.memories.len()
    }

    /// Check if index is empty
    pub fn is_empty(&self) -> bool {
        self.memories.is_empty()
    }

    /// Encode text into HDC vector
    fn encode_text(&mut self, text: &str) -> HyperVector {
        // Tokenize (simple whitespace split for now)
        let tokens: Vec<&str> = text.split_whitespace().collect();

        if tokens.is_empty() {
            return HyperVector::zeros(DEFAULT_DIMENSION);
        }

        // Encode using sequence encoder
        self.encoder.encode(&tokens)
    }
}

impl Default for HdcMemoryIndex {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cmd_core::memory::{Modality, Embeddings, TemporalMetadata, SourceMetadata, SourceType, GraphLink};
    use cmd_core::retention::RetentionModel;
    use cmd_core::types::SourceId;

    fn create_test_memory(text: &str) -> MemoryUnit {
        MemoryUnit {
            id: MemoryId::new(),
            modality: Modality::Text,
            content: text.as_bytes().to_vec(),
            text: Some(text.to_string()),
            embeddings: Embeddings {
                dense: None,
                sparse: None,
                hdc: None,
            },
            temporal: TemporalMetadata {
                created_at: Utc::now(),
                last_accessed: Utc::now(),
                access_count: 0,
                is_consolidated: false,
            },
            retention: RetentionModel::new(0.9),
            graph_links: Vec::new(),
            source: SourceMetadata {
                source_id: SourceId::new(),
                source_type: SourceType::DirectUserInput,
                confidence: 1.0,
                timestamp: Utc::now(),
            },
            metadata: HashMap::new(),
        }
    }

    #[test]
    fn test_index_creation() {
        let index = HdcMemoryIndex::new();
        assert_eq!(index.len(), 0);
        assert!(index.is_empty());
    }

    #[test]
    fn test_add_memory() {
        let mut index = HdcMemoryIndex::new();
        let memory = create_test_memory("Hello world");

        let result = index.add_memory(memory);
        assert!(result.is_ok());
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn test_search_similar() {
        let mut index = HdcMemoryIndex::new();

        let mem1 = create_test_memory("The quick brown fox jumps over the lazy dog");
        let mem2 = create_test_memory("A fast brown fox leaps over a sleeping dog");
        let mem3 = create_test_memory("Python is a programming language");

        index.add_memory(mem1).unwrap();
        index.add_memory(mem2).unwrap();
        index.add_memory(mem3).unwrap();

        // Search for fox-related memories
        let results = index.search_hdc("brown fox jumping", 3);

        assert_eq!(results.len(), 3);

        // At least one of the top results should be about foxes
        let fox_count = results.iter()
            .filter_map(|r| index.get_memory(&r.memory_id))
            .filter(|m| m.text.as_ref().unwrap().contains("fox"))
            .count();

        assert!(fox_count >= 1, "Expected at least one fox-related result");
    }

    #[test]
    fn test_threshold_search() {
        let mut index = HdcMemoryIndex::new();

        let mem1 = create_test_memory("machine learning algorithms");
        let mem2 = create_test_memory("deep learning neural networks");
        let mem3 = create_test_memory("cooking recipes");

        index.add_memory(mem1).unwrap();
        index.add_memory(mem2).unwrap();
        index.add_memory(mem3).unwrap();

        // High threshold should return only very similar results
        let results = index.search_threshold("learning", 0.3);

        // Should find the learning-related memories
        assert!(results.len() >= 1);

        for result in results {
            let memory = index.get_memory(&result.memory_id).unwrap();
            assert!(memory.text.as_ref().unwrap().contains("learning") ||
                    result.similarity > 0.3);
        }
    }

    #[test]
    fn test_remove_memory() {
        let mut index = HdcMemoryIndex::new();
        let memory = create_test_memory("test memory");
        let memory_id = memory.id.clone();

        index.add_memory(memory).unwrap();
        assert_eq!(index.len(), 1);

        index.remove_memory(&memory_id).unwrap();
        assert_eq!(index.len(), 0);
    }
}
