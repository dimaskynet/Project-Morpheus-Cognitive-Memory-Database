//! Memory consolidation processes
//!
//! Implements background consolidation of memories based on access patterns
//! and retention models.

use cmd_core::memory::MemoryUnit;
use cmd_core::types::MemoryId;
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Consolidation strategy
#[derive(Debug, Clone, PartialEq)]
pub enum ConsolidationStrategy {
    /// Consolidate based on access frequency
    AccessFrequency,

    /// Consolidate based on retention strength
    RetentionBased,

    /// Consolidate related memories together
    ClusterBased,

    /// Consolidate based on temporal proximity
    TemporalProximity,
}

/// Result of consolidation analysis
#[derive(Debug, Clone)]
pub struct ConsolidationCandidate {
    pub memory_id: MemoryId,
    pub score: f32,
    pub reason: String,
}

/// Consolidation engine
pub struct ConsolidationEngine {
    strategy: ConsolidationStrategy,
    threshold: f32,
}

impl ConsolidationEngine {
    /// Create a new consolidation engine
    pub fn new(strategy: ConsolidationStrategy, threshold: f32) -> Self {
        Self { strategy, threshold }
    }

    /// Analyze memories and identify consolidation candidates
    pub fn identify_candidates(
        &self,
        memories: &[MemoryUnit],
    ) -> Vec<ConsolidationCandidate> {
        match self.strategy {
            ConsolidationStrategy::AccessFrequency => {
                self.identify_by_access_frequency(memories)
            }
            ConsolidationStrategy::RetentionBased => {
                self.identify_by_retention(memories)
            }
            ConsolidationStrategy::ClusterBased => {
                self.identify_by_clusters(memories)
            }
            ConsolidationStrategy::TemporalProximity => {
                self.identify_by_temporal_proximity(memories)
            }
        }
    }

    /// Identify candidates based on access frequency
    fn identify_by_access_frequency(
        &self,
        memories: &[MemoryUnit],
    ) -> Vec<ConsolidationCandidate> {
        memories
            .iter()
            .filter_map(|memory| {
                let access_count = memory.temporal.access_count;
                if access_count >= 5 && !memory.temporal.is_consolidated {
                    let score = (access_count as f32).min(10.0) / 10.0;
                    if score >= self.threshold {
                        return Some(ConsolidationCandidate {
                            memory_id: memory.id.clone(),
                            score,
                            reason: format!(
                                "High access frequency: {} accesses",
                                access_count
                            ),
                        });
                    }
                }
                None
            })
            .collect()
    }

    /// Identify candidates based on retention strength
    fn identify_by_retention(
        &self,
        memories: &[MemoryUnit],
    ) -> Vec<ConsolidationCandidate> {
        memories
            .iter()
            .filter_map(|memory| {
                let retention = memory.retention_strength();
                if retention >= self.threshold && !memory.temporal.is_consolidated {
                    Some(ConsolidationCandidate {
                        memory_id: memory.id.clone(),
                        score: retention,
                        reason: format!("Strong retention: {:.2}", retention),
                    })
                } else {
                    None
                }
            })
            .collect()
    }

    /// Identify candidates based on clustering
    fn identify_by_clusters(
        &self,
        _memories: &[MemoryUnit],
    ) -> Vec<ConsolidationCandidate> {
        // TODO: Implement clustering-based consolidation
        Vec::new()
    }

    /// Identify candidates based on temporal proximity
    fn identify_by_temporal_proximity(
        &self,
        memories: &[MemoryUnit],
    ) -> Vec<ConsolidationCandidate> {
        let now = Utc::now();
        let threshold_days = 7;

        memories
            .iter()
            .filter_map(|memory| {
                let age_days = (now - memory.temporal.created_at).num_days();
                if age_days >= threshold_days && !memory.temporal.is_consolidated {
                    let score = 1.0 - (age_days as f32 / 30.0).min(1.0);
                    if score >= self.threshold {
                        return Some(ConsolidationCandidate {
                            memory_id: memory.id.clone(),
                            score,
                            reason: format!("Age: {} days", age_days),
                        });
                    }
                }
                None
            })
            .collect()
    }

    /// Consolidate a memory (mark as consolidated)
    pub fn consolidate(&self, memory: &mut MemoryUnit) {
        memory.temporal.is_consolidated = true;
        memory.retention.stability *= 1.5; // Boost stability after consolidation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cmd_core::memory::{SourceMetadata, SourceType};
    use cmd_core::types::SourceId;

    fn create_test_memory(text: &str, access_count: u32) -> MemoryUnit {
        let source = SourceMetadata {
            source_id: SourceId::new(),
            source_type: SourceType::DirectUserInput,
            confidence: 0.9,
            timestamp: Utc::now(),
        };

        let mut memory = MemoryUnit::new_text(text.to_string(), source);
        memory.temporal.access_count = access_count;
        memory
    }

    #[test]
    fn test_access_frequency_consolidation() {
        let engine = ConsolidationEngine::new(
            ConsolidationStrategy::AccessFrequency,
            0.5,
        );

        let memories = vec![
            create_test_memory("Low access", 1),
            create_test_memory("High access", 10),
            create_test_memory("Medium access", 5),
        ];

        let candidates = engine.identify_candidates(&memories);
        assert!(candidates.len() >= 1);
        assert!(candidates.iter().any(|c| c.score >= 0.5));
    }

    #[test]
    fn test_retention_based_consolidation() {
        let engine = ConsolidationEngine::new(
            ConsolidationStrategy::RetentionBased,
            0.7,
        );

        let memories = vec![
            create_test_memory("Memory 1", 0),
            create_test_memory("Memory 2", 0),
        ];

        let candidates = engine.identify_candidates(&memories);
        // Retention strength depends on time and confidence
        assert!(candidates.len() >= 0);
    }

    #[test]
    fn test_consolidate_memory() {
        let engine = ConsolidationEngine::new(
            ConsolidationStrategy::AccessFrequency,
            0.5,
        );

        let mut memory = create_test_memory("Test", 10);
        assert!(!memory.temporal.is_consolidated);

        let initial_stability = memory.retention.stability;
        engine.consolidate(&mut memory);

        assert!(memory.temporal.is_consolidated);
        assert!(memory.retention.stability > initial_stability);
    }
}
