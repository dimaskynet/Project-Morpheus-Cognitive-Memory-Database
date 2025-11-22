//! Memory unit structures and operations

use crate::types::{MemoryId, NodeId, SourceId};
use crate::retention::RetentionModel;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Modality of memory content
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum Modality {
    Text,
    Image,
    Audio,
    Structured,
    Mixed,
}

/// Link to a node in the knowledge graph
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphLink {
    pub node_id: NodeId,
    pub relationship: String,
    pub weight: f32,
}

/// Source metadata for tracking origin of information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SourceMetadata {
    pub source_id: SourceId,
    pub source_type: SourceType,
    pub confidence: f32,
    pub timestamp: DateTime<Utc>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SourceType {
    DirectUserInput,
    ToolOutput,
    LLMExtraction,
    Consolidation,
    Inference,
}

/// The fundamental unit of memory in CMD
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUnit {
    /// Unique identifier (UUIDv7 for temporal sorting)
    pub id: MemoryId,

    /// Type of content
    pub modality: Modality,

    /// Raw content payload
    pub content: Vec<u8>,

    /// Human-readable text representation (if applicable)
    pub text: Option<String>,

    /// Vector embeddings
    pub embeddings: Embeddings,

    /// Temporal characteristics
    pub temporal: TemporalMetadata,

    /// Retention model for forgetting curve
    pub retention: RetentionModel,

    /// Links to knowledge graph nodes
    pub graph_links: Vec<GraphLink>,

    /// Source information
    pub source: SourceMetadata,

    /// Additional metadata
    pub metadata: HashMap<String, serde_json::Value>,
}

/// Different vector representations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Embeddings {
    /// Dense vector (e.g., OpenAI embeddings)
    pub dense: Option<Vec<f32>>,

    /// Sparse vector (e.g., BM25/SPLADE for keyword search)
    pub sparse: Option<HashMap<u32, f32>>,

    /// HDC binary vector for structural encoding
    pub hdc: Option<Vec<u8>>,
}

/// Temporal metadata for tracking memory lifecycle
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TemporalMetadata {
    pub created_at: DateTime<Utc>,
    pub last_accessed: DateTime<Utc>,
    pub access_count: u32,
    pub is_consolidated: bool,
}

impl MemoryUnit {
    /// Create a new memory unit with text content
    pub fn new_text(content: String, source: SourceMetadata) -> Self {
        let now = Utc::now();

        Self {
            id: MemoryId::new(),
            modality: Modality::Text,
            content: content.as_bytes().to_vec(),
            text: Some(content),
            embeddings: Embeddings {
                dense: None,
                sparse: None,
                hdc: None,
            },
            temporal: TemporalMetadata {
                created_at: now,
                last_accessed: now,
                access_count: 0,
                is_consolidated: false,
            },
            retention: RetentionModel::new(source.confidence),
            graph_links: Vec::new(),
            source,
            metadata: HashMap::new(),
        }
    }

    /// Record access to this memory
    pub fn record_access(&mut self, success: bool) {
        self.temporal.last_accessed = Utc::now();
        self.temporal.access_count += 1;
        self.retention.update_on_recall(success);
    }

    /// Calculate current retention strength
    pub fn retention_strength(&self) -> f32 {
        self.retention.retention_strength(Utc::now())
    }

    /// Check if memory should be forgotten
    pub fn should_forget(&self, threshold: f32) -> bool {
        self.retention_strength() < threshold
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_unit_creation() {
        let source = SourceMetadata {
            source_id: SourceId::new(),
            source_type: SourceType::DirectUserInput,
            confidence: 0.9,
            timestamp: Utc::now(),
        };

        let memory = MemoryUnit::new_text(
            "User prefers dark mode".to_string(),
            source,
        );

        assert_eq!(memory.modality, Modality::Text);
        assert_eq!(memory.text, Some("User prefers dark mode".to_string()));
        assert!(!memory.temporal.is_consolidated);
        assert_eq!(memory.temporal.access_count, 0);
    }

    #[test]
    fn test_memory_access_tracking() {
        let source = SourceMetadata {
            source_id: SourceId::new(),
            source_type: SourceType::DirectUserInput,
            confidence: 0.9,
            timestamp: Utc::now(),
        };

        let mut memory = MemoryUnit::new_text(
            "Test memory".to_string(),
            source,
        );

        let initial_strength = memory.retention_strength();
        memory.record_access(true);

        assert_eq!(memory.temporal.access_count, 1);
        // After successful recall, stability should increase
        assert!(memory.retention.stability > 1.0);
    }
}