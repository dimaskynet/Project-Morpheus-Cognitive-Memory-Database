//! Storage layer for Cognitive Memory Database
//!
//! Provides persistent storage for memories using:
//! - Episodic memory: LanceDB for vector storage with Parquet backing
//! - Semantic memory: KuzuDB for graph relationships
//! - In-memory: Simple HashMap-based storage for testing

#[cfg(feature = "lancedb")]
pub mod episodic;

#[cfg(feature = "kuzu")]
pub mod semantic;

pub mod memory;

use cmd_core::memory::MemoryUnit;
use cmd_core::types::{MemoryId, NodeId};
use async_trait::async_trait;
use std::collections::HashMap;

/// Storage errors
#[derive(thiserror::Error, Debug)]
pub enum StorageError {
    #[error("Memory not found: {0}")]
    MemoryNotFound(MemoryId),

    #[error("Node not found: {0}")]
    NodeNotFound(NodeId),

    #[error("Database error: {0}")]
    DatabaseError(String),

    #[error("Serialization error: {0}")]
    SerializationError(String),

    #[error("IO error: {0}")]
    IoError(#[from] std::io::Error),

    #[error("Unknown error: {0}")]
    Unknown(String),
}

pub type Result<T> = std::result::Result<T, StorageError>;

/// Trait for episodic memory storage (vector-based)
#[async_trait]
pub trait EpisodicStorage: Send + Sync {
    /// Store a memory unit
    async fn store(&mut self, memory: MemoryUnit) -> Result<()>;

    /// Retrieve a memory by ID
    async fn retrieve(&self, id: &MemoryId) -> Result<MemoryUnit>;

    /// Update an existing memory
    async fn update(&mut self, memory: MemoryUnit) -> Result<()>;

    /// Delete a memory
    async fn delete(&mut self, id: &MemoryId) -> Result<()>;

    /// Search by vector similarity
    async fn search_vector(
        &self,
        query_vector: &[f32],
        k: usize,
        filter: Option<HashMap<String, serde_json::Value>>,
    ) -> Result<Vec<(MemoryId, f32)>>;

    /// Search by HDC vector
    async fn search_hdc(
        &self,
        query_hdc: &[u8],
        k: usize,
    ) -> Result<Vec<(MemoryId, f32)>>;

    /// List all memory IDs
    async fn list_ids(&self) -> Result<Vec<MemoryId>>;

    /// Get storage statistics
    async fn stats(&self) -> Result<StorageStats>;
}

/// Trait for semantic memory storage (graph-based)
#[async_trait]
pub trait SemanticStorage: Send + Sync {
    /// Create a node in the knowledge graph
    async fn create_node(
        &mut self,
        node_id: NodeId,
        labels: Vec<String>,
        properties: HashMap<String, serde_json::Value>,
    ) -> Result<()>;

    /// Get node properties
    async fn get_node(&self, node_id: &NodeId) -> Result<GraphNode>;

    /// Create a relationship between nodes
    async fn create_edge(
        &mut self,
        from: NodeId,
        to: NodeId,
        relationship: String,
        properties: HashMap<String, serde_json::Value>,
    ) -> Result<()>;

    /// Query the graph using Cypher-like syntax
    async fn query(
        &self,
        query: &str,
        params: HashMap<String, serde_json::Value>,
    ) -> Result<Vec<HashMap<String, serde_json::Value>>>;

    /// Find neighbors of a node
    async fn get_neighbors(
        &self,
        node_id: &NodeId,
        relationship_type: Option<String>,
        depth: usize,
    ) -> Result<Vec<GraphNode>>;

    /// Delete a node and its relationships
    async fn delete_node(&mut self, node_id: &NodeId) -> Result<()>;
}

/// Graph node representation
#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: NodeId,
    pub labels: Vec<String>,
    pub properties: HashMap<String, serde_json::Value>,
}

/// Storage statistics
#[derive(Debug, Clone)]
pub struct StorageStats {
    pub total_memories: u64,
    pub total_nodes: u64,
    pub total_edges: u64,
    pub disk_usage_bytes: u64,
    pub index_size_bytes: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_storage_error() {
        let err = StorageError::MemoryNotFound(MemoryId::new());
        assert!(err.to_string().contains("Memory not found"));
    }
}
