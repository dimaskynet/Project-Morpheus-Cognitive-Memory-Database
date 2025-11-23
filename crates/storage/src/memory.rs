//! In-memory storage implementation for testing and development
//!
//! Provides simple HashMap-based storage without external dependencies.

use crate::{EpisodicStorage, GraphNode, Result, SemanticStorage, StorageError, StorageStats};
use cmd_core::memory::MemoryUnit;
use cmd_core::types::{MemoryId, NodeId};
use async_trait::async_trait;
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;

/// In-memory episodic storage
#[derive(Clone)]
pub struct InMemoryEpisodicStorage {
    memories: Arc<RwLock<HashMap<MemoryId, MemoryUnit>>>,
}

impl InMemoryEpisodicStorage {
    /// Create a new in-memory storage
    pub fn new() -> Self {
        Self {
            memories: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Get number of stored memories
    pub fn len(&self) -> usize {
        self.memories.read().len()
    }

    /// Check if storage is empty
    pub fn is_empty(&self) -> bool {
        self.memories.read().is_empty()
    }
}

impl Default for InMemoryEpisodicStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl EpisodicStorage for InMemoryEpisodicStorage {
    async fn store(&mut self, memory: MemoryUnit) -> Result<()> {
        self.memories.write().insert(memory.id.clone(), memory);
        Ok(())
    }

    async fn retrieve(&self, id: &MemoryId) -> Result<MemoryUnit> {
        self.memories
            .read()
            .get(id)
            .cloned()
            .ok_or_else(|| StorageError::MemoryNotFound(id.clone()))
    }

    async fn update(&mut self, memory: MemoryUnit) -> Result<()> {
        if !self.memories.read().contains_key(&memory.id) {
            return Err(StorageError::MemoryNotFound(memory.id.clone()));
        }
        self.memories.write().insert(memory.id.clone(), memory);
        Ok(())
    }

    async fn delete(&mut self, id: &MemoryId) -> Result<()> {
        self.memories
            .write()
            .remove(id)
            .ok_or_else(|| StorageError::MemoryNotFound(id.clone()))?;
        Ok(())
    }

    async fn search_vector(
        &self,
        query_vector: &[f32],
        k: usize,
        _filter: Option<HashMap<String, JsonValue>>,
    ) -> Result<Vec<(MemoryId, f32)>> {
        let memories = self.memories.read();

        // Simple cosine similarity search
        let mut results: Vec<(MemoryId, f32)> = memories
            .iter()
            .filter_map(|(id, mem)| {
                mem.embeddings.dense.as_ref().map(|dense| {
                    let similarity = cosine_similarity(query_vector, dense);
                    (id.clone(), similarity)
                })
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(k);

        Ok(results)
    }

    async fn search_hdc(
        &self,
        query_hdc: &[u8],
        k: usize,
    ) -> Result<Vec<(MemoryId, f32)>> {
        let memories = self.memories.read();

        // Simple Hamming distance search
        let mut results: Vec<(MemoryId, f32)> = memories
            .iter()
            .filter_map(|(id, mem)| {
                mem.embeddings.hdc.as_ref().map(|hdc| {
                    let distance = hamming_distance(query_hdc, hdc);
                    let similarity = 1.0 - (distance as f32 / (query_hdc.len() * 8) as f32);
                    (id.clone(), similarity)
                })
            })
            .collect();

        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        results.truncate(k);

        Ok(results)
    }

    async fn list_ids(&self) -> Result<Vec<MemoryId>> {
        Ok(self.memories.read().keys().cloned().collect())
    }

    async fn stats(&self) -> Result<StorageStats> {
        let count = self.memories.read().len() as u64;
        Ok(StorageStats {
            total_memories: count,
            total_nodes: 0,
            total_edges: 0,
            disk_usage_bytes: 0,
            index_size_bytes: 0,
        })
    }
}

/// In-memory semantic storage
#[derive(Clone)]
pub struct InMemorySemanticStorage {
    nodes: Arc<RwLock<HashMap<NodeId, GraphNode>>>,
    edges: Arc<RwLock<Vec<GraphEdge>>>,
}

#[derive(Debug, Clone)]
struct GraphEdge {
    from: NodeId,
    to: NodeId,
    relationship: String,
    properties: HashMap<String, JsonValue>,
}

impl InMemorySemanticStorage {
    /// Create a new in-memory semantic storage
    pub fn new() -> Self {
        Self {
            nodes: Arc::new(RwLock::new(HashMap::new())),
            edges: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Get number of nodes
    pub fn node_count(&self) -> usize {
        self.nodes.read().len()
    }

    /// Get number of edges
    pub fn edge_count(&self) -> usize {
        self.edges.read().len()
    }
}

impl Default for InMemorySemanticStorage {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait]
impl SemanticStorage for InMemorySemanticStorage {
    async fn create_node(
        &mut self,
        node_id: NodeId,
        labels: Vec<String>,
        properties: HashMap<String, JsonValue>,
    ) -> Result<()> {
        let node = GraphNode {
            id: node_id.clone(),
            labels,
            properties,
        };
        self.nodes.write().insert(node_id, node);
        Ok(())
    }

    async fn get_node(&self, node_id: &NodeId) -> Result<GraphNode> {
        self.nodes
            .read()
            .get(node_id)
            .cloned()
            .ok_or_else(|| StorageError::NodeNotFound(node_id.clone()))
    }

    async fn create_edge(
        &mut self,
        from: NodeId,
        to: NodeId,
        relationship: String,
        properties: HashMap<String, JsonValue>,
    ) -> Result<()> {
        // Verify both nodes exist
        if !self.nodes.read().contains_key(&from) {
            return Err(StorageError::NodeNotFound(from));
        }
        if !self.nodes.read().contains_key(&to) {
            return Err(StorageError::NodeNotFound(to.clone()));
        }

        let edge = GraphEdge {
            from,
            to,
            relationship,
            properties,
        };
        self.edges.write().push(edge);
        Ok(())
    }

    async fn query(
        &self,
        _query: &str,
        _params: HashMap<String, JsonValue>,
    ) -> Result<Vec<HashMap<String, JsonValue>>> {
        // Simple in-memory implementation doesn't support full query language
        Ok(Vec::new())
    }

    async fn get_neighbors(
        &self,
        node_id: &NodeId,
        relationship_type: Option<String>,
        depth: usize,
    ) -> Result<Vec<GraphNode>> {
        if depth == 0 {
            return Ok(Vec::new());
        }

        let edges = self.edges.read();
        let nodes = self.nodes.read();

        let mut visited = HashMap::new();
        visited.insert(node_id.clone(), 0);

        let mut queue = vec![(node_id.clone(), 0)];
        let mut neighbors = Vec::new();

        while let Some((current_id, current_depth)) = queue.pop() {
            if current_depth >= depth {
                continue;
            }

            // Find outgoing edges
            for edge in edges.iter() {
                if edge.from == current_id {
                    // Check relationship type filter
                    if let Some(ref rel_type) = relationship_type {
                        if &edge.relationship != rel_type {
                            continue;
                        }
                    }

                    // Add neighbor if not visited at this depth or closer
                    let next_depth = current_depth + 1;
                    if !visited.contains_key(&edge.to)
                        || visited[&edge.to] > next_depth
                    {
                        visited.insert(edge.to.clone(), next_depth);
                        queue.push((edge.to.clone(), next_depth));

                        if let Some(node) = nodes.get(&edge.to) {
                            neighbors.push(node.clone());
                        }
                    }
                }
            }
        }

        Ok(neighbors)
    }

    async fn delete_node(&mut self, node_id: &NodeId) -> Result<()> {
        self.nodes
            .write()
            .remove(node_id)
            .ok_or_else(|| StorageError::NodeNotFound(node_id.clone()))?;

        // Remove all edges connected to this node
        self.edges
            .write()
            .retain(|edge| edge.from != *node_id && edge.to != *node_id);

        Ok(())
    }
}

/// Calculate cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        return 0.0;
    }

    dot_product / (norm_a * norm_b)
}

/// Calculate Hamming distance between two byte arrays
fn hamming_distance(a: &[u8], b: &[u8]) -> usize {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x ^ y).count_ones() as usize)
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cmd_core::memory::{SourceMetadata, SourceType};
    use cmd_core::types::SourceId;
    use chrono::Utc;

    #[tokio::test]
    async fn test_episodic_store_retrieve() {
        let mut storage = InMemoryEpisodicStorage::new();

        let source = SourceMetadata {
            source_id: SourceId::new(),
            source_type: SourceType::DirectUserInput,
            confidence: 0.9,
            timestamp: Utc::now(),
        };

        let memory = MemoryUnit::new_text("Test memory".to_string(), source);
        let memory_id = memory.id.clone();

        storage.store(memory).await.unwrap();

        let retrieved = storage.retrieve(&memory_id).await.unwrap();
        assert_eq!(retrieved.text, Some("Test memory".to_string()));
    }

    #[tokio::test]
    async fn test_episodic_delete() {
        let mut storage = InMemoryEpisodicStorage::new();

        let source = SourceMetadata {
            source_id: SourceId::new(),
            source_type: SourceType::DirectUserInput,
            confidence: 0.9,
            timestamp: Utc::now(),
        };

        let memory = MemoryUnit::new_text("Test".to_string(), source);
        let memory_id = memory.id.clone();

        storage.store(memory).await.unwrap();
        assert_eq!(storage.len(), 1);

        storage.delete(&memory_id).await.unwrap();
        assert_eq!(storage.len(), 0);
    }

    #[tokio::test]
    async fn test_semantic_node_creation() {
        let mut storage = InMemorySemanticStorage::new();

        let node_id = NodeId::new("node1");
        let labels = vec!["Person".to_string()];
        let mut properties = HashMap::new();
        properties.insert("name".to_string(), JsonValue::String("Alice".to_string()));

        storage
            .create_node(node_id.clone(), labels, properties)
            .await
            .unwrap();

        let node = storage.get_node(&node_id).await.unwrap();
        assert_eq!(node.labels[0], "Person");
    }

    #[tokio::test]
    async fn test_semantic_edge_creation() {
        let mut storage = InMemorySemanticStorage::new();

        let node1 = NodeId::new("node1");
        let node2 = NodeId::new("node2");

        storage
            .create_node(node1.clone(), vec!["Person".to_string()], HashMap::new())
            .await
            .unwrap();
        storage
            .create_node(node2.clone(), vec!["Person".to_string()], HashMap::new())
            .await
            .unwrap();

        storage
            .create_edge(node1.clone(), node2.clone(), "KNOWS".to_string(), HashMap::new())
            .await
            .unwrap();

        assert_eq!(storage.edge_count(), 1);
    }

    #[tokio::test]
    async fn test_semantic_neighbors() {
        let mut storage = InMemorySemanticStorage::new();

        let node1 = NodeId::new("node1");
        let node2 = NodeId::new("node2");
        let node3 = NodeId::new("node3");

        storage
            .create_node(node1.clone(), vec!["A".to_string()], HashMap::new())
            .await
            .unwrap();
        storage
            .create_node(node2.clone(), vec!["B".to_string()], HashMap::new())
            .await
            .unwrap();
        storage
            .create_node(node3.clone(), vec!["C".to_string()], HashMap::new())
            .await
            .unwrap();

        storage
            .create_edge(node1.clone(), node2.clone(), "REL".to_string(), HashMap::new())
            .await
            .unwrap();
        storage
            .create_edge(node2.clone(), node3.clone(), "REL".to_string(), HashMap::new())
            .await
            .unwrap();

        let neighbors = storage.get_neighbors(&node1, None, 2).await.unwrap();
        assert_eq!(neighbors.len(), 2);
    }

    #[test]
    fn test_cosine_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert_eq!(cosine_similarity(&a, &b), 1.0);

        let c = vec![1.0, 0.0, 0.0];
        let d = vec![0.0, 1.0, 0.0];
        assert_eq!(cosine_similarity(&c, &d), 0.0);
    }

    #[test]
    fn test_hamming_distance() {
        let a = &[0b10101010];
        let b = &[0b10101010];
        assert_eq!(hamming_distance(a, b), 0);

        let c = &[0b11111111];
        let d = &[0b00000000];
        assert_eq!(hamming_distance(c, d), 8);
    }
}
