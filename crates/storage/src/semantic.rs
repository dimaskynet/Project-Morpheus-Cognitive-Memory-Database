//! Semantic memory storage using KuzuDB
//!
//! Provides graph-based storage for knowledge relationships using
//! KuzuDB's embedded graph database.

use crate::{GraphNode, Result, SemanticStorage, StorageError};
use cmd_core::types::NodeId;
use async_trait::async_trait;
use kuzu::{Connection, Database, SystemConfig};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// KuzuDB-backed semantic storage
pub struct KuzuDbStorage {
    database: Arc<Database>,
    connection: Connection,
}

impl KuzuDbStorage {
    /// Create a new KuzuDB storage instance
    pub async fn new<P: AsRef<Path>>(path: P) -> Result<Self> {
        let path_str = path.as_ref().to_str()
            .ok_or_else(|| StorageError::DatabaseError("Invalid path".to_string()))?;

        let system_config = SystemConfig::default();
        let database = Database::new(path_str, system_config)
            .map_err(|e| StorageError::DatabaseError(format!("Failed to open database: {:?}", e)))?;

        let database = Arc::new(database);
        let connection = Connection::new(&database)
            .map_err(|e| StorageError::DatabaseError(format!("Failed to create connection: {:?}", e)))?;

        Ok(Self {
            database,
            connection,
        })
    }

    /// Initialize the graph schema
    pub async fn initialize(&mut self) -> Result<()> {
        // Create Node table
        self.connection
            .query("CREATE NODE TABLE IF NOT EXISTS MemoryNode(id STRING, labels STRING[], properties STRING, PRIMARY KEY(id))")
            .map_err(|e| StorageError::DatabaseError(format!("Failed to create node table: {:?}", e)))?
            .execute()
            .map_err(|e| StorageError::DatabaseError(format!("Failed to execute: {:?}", e)))?;

        // Create Relationship table
        self.connection
            .query("CREATE REL TABLE IF NOT EXISTS Relationship(FROM MemoryNode TO MemoryNode, type STRING, properties STRING)")
            .map_err(|e| StorageError::DatabaseError(format!("Failed to create rel table: {:?}", e)))?
            .execute()
            .map_err(|e| StorageError::DatabaseError(format!("Failed to execute: {:?}", e)))?;

        Ok(())
    }

    /// Convert properties HashMap to JSON string
    fn properties_to_json(properties: &HashMap<String, JsonValue>) -> Result<String> {
        serde_json::to_string(properties)
            .map_err(|e| StorageError::SerializationError(e.to_string()))
    }

    /// Convert JSON string to properties HashMap
    fn json_to_properties(json: &str) -> Result<HashMap<String, JsonValue>> {
        serde_json::from_str(json)
            .map_err(|e| StorageError::SerializationError(e.to_string()))
    }

    /// Convert labels Vec to JSON array string
    fn labels_to_json(labels: &[String]) -> Result<String> {
        serde_json::to_string(labels)
            .map_err(|e| StorageError::SerializationError(e.to_string()))
    }
}

#[async_trait]
impl SemanticStorage for KuzuDbStorage {
    async fn create_node(
        &mut self,
        node_id: NodeId,
        labels: Vec<String>,
        properties: HashMap<String, JsonValue>,
    ) -> Result<()> {
        let id_str = node_id.to_string();
        let labels_json = Self::labels_to_json(&labels)?;
        let properties_json = Self::properties_to_json(&properties)?;

        let query = format!(
            "CREATE (n:MemoryNode {{id: '{}', labels: {}, properties: '{}'}})",
            id_str,
            labels_json,
            properties_json.replace('\'', "''")
        );

        self.connection
            .query(&query)
            .map_err(|e| StorageError::DatabaseError(format!("Query creation failed: {:?}", e)))?
            .execute()
            .map_err(|e| StorageError::DatabaseError(format!("Node creation failed: {:?}", e)))?;

        Ok(())
    }

    async fn get_node(&self, node_id: &NodeId) -> Result<GraphNode> {
        let id_str = node_id.to_string();

        let query = format!(
            "MATCH (n:MemoryNode) WHERE n.id = '{}' RETURN n.id, n.labels, n.properties",
            id_str
        );

        let mut result = self.connection
            .query(&query)
            .map_err(|e| StorageError::DatabaseError(format!("Query failed: {:?}", e)))?
            .execute()
            .map_err(|e| StorageError::DatabaseError(format!("Execute failed: {:?}", e)))?;

        // TODO: Parse result and convert to GraphNode
        // For now, return a placeholder
        Err(StorageError::NodeNotFound(node_id.clone()))
    }

    async fn create_edge(
        &mut self,
        from: NodeId,
        to: NodeId,
        relationship: String,
        properties: HashMap<String, JsonValue>,
    ) -> Result<()> {
        let from_str = from.to_string();
        let to_str = to.to_string();
        let properties_json = Self::properties_to_json(&properties)?;

        let query = format!(
            "MATCH (a:MemoryNode), (b:MemoryNode) WHERE a.id = '{}' AND b.id = '{}' CREATE (a)-[:Relationship {{type: '{}', properties: '{}'}}]->(b)",
            from_str,
            to_str,
            relationship,
            properties_json.replace('\'', "''")
        );

        self.connection
            .query(&query)
            .map_err(|e| StorageError::DatabaseError(format!("Query creation failed: {:?}", e)))?
            .execute()
            .map_err(|e| StorageError::DatabaseError(format!("Edge creation failed: {:?}", e)))?;

        Ok(())
    }

    async fn query(
        &self,
        query_str: &str,
        _params: HashMap<String, JsonValue>,
    ) -> Result<Vec<HashMap<String, JsonValue>>> {
        let mut result = self.connection
            .query(query_str)
            .map_err(|e| StorageError::DatabaseError(format!("Query failed: {:?}", e)))?
            .execute()
            .map_err(|e| StorageError::DatabaseError(format!("Execute failed: {:?}", e)))?;

        // TODO: Parse results into HashMap
        Ok(Vec::new())
    }

    async fn get_neighbors(
        &self,
        node_id: &NodeId,
        relationship_type: Option<String>,
        depth: usize,
    ) -> Result<Vec<GraphNode>> {
        let id_str = node_id.to_string();

        let rel_filter = match relationship_type {
            Some(ref t) => format!("WHERE r.type = '{}'", t),
            None => String::new(),
        };

        let query = format!(
            "MATCH (n:MemoryNode)-[r:Relationship*1..{}]-(neighbor:MemoryNode) WHERE n.id = '{}' {} RETURN DISTINCT neighbor.id, neighbor.labels, neighbor.properties",
            depth,
            id_str,
            rel_filter
        );

        let mut result = self.connection
            .query(&query)
            .map_err(|e| StorageError::DatabaseError(format!("Query failed: {:?}", e)))?
            .execute()
            .map_err(|e| StorageError::DatabaseError(format!("Execute failed: {:?}", e)))?;

        // TODO: Parse results into GraphNode vector
        Ok(Vec::new())
    }

    async fn delete_node(&mut self, node_id: &NodeId) -> Result<()> {
        let id_str = node_id.to_string();

        let query = format!(
            "MATCH (n:MemoryNode) WHERE n.id = '{}' DETACH DELETE n",
            id_str
        );

        self.connection
            .query(&query)
            .map_err(|e| StorageError::DatabaseError(format!("Query failed: {:?}", e)))?
            .execute()
            .map_err(|e| StorageError::DatabaseError(format!("Delete failed: {:?}", e)))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_storage_creation() {
        let dir = tempdir().unwrap();
        let storage = KuzuDbStorage::new(dir.path()).await;
        assert!(storage.is_ok());
    }

    #[tokio::test]
    async fn test_initialize() {
        let dir = tempdir().unwrap();
        let mut storage = KuzuDbStorage::new(dir.path()).await.unwrap();
        let result = storage.initialize().await;
        assert!(result.is_ok());
    }

    #[tokio::test]
    async fn test_properties_serialization() {
        let mut props = HashMap::new();
        props.insert("key".to_string(), JsonValue::String("value".to_string()));

        let json = KuzuDbStorage::properties_to_json(&props).unwrap();
        assert!(json.contains("key"));

        let parsed = KuzuDbStorage::json_to_properties(&json).unwrap();
        assert_eq!(parsed.get("key").unwrap().as_str().unwrap(), "value");
    }
}
