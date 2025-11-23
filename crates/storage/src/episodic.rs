//! Episodic memory storage using LanceDB
//!
//! Provides vector-based storage and retrieval of memories using
//! LanceDB's columnar Parquet format with vector indexing.

use crate::{EpisodicStorage, Result, StorageError, StorageStats};
use cmd_core::memory::{MemoryUnit, Modality};
use cmd_core::types::MemoryId;
use async_trait::async_trait;
use lancedb::{Connection, Table};
use arrow::array::{
    Array, ArrayRef, BinaryArray, Float32Array, RecordBatch, StringArray,
    StructArray, UInt32Array, UInt64Array,
};
use arrow::datatypes::{DataType, Field, Schema};
use serde_json::Value as JsonValue;
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// LanceDB-backed episodic storage
pub struct LanceDbStorage {
    connection: Connection,
    table_name: String,
}

impl LanceDbStorage {
    /// Create a new LanceDB storage instance
    pub async fn new<P: AsRef<Path>>(path: P, table_name: &str) -> Result<Self> {
        let uri = path.as_ref().to_str()
            .ok_or_else(|| StorageError::DatabaseError("Invalid path".to_string()))?;

        let connection = lancedb::connect(uri)
            .execute()
            .await
            .map_err(|e| StorageError::DatabaseError(format!("Failed to connect: {}", e)))?;

        Ok(Self {
            connection,
            table_name: table_name.to_string(),
        })
    }

    /// Initialize the table schema if it doesn't exist
    pub async fn initialize(&mut self) -> Result<()> {
        let schema = Self::create_schema();

        // Check if table exists
        let table_names = self.connection.table_names()
            .execute()
            .await
            .map_err(|e| StorageError::DatabaseError(format!("Failed to list tables: {}", e)))?;

        if !table_names.contains(&self.table_name) {
            // Create empty table with schema
            let empty_batch = RecordBatch::new_empty(Arc::new(schema));

            self.connection
                .create_table(&self.table_name, vec![empty_batch])
                .execute()
                .await
                .map_err(|e| StorageError::DatabaseError(format!("Failed to create table: {}", e)))?;
        }

        Ok(())
    }

    /// Create the Arrow schema for memory storage
    fn create_schema() -> Schema {
        Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("modality", DataType::Utf8, false),
            Field::new("content", DataType::Binary, false),
            Field::new("text", DataType::Utf8, true),
            Field::new("dense_vector", DataType::List(Arc::new(Field::new("item", DataType::Float32, true))), true),
            Field::new("hdc_vector", DataType::Binary, true),
            Field::new("created_at", DataType::UInt64, false),
            Field::new("last_accessed", DataType::UInt64, false),
            Field::new("access_count", DataType::UInt32, false),
            Field::new("is_consolidated", DataType::Boolean, false),
            Field::new("retention_strength", DataType::Float32, false),
            Field::new("source_type", DataType::Utf8, false),
            Field::new("confidence", DataType::Float32, false),
            Field::new("metadata", DataType::Utf8, true),
        ])
    }

    /// Convert MemoryUnit to Arrow RecordBatch
    fn memory_to_record_batch(memories: Vec<MemoryUnit>) -> Result<RecordBatch> {
        let schema = Arc::new(Self::create_schema());

        let mut ids = Vec::new();
        let mut modalities = Vec::new();
        let mut contents = Vec::new();
        let mut texts = Vec::new();
        let mut dense_vectors: Vec<Option<Vec<f32>>> = Vec::new();
        let mut hdc_vectors = Vec::new();
        let mut created_ats = Vec::new();
        let mut last_accesseds = Vec::new();
        let mut access_counts = Vec::new();
        let mut is_consolidated_flags = Vec::new();
        let mut retention_strengths = Vec::new();
        let mut source_types = Vec::new();
        let mut confidences = Vec::new();
        let mut metadatas = Vec::new();

        for memory in memories {
            ids.push(memory.id.to_string());
            modalities.push(format!("{:?}", memory.modality));
            contents.push(memory.content);
            texts.push(memory.text);
            dense_vectors.push(memory.embeddings.dense);
            hdc_vectors.push(memory.embeddings.hdc);
            created_ats.push(memory.temporal.created_at.timestamp_millis() as u64);
            last_accesseds.push(memory.temporal.last_accessed.timestamp_millis() as u64);
            access_counts.push(memory.temporal.access_count);
            is_consolidated_flags.push(memory.temporal.is_consolidated);
            retention_strengths.push(memory.retention_strength());
            source_types.push(format!("{:?}", memory.source.source_type));
            confidences.push(memory.source.confidence);

            let metadata_json = serde_json::to_string(&memory.metadata)
                .map_err(|e| StorageError::SerializationError(e.to_string()))?;
            metadatas.push(Some(metadata_json));
        }

        // Build Arrow arrays
        let id_array = Arc::new(StringArray::from(ids)) as ArrayRef;
        let modality_array = Arc::new(StringArray::from(modalities)) as ArrayRef;
        let content_array = Arc::new(BinaryArray::from(contents)) as ArrayRef;
        let text_array = Arc::new(StringArray::from(texts)) as ArrayRef;

        // TODO: Properly handle dense vector lists
        // For now, storing as null
        let dense_vector_array = Arc::new(arrow::array::ListArray::from(vec![] as Vec<Option<Vec<f32>>>)) as ArrayRef;

        let hdc_vector_array = Arc::new(BinaryArray::from(hdc_vectors)) as ArrayRef;
        let created_at_array = Arc::new(UInt64Array::from(created_ats)) as ArrayRef;
        let last_accessed_array = Arc::new(UInt64Array::from(last_accesseds)) as ArrayRef;
        let access_count_array = Arc::new(UInt32Array::from(access_counts)) as ArrayRef;
        let is_consolidated_array = Arc::new(arrow::array::BooleanArray::from(is_consolidated_flags)) as ArrayRef;
        let retention_strength_array = Arc::new(Float32Array::from(retention_strengths)) as ArrayRef;
        let source_type_array = Arc::new(StringArray::from(source_types)) as ArrayRef;
        let confidence_array = Arc::new(Float32Array::from(confidences)) as ArrayRef;
        let metadata_array = Arc::new(StringArray::from(metadatas)) as ArrayRef;

        RecordBatch::try_new(
            schema,
            vec![
                id_array,
                modality_array,
                content_array,
                text_array,
                dense_vector_array,
                hdc_vector_array,
                created_at_array,
                last_accessed_array,
                access_count_array,
                is_consolidated_array,
                retention_strength_array,
                source_type_array,
                confidence_array,
                metadata_array,
            ],
        )
        .map_err(|e| StorageError::SerializationError(format!("Failed to create batch: {}", e)))
    }

    /// Get or create the table
    async fn get_table(&self) -> Result<Table> {
        self.connection
            .open_table(&self.table_name)
            .execute()
            .await
            .map_err(|e| StorageError::DatabaseError(format!("Failed to open table: {}", e)))
    }
}

#[async_trait]
impl EpisodicStorage for LanceDbStorage {
    async fn store(&mut self, memory: MemoryUnit) -> Result<()> {
        let batch = Self::memory_to_record_batch(vec![memory])?;

        let mut table = self.get_table().await?;
        table.add(vec![batch])
            .execute()
            .await
            .map_err(|e| StorageError::DatabaseError(format!("Failed to store memory: {}", e)))?;

        Ok(())
    }

    async fn retrieve(&self, id: &MemoryId) -> Result<MemoryUnit> {
        let table = self.get_table().await?;

        // Query by ID
        let query = format!("id = '{}'", id);
        let results = table
            .query()
            .filter(&query)
            .execute()
            .await
            .map_err(|e| StorageError::DatabaseError(format!("Query failed: {}", e)))?;

        // TODO: Convert RecordBatch back to MemoryUnit
        // For now, return a placeholder error
        Err(StorageError::MemoryNotFound(id.clone()))
    }

    async fn update(&mut self, memory: MemoryUnit) -> Result<()> {
        // LanceDB doesn't support in-place updates
        // We need to delete and re-insert
        self.delete(&memory.id).await?;
        self.store(memory).await
    }

    async fn delete(&mut self, id: &MemoryId) -> Result<()> {
        let mut table = self.get_table().await?;

        let predicate = format!("id = '{}'", id);
        table.delete(&predicate)
            .await
            .map_err(|e| StorageError::DatabaseError(format!("Delete failed: {}", e)))?;

        Ok(())
    }

    async fn search_vector(
        &self,
        query_vector: &[f32],
        k: usize,
        _filter: Option<HashMap<String, JsonValue>>,
    ) -> Result<Vec<(MemoryId, f32)>> {
        let table = self.get_table().await?;

        // Perform vector similarity search
        // TODO: Implement proper vector search with LanceDB
        // For now return empty results
        Ok(Vec::new())
    }

    async fn search_hdc(
        &self,
        _query_hdc: &[u8],
        _k: usize,
    ) -> Result<Vec<(MemoryId, f32)>> {
        // TODO: Implement HDC vector search
        Ok(Vec::new())
    }

    async fn list_ids(&self) -> Result<Vec<MemoryId>> {
        let table = self.get_table().await?;

        // Get all IDs
        // TODO: Implement proper ID listing
        Ok(Vec::new())
    }

    async fn stats(&self) -> Result<StorageStats> {
        // TODO: Implement stats collection
        Ok(StorageStats {
            total_memories: 0,
            total_nodes: 0,
            total_edges: 0,
            disk_usage_bytes: 0,
            index_size_bytes: 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cmd_core::memory::{SourceMetadata, SourceType};
    use cmd_core::types::SourceId;
    use chrono::Utc;
    use tempfile::tempdir;

    #[tokio::test]
    async fn test_storage_creation() {
        let dir = tempdir().unwrap();
        let storage = LanceDbStorage::new(dir.path(), "memories").await;
        assert!(storage.is_ok());
    }

    #[tokio::test]
    async fn test_schema_creation() {
        let schema = LanceDbStorage::create_schema();
        assert_eq!(schema.fields().len(), 14);
        assert_eq!(schema.field(0).name(), "id");
    }

    #[tokio::test]
    async fn test_initialize() {
        let dir = tempdir().unwrap();
        let mut storage = LanceDbStorage::new(dir.path(), "memories").await.unwrap();
        let result = storage.initialize().await;
        assert!(result.is_ok());
    }
}
