//! Memory Manager - Unified interface for cognitive memory operations
//!
//! Integrates storage, search, and conflict resolution into a cohesive system.

pub mod config;
pub mod consolidation;

use cmd_core::memory::{MemoryUnit, SourceMetadata, SourceType};
use cmd_core::types::{MemoryId, NodeId};
use cmd_storage::{EpisodicStorage, SemanticStorage, StorageError};
use cmd_search::{HdcMemoryIndex, SearchResult};
use cmd_resolver::deterministic::DeterministicResolver;
use cmd_resolver::types::{Conflict, ConflictType, ResolutionResult};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use std::sync::Arc;
use parking_lot::RwLock;
use dashmap::DashMap;

pub use config::ManagerConfig;

/// Memory Manager errors
#[derive(thiserror::Error, Debug)]
pub enum ManagerError {
    #[error("Storage error: {0}")]
    StorageError(#[from] StorageError),

    #[error("Memory not found: {0}")]
    MemoryNotFound(MemoryId),

    #[error("Conflict resolution failed: {0}")]
    ConflictResolutionFailed(String),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("Consolidation error: {0}")]
    ConsolidationError(String),

    #[error("Unknown error: {0}")]
    Unknown(String),
}

pub type Result<T> = std::result::Result<T, ManagerError>;

/// Statistics about memory manager operations
#[derive(Debug, Clone, Default)]
pub struct ManagerStats {
    pub total_memories: u64,
    pub total_searches: u64,
    pub total_conflicts_resolved: u64,
    pub total_consolidations: u64,
    pub total_forgotten: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
}

/// The main Memory Manager coordinating all memory operations
pub struct MemoryManager<E, S>
where
    E: EpisodicStorage,
    S: SemanticStorage,
{
    /// Episodic storage backend
    episodic_storage: Arc<RwLock<E>>,

    /// Semantic storage backend
    semantic_storage: Arc<RwLock<S>>,

    /// HDC-based search index
    search_index: Arc<RwLock<HdcMemoryIndex>>,

    /// Conflict resolver
    resolver: Arc<RwLock<DeterministicResolver>>,

    /// Configuration
    config: ManagerConfig,

    /// Statistics
    stats: Arc<DashMap<String, u64>>,

    /// Active consolidation tasks
    consolidation_tasks: Arc<DashMap<MemoryId, ConsolidationTask>>,
}

#[derive(Debug, Clone)]
struct ConsolidationTask {
    started_at: DateTime<Utc>,
    status: ConsolidationStatus,
}

#[derive(Debug, Clone, PartialEq)]
enum ConsolidationStatus {
    Pending,
    InProgress,
    Completed,
    Failed(String),
}

impl<E, S> MemoryManager<E, S>
where
    E: EpisodicStorage + Send + Sync + 'static,
    S: SemanticStorage + Send + Sync + 'static,
{
    /// Create a new Memory Manager
    pub fn new(
        episodic_storage: E,
        semantic_storage: S,
        config: ManagerConfig,
    ) -> Self {
        Self {
            episodic_storage: Arc::new(RwLock::new(episodic_storage)),
            semantic_storage: Arc::new(RwLock::new(semantic_storage)),
            search_index: Arc::new(RwLock::new(HdcMemoryIndex::new())),
            resolver: Arc::new(RwLock::new(DeterministicResolver::new())),
            config,
            stats: Arc::new(DashMap::new()),
            consolidation_tasks: Arc::new(DashMap::new()),
        }
    }

    /// Add a new memory to the system
    pub async fn add_memory(&self, mut memory: MemoryUnit) -> Result<MemoryId> {
        let memory_id = memory.id.clone();

        // Check for conflicts with existing memories
        if self.config.enable_conflict_detection {
            if let Err(e) = self.detect_and_resolve_conflicts(&memory).await {
                tracing::warn!("Conflict detection failed: {}", e);
            }
        }

        // Encode with HDC if not already encoded
        if memory.embeddings.hdc.is_none() {
            memory = self.encode_hdc(memory).await?;
        }

        // Store in episodic storage
        {
            let mut storage = self.episodic_storage.write();
            storage.store(memory.clone()).await?;
        }

        // Add to search index
        {
            let mut index = self.search_index.write();
            index.add_memory(memory.clone())
                .map_err(|e| ManagerError::InvalidOperation(e))?;
        }

        // Update statistics
        self.increment_stat("total_memories");

        tracing::debug!("Added memory: {}", memory_id);
        Ok(memory_id)
    }

    /// Retrieve a memory by ID
    pub async fn get_memory(&self, id: &MemoryId) -> Result<MemoryUnit> {
        let storage = self.episodic_storage.read();
        let mut memory = storage.retrieve(id).await?;

        // Record access
        memory.record_access(true);

        // Update in storage
        drop(storage);
        {
            let mut storage = self.episodic_storage.write();
            storage.update(memory.clone()).await?;
        }

        Ok(memory)
    }

    /// Search memories using HDC-based similarity
    pub async fn search(&self, query: &str, k: usize) -> Result<Vec<SearchResult>> {
        self.increment_stat("total_searches");

        let mut index = self.search_index.write();
        let results = index.search_hdc(query, k);

        Ok(results)
    }

    /// Search memories with temporal filter
    pub async fn search_temporal(
        &self,
        query: &str,
        k: usize,
        since: Option<DateTime<Utc>>,
        until: Option<DateTime<Utc>>,
    ) -> Result<Vec<SearchResult>> {
        self.increment_stat("total_searches");

        let mut index = self.search_index.write();
        let results = index.search_temporal(query, k, since, until);

        Ok(results)
    }

    /// Delete a memory
    pub async fn delete_memory(&self, id: &MemoryId) -> Result<()> {
        // Remove from episodic storage
        {
            let mut storage = self.episodic_storage.write();
            storage.delete(id).await?;
        }

        // Remove from search index
        {
            let mut index = self.search_index.write();
            index.remove_memory(id)
                .map_err(|e| ManagerError::InvalidOperation(e))?;
        }

        tracing::debug!("Deleted memory: {}", id);
        Ok(())
    }

    /// Run forgetting process - remove memories below retention threshold
    pub async fn forget_weak_memories(&self, threshold: f32) -> Result<u64> {
        let storage = self.episodic_storage.read();
        let all_ids = storage.list_ids().await?;
        drop(storage);

        let mut forgotten_count = 0u64;

        for id in all_ids {
            let storage = self.episodic_storage.read();
            if let Ok(memory) = storage.retrieve(&id).await {
                if memory.should_forget(threshold) {
                    drop(storage);
                    self.delete_memory(&id).await?;
                    forgotten_count += 1;
                    tracing::debug!("Forgot weak memory: {}", id);
                }
            }
        }

        self.stats.insert("total_forgotten".to_string(), forgotten_count);
        Ok(forgotten_count)
    }

    /// Get memory manager statistics
    pub fn get_stats(&self) -> ManagerStats {
        ManagerStats {
            total_memories: self.get_stat("total_memories"),
            total_searches: self.get_stat("total_searches"),
            total_conflicts_resolved: self.get_stat("total_conflicts_resolved"),
            total_consolidations: self.get_stat("total_consolidations"),
            total_forgotten: self.get_stat("total_forgotten"),
            cache_hits: self.get_stat("cache_hits"),
            cache_misses: self.get_stat("cache_misses"),
        }
    }

    /// Encode memory with HDC
    async fn encode_hdc(&self, mut memory: MemoryUnit) -> Result<MemoryUnit> {
        use cmd_hdc::encoder::SequenceEncoder;
        use cmd_hdc::DEFAULT_DIMENSION;

        if let Some(ref text) = memory.text {
            let mut encoder = SequenceEncoder::new(DEFAULT_DIMENSION);
            let tokens: Vec<&str> = text.split_whitespace().collect();
            let hdc_vector = encoder.encode(&tokens);

            // Convert to bytes
            memory.embeddings.hdc = Some(hdc_vector.to_bytes());
        }

        Ok(memory)
    }

    /// Detect and resolve conflicts
    async fn detect_and_resolve_conflicts(&self, _memory: &MemoryUnit) -> Result<()> {
        // TODO: Implement conflict detection logic
        // For now, just return Ok
        Ok(())
    }

    /// Increment a statistic counter
    fn increment_stat(&self, key: &str) {
        self.stats.entry(key.to_string())
            .and_modify(|v| *v += 1)
            .or_insert(1);
    }

    /// Get a statistic value
    fn get_stat(&self, key: &str) -> u64 {
        self.stats.get(key).map(|v| *v).unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cmd_storage::memory::{InMemoryEpisodicStorage, InMemorySemanticStorage};
    use cmd_core::types::SourceId;

    fn create_test_memory(text: &str) -> MemoryUnit {
        let source = SourceMetadata {
            source_id: SourceId::new(),
            source_type: SourceType::DirectUserInput,
            confidence: 0.9,
            timestamp: Utc::now(),
        };

        MemoryUnit::new_text(text.to_string(), source)
    }

    #[tokio::test]
    async fn test_manager_creation() {
        let episodic = InMemoryEpisodicStorage::new();
        let semantic = InMemorySemanticStorage::new();
        let config = ManagerConfig::default();

        let manager = MemoryManager::new(episodic, semantic, config);
        let stats = manager.get_stats();

        assert_eq!(stats.total_memories, 0);
    }

    #[tokio::test]
    async fn test_add_and_retrieve_memory() {
        let episodic = InMemoryEpisodicStorage::new();
        let semantic = InMemorySemanticStorage::new();
        let config = ManagerConfig::default();

        let manager = MemoryManager::new(episodic, semantic, config);

        let memory = create_test_memory("Test memory");
        let memory_id = memory.id.clone();

        manager.add_memory(memory).await.unwrap();

        let retrieved = manager.get_memory(&memory_id).await.unwrap();
        assert_eq!(retrieved.text, Some("Test memory".to_string()));
        assert_eq!(retrieved.temporal.access_count, 1);
    }

    #[tokio::test]
    async fn test_search_memory() {
        let episodic = InMemoryEpisodicStorage::new();
        let semantic = InMemorySemanticStorage::new();
        let config = ManagerConfig::default();

        let manager = MemoryManager::new(episodic, semantic, config);

        let mem1 = create_test_memory("The quick brown fox");
        let mem2 = create_test_memory("A lazy dog sleeps");
        let mem3 = create_test_memory("Python programming");

        manager.add_memory(mem1).await.unwrap();
        manager.add_memory(mem2).await.unwrap();
        manager.add_memory(mem3).await.unwrap();

        let results = manager.search("fox brown", 5).await.unwrap();
        assert!(results.len() > 0);
    }

    #[tokio::test]
    async fn test_delete_memory() {
        let episodic = InMemoryEpisodicStorage::new();
        let semantic = InMemorySemanticStorage::new();
        let config = ManagerConfig::default();

        let manager = MemoryManager::new(episodic, semantic, config);

        let memory = create_test_memory("To be deleted");
        let memory_id = memory.id.clone();

        manager.add_memory(memory).await.unwrap();
        manager.delete_memory(&memory_id).await.unwrap();

        let result = manager.get_memory(&memory_id).await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_forget_weak_memories() {
        let episodic = InMemoryEpisodicStorage::new();
        let semantic = InMemorySemanticStorage::new();
        let config = ManagerConfig::default();

        let manager = MemoryManager::new(episodic, semantic, config);

        // Add some memories
        for i in 0..5 {
            let memory = create_test_memory(&format!("Memory {}", i));
            manager.add_memory(memory).await.unwrap();
        }

        // Forget with very low threshold (should not forget anything immediately)
        let forgotten = manager.forget_weak_memories(0.1).await.unwrap();
        assert_eq!(forgotten, 0);
    }

    #[tokio::test]
    async fn test_statistics() {
        let episodic = InMemoryEpisodicStorage::new();
        let semantic = InMemorySemanticStorage::new();
        let config = ManagerConfig::default();

        let manager = MemoryManager::new(episodic, semantic, config);

        manager.add_memory(create_test_memory("Memory 1")).await.unwrap();
        manager.add_memory(create_test_memory("Memory 2")).await.unwrap();
        manager.search("test", 5).await.unwrap();

        let stats = manager.get_stats();
        assert_eq!(stats.total_memories, 2);
        assert_eq!(stats.total_searches, 1);
    }
}
