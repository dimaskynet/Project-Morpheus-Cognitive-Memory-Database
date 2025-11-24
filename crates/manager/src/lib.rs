//! Memory Manager - Unified interface for cognitive memory operations
//!
//! Integrates storage, search, and conflict resolution into a cohesive system.

pub mod config;
pub mod consolidation;

use cmd_core::memory::{MemoryUnit, SourceMetadata, SourceType, EmotionalValence, PADVector, GoalStatus};
use cmd_core::types::{MemoryId, NodeId};
use cmd_storage::{EpisodicStorage, SemanticStorage, StorageError};
use cmd_search::{HdcMemoryIndex, SearchResult};
use cmd_resolver::deterministic::DeterministicResolver;
use cmd_resolver::types::{Conflict, ConflictType, ResolutionResult};
use async_trait::async_trait;
use chrono::{DateTime, Utc};
use std::sync::Arc;
use tokio::sync::RwLock;
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

/// Statistics about emotional memories
#[derive(Debug, Clone, Default)]
pub struct EmotionalStats {
    pub positive_count: u64,
    pub negative_count: u64,
    pub neutral_count: u64,
    pub average_intensity: f32,
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
            let mut storage = self.episodic_storage.write().await;
            storage.store(memory.clone()).await?;
        }

        // Add to search index
        {
            let mut index = self.search_index.write().await;
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
        let storage = self.episodic_storage.read().await;
        let mut memory = storage.retrieve(id).await?;

        // Record access
        memory.record_access(true);

        // Update in storage
        drop(storage);
        {
            let mut storage = self.episodic_storage.write().await;
            storage.update(memory.clone()).await?;
        }

        Ok(memory)
    }

    /// Search memories using HDC-based similarity
    pub async fn search(&self, query: &str, k: usize) -> Result<Vec<SearchResult>> {
        self.increment_stat("total_searches");

        let mut index = self.search_index.write().await;
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

        let mut index = self.search_index.write().await;
        let results = index.search_temporal(query, k, since, until);

        Ok(results)
    }

    /// Delete a memory
    pub async fn delete_memory(&self, id: &MemoryId) -> Result<()> {
        // Remove from episodic storage
        {
            let mut storage = self.episodic_storage.write().await;
            storage.delete(id).await?;
        }

        // Remove from search index
        {
            let mut index = self.search_index.write().await;
            index.remove_memory(id)
                .map_err(|e| ManagerError::InvalidOperation(e))?;
        }

        tracing::debug!("Deleted memory: {}", id);
        Ok(())
    }

    /// Run forgetting process - remove memories below retention threshold
    pub async fn forget_weak_memories(&self, threshold: f32) -> Result<u64> {
        let storage = self.episodic_storage.read().await;
        let all_ids = storage.list_ids().await?;
        drop(storage);

        let mut forgotten_count = 0u64;

        for id in all_ids {
            let storage = self.episodic_storage.read().await;
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

    /// Search memories by emotional valence
    pub async fn search_by_emotion(
        &self,
        valence: EmotionalValence,
        k: usize,
    ) -> Result<Vec<MemoryUnit>> {
        self.increment_stat("total_searches");

        let storage = self.episodic_storage.read().await;
        let all_ids = storage.list_ids().await?;
        drop(storage);

        let mut results = Vec::new();

        for id in all_ids {
            let storage = self.episodic_storage.read().await;
            if let Ok(memory) = storage.retrieve(&id).await {
                if memory.emotional_valence() == Some(valence) {
                    results.push(memory);
                }
            }
        }

        // Sort by retention strength (stronger memories first)
        results.sort_by(|a, b| {
            b.retention_strength()
                .partial_cmp(&a.retention_strength())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit to k results
        results.truncate(k);

        Ok(results)
    }

    /// Search memories by emotional similarity to a given PAD vector
    pub async fn search_by_emotional_similarity(
        &self,
        target_emotion: PADVector,
        k: usize,
        max_distance: f32,
    ) -> Result<Vec<(MemoryUnit, f32)>> {
        self.increment_stat("total_searches");

        let storage = self.episodic_storage.read().await;
        let all_ids = storage.list_ids().await?;
        drop(storage);

        let mut results: Vec<(MemoryUnit, f32)> = Vec::new();

        for id in all_ids {
            let storage = self.episodic_storage.read().await;
            if let Ok(memory) = storage.retrieve(&id).await {
                if let Some(ref emotional) = memory.emotional {
                    let distance = target_emotion.distance(&emotional.pad_vector);
                    if distance <= max_distance {
                        results.push((memory, distance));
                    }
                }
            }
        }

        // Sort by distance (closer emotions first)
        results.sort_by(|a, b| {
            a.1.partial_cmp(&b.1)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Limit to k results
        results.truncate(k);

        Ok(results)
    }

    /// Get all active prospective memories (goals/intentions)
    pub async fn get_active_intentions(&self) -> Result<Vec<MemoryUnit>> {
        let storage = self.episodic_storage.read().await;
        let all_ids = storage.list_ids().await?;
        drop(storage);

        let mut intentions = Vec::new();

        for id in all_ids {
            let storage = self.episodic_storage.read().await;
            if let Ok(memory) = storage.retrieve(&id).await {
                if memory.is_prospective() {
                    if let Some(ref intention) = memory.intention {
                        if intention.status == GoalStatus::Active
                            || intention.status == GoalStatus::Pending
                        {
                            intentions.push(memory);
                        }
                    }
                }
            }
        }

        // Sort by priority (higher priority first)
        intentions.sort_by(|a, b| {
            let priority_a = a.intention.as_ref().map(|i| i.priority).unwrap_or(0.0);
            let priority_b = b.intention.as_ref().map(|i| i.priority).unwrap_or(0.0);
            priority_b
                .partial_cmp(&priority_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(intentions)
    }

    /// Get intentions that should trigger now
    pub async fn get_triggerable_intentions(&self) -> Result<Vec<MemoryUnit>> {
        let storage = self.episodic_storage.read().await;
        let all_ids = storage.list_ids().await?;
        drop(storage);

        let mut triggerable = Vec::new();

        for id in all_ids {
            let storage = self.episodic_storage.read().await;
            if let Ok(memory) = storage.retrieve(&id).await {
                if let Some(ref intention) = memory.intention {
                    if intention.should_trigger_now() {
                        triggerable.push(memory);
                    }
                }
            }
        }

        // Sort by priority
        triggerable.sort_by(|a, b| {
            let priority_a = a.intention.as_ref().map(|i| i.priority).unwrap_or(0.0);
            let priority_b = b.intention.as_ref().map(|i| i.priority).unwrap_or(0.0);
            priority_b
                .partial_cmp(&priority_a)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        Ok(triggerable)
    }

    /// Complete a prospective memory (mark goal as completed)
    pub async fn complete_intention(&self, id: &MemoryId) -> Result<()> {
        let storage = self.episodic_storage.read().await;
        let mut memory = storage.retrieve(id).await?;
        drop(storage);

        if let Some(ref mut intention) = memory.intention {
            intention.complete();

            let mut storage = self.episodic_storage.write().await;
            storage.update(memory).await?;

            tracing::debug!("Completed intention: {}", id);
            Ok(())
        } else {
            Err(ManagerError::InvalidOperation(
                "Memory is not a prospective memory".to_string(),
            ))
        }
    }

    /// Cancel a prospective memory
    pub async fn cancel_intention(&self, id: &MemoryId) -> Result<()> {
        let storage = self.episodic_storage.read().await;
        let mut memory = storage.retrieve(id).await?;
        drop(storage);

        if let Some(ref mut intention) = memory.intention {
            intention.cancel();

            let mut storage = self.episodic_storage.write().await;
            storage.update(memory).await?;

            tracing::debug!("Cancelled intention: {}", id);
            Ok(())
        } else {
            Err(ManagerError::InvalidOperation(
                "Memory is not a prospective memory".to_string(),
            ))
        }
    }

    /// Update emotional state of a memory
    pub async fn update_emotion(&self, id: &MemoryId, emotion: PADVector) -> Result<()> {
        let storage = self.episodic_storage.read().await;
        let mut memory = storage.retrieve(id).await?;
        drop(storage);

        memory.update_emotion(emotion);

        let mut storage = self.episodic_storage.write().await;
        storage.update(memory).await?;

        tracing::debug!("Updated emotion for memory: {}", id);
        Ok(())
    }

    /// Get emotional statistics
    pub async fn get_emotional_stats(&self) -> Result<EmotionalStats> {
        let storage = self.episodic_storage.read().await;
        let all_ids = storage.list_ids().await?;
        drop(storage);

        let mut positive_count = 0u64;
        let mut negative_count = 0u64;
        let mut neutral_count = 0u64;
        let mut total_intensity = 0.0f32;
        let mut emotion_count = 0u64;

        for id in all_ids {
            let storage = self.episodic_storage.read().await;
            if let Ok(memory) = storage.retrieve(&id).await {
                if let Some(ref emotional) = memory.emotional {
                    match emotional.valence {
                        Some(EmotionalValence::Positive) => positive_count += 1,
                        Some(EmotionalValence::Negative) => negative_count += 1,
                        Some(EmotionalValence::Neutral) => neutral_count += 1,
                        None => {}
                    }

                    total_intensity += emotional.pad_vector.intensity();
                    emotion_count += 1;
                }
            }
        }

        let average_intensity = if emotion_count > 0 {
            total_intensity / emotion_count as f32
        } else {
            0.0
        };

        Ok(EmotionalStats {
            positive_count,
            negative_count,
            neutral_count,
            average_intensity,
        })
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

    #[tokio::test]
    async fn test_add_memory_with_emotion() {
        use cmd_core::memory::PADVector;

        let episodic = InMemoryEpisodicStorage::new();
        let semantic = InMemorySemanticStorage::new();
        let config = ManagerConfig::default();

        let manager = MemoryManager::new(episodic, semantic, config);

        let source = SourceMetadata {
            source_id: SourceId::new(),
            source_type: SourceType::DirectUserInput,
            confidence: 0.9,
            timestamp: Utc::now(),
        };

        let emotion = PADVector::new(0.7, 0.4, 0.3);
        let memory = MemoryUnit::new_text_with_emotion(
            "User is happy about progress".to_string(),
            source,
            emotion,
        );
        let memory_id = memory.id.clone();

        manager.add_memory(memory).await.unwrap();

        let retrieved = manager.get_memory(&memory_id).await.unwrap();
        assert!(retrieved.emotional.is_some());
        assert_eq!(
            retrieved.emotional_valence(),
            Some(EmotionalValence::Positive)
        );
    }

    #[tokio::test]
    async fn test_search_by_emotion() {
        use cmd_core::memory::PADVector;

        let episodic = InMemoryEpisodicStorage::new();
        let semantic = InMemorySemanticStorage::new();
        let config = ManagerConfig::default();

        let manager = MemoryManager::new(episodic, semantic, config);

        let source1 = SourceMetadata {
            source_id: SourceId::new(),
            source_type: SourceType::DirectUserInput,
            confidence: 0.9,
            timestamp: Utc::now(),
        };

        let source2 = SourceMetadata {
            source_id: SourceId::new(),
            source_type: SourceType::DirectUserInput,
            confidence: 0.9,
            timestamp: Utc::now(),
        };

        // Add positive memory
        let positive_emotion = PADVector::new(0.8, 0.5, 0.4);
        let positive_memory = MemoryUnit::new_text_with_emotion(
            "Great success!".to_string(),
            source1,
            positive_emotion,
        );
        manager.add_memory(positive_memory).await.unwrap();

        // Add negative memory
        let negative_emotion = PADVector::new(-0.6, -0.3, -0.2);
        let negative_memory = MemoryUnit::new_text_with_emotion(
            "Encountered error".to_string(),
            source2,
            negative_emotion,
        );
        manager.add_memory(negative_memory).await.unwrap();

        // Search for positive memories
        let positive_results = manager
            .search_by_emotion(EmotionalValence::Positive, 10)
            .await
            .unwrap();
        assert_eq!(positive_results.len(), 1);
        assert_eq!(
            positive_results[0].text,
            Some("Great success!".to_string())
        );

        // Search for negative memories
        let negative_results = manager
            .search_by_emotion(EmotionalValence::Negative, 10)
            .await
            .unwrap();
        assert_eq!(negative_results.len(), 1);
    }

    #[tokio::test]
    async fn test_search_by_emotional_similarity() {
        use cmd_core::memory::PADVector;

        let episodic = InMemoryEpisodicStorage::new();
        let semantic = InMemorySemanticStorage::new();
        let config = ManagerConfig::default();

        let manager = MemoryManager::new(episodic, semantic, config);

        let source = SourceMetadata {
            source_id: SourceId::new(),
            source_type: SourceType::DirectUserInput,
            confidence: 0.9,
            timestamp: Utc::now(),
        };

        // Add memory with specific emotion
        let emotion = PADVector::new(0.7, 0.5, 0.3);
        let memory = MemoryUnit::new_text_with_emotion(
            "Happy memory".to_string(),
            source,
            emotion,
        );
        manager.add_memory(memory).await.unwrap();

        // Search for similar emotion
        let target_emotion = PADVector::new(0.8, 0.4, 0.2);
        let results = manager
            .search_by_emotional_similarity(target_emotion, 10, 0.5)
            .await
            .unwrap();

        assert!(results.len() > 0);
        assert!(results[0].1 < 0.5); // Distance should be less than max
    }

    #[tokio::test]
    async fn test_update_emotion() {
        use cmd_core::memory::PADVector;

        let episodic = InMemoryEpisodicStorage::new();
        let semantic = InMemorySemanticStorage::new();
        let config = ManagerConfig::default();

        let manager = MemoryManager::new(episodic, semantic, config);

        let memory = create_test_memory("Test memory");
        let memory_id = memory.id.clone();
        manager.add_memory(memory).await.unwrap();

        // Update with emotion
        let emotion = PADVector::new(0.6, 0.3, 0.2);
        manager.update_emotion(&memory_id, emotion).await.unwrap();

        let retrieved = manager.get_memory(&memory_id).await.unwrap();
        assert!(retrieved.emotional.is_some());
        assert_eq!(
            retrieved.emotional_valence(),
            Some(EmotionalValence::Positive)
        );
    }

    #[tokio::test]
    async fn test_prospective_memory() {
        use cmd_core::memory::TriggerCondition;

        let episodic = InMemoryEpisodicStorage::new();
        let semantic = InMemorySemanticStorage::new();
        let config = ManagerConfig::default();

        let manager = MemoryManager::new(episodic, semantic, config);

        let source = SourceMetadata {
            source_id: SourceId::new(),
            source_type: SourceType::DirectUserInput,
            confidence: 1.0,
            timestamp: Utc::now(),
        };

        // Add intention
        let intention = MemoryUnit::new_intention(
            "Complete research".to_string(),
            TriggerCondition::Immediate,
            0.9,
            source,
        );
        let intention_id = intention.id.clone();
        manager.add_memory(intention).await.unwrap();

        // Get active intentions
        let active = manager.get_active_intentions().await.unwrap();
        assert_eq!(active.len(), 1);
        assert_eq!(active[0].id, intention_id);
    }

    #[tokio::test]
    async fn test_complete_intention() {
        use cmd_core::memory::{TriggerCondition, GoalStatus};

        let episodic = InMemoryEpisodicStorage::new();
        let semantic = InMemorySemanticStorage::new();
        let config = ManagerConfig::default();

        let manager = MemoryManager::new(episodic, semantic, config);

        let source = SourceMetadata {
            source_id: SourceId::new(),
            source_type: SourceType::DirectUserInput,
            confidence: 1.0,
            timestamp: Utc::now(),
        };

        let intention = MemoryUnit::new_intention(
            "Task to complete".to_string(),
            TriggerCondition::Immediate,
            0.8,
            source,
        );
        let intention_id = intention.id.clone();
        manager.add_memory(intention).await.unwrap();

        // Complete the intention
        manager.complete_intention(&intention_id).await.unwrap();

        // Verify it's completed
        let retrieved = manager.get_memory(&intention_id).await.unwrap();
        assert_eq!(
            retrieved.intention.as_ref().unwrap().status,
            GoalStatus::Completed
        );
        assert!(retrieved.intention.as_ref().unwrap().completed_at.is_some());

        // Should not appear in active intentions
        let active = manager.get_active_intentions().await.unwrap();
        assert_eq!(active.len(), 0);
    }

    #[tokio::test]
    async fn test_cancel_intention() {
        use cmd_core::memory::{TriggerCondition, GoalStatus};

        let episodic = InMemoryEpisodicStorage::new();
        let semantic = InMemorySemanticStorage::new();
        let config = ManagerConfig::default();

        let manager = MemoryManager::new(episodic, semantic, config);

        let source = SourceMetadata {
            source_id: SourceId::new(),
            source_type: SourceType::DirectUserInput,
            confidence: 1.0,
            timestamp: Utc::now(),
        };

        let intention = MemoryUnit::new_intention(
            "Task to cancel".to_string(),
            TriggerCondition::Immediate,
            0.7,
            source,
        );
        let intention_id = intention.id.clone();
        manager.add_memory(intention).await.unwrap();

        // Cancel the intention
        manager.cancel_intention(&intention_id).await.unwrap();

        // Verify it's cancelled
        let retrieved = manager.get_memory(&intention_id).await.unwrap();
        assert_eq!(
            retrieved.intention.as_ref().unwrap().status,
            GoalStatus::Cancelled
        );
    }

    #[tokio::test]
    async fn test_triggerable_intentions() {
        use cmd_core::memory::TriggerCondition;

        let episodic = InMemoryEpisodicStorage::new();
        let semantic = InMemorySemanticStorage::new();
        let config = ManagerConfig::default();

        let manager = MemoryManager::new(episodic, semantic, config);

        let source1 = SourceMetadata {
            source_id: SourceId::new(),
            source_type: SourceType::DirectUserInput,
            confidence: 1.0,
            timestamp: Utc::now(),
        };

        let source2 = SourceMetadata {
            source_id: SourceId::new(),
            source_type: SourceType::DirectUserInput,
            confidence: 1.0,
            timestamp: Utc::now(),
        };

        // Add immediate intention
        let immediate = MemoryUnit::new_intention(
            "Immediate task".to_string(),
            TriggerCondition::Immediate,
            0.9,
            source1,
        );
        manager.add_memory(immediate).await.unwrap();

        // Add future intention
        let future = Utc::now() + chrono::Duration::hours(1);
        let future_intention = MemoryUnit::new_intention(
            "Future task".to_string(),
            TriggerCondition::TimeBasedAt(future),
            0.8,
            source2,
        );
        manager.add_memory(future_intention).await.unwrap();

        // Only immediate should be triggerable
        let triggerable = manager.get_triggerable_intentions().await.unwrap();
        assert_eq!(triggerable.len(), 1);
        assert_eq!(
            triggerable[0].text,
            Some("Immediate task".to_string())
        );
    }

    #[tokio::test]
    async fn test_emotional_stats() {
        use cmd_core::memory::PADVector;

        let episodic = InMemoryEpisodicStorage::new();
        let semantic = InMemorySemanticStorage::new();
        let config = ManagerConfig::default();

        let manager = MemoryManager::new(episodic, semantic, config);

        let source1 = SourceMetadata {
            source_id: SourceId::new(),
            source_type: SourceType::DirectUserInput,
            confidence: 0.9,
            timestamp: Utc::now(),
        };

        let source2 = SourceMetadata {
            source_id: SourceId::new(),
            source_type: SourceType::DirectUserInput,
            confidence: 0.9,
            timestamp: Utc::now(),
        };

        // Add positive memory
        let positive = MemoryUnit::new_text_with_emotion(
            "Happy".to_string(),
            source1,
            PADVector::new(0.7, 0.3, 0.2),
        );
        manager.add_memory(positive).await.unwrap();

        // Add negative memory
        let negative = MemoryUnit::new_text_with_emotion(
            "Sad".to_string(),
            source2,
            PADVector::new(-0.6, -0.2, -0.1),
        );
        manager.add_memory(negative).await.unwrap();

        let stats = manager.get_emotional_stats().await.unwrap();
        assert_eq!(stats.positive_count, 1);
        assert_eq!(stats.negative_count, 1);
        assert!(stats.average_intensity > 0.0);
    }
}
