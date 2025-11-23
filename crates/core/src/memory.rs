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

    /// Emotional context (PAD model)
    pub emotional: Option<EmotionalMetadata>,

    /// Prospective memory (goals/intentions)
    pub intention: Option<ProspectiveMemory>,

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

/// PAD emotional model (Pleasure, Arousal, Dominance)
/// 3D emotional space for memory tagging
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PADVector {
    /// Pleasure dimension: -1.0 (displeasure) to 1.0 (pleasure)
    pub pleasure: f32,

    /// Arousal dimension: -1.0 (calm) to 1.0 (excited)
    pub arousal: f32,

    /// Dominance dimension: -1.0 (submissive) to 1.0 (dominant)
    pub dominance: f32,
}

impl PADVector {
    /// Create a new PAD vector
    pub fn new(pleasure: f32, arousal: f32, dominance: f32) -> Self {
        Self {
            pleasure: pleasure.clamp(-1.0, 1.0),
            arousal: arousal.clamp(-1.0, 1.0),
            dominance: dominance.clamp(-1.0, 1.0),
        }
    }

    /// Neutral emotional state
    pub fn neutral() -> Self {
        Self {
            pleasure: 0.0,
            arousal: 0.0,
            dominance: 0.0,
        }
    }

    /// Calculate Euclidean distance to another PAD vector
    pub fn distance(&self, other: &PADVector) -> f32 {
        let dp = self.pleasure - other.pleasure;
        let da = self.arousal - other.arousal;
        let dd = self.dominance - other.dominance;
        (dp * dp + da * da + dd * dd).sqrt()
    }

    /// Get emotional intensity (distance from neutral)
    pub fn intensity(&self) -> f32 {
        self.distance(&Self::neutral())
    }
}

/// Emotional metadata for memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmotionalMetadata {
    /// Current emotional state (PAD vector)
    pub pad_vector: PADVector,

    /// History of emotional states over time
    pub mood_history: Vec<(DateTime<Utc>, PADVector)>,

    /// Emotional valence tag (optional simplified label)
    pub valence: Option<EmotionalValence>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmotionalValence {
    Positive,
    Negative,
    Neutral,
}

impl EmotionalMetadata {
    /// Create neutral emotional metadata
    pub fn neutral() -> Self {
        Self {
            pad_vector: PADVector::neutral(),
            mood_history: Vec::new(),
            valence: Some(EmotionalValence::Neutral),
        }
    }

    /// Create from PAD vector
    pub fn from_pad(pad: PADVector) -> Self {
        let valence = if pad.pleasure > 0.3 {
            Some(EmotionalValence::Positive)
        } else if pad.pleasure < -0.3 {
            Some(EmotionalValence::Negative)
        } else {
            Some(EmotionalValence::Neutral)
        };

        Self {
            pad_vector: pad,
            mood_history: vec![(Utc::now(), pad)],
            valence,
        }
    }

    /// Update emotional state
    pub fn update(&mut self, new_pad: PADVector) {
        self.mood_history.push((Utc::now(), new_pad));
        self.pad_vector = new_pad;

        // Update valence
        self.valence = if new_pad.pleasure > 0.3 {
            Some(EmotionalValence::Positive)
        } else if new_pad.pleasure < -0.3 {
            Some(EmotionalValence::Negative)
        } else {
            Some(EmotionalValence::Neutral)
        };
    }
}

/// Prospective memory - goals and intentions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProspectiveMemory {
    /// Goal or intention description
    pub goal: String,

    /// Trigger condition for this memory
    pub trigger: TriggerCondition,

    /// Optional deadline
    pub deadline: Option<DateTime<Utc>>,

    /// Priority (0.0 to 1.0)
    pub priority: f32,

    /// Status of the goal
    pub status: GoalStatus,

    /// When this intention was created
    pub created_at: DateTime<Utc>,

    /// When this intention was fulfilled/cancelled
    pub completed_at: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TriggerCondition {
    /// Trigger at specific time
    TimeBasedAt(DateTime<Utc>),

    /// Trigger after duration
    TimeBasedAfter(std::time::Duration),

    /// Trigger when specific event occurs
    EventBased(String),

    /// Trigger when context matches
    ContextBased(Vec<String>),

    /// Trigger immediately
    Immediate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GoalStatus {
    Pending,
    Active,
    Completed,
    Cancelled,
    Failed,
}

impl ProspectiveMemory {
    /// Create a new prospective memory
    pub fn new(goal: String, trigger: TriggerCondition, priority: f32) -> Self {
        Self {
            goal,
            trigger,
            deadline: None,
            priority: priority.clamp(0.0, 1.0),
            status: GoalStatus::Pending,
            created_at: Utc::now(),
            completed_at: None,
        }
    }

    /// Mark as completed
    pub fn complete(&mut self) {
        self.status = GoalStatus::Completed;
        self.completed_at = Some(Utc::now());
    }

    /// Mark as cancelled
    pub fn cancel(&mut self) {
        self.status = GoalStatus::Cancelled;
        self.completed_at = Some(Utc::now());
    }

    /// Check if should trigger based on current time
    pub fn should_trigger_now(&self) -> bool {
        if self.status != GoalStatus::Pending && self.status != GoalStatus::Active {
            return false;
        }

        match &self.trigger {
            TriggerCondition::TimeBasedAt(time) => Utc::now() >= *time,
            TriggerCondition::Immediate => true,
            _ => false, // Other conditions need external evaluation
        }
    }
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
            emotional: None,
            intention: None,
            metadata: HashMap::new(),
        }
    }

    /// Create a new memory with emotional context
    pub fn new_text_with_emotion(
        content: String,
        source: SourceMetadata,
        emotion: PADVector,
    ) -> Self {
        let mut memory = Self::new_text(content, source);
        memory.emotional = Some(EmotionalMetadata::from_pad(emotion));
        memory
    }

    /// Create a prospective memory (goal/intention)
    pub fn new_intention(
        goal: String,
        trigger: TriggerCondition,
        priority: f32,
        source: SourceMetadata,
    ) -> Self {
        let now = Utc::now();
        let intention = ProspectiveMemory::new(goal.clone(), trigger, priority);

        Self {
            id: MemoryId::new(),
            modality: Modality::Structured,
            content: goal.as_bytes().to_vec(),
            text: Some(goal),
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
            retention: RetentionModel::new(1.0), // High retention for goals
            graph_links: Vec::new(),
            source,
            emotional: None,
            intention: Some(intention),
            metadata: HashMap::new(),
        }
    }

    /// Update emotional state of this memory
    pub fn update_emotion(&mut self, emotion: PADVector) {
        if let Some(ref mut emotional) = self.emotional {
            emotional.update(emotion);
        } else {
            self.emotional = Some(EmotionalMetadata::from_pad(emotion));
        }
    }

    /// Check if this is a prospective memory
    pub fn is_prospective(&self) -> bool {
        self.intention.is_some()
    }

    /// Get emotional valence if available
    pub fn emotional_valence(&self) -> Option<EmotionalValence> {
        self.emotional.as_ref().and_then(|e| e.valence)
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

        memory.record_access(true);

        assert_eq!(memory.temporal.access_count, 1);
        // After successful recall, stability should increase
        assert!(memory.retention.stability > 1.0);
    }

    #[test]
    fn test_pad_vector_creation() {
        let pad = PADVector::new(0.8, 0.5, -0.3);
        assert_eq!(pad.pleasure, 0.8);
        assert_eq!(pad.arousal, 0.5);
        assert_eq!(pad.dominance, -0.3);

        // Test clamping
        let pad_clamped = PADVector::new(1.5, -2.0, 0.0);
        assert_eq!(pad_clamped.pleasure, 1.0);
        assert_eq!(pad_clamped.arousal, -1.0);
        assert_eq!(pad_clamped.dominance, 0.0);
    }

    #[test]
    fn test_pad_vector_distance() {
        let happy = PADVector::new(0.8, 0.3, 0.5);
        let sad = PADVector::new(-0.6, -0.2, -0.3);
        let neutral = PADVector::neutral();

        // Distance between happy and sad should be large
        let distance = happy.distance(&sad);
        assert!(distance > 1.0);

        // Distance from self should be zero
        assert_eq!(happy.distance(&happy), 0.0);

        // Neutral should be equidistant to happy and sad (approximately)
        let d_happy = neutral.distance(&happy);
        let d_sad = neutral.distance(&sad);
        assert!((d_happy - d_sad).abs() < 0.5);
    }

    #[test]
    fn test_pad_vector_intensity() {
        let neutral = PADVector::neutral();
        assert_eq!(neutral.intensity(), 0.0);

        let intense = PADVector::new(1.0, 1.0, 1.0);
        let intensity = intense.intensity();
        assert!(intensity > 1.5); // sqrt(3) ≈ 1.73

        let mild = PADVector::new(0.3, 0.2, 0.1);
        assert!(mild.intensity() < intense.intensity());
    }

    #[test]
    fn test_emotional_metadata_creation() {
        let pad = PADVector::new(0.5, 0.3, 0.2);
        let emotional = EmotionalMetadata::from_pad(pad);

        assert_eq!(emotional.pad_vector.pleasure, 0.5);
        assert_eq!(emotional.valence, Some(EmotionalValence::Positive));
        assert_eq!(emotional.mood_history.len(), 1);
    }

    #[test]
    fn test_emotional_metadata_valence_classification() {
        let positive = EmotionalMetadata::from_pad(PADVector::new(0.6, 0.0, 0.0));
        assert_eq!(positive.valence, Some(EmotionalValence::Positive));

        let negative = EmotionalMetadata::from_pad(PADVector::new(-0.6, 0.0, 0.0));
        assert_eq!(negative.valence, Some(EmotionalValence::Negative));

        let neutral = EmotionalMetadata::from_pad(PADVector::new(0.1, 0.0, 0.0));
        assert_eq!(neutral.valence, Some(EmotionalValence::Neutral));
    }

    #[test]
    fn test_emotional_metadata_update() {
        let mut emotional = EmotionalMetadata::neutral();
        assert_eq!(emotional.mood_history.len(), 0);

        let new_pad = PADVector::new(0.7, 0.5, 0.3);
        emotional.update(new_pad);

        assert_eq!(emotional.pad_vector.pleasure, 0.7);
        assert_eq!(emotional.valence, Some(EmotionalValence::Positive));
        assert_eq!(emotional.mood_history.len(), 1);

        // Update again
        let another_pad = PADVector::new(-0.5, 0.2, -0.1);
        emotional.update(another_pad);
        assert_eq!(emotional.mood_history.len(), 2);
        assert_eq!(emotional.valence, Some(EmotionalValence::Negative));
    }

    #[test]
    fn test_prospective_memory_creation() {
        let goal = "Complete research paper".to_string();
        let trigger = TriggerCondition::Immediate;
        let priority = 0.8;

        let prospective = ProspectiveMemory::new(goal.clone(), trigger, priority);

        assert_eq!(prospective.goal, goal);
        assert_eq!(prospective.priority, 0.8);
        assert_eq!(prospective.status, GoalStatus::Pending);
        assert!(prospective.completed_at.is_none());
    }

    #[test]
    fn test_prospective_memory_priority_clamping() {
        let prospective = ProspectiveMemory::new(
            "Test".to_string(),
            TriggerCondition::Immediate,
            1.5, // Should be clamped to 1.0
        );
        assert_eq!(prospective.priority, 1.0);

        let prospective2 = ProspectiveMemory::new(
            "Test2".to_string(),
            TriggerCondition::Immediate,
            -0.5, // Should be clamped to 0.0
        );
        assert_eq!(prospective2.priority, 0.0);
    }

    #[test]
    fn test_prospective_memory_complete() {
        let mut prospective = ProspectiveMemory::new(
            "Test goal".to_string(),
            TriggerCondition::Immediate,
            0.7,
        );

        assert_eq!(prospective.status, GoalStatus::Pending);
        assert!(prospective.completed_at.is_none());

        prospective.complete();

        assert_eq!(prospective.status, GoalStatus::Completed);
        assert!(prospective.completed_at.is_some());
    }

    #[test]
    fn test_prospective_memory_cancel() {
        let mut prospective = ProspectiveMemory::new(
            "Test goal".to_string(),
            TriggerCondition::Immediate,
            0.5,
        );

        prospective.cancel();

        assert_eq!(prospective.status, GoalStatus::Cancelled);
        assert!(prospective.completed_at.is_some());
    }

    #[test]
    fn test_prospective_memory_immediate_trigger() {
        let prospective = ProspectiveMemory::new(
            "Immediate task".to_string(),
            TriggerCondition::Immediate,
            0.9,
        );

        assert!(prospective.should_trigger_now());
    }

    #[test]
    fn test_prospective_memory_time_based_trigger() {
        // Future trigger - should not trigger now
        let future = Utc::now() + chrono::Duration::hours(1);
        let prospective_future = ProspectiveMemory::new(
            "Future task".to_string(),
            TriggerCondition::TimeBasedAt(future),
            0.8,
        );
        assert!(!prospective_future.should_trigger_now());

        // Past trigger - should trigger now
        let past = Utc::now() - chrono::Duration::hours(1);
        let prospective_past = ProspectiveMemory::new(
            "Past task".to_string(),
            TriggerCondition::TimeBasedAt(past),
            0.8,
        );
        assert!(prospective_past.should_trigger_now());
    }

    #[test]
    fn test_memory_with_emotion() {
        let source = SourceMetadata {
            source_id: SourceId::new(),
            source_type: SourceType::DirectUserInput,
            confidence: 0.9,
            timestamp: Utc::now(),
        };

        let emotion = PADVector::new(0.7, 0.4, 0.3);
        let memory = MemoryUnit::new_text_with_emotion(
            "User expressed excitement about the project".to_string(),
            source,
            emotion,
        );

        assert!(memory.emotional.is_some());
        assert_eq!(memory.emotional_valence(), Some(EmotionalValence::Positive));
        assert!(!memory.is_prospective());
    }

    #[test]
    fn test_memory_update_emotion() {
        let source = SourceMetadata {
            source_id: SourceId::new(),
            source_type: SourceType::DirectUserInput,
            confidence: 0.9,
            timestamp: Utc::now(),
        };

        let mut memory = MemoryUnit::new_text("Test memory".to_string(), source);
        assert!(memory.emotional.is_none());

        // Add emotional context
        let emotion = PADVector::new(0.5, 0.3, 0.2);
        memory.update_emotion(emotion);

        assert!(memory.emotional.is_some());
        assert_eq!(memory.emotional_valence(), Some(EmotionalValence::Positive));

        // Update emotional context
        let new_emotion = PADVector::new(-0.6, -0.2, -0.3);
        memory.update_emotion(new_emotion);

        assert_eq!(memory.emotional_valence(), Some(EmotionalValence::Negative));
        assert_eq!(memory.emotional.as_ref().unwrap().mood_history.len(), 2);
    }

    #[test]
    fn test_intention_memory() {
        let source = SourceMetadata {
            source_id: SourceId::new(),
            source_type: SourceType::DirectUserInput,
            confidence: 1.0,
            timestamp: Utc::now(),
        };

        let memory = MemoryUnit::new_intention(
            "Implement cognitive architecture".to_string(),
            TriggerCondition::ContextBased(vec!["coding".to_string(), "development".to_string()]),
            0.9,
            source,
        );

        assert!(memory.is_prospective());
        assert_eq!(memory.modality, Modality::Structured);
        assert_eq!(memory.intention.as_ref().unwrap().priority, 0.9);
        assert_eq!(memory.intention.as_ref().unwrap().status, GoalStatus::Pending);
        // High retention for goals
        assert!(memory.retention_strength() > 0.9);
    }
}