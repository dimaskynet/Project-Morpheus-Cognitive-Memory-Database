//! Types for conflict detection and resolution

use cmd_core::crdt::FactVersion;
use cmd_core::types::SourceId;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Types of conflicts that can occur in the memory system
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum ConflictType {
    /// Newer fact supersedes older one
    TemporalSupersession,

    /// Updating attributes of existing entity
    AttributeUpdate,

    /// Direct contradiction (A and ¬A)
    ContradictoryFacts,

    /// Partial overlap in information
    PartialOverlap,

    /// Causal chain conflict (A→B, B→C, A→¬C)
    CausalChain,

    /// Different sources disagree
    SourceDisagreement,

    /// Same fact from multiple sources
    DuplicateInformation,

    /// Ambiguous reference resolution
    ReferenceAmbiguity,
}

/// Strategy for resolving conflicts
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ResolutionStrategy {
    // Level 1: Fully deterministic (90% of cases)

    /// Keep the newest fact based on timestamp
    KeepNewest,

    /// Keep the oldest fact (for stable facts)
    KeepOldest,

    /// Merge using CRDT operations
    MergeViaCRDT,

    /// Trust source with higher confidence
    TrustHigherSource,

    /// Apply logical rules (Datalog)
    ApplyLogicRules,

    /// Keep both versions with different confidence
    VersionBranching,

    // Level 2: Semi-deterministic (8%)

    /// Use consensus from multiple sources
    ConsensusVoting,

    /// Apply domain-specific rules
    DomainSpecificRules(String),

    // Level 3: External resolution (2%)

    /// Defer to user for manual resolution
    DeferToUser,

    /// Use local LLM for resolution (fallback)
    LocalLLM,
}

/// A detected conflict between facts
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Conflict {
    /// Unique identifier for this conflict
    pub id: String,

    /// Type of conflict detected
    pub conflict_type: ConflictType,

    /// The existing fact
    pub existing_fact: FactVersion,

    /// The new/conflicting fact
    pub new_fact: FactVersion,

    /// Confidence in conflict detection (0.0 to 1.0)
    pub confidence: f32,

    /// Additional context for resolution
    pub context: ConflictContext,
}

/// Additional context for conflict resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConflictContext {
    /// Related facts that might influence resolution
    pub related_facts: Vec<FactVersion>,

    /// Source trust scores
    pub source_trust: HashMap<SourceId, f32>,

    /// Domain this conflict relates to
    pub domain: Option<String>,

    /// Previous resolutions of similar conflicts
    pub precedents: Vec<ResolutionPrecedent>,
}

/// Record of a previous conflict resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionPrecedent {
    /// Similar conflict type
    pub conflict_type: ConflictType,

    /// Strategy that was used
    pub strategy_used: ResolutionStrategy,

    /// Whether the resolution was successful
    pub success: bool,

    /// Confidence in this precedent's relevance
    pub relevance: f32,
}

/// Result of conflict resolution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResolutionResult {
    /// The strategy that was applied
    pub strategy_used: ResolutionStrategy,

    /// The resolved fact(s)
    pub resolved_facts: Vec<FactVersion>,

    /// Facts that were discarded
    pub discarded_facts: Vec<FactVersion>,

    /// Confidence in the resolution
    pub confidence: f32,

    /// Explanation of the resolution
    pub explanation: String,

    /// Whether manual review is recommended
    pub needs_review: bool,
}

/// Statistics about conflict resolution
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct ResolutionStats {
    /// Total conflicts processed
    pub total_conflicts: u64,

    /// Conflicts resolved deterministically
    pub deterministic_resolutions: u64,

    /// Conflicts requiring external help
    pub external_resolutions: u64,

    /// Average resolution time in microseconds
    pub avg_resolution_time_us: u64,

    /// Success rate (0.0 to 1.0)
    pub success_rate: f32,

    /// Breakdown by conflict type
    pub by_type: HashMap<String, u64>,

    /// Breakdown by strategy used
    pub by_strategy: HashMap<String, u64>,
}

impl ConflictType {
    /// Get the default resolution strategy for this conflict type
    pub fn default_strategy(&self) -> ResolutionStrategy {
        match self {
            Self::TemporalSupersession => ResolutionStrategy::KeepNewest,
            Self::AttributeUpdate => ResolutionStrategy::MergeViaCRDT,
            Self::ContradictoryFacts => ResolutionStrategy::TrustHigherSource,
            Self::PartialOverlap => ResolutionStrategy::MergeViaCRDT,
            Self::CausalChain => ResolutionStrategy::ApplyLogicRules,
            Self::SourceDisagreement => ResolutionStrategy::ConsensusVoting,
            Self::DuplicateInformation => ResolutionStrategy::KeepNewest,
            Self::ReferenceAmbiguity => ResolutionStrategy::DeferToUser,
        }
    }

    /// Check if this conflict type can be resolved deterministically
    pub fn is_deterministic(&self) -> bool {
        matches!(
            self,
            Self::TemporalSupersession
            | Self::AttributeUpdate
            | Self::DuplicateInformation
        )
    }
}