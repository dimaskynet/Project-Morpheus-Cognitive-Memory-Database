//! Conflict resolution module for Cognitive Memory Database

pub mod deterministic;
pub mod strategies;
pub mod types;

pub use deterministic::{DeterministicResolver, DomainRules};
pub use strategies::SourceTrustManager;
pub use types::{
    Conflict, ConflictContext, ConflictType, ResolutionPrecedent,
    ResolutionResult, ResolutionStats, ResolutionStrategy,
};