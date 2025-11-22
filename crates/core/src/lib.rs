//! Core data structures and traits for Cognitive Memory Database

pub mod memory;
pub mod retention;
pub mod types;
pub mod crdt;

pub use memory::{MemoryUnit, Modality};
pub use retention::RetentionModel;
pub use types::{MemoryId, SourceId, NodeId, EdgeId};
pub use crdt::{FactVersion, VectorClock};

/// Core error types
#[derive(thiserror::Error, Debug)]
pub enum CmdError {
    #[error("Memory not found: {0}")]
    MemoryNotFound(MemoryId),

    #[error("Conflict detected: {0}")]
    ConflictDetected(String),

    #[error("Invalid operation: {0}")]
    InvalidOperation(String),

    #[error("Serialization error: {0}")]
    SerializationError(#[from] bincode::Error),

    #[error("Storage error: {0}")]
    StorageError(String),

    #[error("Unknown error: {0}")]
    Unknown(#[from] anyhow::Error),
}

pub type Result<T> = std::result::Result<T, CmdError>;