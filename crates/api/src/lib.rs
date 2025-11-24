//! API layer for Cognitive Memory Database

pub mod rest;
#[cfg(feature = "python")]
pub mod python;

// Re-export main router for convenience
pub use rest::create_router;