//! API layer for Cognitive Memory Database

pub mod rest;
#[cfg(feature = "python")]
pub mod python;

// Temporary placeholder
pub fn init() {
    println!("API module initialized");
}