//! Hyperdimensional Computing (HDC) for cognitive memory
//!
//! HDC uses high-dimensional binary vectors (typically 10,000 bits) to represent
//! and manipulate information in a way that mimics brain computation.

#![cfg_attr(feature = "nightly", feature(portable_simd))]

pub mod hypervector;
pub mod operations;
pub mod encoder;
pub mod similarity;
pub mod simd;

pub use hypervector::HyperVector;
pub use operations::{bind, bundle, permute};
pub use similarity::cosine_similarity;

/// Default dimension for hypervectors (10,000 bits)
pub const DEFAULT_DIMENSION: usize = 10_000;

/// Initialize HDC module
pub fn init() -> Result<(), String> {
    // Check if we have SIMD support
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            println!("HDC: AVX2 support detected - using SIMD optimizations");
        } else if is_x86_feature_detected!("sse4.2") {
            println!("HDC: SSE4.2 support detected - using partial SIMD");
        } else {
            println!("HDC: No SIMD support - using scalar operations");
        }
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        println!("HDC: Non-x86_64 architecture - using scalar operations");
    }

    Ok(())
}