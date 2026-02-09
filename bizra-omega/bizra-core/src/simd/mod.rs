//! SIMD Acceleration Layer — AVX-512 optimizations for critical paths
//!
//! Giants: Shannon (information limits), Intel (AVX-512), Gerganov (SIMD LLM)

/// Batch Ed25519 signature verification.
pub mod batch;
/// Branchless SIMD gate validation.
pub mod gates;
/// Parallel BLAKE3 hashing.
pub mod hash;

pub use batch::verify_signatures_batch;
pub use gates::validate_gates_batch;
pub use hash::blake3_parallel;
