//! Proof-Carrying Inference Protocol

/// Cryptographically signed message containers.
pub mod envelope;
/// Tiered verification gate chain (Schema → Ihsan → SNR).
pub mod gates;
/// Kleisli gate chain — category-theoretic formalization (HP-02 dual-stack monad).
pub mod kleisli;
/// Protocol-level reject/success codes.
pub mod reject_codes;
/// Gate verdict — cross-layer contract #2 (mission evaluation result).
pub mod verdict;

pub use envelope::PCIEnvelope;
pub use gates::{Gate, GateChain, GateContext, GateResult, GateTier};
pub use reject_codes::RejectCode;
pub use verdict::{GateVerdict, ProofStatus, VerdictStatus};
