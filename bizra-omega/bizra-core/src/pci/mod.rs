//! Proof-Carrying Inference Protocol

/// Cryptographically signed message containers.
pub mod envelope;
/// Tiered verification gate chain (Schema → Ihsan → SNR).
pub mod gates;
/// Protocol-level reject/success codes.
pub mod reject_codes;

pub use envelope::PCIEnvelope;
pub use gates::{Gate, GateChain, GateContext, GateResult, GateTier};
pub use reject_codes::RejectCode;
