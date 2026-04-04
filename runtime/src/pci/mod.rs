// src/pci/mod.rs - Proof-Carrying Inference Protocol
//
// Cryptographic message protocol for PAT↔SAT communication.
//
// Version: 1.0.0
// Status: PRODUCTION
// Alignment: BIZRA_SOT.md Section 3.1 (Ihsān IM ≥ 0.95)

pub mod envelope;
pub mod gates;
pub mod reject_codes;
pub mod types;

pub use envelope::*;
pub use gates::*;
pub use reject_codes::*;
pub use types::*;
