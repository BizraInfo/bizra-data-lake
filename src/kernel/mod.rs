//! BIZRA Sovereign Kernel
//!
//! The Formal Verification Layer (L0-L2) that mathematically proves
//! all agent actions comply with the BIZRA Constitution before execution.
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    SOVEREIGN KERNEL                             │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  L0: Constitution     │  The Covenant (3 Invariants)            │
//! │  L1: Formal Verifier  │  Z3 SMT Solver                          │
//! │  L2: Execution Gate   │  CPU Execution Control                  │
//! └─────────────────────────────────────────────────────────────────┘
//! ```
//!
//! # The Three Invariants (The Covenant)
//!
//! 1. **Anti-Debt (Riba == 0)**: No interest-based transactions
//! 2. **Ihsan Floor (>= 0.99)**: Excellence threshold for all actions
//! 3. **Anti-Assumption (Evidence > 0)**: No execution without evidence
//!
//! # Features
//!
//! This module requires the `z3-solver` feature to enable formal verification.
//! Without this feature, a fallback implementation is used that performs
//! runtime checks without Z3 SMT proving.
//!
//! ```bash
//! # Build with Z3 support (requires z3 library installed)
//! cargo build --features z3-solver
//!
//! # Build without Z3 (uses fallback verification)
//! cargo build
//! ```
//!
//! # Usage
//!
//! ```rust,ignore
//! use crate::kernel::{SovereignKernel, AgentAction, create_verification_context};
//!
//! let ctx = create_verification_context();
//! let kernel = SovereignKernel::new(&ctx);
//!
//! let action = AgentAction { /* ... */ };
//! let result = kernel.verify_intent(&action)?;
//!
//! if result.verified {
//!     // Execute the action - verified safe
//! } else {
//!     // Block execution - constitutional violation detected
//! }
//! ```

pub mod contract;

#[cfg(feature = "z3-solver")]
mod sovereign_gate;

#[cfg(not(feature = "z3-solver"))]
mod sovereign_gate_fallback;

#[cfg(feature = "z3-solver")]
pub use sovereign_gate::{
    create_z3_context, verify_action, ActionContext, ActionMetadata, AgentAction, EvidenceAtom,
    SovereignKernel, VerificationResult,
};

// Alias for compatibility
#[cfg(feature = "z3-solver")]
pub use sovereign_gate::create_z3_context as create_verification_context;

#[cfg(not(feature = "z3-solver"))]
pub use sovereign_gate_fallback::{
    create_verification_context, verify_action, ActionContext, ActionMetadata, AgentAction,
    EvidenceAtom, SovereignKernel, VerificationContext, VerificationResult,
};
