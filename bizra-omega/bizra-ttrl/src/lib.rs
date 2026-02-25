//! # bizra-ttrl — Four Paper Upgrades for the BIZRA Omni-Kernel
//!
//! This crate wires four research-paper breakthroughs into the existing
//! `bizra-agent` orchestration stack, upgrading each of the 8 Omni-Kernel
//! lines without replacing existing infrastructure.
//!
//! ## Module map
//!
//! | Module              | Paper                        | Omni-Kernel Line |
//! |---------------------|------------------------------|------------------|
//! | `decision_pivot`    | Chain of Reasoning           | Line 1 (HHMM)    |
//! | `engram`            | DeepSeek MoE Engram          | Line 2 (cache)   |
//! | `sso`               | Spectral Sphere Optimizer    | Line 5 (İhsān)   |
//! | `ttrl_engine`       | Test-Time Reinforcement Lrn  | Line 7 (ledger)  |
//! | `metabolic_ledger`  | TTRL economics + decay       | Line 7 (ledger)  |
//!
//! ## Usage in `OmniKernel`
//! ```rust,ignore
//! use bizra_ttrl::{
//!     decision_pivot::ReasoningChain,
//!     engram::EngramCache,
//!     sso::SpectralSphereConstraint,
//!     ttrl_engine::TtrlEngine,
//!     metabolic_ledger::MetabolicLedger,
//! };
//! ```
//!
//! Standing on Giants:
//! - Shannon (1948): Information theory
//! - Al-Ghazali (1095): Iḥsān as incremental excellence
//! - TTRL paper (2025): Self-improving on-device RL
//! - SSO paper (2025): Spectral-sphere stability
//! - DeepSeek MoE (2025): Engram as O(1) factual memory

pub mod decision_pivot;
pub mod engram;
pub mod metabolic_ledger;
pub mod sso;
pub mod ttrl_engine;

// Re-export the most-used types for ergonomic `use bizra_ttrl::*`.
pub use decision_pivot::{DecisionPivot, HhmmLevel, ReasoningChain, PIVOT_IHSAN_DEFAULT};
pub use engram::{EngramCache, EngramResult};
pub use metabolic_ledger::{MetabolicLedger, PoiYield};
pub use sso::{SpectralNorm, SpectralSphereConstraint, SSO_DEFAULT_EPSILON};
pub use ttrl_engine::{GrpoUpdate, TtrlEngine};
