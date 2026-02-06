//! BIZRA Sovereign Orchestrator — The Unified Intelligence Layer
//!
//! This module embodies the peak integration of all BIZRA components,
//! implementing Graph-of-Thoughts reasoning and SNR-maximizing autonomy.
//!
//! # Standing on the Shoulders of Giants
//!
//! - **Shannon**: Information theory, entropy bounds (RE = ΔE)
//! - **Lamport**: Distributed consensus, Byzantine fault tolerance
//! - **Vaswani**: Attention mechanisms, transformer architecture
//! - **Besta**: Graph-of-Thoughts reasoning patterns
//! - **Torvalds**: Unix philosophy, composable systems
//! - **Fielding**: REST architectural constraints
//! - **Gerganov**: SIMD optimization, llama.cpp efficiency
//!
//! # Architecture
//!
//! ```text
//! ┌─────────────────────────────────────────────────────────────────┐
//! │                    SovereignOrchestrator                        │
//! ├─────────────────────────────────────────────────────────────────┤
//! │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
//! │  │  Identity   │  │  Protocol   │  │  Inference  │             │
//! │  │   Engine    │  │   Engine    │  │   Engine    │             │
//! │  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
//! │         │                │                │                     │
//! │         └────────────────┼────────────────┘                     │
//! │                          ▼                                      │
//! │              ┌───────────────────────┐                          │
//! │              │   Graph-of-Thoughts   │                          │
//! │              │      Reasoner         │                          │
//! │              └───────────────────────┘                          │
//! │                          │                                      │
//! │                          ▼                                      │
//! │              ┌───────────────────────┐                          │
//! │              │   SNR Maximizer       │                          │
//! │              │   (Autonomous)        │                          │
//! │              └───────────────────────┘                          │
//! └─────────────────────────────────────────────────────────────────┘
//! ```

pub mod error;
pub mod giants;
pub mod graph_of_thoughts;
pub mod omega;
pub mod orchestrator;
pub mod snr_engine;

pub use error::{ErrorContext, SovereignError, SovereignResult};
pub use giants::{Contribution, Giant, GiantRegistry};
pub use graph_of_thoughts::{
    AggregateResult, GraphStats, ReasoningPath, ThoughtGraph, ThoughtNode, ThoughtType,
};
pub use omega::{CircuitState, OmegaConfig, OmegaEngine, OmegaMetrics};
pub use orchestrator::{OrchestratorConfig, SovereignOrchestrator};
pub use snr_engine::{SNRConfig, SNREngine, SignalMetrics};
