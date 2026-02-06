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
pub mod orchestrator;
pub mod graph_of_thoughts;
pub mod snr_engine;
pub mod giants;
pub mod omega;

pub use error::{SovereignError, SovereignResult, ErrorContext};
pub use orchestrator::{SovereignOrchestrator, OrchestratorConfig};
pub use graph_of_thoughts::{ThoughtGraph, ThoughtNode, ThoughtType, ReasoningPath, AggregateResult, GraphStats};
pub use snr_engine::{SNREngine, SignalMetrics, SNRConfig};
pub use giants::{GiantRegistry, Giant, Contribution};
pub use omega::{OmegaEngine, OmegaConfig, OmegaMetrics, CircuitState};
