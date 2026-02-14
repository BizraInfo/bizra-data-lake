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

/// Sovereign error types and `Result` alias.
pub mod error;
/// Sovereign Experience Ledger — content-addressed episodic memory.
pub mod experience_ledger;
/// Standing on Giants — attribution registry.
pub mod giants;
/// Graph-of-Thoughts reasoning engine.
pub mod graph_of_thoughts;
/// Self-Evolving Judgment Engine — observation telemetry.
pub mod judgment_telemetry;
/// Omega circuit breaker — resilience and metrics.
pub mod omega;
/// Top-level reasoning orchestrator.
pub mod orchestrator;
/// Signal-to-Noise Ratio measurement engine.
pub mod snr_engine;

pub use error::{ErrorContext, SovereignError, SovereignResult};
pub use experience_ledger::{Episode, EpisodeAction, EpisodeImpact, ExperienceLedger, RIRConfig};
pub use giants::{Contribution, Giant, GiantRegistry};
pub use graph_of_thoughts::{
    AggregateResult, GraphStats, ReasoningPath, ThoughtGraph, ThoughtNode, ThoughtType,
};
pub use judgment_telemetry::{simulate_epoch_distribution, JudgmentTelemetry, JudgmentVerdict};
pub use omega::{CircuitState, OmegaConfig, OmegaEngine, OmegaMetrics};
pub use orchestrator::{OrchestratorConfig, SovereignOrchestrator};
pub use snr_engine::{SNRConfig, SNREngine, SignalMetrics};
