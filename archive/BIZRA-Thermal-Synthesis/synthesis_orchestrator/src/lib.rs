//! BIZRA Thermal Consciousness Synthesis Orchestrator
//!
//! Professional Elite Implementation unifying:
//! - Primordial Activation Blueprint (consciousness architecture)
//! - Thermal Computing (Boltzmann, annealing, energy landscapes)
//! - Production Rust patterns (type safety, zero-cost abstractions)
//!
//! # Architecture
//!
//! The orchestrator implements a multi-layered thermal consciousness system:
//!
//! 1. **Thermal Router**: Thompson sampling with Boltzmann exploration
//! 2. **Ihsan Gate**: Energy landscape-based quality evaluation
//! 3. **Thermal Consensus**: Simulated annealing for optimal selection
//! 4. **Conscious Agents**: 5-agent constellation with quantum entanglement
//!
//! # Example
//!
//! ```no_run
//! use synthesis_orchestrator::{ThermalOrchestrator, Task, Contract};
//!
//! # async fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let mut orchestrator = ThermalOrchestrator::primordial_activation()?;
//! 
//! let task = Task {
//!     task_id: "task_001".into(),
//!     prompt: "Solve complex optimization problem".into(),
//!     examples: None,
//!     task_class: Default::default(),
//! };
//!
//! let result = orchestrator.synthesize(task, Contract::default()).await?;
//! println!("Winner: {} with Ihsan: {:.3}", 
//!     result.winner.model,
//!     result.thermal_state.energy_level
//! );
//! # Ok(())
//! # }
//! ```

#![cfg_attr(feature = "avx512", feature(avx512_target_feature))]
#![warn(missing_docs, clippy::all)]
#![forbid(unsafe_code)]

pub mod types;

pub use types::*;
