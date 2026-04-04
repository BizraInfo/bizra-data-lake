// src/unified/mod.rs - SAPE Ultimate Integration Module
//
// The Unified Host Architecture - Cybernetic Organism
// =====================================================
// Brain (Python) + Body (Rust) + Soul (Ihsān Protocol)
//
// This module implements the synthesis from SAPE v1.∞ analysis:
// - Sidecar Pattern: Python cognitive engine as stateless microservice
// - Symbolic Grounding: WisdomAtom with Horn clause representation
// - Proactive Attestation: Cryptographic behavior logging
// - Pipelined Consciousness: Async SAT updating PAT intuition

pub mod attestor;
pub mod cognitive_bridge;
pub mod orchestrator;
pub mod wisdom;

pub use attestor::CryptographicAttestor;
pub use cognitive_bridge::CognitiveBridge;
pub use orchestrator::UnifiedOrchestrator;
pub use wisdom::{ActionPrimitive, Symbol, WisdomAtom, WisdomStore};
