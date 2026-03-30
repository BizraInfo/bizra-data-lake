//! TopologyCanon v1 — Frozen Sovereign Topology
//!
//! The canonical registry of BIZRA's agent, node, and network topology.
//! This is the TOPOLOGY_CANON: the frozen truth about what entities
//! exist, what roles they play, and how they relate.
//!
//! Every surface (UI, API, docs, logs) MUST reference this canon.
//! Discrepancies between surfaces and canon are bugs, not features.
//!
//! Canonical counts:
//!   PAT-7: 7 Personal Agentic Team agents (user-sovereign)
//!   SAT-5: 5 Shared Agentic Team agents (system-level)
//!   Total: 12 agents per node

use serde::{Deserialize, Serialize};

/// PAT-7 agent definitions (Personal Agentic Team).
/// These are the user's sovereign council.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PatAgent {
    /// P1: Strategic planning and decomposition.
    Atlas,
    /// P2: Knowledge discovery and research.
    Oracle,
    /// P3: Code implementation and building.
    Forge,
    /// P4: Quality assessment and scoring.
    Judge,
    /// P5: Constitutional verification and ethics.
    Crown,
    /// P6: Delivery, publishing, and communication.
    Herald,
    /// P7: Cross-agent orchestration and integration.
    Nexus,
}

impl PatAgent {
    /// All PAT agents in canonical order.
    pub const ALL: [PatAgent; 7] = [
        Self::Atlas, Self::Oracle, Self::Forge, Self::Judge,
        Self::Crown, Self::Herald, Self::Nexus,
    ];

    /// Protocol callsign (used in receipts and logs).
    pub fn callsign(&self) -> &'static str {
        match self {
            Self::Atlas => "ATLAS",
            Self::Oracle => "ORACLE",
            Self::Forge => "FORGE",
            Self::Judge => "JUDGE",
            Self::Crown => "CROWN",
            Self::Herald => "HERALD",
            Self::Nexus => "NEXUS",
        }
    }

    /// Human-readable role name.
    pub fn role(&self) -> &'static str {
        match self {
            Self::Atlas => "Planner",
            Self::Oracle => "Researcher",
            Self::Forge => "Builder",
            Self::Judge => "Evaluator",
            Self::Crown => "Verifier",
            Self::Herald => "Publisher",
            Self::Nexus => "Integrator",
        }
    }

    /// Canonical agent index (P1-P7).
    pub fn index(&self) -> u8 {
        match self {
            Self::Atlas => 1,
            Self::Oracle => 2,
            Self::Forge => 3,
            Self::Judge => 4,
            Self::Crown => 5,
            Self::Herald => 6,
            Self::Nexus => 7,
        }
    }
}

/// SAT-5 agent definitions (Shared Agentic Team).
/// These are the system's immune system.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SatAgent {
    /// S1: Security monitoring and threat detection.
    Sentinel,
    /// S2: Quality scoring and Ihsan measurement.
    OracleSat,
    /// S3: Receipt chain integrity and ledger management.
    Ledger,
    /// S4: Task routing and resource coordination.
    Conductor,
    /// S5: External communication and federation.
    Ambassador,
}

impl SatAgent {
    /// All SAT agents in canonical order.
    pub const ALL: [SatAgent; 5] = [
        Self::Sentinel, Self::OracleSat, Self::Ledger,
        Self::Conductor, Self::Ambassador,
    ];

    /// Protocol callsign.
    pub fn callsign(&self) -> &'static str {
        match self {
            Self::Sentinel => "SENTINEL",
            Self::OracleSat => "ORACLE_SAT",
            Self::Ledger => "LEDGER",
            Self::Conductor => "CONDUCTOR",
            Self::Ambassador => "AMBASSADOR",
        }
    }

    /// Canonical agent index (S1-S5).
    pub fn index(&self) -> u8 {
        match self {
            Self::Sentinel => 1,
            Self::OracleSat => 2,
            Self::Ledger => 3,
            Self::Conductor => 4,
            Self::Ambassador => 5,
        }
    }
}

/// The canonical topology of a BIZRA sovereign node.
pub struct TopologyCanon;

impl TopologyCanon {
    /// Canonical PAT agent count.
    pub const PAT_COUNT: usize = 7;
    /// Canonical SAT agent count.
    pub const SAT_COUNT: usize = 5;
    /// Total agents per node.
    pub const TOTAL_AGENTS: usize = Self::PAT_COUNT + Self::SAT_COUNT;
    /// Gate chain order (canonical, from gates.rs:225).
    pub const GATE_ORDER: &'static [&'static str] = &["Schema", "Ihsan", "SNR"];
    /// Verdict precedence (canonical, from verdict.rs).
    pub const VERDICT_PRECEDENCE: &'static [&'static str] = &[
        "RIBA", "ZANN", "FATE", "Ihsan", "SNR",
    ];

    /// Validate that a PAT count matches canon.
    pub fn validate_pat_count(count: usize) -> bool {
        count == Self::PAT_COUNT
    }

    /// Validate that a SAT count matches canon.
    pub fn validate_sat_count(count: usize) -> bool {
        count == Self::SAT_COUNT
    }

    /// Validate total agent count.
    pub fn validate_total(count: usize) -> bool {
        count == Self::TOTAL_AGENTS
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pat_canonical_count() {
        assert_eq!(PatAgent::ALL.len(), TopologyCanon::PAT_COUNT);
        assert_eq!(TopologyCanon::PAT_COUNT, 7);
    }

    #[test]
    fn test_sat_canonical_count() {
        assert_eq!(SatAgent::ALL.len(), TopologyCanon::SAT_COUNT);
        assert_eq!(TopologyCanon::SAT_COUNT, 5);
    }

    #[test]
    fn test_total_agents() {
        assert_eq!(TopologyCanon::TOTAL_AGENTS, 12);
        assert!(TopologyCanon::validate_total(12));
        assert!(!TopologyCanon::validate_total(11));
    }

    #[test]
    fn test_pat_callsigns_unique() {
        let callsigns: Vec<_> = PatAgent::ALL.iter().map(|a| a.callsign()).collect();
        let unique: std::collections::HashSet<_> = callsigns.iter().collect();
        assert_eq!(callsigns.len(), unique.len());
    }

    #[test]
    fn test_sat_callsigns_unique() {
        let callsigns: Vec<_> = SatAgent::ALL.iter().map(|a| a.callsign()).collect();
        let unique: std::collections::HashSet<_> = callsigns.iter().collect();
        assert_eq!(callsigns.len(), unique.len());
    }

    #[test]
    fn test_gate_chain_order() {
        assert_eq!(TopologyCanon::GATE_ORDER, &["Schema", "Ihsan", "SNR"]);
    }

    #[test]
    fn test_verdict_precedence_order() {
        assert_eq!(TopologyCanon::VERDICT_PRECEDENCE[0], "RIBA");
        assert_eq!(TopologyCanon::VERDICT_PRECEDENCE[4], "SNR");
    }

    #[test]
    fn test_pat_indices_sequential() {
        for (i, agent) in PatAgent::ALL.iter().enumerate() {
            assert_eq!(agent.index() as usize, i + 1);
        }
    }
}
