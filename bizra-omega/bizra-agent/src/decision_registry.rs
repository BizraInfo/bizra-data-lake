// bizra-agent/src/decision_registry.rs
// ============================================================
// GENESIS Decision Registry — glass-box, retrieval-only artifacts
// ============================================================

use std::collections::HashMap;

use crate::hash_namespace::{parse_hex_32, ActionHash, TriggerHash};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CognitiveMode {
    System1,
    System2,
}

impl CognitiveMode {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::System1 => "system1",
            Self::System2 => "system2",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MissionPhase {
    Meaning,
    TruthFinding,
    Execution,
    Compression,
}

impl MissionPhase {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Meaning => "meaning",
            Self::TruthFinding => "truth_finding",
            Self::Execution => "execution",
            Self::Compression => "compression",
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RejectedAlternative {
    pub route: String,
    pub reason: String,
}

#[derive(Debug, Clone)]
pub struct DecisionArtifact {
    pub action_hash: ActionHash,
    pub trigger_hash: TriggerHash,
    pub decision_mode: CognitiveMode,
    pub mission_phase: MissionPhase,
    pub micro_path: Vec<String>,
    pub chosen_route: String,
    pub rejected_alternatives: Vec<RejectedAlternative>,
    pub guardian_verdict: bool,
    pub ihsan_at_decision: f32,
    pub snr_at_decision: f32,
    pub timestamp: u64,
    pub policy_hash: [u8; 32],
}

pub struct DecisionRegistry {
    by_action: HashMap<ActionHash, DecisionArtifact>,
    append_order: Vec<ActionHash>,
    max_entries: usize,
}

impl DecisionRegistry {
    pub fn new(max_entries: usize) -> Self {
        Self {
            by_action: HashMap::new(),
            append_order: Vec::new(),
            max_entries: max_entries.max(1),
        }
    }

    pub fn append(&mut self, artifact: DecisionArtifact) {
        let key = artifact.action_hash;
        self.by_action.insert(key, artifact);
        self.append_order.retain(|k| *k != key);
        self.append_order.push(key);
        self.evict_if_needed();
    }

    pub fn get(&self, action_hash: &ActionHash) -> Option<&DecisionArtifact> {
        self.by_action.get(action_hash)
    }

    pub fn get_by_hex(&self, action_hash_hex: &str) -> Option<&DecisionArtifact> {
        let bytes = parse_hex_32(action_hash_hex)?;
        let action_hash = ActionHash(bytes);
        self.get(&action_hash)
    }

    pub fn len(&self) -> usize {
        self.by_action.len()
    }

    pub fn is_empty(&self) -> bool {
        self.by_action.is_empty()
    }

    fn evict_if_needed(&mut self) {
        while self.by_action.len() > self.max_entries {
            let Some(oldest) = self.append_order.first().copied() else {
                break;
            };
            self.append_order.remove(0);
            self.by_action.remove(&oldest);
        }
    }
}

impl Default for DecisionRegistry {
    fn default() -> Self {
        Self::new(4096)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_artifact(action: [u8; 32], trigger: [u8; 32], ts: u64) -> DecisionArtifact {
        DecisionArtifact {
            action_hash: ActionHash(action),
            trigger_hash: TriggerHash(trigger),
            decision_mode: CognitiveMode::System2,
            mission_phase: MissionPhase::TruthFinding,
            micro_path: vec![
                "Retrieve".to_string(),
                "Verify".to_string(),
                "Act".to_string(),
            ],
            chosen_route: "RetrieveContext>GenerateResponse".to_string(),
            rejected_alternatives: vec![RejectedAlternative {
                route: "GenerateOnly".to_string(),
                reason: "context_required".to_string(),
            }],
            guardian_verdict: true,
            ihsan_at_decision: 0.97,
            snr_at_decision: 0.92,
            timestamp: ts,
            policy_hash: [9u8; 32],
        }
    }

    #[test]
    fn append_and_get_by_hex_roundtrip() {
        let mut reg = DecisionRegistry::new(8);
        let art = sample_artifact([1u8; 32], [2u8; 32], 100);
        let key_hex = art.action_hash.to_hex();
        reg.append(art);
        let fetched = reg.get_by_hex(&key_hex).expect("artifact should exist");
        assert_eq!(fetched.decision_mode, CognitiveMode::System2);
        assert_eq!(fetched.mission_phase, MissionPhase::TruthFinding);
    }

    #[test]
    fn bounded_registry_evicts_oldest() {
        let mut reg = DecisionRegistry::new(2);
        reg.append(sample_artifact([1u8; 32], [1u8; 32], 1));
        reg.append(sample_artifact([2u8; 32], [2u8; 32], 2));
        reg.append(sample_artifact([3u8; 32], [3u8; 32], 3));
        assert_eq!(reg.len(), 2);
        assert!(reg.get(&ActionHash([1u8; 32])).is_none());
        assert!(reg.get(&ActionHash([3u8; 32])).is_some());
    }
}
