// bizra-agent/src/sub_agent.rs
// ============================================================
// Sub-Agent model with degraded permit inheritance
// ============================================================

use std::collections::HashSet;

use bizra_telescript::Capability;

use crate::action_types::ActionPlan;
use crate::spawn_policy::{SpawnDenied, SpawnPolicy};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SubAgentStatus {
    Pending,
    Running,
    Complete(String),
    Failed(String),
    Killed(String),
}

#[derive(Debug, Clone)]
pub struct SubAgentPermit {
    pub capabilities: Vec<Capability>,
    pub max_actions: u32,
    pub max_llm_calls: u8,
    pub ttl_seconds: u64,
    pub max_spawn_depth: u8,
    pub policy_hash: [u8; 32],
    pub guardian_required: bool,
}

impl SubAgentPermit {
    pub fn degrade(
        &self,
        factor: f32,
        required_capabilities: &[Capability],
        parent_remaining_ttl: u64,
    ) -> Self {
        let required: HashSet<Capability> = required_capabilities.iter().copied().collect();
        let caps = self
            .capabilities
            .iter()
            .copied()
            .filter(|c| required.contains(c))
            .collect::<Vec<_>>();

        let scaled_actions = ((self.max_actions as f32) * factor).floor() as u32;
        let scaled_llm = ((self.max_llm_calls as f32) * factor).floor() as u8;
        let scaled_ttl = ((self.ttl_seconds as f32) * factor).floor() as u64;
        let ttl_seconds = scaled_ttl.min(parent_remaining_ttl).max(1);

        Self {
            capabilities: caps,
            max_actions: scaled_actions.max(1),
            max_llm_calls: scaled_llm.max(1),
            ttl_seconds,
            max_spawn_depth: self.max_spawn_depth.saturating_sub(1),
            policy_hash: self.policy_hash,
            guardian_required: true,
        }
    }

    pub fn allows(&self, capability: Capability) -> bool {
        self.capabilities.contains(&capability)
    }
}

#[derive(Debug, Clone)]
pub struct SubAgent {
    pub id: String,
    pub parent_id: String,
    pub depth: u8,
    pub permit: SubAgentPermit,
    pub task: ActionPlan,
    pub status: SubAgentStatus,
    pub spawn_time: u64,
    pub deadline: u64,
    pub children_spawned: u8,
}

impl SubAgent {
    pub fn root(policy_hash: [u8; 32], now: u64) -> Self {
        Self {
            id: "root".to_string(),
            parent_id: "root".to_string(),
            depth: 0,
            permit: SubAgentPermit {
                capabilities: vec![Capability::Compute, Capability::Network, Capability::Store],
                max_actions: 500,
                max_llm_calls: 10,
                ttl_seconds: 300,
                max_spawn_depth: 2,
                policy_hash,
                guardian_required: true,
            },
            task: ActionPlan {
                plan_id: "root".to_string(),
                created_at: now,
                steps: Vec::new(),
            },
            status: SubAgentStatus::Running,
            spawn_time: now,
            deadline: now + 300,
            children_spawned: 0,
        }
    }

    pub fn kill_if_expired(&mut self, now: u64) -> bool {
        if now > self.deadline && !matches!(self.status, SubAgentStatus::Killed(_)) {
            self.status = SubAgentStatus::Killed("ttl_expired".to_string());
            return true;
        }
        false
    }
}

pub struct SubAgentSpawner {
    pub policy: SpawnPolicy,
    active_count: u16,
}

impl SubAgentSpawner {
    pub fn new(policy: SpawnPolicy) -> Self {
        Self {
            policy,
            active_count: 0,
        }
    }

    pub fn active_count(&self) -> u16 {
        self.active_count
    }

    pub fn spawn(
        &mut self,
        parent: &mut SubAgent,
        task: ActionPlan,
        required_capabilities: Vec<Capability>,
        now: u64,
        guardian_approved: bool,
    ) -> Result<SubAgent, SpawnDenied> {
        if parent.depth >= self.policy.max_depth {
            return Err(SpawnDenied::MaxDepthReached);
        }
        if parent.children_spawned >= self.policy.max_children_per_agent {
            return Err(SpawnDenied::MaxChildrenReached);
        }
        if self.active_count >= self.policy.max_total_active {
            return Err(SpawnDenied::GlobalLimitReached);
        }
        if self.policy.guardian_on_spawn && !guardian_approved {
            return Err(SpawnDenied::GuardianDenied);
        }

        let parent_remaining_ttl = parent.deadline.saturating_sub(now).max(1);
        let child_permit = parent.permit.degrade(
            self.policy.permit_degradation,
            &required_capabilities,
            parent_remaining_ttl,
        );
        if child_permit.capabilities.is_empty() {
            return Err(SpawnDenied::PermitInsufficient);
        }

        parent.children_spawned += 1;
        self.active_count += 1;
        let id = format!("{}-{}", parent.id, parent.children_spawned);
        Ok(SubAgent {
            id,
            parent_id: parent.id.clone(),
            depth: parent.depth + 1,
            permit: child_permit.clone(),
            task,
            status: SubAgentStatus::Pending,
            spawn_time: now,
            deadline: now + child_permit.ttl_seconds,
            children_spawned: 0,
        })
    }

    pub fn mark_complete(&mut self) {
        self.active_count = self.active_count.saturating_sub(1);
    }
}

impl Default for SubAgentSpawner {
    fn default() -> Self {
        Self::new(SpawnPolicy::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action_types::{ActionChannel, ActionKind, PlannedStep};

    fn simple_plan(id: &str, ts: u64) -> ActionPlan {
        ActionPlan {
            plan_id: id.to_string(),
            created_at: ts,
            steps: vec![PlannedStep {
                channel: ActionChannel::DesktopRpc,
                kind: ActionKind::Click,
                payload: "{\"target\":\"x\"}".to_string(),
            }],
        }
    }

    #[test]
    fn child_permit_is_subset_and_degraded() {
        let parent = SubAgentPermit {
            capabilities: vec![Capability::Compute, Capability::Network, Capability::Store],
            max_actions: 100,
            max_llm_calls: 10,
            ttl_seconds: 120,
            max_spawn_depth: 2,
            policy_hash: [1u8; 32],
            guardian_required: true,
        };
        let child = parent.degrade(0.5, &[Capability::Network], 100);
        assert_eq!(child.capabilities, vec![Capability::Network]);
        assert!(child.max_actions <= parent.max_actions);
        assert!(child.ttl_seconds <= 100);
    }

    #[test]
    fn enforces_spawn_limits() {
        let mut spawner = SubAgentSpawner::new(SpawnPolicy {
            max_depth: 1,
            ..Default::default()
        });
        let mut root = SubAgent::root([2u8; 32], 10);
        let child = spawner
            .spawn(
                &mut root,
                simple_plan("p1", 10),
                vec![Capability::Compute],
                10,
                true,
            )
            .expect("first spawn succeeds");
        let mut child_mut = child.clone();
        let err = spawner
            .spawn(
                &mut child_mut,
                simple_plan("p2", 11),
                vec![Capability::Compute],
                11,
                true,
            )
            .unwrap_err();
        assert_eq!(err, SpawnDenied::MaxDepthReached);
    }
}
