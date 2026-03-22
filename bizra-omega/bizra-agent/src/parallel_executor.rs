// bizra-agent/src/parallel_executor.rs
// ============================================================
// Parallel Executor — bounded concurrent sub-agent execution
// ============================================================

use std::{
    sync::{Arc, Mutex},
    thread,
};

use bizra_telescript::Capability;

use crate::{
    action_types::ActionPlan,
    spawn_policy::{SpawnDenied, SpawnPolicy},
    sub_agent::{SubAgent, SubAgentSpawner, SubAgentStatus},
};

#[derive(Debug, Clone)]
pub struct SubAgentResult {
    pub agent_id: String,
    pub parent_id: String,
    pub status: SubAgentStatus,
    pub duration_ms: u64,
}

pub struct ParallelExecutor {
    pub spawner: SubAgentSpawner,
}

impl ParallelExecutor {
    pub fn new(policy: SpawnPolicy) -> Self {
        Self {
            spawner: SubAgentSpawner::new(policy),
        }
    }

    pub fn execute_parallel(
        &mut self,
        parent: &mut SubAgent,
        plans: Vec<ActionPlan>,
        now: u64,
    ) -> Vec<SubAgentResult> {
        let mut spawned = Vec::new();
        for plan in plans {
            match self
                .spawner
                .spawn(parent, plan, vec![Capability::Compute], now, true)
            {
                Ok(mut child) => {
                    child.status = SubAgentStatus::Running;
                    spawned.push(child);
                }
                Err(_denied) => {
                    // Denials are dropped from active execution set by design.
                }
            }
        }

        let results: Arc<Mutex<Vec<SubAgentResult>>> = Arc::new(Mutex::new(Vec::new()));
        let mut handles = Vec::new();
        for child in spawned {
            let results_ref = Arc::clone(&results);
            handles.push(thread::spawn(move || {
                // Simulated work for deterministic testability.
                let status = if child.task.steps.is_empty() {
                    SubAgentStatus::Failed("empty_plan".to_string())
                } else {
                    SubAgentStatus::Complete("ok".to_string())
                };
                let out = SubAgentResult {
                    agent_id: child.id.clone(),
                    parent_id: child.parent_id.clone(),
                    status,
                    duration_ms: 1,
                };
                if let Ok(mut guard) = results_ref.lock() {
                    guard.push(out);
                }
            }));
        }

        for h in handles {
            let _ = h.join();
            self.spawner.mark_complete();
        }
        results.lock().map(|v| v.clone()).unwrap_or_default()
    }

    pub fn try_spawn_only(
        &mut self,
        parent: &mut SubAgent,
        plan: ActionPlan,
        now: u64,
    ) -> Result<SubAgent, SpawnDenied> {
        self.spawner
            .spawn(parent, plan, vec![Capability::Compute], now, true)
    }
}

impl Default for ParallelExecutor {
    fn default() -> Self {
        Self::new(SpawnPolicy::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action_types::{ActionChannel, ActionKind, PlannedStep};

    fn plan(id: &str) -> ActionPlan {
        ActionPlan {
            plan_id: id.to_string(),
            created_at: 1,
            steps: vec![PlannedStep {
                channel: ActionChannel::DesktopRpc,
                kind: ActionKind::Click,
                payload: "{\"target\":\"x\"}".to_string(),
            }],
        }
    }

    #[test]
    fn executes_parallel_and_collects_results() {
        let mut exec = ParallelExecutor::default();
        let mut root = SubAgent::root([9u8; 32], 1);
        let out = exec.execute_parallel(&mut root, vec![plan("a"), plan("b")], 2);
        assert_eq!(out.len(), 2);
        assert!(out
            .iter()
            .all(|r| matches!(r.status, SubAgentStatus::Complete(_))));
    }

    #[test]
    fn active_limit_is_enforced() {
        let mut exec = ParallelExecutor::new(SpawnPolicy {
            max_total_active: 1,
            ..Default::default()
        });
        let mut root = SubAgent::root([3u8; 32], 1);
        let first = exec.try_spawn_only(&mut root, plan("a"), 2);
        assert!(first.is_ok());
        let second = exec.try_spawn_only(&mut root, plan("b"), 2);
        assert_eq!(second.unwrap_err(), SpawnDenied::GlobalLimitReached);
    }
}
