// bizra-agent/src/permit_guard.rs
// ============================================================
// Permit Guard — local capability + resource budget enforcement
// ============================================================

use std::collections::VecDeque;

use bizra_telescript::{Capability, Permit};

use crate::action_types::{ActionChannel, ActionError, ActionKind, ActionPlan, PlannedStep};

#[derive(Debug, Clone)]
pub struct PermitBudgetConfig {
    pub max_actions_per_minute: u32,
    pub max_actions_per_session: u32,
    pub max_desktop_clicks_per_plan: u32,
    pub max_file_writes_per_plan: u32,
    pub max_llm_calls_per_plan: u32,
    pub ttl_seconds: u64,
}

impl Default for PermitBudgetConfig {
    fn default() -> Self {
        Self {
            max_actions_per_minute: 30,
            max_actions_per_session: 500,
            max_desktop_clicks_per_plan: 20,
            max_file_writes_per_plan: 5,
            max_llm_calls_per_plan: 10,
            ttl_seconds: 300,
        }
    }
}

#[derive(Debug, Clone, Default)]
pub struct PermitUsage {
    pub session_actions: u32,
    pub plan_clicks: u32,
    pub plan_file_writes: u32,
    pub plan_llm_calls: u32,
    action_timestamps: VecDeque<u64>,
}

impl PermitUsage {
    pub fn reset_for_new_plan(&mut self) {
        self.plan_clicks = 0;
        self.plan_file_writes = 0;
        self.plan_llm_calls = 0;
    }
}

pub struct PermitGuard {
    config: PermitBudgetConfig,
}

impl PermitGuard {
    pub fn new(config: PermitBudgetConfig) -> Self {
        Self { config }
    }

    pub fn config(&self) -> &PermitBudgetConfig {
        &self.config
    }

    pub fn capability_for_step(step: &PlannedStep) -> Capability {
        match (step.channel, step.kind) {
            (_, ActionKind::Click) | (_, ActionKind::TypeText) | (_, ActionKind::InvokeSkill) => {
                Capability::Compute
            }
            (_, ActionKind::WriteFile) => Capability::Store,
            _ => Capability::Network,
        }
    }

    pub fn validate_plan(
        &self,
        plan: &ActionPlan,
        permit: &Permit,
        usage: &PermitUsage,
        now: u64,
    ) -> Result<(), ActionError> {
        if !permit.verify() {
            return Err(ActionError::new(
                "PERMIT_INVALID",
                "Permit failed integrity/expiry verification",
            ));
        }

        // Dry-run: validate against a shadow copy so real usage is not mutated.
        let mut shadow = usage.clone();
        shadow.reset_for_new_plan();
        for step in &plan.steps {
            self.validate_step(step, permit, &shadow, now)?;
            self.record_step_usage(step, &mut shadow, now);
        }
        Ok(())
    }

    pub fn validate_step(
        &self,
        step: &PlannedStep,
        permit: &Permit,
        usage: &PermitUsage,
        now: u64,
    ) -> Result<(), ActionError> {
        step.validate()?;

        let required = Self::capability_for_step(step);
        if !permit.has_capability(required) {
            return Err(ActionError::new(
                "PERMIT_DENIED",
                format!("Permit missing required capability: {:?}", required).as_str(),
            ));
        }

        if usage.session_actions >= self.config.max_actions_per_session {
            return Err(ActionError::new(
                "SESSION_BUDGET_EXCEEDED",
                "Action session budget exceeded",
            ));
        }

        let in_last_minute = usage
            .action_timestamps
            .iter()
            .filter(|&&ts| now.saturating_sub(ts) <= 60)
            .count() as u32;
        if in_last_minute >= self.config.max_actions_per_minute {
            return Err(ActionError::new(
                "RATE_LIMITED",
                "Action rate limit exceeded in rolling minute window",
            ));
        }

        match (step.channel, step.kind) {
            (ActionChannel::DesktopRpc, ActionKind::Click) => {
                if usage.plan_clicks >= self.config.max_desktop_clicks_per_plan {
                    return Err(ActionError::new(
                        "PLAN_CLICK_BUDGET_EXCEEDED",
                        "Desktop click budget exceeded for action plan",
                    ));
                }
            }
            (_, ActionKind::WriteFile) => {
                if usage.plan_file_writes >= self.config.max_file_writes_per_plan {
                    return Err(ActionError::new(
                        "PLAN_FILE_WRITE_BUDGET_EXCEEDED",
                        "File write budget exceeded for action plan",
                    ));
                }
            }
            (_, ActionKind::ToolCall) | (_, ActionKind::Query) => {
                if usage.plan_llm_calls >= self.config.max_llm_calls_per_plan {
                    return Err(ActionError::new(
                        "PLAN_LLM_BUDGET_EXCEEDED",
                        "LLM/remote call budget exceeded for action plan",
                    ));
                }
            }
            _ => {}
        }

        Ok(())
    }

    pub fn record_step_usage(&self, step: &PlannedStep, usage: &mut PermitUsage, now: u64) {
        usage.session_actions += 1;
        usage.action_timestamps.push_back(now);
        while let Some(front) = usage.action_timestamps.front().copied() {
            if now.saturating_sub(front) > 60 {
                let _ = usage.action_timestamps.pop_front();
            } else {
                break;
            }
        }

        match (step.channel, step.kind) {
            (ActionChannel::DesktopRpc, ActionKind::Click) => usage.plan_clicks += 1,
            (_, ActionKind::WriteFile) => usage.plan_file_writes += 1,
            (_, ActionKind::ToolCall) | (_, ActionKind::Query) => usage.plan_llm_calls += 1,
            _ => {}
        }
    }
}

impl Default for PermitGuard {
    fn default() -> Self {
        Self::new(PermitBudgetConfig::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action_types::{ActionChannel, ActionKind, ActionPlan, PlannedStep};
    use bizra_telescript::{Authority, Permit, ResourceLimits};

    fn permit_with_caps(capabilities: Vec<Capability>) -> Permit {
        Permit::new(
            Authority::genesis(),
            capabilities,
            ResourceLimits::default(),
            600,
        )
    }

    #[test]
    fn validates_capability_mapping() {
        let guard = PermitGuard::default();
        let mut usage = PermitUsage::default();
        let permit = permit_with_caps(vec![Capability::Compute]);
        let step = PlannedStep {
            channel: ActionChannel::DesktopRpc,
            kind: ActionKind::Click,
            payload: "{\"target\":\"btn\"}".to_string(),
        };
        assert!(guard.validate_step(&step, &permit, &usage, 100).is_ok());
        guard.record_step_usage(&step, &mut usage, 100);
        assert_eq!(usage.session_actions, 1);
    }

    #[test]
    fn rejects_missing_capability() {
        let guard = PermitGuard::default();
        let usage = PermitUsage::default();
        let permit = permit_with_caps(vec![Capability::Store]);
        let step = PlannedStep {
            channel: ActionChannel::DesktopRpc,
            kind: ActionKind::Click,
            payload: "{\"target\":\"btn\"}".to_string(),
        };
        let err = guard
            .validate_step(&step, &permit, &usage, 100)
            .unwrap_err();
        assert_eq!(err.code, "PERMIT_DENIED");
    }

    #[test]
    fn enforces_plan_click_budget() {
        let guard = PermitGuard::default();
        let usage = PermitUsage {
            plan_clicks: guard.config.max_desktop_clicks_per_plan,
            ..Default::default()
        };
        let permit = permit_with_caps(vec![Capability::Compute]);
        let step = PlannedStep {
            channel: ActionChannel::DesktopRpc,
            kind: ActionKind::Click,
            payload: "{\"target\":\"btn\"}".to_string(),
        };
        let err = guard
            .validate_step(&step, &permit, &usage, 100)
            .unwrap_err();
        assert_eq!(err.code, "PLAN_CLICK_BUDGET_EXCEEDED");
    }

    #[test]
    fn validates_full_plan() {
        let guard = PermitGuard::default();
        let permit = permit_with_caps(vec![Capability::Compute, Capability::Store]);
        let plan = ActionPlan {
            plan_id: "pln".to_string(),
            created_at: 1,
            steps: vec![
                PlannedStep {
                    channel: ActionChannel::DesktopRpc,
                    kind: ActionKind::Click,
                    payload: "{\"target\":\"open\"}".to_string(),
                },
                PlannedStep {
                    channel: ActionChannel::FileOp,
                    kind: ActionKind::WriteFile,
                    payload: "{\"path\":\"/tmp/x\",\"content\":\"ok\"}".to_string(),
                },
            ],
        };
        let usage = PermitUsage::default();
        assert!(guard.validate_plan(&plan, &permit, &usage, 100).is_ok());
    }
}
