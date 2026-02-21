// bizra-agent/src/action_bus.rs
// ============================================================
// Action Bus — validated, fail-closed dispatch contract
// ============================================================

use std::collections::HashMap;

use bizra_telescript::Permit;

use crate::action_types::{
    ActionChannel, ActionError, ActionExecutionStatus, ActionPlan, ActionResult, PlannedStep,
};
use crate::permit_guard::{PermitGuard, PermitUsage};

pub struct ActionBus {
    permit_guard: PermitGuard,
    executors: HashMap<ActionChannel, bool>,
}

impl ActionBus {
    pub fn new(permit_guard: PermitGuard) -> Self {
        Self {
            permit_guard,
            executors: HashMap::new(),
        }
    }

    pub fn register_executor(&mut self, channel: ActionChannel) {
        self.executors.insert(channel, true);
    }

    pub fn unregister_executor(&mut self, channel: ActionChannel) {
        self.executors.remove(&channel);
    }

    pub fn has_executor(&self, channel: ActionChannel) -> bool {
        self.executors.get(&channel).copied().unwrap_or(false)
    }

    pub fn validate_plan(
        &self,
        plan: &ActionPlan,
        permit: &Permit,
        usage: &PermitUsage,
        now: u64,
    ) -> Result<(), ActionError> {
        plan.validate()?;
        self.permit_guard.validate_plan(plan, permit, usage, now)?;
        for step in &plan.steps {
            if !self.has_executor(step.channel) {
                return Err(ActionError::new(
                    "MISSING_EXECUTOR",
                    format!(
                        "No executor registered for channel {}",
                        step.channel.as_str()
                    )
                    .as_str(),
                ));
            }
        }
        Ok(())
    }

    pub fn dispatch_step(
        &self,
        step: &PlannedStep,
        permit: &Permit,
        usage: &mut PermitUsage,
        now: u64,
    ) -> Result<ActionResult, ActionError> {
        if !self.has_executor(step.channel) {
            return Err(ActionError::new(
                "MISSING_EXECUTOR",
                format!(
                    "No executor registered for channel {}",
                    step.channel.as_str()
                )
                .as_str(),
            ));
        }
        self.permit_guard.validate_step(step, permit, usage, now)?;
        self.permit_guard.record_step_usage(step, usage, now);

        Ok(ActionResult {
            action_id: format!("act_{}_{}", step.channel.as_str(), now),
            plan_id: "adhoc".to_string(),
            status: ActionExecutionStatus::Running,
            message: format!("dispatch:{}:{}", step.channel.as_str(), step.kind.as_str()),
            started_at: now,
            finished_at: now,
        })
    }
}

impl Default for ActionBus {
    fn default() -> Self {
        let mut bus = Self::new(PermitGuard::default());
        bus.register_executor(ActionChannel::DesktopRpc);
        bus.register_executor(ActionChannel::ToolCall);
        bus.register_executor(ActionChannel::LlmCall);
        bus.register_executor(ActionChannel::FileOp);
        bus.register_executor(ActionChannel::BrowserNav);
        bus
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::action_types::{ActionChannel, ActionKind, ActionPlan, PlannedStep};
    use bizra_telescript::{Authority, Capability, Permit, ResourceLimits};

    fn permit_all() -> Permit {
        Permit::new(
            Authority::genesis(),
            vec![Capability::Compute, Capability::Network, Capability::Store],
            ResourceLimits::default(),
            600,
        )
    }

    #[test]
    fn fails_when_executor_missing() {
        let mut bus = ActionBus::new(PermitGuard::default());
        bus.register_executor(ActionChannel::DesktopRpc);
        let plan = ActionPlan {
            plan_id: "pln".to_string(),
            created_at: 1,
            steps: vec![PlannedStep {
                channel: ActionChannel::BrowserNav,
                kind: ActionKind::Navigate,
                payload: "{\"url\":\"https://example.com\"}".to_string(),
            }],
        };
        let usage = PermitUsage::default();
        let err = bus
            .validate_plan(&plan, &permit_all(), &usage, 100)
            .unwrap_err();
        assert_eq!(err.code, "MISSING_EXECUTOR");
    }

    #[test]
    fn validates_and_dispatches_step() {
        let bus = ActionBus::default();
        let step = PlannedStep {
            channel: ActionChannel::DesktopRpc,
            kind: ActionKind::Click,
            payload: "{\"target\":\"ok\"}".to_string(),
        };
        let mut usage = PermitUsage::default();
        let out = bus
            .dispatch_step(&step, &permit_all(), &mut usage, 123)
            .expect("dispatch ok");
        assert_eq!(out.status, ActionExecutionStatus::Running);
        assert!(out.message.contains("dispatch:DesktopRpc:Click"));
    }
}
