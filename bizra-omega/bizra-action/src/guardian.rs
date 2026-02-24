//! # The Guardian — Constitutional Gate on Every Action
//!
//! The Guardian is the Daughter Test encoded in software.
//! Before any action touches the real world, the Guardian asks:
//! "Would I trust this action to serve someone I love?"
//!
//! ## Gating Logic
//!
//! 1. **Permit check**: Is this channel permitted?
//! 2. **Iḥsān check**: Does the plan score meet the risk-adjusted threshold?
//! 3. **Scope check**: Is the action within permitted boundaries?
//! 4. **Resource check**: Is there budget remaining?
//! 5. **HITL check**: Does this action require human confirmation?
//!
//! All five must pass. Any failure produces a denial with a specific
//! violation type — never a generic "denied." The denial itself becomes
//! part of the constitutional receipt.
//!
//! ## Standing on Giants
//! - **Lampson (1974)**: Capability-based security — permits define what's allowed
//! - **Al-Ghazali**: The Guardian embodies إحسان — excellence through constraint

use crate::types::*;

/// The Guardian — constitutional gatekeeper for all actions.
#[derive(Debug)]
pub struct Guardian {
    /// Counters for monitoring.
    actions_approved: u64,
    actions_denied: u64,
    hitl_requests: u64,

    /// Current resource usage (monotonically increasing per session).
    resource_used: u64,

    /// Consecutive approvals (for stability tracking).
    consecutive_approvals: u64,

    /// Whether the Guardian is in strict mode (higher thresholds).
    strict_mode: bool,
}

impl Guardian {
    /// Create a new Guardian in standard mode.
    pub fn new() -> Self {
        Self {
            actions_approved: 0,
            actions_denied: 0,
            hitl_requests: 0,
            resource_used: 0,
            consecutive_approvals: 0,
            strict_mode: false,
        }
    }

    /// Create a Guardian in strict mode (for visiting agents).
    pub fn strict() -> Self {
        let mut g = Self::new();
        g.strict_mode = true;
        g
    }

    /// Evaluate an action against its permit and constitutional constraints.
    ///
    /// This is the core gate. Every action in BIZRA passes through here.
    /// Returns a verdict: Approved, Denied (with reason), or RequiresHitl.
    pub fn evaluate(&mut self, envelope: &ActionEnvelope) -> GuardianVerdict {
        let channel = envelope.action.channel();
        let permit = &envelope.permit;
        let risk = channel.risk_level();

        // ── Gate 1: Channel Permission ──────────────────────
        if !permit.allows_channel(&channel) {
            self.record_denial();
            return GuardianVerdict::Denied {
                reason: format!(
                    "Channel '{}' not permitted by current permit",
                    channel.name()
                ),
                violation: GuardianViolation::ChannelNotPermitted { channel },
            };
        }

        // ── Gate 2: Desktop Permission ──────────────────────
        if channel == Channel::Ahk && !permit.allow_desktop {
            self.record_denial();
            return GuardianVerdict::Denied {
                reason: "Desktop manipulation not permitted by current permit".into(),
                violation: GuardianViolation::DesktopNotPermitted,
            };
        }

        // ── Gate 3: Network Permission ──────────────────────
        if matches!(channel, Channel::Browser | Channel::Telescript) && !permit.allow_network {
            self.record_denial();
            return GuardianVerdict::Denied {
                reason: "Network egress not permitted by current permit".into(),
                violation: GuardianViolation::NetworkNotPermitted,
            };
        }

        // ── Gate 4: Iḥsān Threshold ────────────────────────
        let required_ihsan = if self.strict_mode {
            // Strict mode: higher bar
            (risk.min_ihsan() + 0.02).min(1.0)
        } else {
            risk.min_ihsan()
        };

        if envelope.plan_ihsan.value() < required_ihsan {
            self.record_denial();
            return GuardianVerdict::Denied {
                reason: format!(
                    "Iḥsān score {:.4} below threshold {:.4} for {} risk",
                    envelope.plan_ihsan.value(),
                    required_ihsan,
                    match risk {
                        RiskLevel::Low => "low",
                        RiskLevel::Medium => "medium",
                        RiskLevel::High => "high",
                    }
                ),
                violation: GuardianViolation::IhsanBelowThreshold {
                    score: envelope.plan_ihsan.value(),
                    required: required_ihsan,
                },
            };
        }

        // ── Gate 5: File Scope Check ────────────────────────
        if let Some(path) = self.extract_path(&envelope.action) {
            if !self.path_in_scope(&path, &permit.fs_scope) {
                self.record_denial();
                return GuardianVerdict::Denied {
                    reason: format!("Path '{}' outside permitted scope", path),
                    violation: GuardianViolation::PathOutOfScope {
                        path,
                        scope: permit.fs_scope.clone(),
                    },
                };
            }
        }

        // ── Gate 6: Resource Budget ─────────────────────────
        let action_cost = self.estimate_cost(&envelope.action);
        if self.resource_used + action_cost > permit.resource_limit {
            self.record_denial();
            return GuardianVerdict::Denied {
                reason: format!(
                    "Resource budget exceeded: used {} + cost {} > limit {}",
                    self.resource_used, action_cost, permit.resource_limit
                ),
                violation: GuardianViolation::ResourceExceeded {
                    used: self.resource_used + action_cost,
                    limit: permit.resource_limit,
                },
            };
        }

        // ── Gate 7: HITL Check ──────────────────────────────
        if permit.requires_hitl && risk >= RiskLevel::Medium {
            self.hitl_requests += 1;
            return GuardianVerdict::RequiresHitl {
                reason: format!(
                    "Human confirmation required for {} action on {} channel",
                    match risk {
                        RiskLevel::Low => "low-risk",
                        RiskLevel::Medium => "medium-risk",
                        RiskLevel::High => "high-risk",
                    },
                    channel.name()
                ),
                action_summary: envelope.action.summary(),
            };
        }

        // ── All gates passed ────────────────────────────────
        self.resource_used += action_cost;
        self.record_approval();

        GuardianVerdict::Approved {
            reason: "All constitutional gates passed",
        }
    }

    // ── Internal helpers ────────────────────────────────────

    fn record_approval(&mut self) {
        self.actions_approved += 1;
        self.consecutive_approvals += 1;
    }

    fn record_denial(&mut self) {
        self.actions_denied += 1;
        self.consecutive_approvals = 0;
    }

    /// Extract file path from action, if applicable.
    fn extract_path(&self, action: &BizraAction) -> Option<String> {
        match action {
            BizraAction::FileCreate { path, .. }
            | BizraAction::FileRead { path }
            | BizraAction::FileDelete { path } => Some(path.clone()),
            _ => None,
        }
    }

    /// Check if a path falls within permitted scope.
    fn path_in_scope(&self, path: &str, scope: &[String]) -> bool {
        if scope.is_empty() {
            return false;
        }
        scope
            .iter()
            .any(|s| if s == "*" { true } else { path.starts_with(s) })
    }

    /// Estimate the resource cost of an action.
    fn estimate_cost(&self, action: &BizraAction) -> u64 {
        match action {
            // AHK actions: low cost (local execution)
            BizraAction::AhkClick { .. } => 10,
            BizraAction::AhkType { .. } => 10,
            BizraAction::AhkRead { .. } => 5,
            BizraAction::AhkReflex { .. } => 5,
            BizraAction::AhkLaunch { .. } => 20,
            BizraAction::AhkPerceive => 15,

            // LLM actions: high cost (GPU inference)
            BizraAction::LlmQuery { max_tokens, .. } => *max_tokens as u64 * 2,
            BizraAction::LlmStream { max_tokens, .. } => *max_tokens as u64 * 2,

            // Memory: moderate
            BizraAction::MemoryStore { content, .. } => content.len() as u64 / 10,
            BizraAction::MemoryRecall { top_k, .. } => *top_k as u64 * 5,
            BizraAction::MemoryUpdateKnownMe { .. } => 10,

            // MCP: moderate
            BizraAction::McpToolCall { .. } => 100,

            // File: low
            BizraAction::FileCreate { content, .. } => content.len() as u64 / 100,
            BizraAction::FileRead { .. } => 5,
            BizraAction::FileDelete { .. } => 5,

            // Browser: moderate
            BizraAction::BrowserNavigate { .. } => 50,
            BizraAction::BrowserFetch { .. } => 100,

            // Response: low (returning to user)
            BizraAction::RespondToUser { .. } => 10,

            // Telescript: high (agent travel)
            BizraAction::TelescriptGo { agent_state, .. } => agent_state.len() as u64,
            BizraAction::TelescriptMeet { .. } => 200,
        }
    }

    // ── Public stats ────────────────────────────────────────

    /// Total actions evaluated.
    pub fn total_evaluated(&self) -> u64 {
        self.actions_approved + self.actions_denied + self.hitl_requests
    }

    /// Approval rate.
    pub fn approval_rate(&self) -> f64 {
        let total = self.total_evaluated();
        if total == 0 {
            1.0
        } else {
            self.actions_approved as f64 / total as f64
        }
    }

    /// Current resource usage.
    pub fn resource_used(&self) -> u64 {
        self.resource_used
    }

    /// Consecutive approvals (stability metric).
    pub fn consecutive_approvals(&self) -> u64 {
        self.consecutive_approvals
    }

    /// Guardian health snapshot.
    pub fn health(&self) -> GuardianHealth {
        GuardianHealth {
            total_evaluated: self.total_evaluated(),
            approved: self.actions_approved,
            denied: self.actions_denied,
            hitl_requested: self.hitl_requests,
            approval_rate: self.approval_rate(),
            resource_used: self.resource_used,
            consecutive_approvals: self.consecutive_approvals,
            strict_mode: self.strict_mode,
        }
    }
}

impl Default for Guardian {
    fn default() -> Self {
        Self::new()
    }
}

/// Guardian health snapshot for monitoring.
#[derive(Debug, Clone)]
pub struct GuardianHealth {
    pub total_evaluated: u64,
    pub approved: u64,
    pub denied: u64,
    pub hitl_requested: u64,
    pub approval_rate: f64,
    pub resource_used: u64,
    pub consecutive_approvals: u64,
    pub strict_mode: bool,
}
