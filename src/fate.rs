// src/fate.rs - FATE (Fail-Safe Agentic Trust Escalation) Module
// Handles quarantine, escalation, and human review routing
//
// PERSISTENCE: Uses Redis (Synapse) for durable escalation storage

use crate::sat::RejectionCode;
use crate::synapse::SynapseClient;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use tracing::{info, warn};

/// Global counter for escalation IDs
static ESCALATION_COUNTER: AtomicU64 = AtomicU64::new(1);

/// FATE escalation severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EscalationLevel {
    /// Low: Informational, auto-resolved
    Low,
    /// Medium: Requires logging, may need review
    Medium,
    /// High: Requires human review before proceeding
    High,
    /// Critical: Immediate block, security team notification
    Critical,
}

/// FATE escalation record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Escalation {
    /// Unique escalation ID
    pub id: String,
    /// Timestamp of escalation
    pub timestamp: DateTime<Utc>,
    /// Severity level
    pub level: EscalationLevel,
    /// Source component (SAT, PAT, Ihsan, etc.)
    pub source: String,
    /// Rejection code that triggered escalation
    pub rejection_code: String,
    /// Human-readable reason
    pub reason: String,
    /// Original request context (sanitized)
    pub context: HashMap<String, String>,
    /// Resolution status
    pub status: EscalationStatus,
    /// Recommended action
    pub recommended_action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum EscalationStatus {
    /// Pending human review
    Pending,
    /// Under review
    InReview,
    /// Approved to proceed
    Approved,
    /// Permanently blocked
    Blocked,
    /// Auto-resolved (low severity)
    AutoResolved,
}

/// FATE Coordinator - manages escalations and quarantine
pub struct FATECoordinator {
    /// Pending escalations (in-memory cache)
    pending_escalations: Vec<Escalation>,
    /// Redis client for persistence (optional)
    synapse: Option<SynapseClient>,
}

impl FATECoordinator {
    pub fn new() -> Self {
        info!("⚖️  Initializing FATE (Fail-Safe Agentic Trust Escalation)");
        Self {
            pending_escalations: Vec::new(),
            synapse: None,
        }
    }
    
    /// Create with Redis persistence
    pub fn with_synapse(synapse: SynapseClient) -> Self {
        info!("⚖️  Initializing FATE with Redis persistence");
        Self {
            pending_escalations: Vec::new(),
            synapse: Some(synapse),
        }
    }
    
    /// Create from environment (auto-detect Redis)
    pub async fn from_env() -> Self {
        match crate::synapse::SynapseClient::from_env().await {
            Ok(synapse) if synapse.is_available() => {
                info!("⚖️  FATE connected to Redis for persistent escalations");
                Self::with_synapse(synapse)
            }
            _ => {
                warn!("⚖️  FATE running without Redis (in-memory only)");
                Self::new()
            }
        }
    }

    /// Escalate a SAT rejection through FATE
    pub fn escalate_rejection(
        &mut self,
        rejection_codes: &[RejectionCode],
        task: &str,
        context: &HashMap<String, String>,
    ) -> Escalation {
        let level = Self::determine_level(rejection_codes);
        let id = format!(
            "FATE-{:06}",
            ESCALATION_COUNTER.fetch_add(1, Ordering::SeqCst)
        );

        // Sanitize context (remove potentially sensitive data)
        let sanitized_context: HashMap<String, String> = context
            .iter()
            .map(|(k, v)| {
                let sanitized_v = if k.to_lowercase().contains("password")
                    || k.to_lowercase().contains("secret")
                    || k.to_lowercase().contains("key")
                {
                    "[REDACTED]".to_string()
                } else if v.len() > 200 {
                    format!("{}...[truncated]", &v[..200])
                } else {
                    v.clone()
                };
                (k.clone(), sanitized_v)
            })
            .collect();

        let primary_rejection = rejection_codes.first().cloned().unwrap_or_else(|| {
            RejectionCode::ConsistencyFailure("Unknown rejection".to_string())
        });

        let reason = format!(
            "Task '{}' rejected by SAT: {}",
            if task.len() > 100 { &task[..100] } else { task },
            primary_rejection
        );

        let recommended_action = match &primary_rejection {
            RejectionCode::SecurityThreat(_) => {
                "BLOCK: Security threat detected. Do not execute under any circumstances."
            }
            RejectionCode::EthicsViolation(_) => {
                "BLOCK: Ethics violation. Requires ethics review before any action."
            }
            RejectionCode::Quarantine(_) => {
                "REVIEW: Uncertain request. Human judgment required before proceeding."
            }
            RejectionCode::PerformanceBudgetExceeded(_) => {
                "OPTIMIZE: Request exceeds performance budget. Consider breaking into smaller tasks."
            }
            RejectionCode::ConsistencyFailure(_) => {
                "CLARIFY: Request contains contradictions. Request clarification from user."
            }
            RejectionCode::ResourceConstraintViolated(_) => {
                "DEFER: Insufficient resources. Queue for later or reduce scope."
            }
        }
        .to_string();

        let escalation = Escalation {
            id: id.clone(),
            timestamp: Utc::now(),
            level: level.clone(),
            source: "SAT".to_string(),
            rejection_code: primary_rejection.to_string(),
            reason: reason.clone(),
            context: sanitized_context,
            status: match level {
                EscalationLevel::Low => EscalationStatus::AutoResolved,
                _ => EscalationStatus::Pending,
            },
            recommended_action,
        };

        match &level {
            EscalationLevel::Critical => {
                warn!(
                    escalation_id = %id,
                    level = ?level,
                    reason = %reason,
                    "🚨 FATE CRITICAL ESCALATION - Immediate security review required"
                );
            }
            EscalationLevel::High => {
                warn!(
                    escalation_id = %id,
                    level = ?level,
                    reason = %reason,
                    "⚠️ FATE HIGH ESCALATION - Human review required"
                );
            }
            EscalationLevel::Medium => {
                info!(
                    escalation_id = %id,
                    level = ?level,
                    reason = %reason,
                    "📋 FATE MEDIUM ESCALATION - Logged for review"
                );
            }
            EscalationLevel::Low => {
                info!(
                    escalation_id = %id,
                    level = ?level,
                    reason = %reason,
                    "ℹ️ FATE LOW ESCALATION - Auto-resolved"
                );
            }
        }

        // Store pending escalations (not auto-resolved)
        if escalation.status == EscalationStatus::Pending {
            // Note: Redis persistence happens via async method persist_to_synapse()
            // Also keep in memory for fast access
            self.pending_escalations.push(escalation.clone());
        }

        escalation
    }
    
    /// Persist escalation to Redis (call this separately if synapse is available)
    pub async fn persist_to_synapse(&self, escalation: &Escalation) -> Result<(), anyhow::Error> {
        if let Some(ref synapse) = self.synapse {
            let json = serde_json::to_string(escalation)?;
            synapse.push_fate_escalation(&escalation.id, &json).await?;
        }
        Ok(())
    }
    
    /// Get pending escalations from memory
    pub fn get_pending_escalations(&self) -> Vec<Escalation> {
        self.pending_escalations.clone()
    }
    
    /// Pop next pending escalation for review (async for Redis)
    pub async fn pop_pending_escalation_async(&mut self) -> Option<Escalation> {
        if let Some(ref synapse) = self.synapse {
            if let Ok(Some(json)) = synapse.pop_pending_escalation().await {
                if let Ok(escalation) = serde_json::from_str::<Escalation>(&json) {
                    // Remove from memory cache too
                    self.pending_escalations.retain(|e| e.id != escalation.id);
                    return Some(escalation);
                }
            }
        }
        self.pending_escalations.pop()
    }
    
    /// Resolve an escalation with Redis persistence (async)
    pub async fn resolve_escalation_async(&mut self, escalation_id: &str, approved: bool) -> bool {
        if let Some(ref synapse) = self.synapse {
            let resolution = if approved { "approved" } else { "blocked" };
            if synapse.resolve_escalation(escalation_id, resolution).await.is_ok() {
                self.pending_escalations.retain(|e| e.id != escalation_id);
                return true;
            }
        }
        
        // Fallback to memory-only resolution
        if let Some(pos) = self.pending_escalations.iter().position(|e| e.id == escalation_id) {
            let mut esc = self.pending_escalations.remove(pos);
            esc.status = if approved {
                EscalationStatus::Approved
            } else {
                EscalationStatus::Blocked
            };
            true
        } else {
            false
        }
    }

    /// Escalate an Ihsān threshold failure
    pub fn escalate_ihsan_failure(
        &mut self,
        env: &str,
        artifact_class: &str,
        score: f64,
        threshold: f64,
        context: &HashMap<String, String>,
    ) -> Escalation {
        let id = format!(
            "FATE-{:06}",
            ESCALATION_COUNTER.fetch_add(1, Ordering::SeqCst)
        );

        let reason = format!(
            "Ihsān gate failed: env={} artifact_class={} score={:.4} < threshold={:.4}",
            env, artifact_class, score, threshold
        );

        let escalation = Escalation {
            id: id.clone(),
            timestamp: Utc::now(),
            level: EscalationLevel::High,
            source: "IHSAN".to_string(),
            rejection_code: format!("IHSAN_THRESHOLD_FAILURE(score={:.4})", score),
            reason: reason.clone(),
            context: context.clone(),
            status: EscalationStatus::Pending,
            recommended_action: format!(
                "IMPROVE: Current Ihsān score ({:.4}) below {} threshold ({:.4}). Review quality dimensions.",
                score, env, threshold
            ),
        };

        warn!(
            escalation_id = %id,
            env = %env,
            score = score,
            threshold = threshold,
            "⚠️ FATE IHSAN ESCALATION - Quality threshold not met"
        );

        self.pending_escalations.push(escalation.clone());
        escalation
    }

    /// Determine escalation level from rejection codes
    fn determine_level(rejection_codes: &[RejectionCode]) -> EscalationLevel {
        for code in rejection_codes {
            match code {
                RejectionCode::SecurityThreat(_) => return EscalationLevel::Critical,
                RejectionCode::EthicsViolation(_) => return EscalationLevel::Critical,
                RejectionCode::Quarantine(_) => return EscalationLevel::High,
                _ => {}
            }
        }

        // Check for multiple moderate rejections
        let moderate_count = rejection_codes
            .iter()
            .filter(|c| {
                matches!(
                    c,
                    RejectionCode::PerformanceBudgetExceeded(_)
                        | RejectionCode::ConsistencyFailure(_)
                        | RejectionCode::ResourceConstraintViolated(_)
                )
            })
            .count();

        if moderate_count >= 2 {
            EscalationLevel::Medium
        } else if moderate_count == 1 {
            EscalationLevel::Low
        } else {
            EscalationLevel::Low
        }
    }

    /// Get all pending escalations
    pub fn pending_escalations(&self) -> &[Escalation] {
        &self.pending_escalations
    }

    /// Get pending escalation count
    pub fn pending_count(&self) -> usize {
        self.pending_escalations.len()
    }

    /// Resolve an escalation (for future human-in-the-loop)
    pub fn resolve_escalation(&mut self, id: &str, status: EscalationStatus) -> Option<&Escalation> {
        if let Some(escalation) = self.pending_escalations.iter_mut().find(|e| e.id == id) {
            escalation.status = status;
            Some(escalation)
        } else {
            None
        }
    }
}

impl Default for FATECoordinator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_security_escalation_is_critical() {
        let mut fate = FATECoordinator::new();
        let codes = vec![RejectionCode::SecurityThreat("SQL injection".to_string())];
        let escalation = fate.escalate_rejection(&codes, "test task", &HashMap::new());
        
        assert_eq!(escalation.level, EscalationLevel::Critical);
        assert_eq!(escalation.source, "SAT");
        assert!(escalation.rejection_code.contains("SECURITY_THREAT"));
    }

    #[test]
    fn test_quarantine_escalation_is_high() {
        let mut fate = FATECoordinator::new();
        let codes = vec![RejectionCode::Quarantine("uncertain intent".to_string())];
        let escalation = fate.escalate_rejection(&codes, "ambiguous task", &HashMap::new());
        
        assert_eq!(escalation.level, EscalationLevel::High);
        assert_eq!(escalation.status, EscalationStatus::Pending);
    }

    #[test]
    fn test_context_sanitization() {
        let mut fate = FATECoordinator::new();
        let codes = vec![RejectionCode::ConsistencyFailure("test".to_string())];
        let mut context = HashMap::new();
        context.insert("password".to_string(), "secret123".to_string());
        context.insert("user_input".to_string(), "normal_value".to_string());
        
        let escalation = fate.escalate_rejection(&codes, "test", &context);
        
        assert_eq!(escalation.context.get("password"), Some(&"[REDACTED]".to_string()));
        assert_eq!(escalation.context.get("user_input"), Some(&"normal_value".to_string()));
    }

    #[test]
    fn test_ihsan_escalation() {
        let mut fate = FATECoordinator::new();
        let escalation = fate.escalate_ihsan_failure(
            "ci",
            "docs",
            0.75,
            0.90,
            &HashMap::new(),
        );
        
        assert_eq!(escalation.level, EscalationLevel::High);
        assert_eq!(escalation.source, "IHSAN");
        assert!(escalation.reason.contains("0.75"));
    }
}
