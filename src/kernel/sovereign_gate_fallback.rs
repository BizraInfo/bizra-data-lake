/*
 * BIZRA SOVEREIGN KERNEL v1.0 - Fallback Implementation
 *
 * This module provides a fallback verification implementation when Z3 SMT solver
 * is not available. It performs runtime checks instead of formal SMT verification.
 *
 * The same three invariants (The Covenant) are enforced:
 *   1. Anti-Debt (Riba == 0) - No interest-based transactions
 *   2. Ihsan Floor (>= 0.99) - Excellence threshold
 *   3. Anti-Assumption (Evidence > 0) - No hallucinations allowed
 *
 * NOTE: This fallback does NOT provide mathematical proof - it only performs
 * runtime validation. For production deployments requiring formal verification,
 * build with the `z3-solver` feature enabled.
 */

use crate::errors::BridgeError;
use tracing::{debug, info, warn, instrument};

/// Represents an agent's proposed action for verification
#[derive(Debug, Clone)]
pub struct AgentAction {
    /// The Ihsan excellence score (0.0 - 1.0)
    pub ihsan: f64,
    /// Action metadata including financial parameters
    pub metadata: ActionMetadata,
    /// Context with evidence atoms
    pub context: ActionContext,
}

#[derive(Debug, Clone)]
pub struct ActionMetadata {
    /// Any proposed interest rate (must be 0 for compliance)
    pub proposed_interest: u32,
    /// Action type identifier
    pub action_type: String,
    /// Optional monetary value involved
    pub monetary_value: Option<u64>,
}

#[derive(Debug, Clone)]
pub struct ActionContext {
    /// Evidence atoms supporting the action (from Data Lake)
    pub atoms: Vec<EvidenceAtom>,
    /// Source tier (M1-M6)
    pub source_tier: String,
    /// Query that generated this context
    pub query: String,
}

#[derive(Debug, Clone)]
pub struct EvidenceAtom {
    /// Unique identifier for the evidence
    pub id: String,
    /// Content of the evidence
    pub content: String,
    /// Confidence score (0.0 - 1.0)
    pub confidence: f64,
    /// Source reference (e.g., Data Lake node ID)
    pub source: String,
}

/// Result of verification
#[derive(Debug, Clone)]
pub struct VerificationResult {
    /// Whether the action satisfies the constitution
    pub verified: bool,
    /// Human-readable explanation
    pub explanation: String,
    /// Which invariants were checked
    pub invariants_checked: Vec<String>,
    /// Any violations detected
    pub violations: Vec<String>,
    /// Verification method status
    pub solver_status: String,
}

/// Verification context (placeholder for Z3 Context compatibility)
#[derive(Debug, Clone, Default)]
pub struct VerificationContext {
    /// Mode indicator
    pub mode: String,
}

impl VerificationContext {
    pub fn new() -> Self {
        Self {
            mode: "fallback_runtime".to_string(),
        }
    }
}

/// The Sovereign Kernel - Fallback Verification Engine
///
/// Performs runtime verification of agent actions against the BIZRA Constitution.
/// This is a fallback implementation when Z3 SMT Solver is not available.
pub struct SovereignKernel {
    invariant_names: Vec<String>,
    context_mode: String,
}

impl SovereignKernel {
    /// Create a new Sovereign Kernel
    pub fn new(_ctx: &VerificationContext) -> Self {
        let invariant_names = vec![
            "Anti-Debt (Riba == 0)".to_string(),
            "Ihsan Floor (>= 0.99)".to_string(),
            "Anti-Assumption (Evidence > 0)".to_string(),
        ];

        info!("Sovereign Kernel initialized (fallback mode) with {} constitutional invariants",
              invariant_names.len());

        Self {
            invariant_names,
            context_mode: "fallback_runtime".to_string(),
        }
    }

    /// Verify an agent's proposed action against the Constitution
    ///
    /// This performs runtime checks (not formal SMT verification).
    ///
    /// # Arguments
    /// * `action` - The agent's proposed action to verify
    ///
    /// # Returns
    /// * `Ok(VerificationResult)` - Verification completed
    /// * `Err(BridgeError)` - Verification could not be completed
    #[instrument(skip(self, action), fields(ihsan = %action.ihsan, evidence = %action.context.atoms.len()))]
    pub fn verify_intent(&self, action: &AgentAction) -> Result<VerificationResult, BridgeError> {
        let violations = self.identify_violations(action);

        if violations.is_empty() {
            info!("Verification PASSED (runtime): Action satisfies Constitution");
            Ok(VerificationResult {
                verified: true,
                explanation: "Action verified via runtime checks (Z3 not available)".to_string(),
                invariants_checked: self.invariant_names.clone(),
                violations: vec![],
                solver_status: "RUNTIME_PASS".to_string(),
            })
        } else {
            warn!("Verification FAILED (runtime): Action violates Constitution - {:?}", violations);
            Ok(VerificationResult {
                verified: false,
                explanation: format!("Action violates constitutional invariants: {:?}", violations),
                invariants_checked: self.invariant_names.clone(),
                violations,
                solver_status: "RUNTIME_FAIL".to_string(),
            })
        }
    }

    /// Identify which specific invariants were violated
    fn identify_violations(&self, action: &AgentAction) -> Vec<String> {
        let mut violations = Vec::new();

        // Check Anti-Debt (Riba == 0)
        if action.metadata.proposed_interest > 0 {
            violations.push(format!(
                "Anti-Debt violation: proposed_interest = {} (must be 0)",
                action.metadata.proposed_interest
            ));
            debug!("Anti-Debt invariant violated: interest = {}", action.metadata.proposed_interest);
        }

        // Check Ihsan Floor (>= 0.99)
        if action.ihsan < 0.99 {
            violations.push(format!(
                "Ihsan Floor violation: ihsan = {:.4} (must be >= 0.99)",
                action.ihsan
            ));
            debug!("Ihsan Floor invariant violated: ihsan = {:.4}", action.ihsan);
        }

        // Check Anti-Assumption (Evidence > 0)
        if action.context.atoms.is_empty() {
            violations.push("Anti-Assumption violation: no evidence atoms provided".to_string());
            debug!("Anti-Assumption invariant violated: no evidence");
        }

        violations
    }

    /// Quick verification that only checks Ihsan score
    /// Used for fast-path validation before full verification
    pub fn quick_ihsan_check(&self, ihsan: f64) -> bool {
        ihsan >= 0.99
    }

    /// Verify a batch of actions (for parallel processing)
    pub fn verify_batch(&self, actions: &[AgentAction]) -> Vec<Result<VerificationResult, BridgeError>> {
        actions.iter().map(|a| self.verify_intent(a)).collect()
    }
}

/// Create a verification context (fallback mode)
pub fn create_verification_context() -> VerificationContext {
    VerificationContext::new()
}

/// Convenience function to verify an action
pub fn verify_action(action: &AgentAction) -> Result<VerificationResult, BridgeError> {
    let ctx = create_verification_context();
    let kernel = SovereignKernel::new(&ctx);
    kernel.verify_intent(action)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_action(ihsan: f64, interest: u32, evidence_count: usize) -> AgentAction {
        let atoms: Vec<EvidenceAtom> = (0..evidence_count)
            .map(|i| EvidenceAtom {
                id: format!("atom_{}", i),
                content: format!("Test evidence {}", i),
                confidence: 0.95,
                source: "M6_SOVEREIGN".to_string(),
            })
            .collect();

        AgentAction {
            ihsan,
            metadata: ActionMetadata {
                proposed_interest: interest,
                action_type: "test".to_string(),
                monetary_value: None,
            },
            context: ActionContext {
                atoms,
                source_tier: "M6".to_string(),
                query: "test query".to_string(),
            },
        }
    }

    #[test]
    fn test_compliant_action_passes() {
        let ctx = create_verification_context();
        let kernel = SovereignKernel::new(&ctx);

        // Compliant action: Ihsan >= 0.99, interest = 0, evidence > 0
        let action = create_test_action(0.99, 0, 5);
        let result = kernel.verify_intent(&action).unwrap();

        assert!(result.verified, "Compliant action should pass verification");
        assert!(result.violations.is_empty());
        assert_eq!(result.solver_status, "RUNTIME_PASS");
    }

    #[test]
    fn test_low_ihsan_fails() {
        let ctx = create_verification_context();
        let kernel = SovereignKernel::new(&ctx);

        // Low Ihsan: 0.85 < 0.99
        let action = create_test_action(0.85, 0, 5);
        let result = kernel.verify_intent(&action).unwrap();

        assert!(!result.verified, "Low Ihsan action should fail verification");
        assert!(result.violations.iter().any(|v| v.contains("Ihsan Floor")));
    }

    #[test]
    fn test_interest_fails() {
        let ctx = create_verification_context();
        let kernel = SovereignKernel::new(&ctx);

        // Interest > 0 (Riba violation)
        let action = create_test_action(0.99, 5, 5);
        let result = kernel.verify_intent(&action).unwrap();

        assert!(!result.verified, "Interest-bearing action should fail verification");
        assert!(result.violations.iter().any(|v| v.contains("Anti-Debt")));
    }

    #[test]
    fn test_no_evidence_fails() {
        let ctx = create_verification_context();
        let kernel = SovereignKernel::new(&ctx);

        // No evidence atoms (hallucination risk)
        let action = create_test_action(0.99, 0, 0);
        let result = kernel.verify_intent(&action).unwrap();

        assert!(!result.verified, "Action without evidence should fail verification");
        assert!(result.violations.iter().any(|v| v.contains("Anti-Assumption")));
    }

    #[test]
    fn test_quick_ihsan_check() {
        let ctx = create_verification_context();
        let kernel = SovereignKernel::new(&ctx);

        assert!(kernel.quick_ihsan_check(0.99));
        assert!(kernel.quick_ihsan_check(1.0));
        assert!(!kernel.quick_ihsan_check(0.98));
        assert!(!kernel.quick_ihsan_check(0.5));
    }
}
