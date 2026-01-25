/*
 * BIZRA SOVEREIGN KERNEL v1.0
 * The "FATE Gate" (Formalized Alignment & Transcendence Engine)
 *
 * This module uses Z3 SMT Solver to mathematically prove that every action
 * of the AI complies with the BIZRA Constitution BEFORE the CPU executes it.
 *
 * Logic: Every Agent Proposal 'P' must satisfy Constitution 'C'.
 * Proof: Solve(P AND C) == SAT.
 *
 * Three Invariants (The Covenant):
 *   1. Anti-Debt (Riba == 0) - No interest-based transactions
 *   2. Ihsan Floor (>= 0.99) - Excellence threshold
 *   3. Anti-Assumption (Evidence > 0) - No hallucinations allowed
 */

use crate::errors::BridgeError;
use tracing::{debug, info, warn, instrument};
use z3::{ast::{Ast, Bool, Int}, Config, Context, SatResult, Solver};

/// Represents an agent's proposed action for formal verification
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

/// Result of formal verification
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
    /// Z3 solver status
    pub solver_status: String,
}

/// The Sovereign Kernel - Formal Verification Engine
///
/// Uses Z3 SMT Solver to prove agent actions comply with the BIZRA Constitution.
/// This is the "Logic Gate of the Covenant" - no action executes without proof.
pub struct SovereignKernel<'ctx> {
    ctx: &'ctx Context,
    constitution: Vec<Bool<'ctx>>,
    invariant_names: Vec<String>,
}

impl<'ctx> SovereignKernel<'ctx> {
    /// Create a new Sovereign Kernel with the given Z3 context
    ///
    /// # Example
    /// ```rust,ignore
    /// let cfg = Config::new();
    /// let ctx = Context::new(&cfg);
    /// let kernel = SovereignKernel::new(&ctx);
    /// ```
    pub fn new(ctx: &'ctx Context) -> Self {
        let mut kernel = Self {
            ctx,
            constitution: Vec::new(),
            invariant_names: Vec::new(),
        };
        kernel.initialize_invariants();
        info!("Sovereign Kernel initialized with {} constitutional invariants",
              kernel.constitution.len());
        kernel
    }

    /// Initialize the constitutional invariants (The Covenant)
    ///
    /// These are hard-coded into the "physics" of the system:
    /// 1. Anti-Debt (Riba == 0)
    /// 2. Ihsan Floor (>= 99%)
    /// 3. Anti-Assumption (Evidence > 0)
    fn initialize_invariants(&mut self) {
        let zero = Int::from_i64(self.ctx, 0);
        let threshold = Int::from_i64(self.ctx, 99);

        // Invariant 1: Anti-Debt (Riba == 0)
        // The system CANNOT propose interest-bearing transactions
        let interest_rate = Int::new_const(self.ctx, "interest_rate");
        let anti_debt = interest_rate._eq(&zero);
        self.constitution.push(anti_debt);
        self.invariant_names.push("Anti-Debt (Riba == 0)".to_string());
        debug!("Loaded invariant: Anti-Debt (Riba == 0)");

        // Invariant 2: Ihsan Floor (Ihsan >= 0.99)
        // All actions must meet the excellence threshold
        let ihsan_metric = Int::new_const(self.ctx, "ihsan_metric");
        let ihsan_floor = ihsan_metric.ge(&threshold);
        self.constitution.push(ihsan_floor);
        self.invariant_names.push("Ihsan Floor (>= 0.99)".to_string());
        debug!("Loaded invariant: Ihsan Floor (>= 0.99)");

        // Invariant 3: Anti-Assumption (Evidence > 0)
        // No action without evidence from the Data Lake
        let evidence_count = Int::new_const(self.ctx, "evidence_count");
        let anti_assumption = evidence_count.gt(&zero);
        self.constitution.push(anti_assumption);
        self.invariant_names.push("Anti-Assumption (Evidence > 0)".to_string());
        debug!("Loaded invariant: Anti-Assumption (Evidence > 0)");
    }

    /// Formally verify an agent's proposed action against the Constitution
    ///
    /// This is the "Money Shot" - the mathematical proof that the action
    /// complies with all invariants before execution is allowed.
    ///
    /// # Arguments
    /// * `action` - The agent's proposed action to verify
    ///
    /// # Returns
    /// * `Ok(VerificationResult)` - Verification completed (may be verified or not)
    /// * `Err(BridgeError)` - Verification could not be completed
    #[instrument(skip(self, action), fields(ihsan = %action.ihsan, evidence = %action.context.atoms.len()))]
    pub fn verify_intent(&self, action: &AgentAction) -> Result<VerificationResult, BridgeError> {
        let solver = Solver::new(self.ctx);

        // Load the Constitution (all invariants must hold)
        for invariant in &self.constitution {
            solver.assert(invariant);
        }

        // Map the Agent's Action to First-Order Logic
        let action_ihsan = Int::from_i64(self.ctx, (action.ihsan * 100.0) as i64);
        let action_interest = Int::from_i64(self.ctx, action.metadata.proposed_interest as i64);
        let action_evidence = Int::from_i64(self.ctx, action.context.atoms.len() as i64);

        // Assert the Action's parameters match the constitutional variables
        let ihsan_var = Int::new_const(self.ctx, "ihsan_metric");
        let interest_var = Int::new_const(self.ctx, "interest_rate");
        let evidence_var = Int::new_const(self.ctx, "evidence_count");

        solver.assert(&ihsan_var._eq(&action_ihsan));
        solver.assert(&interest_var._eq(&action_interest));
        solver.assert(&evidence_var._eq(&action_evidence));

        // CHECK: Is the Action logically consistent with the Covenant?
        let result = match solver.check() {
            SatResult::Sat => {
                info!("Verification PASSED: Action satisfies Constitution");
                VerificationResult {
                    verified: true,
                    explanation: "Action mathematically proven to comply with Constitution".to_string(),
                    invariants_checked: self.invariant_names.clone(),
                    violations: vec![],
                    solver_status: "SAT".to_string(),
                }
            }
            SatResult::Unsat => {
                // Identify which invariants were violated
                let violations = self.identify_violations(action);
                warn!("Verification FAILED: Action violates Constitution - {:?}", violations);
                VerificationResult {
                    verified: false,
                    explanation: format!("Action violates constitutional invariants: {:?}", violations),
                    invariants_checked: self.invariant_names.clone(),
                    violations,
                    solver_status: "UNSAT".to_string(),
                }
            }
            SatResult::Unknown => {
                warn!("Verification INCONCLUSIVE: Z3 solver returned Unknown");
                return Err(BridgeError::ProtocolError(
                    "Formal verification inconclusive - cannot prove or disprove compliance".to_string()
                ));
            }
        };

        Ok(result)
    }

    /// Identify which specific invariants were violated
    fn identify_violations(&self, action: &AgentAction) -> Vec<String> {
        let mut violations = Vec::new();

        // Check Anti-Debt
        if action.metadata.proposed_interest > 0 {
            violations.push(format!(
                "Anti-Debt violation: proposed_interest = {} (must be 0)",
                action.metadata.proposed_interest
            ));
        }

        // Check Ihsan Floor
        if action.ihsan < 0.99 {
            violations.push(format!(
                "Ihsan Floor violation: ihsan = {:.4} (must be >= 0.99)",
                action.ihsan
            ));
        }

        // Check Anti-Assumption
        if action.context.atoms.is_empty() {
            violations.push("Anti-Assumption violation: no evidence atoms provided".to_string());
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

/// Global Z3 context factory
/// Creates a new Z3 context with optimized settings
pub fn create_z3_context() -> Context {
    let mut cfg = Config::new();
    cfg.set_proof_generation(false); // Disable proof generation for speed
    cfg.set_model_generation(true);  // Enable model generation for debugging
    Context::new(&cfg)
}

/// Convenience function to verify an action
pub fn verify_action(action: &AgentAction) -> Result<VerificationResult, BridgeError> {
    let ctx = create_z3_context();
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
        let ctx = create_z3_context();
        let kernel = SovereignKernel::new(&ctx);

        // Compliant action: Ihsan >= 0.99, interest = 0, evidence > 0
        let action = create_test_action(0.99, 0, 5);
        let result = kernel.verify_intent(&action).unwrap();

        assert!(result.verified, "Compliant action should pass verification");
        assert!(result.violations.is_empty());
        assert_eq!(result.solver_status, "SAT");
    }

    #[test]
    fn test_low_ihsan_fails() {
        let ctx = create_z3_context();
        let kernel = SovereignKernel::new(&ctx);

        // Low Ihsan: 0.85 < 0.99
        let action = create_test_action(0.85, 0, 5);
        let result = kernel.verify_intent(&action).unwrap();

        assert!(!result.verified, "Low Ihsan action should fail verification");
        assert!(result.violations.iter().any(|v| v.contains("Ihsan Floor")));
    }

    #[test]
    fn test_interest_fails() {
        let ctx = create_z3_context();
        let kernel = SovereignKernel::new(&ctx);

        // Interest > 0 (Riba violation)
        let action = create_test_action(0.99, 5, 5);
        let result = kernel.verify_intent(&action).unwrap();

        assert!(!result.verified, "Interest-bearing action should fail verification");
        assert!(result.violations.iter().any(|v| v.contains("Anti-Debt")));
    }

    #[test]
    fn test_no_evidence_fails() {
        let ctx = create_z3_context();
        let kernel = SovereignKernel::new(&ctx);

        // No evidence atoms (hallucination risk)
        let action = create_test_action(0.99, 0, 0);
        let result = kernel.verify_intent(&action).unwrap();

        assert!(!result.verified, "Action without evidence should fail verification");
        assert!(result.violations.iter().any(|v| v.contains("Anti-Assumption")));
    }

    #[test]
    fn test_quick_ihsan_check() {
        let ctx = create_z3_context();
        let kernel = SovereignKernel::new(&ctx);

        assert!(kernel.quick_ihsan_check(0.99));
        assert!(kernel.quick_ihsan_check(1.0));
        assert!(!kernel.quick_ihsan_check(0.98));
        assert!(!kernel.quick_ihsan_check(0.5));
    }
}
