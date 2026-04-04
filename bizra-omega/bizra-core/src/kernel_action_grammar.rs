//! Kernel Action Grammar — Constitutional Computability Doctrine v1
//!
//! Defines the finite set of actions the microkernel is allowed to decide
//! directly as blocking syscalls. Everything outside this grammar must be
//! escalated to bounded review or constitutional judiciary.
//!
//! Design principle: kernel law must be decidable. Any blocking syscall
//! must terminate within a fixed budget and return one of a finite set
//! of verdicts. Not all ethics are kernel law.
//!
//! Standing on Giants: Gödel (incompleteness), Turing (decidability),
//! L4/seL4 (microkernel verification), BIZRA constitutional spine.

use std::time::Duration;

// ─────────────────────────────────────────────────────────
// Constitutional Verdict
// ─────────────────────────────────────────────────────────

/// The four possible outcomes of any constitutional evaluation.
///
/// Path C doctrine: not everything is PERMIT or REJECT.
/// Without REVIEW and SCORE_ONLY, the system will either lie or deadlock.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConstitutionalVerdict {
    /// Action is constitutionally valid. Proceed.
    Permit,
    /// Action violates hard constitutional law. Block unconditionally.
    Reject,
    /// Action requires bounded review — too complex for kernel decision.
    /// Low-risk actions may proceed with monitoring; high-risk fail closed.
    Review,
    /// Advisory evaluation only — produces score but does not gate.
    /// Used by Layer 3 (judiciary/advisory) for post-hoc analysis.
    ScoreOnly,
}

impl ConstitutionalVerdict {
    /// Whether this verdict allows the action to proceed immediately.
    pub fn is_permissive(&self) -> bool {
        matches!(self, Self::Permit | Self::ScoreOnly)
    }

    /// Whether this verdict blocks the action.
    pub fn is_blocking(&self) -> bool {
        matches!(self, Self::Reject)
    }

    /// Whether this verdict requires escalation.
    pub fn requires_escalation(&self) -> bool {
        matches!(self, Self::Review)
    }
}

// ─────────────────────────────────────────────────────────
// Constitutional Layer
// ─────────────────────────────────────────────────────────

/// The three layers of the stratified constitution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConstitutionalLayer {
    /// Layer 1 — Hard Constitution (kernel law).
    /// Finite-state, bounded-time, deterministic, replayable, receipt-native.
    /// This is physics.
    HardLaw,

    /// Layer 2 — Bounded Constitutional Review.
    /// Gate-relevant under explicit computational budgets.
    /// Timeout-aware, fail-closed for high-risk classes.
    BoundedReview,

    /// Layer 3 — Constitutional Judiciary / Advisory.
    /// Full expressiveness: 8D Ihsan, whole-graph Adl, Guardian Council.
    /// Produces scores, explanations, appeals — not blocking syscalls.
    /// This is jurisprudence.
    Judiciary,
}

// ─────────────────────────────────────────────────────────
// Kernel Action Grammar
// ─────────────────────────────────────────────────────────

/// The finite set of actions the microkernel may decide as blocking syscalls.
///
/// Every action in this grammar must be:
/// - finite-state
/// - bounded-time (terminates within `max_budget`)
/// - deterministic (same inputs → same verdict)
/// - replayable (receipt captures full decision context)
/// - receipt-native (produces or consumes CanonicalReceipt)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelAction {
    /// Verify Ed25519 signature on a receipt or attestation.
    SignatureVerify,
    /// Validate input against schema (JSON Schema / type check).
    SchemaValidate,
    /// Check ReceiptStateMachine transition legality.
    TransitionCheck,
    /// Compare pre-computed scalar against constitutional threshold.
    /// Examples: Ihsan >= 0.95, SNR >= 0.99, Gini <= 0.35.
    ThresholdCompare,
    /// Apply non-bypassable constitutional veto.
    /// RIBA_ZERO, ZANN_ZERO, and other frozen invariants.
    ConstitutionalVeto,
    /// Enforce resource cap / risk cap / local invariant.
    ResourceCap,
    /// Verify BLAKE3 chain integrity (receipt chaining).
    ChainIntegrity,
    /// Verify receipt completeness (no gaps in provenance).
    ReceiptCompleteness,
    /// Mint SEED token from verified work proof.
    SeedMint,
    /// Verify BLOOM accrual/decay from sustained excellence.
    BloomAccrue,
}

impl KernelAction {
    /// Maximum time budget for this kernel action.
    /// Hard constitutional law: every syscall must terminate within this bound.
    pub fn max_budget(&self) -> Duration {
        match self {
            Self::SignatureVerify => Duration::from_millis(5),
            Self::SchemaValidate => Duration::from_millis(10),
            Self::TransitionCheck => Duration::from_millis(2),
            Self::ThresholdCompare => Duration::from_micros(100),
            Self::ConstitutionalVeto => Duration::from_micros(50),
            Self::ResourceCap => Duration::from_millis(1),
            Self::ChainIntegrity => Duration::from_millis(5),
            Self::ReceiptCompleteness => Duration::from_millis(10),
            Self::SeedMint => Duration::from_millis(20),
            Self::BloomAccrue => Duration::from_millis(15),
        }
    }

    /// Which constitutional layer owns this action.
    /// All KernelActions are Layer 1 by definition.
    pub fn layer(&self) -> ConstitutionalLayer {
        ConstitutionalLayer::HardLaw
    }

    /// Whether this action is a frozen invariant (cannot be modified by governance).
    pub fn is_frozen(&self) -> bool {
        matches!(self, Self::ConstitutionalVeto | Self::ChainIntegrity)
    }
}

// ─────────────────────────────────────────────────────────
// Review Action Grammar
// ─────────────────────────────────────────────────────────

/// Actions that require bounded constitutional review (Layer 2).
/// These are gate-relevant but operate under explicit computational budgets.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ReviewAction {
    /// Project multi-dimensional Ihsan score from weighted components.
    IhsanProjection,
    /// Compute approximate Adl delta (Gini impact of this action).
    AdlDeltaApprox,
    /// Bounded provenance traversal (verify chain depth N).
    ProvenanceTraversal,
    /// SMT-bounded formal verification (Z3 with timeout).
    FormalVerification,
    /// Regime-conditioned check on pre-aggregated state.
    RegimeCheck,
}

/// Computational budget for a bounded review action.
#[derive(Debug, Clone)]
pub struct ReviewBudget {
    /// Maximum wall-clock time for this evaluation.
    pub max_time: Duration,
    /// Maximum memory (bytes) for this evaluation.
    pub max_memory: usize,
    /// Maximum provenance chain depth to traverse.
    pub max_depth: u32,
    /// Whether approximate algorithms are permitted.
    pub allow_approximation: bool,
    /// Fallback verdict if budget is exceeded.
    pub fallback: ConstitutionalVerdict,
}

impl ReviewAction {
    /// Default budget for this review action.
    pub fn default_budget(&self) -> ReviewBudget {
        match self {
            Self::IhsanProjection => ReviewBudget {
                max_time: Duration::from_millis(100),
                max_memory: 4 * 1024 * 1024, // 4 MB
                max_depth: 8,
                allow_approximation: true,
                fallback: ConstitutionalVerdict::Review,
            },
            Self::AdlDeltaApprox => ReviewBudget {
                max_time: Duration::from_millis(200),
                max_memory: 8 * 1024 * 1024, // 8 MB
                max_depth: 16,
                allow_approximation: true,
                fallback: ConstitutionalVerdict::Reject, // high-risk: fail closed
            },
            Self::ProvenanceTraversal => ReviewBudget {
                max_time: Duration::from_millis(500),
                max_memory: 16 * 1024 * 1024, // 16 MB
                max_depth: 64,
                allow_approximation: false,
                fallback: ConstitutionalVerdict::Review,
            },
            Self::FormalVerification => ReviewBudget {
                max_time: Duration::from_secs(5),
                max_memory: 64 * 1024 * 1024, // 64 MB
                max_depth: 32,
                allow_approximation: false,
                fallback: ConstitutionalVerdict::Reject, // high-risk: fail closed
            },
            Self::RegimeCheck => ReviewBudget {
                max_time: Duration::from_millis(50),
                max_memory: 2 * 1024 * 1024, // 2 MB
                max_depth: 4,
                allow_approximation: true,
                fallback: ConstitutionalVerdict::Permit,
            },
        }
    }

    /// Which constitutional layer owns this action.
    pub fn layer(&self) -> ConstitutionalLayer {
        ConstitutionalLayer::BoundedReview
    }
}

// ─────────────────────────────────────────────────────────
// Judiciary / Advisory Actions (Layer 3)
// ─────────────────────────────────────────────────────────

/// Actions owned by the constitutional judiciary (Layer 3).
/// These produce evidence, scores, and explanations — never blocking verdicts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum JudiciaryAction {
    /// Full 8-dimensional contextual Ihsan evaluation.
    ContextualIhsan,
    /// Whole-graph distributional justice analysis.
    GraphAdl,
    /// Long-horizon Amanah (trustworthiness over time).
    LongHorizonAmanah,
    /// Guardian Council deliberation.
    GuardianDeliberation,
    /// Counterfactual simulation (what-if analysis).
    CounterfactualSim,
    /// Market-regime contextualization.
    MarketRegime,
    /// Ethical interpretation beyond kernel grammar.
    EthicalInterpretation,
    /// Appeal of a kernel or review verdict.
    VerdictAppeal,
}

impl JudiciaryAction {
    /// Which constitutional layer owns this action.
    pub fn layer(&self) -> ConstitutionalLayer {
        ConstitutionalLayer::Judiciary
    }

    /// Judiciary actions always produce SCORE_ONLY verdicts.
    /// They inform but do not block.
    pub fn verdict_type(&self) -> ConstitutionalVerdict {
        ConstitutionalVerdict::ScoreOnly
    }
}

// ─────────────────────────────────────────────────────────
// Verdict Receipt Extension
// ─────────────────────────────────────────────────────────

/// Extended verdict metadata for constitutional receipts.
/// Records not just yes/no, but what was decided where.
#[derive(Debug, Clone)]
pub struct VerdictReceipt {
    /// The verdict itself.
    pub verdict: ConstitutionalVerdict,
    /// Which layer produced this verdict.
    pub layer: ConstitutionalLayer,
    /// Was an approximation used?
    pub approximated: bool,
    /// Was the budget exceeded? (Layer 2 only)
    pub budget_exceeded: bool,
    /// Was this escalated from a lower layer?
    pub escalated_from: Option<ConstitutionalLayer>,
    /// Human-readable reason code.
    pub reason: &'static str,
}

// ─────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kernel_actions_are_hard_law() {
        for action in [
            KernelAction::SignatureVerify,
            KernelAction::SchemaValidate,
            KernelAction::TransitionCheck,
            KernelAction::ThresholdCompare,
            KernelAction::ConstitutionalVeto,
            KernelAction::ResourceCap,
            KernelAction::ChainIntegrity,
            KernelAction::ReceiptCompleteness,
            KernelAction::SeedMint,
            KernelAction::BloomAccrue,
        ] {
            assert_eq!(action.layer(), ConstitutionalLayer::HardLaw);
        }
    }

    #[test]
    fn kernel_budgets_are_bounded() {
        // All kernel actions must complete within 100ms
        let max_kernel = Duration::from_millis(100);
        for action in [
            KernelAction::SignatureVerify,
            KernelAction::SchemaValidate,
            KernelAction::TransitionCheck,
            KernelAction::ThresholdCompare,
            KernelAction::ConstitutionalVeto,
            KernelAction::ResourceCap,
            KernelAction::ChainIntegrity,
            KernelAction::ReceiptCompleteness,
            KernelAction::SeedMint,
            KernelAction::BloomAccrue,
        ] {
            assert!(
                action.max_budget() <= max_kernel,
                "{:?} budget {:?} exceeds kernel max {:?}",
                action,
                action.max_budget(),
                max_kernel
            );
        }
    }

    #[test]
    fn constitutional_vetoes_are_frozen() {
        assert!(KernelAction::ConstitutionalVeto.is_frozen());
        assert!(KernelAction::ChainIntegrity.is_frozen());
        assert!(!KernelAction::SchemaValidate.is_frozen());
    }

    #[test]
    fn review_fallbacks_are_safe() {
        // High-risk review actions must fail closed
        assert_eq!(
            ReviewAction::FormalVerification.default_budget().fallback,
            ConstitutionalVerdict::Reject
        );
        assert_eq!(
            ReviewAction::AdlDeltaApprox.default_budget().fallback,
            ConstitutionalVerdict::Reject
        );
    }

    #[test]
    fn judiciary_never_blocks() {
        for action in [
            JudiciaryAction::ContextualIhsan,
            JudiciaryAction::GraphAdl,
            JudiciaryAction::GuardianDeliberation,
            JudiciaryAction::VerdictAppeal,
        ] {
            assert_eq!(action.verdict_type(), ConstitutionalVerdict::ScoreOnly);
            assert_eq!(action.layer(), ConstitutionalLayer::Judiciary);
        }
    }

    #[test]
    fn verdict_semantics() {
        assert!(ConstitutionalVerdict::Permit.is_permissive());
        assert!(ConstitutionalVerdict::ScoreOnly.is_permissive());
        assert!(ConstitutionalVerdict::Reject.is_blocking());
        assert!(ConstitutionalVerdict::Review.requires_escalation());
        assert!(!ConstitutionalVerdict::Permit.is_blocking());
        assert!(!ConstitutionalVerdict::Reject.requires_escalation());
    }
}
