//! BIZRA Admissibility v1 Freeze — §7 GateVerdict + RejectedClaim + Five-Gate Chain
//!
//! بسم الله الرحمن الرحيم
//!
//! File: crates/bizra-kernel/src/admissibility/freeze_v1.rs
//! Authority: Manifest v0.2 Canon, §5 (Authority Model), §7 (Canonical Contracts)
//! Build Step: 3 of 8 (§17)
//! Truth Target: PROVEN
//! Depends on: Step 2 (Receipt v1 freeze — receipt_freeze_v1.rs)
//!
//! This file delivers:
//!
//!   1. GateVerdict — §7 frozen contract. Four verdicts only: PERMIT, REJECT,
//!      REVIEW, SCORE_ONLY. No additional verdicts without constitutional amendment.
//!
//!   2. RejectedClaim — §7 frozen contract. Denied claim with reason and
//!      remediation path. Emitted alongside REJECT verdicts.
//!
//!   3. AdmissibilityClaim — the input structure representing a claim to be evaluated.
//!
//!   4. InvariantGate trait — each of the five invariants (§3) is a discrete gate
//!      that evaluates a claim and produces a GateVerdict.
//!
//!   5. AdmissibilityChain — the fail-closed pipeline that runs a claim through
//!      all five gates in order. Any REJECT stops the pipeline immediately.
//!
//! Plane: Kernel (Layer 1 — DEFINES LAW)
//! Authority rule: bounded and decidable. Every evaluation terminates in finite
//! time with a definitive verdict. No unbounded loops. No external calls.
//!
//! §5: "Hot-path law must remain bounded and decidable — meaning every
//! admissibility evaluation must terminate in finite time with a definitive verdict."

use crate::canonical_hasher::blake3_domain;
use crate::receipts::{
    Blake3Hash, ByteReader, DecodeError, ReceiptKind, ReceiptPayload, ReceiptPayloadDecode,
};

// ════════════════════════════════════════════════════════════
// GateVerdict — §7 Frozen Contract
// ════════════════════════════════════════════════════════════

/// The four canonical verdicts per Manifest §5.
/// No additional verdict types may be introduced without constitutional amendment.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Verdict {
    /// Claim is admissible. Proceeds to execution (S5).
    Permit = 0x01,
    /// Claim is denied. RejectedClaim emitted. Does NOT proceed.
    Reject = 0x02,
    /// Claim requires human or higher-layer review. Paused.
    Review = 0x03,
    /// Non-binding assessment. Does not affect claim lifecycle.
    ScoreOnly = 0x04,
}

impl Verdict {
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0x01 => Some(Self::Permit),
            0x02 => Some(Self::Reject),
            0x03 => Some(Self::Review),
            0x04 => Some(Self::ScoreOnly),
            _ => None,
        }
    }

    pub fn is_terminal(&self) -> bool {
        matches!(self, Verdict::Permit | Verdict::Reject)
    }

    pub fn allows_execution(&self) -> bool {
        matches!(self, Verdict::Permit)
    }
}

/// The canonical admissibility evaluation result per §7 Table 7-1.
///
/// FROZEN after Step 3 completes.
///
/// §7 specifies:
///   "verdict, reason, scorer_id, chain_ref, timestamp"
///   Plane: Kernel → All
///   Lifetime: Immutable once issued
#[derive(Debug, Clone)]
pub struct GateVerdict {
    // ── §7 required fields ──
    /// The verdict: PERMIT, REJECT, REVIEW, or SCORE_ONLY.
    pub verdict: Verdict,

    /// Human-readable reason for the verdict.
    /// For PERMIT: which invariants were satisfied.
    /// For REJECT: which invariant was violated and why.
    pub reason: String,

    /// Identifier of the gate that produced this verdict.
    /// One of: "IHSAN_FLOOR", "ZANN_ZERO", "RIBA_ZERO",
    /// "CLAIM_MUST_BIND", "NO_SHADOW_STATE", or "CHAIN" for aggregate.
    pub scorer_id: String,

    /// Reference to the claim being evaluated.
    pub chain_ref: Blake3Hash,

    /// Monotonic timestamp (nanoseconds).
    pub timestamp_ns: u64,

    // ── Operational extensions ──
    /// Which invariant (if any) caused the verdict.
    pub invariant: Option<Invariant>,

    /// Numerical score (0.0-1.0) for SCORE_ONLY verdicts or gate metrics.
    pub score: Option<f64>,
}

impl ReceiptPayload for GateVerdict {
    fn kind(&self) -> ReceiptKind {
        ReceiptKind::GovernanceDecision
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(256);
        buf.push(self.verdict as u8);
        // reason: length-prefixed
        buf.extend_from_slice(&(self.reason.len() as u32).to_le_bytes());
        buf.extend_from_slice(self.reason.as_bytes());
        // scorer_id: length-prefixed
        buf.extend_from_slice(&(self.scorer_id.len() as u32).to_le_bytes());
        buf.extend_from_slice(self.scorer_id.as_bytes());
        // chain_ref
        buf.extend_from_slice(&self.chain_ref);
        // timestamp
        buf.extend_from_slice(&self.timestamp_ns.to_le_bytes());
        // invariant: option discriminant + value
        match &self.invariant {
            None => buf.push(0x00),
            Some(inv) => {
                buf.push(0x01);
                buf.push(*inv as u8);
            }
        }
        // score: option discriminant + value
        match self.score {
            None => buf.push(0x00),
            Some(s) => {
                buf.push(0x01);
                buf.extend_from_slice(&s.to_le_bytes());
            }
        }
        buf
    }

    fn hash(&self) -> Blake3Hash {
        blake3_domain("bizra-gate-verdict-v1", &self.canonical_bytes())
    }
}

impl ReceiptPayloadDecode for GateVerdict {
    fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, DecodeError> {
        let mut r = ByteReader::new(bytes);

        let verdict_byte = r.read_u8()?;
        let verdict = Verdict::from_byte(verdict_byte).ok_or(DecodeError::UnknownDiscriminant {
            field: "GateVerdict.verdict",
            byte: verdict_byte,
        })?;

        let reason_bytes = r.read_length_prefixed()?;
        let reason = std::str::from_utf8(reason_bytes)
            .map_err(|e| DecodeError::Utf8(e.to_string()))?
            .to_string();

        let scorer_bytes = r.read_length_prefixed()?;
        let scorer_id = std::str::from_utf8(scorer_bytes)
            .map_err(|e| DecodeError::Utf8(e.to_string()))?
            .to_string();

        let chain_ref = r.read_hash()?;
        let timestamp_ns = r.read_u64()?;

        let inv_disc = r.read_u8()?;
        let invariant = if inv_disc == 0x01 {
            let inv_byte = r.read_u8()?;
            Some(
                Invariant::from_byte(inv_byte).ok_or(DecodeError::UnknownDiscriminant {
                    field: "GateVerdict.invariant",
                    byte: inv_byte,
                })?,
            )
        } else {
            None
        };

        let score_disc = r.read_u8()?;
        let score = if score_disc == 0x01 {
            Some(r.read_f64()?)
        } else {
            None
        };

        Ok(GateVerdict {
            verdict,
            reason,
            scorer_id,
            chain_ref,
            timestamp_ns,
            invariant,
            score,
        })
    }
}

// ════════════════════════════════════════════════════════════
// RejectedClaim — §7 Frozen Contract
// ════════════════════════════════════════════════════════════

/// Emitted alongside a REJECT verdict per §7 Table 7-1.
///
/// §7 specifies:
///   "claim_ref, reject_reason, remediation_path, escalation_option"
///   Plane: Kernel → Face
///   Lifetime: Until remediated
#[derive(Debug, Clone)]
pub struct RejectedClaim {
    /// Reference to the rejected claim.
    pub claim_ref: Blake3Hash,

    /// Which invariant caused the rejection.
    pub invariant: Invariant,

    /// Human-readable rejection reason.
    pub reject_reason: String,

    /// Suggested path to remediate the claim so it can be re-submitted.
    pub remediation_path: String,

    /// Whether the claim can be escalated to REVIEW instead.
    pub escalation_allowed: bool,
}

// ════════════════════════════════════════════════════════════
// The Five Invariants — §3 Core Pillars
// ════════════════════════════════════════════════════════════

/// The five core invariants from Manifest §3.
/// These are the pillars. Each is a discrete gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum Invariant {
    IhsanFloor = 0x01,    // P1: excellence ≥ 0.95
    ZannZero = 0x02,      // P2: no claim without evidence
    RibaZero = 0x03,      // P3: no extractive economic pattern
    ClaimMustBind = 0x04, // P4: no claim without receipt/artifact
    NoShadowState = 0x05, // P5: no UI simulates truth independently
}

impl Invariant {
    pub fn from_byte(b: u8) -> Option<Self> {
        match b {
            0x01 => Some(Self::IhsanFloor),
            0x02 => Some(Self::ZannZero),
            0x03 => Some(Self::RibaZero),
            0x04 => Some(Self::ClaimMustBind),
            0x05 => Some(Self::NoShadowState),
            _ => None,
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            Self::IhsanFloor => "IHSAN_FLOOR",
            Self::ZannZero => "ZANN_ZERO",
            Self::RibaZero => "RIBA_ZERO",
            Self::ClaimMustBind => "CLAIM_MUST_BIND",
            Self::NoShadowState => "NO_SHADOW_STATE",
        }
    }
}

// All five, in evaluation order (fail-closed: first REJECT stops chain)
pub const INVARIANT_ORDER: [Invariant; 5] = [
    Invariant::ZannZero, // Check evidence first — no point evaluating rest if unbound
    Invariant::ClaimMustBind, // Then check artifact binding
    Invariant::RibaZero, // Then economic pattern
    Invariant::NoShadowState, // Then shadow state
    Invariant::IhsanFloor, // Ihsan last — it's the quality floor after structural checks pass
];

// ════════════════════════════════════════════════════════════
// AdmissibilityClaim — input to the gate chain
// ════════════════════════════════════════════════════════════

/// A claim submitted for admissibility evaluation at Stage S4.
///
/// The claim carries all the metadata the gate chain needs to
/// evaluate against the five invariants.
#[derive(Debug, Clone)]
pub struct AdmissibilityClaim {
    /// Hash identifying this claim (from MissionEnvelope).
    pub claim_id: Blake3Hash,

    /// Does this claim carry binding evidence?
    /// Gate: ZANN_ZERO checks this.
    pub has_evidence: bool,

    /// Hash of the evidence, if present.
    /// Gate: CLAIM_MUST_BIND checks this is non-zero when has_evidence is true.
    pub evidence_hash: Option<Blake3Hash>,

    /// Does this claim involve an economic pattern?
    /// Gate: RIBA_ZERO checks if the pattern is extractive.
    pub economic_pattern: Option<EconomicPattern>,

    /// Does this claim mutate operator-visible state?
    /// Gate: NO_SHADOW_STATE checks if it derives from canonical runtime.
    pub state_mutation: Option<StateMutation>,

    /// Quality score for this claim (0.0-1.0).
    /// Gate: IHSAN_FLOOR checks this is ≥ 0.95.
    pub quality_score: f64,

    /// Monotonic timestamp for the claim.
    pub timestamp_ns: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum EconomicPattern {
    /// No economic activity.
    None,
    /// Value exchange between peers (halal).
    PeerExchange,
    /// Profit-loss sharing (mudarabah — halal).
    ProfitSharing,
    /// Fixed-return lending (riba — HARAM, will be rejected).
    FixedReturnLending,
    /// Hidden fee extraction (riba — HARAM, will be rejected).
    HiddenFeeExtraction,
    /// Asymmetric information exploitation (riba — HARAM).
    AsymmetricExploitation,
}

impl EconomicPattern {
    pub fn is_extractive(&self) -> bool {
        matches!(
            self,
            EconomicPattern::FixedReturnLending
                | EconomicPattern::HiddenFeeExtraction
                | EconomicPattern::AsymmetricExploitation
        )
    }
}

#[derive(Debug, Clone)]
pub struct StateMutation {
    /// Does this mutation derive from the canonical runtime?
    pub derives_from_canonical: bool,
    /// Is this a Face-plane-only mutation (prohibited)?
    pub face_only: bool,
}

// ════════════════════════════════════════════════════════════
// InvariantGate trait — each invariant is a discrete evaluator
// ════════════════════════════════════════════════════════════

/// A single invariant gate. Evaluates one claim against one invariant.
///
/// §5: "Hot-path law must remain bounded and decidable."
/// Every gate implementation MUST terminate in O(1) or O(n) where n
/// is the size of the claim's metadata. No external calls. No loops
/// that depend on runtime state. Pure evaluation.
pub trait InvariantGate: Send + Sync {
    /// Which invariant this gate enforces.
    fn invariant(&self) -> Invariant;

    /// Evaluate the claim. Returns PERMIT or REJECT.
    /// REVIEW is only returned if the gate cannot determine the verdict
    /// with the available metadata (escalation to human).
    fn evaluate(&self, claim: &AdmissibilityClaim) -> GateVerdict;
}

// ════════════════════════════════════════════════════════════
// Five Gate Implementations
// ════════════════════════════════════════════════════════════

/// P2: ZANN_ZERO — No claim without binding evidence.
pub struct ZannZeroGate;

impl InvariantGate for ZannZeroGate {
    fn invariant(&self) -> Invariant {
        Invariant::ZannZero
    }

    fn evaluate(&self, claim: &AdmissibilityClaim) -> GateVerdict {
        if claim.has_evidence {
            GateVerdict {
                verdict: Verdict::Permit,
                reason: "Claim carries evidence binding".into(),
                scorer_id: "ZANN_ZERO".into(),
                chain_ref: claim.claim_id,
                timestamp_ns: claim.timestamp_ns,
                invariant: Some(Invariant::ZannZero),
                score: Some(1.0),
            }
        } else {
            GateVerdict {
                verdict: Verdict::Reject,
                reason: "ZANN_ZERO violation: claim promoted without binding evidence".into(),
                scorer_id: "ZANN_ZERO".into(),
                chain_ref: claim.claim_id,
                timestamp_ns: claim.timestamp_ns,
                invariant: Some(Invariant::ZannZero),
                score: Some(0.0),
            }
        }
    }
}

/// P4: CLAIM_MUST_BIND — No claim canonical without evidence hash.
pub struct ClaimMustBindGate;

impl InvariantGate for ClaimMustBindGate {
    fn invariant(&self) -> Invariant {
        Invariant::ClaimMustBind
    }

    fn evaluate(&self, claim: &AdmissibilityClaim) -> GateVerdict {
        let bound = match &claim.evidence_hash {
            Some(h) => *h != [0u8; 32], // non-zero hash = bound
            None => false,
        };

        if bound {
            GateVerdict {
                verdict: Verdict::Permit,
                reason: "Claim bound to evidence artifact".into(),
                scorer_id: "CLAIM_MUST_BIND".into(),
                chain_ref: claim.claim_id,
                timestamp_ns: claim.timestamp_ns,
                invariant: Some(Invariant::ClaimMustBind),
                score: Some(1.0),
            }
        } else {
            GateVerdict {
                verdict: Verdict::Reject,
                reason: "CLAIM_MUST_BIND violation: no evidence hash or zero hash".into(),
                scorer_id: "CLAIM_MUST_BIND".into(),
                chain_ref: claim.claim_id,
                timestamp_ns: claim.timestamp_ns,
                invariant: Some(Invariant::ClaimMustBind),
                score: Some(0.0),
            }
        }
    }
}

/// P3: RIBA_ZERO — No extractive economic pattern.
pub struct RibaZeroGate;

impl InvariantGate for RibaZeroGate {
    fn invariant(&self) -> Invariant {
        Invariant::RibaZero
    }

    fn evaluate(&self, claim: &AdmissibilityClaim) -> GateVerdict {
        match &claim.economic_pattern {
            None | Some(EconomicPattern::None) => GateVerdict {
                verdict: Verdict::Permit,
                reason: "No economic pattern present".into(),
                scorer_id: "RIBA_ZERO".into(),
                chain_ref: claim.claim_id,
                timestamp_ns: claim.timestamp_ns,
                invariant: Some(Invariant::RibaZero),
                score: Some(1.0),
            },
            Some(pattern) if !pattern.is_extractive() => GateVerdict {
                verdict: Verdict::Permit,
                reason: format!("Economic pattern {:?} is non-extractive", pattern),
                scorer_id: "RIBA_ZERO".into(),
                chain_ref: claim.claim_id,
                timestamp_ns: claim.timestamp_ns,
                invariant: Some(Invariant::RibaZero),
                score: Some(1.0),
            },
            Some(pattern) => GateVerdict {
                verdict: Verdict::Reject,
                reason: format!(
                    "RIBA_ZERO violation: extractive pattern {:?} detected",
                    pattern
                ),
                scorer_id: "RIBA_ZERO".into(),
                chain_ref: claim.claim_id,
                timestamp_ns: claim.timestamp_ns,
                invariant: Some(Invariant::RibaZero),
                score: Some(0.0),
            },
        }
    }
}

/// P5: NO_SHADOW_STATE — No UI simulates truth independently.
pub struct NoShadowStateGate;

impl InvariantGate for NoShadowStateGate {
    fn invariant(&self) -> Invariant {
        Invariant::NoShadowState
    }

    fn evaluate(&self, claim: &AdmissibilityClaim) -> GateVerdict {
        match &claim.state_mutation {
            None => GateVerdict {
                verdict: Verdict::Permit,
                reason: "No state mutation in this claim".into(),
                scorer_id: "NO_SHADOW_STATE".into(),
                chain_ref: claim.claim_id,
                timestamp_ns: claim.timestamp_ns,
                invariant: Some(Invariant::NoShadowState),
                score: Some(1.0),
            },
            Some(sm) if sm.derives_from_canonical && !sm.face_only => GateVerdict {
                verdict: Verdict::Permit,
                reason: "State mutation derives from canonical runtime".into(),
                scorer_id: "NO_SHADOW_STATE".into(),
                chain_ref: claim.claim_id,
                timestamp_ns: claim.timestamp_ns,
                invariant: Some(Invariant::NoShadowState),
                score: Some(1.0),
            },
            Some(sm) => {
                let reason = if sm.face_only {
                    "NO_SHADOW_STATE violation: Face-only state mutation (UI simulating truth)"
                } else {
                    "NO_SHADOW_STATE violation: mutation does not derive from canonical runtime"
                };
                GateVerdict {
                    verdict: Verdict::Reject,
                    reason: reason.into(),
                    scorer_id: "NO_SHADOW_STATE".into(),
                    chain_ref: claim.claim_id,
                    timestamp_ns: claim.timestamp_ns,
                    invariant: Some(Invariant::NoShadowState),
                    score: Some(0.0),
                }
            }
        }
    }
}

/// P1: IHSAN_FLOOR — Excellence ≥ 0.95 in all operator-visible paths.
pub struct IhsanFloorGate {
    /// The floor. Hardcoded 0.95 per commit 0115016b.
    /// Exists as a field for testability, but production MUST use 0.95.
    pub floor: f64,
}

impl IhsanFloorGate {
    /// Canonical production instance. Floor = 0.95. Non-negotiable.
    pub fn canonical() -> Self {
        IhsanFloorGate { floor: 0.95 }
    }
}

impl InvariantGate for IhsanFloorGate {
    fn invariant(&self) -> Invariant {
        Invariant::IhsanFloor
    }

    fn evaluate(&self, claim: &AdmissibilityClaim) -> GateVerdict {
        if claim.quality_score >= self.floor {
            GateVerdict {
                verdict: Verdict::Permit,
                reason: format!(
                    "Ihsan score {:.4} ≥ floor {:.4}",
                    claim.quality_score, self.floor
                ),
                scorer_id: "IHSAN_FLOOR".into(),
                chain_ref: claim.claim_id,
                timestamp_ns: claim.timestamp_ns,
                invariant: Some(Invariant::IhsanFloor),
                score: Some(claim.quality_score),
            }
        } else {
            GateVerdict {
                verdict: Verdict::Reject,
                reason: format!(
                    "IHSAN_FLOOR violation: score {:.4} below floor {:.4}",
                    claim.quality_score, self.floor
                ),
                scorer_id: "IHSAN_FLOOR".into(),
                chain_ref: claim.claim_id,
                timestamp_ns: claim.timestamp_ns,
                invariant: Some(Invariant::IhsanFloor),
                score: Some(claim.quality_score),
            }
        }
    }
}

// ════════════════════════════════════════════════════════════
// AdmissibilityChain — the fail-closed pipeline
// ════════════════════════════════════════════════════════════

/// Result of running the full admissibility chain.
#[derive(Debug, Clone)]
pub struct AdmissibilityResult {
    /// The aggregate verdict. PERMIT only if ALL gates PERMIT.
    pub verdict: Verdict,

    /// All individual gate verdicts, in evaluation order.
    pub gate_verdicts: Vec<GateVerdict>,

    /// If REJECT: the RejectedClaim with remediation path.
    pub rejected: Option<RejectedClaim>,
}

/// The five-gate admissibility chain.
///
/// §6 Stage S4: "Gate chain evaluates claim."
/// §3: "Any violation triggers an immediate REJECT verdict."
///
/// Fail-closed: gates run in INVARIANT_ORDER. First REJECT stops the chain.
/// All gates must PERMIT for the aggregate verdict to be PERMIT.
pub struct AdmissibilityChain {
    gates: Vec<Box<dyn InvariantGate>>,
}

impl AdmissibilityChain {
    /// Build the canonical five-gate chain with production thresholds.
    pub fn canonical() -> Self {
        AdmissibilityChain {
            gates: vec![
                Box::new(ZannZeroGate),
                Box::new(ClaimMustBindGate),
                Box::new(RibaZeroGate),
                Box::new(NoShadowStateGate),
                Box::new(IhsanFloorGate::canonical()),
            ],
        }
    }

    /// Evaluate a claim through the entire chain.
    ///
    /// §5: "bounded and decidable." This method runs in O(5) — exactly
    /// five gate evaluations at most. No loops. No external calls.
    /// Terminates with a definitive verdict.
    pub fn evaluate(&self, claim: &AdmissibilityClaim) -> AdmissibilityResult {
        let mut verdicts = Vec::with_capacity(self.gates.len());

        for gate in &self.gates {
            let gv = gate.evaluate(claim);
            let is_reject = gv.verdict == Verdict::Reject;
            verdicts.push(gv);

            if is_reject {
                // Fail-closed: first REJECT stops the chain.
                let last = verdicts.last().unwrap();
                let invariant = last.invariant.unwrap_or(Invariant::ZannZero);

                let rejected = RejectedClaim {
                    claim_ref: claim.claim_id,
                    invariant,
                    reject_reason: last.reason.clone(),
                    remediation_path: Self::remediation_for(invariant),
                    escalation_allowed: matches!(
                        invariant,
                        Invariant::IhsanFloor | Invariant::NoShadowState
                    ),
                };

                return AdmissibilityResult {
                    verdict: Verdict::Reject,
                    gate_verdicts: verdicts,
                    rejected: Some(rejected),
                };
            }
        }

        // All gates PERMIT
        AdmissibilityResult {
            verdict: Verdict::Permit,
            gate_verdicts: verdicts,
            rejected: None,
        }
    }

    /// Suggested remediation path for each invariant violation.
    fn remediation_for(inv: Invariant) -> String {
        match inv {
            Invariant::ZannZero => "Attach binding evidence to the claim before resubmitting. \
                 Evidence must be hashable and verifiable."
                .into(),
            Invariant::ClaimMustBind => "Provide a non-zero evidence_hash linking the claim to a \
                 ReceiptArtifact or other canonical proof artifact."
                .into(),
            Invariant::RibaZero => "Remove the extractive economic pattern from the claim. \
                 Use PeerExchange or ProfitSharing (mudarabah) instead of \
                 FixedReturnLending or HiddenFeeExtraction."
                .into(),
            Invariant::NoShadowState => "Ensure the state mutation derives from canonical runtime \
                 truth (Proof plane), not from Face-only computation. \
                 The UI must reveal, never simulate."
                .into(),
            Invariant::IhsanFloor => "Improve claim quality score to ≥ 0.95 before resubmitting. \
                 Add tests, documentation, or constitutional alignment evidence \
                 to raise the Ihsan score."
                .into(),
        }
    }
}

// ════════════════════════════════════════════════════════════
// Tests — proving each invariant gate fires correctly
// ════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn base_claim() -> AdmissibilityClaim {
        AdmissibilityClaim {
            claim_id: [1u8; 32],
            has_evidence: true,
            evidence_hash: Some([2u8; 32]),
            economic_pattern: None,
            state_mutation: None,
            quality_score: 0.97, // above 0.95 floor
            timestamp_ns: 1000,
        }
    }

    // ── Test 1: Clean claim passes all five gates ──

    #[test]
    fn test_clean_claim_permits() {
        let chain = AdmissibilityChain::canonical();
        let claim = base_claim();
        let result = chain.evaluate(&claim);

        assert_eq!(result.verdict, Verdict::Permit);
        assert_eq!(result.gate_verdicts.len(), 5, "All 5 gates must evaluate");
        assert!(result.rejected.is_none());

        for gv in &result.gate_verdicts {
            assert_eq!(
                gv.verdict,
                Verdict::Permit,
                "Gate {} should PERMIT clean claim",
                gv.scorer_id
            );
        }
    }

    // ── Test 2: ZANN_ZERO rejects claim without evidence ──

    #[test]
    fn test_zann_zero_rejects_no_evidence() {
        let chain = AdmissibilityChain::canonical();
        let mut claim = base_claim();
        claim.has_evidence = false;

        let result = chain.evaluate(&claim);

        assert_eq!(result.verdict, Verdict::Reject);
        assert_eq!(
            result.gate_verdicts.len(),
            1,
            "Chain should stop at first REJECT (ZANN_ZERO is gate 1)"
        );
        assert_eq!(
            result.rejected.as_ref().unwrap().invariant,
            Invariant::ZannZero
        );
    }

    // ── Test 3: CLAIM_MUST_BIND rejects zero evidence hash ──

    #[test]
    fn test_claim_must_bind_rejects_zero_hash() {
        let chain = AdmissibilityChain::canonical();
        let mut claim = base_claim();
        claim.evidence_hash = Some([0u8; 32]); // zero hash = unbound

        let result = chain.evaluate(&claim);

        assert_eq!(result.verdict, Verdict::Reject);
        assert_eq!(
            result.rejected.as_ref().unwrap().invariant,
            Invariant::ClaimMustBind
        );
    }

    // ── Test 4: RIBA_ZERO rejects fixed-return lending ──

    #[test]
    fn test_riba_zero_rejects_fixed_return() {
        let chain = AdmissibilityChain::canonical();
        let mut claim = base_claim();
        claim.economic_pattern = Some(EconomicPattern::FixedReturnLending);

        let result = chain.evaluate(&claim);

        assert_eq!(result.verdict, Verdict::Reject);
        assert_eq!(
            result.rejected.as_ref().unwrap().invariant,
            Invariant::RibaZero
        );
    }

    // ── Test 5: RIBA_ZERO permits profit-sharing (mudarabah) ──

    #[test]
    fn test_riba_zero_permits_mudarabah() {
        let chain = AdmissibilityChain::canonical();
        let mut claim = base_claim();
        claim.economic_pattern = Some(EconomicPattern::ProfitSharing);

        let result = chain.evaluate(&claim);
        assert_eq!(result.verdict, Verdict::Permit);
    }

    // ── Test 6: NO_SHADOW_STATE rejects face-only mutation ──

    #[test]
    fn test_no_shadow_state_rejects_face_only() {
        let chain = AdmissibilityChain::canonical();
        let mut claim = base_claim();
        claim.state_mutation = Some(StateMutation {
            derives_from_canonical: false,
            face_only: true,
        });

        let result = chain.evaluate(&claim);

        assert_eq!(result.verdict, Verdict::Reject);
        assert_eq!(
            result.rejected.as_ref().unwrap().invariant,
            Invariant::NoShadowState
        );
    }

    // ── Test 7: NO_SHADOW_STATE permits canonical mutation ──

    #[test]
    fn test_no_shadow_state_permits_canonical() {
        let chain = AdmissibilityChain::canonical();
        let mut claim = base_claim();
        claim.state_mutation = Some(StateMutation {
            derives_from_canonical: true,
            face_only: false,
        });

        let result = chain.evaluate(&claim);
        assert_eq!(result.verdict, Verdict::Permit);
    }

    // ── Test 8: IHSAN_FLOOR rejects below 0.95 ──

    #[test]
    fn test_ihsan_floor_rejects_below_threshold() {
        let chain = AdmissibilityChain::canonical();
        let mut claim = base_claim();
        claim.quality_score = 0.94; // below 0.95

        let result = chain.evaluate(&claim);

        assert_eq!(result.verdict, Verdict::Reject);
        assert_eq!(
            result.rejected.as_ref().unwrap().invariant,
            Invariant::IhsanFloor
        );
        assert!(
            result.rejected.as_ref().unwrap().escalation_allowed,
            "Ihsan rejection should allow escalation to REVIEW"
        );
    }

    // ── Test 9: IHSAN_FLOOR permits exactly 0.95 ──

    #[test]
    fn test_ihsan_floor_permits_exact_threshold() {
        let chain = AdmissibilityChain::canonical();
        let mut claim = base_claim();
        claim.quality_score = 0.95; // exactly at floor

        let result = chain.evaluate(&claim);
        assert_eq!(result.verdict, Verdict::Permit);
    }

    // ── Test 10: Fail-closed — chain stops at first REJECT ──

    #[test]
    fn test_fail_closed_stops_at_first_reject() {
        let chain = AdmissibilityChain::canonical();

        // Claim that fails ZANN_ZERO (gate 1) AND has low Ihsan (gate 5)
        let mut claim = base_claim();
        claim.has_evidence = false; // fails gate 1
        claim.quality_score = 0.50; // would fail gate 5

        let result = chain.evaluate(&claim);

        assert_eq!(result.verdict, Verdict::Reject);
        // Only 1 gate should have evaluated (ZANN_ZERO stops the chain)
        assert_eq!(
            result.gate_verdicts.len(),
            1,
            "Fail-closed: must stop at first REJECT, not evaluate remaining gates"
        );
        assert_eq!(
            result.rejected.as_ref().unwrap().invariant,
            Invariant::ZannZero,
            "First gate (ZANN_ZERO) should be the rejector, not IHSAN_FLOOR"
        );
    }

    // ── Test 11: GateVerdict round-trip ──

    #[test]
    fn test_gate_verdict_roundtrip() {
        let gv = GateVerdict {
            verdict: Verdict::Reject,
            reason: "RIBA_ZERO: extractive pattern detected".into(),
            scorer_id: "RIBA_ZERO".into(),
            chain_ref: [42u8; 32],
            timestamp_ns: 999_999,
            invariant: Some(Invariant::RibaZero),
            score: Some(0.0),
        };

        let bytes = gv.canonical_bytes();
        let decoded = GateVerdict::from_canonical_bytes(&bytes).unwrap();

        assert_eq!(decoded.verdict, Verdict::Reject);
        assert_eq!(decoded.reason, "RIBA_ZERO: extractive pattern detected");
        assert_eq!(decoded.scorer_id, "RIBA_ZERO");
        assert_eq!(decoded.chain_ref, [42u8; 32]);
        assert_eq!(decoded.timestamp_ns, 999_999);
        assert_eq!(decoded.invariant, Some(Invariant::RibaZero));
        assert_eq!(decoded.score, Some(0.0));
    }

    // ── Test 12: Verdict enum completeness ──

    #[test]
    fn test_verdict_properties() {
        assert!(Verdict::Permit.allows_execution());
        assert!(!Verdict::Reject.allows_execution());
        assert!(!Verdict::Review.allows_execution());
        assert!(!Verdict::ScoreOnly.allows_execution());

        assert!(Verdict::Permit.is_terminal());
        assert!(Verdict::Reject.is_terminal());
        assert!(!Verdict::Review.is_terminal());
        assert!(!Verdict::ScoreOnly.is_terminal());
    }
}
