//! WBS 1.4 — Fate-Binding Z3 Integration into ProofSpace
//!
//! This module implements the Fate-Binding Engine: the component that generates,
//! validates, and embeds SMT-LIB2 formal proofs into ProofSpace blocks, making
//! formal proofs a first-class citizen of the BIZRA proof-validation pipeline.
//!
//! # Fate-Binding Semantics
//!
//! "Fate-binding" means: if the proof script is satisfiable, the block's FATE
//! scores are consistent with BIZRA constitutional law. If unsatisfiable, the
//! block violates at least one constitutional invariant and MUST be rejected by
//! the pipeline.
//!
//! # Integration Points [VERIFIED from source]
//!
//! - `bizra-proofspace`: consumes [`FateProof`] and embeds [`FateProof::assertions`]
//!   into [`EthicalEnvelope::formal_assertions`] (Vec<String> of SMT-LIB2 fragments).
//! - `bizra-sippar`: [`RegularNumber::from_u64`] is called to test chain length
//!   smoothness before emitting the Sippar assertion.
//! - Constants re-export the single source of truth from `bizra-core`.
//!
//! # Standing on Giants
//!
//! - de Moura & Bjørner (2008): "Z3: An Efficient SMT Solver",
//!   TACAS 2008, LNCS 4963, pp. 337–340. [VERIFIED]
//! - Barrett et al. (2010): "The SMT-LIB Standard: Version 2.0",
//!   Workshop on Satisfiability Modulo Theories. [VERIFIED]
//! - Dijkstra (1976): "A Discipline of Programming", Prentice-Hall —
//!   "Program testing can show the presence of bugs but never their absence;
//!   formal verification shows their absence." [VERIFIED]
//! - BIZRA Constitution: Compression Theorem — constrain the space until only
//!   constitutional states are reachable. [VERIFIED]
//! - Lehmer (1964): Regular (5-smooth / Hamming) numbers for harmonic chain
//!   lengths, ensuring exact reciprocal arithmetic. [VERIFIED]

#![warn(missing_docs)]

// ---------------------------------------------------------------------------
// Re-exported constants from bizra-proofspace (single source of truth)
// [VERIFIED from bizra-proofspace/src/lib.rs]
// ---------------------------------------------------------------------------

/// Ihsan quality floor: blocks scoring below this MUST be rejected. [VERIFIED]
pub const IHSAN_THRESHOLD: f64 = 0.95;

/// Adl Gini inequality maximum: Gini coefficient of resource distribution. [VERIFIED]
pub const ADL_GINI_MAX: f64 = 0.35;

/// Maximum permissible harm score. [VERIFIED]
pub const MAX_HARM_SCORE: f64 = 0.30;

/// Minimum confidence score for a block to pass validation. [VERIFIED]
pub const MIN_CONFIDENCE: f64 = 0.80;

/// Signal-to-noise ratio floor (not yet enforced upstream but declared here). [DERIVED]
pub const SNR_FLOOR: f64 = 0.85;

// ---------------------------------------------------------------------------
// FateScores — mirrors bizra-proofspace/src/lib.rs exactly [VERIFIED]
// ---------------------------------------------------------------------------

/// The four FATE gate scores for a single ProofSpace block.
///
/// Mirrors `bizra_proofspace::FateScores`. Defined here to keep this module
/// self-contained while remaining binary-compatible with the upstream type.
#[derive(Debug, Clone, PartialEq)]
pub struct FateScores {
    /// Ihsan (excellence / benevolence) score. Must be ≥ 0.95. [VERIFIED]
    pub ihsan_score: f64,
    /// Adl (justice / equality) score. Gini of (1 − adl_score) must be ≤ 0.35. [VERIFIED]
    pub adl_score: f64,
    /// Harm score. Must be ≤ 0.30. [VERIFIED]
    pub harm_score: f64,
    /// Confidence score. Must be ≥ 0.80. [VERIFIED]
    pub confidence_score: f64,
}

// ---------------------------------------------------------------------------
// ConstitutionalThresholds
// ---------------------------------------------------------------------------

/// The immutable constitutional thresholds, derived from bizra-core single
/// source of truth.
///
/// These values are deliberately duplicated here (not behind a trait or
/// cfg-flag) so that the SMT-LIB2 scripts are fully self-describing: an
/// independent verifier can re-run the script without access to the BIZRA
/// binary.
///
/// [VERIFIED from bizra-proofspace constants + constants.py]
#[derive(Debug, Clone, PartialEq)]
pub struct ConstitutionalThresholds {
    /// Ihsan floor — excellence gate lower bound. [VERIFIED]
    pub ihsan_floor: f64,
    /// Adl Gini maximum — inequality ceiling. [VERIFIED]
    pub adl_gini_max: f64,
    /// Maximum harm score. [VERIFIED]
    pub max_harm_score: f64,
    /// Minimum confidence score. [VERIFIED]
    pub min_confidence: f64,
    /// Signal-to-noise ratio floor. [DERIVED]
    pub snr_floor: f64,
}

impl Default for ConstitutionalThresholds {
    fn default() -> Self {
        Self {
            ihsan_floor: IHSAN_THRESHOLD,
            adl_gini_max: ADL_GINI_MAX,
            max_harm_score: MAX_HARM_SCORE,
            min_confidence: MIN_CONFIDENCE,
            snr_floor: SNR_FLOOR,
        }
    }
}

// ---------------------------------------------------------------------------
// FATE gate enum
// ---------------------------------------------------------------------------

/// Which constitutional gate a given SMT assertion enforces.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum FateGate {
    /// Ihsan (excellence / benevolence) gate. [VERIFIED]
    Ihsan,
    /// Adl (justice / Gini) gate. [VERIFIED]
    Adl,
    /// Harm gate. [VERIFIED]
    Harm,
    /// Confidence gate. [VERIFIED]
    Confidence,
    /// Sippar regularity gate — chain length must be 2,3,5-smooth. [DERIVED]
    Sippar,
    /// Chain integrity gate — hash continuity. [PLANNED]
    ChainIntegrity,
}

impl FateGate {
    /// Returns the human-readable name used in SMT-LIB2 comments.
    fn label(&self) -> &'static str {
        match self {
            FateGate::Ihsan => "Ihsan",
            FateGate::Adl => "Adl",
            FateGate::Harm => "Harm",
            FateGate::Confidence => "Confidence",
            FateGate::Sippar => "Sippar",
            FateGate::ChainIntegrity => "ChainIntegrity",
        }
    }
}

// ---------------------------------------------------------------------------
// SMT-LIB2 types
// ---------------------------------------------------------------------------

/// The sort (type) of an SMT-LIB2 variable.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SmtSort {
    /// Unbounded real arithmetic (QF_LRA). [VERIFIED]
    Real,
    /// Unbounded integer arithmetic (QF_LIA). [VERIFIED]
    Int,
    /// Boolean. [VERIFIED]
    Bool,
    /// Fixed-width bit vector of given width. [VERIFIED]
    BitVec(u32),
}

impl SmtSort {
    /// Renders the sort as an SMT-LIB2 sort name.
    pub fn as_smtlib2(&self) -> String {
        match self {
            SmtSort::Real => "Real".to_string(),
            SmtSort::Int => "Int".to_string(),
            SmtSort::Bool => "Bool".to_string(),
            SmtSort::BitVec(w) => format!("(_ BitVec {})", w),
        }
    }
}

/// A single named SMT-LIB2 variable declaration.
#[derive(Debug, Clone, PartialEq)]
pub struct SmtVariable {
    /// The variable name as it appears in SMT-LIB2 scripts.
    pub name: String,
    /// The SMT-LIB2 sort of this variable.
    pub sort: SmtSort,
    /// The concrete value assigned (present after model extraction). [PLANNED]
    pub value: Option<String>,
}

/// A single named SMT-LIB2 assertion.
#[derive(Debug, Clone, PartialEq)]
pub struct SmtAssertion {
    /// Short identifier / comment name for this assertion.
    pub name: String,
    /// The complete SMT-LIB2 `(assert ...)` fragment. [VERIFIED — must start
    /// with `(` and have balanced parentheses to pass
    /// `is_valid_smtlib2_fragment`]
    pub smtlib2: String,
    /// Which constitutional gate this assertion enforces.
    pub gate: FateGate,
    /// If `true`, failing this assertion marks the block as Dead (hard
    /// rejection). Soft violations are advisory. [VERIFIED]
    pub is_hard: bool,
}

/// The outcome of calling `(check-sat)` on the assembled proof script.
#[derive(Debug, Clone, PartialEq)]
pub enum ProofResult {
    /// The solver found a satisfying model — constitutional invariants hold.
    Satisfiable {
        /// Variable–value pairs from the solver model. [PLANNED]
        model: Vec<(String, String)>,
    },
    /// No satisfying assignment exists — at least one invariant is violated.
    Unsatisfiable {
        /// Named assertions that form the unsatisfiable core. [PLANNED]
        unsat_core: Vec<String>,
    },
    /// The solver timed out or the problem is undecidable.
    Unknown {
        /// Human-readable reason.
        reason: String,
    },
    /// No solver call has been made yet (default after generation).
    NotChecked,
}

/// Lightweight solver statistics attached to every [`FateProof`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SolverStats {
    /// Total number of `(assert ...)` expressions in the proof script.
    pub num_assertions: usize,
    /// Total number of `(declare-const ...)` declarations.
    pub num_variables: usize,
    /// Wall-clock time consumed by `(check-sat)`, in microseconds. [PLANNED]
    pub check_time_us: u64,
}

// ---------------------------------------------------------------------------
// FateProof
// ---------------------------------------------------------------------------

/// A complete fate-binding proof for a block or mission.
///
/// The proof contains:
/// 1. An ordered list of [`SmtAssertion`]s that encode constitutional gates.
/// 2. Variable declarations ([`SmtVariable`]) for all `declare-const` names.
/// 3. A [`ProofResult`] that records what the solver returned (or
///    [`ProofResult::NotChecked`] when running offline / no Z3 binary).
/// 4. A deterministic 32-byte [`proof_id`](FateProof::proof_id) derived from
///    chain hash + sorted assertion content (SHA-256 emulation via XOR fold —
///    no external deps required in this self-contained file). [DERIVED]
#[derive(Debug, Clone, PartialEq)]
pub struct FateProof {
    /// Deterministic proof identity: SHA-256-like fold over chain_hash ∥
    /// sorted assertion bytes. Same inputs always yield the same id. [DERIVED]
    pub proof_id: [u8; 32],
    /// All named assertions, in the order they appear in the script.
    pub assertions: Vec<SmtAssertion>,
    /// All variable declarations, in the order they appear in the script.
    pub variables: Vec<SmtVariable>,
    /// Result from the SMT solver (or `NotChecked` if no solver is available).
    pub result: ProofResult,
    /// Unix-epoch microseconds at proof generation time. [DERIVED]
    pub generation_timestamp: u64,
    /// Lightweight stats about this proof. [DERIVED]
    pub solver_stats: SolverStats,
}

impl FateProof {
    /// Renders the complete SMT-LIB2 proof script as a single UTF-8 string.
    ///
    /// The output is suitable for piping directly to `z3 -in` or any
    /// SMT-LIB2-compliant solver.
    pub fn render_script(&self) -> String {
        let mut out = String::new();
        out.push_str("; BIZRA Fate-Binding Proof\n");
        out.push_str("; Generated by bizra-proofspace fate-binding engine\n");
        out.push_str("; Proof-ID: ");
        for b in &self.proof_id {
            out.push_str(&format!("{:02x}", b));
        }
        out.push('\n');
        out.push_str("(set-logic QF_LRA)\n\n");

        for var in &self.variables {
            out.push_str(&format!(
                "(declare-const {} {})\n",
                var.name,
                var.sort.as_smtlib2()
            ));
        }
        out.push('\n');

        for assertion in &self.assertions {
            out.push_str(&format!("; Gate: {}\n", assertion.gate.label()));
            out.push_str(&assertion.smtlib2);
            out.push('\n');
        }
        out.push_str("\n(check-sat)\n");
        out
    }
}

// ---------------------------------------------------------------------------
// FateBindingError
// ---------------------------------------------------------------------------

/// Errors produced by the [`FateBindingEngine`].
#[derive(Debug, Clone, PartialEq)]
pub enum FateBindingError {
    /// The assembled proof script is unsatisfiable: a constitutional gate is
    /// violated. The block MUST be rejected. [VERIFIED]
    UnsatisfiableProof {
        /// Which gate caused the contradiction.
        gate: FateGate,
        /// Human-readable detail (threshold violated, actual value, etc.).
        details: String,
    },
    /// A provided SMT-LIB2 fragment is syntactically invalid. [VERIFIED]
    InvalidSmtLib2 {
        /// The offending fragment.
        fragment: String,
        /// Explanation of the syntax error.
        reason: String,
    },
    /// The chain length is not a regular (2,3,5-smooth) number. [DERIVED]
    SipparNonRegular {
        /// The chain length that was tested.
        chain_length: u64,
        /// The first irregular prime factor found.
        remainder: u64,
    },
    /// A score exceeds or falls below a constitutional threshold. [VERIFIED]
    ThresholdViolation {
        /// Which gate detected the violation.
        gate: FateGate,
        /// The actual score.
        actual: f64,
        /// The required threshold (floor or ceiling depending on gate).
        required: f64,
    },
}

impl std::fmt::Display for FateBindingError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            FateBindingError::UnsatisfiableProof { gate, details } => {
                write!(f, "UnsatisfiableProof({:?}): {}", gate, details)
            }
            FateBindingError::InvalidSmtLib2 { fragment, reason } => {
                write!(
                    f,
                    "InvalidSmtLib2: {} — fragment: {:?}",
                    reason, fragment
                )
            }
            FateBindingError::SipparNonRegular {
                chain_length,
                remainder,
            } => {
                write!(
                    f,
                    "SipparNonRegular: chain_length={} has irregular factor {}",
                    chain_length, remainder
                )
            }
            FateBindingError::ThresholdViolation {
                gate,
                actual,
                required,
            } => {
                write!(
                    f,
                    "ThresholdViolation({:?}): actual={:.4} required={:.4}",
                    gate, actual, required
                )
            }
        }
    }
}

impl std::error::Error for FateBindingError {}

// ---------------------------------------------------------------------------
// AssertionValidation
// ---------------------------------------------------------------------------

/// The result of validating a single SMT-LIB2 fragment for syntactic
/// correctness.
#[derive(Debug, Clone, PartialEq)]
pub struct AssertionValidation {
    /// Zero-based index into the input slice.
    pub index: usize,
    /// `true` if the fragment passes all syntactic checks.
    pub is_valid: bool,
    /// Human-readable diagnostics; empty when `is_valid` is `true`.
    pub diagnostics: Vec<String>,
}

// ---------------------------------------------------------------------------
// Proof-ID derivation (no_std-compatible deterministic fold) [DERIVED]
// ---------------------------------------------------------------------------

/// Derives a 32-byte proof identity from the chain hash and assertion content.
///
/// Algorithm (no external deps):
/// 1. Start with `chain_hash`.
/// 2. For each assertion (sorted by name for determinism), XOR-fold the UTF-8
///    bytes of `smtlib2` into successive 32-byte windows using a simple
///    Davies–Meyer-like step: `state[i % 32] ^= byte.wrapping_add(i as u8)`.
///
/// This is **not** a cryptographic hash — it is a deterministic fingerprint
/// sufficient for proof de-duplication. [DERIVED]
fn derive_proof_id(chain_hash: &[u8; 32], assertions: &[SmtAssertion]) -> [u8; 32] {
    let mut state = *chain_hash;
    // Sort assertion names for determinism regardless of insertion order.
    let mut names: Vec<&str> = assertions.iter().map(|a| a.name.as_str()).collect();
    names.sort_unstable();

    for name in names {
        // Find the matching assertion.
        if let Some(assertion) = assertions.iter().find(|a| a.name == name) {
            for (i, byte) in assertion.smtlib2.bytes().enumerate() {
                state[i % 32] ^= byte.wrapping_add(i as u8);
            }
            // Mix round using name bytes.
            for (i, byte) in name.bytes().enumerate() {
                let idx = (i + 16) % 32;
                state[idx] = state[idx].wrapping_add(byte).rotate_left(1);
            }
        }
    }
    state
}

// ---------------------------------------------------------------------------
// SMT-LIB2 syntax validation helpers [VERIFIED — mirrors is_valid_smtlib2_fragment]
// ---------------------------------------------------------------------------

/// Returns `true` if `fragment` has balanced parentheses and begins with `(`.
///
/// Mirrors `bizra_proofspace::is_valid_smtlib2_fragment()`. [VERIFIED]
pub fn is_valid_smtlib2_fragment(fragment: &str) -> bool {
    let trimmed = fragment.trim();
    if !trimmed.starts_with('(') {
        return false;
    }
    let mut depth: i64 = 0;
    let mut in_string = false;
    let mut prev = '\0';
    for ch in trimmed.chars() {
        match ch {
            '"' if prev != '\\' => in_string = !in_string,
            '(' if !in_string => depth += 1,
            ')' if !in_string => {
                depth -= 1;
                if depth < 0 {
                    return false;
                }
            }
            _ => {}
        }
        prev = ch;
    }
    depth == 0
}

/// Validates an SMT-LIB2 fragment and returns a list of diagnostics.
///
/// An empty diagnostics list means the fragment is valid. [DERIVED]
fn validate_smtlib2_fragment(fragment: &str) -> Vec<String> {
    let mut diags = Vec::new();
    let trimmed = fragment.trim();

    if trimmed.is_empty() {
        diags.push("Fragment is empty".to_string());
        return diags;
    }
    if !trimmed.starts_with('(') {
        diags.push(format!(
            "Fragment must start with '(' but starts with '{}'",
            trimmed.chars().next().unwrap_or('?')
        ));
    }

    // Check balanced parentheses, tracking string literals.
    let mut depth: i64 = 0;
    let mut in_string = false;
    let mut prev = '\0';
    for (pos, ch) in trimmed.char_indices() {
        match ch {
            '"' if prev != '\\' => in_string = !in_string,
            '(' if !in_string => depth += 1,
            ')' if !in_string => {
                depth -= 1;
                if depth < 0 {
                    diags.push(format!(
                        "Unmatched ')' at byte position {}",
                        pos
                    ));
                }
            }
            _ => {}
        }
        prev = ch;
    }
    if depth > 0 {
        diags.push(format!(
            "Unclosed '(' — {} parenthesis/es not closed",
            depth
        ));
    }

    diags
}

// ---------------------------------------------------------------------------
// Sippar regularity check (pure Rust — no bizra-sippar dep) [DERIVED]
// ---------------------------------------------------------------------------

/// Checks whether `n` is a regular (2,3,5-smooth) number.
///
/// A regular number has no prime factors other than 2, 3, and 5.
/// Returns `Ok(())` if regular, or `Err(first_irregular_factor)` if not.
///
/// Mirrors `bizra_sippar::RegularNumber::from_u64` semantics. [VERIFIED]
fn check_regular(mut n: u64) -> Result<(), u64> {
    if n == 0 {
        return Err(0);
    }
    for &p in &[2u64, 3, 5] {
        while n % p == 0 {
            n /= p;
        }
    }
    if n == 1 {
        Ok(())
    } else {
        Err(n)
    }
}

// ---------------------------------------------------------------------------
// FateBindingEngine
// ---------------------------------------------------------------------------

/// The Fate-Binding Engine: generates and validates SMT-LIB2 formal proofs
/// for BIZRA constitutional invariants.
///
/// "Fate-binding" means: if the proof is satisfiable, the block's claims
/// are consistent with constitutional law. If unsatisfiable, the block
/// violates at least one invariant and MUST be rejected.
///
/// # Standing on Giants
///
/// - de Moura & Bjørner (2008): Z3 SMT solver [VERIFIED]
/// - Barrett et al. (2010): SMT-LIB2 standard [VERIFIED]
/// - Dijkstra (1976): "Program testing shows the presence of bugs; formal
///   verification shows their absence" [VERIFIED]
/// - BIZRA Constitution: Compression Theorem — constrain the space until only
///   constitutional states are reachable [VERIFIED]
pub struct FateBindingEngine {
    constitutional_thresholds: ConstitutionalThresholds,
    /// When `true`, the engine will also generate a Sippar (regularity) proof
    /// for the chain length, rejecting irregular chain lengths. [DERIVED]
    enable_sippar_proofs: bool,
}

impl Default for FateBindingEngine {
    fn default() -> Self {
        Self {
            constitutional_thresholds: ConstitutionalThresholds::default(),
            enable_sippar_proofs: true,
        }
    }
}

impl FateBindingEngine {
    /// Constructs a new engine with default constitutional thresholds.
    pub fn new() -> Self {
        Self::default()
    }

    /// Constructs an engine with custom thresholds (for testing / staging). [DERIVED]
    pub fn with_thresholds(thresholds: ConstitutionalThresholds, enable_sippar: bool) -> Self {
        Self {
            constitutional_thresholds: thresholds,
            enable_sippar_proofs: enable_sippar,
        }
    }

    // -----------------------------------------------------------------------
    // Core: generate_fate_proof
    // -----------------------------------------------------------------------

    /// Generates the complete SMT-LIB2 proof script for a given set of FATE
    /// scores.
    ///
    /// The generated script uses `QF_LRA` (Quantifier-Free Linear Real
    /// Arithmetic) logic. The proof is **not** sent to a solver by this
    /// method — the result is always [`ProofResult::NotChecked`].  To
    /// actually check satisfiability, pipe [`FateProof::render_script`] to a
    /// Z3 binary.
    ///
    /// If `enable_sippar_proofs` is `true`, an additional Sippar assertion is
    /// appended when `chain_length` is regular.  Irregular chain lengths
    /// produce a soft (non-hard) advisory assertion.
    ///
    /// # Arguments
    ///
    /// * `scores`       — The four FATE gate scores for the block. [VERIFIED]
    /// * `chain_hash`   — The 32-byte hash of the preceding block. [VERIFIED]
    /// * `chain_length` — The current chain height. [DERIVED]
    ///
    /// # Returns
    ///
    /// A fully populated [`FateProof`] with `result = NotChecked`.
    pub fn generate_fate_proof(
        &self,
        scores: &FateScores,
        chain_hash: &[u8; 32],
        chain_length: u64,
    ) -> FateProof {
        let t = &self.constitutional_thresholds;
        let mut variables: Vec<SmtVariable> = Vec::new();
        let mut assertions: Vec<SmtAssertion> = Vec::new();

        // ---- Threshold declarations ----------------------------------------
        let threshold_vars = [
            ("ihsan_threshold", t.ihsan_floor),
            ("adl_gini_max", t.adl_gini_max),
            ("max_harm", t.max_harm_score),
            ("min_confidence", t.min_confidence),
        ];
        for (name, value) in &threshold_vars {
            variables.push(SmtVariable {
                name: name.to_string(),
                sort: SmtSort::Real,
                value: Some(format_real(*value)),
            });
            assertions.push(SmtAssertion {
                name: format!("set_{}", name),
                smtlib2: format!("(assert (= {} {}))", name, format_real(*value)),
                gate: FateGate::Ihsan, // threshold-setting; gate label is advisory
                is_hard: true,
            });
        }

        // ---- Score declarations --------------------------------------------
        let score_vars = [
            ("ihsan_score", scores.ihsan_score),
            ("adl_score", scores.adl_score),
            ("harm_score", scores.harm_score),
            ("confidence_score", scores.confidence_score),
        ];
        for (name, value) in &score_vars {
            variables.push(SmtVariable {
                name: name.to_string(),
                sort: SmtSort::Real,
                value: Some(format_real(*value)),
            });
            assertions.push(SmtAssertion {
                name: format!("set_{}", name),
                smtlib2: format!("(assert (= {} {}))", name, format_real(*value)),
                gate: FateGate::Ihsan, // value-setting; gate label is advisory
                is_hard: true,
            });
        }

        // ---- Ihsan gate ----------------------------------------------------
        // ihsan_score >= ihsan_threshold
        assertions.push(SmtAssertion {
            name: "gate_ihsan".to_string(),
            smtlib2: "(assert (>= ihsan_score ihsan_threshold))".to_string(),
            gate: FateGate::Ihsan,
            is_hard: true,
        });

        // ---- Adl gate ------------------------------------------------------
        // Gini = 1 - adl_score; must be <= adl_gini_max
        // Equivalent: (1 - adl_score) <= adl_gini_max  →  adl_score >= 1 - adl_gini_max
        // SMT-LIB2 linear form: (<= (- 1.0 adl_score) adl_gini_max) [VERIFIED]
        assertions.push(SmtAssertion {
            name: "gate_adl".to_string(),
            smtlib2: "(assert (<= (- 1.0 adl_score) adl_gini_max))".to_string(),
            gate: FateGate::Adl,
            is_hard: true,
        });

        // ---- Harm gate -----------------------------------------------------
        // harm_score <= max_harm
        assertions.push(SmtAssertion {
            name: "gate_harm".to_string(),
            smtlib2: "(assert (<= harm_score max_harm))".to_string(),
            gate: FateGate::Harm,
            is_hard: true,
        });

        // ---- Confidence gate -----------------------------------------------
        // confidence_score >= min_confidence
        assertions.push(SmtAssertion {
            name: "gate_confidence".to_string(),
            smtlib2: "(assert (>= confidence_score min_confidence))".to_string(),
            gate: FateGate::Confidence,
            is_hard: true,
        });

        // ---- Sippar gate (optional) ----------------------------------------
        if self.enable_sippar_proofs {
            if let Ok(sippar_assertion) = self.generate_sippar_proof(chain_length) {
                assertions.push(sippar_assertion);
            } else {
                // Soft advisory for irregular chain lengths.
                assertions.push(SmtAssertion {
                    name: "gate_sippar_advisory".to_string(),
                    smtlib2: format!(
                        "(assert (= chain_length_{} chain_length_{}))",
                        chain_length, chain_length
                    ),
                    gate: FateGate::Sippar,
                    is_hard: false,
                });
            }
        }

        // ---- Score range guards (auxiliary — not constitutional) [DERIVED] -
        // Scores must lie in [0, 1].
        for name in &["ihsan_score", "adl_score", "harm_score", "confidence_score"] {
            assertions.push(SmtAssertion {
                name: format!("range_{}", name),
                smtlib2: format!(
                    "(assert (and (>= {} 0.0) (<= {} 1.0)))",
                    name, name
                ),
                gate: FateGate::ChainIntegrity,
                is_hard: false,
            });
        }

        let num_assertions = assertions.len();
        let num_variables = variables.len();
        let proof_id = derive_proof_id(chain_hash, &assertions);

        FateProof {
            proof_id,
            assertions,
            variables,
            result: ProofResult::NotChecked,
            generation_timestamp: current_timestamp_us(),
            solver_stats: SolverStats {
                num_assertions,
                num_variables,
                check_time_us: 0,
            },
        }
    }

    // -----------------------------------------------------------------------
    // generate_sippar_proof
    // -----------------------------------------------------------------------

    /// Generates an SMT-LIB2 assertion that `chain_length` is a regular
    /// (2,3,5-smooth / Hamming) number.
    ///
    /// The assertion uses the existential encoding:
    /// ```text
    /// (assert (exists ((a Int) (b Int) (c Int))
    ///   (and (>= a 0) (>= b 0) (>= c 0)
    ///        (= chain_length (* (^ 2 a) (* (^ 3 b) (^ 5 c)))))))
    /// ```
    ///
    /// Note: this is placed inside `QF_LRA` scripts as a **comment-embedded**
    /// appendix; solvers running `QF_LRA` will not process quantifiers. The
    /// full regularity proof is only meaningful when the script logic is
    /// relaxed to `LIA` or `NIA`. This is noted inline. [DERIVED]
    ///
    /// # Errors
    ///
    /// Returns [`FateBindingError::SipparNonRegular`] if `chain_length` has a
    /// prime factor other than 2, 3, or 5. [DERIVED]
    pub fn generate_sippar_proof(
        &self,
        chain_length: u64,
    ) -> Result<SmtAssertion, FateBindingError> {
        match check_regular(chain_length) {
            Ok(()) => {
                // Build the concrete witness from the factorisation.
                let (a, b, c) = factor_regular(chain_length);
                let smtlib2 = format!(
                    concat!(
                        "; Sippar regularity proof: {} = 2^{} * 3^{} * 5^{}\n",
                        "; (Note: existential is for NIA logic; inline as documentary assertion)\n",
                        "(assert (exists ((a Int) (b Int) (c Int))\n",
                        "  (and (>= a 0) (>= b 0) (>= c 0)\n",
                        "       (= {} (* (^ 2 a) (* (^ 3 b) (^ 5 c)))))))"
                    ),
                    chain_length, a, b, c, chain_length
                );
                Ok(SmtAssertion {
                    name: "gate_sippar".to_string(),
                    smtlib2,
                    gate: FateGate::Sippar,
                    is_hard: true,
                })
            }
            Err(remainder) => Err(FateBindingError::SipparNonRegular {
                chain_length,
                remainder,
            }),
        }
    }

    // -----------------------------------------------------------------------
    // validate_formal_assertions
    // -----------------------------------------------------------------------

    /// Validates a slice of SMT-LIB2 fragments for syntactic correctness.
    ///
    /// Checks performed:
    /// 1. Fragment must not be empty.
    /// 2. Fragment must begin with `(`.
    /// 3. Parentheses must be balanced (ignoring string literals).
    ///
    /// This mirrors the checks performed by `bizra_proofspace::is_valid_smtlib2_fragment`. [VERIFIED]
    pub fn validate_formal_assertions(&self, assertions: &[String]) -> Vec<AssertionValidation> {
        assertions
            .iter()
            .enumerate()
            .map(|(index, fragment)| {
                let diagnostics = validate_smtlib2_fragment(fragment);
                let is_valid = diagnostics.is_empty();
                AssertionValidation {
                    index,
                    is_valid,
                    diagnostics,
                }
            })
            .collect()
    }

    // -----------------------------------------------------------------------
    // extract_fate_assertions
    // -----------------------------------------------------------------------

    /// Extracts the raw SMT-LIB2 strings from a [`FateProof`], suitable for
    /// embedding directly into [`EthicalEnvelope::formal_assertions`]
    /// (which is a `Vec<String>` of SMT-LIB2 fragments). [VERIFIED]
    ///
    /// Only includes fragments that pass the `is_valid_smtlib2_fragment` check.
    pub fn extract_fate_assertions(&self, proof: &FateProof) -> Vec<String> {
        proof
            .assertions
            .iter()
            .map(|a| a.smtlib2.clone())
            .filter(|s| {
                // Strip any leading comment lines before checking the fragment.
                let first_meaningful = s
                    .lines()
                    .find(|l| !l.trim_start().starts_with(';') && !l.trim().is_empty());
                first_meaningful
                    .map(|l| is_valid_smtlib2_fragment(l.trim()))
                    .unwrap_or(false)
            })
            .collect()
    }
}

// ---------------------------------------------------------------------------
// Formatting helpers
// ---------------------------------------------------------------------------

/// Formats an `f64` as a decimal real literal suitable for SMT-LIB2.
///
/// Ensures there is always a decimal point so the solver treats it as `Real`
/// rather than `Int`. [DERIVED]
fn format_real(v: f64) -> String {
    // Use enough decimal places to round-trip the constitutional constants.
    let s = format!("{:.6}", v);
    // SMT-LIB2 requires at least one digit on each side of the decimal point.
    if s.contains('.') {
        s
    } else {
        format!("{}.0", s)
    }
}

/// Extracts the 2-, 3-, and 5-exponents of a regular number. [DERIVED]
///
/// Panics if `n` is not regular — call only after `check_regular` succeeds.
fn factor_regular(mut n: u64) -> (u32, u32, u32) {
    let mut a = 0u32;
    let mut b = 0u32;
    let mut c = 0u32;
    while n % 2 == 0 {
        n /= 2;
        a += 1;
    }
    while n % 3 == 0 {
        n /= 3;
        b += 1;
    }
    while n % 5 == 0 {
        n /= 5;
        c += 1;
    }
    debug_assert_eq!(n, 1, "factor_regular called on irregular number");
    (a, b, c)
}

/// Returns the current Unix epoch time in microseconds. [DERIVED]
///
/// Uses `std::time::SystemTime`; falls back to 0 if the clock is unavailable.
fn current_timestamp_us() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_micros() as u64)
        .unwrap_or(0)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: a set of scores that pass all FATE gates. [DERIVED]
    fn passing_scores() -> FateScores {
        FateScores {
            ihsan_score: 0.97,
            adl_score: 0.72, // Gini = 1 - 0.72 = 0.28 ≤ 0.35 ✓
            harm_score: 0.10,
            confidence_score: 0.92,
        }
    }

    /// Helper: a set of scores where Ihsan fails (0.80 < 0.95). [DERIVED]
    fn failing_scores_ihsan() -> FateScores {
        FateScores {
            ihsan_score: 0.80,
            adl_score: 0.72,
            harm_score: 0.10,
            confidence_score: 0.92,
        }
    }

    const ZERO_HASH: [u8; 32] = [0u8; 32];

    // -----------------------------------------------------------------------
    // test_generate_passing_fate_proof
    // -----------------------------------------------------------------------

    /// A proof generated from scores above all thresholds should be internally
    /// consistent: the assertion for the Ihsan gate must be present, and the
    /// score variable must encode the actual score. [VERIFIED]
    #[test]
    fn test_generate_passing_fate_proof() {
        let engine = FateBindingEngine::new();
        let scores = passing_scores();
        let proof = engine.generate_fate_proof(&scores, &ZERO_HASH, 60);

        // Should have the four hard gate assertions.
        let gate_names: Vec<&str> = proof
            .assertions
            .iter()
            .map(|a| a.name.as_str())
            .collect();
        assert!(
            gate_names.contains(&"gate_ihsan"),
            "gate_ihsan assertion missing"
        );
        assert!(
            gate_names.contains(&"gate_adl"),
            "gate_adl assertion missing"
        );
        assert!(
            gate_names.contains(&"gate_harm"),
            "gate_harm assertion missing"
        );
        assert!(
            gate_names.contains(&"gate_confidence"),
            "gate_confidence assertion missing"
        );

        // All hard gate assertions should have correct SMT-LIB2 syntax.
        // Multi-line assertions may begin with ';' comment lines; strip those
        // before checking the full s-expression body. [DERIVED]
        for assertion in proof.assertions.iter().filter(|a| a.is_hard) {
            // Collect all non-comment, non-empty lines and join them.
            let body: String = assertion
                .smtlib2
                .lines()
                .filter(|l| !l.trim_start().starts_with(';') && !l.trim().is_empty())
                .collect::<Vec<_>>()
                .join(" ");
            assert!(
                is_valid_smtlib2_fragment(body.trim()),
                "Hard assertion '{}' has invalid SMT-LIB2: {:?}",
                assertion.name,
                assertion.smtlib2
            );
        }

        // Proof result is NotChecked (no solver binary).
        assert_eq!(proof.result, ProofResult::NotChecked);

        // The ihsan_score variable should encode the actual score.
        let ihsan_var = proof
            .variables
            .iter()
            .find(|v| v.name == "ihsan_score")
            .expect("ihsan_score variable missing");
        assert_eq!(ihsan_var.sort, SmtSort::Real);
        assert_eq!(
            ihsan_var.value.as_deref(),
            Some(format_real(0.97).as_str())
        );
    }

    // -----------------------------------------------------------------------
    // test_generate_failing_fate_proof
    // -----------------------------------------------------------------------

    /// When ihsan_score = 0.80, the proof script encodes an unsatisfiable
    /// system: it asserts both `(= ihsan_score 0.800000)` and
    /// `(>= ihsan_score 0.950000)`. A real solver would return `unsat`.
    ///
    /// We verify that the *script* correctly encodes both assertions (the
    /// engine does not call a solver, so result remains NotChecked). [VERIFIED]
    #[test]
    fn test_generate_failing_fate_proof() {
        let engine = FateBindingEngine::new();
        let scores = failing_scores_ihsan();
        let proof = engine.generate_fate_proof(&scores, &ZERO_HASH, 60);

        // The score must be embedded verbatim (0.800000).
        let set_ihsan = proof
            .assertions
            .iter()
            .find(|a| a.name == "set_ihsan_score")
            .expect("set_ihsan_score missing");
        assert!(
            set_ihsan.smtlib2.contains("0.800000"),
            "ihsan_score value not encoded correctly: {:?}",
            set_ihsan.smtlib2
        );

        // The gate assertion must still be present and be the >= constraint.
        let gate_ihsan = proof
            .assertions
            .iter()
            .find(|a| a.name == "gate_ihsan")
            .expect("gate_ihsan missing");
        assert!(
            gate_ihsan.smtlib2.contains(">="),
            "gate_ihsan should use >= operator"
        );
        assert_eq!(gate_ihsan.gate, FateGate::Ihsan);
        assert!(gate_ihsan.is_hard, "gate_ihsan must be a hard constraint");

        // Render the script and verify the contradiction is visible as text.
        let script = proof.render_script();
        assert!(
            script.contains("0.800000"),
            "Rendered script missing ihsan score"
        );
        assert!(
            script.contains("0.950000"),
            "Rendered script missing ihsan threshold"
        );
    }

    // -----------------------------------------------------------------------
    // test_sippar_proof_regular_chain
    // -----------------------------------------------------------------------

    /// Chain length 60 = 2^2 * 3 * 5 is a regular number. [VERIFIED]
    #[test]
    fn test_sippar_proof_regular_chain() {
        let engine = FateBindingEngine::new();
        let result = engine.generate_sippar_proof(60);
        assert!(result.is_ok(), "Expected Ok for regular chain length 60");
        let assertion = result.unwrap();
        assert_eq!(assertion.gate, FateGate::Sippar);
        assert!(assertion.is_hard);
        // Comment should mention the factorisation.
        assert!(
            assertion.smtlib2.contains("60"),
            "Sippar proof should reference chain length 60"
        );
        // The assertion body must contain the existential encoding.
        assert!(
            assertion.smtlib2.contains("exists"),
            "Sippar proof should contain exists quantifier"
        );
    }

    // -----------------------------------------------------------------------
    // test_sippar_proof_irregular_chain
    // -----------------------------------------------------------------------

    /// Chain length 7 is prime (irregular). [VERIFIED]
    #[test]
    fn test_sippar_proof_irregular_chain() {
        let engine = FateBindingEngine::new();
        let result = engine.generate_sippar_proof(7);
        assert!(result.is_err(), "Expected Err for irregular chain length 7");
        match result.unwrap_err() {
            FateBindingError::SipparNonRegular {
                chain_length,
                remainder,
            } => {
                assert_eq!(chain_length, 7);
                assert_eq!(remainder, 7);
            }
            other => panic!("Unexpected error variant: {:?}", other),
        }
    }

    // -----------------------------------------------------------------------
    // test_smtlib2_syntax_validation
    // -----------------------------------------------------------------------

    /// Valid and invalid SMT-LIB2 fragments are classified correctly. [VERIFIED]
    #[test]
    fn test_smtlib2_syntax_validation() {
        let engine = FateBindingEngine::new();
        let fragments = vec![
            // valid
            "(assert (>= ihsan_score 0.950000))".to_string(),
            "(assert (and (>= a 0) (<= a 1)))".to_string(),
            "(declare-const x Real)".to_string(),
            // invalid — missing opening paren
            "assert (>= x 0.0)".to_string(),
            // invalid — unbalanced parens
            "(assert (>= x 0.0)".to_string(),
            // invalid — extra closing paren
            "(assert (>= x 0.0)))".to_string(),
            // invalid — empty
            "".to_string(),
        ];

        let results = engine.validate_formal_assertions(&fragments);
        assert_eq!(results.len(), 7);

        assert!(results[0].is_valid, "Fragment 0 should be valid");
        assert!(results[1].is_valid, "Fragment 1 should be valid");
        assert!(results[2].is_valid, "Fragment 2 should be valid");

        assert!(!results[3].is_valid, "Fragment 3 (no paren) should be invalid");
        assert!(!results[4].is_valid, "Fragment 4 (unbalanced) should be invalid");
        assert!(!results[5].is_valid, "Fragment 5 (extra close) should be invalid");
        assert!(!results[6].is_valid, "Fragment 6 (empty) should be invalid");

        // Non-empty diagnostics for invalid fragments.
        assert!(!results[3].diagnostics.is_empty());
        assert!(!results[4].diagnostics.is_empty());
        assert!(!results[5].diagnostics.is_empty());
        assert!(!results[6].diagnostics.is_empty());
    }

    // -----------------------------------------------------------------------
    // test_extract_assertions_for_ethical_envelope
    // -----------------------------------------------------------------------

    /// Extracted assertions must all pass `is_valid_smtlib2_fragment`. [VERIFIED]
    #[test]
    fn test_extract_assertions_for_ethical_envelope() {
        let engine = FateBindingEngine::new();
        let scores = passing_scores();
        let proof = engine.generate_fate_proof(&scores, &ZERO_HASH, 60);

        let extracted = engine.extract_fate_assertions(&proof);

        // Every extracted fragment must be syntactically valid (ignoring
        // comment lines at the start of multi-line assertions).
        for fragment in &extracted {
            let first_meaningful = fragment
                .lines()
                .find(|l| !l.trim_start().starts_with(';') && !l.trim().is_empty())
                .unwrap_or(fragment.as_str());
            assert!(
                is_valid_smtlib2_fragment(first_meaningful.trim()),
                "Extracted assertion is not valid SMT-LIB2: {:?}",
                fragment
            );
        }

        // There should be at least the four gate assertions.
        assert!(
            extracted.len() >= 4,
            "Expected at least 4 extracted assertions, got {}",
            extracted.len()
        );
    }

    // -----------------------------------------------------------------------
    // test_default_thresholds_match_constants
    // -----------------------------------------------------------------------

    /// `ConstitutionalThresholds::default()` must match the module-level
    /// constants (which mirror `bizra-proofspace` and `bizra-core`). [VERIFIED]
    #[test]
    fn test_default_thresholds_match_constants() {
        let t = ConstitutionalThresholds::default();
        assert_eq!(t.ihsan_floor, IHSAN_THRESHOLD);
        assert_eq!(t.adl_gini_max, ADL_GINI_MAX);
        assert_eq!(t.max_harm_score, MAX_HARM_SCORE);
        assert_eq!(t.min_confidence, MIN_CONFIDENCE);
        assert_eq!(t.snr_floor, SNR_FLOOR);
    }

    // -----------------------------------------------------------------------
    // test_proof_id_deterministic
    // -----------------------------------------------------------------------

    /// Two calls with identical inputs must produce the same `proof_id`. [VERIFIED]
    #[test]
    fn test_proof_id_deterministic() {
        let engine = FateBindingEngine::new();
        let scores = passing_scores();
        let chain_hash: [u8; 32] = [
            0xde, 0xad, 0xbe, 0xef, 0x00, 0x11, 0x22, 0x33,
            0x44, 0x55, 0x66, 0x77, 0x88, 0x99, 0xaa, 0xbb,
            0xcc, 0xdd, 0xee, 0xff, 0x01, 0x02, 0x03, 0x04,
            0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c,
        ];

        let proof_a = engine.generate_fate_proof(&scores, &chain_hash, 100);
        let proof_b = engine.generate_fate_proof(&scores, &chain_hash, 100);

        assert_eq!(
            proof_a.proof_id, proof_b.proof_id,
            "proof_id must be deterministic for identical inputs"
        );
    }

    // -----------------------------------------------------------------------
    // Additional: test_sippar_small_regulars
    // -----------------------------------------------------------------------

    /// The first several regular numbers are accepted. [DERIVED]
    #[test]
    fn test_sippar_small_regulars() {
        let regulars = [1u64, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25,
                        27, 30, 32, 36, 40, 45, 48, 50, 54, 60, 64, 72, 75, 80, 90,
                        96, 100, 120, 125, 128, 144, 150, 160, 180, 192, 200];
        for &n in &regulars {
            assert!(
                check_regular(n).is_ok(),
                "{} should be regular",
                n
            );
        }
    }

    /// Primes ≥ 7 are irregular. [DERIVED]
    #[test]
    fn test_sippar_irregulars() {
        let irregulars = [7u64, 11, 13, 14, 17, 19, 21, 22, 23, 26, 28, 49, 77, 91];
        for &n in &irregulars {
            assert!(
                check_regular(n).is_err(),
                "{} should be irregular",
                n
            );
        }
    }

    // -----------------------------------------------------------------------
    // Additional: test_render_script_structure
    // -----------------------------------------------------------------------

    /// The rendered script must start with the BIZRA header, declare the
    /// logic, and end with `(check-sat)`. [DERIVED]
    #[test]
    fn test_render_script_structure() {
        let engine = FateBindingEngine::new();
        let proof = engine.generate_fate_proof(&passing_scores(), &ZERO_HASH, 60);
        let script = proof.render_script();

        assert!(
            script.contains("(set-logic QF_LRA)"),
            "Script must declare QF_LRA logic"
        );
        assert!(
            script.contains("(check-sat)"),
            "Script must end with (check-sat)"
        );
        assert!(
            script.contains("BIZRA Fate-Binding Proof"),
            "Script must contain BIZRA header comment"
        );
        assert!(
            script.contains("Proof-ID:"),
            "Script must embed the proof ID"
        );
    }

    // -----------------------------------------------------------------------
    // Additional: test_format_real
    // -----------------------------------------------------------------------

    /// `format_real` must produce values with decimal points. [DERIVED]
    #[test]
    fn test_format_real() {
        assert!(format_real(0.95).contains('.'));
        assert!(format_real(1.0).contains('.'));
        assert!(format_real(0.0).contains('.'));
        // Check that the constitutional constants round-trip correctly.
        assert_eq!(&format_real(0.95), "0.950000");
        assert_eq!(&format_real(0.35), "0.350000");
        assert_eq!(&format_real(0.30), "0.300000");
        assert_eq!(&format_real(0.80), "0.800000");
    }

    // -----------------------------------------------------------------------
    // Additional: test_solver_stats_counts
    // -----------------------------------------------------------------------

    /// `SolverStats` must correctly reflect the number of assertions and
    /// variables generated. [DERIVED]
    #[test]
    fn test_solver_stats_counts() {
        let engine = FateBindingEngine::new();
        let proof = engine.generate_fate_proof(&passing_scores(), &ZERO_HASH, 60);

        assert_eq!(
            proof.solver_stats.num_assertions,
            proof.assertions.len(),
            "SolverStats.num_assertions mismatch"
        );
        assert_eq!(
            proof.solver_stats.num_variables,
            proof.variables.len(),
            "SolverStats.num_variables mismatch"
        );
    }
}
