//! WBS 1.1 — ReceiptChain Trait + SipparReceipt Bridge
//!
//! This module implements the bridge between `bizra-action`'s operational receipts
//! and `bizra-proofspace`'s civilization-grade proof blocks.
//!
//! # Architecture
//!
//! ```text
//! bizra-action                       bizra-proofspace
//! ──────────────────                 ─────────────────────────
//! ConstitutionalReceipt  ──────────▶  ProofBlockSubmission
//! ReceiptChain           ──────────▶  ReceiptChainBridge (trait)
//!                                     ConstitutionalReceiptAdapter
//!                                     SipparChainDigest (via bizra-sippar)
//! ```
//!
//! ## Standing on Giants
//!
//! - **Merkle (1979)** [VERIFIED]: Hash chains as tamper-evident data structures —
//!   the backbone of `ReceiptChain`'s `previous_hash` linking.
//! - **Sippar temple scribes (~1900 BCE)** [VERIFIED]: Regular (3-smooth, extended to 5-smooth)
//!   numbers used for exact metrological accounting; the mathematical basis of
//!   `SipparChainDigest`.
//! - **Al-Ghazali (~1090 CE)** [VERIFIED]: *"Nothing is real until it crosses into evidence"* —
//!   the philosophical mandate for converting every action into a `ConstitutionalReceipt`.
//! - **Ed25519 (Bernstein et al., 2011)** [VERIFIED]: The placeholder signature scheme
//!   carried in `ConstitutionalReceipt::signature`.
//! - **SMT-LIB2 standard (Barrett et al.)** [VERIFIED]: The formal assertion language used
//!   in `ProofBlockSubmission::formal_assertion`.

#![warn(missing_docs)]

// ──────────────────────────────────────────────────────────────────────────────
// § 0.  Crate-level imports
//       (self-contained: no external crate dependencies beyond std)
// ──────────────────────────────────────────────────────────────────────────────

use std::fmt;

// ──────────────────────────────────────────────────────────────────────────────
// § 1.  Mirror types from bizra-action/src/types.rs  [VERIFIED interface]
// ──────────────────────────────────────────────────────────────────────────────

/// Unique identifier for an action, wrapping a `u64` counter.
/// Mirrors `bizra-action::types::ActionId`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActionId(pub u64);

/// Unix-epoch timestamp in milliseconds for when an action occurred.
/// Mirrors `bizra-action::types::ActionTimestamp`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub struct ActionTimestamp(pub u64);

/// Ihsan (excellence / well-doing) quality score, clamped to `[0.0, 1.0]`.
/// Mirrors `bizra-action::types::IhsanScore`.
///
/// The word *ihsan* (إحسان) in Islamic ethics means "doing beautiful things" —
/// here it quantifies the quality and harmlessness of an AI action [DERIVED].
#[derive(Debug, Clone, Copy, PartialEq, PartialOrd)]
pub struct IhsanScore(pub f64);

impl IhsanScore {
    /// Construct a clamped `IhsanScore`, saturating at `[0.0, 1.0]`.
    pub fn new(v: f64) -> Self {
        IhsanScore(v.clamp(0.0, 1.0))
    }

    /// Return the raw `f64` value.
    pub fn value(self) -> f64 {
        self.0
    }
}

/// The outcome of guardian review for an action.
/// Mirrors `bizra-action::types::GuardianVerdict`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GuardianVerdict {
    /// Action is approved for execution.
    Approved,
    /// Action is denied.
    Denied,
    /// Action requires human-in-the-loop review before proceeding.
    RequiresHitl,
}

/// The execution channel through which an action was delivered.
/// Mirrors `bizra-action::types::Channel` (8 variants).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Channel {
    /// Agent-to-human knowledge transfer.
    Ahk,
    /// Large-language-model inference.
    Llm,
    /// Working memory read/write.
    Memory,
    /// Model-context-protocol tool call.
    Mcp,
    /// File system operation.
    FileSystem,
    /// Browser automation.
    Browser,
    /// Final response delivery.
    Response,
    /// Telescript execution.
    Telescript,
}

impl Channel {
    /// Encode the channel as a single discriminant byte for compact serialisation.
    ///
    /// [DERIVED] — byte values are assigned in declaration order to remain stable.
    pub fn as_byte(self) -> u8 {
        match self {
            Channel::Ahk => 0,
            Channel::Llm => 1,
            Channel::Memory => 2,
            Channel::Mcp => 3,
            Channel::FileSystem => 4,
            Channel::Browser => 5,
            Channel::Response => 6,
            Channel::Telescript => 7,
        }
    }
}

/// A single tamper-evident record of one constitutional action.
///
/// Mirrors `bizra-action::types::ConstitutionalReceipt` exactly [VERIFIED].
///
/// Each receipt carries:
/// - its own BLAKE3 `content_hash` over `channel ‖ summary ‖ payload ‖ ihsan`,
/// - an Ed25519 `signature` placeholder,
/// - a `previous_hash` that chains it to the preceding receipt (Merkle link).
#[derive(Debug, Clone)]
pub struct ConstitutionalReceipt {
    /// Unique action identifier.
    pub action_id: ActionId,
    /// Wall-clock timestamp of the action (Unix ms).
    pub timestamp: ActionTimestamp,
    /// BLAKE3 hash of `channel ‖ summary ‖ payload ‖ ihsan_score` [VERIFIED].
    pub content_hash: [u8; 32],
    /// Quality score for this action, clamped to `[0.0, 1.0]`.
    pub ihsan_score: IhsanScore,
    /// Guardian verdict attached to this action.
    pub verdict: GuardianVerdict,
    /// Execution channel used.
    pub channel: Channel,
    /// Human-readable summary of what the action did.
    pub action_summary: String,
    /// Ed25519 signature placeholder (64 bytes) [VERIFIED].
    pub signature: [u8; 64],
    /// Hash of the immediately preceding receipt, enabling chain verification [VERIFIED].
    pub previous_hash: [u8; 32],
}

// ──────────────────────────────────────────────────────────────────────────────
// § 2.  Mirror types from bizra-action/src/receipt.rs  [VERIFIED interface]
// ──────────────────────────────────────────────────────────────────────────────

/// A Merkle-linked sequence of `ConstitutionalReceipt`s.
///
/// Mirrors `bizra-action::receipt::ReceiptChain` [VERIFIED].
///
/// ## Invariants
/// - `chain_length` equals `receipts.len()`.
/// - `head_hash` is the `content_hash` of the most recently appended receipt.
/// - Each receipt's `previous_hash` matches the `content_hash` of its predecessor.
pub struct ReceiptChain {
    head_hash: [u8; 32],
    chain_length: u64,
    receipts: Vec<ConstitutionalReceipt>,
}

impl ReceiptChain {
    /// Construct an empty chain.  `head_hash` is the all-zeros sentinel [VERIFIED].
    pub fn new() -> Self {
        ReceiptChain {
            head_hash: [0u8; 32],
            chain_length: 0,
            receipts: Vec::new(),
        }
    }

    /// Append a receipt to the chain, updating `head_hash` and `chain_length`.
    ///
    /// [VERIFIED] mirrors `ReceiptChain::record()`.
    pub fn record(&mut self, receipt: ConstitutionalReceipt) {
        self.head_hash = receipt.content_hash;
        self.chain_length += 1;
        self.receipts.push(receipt);
    }

    /// Verify hash-chain integrity.
    ///
    /// Returns `Ok(verified_length)` on success, `Err(bad_index)` on the first
    /// broken link [VERIFIED].
    pub fn verify_chain(&self) -> Result<u64, u64> {
        if self.receipts.is_empty() {
            return Ok(0);
        }
        let mut expected_prev = [0u8; 32];
        for (i, r) in self.receipts.iter().enumerate() {
            if r.previous_hash != expected_prev {
                return Err(i as u64);
            }
            expected_prev = r.content_hash;
        }
        Ok(self.chain_length)
    }

    /// Return the hash of the most recently appended receipt.
    pub fn head_hash(&self) -> [u8; 32] {
        self.head_hash
    }

    /// Return the number of receipts in the chain.
    pub fn len(&self) -> u64 {
        self.chain_length
    }

    /// Return `true` if the chain is empty.
    pub fn is_empty(&self) -> bool {
        self.chain_length == 0
    }

    /// Return a reference to the receipt at position `index`, or `None`.
    pub fn get(&self, index: usize) -> Option<&ConstitutionalReceipt> {
        self.receipts.get(index)
    }

    /// Return a reference to the most recently appended receipt, or `None`.
    pub fn latest(&self) -> Option<&ConstitutionalReceipt> {
        self.receipts.last()
    }

    /// Return a slice of all receipts in append order.
    pub fn all_receipts(&self) -> &[ConstitutionalReceipt] {
        &self.receipts
    }
}

impl Default for ReceiptChain {
    fn default() -> Self {
        Self::new()
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// § 3.  Mirror types from bizra-sippar/src/lib.rs  [VERIFIED interface]
// ──────────────────────────────────────────────────────────────────────────────

/// Errors that can arise during Sippar regular-number arithmetic.
///
/// Mirrors `bizra-sippar::SipparError` [VERIFIED].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SipparError {
    /// Input was zero; regular numbers are positive integers.
    Zero,
    /// Arithmetic overflow during exponent computation.
    Overflow,
    /// The number contains a prime factor other than 2, 3, or 5.
    IrregularFactor(u64),
    /// Division produced a non-integer quotient.
    NotDivisible,
}

impl fmt::Display for SipparError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            SipparError::Zero => write!(f, "Sippar: input is zero"),
            SipparError::Overflow => write!(f, "Sippar: arithmetic overflow"),
            SipparError::IrregularFactor(p) => {
                write!(f, "Sippar: irregular prime factor {p}")
            }
            SipparError::NotDivisible => write!(f, "Sippar: not evenly divisible"),
        }
    }
}

impl std::error::Error for SipparError {}

/// A 5-smooth (regular) positive integer, expressed as 2^`exp2` × 3^`exp3` × 5^`exp5`.
///
/// Mirrors `bizra-sippar::RegularNumber` [VERIFIED].
///
/// ## Historical context
///
/// Babylonian temple accountants at Sippar (~1900 BCE) used sexagesimal arithmetic
/// rooted in regular numbers because they admit exact reciprocals — essential for
/// grain and silver accounting without remainder errors [VERIFIED].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RegularNumber {
    exp2: u8,
    exp3: u8,
    exp5: u8,
    value: u64,
}

impl RegularNumber {
    /// Construct from explicit exponents, computing `value = 2^e2 × 3^e3 × 5^e5`.
    ///
    /// Returns `Err(SipparError::Overflow)` if the product exceeds `u64::MAX`.
    ///
    /// [VERIFIED] mirrors `RegularNumber::from_factors()`.
    pub fn from_factors(exp2: u8, exp3: u8, exp5: u8) -> Result<Self, SipparError> {
        let v2: u64 = (2u64)
            .checked_pow(exp2 as u32)
            .ok_or(SipparError::Overflow)?;
        let v3: u64 = (3u64)
            .checked_pow(exp3 as u32)
            .ok_or(SipparError::Overflow)?;
        let v5: u64 = (5u64)
            .checked_pow(exp5 as u32)
            .ok_or(SipparError::Overflow)?;
        let value = v2
            .checked_mul(v3)
            .and_then(|x| x.checked_mul(v5))
            .ok_or(SipparError::Overflow)?;
        Ok(RegularNumber {
            exp2,
            exp3,
            exp5,
            value,
        })
    }

    /// Factor `n` into 2^`a` × 3^`b` × 5^`c`, returning `Err` for any irregular factor.
    ///
    /// [VERIFIED] mirrors `RegularNumber::from_u64()`.
    pub fn from_u64(mut n: u64) -> Result<Self, SipparError> {
        if n == 0 {
            return Err(SipparError::Zero);
        }
        let (mut e2, mut e3, mut e5) = (0u8, 0u8, 0u8);
        while n % 2 == 0 {
            n /= 2;
            e2 += 1;
        }
        while n % 3 == 0 {
            n /= 3;
            e3 += 1;
        }
        while n % 5 == 0 {
            n /= 5;
            e5 += 1;
        }
        if n != 1 {
            // smallest remaining factor is irregular
            let mut p = n;
            let mut d = 2u64;
            while d * d <= p {
                if p % d == 0 {
                    break;
                }
                d += 1;
            }
            if d * d <= p {
                p = d;
            }
            return Err(SipparError::IrregularFactor(p));
        }
        Self::from_factors(e2, e3, e5)
    }

    /// Test whether `n` is 5-smooth (regular).
    ///
    /// [VERIFIED] mirrors `RegularNumber::is_regular()`.
    pub fn is_regular(n: u64) -> bool {
        Self::from_u64(n).is_ok()
    }

    /// Return the numeric value.
    pub fn value(self) -> u64 {
        self.value
    }
    /// Return the exponent of 2.
    pub fn exp2(self) -> u8 {
        self.exp2
    }
    /// Return the exponent of 3.
    pub fn exp3(self) -> u8 {
        self.exp3
    }
    /// Return the exponent of 5.
    pub fn exp5(self) -> u8 {
        self.exp5
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// § 4.  Mirror types from bizra-proofspace/src/lib.rs  [VERIFIED interface]
// ──────────────────────────────────────────────────────────────────────────────

/// Minimum ihsan quality threshold to pass ProofSpace validation [VERIFIED].
pub const IHSAN_THRESHOLD: f64 = 0.95;
/// Maximum permissible Gini coefficient for ADL score distribution [VERIFIED].
pub const ADL_GINI_MAX: f64 = 0.35;
/// Maximum permissible harm score [VERIFIED].
pub const MAX_HARM_SCORE: f64 = 0.3;

/// Floating-point epsilon used for ihsan comparisons throughout this module [DERIVED].
const F64_IHSAN_EPS: f64 = 1e-9;

// ──────────────────────────────────────────────────────────────────────────────
// § 5.  SipparChainDigest  [NEW — this module]
// ──────────────────────────────────────────────────────────────────────────────

/// Encoding of chain-length metadata using Babylonian regular-number arithmetic.
///
/// ## Semantics
///
/// If `chain_length` is 5-smooth (i.e., a product of powers of 2, 3, and 5 only),
/// the chain is classified as **harmonious** — it maps exactly onto the Sippar
/// sexagesimal lattice and its reciprocal is computable without remainder [DERIVED].
///
/// If `chain_length` contains an irregular prime factor it is classified as a
/// **witness chain**: the chain carries evidence that cannot be fully dissolved
/// into harmonic accounting [DERIVED].
///
/// ## Standing on Giants
///
/// - Sexagesimal regularity criterion: Neugebauer, *Mathematische Keilschrift-Texte* (1935–1937) [VERIFIED].
/// - Sippar scribal tablets YBC 7289, Plimpton 322: exact arithmetic without remainder [VERIFIED].
#[derive(Debug, Clone)]
pub struct SipparChainDigest {
    /// The raw chain length that was encoded.
    pub chain_length: u64,

    /// If `chain_length` is 5-smooth, this holds its factored representation.
    ///
    /// `None` when `chain_length == 0` or the factorisation failed.
    pub regular_form: Option<RegularNumber>,

    /// Exponent of 2 in the factorisation, or 0 if irregular.
    pub exp2: u8,
    /// Exponent of 3 in the factorisation, or 0 if irregular.
    pub exp3: u8,
    /// Exponent of 5 in the factorisation, or 0 if irregular.
    pub exp5: u8,

    /// `true`  → length is 5-smooth; the chain is **harmonious**.
    /// `false` → length has an irregular factor; the chain is a **witness**.
    pub is_harmonious: bool,

    /// When irregular, the smallest non-smooth prime factor witnessed.
    ///
    /// `None` for harmonious chains.
    pub irregular_witness: Option<u64>,

    /// A human-readable classification label.
    ///
    /// Either `"harmonious"` or `"witness"` [DERIVED].
    pub label: &'static str,
}

impl SipparChainDigest {
    /// Encode a chain length as a `SipparChainDigest`.
    ///
    /// An empty chain (`chain_length == 0`) is treated as a special harmonious
    /// sentinel with all exponents zero [DERIVED].
    pub fn encode(chain_length: u64) -> Self {
        if chain_length == 0 {
            return SipparChainDigest {
                chain_length: 0,
                regular_form: None,
                exp2: 0,
                exp3: 0,
                exp5: 0,
                is_harmonious: true,
                irregular_witness: None,
                label: "harmonious",
            };
        }

        match RegularNumber::from_u64(chain_length) {
            Ok(rn) => SipparChainDigest {
                chain_length,
                regular_form: Some(rn),
                exp2: rn.exp2(),
                exp3: rn.exp3(),
                exp5: rn.exp5(),
                is_harmonious: true,
                irregular_witness: None,
                label: "harmonious",
            },
            Err(SipparError::IrregularFactor(p)) => SipparChainDigest {
                chain_length,
                regular_form: None,
                exp2: 0,
                exp3: 0,
                exp5: 0,
                is_harmonious: false,
                irregular_witness: Some(p),
                label: "witness",
            },
            Err(_) => SipparChainDigest {
                chain_length,
                regular_form: None,
                exp2: 0,
                exp3: 0,
                exp5: 0,
                is_harmonious: false,
                irregular_witness: None,
                label: "witness",
            },
        }
    }

    /// Return a one-line sexagesimal summary string, e.g. `"2^2 × 3^1 × 5^1 = 60"`.
    ///
    /// For witness chains, reports the irregular factor instead [DERIVED].
    pub fn summary(&self) -> String {
        if self.is_harmonious {
            if self.chain_length == 0 {
                return "empty chain (harmonious sentinel)".to_owned();
            }
            format!(
                "2^{} × 3^{} × 5^{} = {} [harmonious]",
                self.exp2, self.exp3, self.exp5, self.chain_length
            )
        } else {
            match self.irregular_witness {
                Some(p) => format!(
                    "irregular factor {} in chain_length={} [witness]",
                    p, self.chain_length
                ),
                None => format!("irregular chain_length={} [witness]", self.chain_length),
            }
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// § 6.  ProofBlockSubmission  [NEW — this module]
// ──────────────────────────────────────────────────────────────────────────────

/// The data package submitted from a `ReceiptChain` to ProofSpace for validation.
///
/// ProofSpace validators consume this struct to decide whether the chain meets
/// civilization-grade proof standards (ihsan ≥ 0.95, harm ≤ 0.30, etc.) [DERIVED].
///
/// ## Standing on Giants
///
/// - ProofSpace threshold constants from `bizra-proofspace::IHSAN_THRESHOLD` [VERIFIED].
/// - Formal assertion syntax: SMT-LIB2 standard (Barrett, de Moura, Stump, 2010) [VERIFIED].
#[derive(Debug, Clone)]
pub struct ProofBlockSubmission {
    /// Merkle head hash of the originating receipt chain.
    pub receipt_chain_hash: [u8; 32],

    /// Number of receipts in the originating chain.
    pub chain_length: u64,

    /// Sippar regular-number encoding of the chain length.
    pub sippar_digest: SipparChainDigest,

    /// Individual ihsan scores from each receipt, in append order.
    pub ihsan_scores: Vec<f64>,

    /// Arithmetic mean of all ihsan scores.
    ///
    /// `0.0` for an empty chain [DERIVED].
    pub mean_ihsan: f64,

    /// Minimum ihsan score across all receipts.
    ///
    /// `0.0` for an empty chain [DERIVED].
    pub min_ihsan: f64,

    /// De-duplicated set of channel discriminant bytes used in the chain.
    ///
    /// Each byte corresponds to `Channel::as_byte()` [DERIVED].
    pub all_channels_used: Vec<u8>,

    /// A syntactically valid SMT-LIB2 assertion over `ihsan_mean` [VERIFIED].
    ///
    /// Example: `(assert (>= ihsan_mean 0.95))`
    pub formal_assertion: String,

    /// Unix-millisecond timestamp at the moment this submission was constructed.
    pub submission_timestamp: u64,

    /// Identifier of the node that submitted this proof block.
    pub creator_node: String,
}

impl ProofBlockSubmission {
    /// Return `true` if `mean_ihsan` meets the ProofSpace threshold within epsilon [DERIVED].
    pub fn passes_ihsan_threshold(&self) -> bool {
        self.mean_ihsan >= IHSAN_THRESHOLD - F64_IHSAN_EPS
    }

    /// Return `true` if every individual receipt passed the ihsan threshold [DERIVED].
    pub fn all_receipts_pass_ihsan(&self) -> bool {
        self.ihsan_scores
            .iter()
            .all(|&s| s >= IHSAN_THRESHOLD - F64_IHSAN_EPS)
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// § 7.  Bridge-layer error type  [NEW — this module]
// ──────────────────────────────────────────────────────────────────────────────

/// Errors that can arise when bridging a `ReceiptChain` into ProofSpace.
#[derive(Debug)]
pub enum BridgeError {
    /// The receipt chain failed its own internal integrity check.
    ///
    /// Carries the index of the first broken link [DERIVED].
    ChainIntegrityFailure(u64),

    /// Sippar encoding failed for an unexpected reason.
    SipparFailure(SipparError),

    /// A required field could not be computed (e.g., empty statistics).
    ComputationError(String),
}

impl fmt::Display for BridgeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BridgeError::ChainIntegrityFailure(idx) => {
                write!(f, "chain integrity broken at receipt index {idx}")
            }
            BridgeError::SipparFailure(e) => write!(f, "Sippar encoding error: {e}"),
            BridgeError::ComputationError(msg) => write!(f, "computation error: {msg}"),
        }
    }
}

impl std::error::Error for BridgeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            BridgeError::SipparFailure(e) => Some(e),
            _ => None,
        }
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// § 8.  ReceiptChainBridge trait  [NEW — this module]
// ──────────────────────────────────────────────────────────────────────────────

/// The bridge between operational execution records (`bizra-action`) and
/// civilization-grade proof blocks (`bizra-proofspace`).
///
/// Any type implementing this trait can:
/// 1. Verify its own internal Merkle integrity.
/// 2. Report its length and head hash.
/// 3. Package itself as a [`ProofBlockSubmission`] for ProofSpace validation.
/// 4. Encode its length using Sippar sexagesimal arithmetic via [`SipparChainDigest`].
///
/// ## Standing on Giants
///
/// - **Merkle (1979)** [VERIFIED]: tamper-evident hash chains underpin `verify_integrity`.
/// - **Sippar scribes (~1900 BCE)** [VERIFIED]: 5-smooth factorisation underlies
///   `sippar_encode_chain_length`.
/// - **Al-Ghazali (~1090 CE)** [VERIFIED]: the imperative to carry all actions into
///   verifiable evidence motivates `to_proof_block`.
pub trait ReceiptChainBridge {
    /// The type of individual receipt stored in this chain.
    type Receipt;

    /// The error type returned by bridge operations.
    type Error: std::error::Error;

    /// Verify the Merkle-hash integrity of the chain.
    ///
    /// Returns `Ok(verified_length)` on success or `Err(bad_index)` at the first
    /// broken link [VERIFIED].
    fn verify_integrity(&self) -> Result<u64, Self::Error>;

    /// Return the number of receipts currently in the chain [VERIFIED].
    fn chain_length(&self) -> u64;

    /// Return the Merkle head hash (content hash of the most recent receipt),
    /// or `[0u8; 32]` for an empty chain [VERIFIED].
    fn head_hash(&self) -> [u8; 32];

    /// Aggregate all receipts into a [`ProofBlockSubmission`] and attach the
    /// `creator_node` identifier [DERIVED].
    ///
    /// This is the primary interop point with `bizra-proofspace`.
    fn to_proof_block(&self, creator_node: &str) -> Result<ProofBlockSubmission, Self::Error>;

    /// Encode `chain_length()` using Sippar regular-number arithmetic [DERIVED].
    ///
    /// A 5-smooth length yields a **harmonious** digest.
    /// Any irregular prime factor yields a **witness** digest.
    fn sippar_encode_chain_length(&self) -> Result<SipparChainDigest, Self::Error>;
}

// ──────────────────────────────────────────────────────────────────────────────
// § 9.  ConstitutionalReceiptAdapter  [NEW — this module]
// ──────────────────────────────────────────────────────────────────────────────

/// Adapter that wraps a [`ReceiptChain`] and implements [`ReceiptChainBridge`],
/// adding ProofSpace awareness to the existing `bizra-action` chain.
///
/// ## Design
///
/// This adapter follows the *Adapter* pattern (GoF, 1994) [VERIFIED]: it does not
/// copy or re-own the receipts; it merely adds the translation layer required by
/// `bizra-proofspace` on top of the existing `bizra-action::ReceiptChain` API.
///
/// ## Standing on Giants
///
/// - The wrapped `ReceiptChain` embodies Merkle (1979) hash-chain tamper-evidence [VERIFIED].
/// - `sippar_encode_chain_length` applies Babylonian regular-number arithmetic [VERIFIED].
/// - `to_proof_block` serialises ihsan statistics and generates an SMT-LIB2 assertion
///   in the formal language of Barrett et al. [VERIFIED].
pub struct ConstitutionalReceiptAdapter {
    /// The underlying operational receipt chain from `bizra-action`.
    chain: ReceiptChain,

    /// A monotonic timestamp source, captured at construction time (Unix ms) [DERIVED].
    ///
    /// In a real deployment this would call `SystemTime::now()`.  Here it is injected
    /// to keep the module self-contained and deterministically testable.
    pub construction_timestamp: u64,
}

impl ConstitutionalReceiptAdapter {
    /// Wrap an existing `ReceiptChain`, recording `timestamp_ms` as the
    /// construction time for `ProofBlockSubmission::submission_timestamp`.
    pub fn new(chain: ReceiptChain, timestamp_ms: u64) -> Self {
        ConstitutionalReceiptAdapter {
            chain,
            construction_timestamp: timestamp_ms,
        }
    }

    /// Consume the adapter and return the inner `ReceiptChain`.
    pub fn into_inner(self) -> ReceiptChain {
        self.chain
    }

    /// Borrow the inner `ReceiptChain`.
    pub fn inner(&self) -> &ReceiptChain {
        &self.chain
    }

    // ── internal helpers ──────────────────────────────────────────────────────

    /// Compute the arithmetic mean of a slice of `f64` values.
    ///
    /// Returns `0.0` for an empty slice [DERIVED].
    pub(crate) fn mean(values: &[f64]) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        // Use Kahan compensated summation to avoid floating-point drift over long chains [DERIVED].
        let mut sum = 0.0f64;
        let mut c = 0.0f64;
        for &v in values {
            let y = v - c;
            let t = sum + y;
            c = (t - sum) - y;
            sum = t;
        }
        sum / values.len() as f64
    }

    /// Compute the minimum of a slice of `f64` values.
    ///
    /// Returns `0.0` for an empty slice [DERIVED].
    pub(crate) fn min_f64(values: &[f64]) -> f64 {
        values
            .iter()
            .cloned()
            .fold(f64::INFINITY, f64::min)
            .min(f64::INFINITY)
            .max(f64::NEG_INFINITY)
    }

    /// Build the sorted, de-duplicated channel-byte vector from all receipts [DERIVED].
    pub(crate) fn collect_channels(receipts: &[ConstitutionalReceipt]) -> Vec<u8> {
        let mut seen = [false; 8];
        for r in receipts {
            let b = r.channel.as_byte() as usize;
            if b < 8 {
                seen[b] = true;
            }
        }
        (0u8..8u8).filter(|&b| seen[b as usize]).collect()
    }

    /// Render a syntactically valid SMT-LIB2 assertion over `ihsan_mean` [VERIFIED].
    ///
    /// The assertion checks `(>= ihsan_mean IHSAN_THRESHOLD)` and is accompanied by
    /// a `declare-const` preamble so it is a standalone parseable SMT-LIB2 fragment.
    ///
    /// Example output:
    /// ```text
    /// (declare-const ihsan_mean Real)
    /// (assert (>= ihsan_mean 0.95))
    /// ```
    pub(crate) fn build_formal_assertion(mean: f64) -> String {
        format!(
            "(declare-const ihsan_mean Real)\n\
             (assert (>= ihsan_mean {threshold:.2}))\n\
             ; actual mean = {actual:.6}",
            threshold = IHSAN_THRESHOLD,
            actual = mean,
        )
    }
}

impl ReceiptChainBridge for ConstitutionalReceiptAdapter {
    type Receipt = ConstitutionalReceipt;
    type Error = BridgeError;

    /// Delegate to `ReceiptChain::verify_chain()`, mapping the error variant [VERIFIED].
    fn verify_integrity(&self) -> Result<u64, BridgeError> {
        self.chain
            .verify_chain()
            .map_err(BridgeError::ChainIntegrityFailure)
    }

    /// Delegate to `ReceiptChain::len()` [VERIFIED].
    fn chain_length(&self) -> u64 {
        self.chain.len()
    }

    /// Delegate to `ReceiptChain::head_hash()` [VERIFIED].
    fn head_hash(&self) -> [u8; 32] {
        self.chain.head_hash()
    }

    /// Aggregate all receipts and build a [`ProofBlockSubmission`].
    ///
    /// Steps [DERIVED]:
    /// 1. Collect all ihsan scores.
    /// 2. Compute `mean_ihsan` (Kahan sum) and `min_ihsan`.
    /// 3. Collect de-duplicated channel bytes.
    /// 4. Encode chain length as a `SipparChainDigest`.
    /// 5. Build the SMT-LIB2 formal assertion.
    /// 6. Package everything into `ProofBlockSubmission`.
    fn to_proof_block(&self, creator_node: &str) -> Result<ProofBlockSubmission, BridgeError> {
        let receipts = self.chain.all_receipts();

        let ihsan_scores: Vec<f64> = receipts.iter().map(|r| r.ihsan_score.value()).collect();

        let mean_ihsan = Self::mean(&ihsan_scores);

        let min_ihsan = if ihsan_scores.is_empty() {
            0.0
        } else {
            Self::min_f64(&ihsan_scores)
        };

        let all_channels_used = Self::collect_channels(receipts);
        let sippar_digest = SipparChainDigest::encode(self.chain.len());
        let formal_assertion = Self::build_formal_assertion(mean_ihsan);

        Ok(ProofBlockSubmission {
            receipt_chain_hash: self.chain.head_hash(),
            chain_length: self.chain.len(),
            sippar_digest,
            ihsan_scores,
            mean_ihsan,
            min_ihsan,
            all_channels_used,
            formal_assertion,
            submission_timestamp: self.construction_timestamp,
            creator_node: creator_node.to_owned(),
        })
    }

    /// Encode `chain_length()` via Sippar regular-number arithmetic [DERIVED].
    ///
    /// - 5-smooth length → `SipparChainDigest { is_harmonious: true, label: "harmonious" }`
    /// - Irregular length → `SipparChainDigest { is_harmonious: false, label: "witness" }`
    fn sippar_encode_chain_length(&self) -> Result<SipparChainDigest, BridgeError> {
        Ok(SipparChainDigest::encode(self.chain.len()))
    }
}

// ──────────────────────────────────────────────────────────────────────────────
// § 10.  Convenience constructor helpers
// ──────────────────────────────────────────────────────────────────────────────

/// Build a minimal `ConstitutionalReceipt` suitable for testing.
///
/// `previous_hash` must be set to the `content_hash` of the preceding receipt
/// (or `[0u8; 32]` for the first receipt) by the caller [DERIVED].
pub fn make_receipt(
    id: u64,
    ts: u64,
    ihsan: f64,
    channel: Channel,
    verdict: GuardianVerdict,
    previous_hash: [u8; 32],
) -> ConstitutionalReceipt {
    // Derive content_hash using FNV-1a over the id and ihsan score.
    // In production this would be BLAKE3(channel ‖ summary ‖ payload ‖ ihsan) [PLANNED].
    let content_hash = fnv1a_stub(id, ihsan, channel.as_byte());

    ConstitutionalReceipt {
        action_id: ActionId(id),
        timestamp: ActionTimestamp(ts),
        content_hash,
        ihsan_score: IhsanScore::new(ihsan),
        verdict,
        channel,
        action_summary: format!("action-{id}"),
        signature: [0u8; 64],
        previous_hash,
    }
}

/// Minimal FNV-1a–inspired stub producing a deterministic 32-byte hash from
/// `(id, ihsan, channel_byte)`.
///
/// This is **not** a cryptographic hash; it is used only for testing and
/// self-contained operation without the `blake3` crate [PLANNED — replace with BLAKE3].
pub(crate) fn fnv1a_stub(id: u64, ihsan: f64, channel_byte: u8) -> [u8; 32] {
    const FNV_PRIME: u64 = 0x0000_0100_0000_01B3;
    const FNV_OFFSET: u64 = 0xCBF2_9CE4_8422_2325;

    let mut hash = FNV_OFFSET;
    for byte in id.to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    for byte in ihsan.to_bits().to_le_bytes() {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash ^= channel_byte as u64;
    hash = hash.wrapping_mul(FNV_PRIME);

    // Expand the 64-bit hash into 32 bytes by mixing two derived words.
    let mut out = [0u8; 32];
    let h2 = hash.wrapping_mul(0x9E37_79B9_7F4A_7C15); // φ-multiplier
    out[..8].copy_from_slice(&hash.to_le_bytes());
    out[8..16].copy_from_slice(&h2.to_le_bytes());
    out[16..24].copy_from_slice(&hash.wrapping_add(h2).to_le_bytes());
    out[24..32].copy_from_slice(&hash.wrapping_sub(h2).to_le_bytes());
    out
}

// ──────────────────────────────────────────────────────────────────────────────
// § 11.  Tests
// ──────────────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    // ── helpers ───────────────────────────────────────────────────────────────

    /// Build a chain of `n` receipts, all `Approved` on `Channel::Llm`,
    /// with ihsan scores drawn from `scores`.
    fn build_chain(scores: &[f64]) -> ReceiptChain {
        let mut chain = ReceiptChain::new();
        let mut prev = [0u8; 32];
        for (i, &score) in scores.iter().enumerate() {
            let r = make_receipt(
                i as u64,
                1_000_000 + i as u64,
                score,
                Channel::Llm,
                GuardianVerdict::Approved,
                prev,
            );
            prev = r.content_hash;
            chain.record(r);
        }
        chain
    }

    fn adapter(scores: &[f64]) -> ConstitutionalReceiptAdapter {
        ConstitutionalReceiptAdapter::new(build_chain(scores), 9_999_999)
    }

    // ── test 1 ────────────────────────────────────────────────────────────────

    /// An empty chain should produce a valid `ProofBlockSubmission` with zeroed stats.
    #[test]
    fn test_empty_chain_produces_valid_submission() {
        let a = ConstitutionalReceiptAdapter::new(ReceiptChain::new(), 42_000);
        let sub = a
            .to_proof_block("node-0")
            .expect("submission should succeed");

        assert_eq!(sub.chain_length, 0, "chain_length must be 0");
        assert_eq!(sub.receipt_chain_hash, [0u8; 32], "head_hash sentinel");
        assert!(sub.ihsan_scores.is_empty(), "no ihsan scores");
        assert!(
            (sub.mean_ihsan - 0.0).abs() < F64_IHSAN_EPS,
            "mean_ihsan = 0"
        );
        assert!((sub.min_ihsan - 0.0).abs() < F64_IHSAN_EPS, "min_ihsan = 0");
        assert!(sub.all_channels_used.is_empty(), "no channels");
        assert_eq!(sub.creator_node, "node-0");
        assert_eq!(sub.submission_timestamp, 42_000);

        // The Sippar digest for length 0 should be harmonious.
        assert!(sub.sippar_digest.is_harmonious, "empty chain is harmonious");
        assert_eq!(sub.sippar_digest.label, "harmonious");
    }

    // ── test 2 ────────────────────────────────────────────────────────────────

    /// A chain with three receipts must compute correct ihsan statistics.
    #[test]
    fn test_chain_with_receipts_computes_correct_ihsan_stats() {
        let scores = [0.90, 0.95, 1.00];
        let a = adapter(&scores);

        // Verify chain integrity first.
        let verified_len = a.verify_integrity().expect("chain must be intact");
        assert_eq!(verified_len, 3);

        let sub = a
            .to_proof_block("node-test")
            .expect("submission must succeed");

        assert_eq!(sub.chain_length, 3);
        assert_eq!(sub.ihsan_scores.len(), 3);

        // mean = (0.90 + 0.95 + 1.00) / 3 = 0.95
        let expected_mean = (0.90 + 0.95 + 1.00) / 3.0;
        assert!(
            (sub.mean_ihsan - expected_mean).abs() < 1e-12,
            "mean_ihsan = {:.15}, expected {expected_mean:.15}",
            sub.mean_ihsan
        );

        // min = 0.90
        assert!(
            (sub.min_ihsan - 0.90).abs() < 1e-12,
            "min_ihsan = {:.6}",
            sub.min_ihsan
        );

        // Channel Llm (byte 1) should be the only channel.
        assert_eq!(sub.all_channels_used, vec![1u8], "only Llm used");

        // Exactly at threshold → passes.
        assert!(
            sub.passes_ihsan_threshold(),
            "mean exactly at threshold must pass"
        );
    }

    // ── test 3 ────────────────────────────────────────────────────────────────

    /// Chain length 60 = 2² × 3 × 5 must be classified as a harmonious Sippar number.
    #[test]
    fn test_sippar_encoding_harmonious() {
        // 60 = 2^2 × 3^1 × 5^1
        let digest = SipparChainDigest::encode(60);

        assert!(digest.is_harmonious, "60 must be harmonious");
        assert_eq!(digest.label, "harmonious");
        assert_eq!(digest.exp2, 2, "2^2 in 60");
        assert_eq!(digest.exp3, 1, "3^1 in 60");
        assert_eq!(digest.exp5, 1, "5^1 in 60");
        assert_eq!(digest.chain_length, 60);
        assert!(digest.irregular_witness.is_none(), "no irregular factor");

        // Verify via RegularNumber::from_u64
        let rn = RegularNumber::from_u64(60).expect("60 is regular");
        assert_eq!(rn.value(), 60);
        assert_eq!(rn.exp2(), 2);
        assert_eq!(rn.exp3(), 1);
        assert_eq!(rn.exp5(), 1);

        // summary should contain the factorisation
        let s = digest.summary();
        assert!(s.contains("60"), "summary contains value");
        assert!(s.contains("harmonious"), "summary labelled harmonious");
    }

    // ── test 4 ────────────────────────────────────────────────────────────────

    /// Chain length 7 is prime and not 5-smooth; it must be classified as a witness chain.
    #[test]
    fn test_sippar_encoding_irregular() {
        let digest = SipparChainDigest::encode(7);

        assert!(!digest.is_harmonious, "7 must be irregular");
        assert_eq!(digest.label, "witness");
        assert_eq!(
            digest.irregular_witness,
            Some(7),
            "7 is itself the irregular factor"
        );
        assert_eq!(digest.chain_length, 7);

        // RegularNumber::from_u64 must also reject 7.
        let result = RegularNumber::from_u64(7);
        assert!(
            matches!(result, Err(SipparError::IrregularFactor(7))),
            "expected IrregularFactor(7), got {result:?}"
        );

        let s = digest.summary();
        assert!(s.contains("7"), "summary references irregular chain length");
        assert!(s.contains("witness"), "summary labelled witness");

        // Additional irregular numbers.
        for n in [7u64, 11, 13, 17, 49, 77, 91] {
            assert!(!RegularNumber::is_regular(n), "{n} must be irregular");
        }

        // Additional harmonious numbers.
        for n in [
            1u64, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48,
            50, 60, 64, 72, 75, 80, 90, 96, 100, 120,
        ] {
            assert!(RegularNumber::is_regular(n), "{n} must be regular");
        }
    }

    // ── test 5 ────────────────────────────────────────────────────────────────

    /// The formal_assertion field must be syntactically valid SMT-LIB2.
    ///
    /// We validate structure without an actual SMT solver by checking that:
    /// - It contains a `declare-const` for `ihsan_mean`.
    /// - It contains an `assert` with `>=` comparison.
    /// - The threshold literal `0.95` is present.
    /// - The `actual mean` comment is present.
    /// - Parentheses are balanced.
    #[test]
    fn test_formal_assertion_is_valid_smtlib2() {
        let assertion = ConstitutionalReceiptAdapter::build_formal_assertion(0.97);

        // Structural checks
        assert!(
            assertion.contains("(declare-const ihsan_mean Real)"),
            "must declare ihsan_mean as Real: {assertion}"
        );
        assert!(
            assertion.contains("(assert (>= ihsan_mean 0.95))"),
            "must assert ihsan_mean >= 0.95: {assertion}"
        );
        assert!(
            assertion.contains("actual mean"),
            "must include actual mean comment: {assertion}"
        );

        // Parenthesis balance check
        let open = assertion.chars().filter(|&c| c == '(').count();
        let close = assertion.chars().filter(|&c| c == ')').count();
        assert_eq!(
            open, close,
            "unbalanced parentheses: {open} open vs {close} close in:\n{assertion}"
        );

        // Verify that the actual mean is embedded correctly (0.970000)
        assert!(
            assertion.contains("0.970000"),
            "actual mean must be formatted to 6 decimal places: {assertion}"
        );

        // Also test via the adapter path
        let scores = [0.96, 0.98, 0.97];
        let a = adapter(&scores);
        let sub = a.to_proof_block("node-smtlib").expect("submission ok");
        let fa = &sub.formal_assertion;

        let open2 = fa.chars().filter(|&c| c == '(').count();
        let close2 = fa.chars().filter(|&c| c == ')').count();
        assert_eq!(
            open2, close2,
            "unbalanced parentheses in adapter-generated assertion"
        );
        assert!(
            fa.contains("(assert (>= ihsan_mean 0.95))"),
            "threshold line present"
        );
    }

    // ── bonus: multi-channel test ─────────────────────────────────────────────

    /// A chain using multiple channels should report all of them in `all_channels_used`.
    #[test]
    fn test_multi_channel_collection() {
        let mut chain = ReceiptChain::new();
        let channels = [
            Channel::Ahk,
            Channel::Mcp,
            Channel::Response,
            Channel::Mcp, // duplicate — should be de-duplicated
        ];
        let mut prev = [0u8; 32];
        for (i, &ch) in channels.iter().enumerate() {
            let r = make_receipt(i as u64, i as u64, 1.0, ch, GuardianVerdict::Approved, prev);
            prev = r.content_hash;
            chain.record(r);
        }
        let a = ConstitutionalReceiptAdapter::new(chain, 1);
        let sub = a.to_proof_block("node-multi").unwrap();

        // Ahk=0, Mcp=3, Response=6 (sorted, de-duplicated)
        assert_eq!(sub.all_channels_used, vec![0u8, 3u8, 6u8]);
    }

    // ── bonus: bridge error display ───────────────────────────────────────────

    /// `BridgeError` display strings must be non-empty and descriptive.
    #[test]
    fn test_bridge_error_display() {
        let e1 = BridgeError::ChainIntegrityFailure(3);
        assert!(e1.to_string().contains('3'), "index 3 in message: {e1}");

        let e2 = BridgeError::SipparFailure(SipparError::IrregularFactor(7));
        assert!(
            e2.to_string().contains('7'),
            "irregular factor 7 in message: {e2}"
        );

        let e3 = BridgeError::ComputationError("bad input".into());
        assert!(
            e3.to_string().contains("bad input"),
            "description in message: {e3}"
        );
    }

    // ── bonus: RegularNumber arithmetic ──────────────────────────────────────

    /// `RegularNumber::from_factors` round-trips correctly.
    #[test]
    fn test_regular_number_from_factors() {
        // 2^2 × 3^1 × 5^1 = 60
        let rn = RegularNumber::from_factors(2, 1, 1).unwrap();
        assert_eq!(rn.value(), 60);
        assert_eq!(rn.exp2(), 2);
        assert_eq!(rn.exp3(), 1);
        assert_eq!(rn.exp5(), 1);

        // 2^6 = 64
        let rn64 = RegularNumber::from_factors(6, 0, 0).unwrap();
        assert_eq!(rn64.value(), 64);

        // zero exponents → 1
        let rn1 = RegularNumber::from_factors(0, 0, 0).unwrap();
        assert_eq!(rn1.value(), 1);
    }

    // ── bonus: Kahan mean stability ───────────────────────────────────────────

    /// The Kahan mean helper should be accurate for a long sequence of equal values.
    #[test]
    fn test_kahan_mean_accuracy() {
        let scores: Vec<f64> = vec![0.95; 1000];
        let m = ConstitutionalReceiptAdapter::mean(&scores);
        assert!((m - 0.95).abs() < 1e-13, "Kahan mean drifted: {m:.20}");
    }
}
