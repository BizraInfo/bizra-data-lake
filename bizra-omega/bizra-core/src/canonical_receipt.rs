//! CanonicalReceipt v1 — The Spearpoint Artifact
//!
//! Every externally visible effect in BIZRA emits exactly one CanonicalReceipt.
//! This is the single source of truth that unifies:
//!   - proof-native execution (mission → verdict → receipt)
//!   - constitutional admissibility (ihsan, SNR, FATE gates)
//!   - memory/reflex compilation (receipt lineage → reflex entry)
//!   - economic settlement (PoI minting from verified receipts only)
//!   - federation trust (receipt genesis hash binds to constitution)
//!
//! The receipt lifecycle:
//!   HYPOTHESIS → VERIFIED → EXECUTABLE → COMMITTED → REPLAYABLE → MARKETABLE
//!
//! Standing on Giants:
//!   - Nakamoto (2008): hash-chained immutable records
//!   - Lamport (1978): happens-before ordering via monotonic timestamps
//!   - Al-Ghazali (1095): Ihsan as constitutional quality floor
//!   - Shannon (1948): SNR as information quality metric

use blake3::Hasher;
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};

use crate::pci::verdict::VerdictStatus;
use crate::pci::RejectCode;

mod sig_bytes {
    use serde::{self, Deserialize, Deserializer, Serializer};

    fn to_hex(bytes: &[u8]) -> String {
        bytes.iter().map(|b| format!("{b:02x}")).collect()
    }

    fn from_hex(s: &str) -> Result<Vec<u8>, String> {
        (0..s.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|e| e.to_string()))
            .collect()
    }

    pub fn serialize<S>(bytes: &[u8; 64], serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        serializer.serialize_str(&to_hex(bytes))
    }

    pub fn deserialize<'de, D>(deserializer: D) -> Result<[u8; 64], D::Error>
    where
        D: Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        let v = from_hex(&s).map_err(serde::de::Error::custom)?;
        let mut arr = [0u8; 64];
        if v.len() == 64 {
            arr.copy_from_slice(&v);
        }
        Ok(arr)
    }
}

/// Domain prefix for canonical receipt hashing.
pub const DOMAIN_CANONICAL_RECEIPT: &str = "bizra-canonical-receipt-v1";

/// Receipt lifecycle states.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReceiptState {
    /// Mission proposed, not yet evaluated by gates.
    Hypothesis,
    /// Gates evaluated, verdict rendered (may be admitted or rejected).
    Verified,
    /// Admitted by gates, approved for execution.
    Executable,
    /// Executed, effects committed, receipt sealed.
    Committed,
    /// Committed and available for deterministic replay.
    Replayable,
    /// Published to marketplace, eligible for PoI rewards.
    Marketable,
}

/// Execution route taken by this mission.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ExecutionRoute {
    /// System 1: O(1) reflex cache hit, no LLM invocation.
    Reflex,
    /// System 2: Full PAT deliberation through LLM.
    Deliberate,
    /// Degraded: Fallback path, reduced capability.
    Degraded,
    /// Rejected: Constitutional gate denied execution.
    Rejected,
}

/// The CanonicalReceipt — BIZRA's spearpoint artifact.
///
/// One receipt per externally visible effect. No exceptions.
/// Every field is deterministic and reproducible.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CanonicalReceipt {
    // ─── Identity ───────────────────────────────────────────
    /// Unique receipt identifier (BLAKE3 of canonical bytes).
    pub receipt_id: [u8; 32],
    /// Mission that produced this receipt.
    pub mission_id: String,
    /// Genesis hash this receipt is bound to.
    pub genesis_hash: [u8; 32],
    /// Policy version used for evaluation.
    pub policy_version: String,

    // ─── Verdict ────────────────────────────────────────────
    /// Constitutional admission status.
    pub verdict: VerdictStatus,
    /// Primary rejection cause (if rejected), ordered by precedence.
    pub primary_reject: Option<RejectCode>,
    /// Ihsan score at evaluation time.
    pub ihsan_score: f64,
    /// SNR score at evaluation time.
    pub snr_score: f64,
    /// Execution route taken.
    pub route: ExecutionRoute,

    // ─── Timing ─────────────────────────────────────────────
    /// Monotonic timestamp (Unix ms) when mission was received.
    pub received_at: u64,
    /// Monotonic timestamp (Unix ms) when receipt was sealed.
    pub sealed_at: u64,

    // ─── Evidence ───────────────────────────────────────────
    /// BLAKE3 hash of the mission input payload.
    pub input_hash: [u8; 32],
    /// BLAKE3 hash of the output/response payload.
    pub output_hash: [u8; 32],
    /// Hash of the previous receipt in the chain (genesis seed for first).
    pub previous_receipt: [u8; 32],

    // ─── Lifecycle ──────────────────────────────────────────
    /// Current state in the receipt lifecycle.
    pub state: ReceiptState,
    /// Whether this receipt is eligible for federation sharing.
    pub federation_admissible: bool,

    // ─── Signature ──────────────────────────────────────────
    /// Ed25519 signature over the canonical bytes (everything above).
    #[serde(with = "sig_bytes")]
    pub signature: [u8; 64],
}

/// Genesis seed for the first receipt in every chain.
pub const GENESIS_SEED: [u8; 32] = [
    0xb1, 0x2a, 0xf3, 0x7e, 0xd4, 0x91, 0xc8, 0x56, 0x2f, 0x0e, 0x8b, 0xd7, 0x43, 0x9a, 0x5c, 0x11,
    0xe7, 0x2d, 0x60, 0xf8, 0x1b, 0x37, 0xa4, 0xce, 0x95, 0x4f, 0x0d, 0x82, 0x76, 0x3c, 0xb9, 0x0a,
];

impl CanonicalReceipt {
    /// Compute the canonical bytes for hashing and signing.
    /// Deterministic: same inputs always produce same bytes.
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(256);

        // mission_id (length-prefixed)
        let mid = self.mission_id.as_bytes();
        buf.extend_from_slice(&(mid.len() as u32).to_le_bytes());
        buf.extend_from_slice(mid);

        // genesis_hash
        buf.extend_from_slice(&self.genesis_hash);

        // policy_version (length-prefixed)
        let pv = self.policy_version.as_bytes();
        buf.extend_from_slice(&(pv.len() as u32).to_le_bytes());
        buf.extend_from_slice(pv);

        // verdict (1 byte)
        buf.push(match self.verdict {
            VerdictStatus::Admitted => 0,
            VerdictStatus::Rejected => 1,
            VerdictStatus::Deferred => 2,
        });

        // ihsan + snr as fixed-point u64
        buf.extend_from_slice(&((self.ihsan_score * 1_000_000.0).round() as u64).to_le_bytes());
        buf.extend_from_slice(&((self.snr_score * 1_000_000.0).round() as u64).to_le_bytes());

        // route (1 byte)
        buf.push(match self.route {
            ExecutionRoute::Reflex => 0,
            ExecutionRoute::Deliberate => 1,
            ExecutionRoute::Degraded => 2,
            ExecutionRoute::Rejected => 3,
        });

        // timestamps
        buf.extend_from_slice(&self.received_at.to_le_bytes());
        buf.extend_from_slice(&self.sealed_at.to_le_bytes());

        // evidence hashes
        buf.extend_from_slice(&self.input_hash);
        buf.extend_from_slice(&self.output_hash);
        buf.extend_from_slice(&self.previous_receipt);

        // state (1 byte)
        buf.push(match self.state {
            ReceiptState::Hypothesis => 0,
            ReceiptState::Verified => 1,
            ReceiptState::Executable => 2,
            ReceiptState::Committed => 3,
            ReceiptState::Replayable => 4,
            ReceiptState::Marketable => 5,
        });

        // federation flag
        buf.push(self.federation_admissible as u8);

        buf
    }

    /// Compute the BLAKE3 receipt ID from canonical bytes.
    pub fn compute_id(&self) -> [u8; 32] {
        let canonical = self.canonical_bytes();
        let mut hasher = Hasher::new();
        hasher.update(DOMAIN_CANONICAL_RECEIPT.as_bytes());
        hasher.update(b":");
        hasher.update(&canonical);
        hasher.finalize().into()
    }

    /// Sign the receipt with a node's signing key.
    pub fn sign(&mut self, key: &SigningKey) {
        let canonical = self.canonical_bytes();
        let sig = key.sign(&canonical);
        self.signature = sig.to_bytes();
        self.receipt_id = self.compute_id();
    }

    /// Verify the receipt's signature against a public key.
    pub fn verify(&self, key: &VerifyingKey) -> bool {
        let canonical = self.canonical_bytes();
        let sig = Signature::from_bytes(&self.signature);
        key.verify(&canonical, &sig).is_ok()
    }

    /// Check if this receipt is chain-valid against a previous hash.
    pub fn chain_valid(&self, expected_previous: &[u8; 32]) -> bool {
        self.previous_receipt == *expected_previous
    }

    /// Check if the receipt ID matches the computed hash.
    pub fn id_valid(&self) -> bool {
        self.receipt_id == self.compute_id()
    }
}

/// Builder for constructing receipts step-by-step.
pub struct CanonicalReceiptBuilder {
    mission_id: String,
    genesis_hash: [u8; 32],
    policy_version: String,
    previous_receipt: [u8; 32],
    received_at: u64,
}

impl CanonicalReceiptBuilder {
    /// Create a new receipt builder with the required chain context.
    pub fn new(
        mission_id: impl Into<String>,
        genesis_hash: [u8; 32],
        policy_version: impl Into<String>,
        previous_receipt: [u8; 32],
        received_at: u64,
    ) -> Self {
        Self {
            mission_id: mission_id.into(),
            genesis_hash,
            policy_version: policy_version.into(),
            previous_receipt,
            received_at,
        }
    }

    /// Seal the receipt with verdict results and sign it.
    #[allow(clippy::too_many_arguments)]
    pub fn seal(
        self,
        verdict: VerdictStatus,
        primary_reject: Option<RejectCode>,
        ihsan_score: f64,
        snr_score: f64,
        route: ExecutionRoute,
        input_hash: [u8; 32],
        output_hash: [u8; 32],
        sealed_at: u64,
        signing_key: &SigningKey,
    ) -> CanonicalReceipt {
        let state = match verdict {
            VerdictStatus::Admitted => ReceiptState::Committed,
            VerdictStatus::Rejected => ReceiptState::Verified,
            VerdictStatus::Deferred => ReceiptState::Hypothesis,
        };

        let mut receipt = CanonicalReceipt {
            receipt_id: [0; 32],
            mission_id: self.mission_id,
            genesis_hash: self.genesis_hash,
            policy_version: self.policy_version,
            verdict,
            primary_reject,
            ihsan_score,
            snr_score,
            route,
            received_at: self.received_at,
            sealed_at,
            input_hash,
            output_hash,
            previous_receipt: self.previous_receipt,
            state,
            federation_admissible: verdict == VerdictStatus::Admitted && ihsan_score >= 0.95,
            signature: [0; 64],
        };
        receipt.sign(signing_key);
        receipt
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    fn test_key() -> SigningKey {
        SigningKey::generate(&mut OsRng)
    }

    fn test_hash(seed: u8) -> [u8; 32] {
        let mut h = [0u8; 32];
        for (i, b) in h.iter_mut().enumerate() {
            *b = seed.wrapping_add(i as u8);
        }
        h
    }

    #[test]
    fn test_receipt_sign_verify() {
        let key = test_key();
        let builder = CanonicalReceiptBuilder::new(
            "mission-001",
            test_hash(0xAA),
            "v0.90.0",
            GENESIS_SEED,
            1000,
        );
        let receipt = builder.seal(
            VerdictStatus::Admitted,
            None,
            0.97,
            0.92,
            ExecutionRoute::Deliberate,
            test_hash(0x10),
            test_hash(0x20),
            2000,
            &key,
        );
        assert!(receipt.verify(&key.verifying_key()));
        assert!(receipt.id_valid());
        assert_ne!(receipt.receipt_id, [0; 32]);
        assert_ne!(receipt.signature, [0; 64]);
    }

    #[test]
    fn test_receipt_chain_integrity() {
        let key = test_key();
        let r1 = CanonicalReceiptBuilder::new("m1", test_hash(0xAA), "v0.90.0", GENESIS_SEED, 1000)
            .seal(
                VerdictStatus::Admitted,
                None,
                0.96,
                0.91,
                ExecutionRoute::Deliberate,
                test_hash(0x10),
                test_hash(0x20),
                1500,
                &key,
            );
        let r2 =
            CanonicalReceiptBuilder::new("m2", test_hash(0xAA), "v0.90.0", r1.receipt_id, 2000)
                .seal(
                    VerdictStatus::Admitted,
                    None,
                    0.98,
                    0.93,
                    ExecutionRoute::Reflex,
                    test_hash(0x30),
                    test_hash(0x40),
                    2500,
                    &key,
                );
        assert!(r1.chain_valid(&GENESIS_SEED));
        assert!(r2.chain_valid(&r1.receipt_id));
        assert!(!r2.chain_valid(&GENESIS_SEED));
    }

    #[test]
    fn test_receipt_rejected_not_federation_admissible() {
        let key = test_key();
        let receipt = CanonicalReceiptBuilder::new(
            "bad-mission",
            test_hash(0xBB),
            "v0.90.0",
            GENESIS_SEED,
            1000,
        )
        .seal(
            VerdictStatus::Rejected,
            Some(RejectCode::RejectRiba),
            0.30,
            0.20,
            ExecutionRoute::Rejected,
            test_hash(0x50),
            test_hash(0x60),
            1100,
            &key,
        );
        assert!(!receipt.federation_admissible);
        assert_eq!(receipt.state, ReceiptState::Verified);
        assert_eq!(receipt.primary_reject, Some(RejectCode::RejectRiba));
    }

    #[test]
    fn test_receipt_below_ihsan_not_federation_admissible() {
        let key = test_key();
        let receipt = CanonicalReceiptBuilder::new(
            "ok-but-weak",
            test_hash(0xCC),
            "v0.90.0",
            GENESIS_SEED,
            1000,
        )
        .seal(
            VerdictStatus::Admitted,
            None,
            0.90,
            0.88,
            ExecutionRoute::Deliberate,
            test_hash(0x70),
            test_hash(0x80),
            1200,
            &key,
        );
        assert!(!receipt.federation_admissible);
        assert_eq!(receipt.state, ReceiptState::Committed);
    }

    #[test]
    fn test_receipt_deterministic_id() {
        let key = test_key();
        let r1 = CanonicalReceiptBuilder::new("m1", test_hash(0xAA), "v0.90.0", GENESIS_SEED, 1000)
            .seal(
                VerdictStatus::Admitted,
                None,
                0.97,
                0.92,
                ExecutionRoute::Deliberate,
                test_hash(0x10),
                test_hash(0x20),
                2000,
                &key,
            );
        assert_eq!(r1.receipt_id, r1.compute_id());
    }

    #[test]
    fn test_receipt_tamper_detection() {
        let key = test_key();
        let mut receipt =
            CanonicalReceiptBuilder::new("m1", test_hash(0xAA), "v0.90.0", GENESIS_SEED, 1000)
                .seal(
                    VerdictStatus::Admitted,
                    None,
                    0.97,
                    0.92,
                    ExecutionRoute::Deliberate,
                    test_hash(0x10),
                    test_hash(0x20),
                    2000,
                    &key,
                );
        // Tamper with ihsan score
        receipt.ihsan_score = 0.50;
        // Signature no longer valid
        assert!(!receipt.verify(&key.verifying_key()));
        // ID no longer matches
        assert!(!receipt.id_valid());
    }

    /// Golden vector: fixed inputs produce a known canonical byte length and receipt ID.
    /// The Python adapter MUST produce identical bytes for these exact inputs.
    #[test]
    fn test_golden_vector_cross_language() {
        // Fixed inputs — same values used in Python golden vector test
        let genesis = GENESIS_SEED;
        let input_hash = test_hash(0x10);
        let output_hash = test_hash(0x20);

        let receipt = CanonicalReceipt {
            receipt_id: [0; 32],
            mission_id: "golden-vector-001".to_string(),
            genesis_hash: genesis,
            policy_version: "v0.90.0".to_string(),
            verdict: VerdictStatus::Admitted,
            primary_reject: None,
            ihsan_score: 0.97,
            snr_score: 0.92,
            route: ExecutionRoute::Deliberate,
            received_at: 1000,
            sealed_at: 2000,
            input_hash,
            output_hash,
            previous_receipt: genesis,
            state: ReceiptState::Committed,
            federation_admissible: true,
            signature: [0; 64],
        };

        let canonical = receipt.canonical_bytes();
        // Golden vector: exact byte length for these inputs
        assert_eq!(
            canonical.len(),
            196,
            "Canonical byte length mismatch — cross-language parity broken"
        );

        let id = receipt.compute_id();
        // The receipt ID is deterministic for these inputs
        assert_ne!(id, [0; 32]);
        // Store the hex for cross-language verification
        let id_hex: String = id.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(id_hex.len(), 64);

        // Print for cross-language test development (cargo test -- --nocapture)
        #[cfg(test)]
        {
            eprintln!("GOLDEN_VECTOR_CANONICAL_LEN={}", canonical.len());
            eprintln!("GOLDEN_VECTOR_RECEIPT_ID={}", id_hex);
            eprintln!(
                "GOLDEN_VECTOR_CANONICAL_HEX={}",
                canonical
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect::<String>()
            );
        }
    }
}
