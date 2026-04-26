// bizra-mission/src/receipt.rs
// ============================================================
// Mission Receipt — constitutional proof of governed execution
// ============================================================
//
// "Every mission emits a receipt — including failed and
//  timed-out missions. No silent failures."
//
// "No contribution becomes value until it becomes a verified receipt."
// ============================================================

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};

use crate::{
    mission::Mission,
    state::{DegradationReason, FailureCode, MissionState},
};

/// The constitutional receipt. Append-only. Tamper-evident via BLAKE3 chain.
/// Signed by the emitting node's Ed25519 key — no claim without signed receipts.
///
/// Standing on: Bernstein (2006) Ed25519, Aumasson (2015) BLAKE3
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MissionReceipt {
    pub receipt_id: [u8; 32],
    pub mission_id: [u8; 32],
    pub final_state: MissionState,
    pub submitted_at: u64,
    pub completed_at: u64,
    pub states_traversed: Vec<MissionState>,
    pub chosen_model: Option<String>,
    pub ihsan_score: Option<f32>,
    pub snr_score: Option<f32>,
    pub guardian_approved: Option<bool>,
    pub failure_code: Option<FailureCode>,
    pub degradation_reasons: Vec<DegradationReason>,
    pub degradation_tier: u8,
    pub previous_receipt_hash: Option<[u8; 32]>,
    /// Ed25519 signature over the canonical receipt payload (excluding signature).
    /// [0u8; 64] = unsigned.
    #[serde(with = "sig_bytes")]
    pub signature: [u8; 64],
}

impl MissionReceipt {
    /// Create a receipt from a completed/failed/degraded mission.
    /// `previous` chains this receipt to the prior one (None for genesis receipt).
    pub fn from_mission(m: &Mission, previous: Option<[u8; 32]>) -> Self {
        let states_traversed: Vec<MissionState> = m.state_history.iter().map(|t| t.to).collect();

        // Compute degradation tier
        let tier = if m.failure_code.is_some() {
            4 // refused/failed
        } else {
            match m.degradation_reasons.len() {
                0 => 0,     // full
                1 => 1,     // light
                2..=3 => 2, // significant
                _ => 3,     // urp_assisted
            }
        };

        // Receipt ID = BLAKE3(mission_id + final_state + completed_at + previous_hash)
        // Including previous_hash makes the chain tamper-evident:
        // reordering or removing a receipt breaks the hash of every subsequent receipt.
        let completed = m.completed_at.unwrap_or(m.submitted_at);
        let mut hasher = blake3::Hasher::new();
        hasher.update(&m.mission_id);
        hasher.update(&[m.state as u8]);
        hasher.update(&completed.to_le_bytes());
        if let Some(prev) = &previous {
            hasher.update(prev);
        }
        let receipt_id: [u8; 32] = hasher.finalize().into();

        Self {
            receipt_id,
            mission_id: m.mission_id,
            final_state: m.state,
            submitted_at: m.submitted_at,
            completed_at: completed,
            states_traversed,
            chosen_model: m.chosen_model.clone(),
            ihsan_score: m.ihsan_score,
            snr_score: m.snr_score,
            guardian_approved: m.guardian_approved,
            failure_code: m.failure_code.clone(),
            degradation_reasons: m.degradation_reasons.clone(),
            degradation_tier: tier,
            previous_receipt_hash: previous,
            signature: [0u8; 64], // unsigned until sign() is called
        }
    }

    fn signature_payload(&self) -> Vec<u8> {
        #[derive(Serialize)]
        struct MissionReceiptSignaturePayload<'a> {
            receipt_id: &'a [u8; 32],
            mission_id: &'a [u8; 32],
            final_state: MissionState,
            submitted_at: u64,
            completed_at: u64,
            states_traversed: &'a [MissionState],
            chosen_model: Option<&'a str>,
            ihsan_score: Option<f32>,
            snr_score: Option<f32>,
            guardian_approved: Option<bool>,
            failure_code: Option<&'a FailureCode>,
            degradation_reasons: &'a [DegradationReason],
            degradation_tier: u8,
            previous_receipt_hash: Option<&'a [u8; 32]>,
        }

        let payload = MissionReceiptSignaturePayload {
            receipt_id: &self.receipt_id,
            mission_id: &self.mission_id,
            final_state: self.final_state,
            submitted_at: self.submitted_at,
            completed_at: self.completed_at,
            states_traversed: &self.states_traversed,
            chosen_model: self.chosen_model.as_deref(),
            ihsan_score: self.ihsan_score,
            snr_score: self.snr_score,
            guardian_approved: self.guardian_approved,
            failure_code: self.failure_code.as_ref(),
            degradation_reasons: &self.degradation_reasons,
            degradation_tier: self.degradation_tier,
            previous_receipt_hash: self.previous_receipt_hash.as_ref(),
        };

        serde_json::to_vec(&payload).expect("mission receipt signature payload serializes")
    }

    /// Sign this receipt with the node's Ed25519 signing key.
    /// Called exactly once, immediately after from_mission().
    /// The signed message is the canonical receipt payload, preserving the
    /// existing BLAKE3 receipt_id chain while binding the full artifact body.
    pub fn sign(&mut self, signing_key: &SigningKey) {
        let payload = self.signature_payload();
        let sig: Signature = signing_key.sign(&payload);
        self.signature = sig.to_bytes();
    }

    /// Verify this receipt's Ed25519 signature against a public key.
    pub fn verify_signature(&self, verifying_key: &VerifyingKey) -> bool {
        if self.signature == [0u8; 64] {
            return false; // unsigned receipt
        }
        let Ok(sig) = Signature::from_slice(&self.signature) else {
            return false; // malformed signature
        };
        let payload = self.signature_payload();
        verifying_key.verify(&payload, &sig).is_ok()
    }

    /// Full integrity check: hash + signature + chain link.
    pub fn verify_full(
        &self,
        verifying_key: &VerifyingKey,
        previous: Option<&MissionReceipt>,
    ) -> bool {
        if !self.verify_hash() {
            return false;
        }
        if !self.verify_signature(verifying_key) {
            return false;
        }
        if let Some(prev) = previous {
            if !self.verify_chain(prev) {
                return false;
            }
        }
        true
    }

    /// Is this receipt signed?
    pub fn is_signed(&self) -> bool {
        self.signature != [0u8; 64]
    }

    /// Receipt ID as hex string.
    pub fn id_hex(&self) -> String {
        self.receipt_id.iter().map(|b| format!("{b:02x}")).collect()
    }

    /// Is this a successful receipt?
    pub fn is_success(&self) -> bool {
        self.final_state == MissionState::Complete
    }

    /// Is this a degraded receipt?
    pub fn is_degraded(&self) -> bool {
        self.final_state == MissionState::Degraded
    }

    /// Verify the receipt hash integrity (includes chain link).
    pub fn verify_hash(&self) -> bool {
        let mut hasher = blake3::Hasher::new();
        hasher.update(&self.mission_id);
        hasher.update(&[self.final_state as u8]);
        hasher.update(&self.completed_at.to_le_bytes());
        if let Some(prev) = &self.previous_receipt_hash {
            hasher.update(prev);
        }
        let expected: [u8; 32] = hasher.finalize().into();
        expected == self.receipt_id
    }

    /// Verify this receipt chains correctly to the given previous receipt.
    pub fn verify_chain(&self, previous: &MissionReceipt) -> bool {
        self.previous_receipt_hash == Some(previous.receipt_id) && self.verify_hash()
    }

    /// Enforce that this receipt is signed before it leaves the node.
    /// Amanah (أمانة) — no claim without cryptographic attestation.
    ///
    /// Returns Ok(()) if signed, Err with reason if unsigned.
    /// Call this before emitting to chain, returning to user, or persisting.
    pub fn require_signed(&self) -> Result<(), &'static str> {
        if self.is_signed() {
            Ok(())
        } else {
            Err("receipt unsigned — Amanah violation: no claim without signed receipts")
        }
    }
}

/// Serde helper for [u8; 64] (Ed25519 signature).
/// Arrays > 32 don't auto-derive Serialize/Deserialize.
mod sig_bytes {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    pub fn serialize<S: Serializer>(bytes: &[u8; 64], s: S) -> Result<S::Ok, S::Error> {
        bytes.as_slice().serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 64], D::Error> {
        let v: Vec<u8> = Vec::deserialize(d)?;
        v.try_into()
            .map_err(|_| serde::de::Error::custom("expected 64 bytes for signature"))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::StateTransition;

    fn mission_fixture() -> Mission {
        let mut mission = Mission::new([7u8; 32], 100);
        mission.state = MissionState::Complete;
        mission.completed_at = Some(109);
        mission.state_history = vec![
            StateTransition {
                from: MissionState::Submitted,
                to: MissionState::Queued,
                at: 101,
                reason: "queued".to_string(),
            },
            StateTransition {
                from: MissionState::Queued,
                to: MissionState::Running,
                at: 105,
                reason: "running".to_string(),
            },
            StateTransition {
                from: MissionState::Running,
                to: MissionState::Complete,
                at: 109,
                reason: "complete".to_string(),
            },
        ];
        mission.chosen_model = Some("qwen2.5:3b".to_string());
        mission.ihsan_score = Some(0.96);
        mission.snr_score = Some(0.91);
        mission.guardian_approved = Some(true);
        mission
    }

    #[test]
    fn signature_verifies_expected_public_key() {
        let mission = mission_fixture();
        let mut receipt = MissionReceipt::from_mission(&mission, Some([3u8; 32]));
        let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);
        let verifying_key = signing_key.verifying_key();

        receipt.sign(&signing_key);

        assert!(receipt.is_signed());
        assert!(receipt.verify_signature(&verifying_key));
        assert!(receipt.verify_full(&verifying_key, None));
    }

    #[test]
    fn signature_changes_when_signed_payload_changes() {
        let mission = mission_fixture();
        let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);

        let mut receipt = MissionReceipt::from_mission(&mission, Some([9u8; 32]));
        let original_receipt_id = receipt.receipt_id;
        receipt.sign(&signing_key);

        let mut altered = receipt.clone();
        altered.guardian_approved = Some(false);
        altered.sign(&signing_key);

        assert_eq!(altered.receipt_id, original_receipt_id);
        assert_ne!(altered.signature, receipt.signature);
    }

    #[test]
    fn tampering_signed_field_breaks_signature_without_changing_chain_hash() {
        let mission = mission_fixture();
        let signing_key = SigningKey::generate(&mut rand::rngs::OsRng);
        let verifying_key = signing_key.verifying_key();

        let mut receipt = MissionReceipt::from_mission(&mission, Some([5u8; 32]));
        receipt.sign(&signing_key);
        assert!(receipt.verify_hash());
        assert!(receipt.verify_signature(&verifying_key));

        let mut tampered = receipt.clone();
        tampered.chosen_model = Some("phi4-mini".to_string());

        assert!(tampered.verify_hash());
        assert!(!tampered.verify_signature(&verifying_key));
        assert!(!tampered.verify_full(&verifying_key, None));
    }
}
