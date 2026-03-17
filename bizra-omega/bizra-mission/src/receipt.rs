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

use crate::mission::Mission;
use crate::state::{DegradationReason, FailureCode, MissionState};
use serde::{Deserialize, Serialize};

/// The constitutional receipt. Append-only. Tamper-evident via BLAKE3 chain.
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
        }
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
}
