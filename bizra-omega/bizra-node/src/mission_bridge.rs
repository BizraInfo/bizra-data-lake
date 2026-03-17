// bizra-node/src/mission_bridge.rs
// ============================================================
// Mission Bridge — Governed lifecycle wrapping runtime.receive()
// ============================================================
//
// The existing AgentRuntime::receive() path becomes the "Running"
// stage inside the Mission Control Plane's state machine.
// ============================================================

use bizra_agent::runtime::{AgentRuntime, RuntimeResponse};
use bizra_agent::types::{Message, MessageId};
use bizra_hooks::IhsanScore;
use bizra_mission::mission::Mission;
use bizra_mission::preflight::{self, Capability};
use bizra_mission::receipt::MissionReceipt;
use bizra_mission::state::{DegradationReason, FailureCode, MissionState};
use ed25519_dalek::SigningKey;

/// Result of a governed mission execution.
pub struct MissionResult {
    /// The runtime response (backward compatible with existing protocol)
    pub runtime_response: Option<RuntimeResponse>,
    /// Constitutional receipt (proof of governed execution)
    pub receipt: MissionReceipt,
    /// Full mission record (audit trail)
    pub mission: Mission,
}

/// Execute a message through the governed mission lifecycle.
/// `previous_receipt` chains this mission's receipt to the prior one for tamper-evident ordering.
/// `signing_key` signs the receipt with the node's Ed25519 sovereign identity.
pub fn execute_governed_mission(
    runtime: &mut AgentRuntime,
    ihsan: &IhsanScore,
    content: &str,
    timestamp: u64,
    available_models: &[String],
    previous_receipt: Option<[u8; 32]>,
    signing_key: Option<&SigningKey>,
) -> MissionResult {
    let content_hash: [u8; 32] = blake3::hash(content.as_bytes()).into();
    let mut m = Mission::new(content_hash, timestamp);
    if let Some(prev) = previous_receipt {
        m.chain_to(prev);
    }
    let t = timestamp; // base time

    // ── Sign helper — signs receipt before returning ──────
    macro_rules! sign_and_return {
        ($m:expr, $resp:expr) => {{
            let mut receipt = $m.receipt.clone().unwrap();
            if let Some(key) = signing_key {
                receipt.sign(key);
            }
            return MissionResult {
                runtime_response: $resp,
                receipt,
                mission: $m,
            };
        }};
    }

    // ── Preflight ─────────────────────────────────────────
    let pf = preflight::run_preflight(&[Capability::Chat], available_models, None);
    m.preflight = Some(pf.clone());

    if !pf.passed() {
        m.fail(FailureCode::CapabilityNotAvailable, t + 1).unwrap();
        sign_and_return!(m, None);
    }
    if let Some(model) = pf.chosen_model() {
        m.chosen_model = Some(model.to_string());
    }

    // ── Lifecycle stages ──────────────────────────────────
    // Constitutional requirement: transition errors must not be silently discarded.
    // If the state machine rejects a transition, the mission fails with a receipt.
    macro_rules! advance {
        ($m:expr, $state:expr, $ts:expr, $reason:expr) => {
            if let Err(_e) = $m.transition($state, $ts, $reason) {
                $m.fail(FailureCode::StateMachineViolation {
                    from: format!("{:?}", $m.state),
                    to: format!("{:?}", $state),
                }, $ts).unwrap_or(());
                if $m.receipt.is_none() {
                    $m.receipt = Some(MissionReceipt::from_mission(&$m, $m.previous_receipt_hash));
                }
                sign_and_return!($m, None);
            }
        };
    }

    advance!(m, MissionState::Queued, t + 1, "capacity available");
    advance!(m, MissionState::WarmingRetrieval, t + 2, "memory ready");
    advance!(m, MissionState::WarmingModel, t + 3, "model ready");
    advance!(m, MissionState::Retrieving, t + 4, "semantic search");
    advance!(m, MissionState::Routing, t + 5, "intent classified");


    // ── Running — this is where runtime.receive() lives ───
    advance!(m, MissionState::Running, t + 6, "inference started");

    let msg_seq = (timestamp % 1_000_000) as u32;
    let msg = Message::inbound(MessageId::new(msg_seq, 1), content, timestamp, *ihsan);
    let result = runtime.receive(msg, timestamp);

    // ── Scoring ───────────────────────────────────────────
    advance!(m, MissionState::Scoring, t + 7, "scoring response");
    m.ihsan_score = Some(ihsan.as_f64() as f32);
    m.guardian_approved = Some(result.guardian_approved);

    // Guardian veto → fail with receipt
    if !result.guardian_approved {
        m.fail(FailureCode::GuardianVeto, t + 8).unwrap();
        sign_and_return!(m, Some(result));
    }

    // Ihsan below constitutional floor → degrade with receipt
    if ihsan.as_f64() < bizra_core::IHSAN_THRESHOLD {
        m.degrade(vec![DegradationReason::UnscoredResponse], t + 8).unwrap();
        sign_and_return!(m, Some(result));
    }

    // ── Persisting ─────────────────────────────────────────
    advance!(m, MissionState::Persisting, t + 8, "persisting receipt");
    m.response_hash = Some(blake3::hash(result.response.content.as_str().as_bytes()).into());

    // ── Complete ──────────────────────────────────────────
    m.complete(t + 9).unwrap();

    // Sign the receipt with the node's sovereign identity
    let mut receipt = m.receipt.clone().unwrap();
    if let Some(key) = signing_key {
        receipt.sign(key);
    }

    MissionResult {
        runtime_response: Some(result),
        receipt,
        mission: m,
    }
}

/// Extract model names from the substrate resource manifest.
/// Used to populate the available_models list for preflight.
pub fn extract_model_names(manifest: &crate::substrate::ResourceManifest) -> Vec<String> {
    manifest.models.iter().map(|m| m.name.clone()).collect()
}
