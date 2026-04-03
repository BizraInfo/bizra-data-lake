// src/pci/gates.rs - PCI Protocol Verification Gate Chain
//
// Status: FROZEN — Changes require version bump + test vector update
// Semantics: First failure terminates chain (fail-closed)

use chrono::{DateTime, Utc};
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::{Duration, Instant};

use super::envelope::{validate_envelope_schema, validate_nonce, PCIEnvelope};
use super::reject_codes::*;
use super::types::*;

/// Result of a single gate execution
#[derive(Debug, Clone)]
pub struct GateResult {
    pub gate: Gate,
    pub passed: bool,
    pub latency_ms: f64,
    pub rejection: Option<RejectionResponse>,
}

/// Thread-safe nonce cache for replay protection with proper TTL
/// SECURITY FIX: Now uses timestamp-based expiration instead of full clear
pub struct NonceCache {
    cache: Mutex<HashMap<String, Instant>>,
    ttl: Duration,
    max_size: usize,
}

impl NonceCache {
    pub fn new(ttl_seconds: i64, max_size: usize) -> Self {
        Self {
            cache: Mutex::new(HashMap::new()),
            ttl: Duration::from_secs(ttl_seconds as u64),
            max_size,
        }
    }

    /// Check if nonce is new (not seen or expired) and add it
    /// SECURITY FIX: Properly expires old entries instead of full clear
    pub fn check_and_add(&self, nonce: &str) -> bool {
        let mut cache = self.cache.lock().unwrap();
        let now = Instant::now();

        // First, clean up expired entries (LRU with TTL)
        if cache.len() >= self.max_size {
            // Remove all expired entries first
            cache.retain(|_, inserted| now.duration_since(*inserted) < self.ttl);

            // If still too large, remove oldest 25%
            if cache.len() >= self.max_size {
                let mut entries: Vec<_> = cache.iter().map(|(k, v)| (k.clone(), *v)).collect();
                entries.sort_by_key(|(_, v)| *v);
                let to_remove = entries.len() / 4;
                for (key, _) in entries.into_iter().take(to_remove) {
                    cache.remove(&key);
                }
            }
        }

        // Check if nonce exists and is not expired
        if let Some(inserted) = cache.get(nonce) {
            if now.duration_since(*inserted) < self.ttl {
                return false; // Replay detected - nonce is still valid
            }
            // Nonce expired, will be replaced below
        }

        // Add new nonce with current timestamp
        cache.insert(nonce.to_string(), now);
        true // New nonce (or expired one replaced)
    }

    pub fn clear(&self) {
        let mut cache = self.cache.lock().unwrap();
        cache.clear();
    }
}

impl Default for NonceCache {
    fn default() -> Self {
        Self::new(TIMESTAMP_SKEW_SECONDS, 100_000)
    }
}

lazy_static::lazy_static! {
    static ref NONCE_CACHE: NonceCache = NonceCache::default();
}

/// Verification Gate Chain
pub struct GateChain {
    current_policy_hash: String,
    current_state_hash: String,
    ihsan_threshold: f64,
    snr_threshold: f64,
    gates_passed: Vec<Gate>,
    total_latency_ms: f64,
}

impl GateChain {
    pub fn new(
        current_policy_hash: String,
        current_state_hash: String,
        ihsan_threshold: f64,
        snr_threshold: f64,
    ) -> Self {
        Self {
            current_policy_hash,
            current_state_hash,
            ihsan_threshold,
            snr_threshold,
            gates_passed: Vec::new(),
            total_latency_ms: 0.0,
        }
    }

    /// Execute the full gate chain
    pub fn verify(
        &mut self,
        envelope: &PCIEnvelope,
        require_expensive: bool,
    ) -> (bool, Option<RejectionResponse>, Vec<GateResult>) {
        let digest = envelope.compute_digest();
        let timestamp = utc_now_iso();
        let mut results = Vec::new();

        // CHEAP TIER (<10ms)
        let cheap_start = Instant::now();

        // SCHEMA gate
        let schema_result = self.gate_schema(envelope, &digest, &timestamp);
        results.push(schema_result.clone());
        if !schema_result.passed {
            return (false, schema_result.rejection, results);
        }
        self.gates_passed.push(Gate::Schema);

        // SIGNATURE gate
        let sig_result = self.gate_signature(envelope, &digest, &timestamp);
        results.push(sig_result.clone());
        if !sig_result.passed {
            return (false, sig_result.rejection, results);
        }
        self.gates_passed.push(Gate::Signature);

        // TIMESTAMP gate
        let ts_result = self.gate_timestamp(envelope, &digest, &timestamp);
        results.push(ts_result.clone());
        if !ts_result.passed {
            return (false, ts_result.rejection, results);
        }
        self.gates_passed.push(Gate::Timestamp);

        // REPLAY gate
        let replay_result = self.gate_replay(envelope, &digest, &timestamp);
        results.push(replay_result.clone());
        if !replay_result.passed {
            return (false, replay_result.rejection, results);
        }
        self.gates_passed.push(Gate::Replay);

        // ROLE gate
        let role_result = self.gate_role(envelope, &digest, &timestamp);
        results.push(role_result.clone());
        if !role_result.passed {
            return (false, role_result.rejection, results);
        }
        self.gates_passed.push(Gate::Role);

        let cheap_elapsed = cheap_start.elapsed().as_secs_f64() * 1000.0;
        if cheap_elapsed > LATENCY_BUDGET_CHEAP_MS as f64 {
            let rejection = RejectionResponse::rejection(
                RejectCode::RejectBudgetExceeded,
                format!(
                    "CHEAP tier exceeded budget: {:.1}ms > {}ms",
                    cheap_elapsed, LATENCY_BUDGET_CHEAP_MS
                ),
                digest.clone(),
                timestamp.clone(),
                None,
            );
            return (false, Some(rejection), results);
        }

        // MEDIUM TIER (<150ms)
        let medium_start = Instant::now();

        // SNR gate
        let snr_result = self.gate_snr(envelope, &digest, &timestamp);
        results.push(snr_result.clone());
        if !snr_result.passed {
            return (false, snr_result.rejection, results);
        }
        self.gates_passed.push(Gate::Snr);

        // IHSAN gate
        let ihsan_result = self.gate_ihsan(envelope, &digest, &timestamp);
        results.push(ihsan_result.clone());
        if !ihsan_result.passed {
            return (false, ihsan_result.rejection, results);
        }
        self.gates_passed.push(Gate::Ihsan);

        // POLICY gate
        let policy_result = self.gate_policy(envelope, &digest, &timestamp);
        results.push(policy_result.clone());
        if !policy_result.passed {
            return (false, policy_result.rejection, results);
        }
        self.gates_passed.push(Gate::Policy);

        let medium_elapsed = medium_start.elapsed().as_secs_f64() * 1000.0;
        if medium_elapsed > LATENCY_BUDGET_MEDIUM_MS as f64 {
            let rejection = RejectionResponse::rejection(
                RejectCode::RejectBudgetExceeded,
                format!(
                    "MEDIUM tier exceeded budget: {:.1}ms > {}ms",
                    medium_elapsed, LATENCY_BUDGET_MEDIUM_MS
                ),
                digest.clone(),
                timestamp.clone(),
                None,
            );
            return (false, Some(rejection), results);
        }

        // EXPENSIVE TIER (<2000ms) - only if required
        if require_expensive {
            let expensive_start = Instant::now();

            // FATE gate (placeholder - pass by default)
            let fate_result = GateResult {
                gate: Gate::Fate,
                passed: true,
                latency_ms: 0.0,
                rejection: None,
            };
            results.push(fate_result);
            self.gates_passed.push(Gate::Fate);

            // FORMAL gate (placeholder - pass by default)
            let formal_result = GateResult {
                gate: Gate::Formal,
                passed: true,
                latency_ms: 0.0,
                rejection: None,
            };
            results.push(formal_result);
            self.gates_passed.push(Gate::Formal);

            let expensive_elapsed = expensive_start.elapsed().as_secs_f64() * 1000.0;
            if expensive_elapsed > LATENCY_BUDGET_EXPENSIVE_MS as f64 {
                let rejection = RejectionResponse::rejection(
                    RejectCode::RejectBudgetExceeded,
                    format!(
                        "EXPENSIVE tier exceeded budget: {:.1}ms > {}ms",
                        expensive_elapsed, LATENCY_BUDGET_EXPENSIVE_MS
                    ),
                    digest.clone(),
                    timestamp.clone(),
                    None,
                );
                return (false, Some(rejection), results);
            }
        }

        self.total_latency_ms = results.iter().map(|r| r.latency_ms).sum();
        (true, None, results)
    }

    pub fn get_gates_passed(&self) -> &[Gate] {
        &self.gates_passed
    }

    pub fn get_total_latency_ms(&self) -> f64 {
        self.total_latency_ms
    }

    // =========================================================================
    // Individual Gate Implementations
    // =========================================================================

    fn gate_schema(&self, envelope: &PCIEnvelope, digest: &str, timestamp: &str) -> GateResult {
        let start = Instant::now();
        let errors = validate_envelope_schema(envelope);
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        if errors.is_empty() {
            GateResult {
                gate: Gate::Schema,
                passed: true,
                latency_ms,
                rejection: None,
            }
        } else {
            GateResult {
                gate: Gate::Schema,
                passed: false,
                latency_ms,
                rejection: Some(reject_schema(digest, timestamp, &errors.join("; "))),
            }
        }
    }

    fn gate_signature(&self, envelope: &PCIEnvelope, digest: &str, timestamp: &str) -> GateResult {
        let start = Instant::now();
        let valid = envelope.verify_signature();
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        if valid {
            GateResult {
                gate: Gate::Signature,
                passed: true,
                latency_ms,
                rejection: None,
            }
        } else {
            GateResult {
                gate: Gate::Signature,
                passed: false,
                latency_ms,
                rejection: Some(reject_signature(digest, timestamp)),
            }
        }
    }

    fn gate_timestamp(&self, envelope: &PCIEnvelope, digest: &str, timestamp: &str) -> GateResult {
        let start = Instant::now();

        // Parse envelope timestamp
        let envelope_dt = match DateTime::parse_from_rfc3339(&envelope.timestamp) {
            Ok(dt) => dt.with_timezone(&Utc),
            Err(e) => {
                let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
                return GateResult {
                    gate: Gate::Timestamp,
                    passed: false,
                    latency_ms,
                    rejection: Some(reject_schema(
                        digest,
                        timestamp,
                        &format!("Invalid timestamp: {}", e),
                    )),
                };
            }
        };

        let now = Utc::now();
        let skew_seconds = (now - envelope_dt).num_seconds();
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        if skew_seconds > TIMESTAMP_SKEW_SECONDS {
            return GateResult {
                gate: Gate::Timestamp,
                passed: false,
                latency_ms,
                rejection: Some(reject_timestamp_stale(
                    digest,
                    timestamp,
                    &envelope.timestamp,
                    skew_seconds as f64,
                )),
            };
        }

        if skew_seconds < -TIMESTAMP_SKEW_SECONDS {
            return GateResult {
                gate: Gate::Timestamp,
                passed: false,
                latency_ms,
                rejection: Some(RejectionResponse::rejection(
                    RejectCode::RejectTimestampFuture,
                    format!(
                        "Timestamp {} is {}s in the future (max {}s)",
                        envelope.timestamp,
                        skew_seconds.abs(),
                        TIMESTAMP_SKEW_SECONDS
                    ),
                    digest.to_string(),
                    timestamp.to_string(),
                    None,
                )),
            };
        }

        GateResult {
            gate: Gate::Timestamp,
            passed: true,
            latency_ms,
            rejection: None,
        }
    }

    fn gate_replay(&self, envelope: &PCIEnvelope, digest: &str, timestamp: &str) -> GateResult {
        let start = Instant::now();

        if !validate_nonce(&envelope.nonce) {
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            return GateResult {
                gate: Gate::Replay,
                passed: false,
                latency_ms,
                rejection: Some(reject_schema(digest, timestamp, "Invalid nonce format")),
            };
        }

        let is_new = NONCE_CACHE.check_and_add(&envelope.nonce);
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        if is_new {
            GateResult {
                gate: Gate::Replay,
                passed: true,
                latency_ms,
                rejection: None,
            }
        } else {
            GateResult {
                gate: Gate::Replay,
                passed: false,
                latency_ms,
                rejection: Some(reject_replay(digest, timestamp, &envelope.nonce)),
            }
        }
    }

    fn gate_role(&self, envelope: &PCIEnvelope, digest: &str, timestamp: &str) -> GateResult {
        let start = Instant::now();

        let action = envelope.payload.action.to_lowercase();
        let agent_type = &envelope.sender.agent_type;

        let forbidden = match agent_type {
            AgentType::Pat => vec!["commit", "issue_receipt", "modify_state"],
            AgentType::Sat => vec!["propose"],
        };

        let violation = forbidden.iter().any(|f| action.contains(f));
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        if violation {
            GateResult {
                gate: Gate::Role,
                passed: false,
                latency_ms,
                rejection: Some(reject_role_violation(
                    digest,
                    timestamp,
                    agent_type.as_str(),
                    &envelope.payload.action,
                )),
            }
        } else {
            GateResult {
                gate: Gate::Role,
                passed: true,
                latency_ms,
                rejection: None,
            }
        }
    }

    fn gate_snr(&self, envelope: &PCIEnvelope, digest: &str, timestamp: &str) -> GateResult {
        let start = Instant::now();
        let snr = envelope.metadata.snr_score;
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        if snr < self.snr_threshold {
            GateResult {
                gate: Gate::Snr,
                passed: false,
                latency_ms,
                rejection: Some(reject_snr(digest, timestamp, snr, self.snr_threshold)),
            }
        } else {
            GateResult {
                gate: Gate::Snr,
                passed: true,
                latency_ms,
                rejection: None,
            }
        }
    }

    fn gate_ihsan(&self, envelope: &PCIEnvelope, digest: &str, timestamp: &str) -> GateResult {
        let start = Instant::now();
        let ihsan = envelope.metadata.ihsan_score;
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        if ihsan < self.ihsan_threshold {
            GateResult {
                gate: Gate::Ihsan,
                passed: false,
                latency_ms,
                rejection: Some(reject_ihsan(digest, timestamp, ihsan, self.ihsan_threshold)),
            }
        } else {
            GateResult {
                gate: Gate::Ihsan,
                passed: true,
                latency_ms,
                rejection: None,
            }
        }
    }

    fn gate_policy(&self, envelope: &PCIEnvelope, digest: &str, timestamp: &str) -> GateResult {
        let start = Instant::now();
        let matches = envelope.payload.policy_hash == self.current_policy_hash;
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

        if matches {
            GateResult {
                gate: Gate::Policy,
                passed: true,
                latency_ms,
                rejection: None,
            }
        } else {
            GateResult {
                gate: Gate::Policy,
                passed: false,
                latency_ms,
                rejection: Some(RejectionResponse::rejection(
                    RejectCode::RejectPolicyMismatch,
                    "policy_hash doesn't match current constitution".to_string(),
                    digest.to_string(),
                    timestamp.to_string(),
                    None,
                )),
            }
        }
    }
}

/// High-level verification API
pub fn verify_envelope(
    envelope: &PCIEnvelope,
    policy_hash: &str,
    state_hash: &str,
    ihsan_threshold: f64,
    snr_threshold: f64,
    require_expensive: bool,
) -> (bool, Option<RejectionResponse>, Vec<GateResult>) {
    let mut chain = GateChain::new(
        policy_hash.to_string(),
        state_hash.to_string(),
        ihsan_threshold,
        snr_threshold,
    );
    chain.verify(envelope, require_expensive)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pci::envelope::EnvelopeBuilder;
    use ed25519_dalek::SigningKey;

    /// Create a properly signed test envelope with Ed25519
    fn create_test_envelope(ihsan: f64, snr: f64) -> PCIEnvelope {
        // Generate a deterministic keypair for testing (seeded from fixed bytes)
        let seed: [u8; 32] = [
            0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e,
            0x0f, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1a, 0x1b, 0x1c,
            0x1d, 0x1e, 0x1f, 0x20,
        ];
        let signing_key = SigningKey::from_bytes(&seed);
        let verifying_key = signing_key.verifying_key();
        let public_key_hex = hex::encode(verifying_key.as_bytes());

        let policy_hash = "a".repeat(64);
        let state_hash = "b".repeat(64);

        let envelope = EnvelopeBuilder::new()
            .with_sender(AgentType::Pat, "pat-001", &public_key_hex)
            .with_action("propose", serde_json::json!({"task": "analyze"}))
            .with_policy(&policy_hash)
            .with_state(&state_hash)
            .with_scores(ihsan, snr)
            .build()
            .unwrap();

        // Sign the envelope
        envelope.sign(&seed).expect("Failed to sign test envelope")
    }

    #[test]
    fn test_gate_chain_passes() {
        NONCE_CACHE.clear();

        let envelope = create_test_envelope(0.97, 0.85);
        let policy_hash = "a".repeat(64);
        let state_hash = "b".repeat(64);

        let (passed, rejection, results) =
            verify_envelope(&envelope, &policy_hash, &state_hash, 0.95, 0.70, false);

        assert!(passed, "Gate chain should pass. Rejection: {:?}", rejection);
        assert!(rejection.is_none());
        assert!(!results.is_empty());
    }

    #[test]
    fn test_ihsan_gate_fails() {
        NONCE_CACHE.clear();

        let envelope = create_test_envelope(0.80, 0.85); // Below threshold
        let policy_hash = "a".repeat(64);
        let state_hash = "b".repeat(64);

        let (passed, rejection, _) =
            verify_envelope(&envelope, &policy_hash, &state_hash, 0.95, 0.70, false);

        assert!(!passed);
        assert!(rejection.is_some());
        let rej = rejection.unwrap();
        assert_eq!(rej.code, RejectCode::RejectIhsanBelowMin.as_u8());
    }

    #[test]
    fn test_snr_gate_fails() {
        NONCE_CACHE.clear();

        let envelope = create_test_envelope(0.97, 0.50); // SNR below threshold
        let policy_hash = "a".repeat(64);
        let state_hash = "b".repeat(64);

        let (passed, rejection, _) =
            verify_envelope(&envelope, &policy_hash, &state_hash, 0.95, 0.70, false);

        assert!(!passed);
        assert!(rejection.is_some());
        let rej = rejection.unwrap();
        assert_eq!(rej.code, RejectCode::RejectSnrBelowMin.as_u8());
    }

    #[test]
    fn test_nonce_replay_fails() {
        NONCE_CACHE.clear();

        let envelope = create_test_envelope(0.97, 0.85);
        let policy_hash = "a".repeat(64);
        let state_hash = "b".repeat(64);

        // First request passes
        let (passed1, _, _) =
            verify_envelope(&envelope, &policy_hash, &state_hash, 0.95, 0.70, false);
        assert!(passed1);

        // Same nonce should fail (replay)
        let (passed2, rejection, _) =
            verify_envelope(&envelope, &policy_hash, &state_hash, 0.95, 0.70, false);
        assert!(!passed2);
        assert!(rejection.is_some());
        assert_eq!(
            rejection.unwrap().code,
            RejectCode::RejectNonceReplay.as_u8()
        );
    }
}
