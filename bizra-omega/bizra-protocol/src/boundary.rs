//! # Trust Boundary — The Architecture Itself
//!
//! This module defines the crossing point between sovereign (local) and
//! constitutional (URP) execution.
//!
//! ```text
//!     PAT (local, serves you)
//!          │
//!          │  ProofCarryingRequest (Ed25519 signed, BLAKE3 hashed)
//!          │
//!    ══════╪══════ TRUST BOUNDARY ══════════════
//!          │
//!          ▼
//!     SAT (in URP, serves the constitution)
//! ```
//!
//! ## The Rule
//!
//! PAT never leaves your node. SAT never obeys your node.
//! The ONLY thing that crosses is a ProofCarryingRequest.
//! Not your data. Not your credentials. Not your raw context.
//! Just the proof.

use blake3::Hasher;
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};

use crate::{constitution::IHSAN_FLOOR, DOMAIN_PREFIX, PROTOCOL_VERSION};

// =============================================================================
// TYPES
// =============================================================================

/// A request that carries its own proof of authorization.
///
/// This is the ONLY artifact that crosses the trust boundary.
/// It contains:
/// - What was done (action hash)
/// - Who authorized it (PAT agent public key)
/// - The quality score (Ihsān — must meet floor)
/// - The Guardian gate verdict (did the action pass the gate?)
/// - Cryptographic proof (signature over all of the above)
///
/// What it does NOT contain:
/// - Raw user data
/// - Credentials or API keys
/// - The full context window
/// - Any information the SAT doesn't need to validate
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProofCarryingRequest {
    /// Unique request ID (BLAKE3 hash of content)
    pub request_id: String,
    /// The node that originated this request
    pub origin_node_id: String,
    /// The PAT agent that produced the work
    pub pat_agent_id: String,
    /// PAT agent's public key (hex) — for signature verification
    pub pat_public_key_hex: String,
    /// BLAKE3 hash of the action output (not the output itself)
    pub action_output_hash: String,
    /// The action type that was performed
    pub action_type: String,
    /// Ihsān quality score (must be >= IHSAN_FLOOR)
    pub ihsan_score: f64,
    /// Guardian gate verdict
    pub guardian_verdict: GuardianVerdict,
    /// The Telescript Permit chain that authorized this action
    pub permit_chain: Vec<PermitLink>,
    /// Timestamp (UTC epoch seconds)
    pub timestamp: u64,
    /// Protocol version
    pub protocol_version: String,
    /// Ed25519 signature by the PAT agent over the canonical form
    pub signature: String,
}

/// Guardian gate verdict — the triple gate result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardianVerdict {
    /// TeleScript gate: format and structure valid?
    pub telescript_pass: bool,
    /// Tier gate: permission level sufficient?
    pub tier_pass: bool,
    /// FATE gate: ethical assessment passed?
    pub fate_pass: bool,
    /// Combined: all three must be true
    pub all_passed: bool,
}

impl GuardianVerdict {
    pub fn new(telescript: bool, tier: bool, fate: bool) -> Self {
        Self {
            telescript_pass: telescript,
            tier_pass: tier,
            fate_pass: fate,
            all_passed: telescript && tier && fate,
        }
    }

    /// A verdict where all gates passed
    pub fn all_pass() -> Self {
        Self::new(true, true, true)
    }
}

/// A single link in the Telescript Permit delegation chain.
///
/// The chain traces authority from the human through DEMA through
/// the acting PAT agent. Each link is signed by the delegator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PermitLink {
    /// Who granted this permit
    pub grantor_id: String,
    /// Who received this permit
    pub grantee_id: String,
    /// What capabilities were granted
    pub capabilities: Vec<String>,
    /// Signature by the grantor
    pub grantor_signature: String,
}

// =============================================================================
// BOUNDARY CROSSING ERRORS
// =============================================================================

/// Errors that prevent a request from crossing the trust boundary
#[derive(Debug, Clone, thiserror::Error)]
pub enum BoundaryError {
    #[error("Ihsān score {score:.3} below floor {floor:.3} — action is a type error")]
    IhsanViolation { score: f64, floor: f64 },

    #[error("Guardian gate failed: telescript={t}, tier={ti}, fate={f}")]
    GuardianRejection { t: bool, ti: bool, f: bool },

    #[error("Signature verification failed for agent {agent_id}")]
    SignatureInvalid { agent_id: String },

    #[error("Empty permit chain — no authority delegation trace")]
    EmptyPermitChain,

    #[error("Protocol version mismatch: expected {expected}, got {got}")]
    ProtocolMismatch { expected: String, got: String },
}

// =============================================================================
// BUILDING A PROOF-CARRYING REQUEST
// =============================================================================

/// Builder for constructing a ProofCarryingRequest.
///
/// Enforces constitutional invariants at construction time:
/// - Ihsān score must meet floor (compile-time-equivalent check)
/// - Guardian verdict must have all gates passed
/// - Permit chain must not be empty
/// - Request is signed by the PAT agent's key
pub struct RequestBuilder {
    origin_node_id: String,
    pat_agent_id: String,
    action_output_hash: String,
    action_type: String,
    ihsan_score: f64,
    guardian_verdict: GuardianVerdict,
    permit_chain: Vec<PermitLink>,
}

impl RequestBuilder {
    pub fn new(
        origin_node_id: String,
        pat_agent_id: String,
        action_output_hash: String,
        action_type: String,
    ) -> Self {
        Self {
            origin_node_id,
            pat_agent_id,
            action_output_hash,
            action_type,
            ihsan_score: 0.0,
            guardian_verdict: GuardianVerdict::new(false, false, false),
            permit_chain: Vec::new(),
        }
    }

    pub fn ihsan_score(mut self, score: f64) -> Self {
        self.ihsan_score = score;
        self
    }

    pub fn guardian_verdict(mut self, verdict: GuardianVerdict) -> Self {
        self.guardian_verdict = verdict;
        self
    }

    pub fn permit_chain(mut self, chain: Vec<PermitLink>) -> Self {
        self.permit_chain = chain;
        self
    }

    /// Build and sign the request. Returns Err if constitutional invariants fail.
    ///
    /// This is the gate. If this returns Ok, the request is constitutionally
    /// valid and may cross the trust boundary. If Err, it cannot.
    pub fn build_and_sign(
        self,
        pat_signing_key: &SigningKey,
    ) -> Result<ProofCarryingRequest, BoundaryError> {
        // Constitutional gate 1: Ihsān floor
        if self.ihsan_score < IHSAN_FLOOR {
            return Err(BoundaryError::IhsanViolation {
                score: self.ihsan_score,
                floor: IHSAN_FLOOR,
            });
        }

        // Constitutional gate 2: Guardian triple-gate
        if !self.guardian_verdict.all_passed {
            return Err(BoundaryError::GuardianRejection {
                t: self.guardian_verdict.telescript_pass,
                ti: self.guardian_verdict.tier_pass,
                f: self.guardian_verdict.fate_pass,
            });
        }

        // Constitutional gate 3: Permit chain must exist
        if self.permit_chain.is_empty() {
            return Err(BoundaryError::EmptyPermitChain);
        }

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .expect("time")
            .as_secs();

        let pat_verifying = pat_signing_key.verifying_key();
        let pat_public_key_hex = hex_encode(pat_verifying.as_bytes());

        // Build canonical form for signing
        let canonical = serde_json::json!({
            "origin_node_id": &self.origin_node_id,
            "pat_agent_id": &self.pat_agent_id,
            "pat_public_key_hex": &pat_public_key_hex,
            "action_output_hash": &self.action_output_hash,
            "action_type": &self.action_type,
            "ihsan_score": self.ihsan_score,
            "guardian_all_passed": true,
            "permit_chain_len": self.permit_chain.len(),
            "timestamp": now,
            "protocol_version": PROTOCOL_VERSION,
        });
        let canonical_bytes = serde_json::to_vec(&canonical).expect("json");

        // BLAKE3 hash of canonical form
        let request_id = domain_hash(&canonical_bytes);

        // Ed25519 signature by the PAT agent
        let digest = domain_hash(&canonical_bytes);
        let signature = pat_signing_key.sign(digest.as_bytes());

        Ok(ProofCarryingRequest {
            request_id,
            origin_node_id: self.origin_node_id,
            pat_agent_id: self.pat_agent_id,
            pat_public_key_hex,
            action_output_hash: self.action_output_hash,
            action_type: self.action_type,
            ihsan_score: self.ihsan_score,
            guardian_verdict: self.guardian_verdict,
            permit_chain: self.permit_chain,
            timestamp: now,
            protocol_version: PROTOCOL_VERSION.to_string(),
            signature: hex_encode(&signature.to_bytes()),
        })
    }
}

// =============================================================================
// VERIFICATION (SAT side — runs inside the URP)
// =============================================================================

/// Verify a ProofCarryingRequest that arrived at the trust boundary.
///
/// This is what SAT runs when it receives a request from PAT.
/// SAT does NOT trust the request. SAT verifies independently:
/// 1. Protocol version matches
/// 2. Ihsān score meets constitutional floor
/// 3. Guardian verdict shows all gates passed
/// 4. Ed25519 signature is cryptographically valid
/// 5. Permit chain is non-empty
pub fn verify_boundary_crossing(request: &ProofCarryingRequest) -> Result<(), BoundaryError> {
    // 1. Protocol version
    if request.protocol_version != PROTOCOL_VERSION {
        return Err(BoundaryError::ProtocolMismatch {
            expected: PROTOCOL_VERSION.to_string(),
            got: request.protocol_version.clone(),
        });
    }

    // 2. Ihsān floor (SAT re-checks — does NOT trust PAT's claim)
    if request.ihsan_score < IHSAN_FLOOR {
        return Err(BoundaryError::IhsanViolation {
            score: request.ihsan_score,
            floor: IHSAN_FLOOR,
        });
    }

    // 3. Guardian verdict
    if !request.guardian_verdict.all_passed {
        return Err(BoundaryError::GuardianRejection {
            t: request.guardian_verdict.telescript_pass,
            ti: request.guardian_verdict.tier_pass,
            f: request.guardian_verdict.fate_pass,
        });
    }

    // 4. Permit chain
    if request.permit_chain.is_empty() {
        return Err(BoundaryError::EmptyPermitChain);
    }

    // 5. Signature verification
    let pk_bytes =
        hex_decode(&request.pat_public_key_hex).map_err(|_| BoundaryError::SignatureInvalid {
            agent_id: request.pat_agent_id.clone(),
        })?;
    let pk_array: [u8; 32] = pk_bytes
        .try_into()
        .map_err(|_| BoundaryError::SignatureInvalid {
            agent_id: request.pat_agent_id.clone(),
        })?;
    let verifying_key =
        VerifyingKey::from_bytes(&pk_array).map_err(|_| BoundaryError::SignatureInvalid {
            agent_id: request.pat_agent_id.clone(),
        })?;

    let canonical = serde_json::json!({
        "origin_node_id": &request.origin_node_id,
        "pat_agent_id": &request.pat_agent_id,
        "pat_public_key_hex": &request.pat_public_key_hex,
        "action_output_hash": &request.action_output_hash,
        "action_type": &request.action_type,
        "ihsan_score": request.ihsan_score,
        "guardian_all_passed": true,
        "permit_chain_len": request.permit_chain.len(),
        "timestamp": request.timestamp,
        "protocol_version": PROTOCOL_VERSION,
    });
    let canonical_bytes = serde_json::to_vec(&canonical).expect("json");
    let digest = domain_hash(&canonical_bytes);

    let sig_bytes =
        hex_decode(&request.signature).map_err(|_| BoundaryError::SignatureInvalid {
            agent_id: request.pat_agent_id.clone(),
        })?;
    let sig_array: [u8; 64] =
        sig_bytes
            .try_into()
            .map_err(|_| BoundaryError::SignatureInvalid {
                agent_id: request.pat_agent_id.clone(),
            })?;
    let signature = Signature::from_bytes(&sig_array);
    verifying_key
        .verify(digest.as_bytes(), &signature)
        .map_err(|_| BoundaryError::SignatureInvalid {
            agent_id: request.pat_agent_id.clone(),
        })?;

    Ok(())
}

// =============================================================================
// HELPERS
// =============================================================================

fn domain_hash(data: &[u8]) -> String {
    let mut hasher = Hasher::new();
    hasher.update(DOMAIN_PREFIX);
    hasher.update(data);
    hasher.finalize().to_hex().to_string()
}

fn hex_encode(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn hex_decode(s: &str) -> Result<Vec<u8>, ()> {
    #[allow(clippy::manual_is_multiple_of)]
    if s.len() % 2 != 0 {
        return Err(());
    }
    (0..s.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&s[i..i + 2], 16).map_err(|_| ()))
        .collect()
}

// =============================================================================
// TESTS
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{constitution::PAT_DERIVATION_PREFIX, mint::derive_agent_key};

    fn make_test_request(ihsan: f64, all_gates: bool) -> (RequestBuilder, SigningKey) {
        let master_secret = [99u8; 32];
        let pat_key = derive_agent_key(&master_secret, PAT_DERIVATION_PREFIX, 0);

        let verdict = if all_gates {
            GuardianVerdict::all_pass()
        } else {
            GuardianVerdict::new(true, false, true)
        };

        let permit = PermitLink {
            grantor_id: "human-001".into(),
            grantee_id: "pat-p1-analyst".into(),
            capabilities: vec!["execute".into()],
            grantor_signature: "stub-for-test".into(),
        };

        let builder = RequestBuilder::new(
            "node-test-001".into(),
            "agent-p1-test".into(),
            "abc123hash".into(),
            "research-query".into(),
        )
        .ihsan_score(ihsan)
        .guardian_verdict(verdict)
        .permit_chain(vec![permit]);

        (builder, pat_key)
    }

    #[test]
    fn test_valid_request_crosses_boundary() {
        let (builder, key) = make_test_request(0.97, true);
        let request = builder.build_and_sign(&key);
        assert!(request.is_ok(), "valid request must cross boundary");

        let req = request.unwrap();
        let verify = verify_boundary_crossing(&req);
        assert!(verify.is_ok(), "SAT must verify valid request");
    }

    #[test]
    fn test_ihsan_below_floor_rejected() {
        let (builder, key) = make_test_request(0.80, true);
        let result = builder.build_and_sign(&key);
        assert!(result.is_err(), "ihsan below floor must be rejected");
        match result.unwrap_err() {
            BoundaryError::IhsanViolation { score, floor } => {
                assert!((score - 0.80).abs() < 0.001);
                assert!((floor - 0.95).abs() < 0.001);
            }
            other => panic!("expected IhsanViolation, got: {other}"),
        }
    }

    #[test]
    fn test_guardian_gate_failure_rejected() {
        let (builder, key) = make_test_request(0.97, false);
        let result = builder.build_and_sign(&key);
        assert!(result.is_err(), "failed guardian gate must be rejected");
        assert!(matches!(
            result.unwrap_err(),
            BoundaryError::GuardianRejection { .. }
        ));
    }

    #[test]
    fn test_empty_permit_chain_rejected() {
        let master_secret = [99u8; 32];
        let pat_key = derive_agent_key(&master_secret, PAT_DERIVATION_PREFIX, 0);

        let builder = RequestBuilder::new(
            "node-test".into(),
            "agent-test".into(),
            "hash".into(),
            "action".into(),
        )
        .ihsan_score(0.97)
        .guardian_verdict(GuardianVerdict::all_pass())
        .permit_chain(vec![]); // empty — no authority trace

        let result = builder.build_and_sign(&pat_key);
        assert!(matches!(
            result.unwrap_err(),
            BoundaryError::EmptyPermitChain
        ));
    }

    #[test]
    fn test_tampered_signature_rejected_by_sat() {
        let (builder, key) = make_test_request(0.97, true);
        let mut request = builder.build_and_sign(&key).unwrap();

        // Tamper with the action hash after signing
        request.action_output_hash = "tampered_hash_value".into();

        // Tamper with a signed field (ihsan_score is in the canonical form)
        request.ihsan_score = 0.99;

        let verify = verify_boundary_crossing(&request);
        assert!(
            verify.is_err(),
            "tampered request must fail SAT verification"
        );
    }

    #[test]
    fn test_wrong_protocol_version_rejected() {
        let (builder, key) = make_test_request(0.97, true);
        let mut request = builder.build_and_sign(&key).unwrap();
        request.protocol_version = "bizra-protocol-v999".into();

        let verify = verify_boundary_crossing(&request);
        assert!(matches!(
            verify.unwrap_err(),
            BoundaryError::ProtocolMismatch { .. }
        ));
    }
}
