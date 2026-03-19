//! # Attestation — SAT's Constitutional Verdict
//!
//! After a ProofCarryingRequest crosses the trust boundary and
//! passes verification, SAT produces an Attestation.
//!
//! The Attestation is:
//! - Counter-signed by the SAT agent (independent of PAT)
//! - The artifact that triggers SEED minting
//! - The proof that the network can verify
//! - Immutable once issued
//!
//! ```text
//! PAT signs request → crosses boundary → SAT verifies →
//! SAT counter-signs → Attestation → SEED mint → proof chain
//! ```

use blake3::Hasher;
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};

use crate::boundary::ProofCarryingRequest;
use crate::{DOMAIN_PREFIX, PROTOCOL_VERSION};

// =============================================================================
// TYPES
// =============================================================================

/// The output of SAT validation — a counter-signed attestation.
///
/// This is the two-party proof: PAT signed the work, SAT signed the verdict.
/// Together they prove: a human's agent did work, and the constitution approved it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Attestation {
    /// Unique attestation ID (BLAKE3 hash)
    pub attestation_id: String,
    /// The original request ID from the PAT side
    pub request_id: String,
    /// The node that originated the work
    pub origin_node_id: String,
    /// PAT agent that did the work
    pub pat_agent_id: String,
    /// PAT's signature (from the ProofCarryingRequest)
    pub pat_signature: String,
    /// SAT agent that validated
    pub sat_agent_id: String,
    /// SAT agent's public key (hex)
    pub sat_public_key_hex: String,
    /// SAT's independent verdict
    pub verdict: SatVerdict,
    /// Ihsān score (verified independently by SAT)
    pub verified_ihsan: f64,
    /// SEED amount to mint (0 if rejected)
    pub seed_mint_amount: u64,
    /// BLAKE3 hash of the complete attestation content
    pub attestation_hash: String,
    /// SAT's Ed25519 counter-signature over the attestation hash
    pub sat_signature: String,
    /// Timestamp
    pub attested_at: u64,
    /// Protocol version
    pub protocol_version: String,
}

/// SAT's verdict — independent of PAT's claim
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SatVerdict {
    /// Constitution approves. SEED mints. Proof chains.
    Approved,
    /// Constitution rejects. No SEED. Halt recorded.
    /// This is NOT a failure — it's the system working correctly.
    /// (The Mint Court's first rejection at SNR 0.577 was proof of governance.)
    Rejected,
    /// SAT needs more evidence before ruling.
    Deferred,
}

/// Errors during attestation
#[derive(Debug, Clone, thiserror::Error)]
pub enum AttestationError {
    #[error("SAT rejected: {reason}")]
    Rejected { reason: String },

    #[error("Signing failed: {0}")]
    SigningError(String),
}

// =============================================================================
// ATTESTATION CREATION (SAT side)
// =============================================================================

/// SAT creates an attestation after verifying a ProofCarryingRequest.
///
/// This is the counter-signature that completes the two-party proof.
/// SAT is independent — it does NOT take instructions from the node.
/// It evaluates the proof and issues its constitutional verdict.
pub fn create_attestation(
    request: &ProofCarryingRequest,
    sat_agent_id: &str,
    sat_signing_key: &SigningKey,
    verdict: SatVerdict,
    verified_ihsan: f64,
    seed_mint_amount: u64,
) -> Result<Attestation, AttestationError> {
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .expect("time")
        .as_secs();

    let sat_verifying = sat_signing_key.verifying_key();
    let sat_public_key_hex = hex_encode(sat_verifying.as_bytes());

    // Build canonical attestation content for hashing
    let content = serde_json::json!({
        "request_id": &request.request_id,
        "origin_node_id": &request.origin_node_id,
        "pat_agent_id": &request.pat_agent_id,
        "pat_signature": &request.signature,
        "sat_agent_id": sat_agent_id,
        "sat_public_key_hex": &sat_public_key_hex,
        "verdict": format!("{:?}", verdict),
        "verified_ihsan": verified_ihsan,
        "seed_mint_amount": seed_mint_amount,
        "timestamp": now,
        "protocol_version": PROTOCOL_VERSION,
    });
    let content_bytes = serde_json::to_vec(&content).expect("json");

    // BLAKE3 hash of attestation content
    let attestation_hash = domain_hash(&content_bytes);

    // Unique attestation ID
    let id_input = format!("{}:{}:{}", &request.request_id, sat_agent_id, now);
    let attestation_id = domain_hash(id_input.as_bytes());

    // SAT's counter-signature
    let digest = domain_hash(&content_bytes);
    let signature = sat_signing_key.sign(digest.as_bytes());

    Ok(Attestation {
        attestation_id,
        request_id: request.request_id.clone(),
        origin_node_id: request.origin_node_id.clone(),
        pat_agent_id: request.pat_agent_id.clone(),
        pat_signature: request.signature.clone(),
        sat_agent_id: sat_agent_id.to_string(),
        sat_public_key_hex,
        verdict,
        verified_ihsan,
        seed_mint_amount,
        attestation_hash,
        sat_signature: hex_encode(&signature.to_bytes()),
        attested_at: now,
        protocol_version: PROTOCOL_VERSION.to_string(),
    })
}

/// Verify an attestation's SAT signature.
///
/// Any node in the network can verify this — that's the point.
/// The attestation is a public proof that constitutional governance happened.
pub fn verify_attestation(attestation: &Attestation) -> Result<(), AttestationError> {
    let pk_bytes = hex_decode(&attestation.sat_public_key_hex)
        .map_err(|_| AttestationError::SigningError("invalid SAT public key hex".into()))?;
    let pk_array: [u8; 32] = pk_bytes
        .try_into()
        .map_err(|_| AttestationError::SigningError("SAT key wrong length".into()))?;
    let verifying_key = VerifyingKey::from_bytes(&pk_array)
        .map_err(|e| AttestationError::SigningError(format!("invalid SAT key: {e}")))?;

    // Reconstruct canonical content
    let content = serde_json::json!({
        "request_id": &attestation.request_id,
        "origin_node_id": &attestation.origin_node_id,
        "pat_agent_id": &attestation.pat_agent_id,
        "pat_signature": &attestation.pat_signature,
        "sat_agent_id": &attestation.sat_agent_id,
        "sat_public_key_hex": &attestation.sat_public_key_hex,
        "verdict": format!("{:?}", attestation.verdict),
        "verified_ihsan": attestation.verified_ihsan,
        "seed_mint_amount": attestation.seed_mint_amount,
        "timestamp": attestation.attested_at,
        "protocol_version": PROTOCOL_VERSION,
    });
    let content_bytes = serde_json::to_vec(&content).expect("json");
    let digest = domain_hash(&content_bytes);

    let sig_bytes = hex_decode(&attestation.sat_signature)
        .map_err(|_| AttestationError::SigningError("invalid signature hex".into()))?;
    let sig_array: [u8; 64] = sig_bytes
        .try_into()
        .map_err(|_| AttestationError::SigningError("signature wrong length".into()))?;
    let signature = Signature::from_bytes(&sig_array);

    verifying_key
        .verify(digest.as_bytes(), &signature)
        .map_err(|e| AttestationError::SigningError(format!("verification failed: {e}")))?;

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
    use crate::boundary::{GuardianVerdict, PermitLink, RequestBuilder};
    use crate::constitution::{PAT_DERIVATION_PREFIX, SAT_DERIVATION_PREFIX};
    use crate::mint::derive_agent_key;

    fn full_round_trip() -> (ProofCarryingRequest, SigningKey) {
        let master = [77u8; 32];
        let pat_key = derive_agent_key(&master, PAT_DERIVATION_PREFIX, 0);
        let permit = PermitLink {
            grantor_id: "human".into(),
            grantee_id: "p1-analyst".into(),
            capabilities: vec!["execute".into()],
            grantor_signature: "stub".into(),
        };

        let request = RequestBuilder::new(
            "node-test".into(),
            "agent-p1".into(),
            "output-hash-abc".into(),
            "analysis".into(),
        )
        .ihsan_score(0.97)
        .guardian_verdict(GuardianVerdict::all_pass())
        .permit_chain(vec![permit])
        .build_and_sign(&pat_key)
        .expect("valid request");

        let sat_key = derive_agent_key(&master, SAT_DERIVATION_PREFIX, 0);
        (request, sat_key)
    }

    #[test]
    fn test_attestation_approved() {
        let (request, sat_key) = full_round_trip();
        let attestation = create_attestation(
            &request,
            "s1-auditor",
            &sat_key,
            SatVerdict::Approved,
            0.97,
            100,
        )
        .expect("attestation");

        assert_eq!(attestation.verdict, SatVerdict::Approved);
        assert_eq!(attestation.seed_mint_amount, 100);
        assert_eq!(attestation.request_id, request.request_id);
    }

    #[test]
    fn test_attestation_rejected() {
        let (request, sat_key) = full_round_trip();
        let attestation = create_attestation(
            &request,
            "s1-auditor",
            &sat_key,
            SatVerdict::Rejected,
            0.97,
            0,
        )
        .expect("attestation");

        assert_eq!(attestation.verdict, SatVerdict::Rejected);
        assert_eq!(attestation.seed_mint_amount, 0);
    }

    #[test]
    fn test_attestation_signature_verifies() {
        let (request, sat_key) = full_round_trip();
        let attestation = create_attestation(
            &request,
            "s1-auditor",
            &sat_key,
            SatVerdict::Approved,
            0.97,
            100,
        )
        .expect("attestation");

        let verify = verify_attestation(&attestation);
        assert!(verify.is_ok(), "valid attestation must verify");
    }

    #[test]
    fn test_tampered_attestation_fails() {
        let (request, sat_key) = full_round_trip();
        let mut attestation = create_attestation(
            &request,
            "s1-auditor",
            &sat_key,
            SatVerdict::Approved,
            0.97,
            100,
        )
        .expect("attestation");

        // Tamper: change seed amount after signing
        attestation.seed_mint_amount = 999999;

        let verify = verify_attestation(&attestation);
        assert!(verify.is_err(), "tampered attestation must fail");
    }

    #[test]
    fn test_two_party_proof_complete() {
        // This is the full circuit:
        // 1. PAT signs the request
        // 2. Request crosses boundary
        // 3. SAT counter-signs attestation
        // 4. Any node verifies both signatures
        let (request, sat_key) = full_round_trip();

        // SAT verifies the boundary crossing
        let boundary_check = crate::boundary::verify_boundary_crossing(&request);
        assert!(boundary_check.is_ok(), "boundary crossing must pass");

        // SAT creates attestation
        let attestation = create_attestation(
            &request,
            "s1-auditor",
            &sat_key,
            SatVerdict::Approved,
            0.97,
            100,
        )
        .expect("attestation");

        // Any node verifies the attestation
        let verify = verify_attestation(&attestation);
        assert!(verify.is_ok(), "attestation must verify");

        // The attestation contains BOTH signatures
        assert!(
            !attestation.pat_signature.is_empty(),
            "PAT signature present"
        );
        assert!(
            !attestation.sat_signature.is_empty(),
            "SAT counter-signature present"
        );

        // Two-party proof is complete:
        // PAT proves "I did this work"
        // SAT proves "the constitution approved it"
    }
}
