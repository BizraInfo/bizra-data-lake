//! Cognition Round — Reasoning Session payload substrate
//! ======================================================
//! Domain tag: bizra-reasoning-v1
//!
//! A cognition round binds a single model inference to the receipt
//! chain under `ReceiptKind::ReasoningSession (0x30)`. The payload
//! carries provenance (who / what / how served the round) and the
//! content hashes of prompt and response — never the content itself.
//!
//! Constitutional boundary (Brain Activation Spec v0.1 §8):
//!   The brain is advisory. The kernel is sovereign.
//!   This payload records *what was said*, not *what is true*.
//!   A separate `GovernanceDecision` receipt (kind 0x40) carries
//!   the kernel's verdict (PASS / BLOCKED / DEGRADED) and binds
//!   back to the round via its hash.
//!
//! Schema parity:
//!   `ProvenanceDescriptor` (and its children) is a schema-parity
//!   mirror of the type in
//!   `bizra-omega/bizra-installer/src/install_receipt.rs`.
//!   The JSON shape is frozen by a test in each crate; any change
//!   must land in both crates in the same commit.

use blake3::Hasher;
use serde::{Deserialize, Serialize};

use crate::receipts::{
    Blake3Hash, ByteReader, DecodeError, ReceiptKind, ReceiptPayload, ReceiptPayloadDecode,
};

// ─────────────────────────────────────────────────────────────
// Provenance (schema-parity mirror of bizra-installer)
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ProvenanceDescriptor {
    pub model_sha256: String,

    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_signer: Option<SignerIdentity>,

    pub provider_identity: ProviderIdentity,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SignerIdentity {
    pub key_id: String,
    pub algorithm: String,
    pub signature_hex: String,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProviderIdentity {
    CoreNone,
    LocalModel { weights_path: String },
    LocalServer { endpoint: String, vendor: String },
    RemoteApi { vendor: String },
}

// ─────────────────────────────────────────────────────────────
// Reasoning Session payload
// ─────────────────────────────────────────────────────────────

const DOMAIN_TAG: &[u8] = b"bizra-reasoning-v1:";

/// One inference round: prompt-hash + response-hash + provenance,
/// chained under `ReceiptKind::ReasoningSession`.
///
/// Latency and ihsan_claim are informational; they participate in
/// the hash (so tampering with them is detectable) but carry no
/// kernel authority.
#[derive(Clone, Debug, PartialEq)]
pub struct ReasoningSessionPayload {
    pub round_version: u32,
    pub provenance: ProvenanceDescriptor,
    pub prompt_hash: Blake3Hash,
    pub response_hash: Blake3Hash,
    pub timestamp_ns: u64,
    pub duration_ns: u64,
    /// Brain's own Ihsān self-claim in [0.0, 1.0]. The kernel verdict
    /// is separate and carried on a GovernanceDecision receipt.
    pub ihsan_claim: Option<f64>,
}

impl ReasoningSessionPayload {
    pub const CURRENT_VERSION: u32 = 1;

    pub fn new(
        provenance: ProvenanceDescriptor,
        prompt_hash: Blake3Hash,
        response_hash: Blake3Hash,
        timestamp_ns: u64,
        duration_ns: u64,
        ihsan_claim: Option<f64>,
    ) -> Self {
        Self {
            round_version: Self::CURRENT_VERSION,
            provenance,
            prompt_hash,
            response_hash,
            timestamp_ns,
            duration_ns,
            ihsan_claim,
        }
    }
}

impl ReceiptPayload for ReasoningSessionPayload {
    fn kind(&self) -> ReceiptKind {
        ReceiptKind::ReasoningSession
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(128);
        out.extend_from_slice(DOMAIN_TAG);
        out.extend_from_slice(&self.round_version.to_le_bytes());

        // Provenance: length-prefixed canonical JSON. JSON is used
        // here (rather than a hand-rolled binary layout) because
        // provenance is schema-shared with bizra-installer's JSON
        // receipts, and using the same encoding avoids a second
        // serializer surface to keep in sync. Determinism requires
        // that serde_json's default field-order for derived types
        // is itself deterministic — which it is (struct field
        // declaration order).
        let prov_json = serde_json::to_string(&self.provenance).unwrap_or_default();
        let prov_len = prov_json.len() as u32;
        out.extend_from_slice(&prov_len.to_le_bytes());
        out.extend_from_slice(prov_json.as_bytes());

        out.extend_from_slice(&self.prompt_hash);
        out.extend_from_slice(&self.response_hash);
        out.extend_from_slice(&self.timestamp_ns.to_le_bytes());
        out.extend_from_slice(&self.duration_ns.to_le_bytes());

        match self.ihsan_claim {
            Some(v) => {
                out.push(1);
                out.extend_from_slice(&v.to_le_bytes());
            }
            None => out.push(0),
        }

        out
    }

    fn hash(&self) -> Blake3Hash {
        let mut hasher = Hasher::new();
        hasher.update(&self.canonical_bytes());
        *hasher.finalize().as_bytes()
    }

    fn timestamp_ns(&self) -> u64 {
        self.timestamp_ns
    }
}

impl ReceiptPayloadDecode for ReasoningSessionPayload {
    fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, DecodeError> {
        if bytes.len() < DOMAIN_TAG.len() || !bytes.starts_with(DOMAIN_TAG) {
            return Err(DecodeError::ShortInput {
                need: DOMAIN_TAG.len(),
                got: bytes.len().min(DOMAIN_TAG.len()),
            });
        }
        let mut r = ByteReader::new(&bytes[DOMAIN_TAG.len()..]);

        let round_version = r.read_u32()?;

        let prov_bytes = r.read_length_prefixed()?;
        let prov_str =
            std::str::from_utf8(prov_bytes).map_err(|e| DecodeError::Utf8(e.to_string()))?;
        let provenance: ProvenanceDescriptor =
            serde_json::from_str(prov_str).map_err(|e| DecodeError::Utf8(e.to_string()))?;

        let prompt_hash = r.read_hash()?;
        let response_hash = r.read_hash()?;
        let timestamp_ns = r.read_u64()?;
        let duration_ns = r.read_u64()?;

        let claim_flag = r.read_u8()?;
        let ihsan_claim = match claim_flag {
            0 => None,
            1 => Some(r.read_f64()?),
            b => {
                return Err(DecodeError::UnknownDiscriminant {
                    field: "ihsan_claim_flag",
                    byte: b,
                })
            }
        };

        Ok(Self {
            round_version,
            provenance,
            prompt_hash,
            response_hash,
            timestamp_ns,
            duration_ns,
            ihsan_claim,
        })
    }
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::receipts::{InMemoryPayloadStore, ReceiptChain};

    fn sample_provenance() -> ProvenanceDescriptor {
        ProvenanceDescriptor {
            model_sha256: "a".repeat(64),
            model_signer: Some(SignerIdentity {
                key_id: "ed25519:bizra-authority-01".into(),
                algorithm: "ed25519".into(),
                signature_hex: "b".repeat(128),
            }),
            provider_identity: ProviderIdentity::LocalModel {
                weights_path: "/home/user/.bizra/models/gemma4-e4b.gguf".into(),
            },
        }
    }

    fn sample_round() -> ReasoningSessionPayload {
        ReasoningSessionPayload::new(
            sample_provenance(),
            [1u8; 32],
            [2u8; 32],
            1_700_000_000_000_000_000,
            42_000_000,
            Some(0.97),
        )
    }

    #[test]
    fn round_kind_is_reasoning_session() {
        let r = sample_round();
        assert_eq!(r.kind(), ReceiptKind::ReasoningSession);
    }

    #[test]
    fn canonical_bytes_roundtrip() {
        let r = sample_round();
        let bytes = r.canonical_bytes();
        let decoded = ReasoningSessionPayload::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(r, decoded);
    }

    #[test]
    fn hash_is_deterministic_across_new_constructions() {
        let r1 = sample_round();
        let r2 = sample_round();
        assert_eq!(r1.hash(), r2.hash());
    }

    #[test]
    fn tamper_on_any_field_changes_hash() {
        let r = sample_round();
        let base = r.hash();

        let mut with_new_claim = r.clone();
        with_new_claim.ihsan_claim = Some(0.50);
        assert_ne!(
            base,
            with_new_claim.hash(),
            "ihsan_claim tamper must detect"
        );

        let mut with_new_response = r.clone();
        with_new_response.response_hash = [9u8; 32];
        assert_ne!(
            base,
            with_new_response.hash(),
            "response_hash tamper must detect"
        );

        let mut with_new_provider = r.clone();
        with_new_provider.provenance.provider_identity = ProviderIdentity::RemoteApi {
            vendor: "impostor".into(),
        };
        assert_ne!(
            base,
            with_new_provider.hash(),
            "provider_identity tamper must detect"
        );
    }

    #[test]
    fn timestamp_ns_reported_through_trait() {
        let r = sample_round();
        assert_eq!(r.timestamp_ns(), 1_700_000_000_000_000_000);
    }

    #[test]
    fn chain_append_and_continuity_verifies() {
        let store = Box::new(InMemoryPayloadStore::new());
        let genesis = [0u8; 32];
        let mut chain = ReceiptChain::new(genesis, store);

        let round = sample_round();
        let appended_hash = chain.append_with_payload(round.clone()).unwrap();

        assert_eq!(chain.head(), appended_hash);
        assert_eq!(chain.len(), 1);
        assert_eq!(chain.latest_timestamp(), Some(round.timestamp_ns));
        assert!(chain.verify_continuity(genesis).is_ok());
    }

    #[test]
    fn fetch_and_decode_from_chain() {
        let store = Box::new(InMemoryPayloadStore::new());
        let genesis = [0u8; 32];
        let mut chain = ReceiptChain::new(genesis, store);

        let round = sample_round();
        let h = chain.append_with_payload(round.clone()).unwrap();

        let recovered: ReasoningSessionPayload = chain.fetch_and_decode(&h).unwrap();
        assert_eq!(recovered, round);
    }

    #[test]
    fn core_none_round_serializes_and_roundtrips() {
        let p = ProvenanceDescriptor {
            model_sha256: String::new(),
            model_signer: None,
            provider_identity: ProviderIdentity::CoreNone,
        };
        let r = ReasoningSessionPayload::new(p.clone(), [0u8; 32], [0u8; 32], 0, 0, None);

        let bytes = r.canonical_bytes();
        let decoded = ReasoningSessionPayload::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(decoded.provenance, p);
        assert_eq!(decoded.ihsan_claim, None);
        assert_eq!(decoded.prompt_hash, [0u8; 32]);
    }

    #[test]
    fn provenance_json_shape_is_stable() {
        // Schema contract: this shape MUST match the identical test in
        // bizra-omega/bizra-installer/src/install_receipt.rs. Any change
        // here requires the same change there in the same commit.
        let p = sample_provenance();
        let v = serde_json::to_value(&p).unwrap();
        let expected = serde_json::json!({
            "model_sha256": "a".repeat(64),
            "model_signer": {
                "key_id": "ed25519:bizra-authority-01",
                "algorithm": "ed25519",
                "signature_hex": "b".repeat(128),
            },
            "provider_identity": {
                "kind": "local_model",
                "weights_path": "/home/user/.bizra/models/gemma4-e4b.gguf",
            },
        });
        assert_eq!(v, expected);
    }

    #[test]
    fn provider_identity_all_variants_roundtrip() {
        let variants = vec![
            ProviderIdentity::CoreNone,
            ProviderIdentity::LocalModel {
                weights_path: "/path/to.gguf".into(),
            },
            ProviderIdentity::LocalServer {
                endpoint: "http://localhost:11434".into(),
                vendor: "ollama".into(),
            },
            ProviderIdentity::RemoteApi {
                vendor: "anthropic".into(),
            },
        ];
        for v in variants {
            let json = serde_json::to_string(&v).unwrap();
            let back: ProviderIdentity = serde_json::from_str(&json).unwrap();
            assert_eq!(v, back);
        }
    }

    #[test]
    fn short_bytes_does_not_panic_on_decode() {
        let short = b"bizra-reasoning-v1:";
        let r = ReasoningSessionPayload::from_canonical_bytes(short);
        assert!(r.is_err());
    }

    #[test]
    fn wrong_domain_tag_rejected() {
        let wrong = b"not-a-reasoning-payload-at-all:";
        let r = ReasoningSessionPayload::from_canonical_bytes(wrong);
        assert!(r.is_err());
    }

    #[test]
    fn round_trip_through_receipt_chain_preserves_hash() {
        let store = Box::new(InMemoryPayloadStore::new());
        let mut chain = ReceiptChain::new([0u8; 32], store);

        let r1 = sample_round();
        let expected_hash = r1.hash();
        let appended = chain.append_with_payload(r1.clone()).unwrap();
        assert_eq!(appended, expected_hash);

        let recovered: ReasoningSessionPayload = chain.fetch_and_decode(&appended).unwrap();
        assert_eq!(recovered.hash(), expected_hash);
    }
}
