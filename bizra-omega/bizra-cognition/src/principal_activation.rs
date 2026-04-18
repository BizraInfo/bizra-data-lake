//! BIZRA Principal Activation — §Cycle-7 G2 Contracts
//!
//! بسم الله الرحمن الرحيم
//!
//! File: bizra-cognition/src/principal_activation.rs
//! Authority: cycle-7/niyyah.md §G2 + §"Writer authority decision (HYBRID)"
//! Cycle position: 7, Phase 2
//! Depends on: receipts.rs (ReceiptKind::PrincipalActivation), canonical_hasher.rs
//!
//! Contracts for turning Mumo's declared-principal intent into a lawful,
//! receipted activation through the mission-runtime connector landed in
//! G1 (add18501):
//!
//!   1. NodeIdentityAnchor — read-only projection of
//!      sovereign_state/identity/credentials.json. The Python stack is
//!      the authoritative writer; Rust only loads + validates.
//!
//!   2. PrincipalActivationEnvelope — canonical Stage-S1→S2 intent capture
//!      specific to principal activation. Binds identity anchor into the
//!      envelope so the downstream mission can be verified against a
//!      concrete node identity rather than a free-text intent alone.
//!
//!   3. PrincipalProfile — derived, rebuildable local cache of the
//!      activated principal. Owned by principal_cache.rs; canonical-hashed
//!      so PrincipalActivationReceipt can bind to it.
//!
//!   4. PrincipalActivationReceipt — chain-sealed proof that a specific
//!      NodeLifecycle mission receipt was bound to a specific node
//!      identity + principal profile. Non-transferable, proof-bearing.
//!      Kind: ReceiptKind::PrincipalActivation (0x61).
//!
//! §10 Proof Law: rejected activations produce no PrincipalActivationReceipt,
//! no on-disk profile, and no local-mint effect. The reject path carries a
//! structured remediation string only.

use std::fs;
use std::path::Path;

use serde_json::Value;

use crate::canonical_hasher::blake3_domain;
use crate::receipts::{
    Blake3Hash, ByteReader, DecodeError, ReceiptKind, ReceiptPayload,
    ReceiptPayloadDecode,
};

// ════════════════════════════════════════════════════════════
// NodeIdentityAnchor — read-only projection of credentials.json
// ════════════════════════════════════════════════════════════

/// Read-only projection of sovereign_state/identity/credentials.json.
///
/// Schema (per Python writer, observed 2026-04-13):
///   { "node_id": "NODE0",
///     "public_key": "<64-char lowercase hex>",
///     "created_at": "<ISO-8601 UTC>" }
///
/// Niyyah §"Writer authority decision": chain truth stays Python-authored;
/// Rust projects these read-only.
#[derive(Debug, Clone)]
pub struct NodeIdentityAnchor {
    pub node_id: String,
    pub public_key_hex: String,
    pub created_at: String,
}

#[derive(Debug)]
pub enum NodeAnchorError {
    FileMissing(String),
    ReadFailed { path: String, msg: String },
    ParseFailed { path: String, msg: String },
    Malformed { path: String, reason: &'static str },
    PubkeyDecode { path: String, reason: String },
}

impl std::fmt::Display for NodeAnchorError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::FileMissing(p) => write!(f, "node identity anchor missing: {}", p),
            Self::ReadFailed { path, msg } => write!(f, "read anchor {}: {}", path, msg),
            Self::ParseFailed { path, msg } => write!(f, "parse anchor {}: {}", path, msg),
            Self::Malformed { path, reason } => {
                write!(f, "anchor {} malformed: {}", path, reason)
            }
            Self::PubkeyDecode { path, reason } => {
                write!(f, "anchor {} pubkey decode failed: {}", path, reason)
            }
        }
    }
}

impl std::error::Error for NodeAnchorError {}

impl NodeIdentityAnchor {
    /// Load and validate a node identity anchor from disk.
    ///
    /// Fails closed on missing file, malformed JSON, missing fields, or
    /// pubkey decode failure. The happy path is the exact schema written
    /// by the Python activation stack on first node boot.
    pub fn load(path: &Path) -> Result<Self, NodeAnchorError> {
        if !path.exists() {
            return Err(NodeAnchorError::FileMissing(path.display().to_string()));
        }
        let bytes = fs::read(path).map_err(|e| NodeAnchorError::ReadFailed {
            path: path.display().to_string(),
            msg: e.to_string(),
        })?;
        let v: Value =
            serde_json::from_slice(&bytes).map_err(|e| NodeAnchorError::ParseFailed {
                path: path.display().to_string(),
                msg: e.to_string(),
            })?;
        let obj = v.as_object().ok_or(NodeAnchorError::Malformed {
            path: path.display().to_string(),
            reason: "root is not an object",
        })?;
        let node_id = obj
            .get("node_id")
            .and_then(|s| s.as_str())
            .ok_or(NodeAnchorError::Malformed {
                path: path.display().to_string(),
                reason: "missing node_id",
            })?
            .to_string();
        let public_key_hex = obj
            .get("public_key")
            .and_then(|s| s.as_str())
            .ok_or(NodeAnchorError::Malformed {
                path: path.display().to_string(),
                reason: "missing public_key",
            })?
            .to_string();
        let created_at = obj
            .get("created_at")
            .and_then(|s| s.as_str())
            .ok_or(NodeAnchorError::Malformed {
                path: path.display().to_string(),
                reason: "missing created_at",
            })?
            .to_string();

        let anchor = NodeIdentityAnchor {
            node_id,
            public_key_hex,
            created_at,
        };
        anchor
            .public_key_bytes()
            .map_err(|e| NodeAnchorError::PubkeyDecode {
                path: path.display().to_string(),
                reason: e,
            })?;
        Ok(anchor)
    }

    /// Decode the hex public key into a 32-byte array.
    pub fn public_key_bytes(&self) -> Result<Blake3Hash, String> {
        if self.public_key_hex.len() != 64 {
            return Err(format!(
                "public_key must be 64 hex chars, got {}",
                self.public_key_hex.len()
            ));
        }
        let mut out = [0u8; 32];
        for i in 0..32 {
            let hi = hex_nibble(self.public_key_hex.as_bytes()[i * 2])?;
            let lo = hex_nibble(self.public_key_hex.as_bytes()[i * 2 + 1])?;
            out[i] = (hi << 4) | lo;
        }
        Ok(out)
    }

    /// Test-only constructor — does not touch disk.
    #[cfg(test)]
    pub(crate) fn for_test(node_id: &str, pubkey_hex: &str, created_at: &str) -> Self {
        NodeIdentityAnchor {
            node_id: node_id.into(),
            public_key_hex: pubkey_hex.into(),
            created_at: created_at.into(),
        }
    }
}

fn hex_nibble(b: u8) -> Result<u8, String> {
    match b {
        b'0'..=b'9' => Ok(b - b'0'),
        b'a'..=b'f' => Ok(b - b'a' + 10),
        b'A'..=b'F' => Ok(b - b'A' + 10),
        _ => Err(format!("non-hex byte 0x{:02x}", b)),
    }
}

// ════════════════════════════════════════════════════════════
// PrincipalActivationEnvelope — activation-specific intent capture
// ════════════════════════════════════════════════════════════

/// The canonical activation-intent text. Kept stable so derived
/// intent_hash values are reproducible across restarts.
pub const CANONICAL_ACTIVATION_INTENT: &str =
    "declare Node0 principal — activate lawful operator face through Dema";

/// Canonical intent for declaring oneself Node0 principal.
///
/// Built from operator input + a loaded NodeIdentityAnchor. The resulting
/// structure feeds `CognitionRuntime::submit_principal_activation`, which
/// wraps the G1 mission-runtime loop.
#[derive(Debug, Clone)]
pub struct PrincipalActivationEnvelope {
    /// Operator-declared principal name (e.g. "Mumo").
    pub principal_name: String,
    /// Declared role. For Cycle-7 Phase 2 this must be "node0_principal".
    /// Narrower roles are reserved for later cycles.
    pub declared_role: String,
    /// 32-byte node public key lifted from the NodeIdentityAnchor.
    pub node_pubkey: Blake3Hash,
    /// Node id string from the anchor (e.g. "NODE0").
    pub node_id: String,
    /// Hash of (canonical_intent || principal_name || node_id).
    pub intent_hash: Blake3Hash,
    /// Monotonic timestamp (nanoseconds).
    pub created_ns: u64,
}

impl PrincipalActivationEnvelope {
    /// Build an activation envelope from operator-facing inputs plus
    /// a loaded NodeIdentityAnchor. Deterministic intent_hash over
    /// (canonical_intent || principal_name || node_id).
    pub fn from_anchor(
        principal_name: String,
        declared_role: String,
        anchor: &NodeIdentityAnchor,
        created_ns: u64,
    ) -> Result<Self, NodeAnchorError> {
        let node_pubkey =
            anchor
                .public_key_bytes()
                .map_err(|e| NodeAnchorError::PubkeyDecode {
                    path: "<in-memory anchor>".into(),
                    reason: e,
                })?;
        let mut intent_buf =
            Vec::with_capacity(CANONICAL_ACTIVATION_INTENT.len() + principal_name.len() + 32);
        intent_buf.extend_from_slice(CANONICAL_ACTIVATION_INTENT.as_bytes());
        intent_buf.push(0);
        intent_buf.extend_from_slice(principal_name.as_bytes());
        intent_buf.push(0);
        intent_buf.extend_from_slice(anchor.node_id.as_bytes());
        let intent_hash = blake3_domain("bizra-principal-activation-intent-v1", &intent_buf);
        Ok(PrincipalActivationEnvelope {
            principal_name,
            declared_role,
            node_pubkey,
            node_id: anchor.node_id.clone(),
            intent_hash,
            created_ns,
        })
    }

    /// Human-readable intent string consumable by MissionEnvelope::from_intent.
    pub fn intent_text(&self) -> String {
        format!(
            "{} — principal={}, role={}, node_id={}",
            CANONICAL_ACTIVATION_INTENT, self.principal_name, self.declared_role, self.node_id,
        )
    }
}

// ════════════════════════════════════════════════════════════
// PrincipalProfile — derived, rebuildable local cache record
// ════════════════════════════════════════════════════════════

/// Locally cached proof that a principal has been activated.
///
/// Niyyah §"Writer authority decision (HYBRID)" §"Storage location":
/// lives in sovereign_state/dema_cache/principal.json. Non-authoritative;
/// may be rebuilt from chain by scanning for the PrincipalActivationReceipt
/// whose fields match this anchor.
#[derive(Debug, Clone)]
pub struct PrincipalProfile {
    /// Stable identity of this principal within this node.
    /// principal_id = blake3("bizra-principal-id-v1", node_pubkey || principal_name)
    pub principal_id: Blake3Hash,
    /// Operator-declared name at activation time.
    pub name: String,
    /// Node id from the anchor at activation time (e.g. "NODE0").
    pub node_id: String,
    /// Declared role (e.g. "node0_principal").
    pub declared_role: String,
    /// NodeLifecycle mission receipt_id sealed by submit_mission under
    /// the activation envelope. This is what the chain actually proves.
    pub activation_receipt_id: Blake3Hash,
    /// Monotonic timestamp of activation (nanoseconds, mission submission).
    pub activation_ns: u64,
}

impl PrincipalProfile {
    pub fn new(
        envelope: &PrincipalActivationEnvelope,
        activation_receipt_id: Blake3Hash,
        activation_ns: u64,
    ) -> Self {
        let mut id_buf = Vec::with_capacity(32 + envelope.principal_name.len());
        id_buf.extend_from_slice(&envelope.node_pubkey);
        id_buf.extend_from_slice(envelope.principal_name.as_bytes());
        let principal_id = blake3_domain("bizra-principal-id-v1", &id_buf);
        PrincipalProfile {
            principal_id,
            name: envelope.principal_name.clone(),
            node_id: envelope.node_id.clone(),
            declared_role: envelope.declared_role.clone(),
            activation_receipt_id,
            activation_ns,
        }
    }

    /// Canonical hash of this profile used to bind it into the
    /// PrincipalActivationReceipt.
    pub fn profile_hash(&self) -> Blake3Hash {
        let mut buf = Vec::with_capacity(160);
        buf.extend_from_slice(&self.principal_id);
        buf.extend_from_slice(&(self.name.len() as u32).to_le_bytes());
        buf.extend_from_slice(self.name.as_bytes());
        buf.extend_from_slice(&(self.node_id.len() as u32).to_le_bytes());
        buf.extend_from_slice(self.node_id.as_bytes());
        buf.extend_from_slice(&(self.declared_role.len() as u32).to_le_bytes());
        buf.extend_from_slice(self.declared_role.as_bytes());
        buf.extend_from_slice(&self.activation_receipt_id);
        buf.extend_from_slice(&self.activation_ns.to_le_bytes());
        blake3_domain("bizra-principal-profile-v1", &buf)
    }
}

// ════════════════════════════════════════════════════════════
// PrincipalActivationReceipt — chain-sealed §Cycle-7 G2 artifact
// ════════════════════════════════════════════════════════════

/// Chain-sealed receipt binding an activation mission's NodeLifecycle
/// receipt to a specific principal profile.
///
/// Appended to the chain as the final artifact of a permitted
/// `submit_principal_activation` call, immediately AFTER the
/// ManifestArtifact emitted by the G1 connector. Kind = 0x61.
#[derive(Debug, Clone)]
pub struct PrincipalActivationReceipt {
    /// Unique receipt id. blake3("…-v1", canonical-bytes-without-id).
    pub receipt_id: Blake3Hash,
    /// NodeLifecycle mission receipt_id this activation binds to.
    pub activation_receipt_ref: Blake3Hash,
    /// Canonical profile_hash() of the persisted PrincipalProfile.
    pub principal_profile_hash: Blake3Hash,
    /// Node pubkey copied from the anchor at activation time.
    pub node_pubkey: Blake3Hash,
    /// principal_id (mirror of PrincipalProfile.principal_id).
    pub principal_id: Blake3Hash,
    /// Monotonic timestamp (nanoseconds).
    pub timestamp_ns: u64,
    /// Chain head at time of this receipt's append (pre-this-receipt).
    pub prev_chain: Blake3Hash,
}

impl PrincipalActivationReceipt {
    pub fn new(
        activation_receipt_ref: Blake3Hash,
        principal_profile_hash: Blake3Hash,
        node_pubkey: Blake3Hash,
        principal_id: Blake3Hash,
        timestamp_ns: u64,
        prev_chain: Blake3Hash,
    ) -> Self {
        let mut r = PrincipalActivationReceipt {
            receipt_id: [0u8; 32],
            activation_receipt_ref,
            principal_profile_hash,
            node_pubkey,
            principal_id,
            timestamp_ns,
            prev_chain,
        };
        r.receipt_id = blake3_domain(
            "bizra-principal-activation-receipt-v1",
            &r.canonical_bytes_without_id(),
        );
        r
    }

    fn canonical_bytes_without_id(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(200);
        buf.extend_from_slice(&self.activation_receipt_ref);
        buf.extend_from_slice(&self.principal_profile_hash);
        buf.extend_from_slice(&self.node_pubkey);
        buf.extend_from_slice(&self.principal_id);
        buf.extend_from_slice(&self.timestamp_ns.to_le_bytes());
        buf.extend_from_slice(&self.prev_chain);
        buf
    }
}

impl ReceiptPayload for PrincipalActivationReceipt {
    fn kind(&self) -> ReceiptKind {
        ReceiptKind::PrincipalActivation
    }

    fn timestamp_ns(&self) -> u64 {
        self.timestamp_ns
    }

    fn canonical_bytes(&self) -> Vec<u8> {
        let mut buf = Vec::with_capacity(232);
        buf.extend_from_slice(&self.receipt_id);
        buf.extend_from_slice(&self.canonical_bytes_without_id());
        buf
    }

    fn hash(&self) -> Blake3Hash {
        self.receipt_id
    }
}

impl ReceiptPayloadDecode for PrincipalActivationReceipt {
    fn from_canonical_bytes(bytes: &[u8]) -> Result<Self, DecodeError> {
        let mut r = ByteReader::new(bytes);
        let receipt_id = r.read_hash()?;
        let activation_receipt_ref = r.read_hash()?;
        let principal_profile_hash = r.read_hash()?;
        let node_pubkey = r.read_hash()?;
        let principal_id = r.read_hash()?;
        let timestamp_ns = r.read_u64()?;
        let prev_chain = r.read_hash()?;
        Ok(PrincipalActivationReceipt {
            receipt_id,
            activation_receipt_ref,
            principal_profile_hash,
            node_pubkey,
            principal_id,
            timestamp_ns,
            prev_chain,
        })
    }
}

// ════════════════════════════════════════════════════════════
// Tests
// ════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    const TEST_PUBKEY_HEX: &str =
        "0232760d5349763eb6b45f57944ffec67d19c36214dd60824c6d0f728d5f762a";

    fn test_anchor() -> NodeIdentityAnchor {
        NodeIdentityAnchor::for_test("NODE0", TEST_PUBKEY_HEX, "2026-04-13T23:54:59Z")
    }

    // ── NodeIdentityAnchor ──

    #[test]
    fn anchor_public_key_bytes_decodes_to_32_bytes() {
        let a = test_anchor();
        let bytes = a.public_key_bytes().unwrap();
        assert_eq!(bytes.len(), 32);
        assert_eq!(bytes[0], 0x02);
        assert_eq!(bytes[31], 0x2a);
    }

    #[test]
    fn anchor_rejects_bad_hex_length() {
        let a = NodeIdentityAnchor::for_test("NODE0", "deadbeef", "now");
        assert!(a.public_key_bytes().is_err());
    }

    #[test]
    fn anchor_rejects_non_hex_chars() {
        let a = NodeIdentityAnchor::for_test(
            "NODE0",
            "ZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZZ",
            "now",
        );
        assert!(a.public_key_bytes().is_err());
    }

    #[test]
    fn anchor_loads_valid_file_from_disk() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(
            tmp.path(),
            br#"{"node_id":"NODE0","public_key":"0232760d5349763eb6b45f57944ffec67d19c36214dd60824c6d0f728d5f762a","created_at":"2026-04-13T23:54:59Z"}"#,
        )
        .unwrap();
        let a = NodeIdentityAnchor::load(tmp.path()).unwrap();
        assert_eq!(a.node_id, "NODE0");
        assert_eq!(a.public_key_hex.len(), 64);
        assert_eq!(a.created_at, "2026-04-13T23:54:59Z");
    }

    #[test]
    fn anchor_load_missing_file_fails_closed() {
        let err = NodeIdentityAnchor::load(std::path::Path::new(
            "/tmp/__definitely_missing_bizra_anchor__",
        ))
        .unwrap_err();
        assert!(matches!(err, NodeAnchorError::FileMissing(_)));
    }

    #[test]
    fn anchor_load_malformed_json_fails_closed() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), b"not json").unwrap();
        let err = NodeIdentityAnchor::load(tmp.path()).unwrap_err();
        assert!(matches!(err, NodeAnchorError::ParseFailed { .. }));
    }

    #[test]
    fn anchor_load_missing_field_fails_closed() {
        let tmp = tempfile::NamedTempFile::new().unwrap();
        std::fs::write(tmp.path(), br#"{"node_id":"X"}"#).unwrap();
        let err = NodeIdentityAnchor::load(tmp.path()).unwrap_err();
        assert!(matches!(err, NodeAnchorError::Malformed { .. }));
    }

    // ── PrincipalActivationEnvelope ──

    #[test]
    fn envelope_intent_hash_is_deterministic() {
        let a = test_anchor();
        let e1 = PrincipalActivationEnvelope::from_anchor(
            "Mumo".into(),
            "node0_principal".into(),
            &a,
            1_000,
        )
        .unwrap();
        let e2 = PrincipalActivationEnvelope::from_anchor(
            "Mumo".into(),
            "node0_principal".into(),
            &a,
            2_000,
        )
        .unwrap();
        assert_eq!(
            e1.intent_hash, e2.intent_hash,
            "intent_hash must not depend on created_ns"
        );
    }

    #[test]
    fn envelope_different_principals_different_intent_hashes() {
        let a = test_anchor();
        let e1 = PrincipalActivationEnvelope::from_anchor(
            "Mumo".into(),
            "node0_principal".into(),
            &a,
            0,
        )
        .unwrap();
        let e2 = PrincipalActivationEnvelope::from_anchor(
            "Other".into(),
            "node0_principal".into(),
            &a,
            0,
        )
        .unwrap();
        assert_ne!(e1.intent_hash, e2.intent_hash);
    }

    #[test]
    fn envelope_intent_text_embeds_principal_and_node() {
        let a = test_anchor();
        let e = PrincipalActivationEnvelope::from_anchor(
            "Mumo".into(),
            "node0_principal".into(),
            &a,
            0,
        )
        .unwrap();
        let t = e.intent_text();
        assert!(t.contains("Mumo"));
        assert!(t.contains("node0_principal"));
        assert!(t.contains("NODE0"));
        assert!(t.contains(CANONICAL_ACTIVATION_INTENT));
    }

    // ── PrincipalProfile ──

    #[test]
    fn profile_id_stable_across_constructions() {
        let a = test_anchor();
        let e = PrincipalActivationEnvelope::from_anchor(
            "Mumo".into(),
            "node0_principal".into(),
            &a,
            0,
        )
        .unwrap();
        let p1 = PrincipalProfile::new(&e, [7u8; 32], 100);
        let p2 = PrincipalProfile::new(&e, [7u8; 32], 100);
        assert_eq!(p1.principal_id, p2.principal_id);
        assert_eq!(p1.profile_hash(), p2.profile_hash());
    }

    #[test]
    fn profile_hash_differs_for_different_activations() {
        let a = test_anchor();
        let e = PrincipalActivationEnvelope::from_anchor(
            "Mumo".into(),
            "node0_principal".into(),
            &a,
            0,
        )
        .unwrap();
        let p1 = PrincipalProfile::new(&e, [7u8; 32], 100);
        let p2 = PrincipalProfile::new(&e, [9u8; 32], 100);
        assert_eq!(
            p1.principal_id, p2.principal_id,
            "principal_id is stable across re-activations of the same principal"
        );
        assert_ne!(
            p1.profile_hash(),
            p2.profile_hash(),
            "profile_hash must bind activation_receipt_id"
        );
    }

    // ── PrincipalActivationReceipt ──

    #[test]
    fn receipt_round_trips_bytes() {
        let r = PrincipalActivationReceipt::new(
            [1u8; 32], [2u8; 32], [3u8; 32], [4u8; 32], 12345, [5u8; 32],
        );
        let bytes = r.canonical_bytes();
        let decoded = PrincipalActivationReceipt::from_canonical_bytes(&bytes).unwrap();
        assert_eq!(decoded.receipt_id, r.receipt_id);
        assert_eq!(decoded.activation_receipt_ref, r.activation_receipt_ref);
        assert_eq!(decoded.principal_profile_hash, r.principal_profile_hash);
        assert_eq!(decoded.node_pubkey, r.node_pubkey);
        assert_eq!(decoded.principal_id, r.principal_id);
        assert_eq!(decoded.timestamp_ns, r.timestamp_ns);
        assert_eq!(decoded.prev_chain, r.prev_chain);
    }

    #[test]
    fn receipt_kind_is_principal_activation() {
        let r = PrincipalActivationReceipt::new(
            [1u8; 32], [2u8; 32], [3u8; 32], [4u8; 32], 12345, [5u8; 32],
        );
        assert_eq!(r.kind(), ReceiptKind::PrincipalActivation);
    }

    #[test]
    fn receipt_id_binds_prev_chain() {
        let r1 = PrincipalActivationReceipt::new(
            [1u8; 32], [2u8; 32], [3u8; 32], [4u8; 32], 12345, [5u8; 32],
        );
        let r2 = PrincipalActivationReceipt::new(
            [1u8; 32], [2u8; 32], [3u8; 32], [4u8; 32], 12345, [6u8; 32],
        );
        assert_ne!(
            r1.receipt_id, r2.receipt_id,
            "receipt_id must bind prev_chain"
        );
    }

    #[test]
    fn receipt_timestamp_advances_chain_latest() {
        use crate::receipts::{InMemoryPayloadStore, ReceiptChain};
        let store = Box::new(InMemoryPayloadStore::new());
        let mut chain = ReceiptChain::new([0u8; 32], store);
        let r = PrincipalActivationReceipt::new(
            [1u8; 32],
            [2u8; 32],
            [3u8; 32],
            [4u8; 32],
            1_700_000_000_000_000_000,
            [0u8; 32],
        );
        chain.append_with_payload(r).unwrap();
        assert_eq!(chain.latest_timestamp(), Some(1_700_000_000_000_000_000));
    }

    #[test]
    fn receipt_kind_byte_roundtrips() {
        assert_eq!(
            ReceiptKind::from_byte(0x61),
            Some(ReceiptKind::PrincipalActivation)
        );
    }
}
