// src/pci/envelope.rs - PCI Protocol Envelope
//
// Status: FROZEN — Changes require version bump + test vector update
// Wire Format: RFC 8785 JSON Canonicalization Scheme (JCS)

use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use super::types::*;

/// Proof-Carrying Inference Envelope
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PCIEnvelope {
    pub version: String,
    pub envelope_id: String,
    pub timestamp: String,
    pub nonce: String,
    pub sender: Sender,
    pub payload: Payload,
    pub metadata: Metadata,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<Signature>,
}

impl PCIEnvelope {
    /// Create a new unsigned envelope
    pub fn new(sender: Sender, payload: Payload, metadata: Metadata) -> Self {
        Self {
            version: PCI_VERSION.to_string(),
            envelope_id: generate_uuid(),
            timestamp: utc_now_iso(),
            nonce: generate_nonce(),
            sender,
            payload,
            metadata,
            signature: None,
        }
    }

    /// Serialize to canonical JSON bytes (RFC 8785 JCS)
    pub fn to_canonical_bytes(&self) -> Vec<u8> {
        canonical_json_bytes(&serde_json::to_value(self).unwrap_or_default())
    }

    /// Serialize to canonical JSON string
    pub fn to_canonical_json(&self) -> String {
        String::from_utf8_lossy(&self.to_canonical_bytes()).to_string()
    }

    /// Compute the domain-separated BLAKE3 digest
    pub fn compute_digest(&self) -> String {
        let canonical = self.to_canonical_bytes();
        domain_separated_digest(&canonical, DOMAIN_PREFIX)
    }

    /// Sign the envelope with Ed25519
    #[cfg(feature = "crypto")]
    pub fn sign(mut self, private_key: &[u8; 32]) -> anyhow::Result<Self> {
        use ed25519_dalek::{Signer, SigningKey};

        let signed_fields = vec![
            "version".into(),
            "envelope_id".into(),
            "timestamp".into(),
            "nonce".into(),
            "sender".into(),
            "payload".into(),
            "metadata".into(),
        ];

        // Create data to sign (without signature field)
        let data_to_sign = self.get_signed_fields_data(&signed_fields);
        let canonical = canonical_json_bytes(&serde_json::to_value(&data_to_sign)?);
        let digest = domain_separated_digest(&canonical, DOMAIN_PREFIX);
        let digest_bytes = hex::decode(&digest)?;

        let signing_key = SigningKey::from_bytes(private_key);
        let signature = signing_key.sign(&digest_bytes);

        self.signature = Some(Signature {
            algorithm: SignatureAlgorithm::Ed25519,
            value: hex::encode(signature.to_bytes()),
            signed_fields,
        });

        Ok(self)
    }

    /// Verify the envelope signature
    #[cfg(feature = "crypto")]
    pub fn verify_signature(&self) -> bool {
        use ed25519_dalek::{Signature as Ed25519Sig, Verifier, VerifyingKey};

        let sig = match &self.signature {
            Some(s) => s,
            None => return false,
        };

        // Get signed fields data
        let data_to_verify = self.get_signed_fields_data(&sig.signed_fields);
        let canonical = match serde_json::to_value(&data_to_verify) {
            Ok(v) => canonical_json_bytes(&v),
            Err(_) => return false,
        };
        let digest = domain_separated_digest(&canonical, DOMAIN_PREFIX);
        let digest_bytes = match hex::decode(&digest) {
            Ok(b) => b,
            Err(_) => return false,
        };

        // Parse public key
        let pk_bytes = match hex::decode(&self.sender.public_key) {
            Ok(b) => b,
            Err(_) => return false,
        };
        if pk_bytes.len() != 32 {
            return false;
        }
        let pk_array: [u8; 32] = match pk_bytes.try_into() {
            Ok(a) => a,
            Err(_) => return false,
        };
        let verifying_key = match VerifyingKey::from_bytes(&pk_array) {
            Ok(k) => k,
            Err(_) => return false,
        };

        // Parse signature
        let sig_bytes = match hex::decode(&sig.value) {
            Ok(b) => b,
            Err(_) => return false,
        };
        if sig_bytes.len() != 64 {
            return false;
        }
        let sig_array: [u8; 64] = match sig_bytes.try_into() {
            Ok(a) => a,
            Err(_) => return false,
        };
        let signature = Ed25519Sig::from_bytes(&sig_array);

        verifying_key.verify(&digest_bytes, &signature).is_ok()
    }

    #[cfg(not(feature = "crypto"))]
    pub fn verify_signature(&self) -> bool {
        // Without crypto feature, signature verification is not available
        self.signature.is_some()
    }

    fn get_signed_fields_data(&self, signed_fields: &[String]) -> serde_json::Value {
        let value = serde_json::to_value(self).unwrap_or_default();
        if let serde_json::Value::Object(map) = value {
            let filtered: serde_json::Map<String, serde_json::Value> = map
                .into_iter()
                .filter(|(k, _)| signed_fields.contains(k))
                .collect();
            serde_json::Value::Object(filtered)
        } else {
            serde_json::Value::Null
        }
    }
}

/// Generate a cryptographically random nonce (32 bytes hex)
pub fn generate_nonce() -> String {
    use rand::RngCore;
    let mut nonce = [0u8; NONCE_BYTES];
    rand::thread_rng().fill_bytes(&mut nonce);
    hex::encode(nonce)
}

/// Validate nonce format
pub fn validate_nonce(nonce: &str) -> bool {
    if nonce.len() != NONCE_BYTES * 2 {
        return false;
    }
    hex::decode(nonce).is_ok()
}

/// Serialize to canonical JSON bytes (RFC 8785 JCS)
/// Keys sorted lexicographically, no whitespace
pub fn canonical_json_bytes(value: &serde_json::Value) -> Vec<u8> {
    fn serialize_value(value: &serde_json::Value, out: &mut Vec<u8>) {
        match value {
            serde_json::Value::Null => out.extend_from_slice(b"null"),
            serde_json::Value::Bool(b) => {
                out.extend_from_slice(if *b { b"true" } else { b"false" })
            }
            serde_json::Value::Number(n) => out.extend_from_slice(n.to_string().as_bytes()),
            serde_json::Value::String(s) => {
                out.push(b'"');
                for c in s.chars() {
                    match c {
                        '"' => out.extend_from_slice(b"\\\""),
                        '\\' => out.extend_from_slice(b"\\\\"),
                        '\n' => out.extend_from_slice(b"\\n"),
                        '\r' => out.extend_from_slice(b"\\r"),
                        '\t' => out.extend_from_slice(b"\\t"),
                        c if c.is_control() => {
                            out.extend_from_slice(format!("\\u{:04x}", c as u32).as_bytes())
                        }
                        c => out.extend_from_slice(c.to_string().as_bytes()),
                    }
                }
                out.push(b'"');
            }
            serde_json::Value::Array(arr) => {
                out.push(b'[');
                for (i, v) in arr.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    serialize_value(v, out);
                }
                out.push(b']');
            }
            serde_json::Value::Object(obj) => {
                out.push(b'{');
                // Sort keys lexicographically (BTreeMap handles this)
                let sorted: BTreeMap<_, _> = obj.iter().collect();
                for (i, (k, v)) in sorted.iter().enumerate() {
                    if i > 0 {
                        out.push(b',');
                    }
                    out.push(b'"');
                    out.extend_from_slice(k.as_bytes());
                    out.push(b'"');
                    out.push(b':');
                    serialize_value(v, out);
                }
                out.push(b'}');
            }
        }
    }

    let mut out = Vec::new();
    serialize_value(value, &mut out);
    out
}

/// Compute BLAKE3 digest
pub fn blake3_digest(data: &[u8]) -> String {
    let hash = blake3::hash(data);
    hash.to_hex().to_string()
}

/// Compute domain-separated BLAKE3 digest
pub fn domain_separated_digest(data: &[u8], domain: &str) -> String {
    let mut combined = Vec::with_capacity(domain.len() + data.len());
    combined.extend_from_slice(domain.as_bytes());
    combined.extend_from_slice(data);
    blake3_digest(&combined)
}

/// Validate envelope schema
pub fn validate_envelope_schema(envelope: &PCIEnvelope) -> Vec<String> {
    let mut errors = Vec::new();

    // Version check
    if envelope.version != PCI_VERSION {
        errors.push(format!(
            "Unsupported version: {} (expected {})",
            envelope.version, PCI_VERSION
        ));
    }

    // Nonce format
    if envelope.nonce.len() != NONCE_BYTES * 2 {
        errors.push(format!(
            "Invalid nonce length: {} (expected {})",
            envelope.nonce.len(),
            NONCE_BYTES * 2
        ));
    }

    // Public key format
    if envelope.sender.public_key.len() != 64 {
        errors.push(format!(
            "Invalid public_key length: {} (expected 64)",
            envelope.sender.public_key.len()
        ));
    }

    // Ihsān score range
    if !(0.0..=1.0).contains(&envelope.metadata.ihsan_score) {
        errors.push(format!(
            "ihsan_score out of range: {}",
            envelope.metadata.ihsan_score
        ));
    }

    // SNR score range
    if !(0.0..=1.0).contains(&envelope.metadata.snr_score) {
        errors.push(format!(
            "snr_score out of range: {}",
            envelope.metadata.snr_score
        ));
    }

    // Signature format (if present)
    if let Some(ref sig) = envelope.signature {
        if sig.value.len() != 128 {
            errors.push(format!(
                "Invalid signature length: {} (expected 128)",
                sig.value.len()
            ));
        }
    }

    errors
}

/// Envelope builder (fluent API)
pub struct EnvelopeBuilder {
    agent_type: Option<AgentType>,
    agent_id: Option<String>,
    public_key: Option<String>,
    action: Option<String>,
    data: serde_json::Value,
    policy_hash: Option<String>,
    state_hash: Option<String>,
    ihsan_score: f64,
    snr_score: f64,
    urgency: Urgency,
}

impl Default for EnvelopeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl EnvelopeBuilder {
    pub fn new() -> Self {
        Self {
            agent_type: None,
            agent_id: None,
            public_key: None,
            action: None,
            data: serde_json::Value::Object(serde_json::Map::new()),
            policy_hash: None,
            state_hash: None,
            ihsan_score: 0.0,
            snr_score: 0.0,
            urgency: Urgency::NearRealTime,
        }
    }

    pub fn with_sender(mut self, agent_type: AgentType, agent_id: &str, public_key: &str) -> Self {
        self.agent_type = Some(agent_type);
        self.agent_id = Some(agent_id.to_string());
        self.public_key = Some(public_key.to_string());
        self
    }

    pub fn with_action(mut self, action: &str, data: serde_json::Value) -> Self {
        self.action = Some(action.to_string());
        self.data = data;
        self
    }

    pub fn with_policy(mut self, policy_hash: &str) -> Self {
        self.policy_hash = Some(policy_hash.to_string());
        self
    }

    pub fn with_state(mut self, state_hash: &str) -> Self {
        self.state_hash = Some(state_hash.to_string());
        self
    }

    pub fn with_scores(mut self, ihsan: f64, snr: f64) -> Self {
        self.ihsan_score = ihsan;
        self.snr_score = snr;
        self
    }

    pub fn with_urgency(mut self, urgency: Urgency) -> Self {
        self.urgency = urgency;
        self
    }

    pub fn build(self) -> Result<PCIEnvelope, String> {
        let agent_type = self.agent_type.ok_or("agent_type is required")?;
        let agent_id = self.agent_id.ok_or("agent_id is required")?;
        let public_key = self.public_key.ok_or("public_key is required")?;
        let action = self.action.ok_or("action is required")?;
        let policy_hash = self.policy_hash.ok_or("policy_hash is required")?;
        let state_hash = self.state_hash.ok_or("state_hash is required")?;

        Ok(PCIEnvelope::new(
            Sender {
                agent_type,
                agent_id,
                public_key,
            },
            Payload {
                action,
                data: self.data,
                policy_hash,
                state_hash,
            },
            Metadata {
                ihsan_score: self.ihsan_score,
                snr_score: self.snr_score,
                urgency: self.urgency,
            },
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_canonical_json() {
        let value = serde_json::json!({
            "zebra": true,
            "apple": 1,
            "mango": "hello"
        });
        let canonical = canonical_json_bytes(&value);
        assert_eq!(
            String::from_utf8_lossy(&canonical),
            r#"{"apple":1,"mango":"hello","zebra":true}"#
        );
    }

    #[test]
    fn test_domain_separated_digest() {
        let data = b"hello";
        let digest = domain_separated_digest(data, "test:");
        assert_eq!(digest.len(), 64); // 32 bytes = 64 hex chars
    }

    #[test]
    fn test_nonce_generation() {
        let nonce = generate_nonce();
        assert_eq!(nonce.len(), 64);
        assert!(validate_nonce(&nonce));
    }

    #[test]
    fn test_envelope_builder() {
        let policy_hash = "a".repeat(64);
        let state_hash = "b".repeat(64);
        let public_key = "c".repeat(64);

        let envelope = EnvelopeBuilder::new()
            .with_sender(AgentType::Pat, "pat-001", &public_key)
            .with_action("propose", serde_json::json!({"task": "analyze"}))
            .with_policy(&policy_hash)
            .with_state(&state_hash)
            .with_scores(0.97, 0.85)
            .build()
            .unwrap();

        assert_eq!(envelope.version, PCI_VERSION);
        assert_eq!(envelope.sender.agent_type, AgentType::Pat);
        assert_eq!(envelope.metadata.ihsan_score, 0.97);
    }

    #[test]
    fn test_validate_envelope() {
        let policy_hash = "a".repeat(64);
        let state_hash = "b".repeat(64);
        let public_key = "c".repeat(64);

        let envelope = EnvelopeBuilder::new()
            .with_sender(AgentType::Pat, "pat-001", &public_key)
            .with_action("propose", serde_json::json!({}))
            .with_policy(&policy_hash)
            .with_state(&state_hash)
            .with_scores(0.97, 0.85)
            .build()
            .unwrap();

        let errors = validate_envelope_schema(&envelope);
        assert!(errors.is_empty(), "Errors: {:?}", errors);
    }
}
