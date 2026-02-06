//! PCI Envelope — Cryptographically signed containers

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::identity::{domain_separated_digest, NodeId, NodeIdentity};
use crate::MAX_TTL_SECONDS;
use super::RejectCode;

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PCIEnvelope<T> {
    pub id: String,
    pub version: String,
    pub sender: NodeId,
    pub timestamp: DateTime<Utc>,
    pub ttl: u64,
    pub content_hash: String,
    pub signature: String,
    pub public_key: String,
    pub provenance: Vec<String>,
    pub payload: T,
}

impl<T: Serialize + for<'de> Deserialize<'de>> PCIEnvelope<T> {
    pub fn create(
        identity: &NodeIdentity,
        payload: T,
        ttl: u64,
        provenance: Vec<String>,
    ) -> Result<Self, RejectCode> {
        let id = format!("pci_{}", &Uuid::new_v4().to_string().replace("-", "")[..16]);
        let timestamp = Utc::now();
        let ttl = ttl.min(MAX_TTL_SECONDS);

        let payload_json = serde_json::to_string(&payload)
            .map_err(|_| RejectCode::RejectSchema)?;
        let content_hash = domain_separated_digest(payload_json.as_bytes());

        let signable = SignableEnvelope {
            id: &id, version: "1.0", sender: identity.node_id(),
            timestamp, ttl, content_hash: &content_hash, provenance: &provenance,
        };
        let signable_json = serde_json::to_string(&signable)
            .map_err(|_| RejectCode::RejectSchema)?;

        let signature = identity.sign(signable_json.as_bytes());
        let public_key = identity.public_key_hex();

        Ok(Self {
            id, version: "1.0".into(), sender: identity.node_id().clone(),
            timestamp, ttl, content_hash, signature, public_key, provenance, payload,
        })
    }

    pub fn verify(&self) -> Result<(), RejectCode> {
        let now = Utc::now();
        let age = now.signed_duration_since(self.timestamp);
        if age.num_seconds() > self.ttl as i64 {
            return Err(RejectCode::RejectExpired);
        }

        let payload_json = serde_json::to_string(&self.payload)
            .map_err(|_| RejectCode::RejectSchema)?;
        let computed_hash = domain_separated_digest(payload_json.as_bytes());
        if computed_hash != self.content_hash {
            return Err(RejectCode::RejectHashMismatch);
        }

        let signable = SignableEnvelope {
            id: &self.id, version: &self.version, sender: &self.sender,
            timestamp: self.timestamp, ttl: self.ttl,
            content_hash: &self.content_hash, provenance: &self.provenance,
        };
        let signable_json = serde_json::to_string(&signable)
            .map_err(|_| RejectCode::RejectSchema)?;

        if !NodeIdentity::verify_with_hex(signable_json.as_bytes(), &self.signature, &self.public_key) {
            return Err(RejectCode::RejectSignature);
        }
        Ok(())
    }
}

#[derive(Serialize)]
struct SignableEnvelope<'a> {
    id: &'a str,
    version: &'a str,
    sender: &'a NodeId,
    timestamp: DateTime<Utc>,
    ttl: u64,
    content_hash: &'a str,
    provenance: &'a [String],
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_verify() {
        let id = NodeIdentity::generate();
        let envelope = PCIEnvelope::create(&id, "test payload".to_string(), 3600, vec![]).unwrap();
        assert!(envelope.verify().is_ok());
    }
}
