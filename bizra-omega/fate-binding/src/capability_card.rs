//! CapabilityCard - PCI-Signed Model Credentials
//!
//! Every model accepted into BIZRA receives a CapabilityCard that
//! certifies its validated capabilities. Cards are signed using
//! Ed25519 for fast verification and include expiration dates.

use ed25519_dalek::{Signer, SigningKey, Verifier, VerifyingKey, Signature};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Duration, Utc};

use crate::{ModelTier, TaskType, IHSAN_THRESHOLD, SNR_THRESHOLD};

/// CapabilityCard - A signed credential for validated models
#[napi(object)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CapabilityCard {
    /// Unique model identifier
    pub model_id: String,

    /// Human-readable model name
    pub model_name: String,

    /// Parameter count (approximate)
    pub parameter_count: Option<i64>,

    /// Quantization level
    pub quantization: String,

    /// Capability tier (EDGE, LOCAL, POOL)
    pub tier: String,

    /// Validated Ihsān score from challenge
    pub ihsan_score: f64,

    /// Validated SNR score from challenge
    pub snr_score: f64,

    /// Maximum context length
    pub max_context: i32,

    /// Measured latency in milliseconds
    pub latency_ms: i32,

    /// Supported task types
    pub tasks_supported: Vec<String>,

    /// Ed25519 signature (hex)
    pub signature: String,

    /// Issuer public key (hex)
    pub issuer_public_key: String,

    /// Issue timestamp (RFC3339)
    pub issued_at: String,

    /// Expiration timestamp (RFC3339)
    pub expires_at: String,

    /// Whether the card has been revoked
    pub revoked: bool,
}

impl CapabilityCard {
    /// Create a new CapabilityCard
    pub fn new(
        model_id: String,
        tier: ModelTier,
        tasks: Vec<TaskType>,
        ihsan_score: f64,
        snr_score: f64,
    ) -> Result<Self> {
        // Validate scores meet thresholds
        if ihsan_score < IHSAN_THRESHOLD {
            return Err(Error::from_reason(format!(
                "Ihsān score {} < threshold {}",
                ihsan_score, IHSAN_THRESHOLD
            )));
        }
        if snr_score < SNR_THRESHOLD {
            return Err(Error::from_reason(format!(
                "SNR score {} < threshold {}",
                snr_score, SNR_THRESHOLD
            )));
        }

        let now = Utc::now();
        let expires = now + Duration::days(90); // 90-day validity

        let tier_str = match tier {
            ModelTier::Edge => "EDGE",
            ModelTier::Local => "LOCAL",
            ModelTier::Pool => "POOL",
        };

        let tasks_str: Vec<String> = tasks.iter().map(|t| format!("{:?}", t)).collect();

        Ok(Self {
            model_id: model_id.clone(),
            model_name: model_id, // Default to ID
            parameter_count: None,
            quantization: "unknown".to_string(),
            tier: tier_str.to_string(),
            ihsan_score,
            snr_score,
            max_context: 2048, // Default
            latency_ms: 0,
            tasks_supported: tasks_str,
            signature: String::new(),
            issuer_public_key: String::new(),
            issued_at: now.to_rfc3339(),
            expires_at: expires.to_rfc3339(),
            revoked: false,
        })
    }

    /// Create the canonical bytes for signing
    fn canonical_bytes(&self) -> Vec<u8> {
        let mut data = Vec::new();
        data.extend(self.model_id.as_bytes());
        data.extend(self.tier.as_bytes());
        data.extend(self.ihsan_score.to_le_bytes());
        data.extend(self.snr_score.to_le_bytes());
        data.extend(self.issued_at.as_bytes());
        data.extend(self.expires_at.as_bytes());
        data
    }

    /// Check if the card is currently valid (not expired, not revoked)
    pub fn is_valid(&self) -> Result<bool> {
        if self.revoked {
            return Ok(false);
        }

        let expires: DateTime<Utc> = self.expires_at.parse()
            .map_err(|_| Error::from_reason("Invalid expiration date"))?;

        Ok(Utc::now() < expires)
    }
}

/// Card issuer for signing CapabilityCards
#[napi]
pub struct CardIssuer {
    signing_key: SigningKey,
    verifying_key: VerifyingKey,
}

#[napi]
impl CardIssuer {
    /// Create a new card issuer with a random keypair
    #[napi(constructor)]
    pub fn new() -> Result<Self> {
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();

        Ok(Self {
            signing_key,
            verifying_key,
        })
    }

    /// Get the issuer's public key (hex)
    #[napi]
    pub fn public_key_hex(&self) -> String {
        hex::encode(self.verifying_key.as_bytes())
    }

    /// Issue (sign) a CapabilityCard
    #[napi]
    pub fn issue(&self, card_json: String) -> Result<String> {
        let mut card: CapabilityCard = serde_json::from_str(&card_json)
            .map_err(|e| Error::from_reason(format!("Invalid card JSON: {}", e)))?;

        // Sign the canonical bytes
        let signature = self.signing_key.sign(&card.canonical_bytes());

        card.signature = hex::encode(signature.to_bytes());
        card.issuer_public_key = self.public_key_hex();

        serde_json::to_string(&card)
            .map_err(|e| Error::from_reason(format!("Serialization error: {}", e)))
    }

    /// Verify a CapabilityCard signature
    #[napi]
    pub fn verify_card(&self, card_json: String) -> Result<bool> {
        let card: CapabilityCard = serde_json::from_str(&card_json)
            .map_err(|e| Error::from_reason(format!("Invalid card JSON: {}", e)))?;

        // Decode signature
        let sig_bytes = hex::decode(&card.signature)
            .map_err(|_| Error::from_reason("Invalid signature hex"))?;

        let signature = Signature::from_slice(&sig_bytes)
            .map_err(|_| Error::from_reason("Invalid signature format"))?;

        // Decode issuer public key
        let pk_bytes = hex::decode(&card.issuer_public_key)
            .map_err(|_| Error::from_reason("Invalid public key hex"))?;

        let pk_array: [u8; 32] = pk_bytes.try_into()
            .map_err(|_| Error::from_reason("Public key must be 32 bytes"))?;

        let verifying_key = VerifyingKey::from_bytes(&pk_array)
            .map_err(|_| Error::from_reason("Invalid public key"))?;

        // Verify
        Ok(verifying_key.verify(&card.canonical_bytes(), &signature).is_ok())
    }
}

/// Standalone card verification (no issuer needed)
#[napi]
pub fn verify_capability_card(card_json: String) -> Result<CapabilityCardValidation> {
    let card: CapabilityCard = serde_json::from_str(&card_json)
        .map_err(|e| Error::from_reason(format!("Invalid card JSON: {}", e)))?;

    // Check expiration
    let is_expired = !card.is_valid().unwrap_or(false);

    // Verify signature
    let sig_bytes = hex::decode(&card.signature)
        .map_err(|_| Error::from_reason("Invalid signature hex"))?;

    let signature = Signature::from_slice(&sig_bytes)
        .map_err(|_| Error::from_reason("Invalid signature format"))?;

    let pk_bytes = hex::decode(&card.issuer_public_key)
        .map_err(|_| Error::from_reason("Invalid public key hex"))?;

    let pk_array: [u8; 32] = pk_bytes.try_into()
        .map_err(|_| Error::from_reason("Public key must be 32 bytes"))?;

    let verifying_key = VerifyingKey::from_bytes(&pk_array)
        .map_err(|_| Error::from_reason("Invalid public key"))?;

    let signature_valid = verifying_key.verify(&card.canonical_bytes(), &signature).is_ok();

    // Check scores still meet thresholds
    let ihsan_valid = card.ihsan_score >= IHSAN_THRESHOLD;
    let snr_valid = card.snr_score >= SNR_THRESHOLD;

    let is_valid = signature_valid && !is_expired && !card.revoked && ihsan_valid && snr_valid;

    let reason = if !signature_valid {
        Some("Invalid signature".to_string())
    } else if is_expired {
        Some("Card expired".to_string())
    } else if card.revoked {
        Some("Card revoked".to_string())
    } else if !ihsan_valid {
        Some(format!("Ihsān score {} below threshold", card.ihsan_score))
    } else if !snr_valid {
        Some(format!("SNR score {} below threshold", card.snr_score))
    } else {
        None
    };

    Ok(CapabilityCardValidation {
        is_valid,
        signature_valid,
        is_expired,
        is_revoked: card.revoked,
        ihsan_valid,
        snr_valid,
        model_id: card.model_id,
        tier: card.tier,
        reason,
    })
}

#[napi(object)]
#[derive(Debug, Clone)]
pub struct CapabilityCardValidation {
    pub is_valid: bool,
    pub signature_valid: bool,
    pub is_expired: bool,
    pub is_revoked: bool,
    pub ihsan_valid: bool,
    pub snr_valid: bool,
    pub model_id: String,
    pub tier: String,
    pub reason: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_capability_card_creation() {
        let card = CapabilityCard::new(
            "test-model".to_string(),
            ModelTier::Local,
            vec![TaskType::Chat],
            0.97,
            0.90,
        ).unwrap();

        assert_eq!(card.model_id, "test-model");
        assert_eq!(card.tier, "LOCAL");
        assert!(card.is_valid().unwrap());
    }

    #[test]
    fn test_card_signing_and_verification() {
        let issuer = CardIssuer::new().unwrap();

        let card = CapabilityCard::new(
            "test-model".to_string(),
            ModelTier::Edge,
            vec![TaskType::Chat, TaskType::Summarization],
            0.96,
            0.88,
        ).unwrap();

        let card_json = serde_json::to_string(&card).unwrap();
        let signed_json = issuer.issue(card_json).unwrap();

        assert!(issuer.verify_card(signed_json.clone()).unwrap());

        let validation = verify_capability_card(signed_json).unwrap();
        assert!(validation.is_valid);
        assert!(validation.signature_valid);
    }

    #[test]
    fn test_below_threshold_rejection() {
        let result = CapabilityCard::new(
            "bad-model".to_string(),
            ModelTier::Edge,
            vec![TaskType::Chat],
            0.90, // Below 0.95 threshold
            0.88,
        );

        assert!(result.is_err());
    }
}
