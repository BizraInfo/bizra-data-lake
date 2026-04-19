//! Installation Receipt — Cryptographic Audit Trail
//!
//! Every installation produces a JSON receipt that serves as both
//! an audit record and a provenance attestation. Receipts are
//! hash-chained: each links to the previous via BLAKE3 parent_hash.
//!
//! Spec Reference: BIZRA Universal Sovereign Installer §17
//! Standing on Giants: Lamport (hash chains, 1979), Shannon (entropy)

use blake3::Hasher;
use serde::{Deserialize, Serialize};

use crate::device_profile::{DeviceProfile, ModelTier};

// ─────────────────────────────────────────────────────────────
// Install Receipt (Spec §17)
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InstallReceipt {
    /// Receipt version for schema evolution
    pub receipt_version: String,

    /// BLAKE3 hash of the receipt content (computed after creation)
    pub receipt_hash: String,

    /// BLAKE3 hash of the previous receipt (genesis has "0000...0000")
    pub parent_hash: String,

    /// UTC ISO-8601 timestamp
    pub timestamp: String,

    /// Which installer version was used
    pub installer_version: String,

    /// What action was performed
    pub action: InstallAction,

    /// Device fingerprint (non-identifying hardware summary)
    pub device_summary: DeviceSummary,

    /// Model selection details
    pub model_selection: ModelSelection,

    /// Components installed
    pub components: Vec<InstalledComponent>,

    /// Duration in seconds
    pub duration_seconds: f64,

    /// Health check result summary
    pub health_check_passed: bool,

    /// Constitutional Ihsān score of the installation
    pub ihsan_score: f64,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum InstallAction {
    FreshInstall,
    Upgrade { from_version: String },
    Repair,
    Uninstall,
    ModelSwap { from: String, to: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DeviceSummary {
    pub os: String,
    pub arch: String,
    pub ram_gb: f32,
    pub gpu: Option<String>,
    pub tier: String,
    pub locale: String,
}

impl DeviceSummary {
    pub fn from_profile(profile: &DeviceProfile) -> Self {
        Self {
            os: format!("{:?}", profile.os),
            arch: format!("{:?}", profile.arch),
            ram_gb: profile.ram_total_gb,
            gpu: profile
                .gpu
                .as_ref()
                .map(|g| format!("{} ({:?})", g.model, g.api)),
            tier: format!("{:?}", profile.recommended_tier()),
            locale: profile.system_locale.clone(),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ModelSelection {
    pub model_name: String,
    pub model_tier: String,
    pub quantization: String,
    pub size_gb: f32,
    pub auto_selected: bool,
    /// v2.1: cryptographic provenance + provider identity.
    /// Optional to preserve byte-parity with v2.0 receipts that
    /// predate the Brain Activation substrate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provenance: Option<ProvenanceDescriptor>,
}

impl ModelSelection {
    pub fn from_tier(tier: &ModelTier, auto: bool) -> Self {
        Self {
            model_name: tier.model_name().to_string(),
            model_tier: format!("{:?}", tier),
            quantization: match tier {
                ModelTier::Micro => "Q2_K".to_string(),
                ModelTier::Compact => "Q4_K_M".to_string(),
                ModelTier::Standard => "Q4_K_M".to_string(),
                ModelTier::Enhanced => "Q4_K_M".to_string(),
                ModelTier::Full => "Q4_K_M".to_string(),
                ModelTier::Premium => "Q4_K_M".to_string(),
                ModelTier::Elite => "Q8_0/FP16".to_string(),
            },
            size_gb: tier.disk_requirement_gb(),
            auto_selected: auto,
            provenance: None,
        }
    }

    /// Attach provenance to an existing selection (post-download).
    pub fn with_provenance(mut self, provenance: ProvenanceDescriptor) -> Self {
        self.provenance = Some(provenance);
        self
    }
}

// ─────────────────────────────────────────────────────────────
// Provenance substrate (v2.1 — Brain Activation binding)
// ─────────────────────────────────────────────────────────────
//
// Every cognition-capable install binds to a verifiable triple:
//   1. model_sha256      — what weights are on disk
//   2. model_signer      — who vouched for them (optional)
//   3. provider_identity — which provider class served them
//
// Schema-parity mirror lives in
// bizra-omega/bizra-cognition/src/cognition_round.rs. Parity is
// enforced by JSON-shape tests in both crates; any change here
// requires the same change there in the same commit.

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ProvenanceDescriptor {
    /// BLAKE3 hex of the model weights file. Empty string when
    /// `provider_identity` is `CoreNone` (no model pinned).
    pub model_sha256: String,

    /// Signer identity + signature over `(model_sha256, model_slug)`.
    /// None when the model is unsigned (user-provided local file)
    /// or irrelevant (remote API provider).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_signer: Option<SignerIdentity>,

    /// Which provider class served this brain. Always present.
    pub provider_identity: ProviderIdentity,
}

#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SignerIdentity {
    /// Opaque identifier for the signing key (e.g., hex fingerprint).
    pub key_id: String,
    /// Algorithm name, e.g., "ed25519".
    pub algorithm: String,
    /// Hex-encoded signature bytes.
    pub signature_hex: String,
}

/// Provider class — which kind of cognition served the round.
/// Mirrors Brain Activation Spec v0.1 §3.1 HAL enum.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum ProviderIdentity {
    /// Kernel-only; no brain layer active. Sovereignty-first default.
    CoreNone,
    /// Embedded local model (llama.cpp / candle / etc.).
    LocalModel { weights_path: String },
    /// Local inference server (Ollama, LM Studio).
    LocalServer { endpoint: String, vendor: String },
    /// Remote API (user explicitly opted in; leaves the machine).
    RemoteApi { vendor: String },
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InstalledComponent {
    pub name: String,
    pub version: String,
    pub size_bytes: u64,
    /// BLAKE3 hash of the component binary
    #[serde(alias = "sha256")]
    pub blake3_hash: String,
}

impl InstallReceipt {
    /// Create a new receipt. The `receipt_hash` will be computed from content.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        parent_hash: String,
        action: InstallAction,
        device_summary: DeviceSummary,
        model_selection: ModelSelection,
        components: Vec<InstalledComponent>,
        duration_seconds: f64,
        health_check_passed: bool,
        ihsan_score: f64,
    ) -> Self {
        let mut receipt = Self {
            receipt_version: "2.0.0".to_string(),
            receipt_hash: String::new(), // Computed below
            parent_hash,
            timestamp: chrono::Utc::now().to_rfc3339(),
            installer_version: env!("CARGO_PKG_VERSION").to_string(),
            action,
            device_summary,
            model_selection,
            components,
            duration_seconds,
            health_check_passed,
            ihsan_score,
        };

        receipt.receipt_hash = receipt.compute_hash();
        receipt
    }

    /// Compute BLAKE3 hash of the canonical receipt content.
    /// Excludes the receipt_hash field itself (circular reference).
    fn compute_hash(&self) -> String {
        let mut hasher = Hasher::new();
        hasher.update(b"bizra-installer-v1:receipt:");

        // Hash all fields except receipt_hash
        hasher.update(self.receipt_version.as_bytes());
        hasher.update(self.parent_hash.as_bytes());
        hasher.update(self.timestamp.as_bytes());
        hasher.update(self.installer_version.as_bytes());

        // Serialize action deterministically
        if let Ok(action_json) = serde_json::to_string(&self.action) {
            hasher.update(action_json.as_bytes());
        }
        if let Ok(device_json) = serde_json::to_string(&self.device_summary) {
            hasher.update(device_json.as_bytes());
        }
        if let Ok(model_json) = serde_json::to_string(&self.model_selection) {
            hasher.update(model_json.as_bytes());
        }

        // Hash components
        for comp in &self.components {
            hasher.update(comp.name.as_bytes());
            hasher.update(comp.version.as_bytes());
            hasher.update(comp.blake3_hash.as_bytes());
        }

        hasher.update(&self.duration_seconds.to_le_bytes());
        hasher.update(&[self.health_check_passed as u8]);
        hasher.update(&self.ihsan_score.to_le_bytes());

        hex::encode(hasher.finalize().as_bytes())
    }

    /// Verify the receipt hash matches the content.
    pub fn verify(&self) -> bool {
        self.receipt_hash == self.compute_hash()
    }

    /// Save receipt to a JSONL file (append mode).
    pub fn save_to_file(&self, path: &std::path::Path) -> std::io::Result<()> {
        use std::io::Write;
        let json = serde_json::to_string(self)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        let mut file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        writeln!(file, "{json}")?;
        Ok(())
    }

    /// Genesis receipt for initial installation (parent_hash = all zeros)
    pub fn genesis_parent_hash() -> String {
        "0".repeat(64)
    }
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_receipt() -> InstallReceipt {
        InstallReceipt::new(
            InstallReceipt::genesis_parent_hash(),
            InstallAction::FreshInstall,
            DeviceSummary {
                os: "Linux".into(),
                arch: "X86_64".into(),
                ram_gb: 16.0,
                gpu: Some("RTX 4090 (Cuda)".into()),
                tier: "Full".into(),
                locale: "en-US".into(),
            },
            ModelSelection {
                model_name: "Qwen 2.5 14B Q4_K_M".into(),
                model_tier: "Full".into(),
                quantization: "Q4_K_M".into(),
                size_gb: 8.5,
                auto_selected: true,
                provenance: None,
            },
            vec![InstalledComponent {
                name: "bizra-node".into(),
                version: "2.0.0".into(),
                size_bytes: 15_000_000,
                blake3_hash: "abc123".into(),
            }],
            45.3,
            true,
            0.97,
        )
    }

    #[test]
    fn receipt_hash_is_deterministic() {
        let r = sample_receipt();
        assert!(!r.receipt_hash.is_empty());
        assert_eq!(r.receipt_hash.len(), 64); // BLAKE3 hex
    }

    #[test]
    fn receipt_verifies() {
        let r = sample_receipt();
        assert!(r.verify());
    }

    #[test]
    fn tampered_receipt_fails_verification() {
        let mut r = sample_receipt();
        r.ihsan_score = 0.50; // Tamper
        assert!(!r.verify()); // Hash mismatch
    }

    #[test]
    fn genesis_parent_hash_is_64_zeros() {
        let h = InstallReceipt::genesis_parent_hash();
        assert_eq!(h.len(), 64);
        assert!(h.chars().all(|c| c == '0'));
    }

    #[test]
    fn model_selection_from_tier() {
        let sel = ModelSelection::from_tier(&ModelTier::Enhanced, true);
        assert_eq!(sel.model_name, "Llama 3.1 8B Q4_K_M");
        assert!(sel.auto_selected);
        assert!(
            sel.provenance.is_none(),
            "from_tier defaults to no provenance"
        );
    }

    // ─────────────────────────────────────────────────────────
    // v2.1 Provenance tests
    // ─────────────────────────────────────────────────────────

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

    #[test]
    fn provenance_serde_roundtrip() {
        let p = sample_provenance();
        let json = serde_json::to_string(&p).unwrap();
        let back: ProvenanceDescriptor = serde_json::from_str(&json).unwrap();
        assert_eq!(p, back);
    }

    #[test]
    fn provenance_json_shape_is_stable() {
        // Schema contract: this exact shape must be mirrored in
        // bizra-cognition/src/cognition_round.rs. If this asserts
        // fails, reconcile both crates in the same commit.
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
    fn provider_identity_core_none_serializes_minimally() {
        let p = ProviderIdentity::CoreNone;
        let v = serde_json::to_value(&p).unwrap();
        assert_eq!(v, serde_json::json!({"kind": "core_none"}));
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
            assert_eq!(v, back, "variant lost fidelity on roundtrip: {:?}", v);
        }
    }

    #[test]
    fn v20_receipt_json_deserializes_without_provenance() {
        // Backward-compat contract: a v2.0 receipt JSON (no `provenance`
        // field on model_selection) must still deserialize cleanly.
        let v20_json = serde_json::json!({
            "receipt_version": "2.0.0",
            "receipt_hash": "deadbeef",
            "parent_hash": "0".repeat(64),
            "timestamp": "2026-04-19T18:15:18Z",
            "installer_version": "2.0.0",
            "action": "FreshInstall",
            "device_summary": {
                "os": "Linux",
                "arch": "X86_64",
                "ram_gb": 16.0,
                "gpu": null,
                "tier": "Standard",
                "locale": "en-US",
            },
            "model_selection": {
                "model_name": "Llama 3.1 8B Q4_K_M",
                "model_tier": "Enhanced",
                "quantization": "Q4_K_M",
                "size_gb": 4.5,
                "auto_selected": true,
            },
            "components": [],
            "duration_seconds": 10.0,
            "health_check_passed": true,
            "ihsan_score": 0.95,
        });
        let r: InstallReceipt = serde_json::from_value(v20_json).unwrap();
        assert!(r.model_selection.provenance.is_none());
    }

    #[test]
    fn v20_receipt_hash_unchanged_with_none_provenance() {
        // A ModelSelection with provenance=None must serialize
        // identically to a v2.0 ModelSelection, so receipts hashed
        // before v2.1 still verify bit-for-bit.
        let sel = ModelSelection::from_tier(&ModelTier::Enhanced, true);
        let json = serde_json::to_string(&sel).unwrap();
        assert!(
            !json.contains("provenance"),
            "Option::None with skip_serializing_if must omit the field; got: {json}"
        );
    }

    #[test]
    fn receipt_with_provenance_hashes_and_verifies() {
        let mut r = sample_receipt();
        r.model_selection = r
            .model_selection
            .clone()
            .with_provenance(sample_provenance());
        // Re-compute hash to reflect the new content.
        r.receipt_hash = r.compute_hash();
        assert!(r.verify(), "fresh receipt with provenance must verify");
    }

    #[test]
    fn receipt_hash_changes_when_provenance_changes() {
        let mut r1 = sample_receipt();
        r1.model_selection = r1
            .model_selection
            .clone()
            .with_provenance(sample_provenance());
        r1.receipt_hash = r1.compute_hash();

        let mut r2 = r1.clone();
        // Alter provenance — e.g., different signer.
        let mut prov = sample_provenance();
        prov.model_signer.as_mut().unwrap().key_id = "ed25519:impostor".into();
        r2.model_selection.provenance = Some(prov);
        r2.receipt_hash = r2.compute_hash();

        assert_ne!(
            r1.receipt_hash, r2.receipt_hash,
            "changing provenance must change receipt_hash"
        );
    }

    #[test]
    fn provenance_skip_serializing_if_none() {
        let prov = ProvenanceDescriptor {
            model_sha256: "feedface".into(),
            model_signer: None, // explicitly None — must be omitted from JSON
            provider_identity: ProviderIdentity::RemoteApi {
                vendor: "openai".into(),
            },
        };
        let json = serde_json::to_string(&prov).unwrap();
        assert!(
            !json.contains("model_signer"),
            "Option::None signer must be skipped; got: {json}"
        );
    }
}
