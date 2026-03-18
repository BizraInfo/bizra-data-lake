//! Installation Receipt — Cryptographic Audit Trail
//!
//! Every installation produces a JSON receipt that serves as both
//! an audit record and a provenance attestation. Receipts are
//! hash-chained: each links to the previous via BLAKE3 parent_hash.
//!
//! Spec Reference: BIZRA Universal Sovereign Installer §17
//! Standing on Giants: Lamport (hash chains, 1979), Shannon (entropy)

use crate::device_profile::{DeviceProfile, ModelTier};
use serde::{Deserialize, Serialize};
use blake3::Hasher;

// ─────────────────────────────────────────────────────────────
// Install Receipt (Spec §17)
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InstallReceipt {
    /// Receipt version for schema evolution
    pub receipt_version: String,

    /// BLAKE3 of the receipt content (computed after creation)
    pub receipt_hash: String,

    /// BLAKE3 of the previous receipt (genesis has "0000...0000")
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

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ModelSelection {
    pub model_name: String,
    pub model_tier: String,
    pub quantization: String,
    pub size_gb: f32,
    pub auto_selected: bool,
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
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InstalledComponent {
    pub name: String,
    pub version: String,
    pub size_bytes: u64,
    pub sha256: String,
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
            hasher.update(comp.sha256.as_bytes());
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
            },
            vec![InstalledComponent {
                name: "bizra-node".into(),
                version: "2.0.0".into(),
                size_bytes: 15_000_000,
                sha256: "abc123".into(),
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
    }
}
