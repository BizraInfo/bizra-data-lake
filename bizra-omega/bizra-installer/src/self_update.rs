//! Self-Update — Delta Patches for Sovereign Nodes
//!
//! Nodes must be able to update themselves without external dependence.
//! Updates use delta patches (binary diff) to minimize bandwidth.
//! Every update is verified via BLAKE3 checksum before applying.
//!
//! Spec Reference: BIZRA Universal Sovereign Installer §14
//! Standing on Giants: Lamport (state transitions), Torvalds (git delta)
//!
//! Constitutional: Updates NEVER bypass constitutional gates.
//! A malformed update is worse than no update.

use blake3::Hasher;
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────
// Update Manifest
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UpdateManifest {
    /// Current version this update applies to
    pub from_version: String,
    /// Target version after update
    pub to_version: String,
    /// Release timestamp (UTC ISO-8601)
    pub released_at: String,
    /// BLAKE3 hash of the full target binary
    #[serde(alias = "target_sha256")]
    pub target_blake3: String,
    /// Patch file URL (if delta update available)
    pub patch_url: Option<String>,
    /// BLAKE3 hash of the patch file
    #[serde(alias = "patch_sha256")]
    pub patch_blake3: Option<String>,
    /// Full binary URL (fallback if delta fails)
    pub full_url: String,
    /// Size of delta patch in bytes
    pub patch_size_bytes: Option<u64>,
    /// Size of full binary in bytes
    pub full_size_bytes: u64,
    /// Whether this update is mandatory (security fix)
    pub mandatory: bool,
    /// Release notes (plain text)
    pub release_notes: String,
    /// Minimum Ihsān score required to apply (constitutional gate)
    pub min_ihsan: f64,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum UpdateStrategy {
    /// Binary diff patch (smallest download)
    DeltaPatch,
    /// Full binary replacement
    FullReplace,
    /// No update needed
    UpToDate,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UpdateResult {
    pub strategy: UpdateStrategy,
    pub success: bool,
    pub from_version: String,
    pub to_version: String,
    pub bytes_downloaded: u64,
    pub detail: String,
}

// ─────────────────────────────────────────────────────────────
// Version Comparison
// ─────────────────────────────────────────────────────────────

/// Simple semver comparison. Returns true if `current` < `target`.
pub fn needs_update(current: &str, target: &str) -> bool {
    let parse = |v: &str| -> Vec<u32> {
        v.trim_start_matches('v')
            .split('.')
            .filter_map(|s| s.parse().ok())
            .collect()
    };
    let c = parse(current);
    let t = parse(target);
    c < t
}

// ─────────────────────────────────────────────────────────────
// Checksum Verification
// ─────────────────────────────────────────────────────────────

/// Verify a file's BLAKE3 hash matches the expected hash.
pub fn verify_file_checksum(path: &std::path::Path, expected_blake3: &str) -> Result<bool, String> {
    let data = std::fs::read(path).map_err(|e| format!("Cannot read {}: {e}", path.display()))?;
    let mut hasher = Hasher::new();
    hasher.update(b"bizra-installer-v1:self-update:");
    hasher.update(&data);
    let computed = hex::encode(hasher.finalize().as_bytes());
    Ok(computed == expected_blake3)
}

// ─────────────────────────────────────────────────────────────
// Update Executor
// ─────────────────────────────────────────────────────────────

/// Determine the best update strategy for a given manifest.
pub fn determine_strategy(manifest: &UpdateManifest, current_version: &str) -> UpdateStrategy {
    if !needs_update(current_version, &manifest.to_version) {
        return UpdateStrategy::UpToDate;
    }

    // Only use delta if patch is available AND from the right version
    if manifest.patch_url.is_some()
        && manifest.patch_blake3.is_some()
        && manifest.from_version == current_version
    {
        UpdateStrategy::DeltaPatch
    } else {
        UpdateStrategy::FullReplace
    }
}

/// Plan an update but don't execute it.
/// Returns what would happen if the update were applied.
pub fn plan_update(manifest: &UpdateManifest, current_version: &str) -> UpdateResult {
    let strategy = determine_strategy(manifest, current_version);

    let bytes = match strategy {
        UpdateStrategy::DeltaPatch => manifest.patch_size_bytes.unwrap_or(0),
        UpdateStrategy::FullReplace => manifest.full_size_bytes,
        UpdateStrategy::UpToDate => 0,
    };

    let detail = match strategy {
        UpdateStrategy::DeltaPatch => format!(
            "Delta patch: {} → {} ({} bytes)",
            current_version, manifest.to_version, bytes
        ),
        UpdateStrategy::FullReplace => {
            format!("Full download: {} ({} bytes)", manifest.to_version, bytes)
        }
        UpdateStrategy::UpToDate => "Already up to date".to_string(),
    };

    UpdateResult {
        strategy,
        success: false, // Not executed yet
        from_version: current_version.to_string(),
        to_version: manifest.to_version.clone(),
        bytes_downloaded: bytes,
        detail,
    }
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_manifest() -> UpdateManifest {
        UpdateManifest {
            from_version: "2.0.0".into(),
            to_version: "2.1.0".into(),
            released_at: "2026-06-01T00:00:00Z".into(),
            target_blake3: "abcd1234".into(),
            patch_url: Some("https://releases.bizra.ai/patches/2.0.0-2.1.0.bsdiff".into()),
            patch_blake3: Some("ef567890".into()),
            full_url: "https://releases.bizra.ai/bin/bizra-node-2.1.0".into(),
            patch_size_bytes: Some(500_000),
            full_size_bytes: 15_000_000,
            mandatory: false,
            release_notes: "Performance improvements".into(),
            min_ihsan: 0.95,
        }
    }

    #[test]
    fn needs_update_true() {
        assert!(needs_update("2.0.0", "2.1.0"));
        assert!(needs_update("1.9.9", "2.0.0"));
        assert!(needs_update("v1.0.0", "v1.0.1"));
    }

    #[test]
    fn needs_update_false() {
        assert!(!needs_update("2.1.0", "2.1.0"));
        assert!(!needs_update("2.2.0", "2.1.0"));
        assert!(!needs_update("3.0.0", "2.1.0"));
    }

    #[test]
    fn delta_patch_when_version_matches() {
        let m = sample_manifest();
        assert_eq!(determine_strategy(&m, "2.0.0"), UpdateStrategy::DeltaPatch);
    }

    #[test]
    fn full_replace_when_version_mismatch() {
        let m = sample_manifest();
        assert_eq!(determine_strategy(&m, "1.9.0"), UpdateStrategy::FullReplace);
    }

    #[test]
    fn up_to_date_when_current() {
        let m = sample_manifest();
        assert_eq!(determine_strategy(&m, "2.1.0"), UpdateStrategy::UpToDate);
        assert_eq!(determine_strategy(&m, "3.0.0"), UpdateStrategy::UpToDate);
    }

    #[test]
    fn plan_delta() {
        let m = sample_manifest();
        let plan = plan_update(&m, "2.0.0");
        assert_eq!(plan.strategy, UpdateStrategy::DeltaPatch);
        assert_eq!(plan.bytes_downloaded, 500_000);
        assert!(!plan.success); // Not executed
    }

    #[test]
    fn plan_full() {
        let m = sample_manifest();
        let plan = plan_update(&m, "1.5.0");
        assert_eq!(plan.strategy, UpdateStrategy::FullReplace);
        assert_eq!(plan.bytes_downloaded, 15_000_000);
    }

    #[test]
    fn plan_up_to_date() {
        let m = sample_manifest();
        let plan = plan_update(&m, "2.1.0");
        assert_eq!(plan.strategy, UpdateStrategy::UpToDate);
        assert_eq!(plan.bytes_downloaded, 0);
    }

    #[test]
    fn full_replace_when_no_patch() {
        let mut m = sample_manifest();
        m.patch_url = None;
        m.patch_blake3 = None;
        assert_eq!(determine_strategy(&m, "2.0.0"), UpdateStrategy::FullReplace);
    }
}
