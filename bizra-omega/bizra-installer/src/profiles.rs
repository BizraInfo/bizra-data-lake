//! Multi-User Profile Manager
//!
//! Shared devices (family tablet, school lab, café PC) each user
//! gets their own sovereign identity and constellation of agents.
//! Profile switching uses PIN + biometric (if available).
//!
//! Spec Reference: BIZRA Universal Sovereign Installer §15
//! Standing on Giants: Al-Ghazali (sovereignty of the individual)
//!
//! Constitutional: Each profile is a sovereign node. Data isolation
//! is absolute — no cross-profile leakage.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};

// ─────────────────────────────────────────────────────────────
// User Profile
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct UserProfile {
    /// Unique profile ID (SHA-256 of creation timestamp + display_name)
    pub profile_id: String,
    /// Display name (user-chosen, not personal ID)
    pub display_name: String,
    /// Ed25519 public key for this profile's node identity
    pub public_key: String,
    /// Preferred locale
    pub locale: String,
    /// Whether this is the primary (first-created) profile
    pub is_primary: bool,
    /// Creation timestamp (UTC ISO-8601)
    pub created_at: String,
    /// Last active timestamp
    pub last_active: Option<String>,
    /// Profile-specific data directory (relative to install dir)
    pub data_dir: String,
    /// SHA-256 of PIN (stored securely; never the PIN itself)
    pub pin_hash: Option<String>,
}

impl UserProfile {
    pub fn new(display_name: &str, locale: &str, is_primary: bool) -> Self {
        let now = chrono::Utc::now().to_rfc3339();
        let mut hasher = Sha256::new();
        hasher.update(now.as_bytes());
        hasher.update(display_name.as_bytes());
        let profile_id = format!("{:x}", hasher.finalize());
        // Use first 16 chars for brevity
        let short_id = &profile_id[..16];

        Self {
            profile_id: short_id.to_string(),
            display_name: display_name.to_string(),
            public_key: String::new(), // Set during identity generation
            locale: locale.to_string(),
            is_primary,
            created_at: now,
            last_active: None,
            data_dir: format!("profiles/{short_id}"),
            pin_hash: None,
        }
    }

    /// Set a PIN for profile protection.
    /// The PIN is SHA-256 hashed with a salt (profile_id) before storage.
    pub fn set_pin(&mut self, pin: &str) {
        let mut hasher = Sha256::new();
        hasher.update(self.profile_id.as_bytes()); // Salt
        hasher.update(pin.as_bytes());
        self.pin_hash = Some(format!("{:x}", hasher.finalize()));
    }

    /// Verify a PIN against the stored hash.
    pub fn verify_pin(&self, pin: &str) -> bool {
        match &self.pin_hash {
            Some(stored) => {
                let mut hasher = Sha256::new();
                hasher.update(self.profile_id.as_bytes());
                hasher.update(pin.as_bytes());
                let computed = format!("{:x}", hasher.finalize());
                // Constant-time comparison to prevent timing attacks
                constant_time_eq(stored.as_bytes(), computed.as_bytes())
            }
            None => true, // No PIN set = no auth required
        }
    }

    /// Get the absolute data directory for this profile
    pub fn absolute_data_dir(&self, install_dir: &Path) -> PathBuf {
        install_dir.join(&self.data_dir)
    }
}

/// Constant-time byte comparison (prevents timing side-channel attacks)
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

// ─────────────────────────────────────────────────────────────
// Profile Manager
// ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProfileRegistry {
    pub profiles: Vec<UserProfile>,
    pub active_profile_id: Option<String>,
    pub max_profiles: u8,
}

impl Default for ProfileRegistry {
    fn default() -> Self {
        Self {
            profiles: Vec::new(),
            active_profile_id: None,
            max_profiles: 8, // Spec §15: max 8 profiles per device
        }
    }
}

impl ProfileRegistry {
    /// Create a new profile. Returns error if max reached.
    pub fn create_profile(
        &mut self,
        display_name: &str,
        locale: &str,
    ) -> Result<UserProfile, String> {
        if self.profiles.len() >= self.max_profiles as usize {
            return Err(format!("Maximum {} profiles reached", self.max_profiles));
        }

        // Check duplicate names
        if self.profiles.iter().any(|p| p.display_name == display_name) {
            return Err(format!("Profile '{}' already exists", display_name));
        }

        let is_primary = self.profiles.is_empty();
        let profile = UserProfile::new(display_name, locale, is_primary);

        // Auto-activate first profile
        if is_primary {
            self.active_profile_id = Some(profile.profile_id.clone());
        }

        self.profiles.push(profile.clone());
        Ok(profile)
    }

    /// Switch active profile (requires PIN if set)
    pub fn switch_profile(&mut self, profile_id: &str, pin: Option<&str>) -> Result<(), String> {
        let profile = self
            .profiles
            .iter()
            .find(|p| p.profile_id == profile_id)
            .ok_or_else(|| format!("Profile {} not found", profile_id))?;

        // Verify PIN if set
        if profile.pin_hash.is_some() {
            let pin = pin.ok_or("PIN required for this profile")?;
            if !profile.verify_pin(pin) {
                return Err("Invalid PIN".to_string());
            }
        }

        self.active_profile_id = Some(profile_id.to_string());

        // Update last_active timestamp
        if let Some(p) = self
            .profiles
            .iter_mut()
            .find(|p| p.profile_id == profile_id)
        {
            p.last_active = Some(chrono::Utc::now().to_rfc3339());
        }

        Ok(())
    }

    /// Get the currently active profile
    pub fn active_profile(&self) -> Option<&UserProfile> {
        self.active_profile_id
            .as_ref()
            .and_then(|id| self.profiles.iter().find(|p| &p.profile_id == id))
    }

    /// Remove a profile (cannot remove primary)
    pub fn remove_profile(&mut self, profile_id: &str) -> Result<(), String> {
        let is_primary = self
            .profiles
            .iter()
            .find(|p| p.profile_id == profile_id)
            .map(|p| p.is_primary)
            .ok_or_else(|| format!("Profile {} not found", profile_id))?;

        if is_primary {
            return Err("Cannot remove primary profile".to_string());
        }

        self.profiles.retain(|p| p.profile_id != profile_id);

        // If removed profile was active, switch to primary
        if self.active_profile_id.as_deref() == Some(profile_id) {
            self.active_profile_id = self
                .profiles
                .iter()
                .find(|p| p.is_primary)
                .map(|p| p.profile_id.clone());
        }

        Ok(())
    }

    /// List all profiles (for switch screen)
    pub fn list_profiles(&self) -> &[UserProfile] {
        &self.profiles
    }
}

/// Save registry to disk
pub fn save_registry(registry: &ProfileRegistry, path: &Path) -> std::io::Result<()> {
    let json = serde_json::to_string_pretty(registry)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    std::fs::write(path, json)
}

/// Load registry from disk
pub fn load_registry(path: &Path) -> std::io::Result<ProfileRegistry> {
    let content = std::fs::read_to_string(path)?;
    serde_json::from_str(&content)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))
}

// ─────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────

#[cfg(test)]
#[allow(clippy::field_reassign_with_default)]
mod tests {
    use super::*;

    #[test]
    fn create_primary_profile() {
        let mut reg = ProfileRegistry::default();
        let p = reg.create_profile("Alice", "en").unwrap();
        assert!(p.is_primary);
        assert_eq!(reg.active_profile().unwrap().display_name, "Alice");
    }

    #[test]
    fn create_secondary_profile() {
        let mut reg = ProfileRegistry::default();
        reg.create_profile("Alice", "en").unwrap();
        let p2 = reg.create_profile("Bob", "ar").unwrap();
        assert!(!p2.is_primary);
        assert_eq!(reg.profiles.len(), 2);
    }

    #[test]
    fn duplicate_name_rejected() {
        let mut reg = ProfileRegistry::default();
        reg.create_profile("Alice", "en").unwrap();
        assert!(reg.create_profile("Alice", "en").is_err());
    }

    #[test]
    fn max_profiles_enforced() {
        let mut reg = ProfileRegistry::default();
        reg.max_profiles = 2;
        reg.create_profile("A", "en").unwrap();
        reg.create_profile("B", "en").unwrap();
        assert!(reg.create_profile("C", "en").is_err());
    }

    #[test]
    fn pin_set_and_verify() {
        let mut profile = UserProfile::new("Test", "en", true);
        profile.set_pin("1234");
        assert!(profile.verify_pin("1234"));
        assert!(!profile.verify_pin("0000"));
        assert!(!profile.verify_pin(""));
    }

    #[test]
    fn profile_without_pin_always_verifies() {
        let profile = UserProfile::new("Test", "en", true);
        assert!(profile.verify_pin("anything"));
    }

    #[test]
    fn switch_profile_with_pin() {
        let mut reg = ProfileRegistry::default();
        let p1 = reg.create_profile("Alice", "en").unwrap();
        let mut p2 = reg.create_profile("Bob", "ar").unwrap();
        p2.set_pin("5678");
        // Update the stored profile with PIN
        if let Some(stored) = reg
            .profiles
            .iter_mut()
            .find(|p| p.profile_id == p2.profile_id)
        {
            stored.pin_hash = p2.pin_hash.clone();
        }

        // Wrong PIN
        assert!(reg.switch_profile(&p2.profile_id, Some("0000")).is_err());
        // No PIN when required
        assert!(reg.switch_profile(&p2.profile_id, None).is_err());
        // Correct PIN
        assert!(reg.switch_profile(&p2.profile_id, Some("5678")).is_ok());
        assert_eq!(reg.active_profile().unwrap().display_name, "Bob");

        // Switch back (no PIN on Alice)
        assert!(reg.switch_profile(&p1.profile_id, None).is_ok());
    }

    #[test]
    fn cannot_remove_primary() {
        let mut reg = ProfileRegistry::default();
        let p = reg.create_profile("Alice", "en").unwrap();
        assert!(reg.remove_profile(&p.profile_id).is_err());
    }

    #[test]
    fn remove_secondary_works() {
        let mut reg = ProfileRegistry::default();
        reg.create_profile("Alice", "en").unwrap();
        let p2 = reg.create_profile("Bob", "ar").unwrap();
        assert!(reg.remove_profile(&p2.profile_id).is_ok());
        assert_eq!(reg.profiles.len(), 1);
    }

    #[test]
    fn remove_active_profile_falls_back_to_primary() {
        let mut reg = ProfileRegistry::default();
        let _p1 = reg.create_profile("Alice", "en").unwrap();
        let p2 = reg.create_profile("Bob", "ar").unwrap();
        reg.switch_profile(&p2.profile_id, None).unwrap();
        reg.remove_profile(&p2.profile_id).unwrap();
        assert_eq!(reg.active_profile().unwrap().display_name, "Alice");
    }

    #[test]
    fn constant_time_eq_works() {
        assert!(constant_time_eq(b"hello", b"hello"));
        assert!(!constant_time_eq(b"hello", b"world"));
        assert!(!constant_time_eq(b"hello", b"hell"));
    }
}
