//! Alpha-100 install configuration
//!
//! Manages the install.toml and provider.env files that describe a
//! completed Alpha-100 installation. All serialization uses TOML.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

/// Full install configuration for an Alpha-100 node.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Alpha100Config {
    /// Unique user hash (u32 for deterministic seeding)
    pub user_hash: u32,
    /// Provider type: "local", "anthropic", "openai"
    pub provider: String,
    /// Local backend: "ollama", "lmstudio"
    pub local_backend: String,
    /// Model identifier
    pub model: String,
    /// BLAKE3 policy hash (64-hex lowercase)
    pub policy_hash: String,
    /// Reflex mode: "disabled", "shadow", "active"
    pub reflex_mode: String,
    /// Absolute path to the bizra-node binary
    pub node_binary_path: String,
    /// Absolute path to the bridge script/binary
    pub bridge_path: String,
    /// Directory for persistent state
    pub state_dir: String,
    /// ISO 8601 timestamp of installation
    pub installed_at: String,
}

impl Alpha100Config {
    /// Create a default configuration for a given user hash.
    /// Uses reasonable defaults that can be overridden before saving.
    pub fn default_for_user(user_hash: u32) -> Self {
        let base = alpha100_dir();
        Self {
            user_hash,
            provider: "local".to_string(),
            local_backend: "ollama".to_string(),
            model: "llama3.1:8b".to_string(),
            policy_hash: String::new(),
            reflex_mode: "shadow".to_string(),
            node_binary_path: base
                .join("bin")
                .join("bizra-node")
                .to_string_lossy()
                .to_string(),
            bridge_path: base.join("bridge").to_string_lossy().to_string(),
            state_dir: base.join("state").to_string_lossy().to_string(),
            installed_at: chrono::Utc::now().to_rfc3339(),
        }
    }
}

/// Returns the canonical Alpha-100 directory: `~/.bizra/alpha100`
pub fn alpha100_dir() -> PathBuf {
    dirs::home_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join(".bizra")
        .join("alpha100")
}

/// Serialize and save the config as `install.toml` inside `base_dir`.
/// Creates `base_dir` if it does not exist.
pub fn save_config(config: &Alpha100Config, base_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(base_dir)
        .with_context(|| format!("Failed to create config dir: {}", base_dir.display()))?;

    let toml_str =
        toml::to_string_pretty(config).context("Failed to serialize Alpha100Config to TOML")?;

    let path = base_dir.join("install.toml");
    std::fs::write(&path, toml_str)
        .with_context(|| format!("Failed to write install.toml to {}", path.display()))?;

    Ok(())
}

/// Load the config from `install.toml` inside `base_dir`.
pub fn load_config(base_dir: &Path) -> Result<Alpha100Config> {
    let path = base_dir.join("install.toml");
    let content = std::fs::read_to_string(&path)
        .with_context(|| format!("Failed to read install.toml from {}", path.display()))?;

    let config: Alpha100Config =
        toml::from_str(&content).context("Failed to parse install.toml")?;

    Ok(config)
}

/// Write `provider.env` with provider-specific environment variables.
/// On Unix, the file is created with 0600 permissions (owner read/write only).
pub fn save_provider_env(config: &Alpha100Config, base_dir: &Path) -> Result<()> {
    std::fs::create_dir_all(base_dir)
        .with_context(|| format!("Failed to create env dir: {}", base_dir.display()))?;

    let env_content = format!(
        "# Alpha-100 provider configuration\n\
         # Generated at: {}\n\
         BIZRA_PROVIDER={}\n\
         BIZRA_LOCAL_BACKEND={}\n\
         BIZRA_MODEL={}\n\
         BIZRA_REFLEX_MODE={}\n\
         BIZRA_STATE_DIR={}\n",
        config.installed_at,
        config.provider,
        config.local_backend,
        config.model,
        config.reflex_mode,
        config.state_dir,
    );

    let path = base_dir.join("provider.env");
    std::fs::write(&path, &env_content)
        .with_context(|| format!("Failed to write provider.env to {}", path.display()))?;

    // Set 0600 permissions on Unix
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let perms = std::fs::Permissions::from_mode(0o600);
        std::fs::set_permissions(&path, perms)
            .with_context(|| format!("Failed to set permissions on {}", path.display()))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_for_user_creates_valid_config() {
        let config = Alpha100Config::default_for_user(42);
        assert_eq!(config.user_hash, 42);
        assert_eq!(config.provider, "local");
        assert_eq!(config.local_backend, "ollama");
        assert_eq!(config.reflex_mode, "shadow");
        assert!(!config.installed_at.is_empty());
    }

    #[test]
    fn save_and_load_roundtrip() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let config = Alpha100Config::default_for_user(99);

        save_config(&config, dir.path()).expect("save");
        let loaded = load_config(dir.path()).expect("load");

        assert_eq!(loaded.user_hash, 99);
        assert_eq!(loaded.provider, config.provider);
        assert_eq!(loaded.model, config.model);
        assert_eq!(loaded.reflex_mode, config.reflex_mode);
        assert_eq!(loaded.policy_hash, config.policy_hash);
    }

    #[test]
    fn load_config_missing_file_returns_error() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let result = load_config(dir.path());
        assert!(result.is_err());
    }

    #[test]
    fn save_provider_env_creates_file() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let config = Alpha100Config::default_for_user(7);

        save_provider_env(&config, dir.path()).expect("save env");

        let path = dir.path().join("provider.env");
        assert!(path.exists());

        let content = std::fs::read_to_string(&path).expect("read env");
        assert!(content.contains("BIZRA_PROVIDER=local"));
        assert!(content.contains("BIZRA_MODEL=llama3.1:8b"));
    }

    #[cfg(unix)]
    #[test]
    fn provider_env_has_restrictive_permissions() {
        use std::os::unix::fs::PermissionsExt;

        let dir = tempfile::tempdir().expect("create temp dir");
        let config = Alpha100Config::default_for_user(1);

        save_provider_env(&config, dir.path()).expect("save env");

        let path = dir.path().join("provider.env");
        let metadata = std::fs::metadata(&path).expect("metadata");
        let mode = metadata.permissions().mode() & 0o777;
        assert_eq!(mode, 0o600, "provider.env must have 0600 permissions");
    }

    #[test]
    fn alpha100_dir_ends_with_expected_path() {
        let dir = alpha100_dir();
        assert!(dir.ends_with(".bizra/alpha100") || dir.ends_with(".bizra\\alpha100"));
    }
}
