// bizra-agent/src/vault_toml.rs
// ============================================================
// TOML Config Backend for KeyVault
// ============================================================
// Reads secrets from install.toml [provider] section.
// Path: $BIZRA_CONFIG_PATH or default "install.toml"
// Lowest priority backend in the layered chain.
// ============================================================

use std::fs;
use std::path::PathBuf;

use crate::key_vault::{SecretString, VaultBackend, VaultError};

/// Backend that reads secrets from a TOML config file's `[provider]` section.
pub struct TomlBackend {
    config_path: PathBuf,
}

impl TomlBackend {
    /// Create with path from `$BIZRA_CONFIG_PATH` or default `install.toml`.
    pub fn new() -> Self {
        let path = std::env::var("BIZRA_CONFIG_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("install.toml"));
        Self { config_path: path }
    }

    /// Create with an explicit config path (for testing).
    pub fn with_path(path: PathBuf) -> Self {
        Self { config_path: path }
    }
}

impl Default for TomlBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl VaultBackend for TomlBackend {
    fn get(&self, key: &str) -> Result<SecretString, VaultError> {
        if !self.config_path.exists() {
            return Err(VaultError::NotFound {
                key: key.to_string(),
            });
        }

        let content = fs::read_to_string(&self.config_path).map_err(|e| VaultError::IoError {
            key: key.to_string(),
            source: e.to_string(),
        })?;

        let table: toml::Value = toml::from_str(&content).map_err(|e| VaultError::ParseError {
            source: e.to_string(),
        })?;

        let value = table
            .get("provider")
            .and_then(|p| p.get(key))
            .and_then(|v| v.as_str());

        match value {
            Some(s) if !s.is_empty() => Ok(SecretString::new(s)),
            _ => Err(VaultError::NotFound {
                key: key.to_string(),
            }),
        }
    }

    fn contains(&self, key: &str) -> bool {
        self.get(key).is_ok()
    }

    fn name(&self) -> &str {
        "toml"
    }
}
