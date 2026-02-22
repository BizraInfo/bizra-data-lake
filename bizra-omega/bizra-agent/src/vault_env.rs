// bizra-agent/src/vault_env.rs
// ============================================================
// Environment Variable Backend for KeyVault
// ============================================================
// Reads secrets from environment variables with BIZRA_ prefix.
// Key mapping: vault key "bridge_token" -> env var "BIZRA_BRIDGE_TOKEN"
// ============================================================

use crate::key_vault::{SecretString, VaultBackend, VaultError};

/// Backend that reads secrets from environment variables.
pub struct EnvBackend {
    prefix: String,
}

impl EnvBackend {
    /// Create with default prefix "BIZRA_".
    pub fn new() -> Self {
        Self {
            prefix: "BIZRA_".to_string(),
        }
    }

    /// Convert a vault key to the corresponding env var name.
    /// "bridge_token" -> "BIZRA_BRIDGE_TOKEN"
    fn env_key(&self, key: &str) -> String {
        format!("{}{}", self.prefix, key.to_uppercase())
    }

    /// Validate that a key only contains safe characters for env var mapping.
    /// Accepts `[a-zA-Z0-9_]`, length 1-128. Rejects empty, too-long, or keys
    /// with shell-special characters that could cause injection.
    fn validate_key(key: &str) -> bool {
        !key.is_empty()
            && key.len() <= 128
            && key.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
    }
}

impl Default for EnvBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl VaultBackend for EnvBackend {
    fn get(&self, key: &str) -> Result<SecretString, VaultError> {
        if !Self::validate_key(key) {
            return Err(VaultError::IoError {
                key: key.to_string(),
                source: "invalid key: must be 1-128 ASCII alphanumeric or underscore characters"
                    .to_string(),
            });
        }
        let env_key = self.env_key(key);
        match std::env::var(&env_key) {
            Ok(value) if value.is_empty() => Err(VaultError::NotFound {
                key: key.to_string(),
            }),
            Ok(value) => Ok(SecretString::new(&value)),
            Err(_) => Err(VaultError::NotFound {
                key: key.to_string(),
            }),
        }
    }

    fn contains(&self, key: &str) -> bool {
        if !Self::validate_key(key) {
            return false;
        }
        let env_key = self.env_key(key);
        std::env::var(&env_key)
            .map(|v| !v.is_empty())
            .unwrap_or(false)
    }

    fn name(&self) -> &str {
        "env"
    }
}
