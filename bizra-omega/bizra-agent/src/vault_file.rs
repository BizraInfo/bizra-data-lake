// bizra-agent/src/vault_file.rs
// ============================================================
// File-Based Backend for KeyVault
// ============================================================
// Reads secrets from $BIZRA_VAULT_DIR/secrets/<key>.secret
// One file per secret, raw content (no JSON wrapping).
//
// Security:
//   - Unix: enforces 0600 permissions
//   - Path traversal blocked (/, \, .., \0 rejected in key names)
//   - Optional BLAKE3 integrity via <key>.secret.hash sidecar
// ============================================================

use std::fs;
use std::path::PathBuf;

use crate::key_vault::{SecretString, VaultBackend, VaultError};

/// Backend that reads secrets from individual files on disk.
pub struct FileBackend {
    vault_dir: PathBuf,
}

impl FileBackend {
    /// Create with directory from `$BIZRA_VAULT_DIR` or default `~/.bizra/vault/`.
    pub fn new() -> Self {
        let dir = std::env::var("BIZRA_VAULT_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| dirs_fallback().join(".bizra").join("vault"));
        Self { vault_dir: dir }
    }

    /// Create with an explicit vault directory (for testing).
    pub fn with_dir(dir: PathBuf) -> Self {
        Self { vault_dir: dir }
    }

    /// Validate that a key name contains no path traversal characters.
    fn validate_key(key: &str) -> Result<(), VaultError> {
        if key.contains('/')
            || key.contains('\\')
            || key.contains("..")
            || key.contains('\0')
            || key.is_empty()
        {
            return Err(VaultError::IoError {
                key: key.to_string(),
                source: "invalid key name: contains path traversal characters".to_string(),
            });
        }
        Ok(())
    }
}

impl Default for FileBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl VaultBackend for FileBackend {
    fn get(&self, key: &str) -> Result<SecretString, VaultError> {
        Self::validate_key(key)?;

        let secret_path = self.vault_dir.join("secrets").join(format!("{key}.secret"));

        if !secret_path.exists() {
            return Err(VaultError::NotFound {
                key: key.to_string(),
            });
        }

        // Reject symlinks to prevent secret redirection attacks.
        if secret_path.is_symlink() {
            return Err(VaultError::PermissionDenied {
                key: key.to_string(),
                path: format!(
                    "symlink detected at {}; refusing to follow",
                    secret_path.display()
                ),
            });
        }

        // Check file permissions (Unix only).
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            let metadata = fs::metadata(&secret_path).map_err(|e| VaultError::IoError {
                key: key.to_string(),
                source: e.to_string(),
            })?;
            let mode = metadata.permissions().mode() & 0o777;
            if mode != 0o600 {
                return Err(VaultError::PermissionDenied {
                    key: key.to_string(),
                    path: format!("mode {:o}, expected 600", mode),
                });
            }
        }

        let content = fs::read_to_string(&secret_path).map_err(|e| VaultError::IoError {
            key: key.to_string(),
            source: e.to_string(),
        })?;
        let secret = content.trim().to_string();

        // Optional BLAKE3 integrity check via sidecar file.
        let hash_path = secret_path.with_extension("secret.hash");
        if hash_path.exists() {
            let expected_hash = fs::read_to_string(&hash_path)
                .map_err(|e| VaultError::IoError {
                    key: key.to_string(),
                    source: format!("failed to read hash sidecar: {e}"),
                })?
                .trim()
                .to_string();

            let actual_hash = blake3::hash(secret.as_bytes()).to_hex().to_string();

            if expected_hash != actual_hash {
                return Err(VaultError::IntegrityFailed {
                    key: key.to_string(),
                    expected: expected_hash,
                    actual: actual_hash,
                });
            }
        }

        if secret.is_empty() {
            return Err(VaultError::NotFound {
                key: key.to_string(),
            });
        }

        Ok(SecretString::new(&secret))
    }

    fn contains(&self, key: &str) -> bool {
        if Self::validate_key(key).is_err() {
            return false;
        }
        self.vault_dir
            .join("secrets")
            .join(format!("{key}.secret"))
            .exists()
    }

    fn name(&self) -> &str {
        "file"
    }
}

/// Fallback home directory resolution (no external crate needed).
fn dirs_fallback() -> PathBuf {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("/tmp"))
}
