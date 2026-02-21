// bizra-agent/src/key_vault.rs
// ============================================================
// Multi-Provider Key Vault — Phase 4 (Sprint 3)
// ============================================================
// Layered secret resolution with zeroize-on-drop semantics.
//
// Resolution order (first match wins):
//   1. Environment variable   (BIZRA_<UPPER_KEY>)
//   2. File-based secret      ($BIZRA_VAULT_DIR/secrets/<key>.secret)
//   3. TOML config            (install.toml [provider].<key>)
//   4. Error::NotFound
//
// Standing on: Diffie-Hellman (1976), Lamport (1982), Al-Ghazali (1095)
// ============================================================

use std::collections::HashMap;
use std::fmt;
use std::sync::atomic::{compiler_fence, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::vault_env::EnvBackend;
use crate::vault_file::FileBackend;
use crate::vault_toml::TomlBackend;

// ── SecretString ─────────────────────────────────────────────

/// Secret string with zeroize-on-drop semantics.
///
/// **Security invariant:** NEVER appears in `Debug`, `Display`, or
/// `serde::Serialize` output. Memory is overwritten with zeros on drop
/// before deallocation.
pub struct SecretString {
    inner: Vec<u8>,
}

impl SecretString {
    /// Create a new secret from a string slice.
    pub fn new(s: &str) -> Self {
        Self {
            inner: s.as_bytes().to_vec(),
        }
    }

    /// Expose the secret value. Use sparingly and never log the result.
    pub fn expose(&self) -> &str {
        std::str::from_utf8(&self.inner).unwrap_or("")
    }

    /// Length of the secret in bytes.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Whether the secret is empty.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Constant-time comparison against a plaintext reference.
    /// Use this instead of `==` to prevent timing side-channels.
    pub fn ct_eq(&self, other: &[u8]) -> bool {
        constant_time_eq(&self.inner, other)
    }
}

/// Constant-time byte comparison to prevent timing attacks on secret comparisons.
/// Returns true if both slices are equal length and content-identical.
///
/// This function always compares every byte even when a difference is found early,
/// preventing an attacker from deducing partial matches via response timing.
pub fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    if a.len() != b.len() {
        return false;
    }
    let mut diff = 0u8;
    for (x, y) in a.iter().zip(b.iter()) {
        diff |= x ^ y;
    }
    diff == 0
}

impl Drop for SecretString {
    fn drop(&mut self) {
        // Overwrite memory with zeros before deallocation.
        for byte in &mut self.inner {
            // Use volatile-style write via ptr to discourage optimization.
            unsafe {
                std::ptr::write_volatile(byte as *mut u8, 0);
            }
        }
        // Compiler fence to prevent reordering past the zeroing.
        compiler_fence(Ordering::SeqCst);
    }
}

impl fmt::Debug for SecretString {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SecretString(***)")
    }
}

impl Clone for SecretString {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
        }
    }
}

// ── VaultError ───────────────────────────────────────────────

/// Errors returned by vault operations.
#[derive(Debug, Clone)]
pub enum VaultError {
    /// Key not found in any backend.
    NotFound { key: String },
    /// I/O error reading the secret.
    IoError { key: String, source: String },
    /// File permissions are too open.
    PermissionDenied { key: String, path: String },
    /// BLAKE3 integrity hash mismatch.
    IntegrityFailed {
        key: String,
        expected: String,
        actual: String,
    },
    /// TOML/config parse failure.
    ParseError { source: String },
}

impl fmt::Display for VaultError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            VaultError::NotFound { key } => write!(f, "secret not found: {key}"),
            VaultError::IoError { key, source } => {
                write!(f, "I/O error for secret '{key}': {source}")
            }
            VaultError::PermissionDenied { key, path } => {
                write!(f, "permission denied for '{key}': {path}")
            }
            VaultError::IntegrityFailed {
                key,
                expected,
                actual,
            } => {
                write!(
                    f,
                    "integrity check failed for '{key}': expected {expected}, got {actual}"
                )
            }
            VaultError::ParseError { source } => write!(f, "parse error: {source}"),
        }
    }
}

// ── VaultBackend trait ───────────────────────────────────────

/// Pluggable backend for secret storage.
pub trait VaultBackend {
    /// Retrieve a secret by key name.
    fn get(&self, key: &str) -> Result<SecretString, VaultError>;

    /// Check if the backend contains the given key (best-effort).
    fn contains(&self, key: &str) -> bool;

    /// Human-readable name for audit logs ("env", "file", "toml").
    fn name(&self) -> &str;
}

// ── Cache + Audit ────────────────────────────────────────────

/// Cached secret with TTL metadata.
struct CachedSecret {
    secret: SecretString,
    fetched_at: u64,
    source: String,
}

/// Audit entry for every vault access.
#[derive(Debug, Clone)]
pub struct VaultAccessEntry {
    /// Key name (NEVER the value).
    pub key: String,
    /// Which backend resolved it.
    pub source: String,
    /// Unix timestamp (seconds).
    pub timestamp: u64,
    /// Whether the lookup succeeded.
    pub success: bool,
}

// ── KeyVault ─────────────────────────────────────────────────

/// Layered secret resolver with caching and audit trail.
///
/// Not `Send`/`Sync` by design — each thread creates its own instance.
pub struct KeyVault {
    backends: Vec<Box<dyn VaultBackend>>,
    cache: HashMap<String, CachedSecret>,
    cache_ttl_secs: u64,
    access_log: Vec<VaultAccessEntry>,
}

/// Maximum audit log entries before trimming.
const MAX_LOG_ENTRIES: usize = 1000;
/// Entries removed when trimming.
const LOG_TRIM_COUNT: usize = 500;

impl KeyVault {
    /// Create a new vault with the default backend chain:
    /// env -> file -> toml.
    pub fn new() -> Self {
        let backends: Vec<Box<dyn VaultBackend>> = vec![
            Box::new(EnvBackend::new()),
            Box::new(FileBackend::new()),
            Box::new(TomlBackend::new()),
        ];
        Self {
            backends,
            cache: HashMap::new(),
            cache_ttl_secs: 300,
            access_log: Vec::new(),
        }
    }

    /// Create a vault with custom backends (for testing).
    pub fn with_backends(backends: Vec<Box<dyn VaultBackend>>) -> Self {
        Self {
            backends,
            cache: HashMap::new(),
            cache_ttl_secs: 300,
            access_log: Vec::new(),
        }
    }

    /// Resolve a secret by key, trying each backend in priority order.
    ///
    /// Results are cached for `cache_ttl_secs` (default 300s).
    pub fn get(&mut self, key: &str) -> Result<SecretString, VaultError> {
        // Check cache first.
        let now = current_time_secs();
        if let Some(cached) = self.cache.get(key) {
            if now.saturating_sub(cached.fetched_at) < self.cache_ttl_secs {
                let source = cached.source.clone();
                let secret = cached.secret.clone();
                self.log_access(key, &source, true);
                return Ok(secret);
            }
            // Cache expired — fall through to backends.
        }

        // Try each backend in priority order.
        // Collect results first to avoid borrow conflict with self.backends.
        let mut found: Option<(SecretString, String)> = None;
        let mut error_sources: Vec<String> = Vec::new();
        for backend in &self.backends {
            match backend.get(key) {
                Ok(secret) => {
                    found = Some((secret, backend.name().to_string()));
                    break;
                }
                Err(VaultError::NotFound { .. }) => {
                    // Try next backend.
                    continue;
                }
                Err(_other) => {
                    // Non-NotFound errors: record and continue to next backend.
                    error_sources.push(backend.name().to_string());
                    continue;
                }
            }
        }

        // Log any error sources encountered during iteration.
        for source in &error_sources {
            self.log_access(key, source, false);
        }

        if let Some((secret, source)) = found {
            self.cache.insert(
                key.to_string(),
                CachedSecret {
                    secret: secret.clone(),
                    fetched_at: now,
                    source: source.clone(),
                },
            );
            self.log_access(key, &source, true);
            return Ok(secret);
        }

        self.log_access(key, "none", false);
        Err(VaultError::NotFound {
            key: key.to_string(),
        })
    }

    /// Invalidate cache for a single key (next `get` re-reads from backend).
    pub fn refresh(&mut self, key: &str) {
        self.cache.remove(key);
    }

    /// Invalidate all cached secrets.
    pub fn refresh_all(&mut self) {
        self.cache.clear();
    }

    /// Access audit log. Contains key names and backends, never secret values.
    pub fn access_log(&self) -> &[VaultAccessEntry] {
        &self.access_log
    }

    /// Record an access event, trimming log if it exceeds capacity.
    fn log_access(&mut self, key: &str, source: &str, success: bool) {
        self.access_log.push(VaultAccessEntry {
            key: key.to_string(),
            source: source.to_string(),
            timestamp: current_time_secs(),
            success,
        });
        if self.access_log.len() > MAX_LOG_ENTRIES {
            self.access_log.drain(0..LOG_TRIM_COUNT);
        }
    }
}

impl Default for KeyVault {
    fn default() -> Self {
        Self::new()
    }
}

/// Returns the current Unix timestamp in seconds.
fn current_time_secs() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

// ── Unit tests ───────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn secret_string_debug_redacted() {
        let s = SecretString::new("hunter2");
        let debug = format!("{:?}", s);
        assert_eq!(debug, "SecretString(***)");
        assert!(!debug.contains("hunter2"));
    }

    #[test]
    fn secret_string_expose() {
        let s = SecretString::new("my-secret");
        assert_eq!(s.expose(), "my-secret");
        assert_eq!(s.len(), 9);
        assert!(!s.is_empty());
    }

    #[test]
    fn secret_string_empty() {
        let s = SecretString::new("");
        assert!(s.is_empty());
        assert_eq!(s.len(), 0);
    }

    #[test]
    fn secret_string_clone() {
        let s = SecretString::new("clone-me");
        let c = s.clone();
        assert_eq!(c.expose(), "clone-me");
    }

    #[test]
    fn vault_error_display() {
        let err = VaultError::NotFound {
            key: "test".to_string(),
        };
        let msg = format!("{err}");
        assert!(msg.contains("test"));
        assert!(!msg.is_empty());
    }

    #[test]
    fn constant_time_eq_equal_slices() {
        assert!(constant_time_eq(b"hello", b"hello"));
        assert!(constant_time_eq(b"", b""));
        assert!(constant_time_eq(b"\x00\xff", b"\x00\xff"));
    }

    #[test]
    fn constant_time_eq_different_content() {
        assert!(!constant_time_eq(b"hello", b"world"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"\x00", b"\x01"));
    }

    #[test]
    fn constant_time_eq_different_lengths() {
        assert!(!constant_time_eq(b"short", b"longer"));
        assert!(!constant_time_eq(b"a", b""));
        assert!(!constant_time_eq(b"", b"b"));
    }

    #[test]
    fn secret_string_ct_eq_matches() {
        let s = SecretString::new("my-token");
        assert!(s.ct_eq(b"my-token"));
        assert!(!s.ct_eq(b"wrong-token"));
        assert!(!s.ct_eq(b"my-toke"));
        assert!(!s.ct_eq(b"my-token-extra"));
    }
}
