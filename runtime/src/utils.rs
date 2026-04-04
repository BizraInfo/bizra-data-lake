// src/utils.rs - Shared Utilities for BIZRA
//
// Extracted common patterns to eliminate duplication across modules.
// See scripts/check_parity.py for cross-boundary verification.

use chrono::{DateTime, Utc};
use sha2::{Digest, Sha256};

/// Generate SHA-256 hash of text, returning hex string
pub fn sha256_text(text: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(text.as_bytes());
    format!("{:x}", hasher.finalize())
}

/// Get current UTC timestamp in ISO 8601 format
pub fn utc_now_iso() -> String {
    Utc::now().to_rfc3339()
}

/// Get current UTC datetime
pub fn utc_now() -> DateTime<Utc> {
    Utc::now()
}

/// Normalize a key for comparison (lowercase, replace hyphens/spaces with underscores)
pub fn normalize_key(raw: &str) -> String {
    raw.trim().to_ascii_lowercase().replace(['-', ' '], "_")
}

/// Clamp a value to [0.0, 1.0] range
pub fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

/// Calculate average confidence from agent results
pub fn avg_confidence<I>(iter: I) -> f64
where
    I: Iterator<Item = f64>,
{
    let mut sum = 0.0;
    let mut count = 0usize;
    for value in iter {
        sum += value;
        count += 1;
    }
    if count == 0 {
        0.0
    } else {
        sum / count as f64
    }
}

/// Generate a unique ID with prefix
pub fn generate_id(prefix: &str, counter: u64) -> String {
    format!("{}-{:06}", prefix, counter)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sha256_text() {
        let hash = sha256_text("hello");
        assert_eq!(hash.len(), 64);
        // Known SHA-256 of "hello"
        assert_eq!(
            hash,
            "2cf24dba5fb0a30e26e83b2ac5b9e29e1b161e5c1fa7425e73043362938b9824"
        );
    }

    #[test]
    fn test_normalize_key() {
        assert_eq!(normalize_key("Hello-World"), "hello_world");
        assert_eq!(normalize_key("  TEST KEY  "), "test_key");
        assert_eq!(normalize_key("already_normal"), "already_normal");
    }

    #[test]
    fn test_clamp01() {
        assert_eq!(clamp01(0.5), 0.5);
        assert_eq!(clamp01(-0.1), 0.0);
        assert_eq!(clamp01(1.5), 1.0);
    }

    #[test]
    fn test_avg_confidence() {
        assert_eq!(avg_confidence([0.8, 0.9, 1.0].into_iter()), 0.9);
        assert_eq!(avg_confidence(std::iter::empty()), 0.0);
    }

    #[test]
    fn test_generate_id() {
        assert_eq!(generate_id("FATE", 42), "FATE-000042");
        assert_eq!(generate_id("RECEIPT", 1), "RECEIPT-000001");
    }
}
