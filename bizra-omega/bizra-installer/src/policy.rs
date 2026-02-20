//! Policy hash canonicalization
//!
//! Ensures deterministic BLAKE3 hashing of policy files regardless of
//! platform line endings. All content is normalized to LF before hashing.

use anyhow::{Context, Result};
use std::path::Path;

/// Normalize policy content to a canonical form:
/// - Replace all CRLF sequences with LF
/// - Trim trailing whitespace from each line
/// - Return the canonical string
pub fn canonicalize_policy(content: &str) -> String {
    content
        .replace("\r\n", "\n")
        .lines()
        .map(|line| line.trim_end())
        .collect::<Vec<&str>>()
        .join("\n")
}

/// Compute the BLAKE3 hash of the canonicalized policy content.
/// Returns a 64-character lowercase hex string.
pub fn compute_policy_hash(content: &str) -> String {
    let canonical = canonicalize_policy(content);
    let hash = blake3::hash(canonical.as_bytes());
    hash.to_hex().to_string()
}

/// Read a policy file from disk, canonicalize its content, and return
/// the BLAKE3 hash as a 64-character lowercase hex string.
pub fn compute_policy_hash_from_file(path: &Path) -> Result<String> {
    let content = std::fs::read_to_string(path)
        .with_context(|| format!("Failed to read policy file: {}", path.display()))?;
    Ok(compute_policy_hash(&content))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crlf_and_lf_produce_same_hash() {
        let lf_content = "line one\nline two\nline three";
        let crlf_content = "line one\r\nline two\r\nline three";

        let hash_lf = compute_policy_hash(lf_content);
        let hash_crlf = compute_policy_hash(crlf_content);

        assert_eq!(hash_lf, hash_crlf);
        assert_eq!(hash_lf.len(), 64);
    }

    #[test]
    fn known_hash_for_known_input() {
        let content = "hello policy";
        let canonical = canonicalize_policy(content);
        let expected = blake3::hash(canonical.as_bytes()).to_hex().to_string();

        let actual = compute_policy_hash(content);
        assert_eq!(actual, expected);
        assert_eq!(actual.len(), 64);
        // Verify it is lowercase hex
        assert!(actual.chars().all(|c| c.is_ascii_hexdigit()));
        assert_eq!(actual, actual.to_lowercase());
    }

    #[test]
    fn trailing_whitespace_is_trimmed() {
        let with_spaces = "line one   \nline two  \n";
        let clean = "line one\nline two\n";

        assert_eq!(
            compute_policy_hash(with_spaces),
            compute_policy_hash(clean)
        );
    }

    #[test]
    fn canonicalize_strips_trailing_spaces_per_line() {
        let input = "alpha   \r\nbeta  \ngamma \r\n";
        let result = canonicalize_policy(input);
        for line in result.lines() {
            assert_eq!(line, line.trim_end());
        }
    }

    #[test]
    fn hash_from_file_missing_file_returns_error() {
        let result = compute_policy_hash_from_file(Path::new("/nonexistent/policy.txt"));
        assert!(result.is_err());
    }

    #[test]
    fn hash_from_file_works_with_tempfile() {
        let dir = tempfile::tempdir().expect("create temp dir");
        let path = dir.path().join("policy.txt");
        std::fs::write(&path, "test policy content\r\n").expect("write");

        let hash = compute_policy_hash_from_file(&path).expect("hash");
        assert_eq!(hash.len(), 64);

        // Should match direct computation
        let direct = compute_policy_hash("test policy content\r\n");
        assert_eq!(hash, direct);
    }
}
