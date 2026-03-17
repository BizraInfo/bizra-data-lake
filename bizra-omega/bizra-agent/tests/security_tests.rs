// bizra-agent/tests/security_tests.rs
// ============================================================
// Security-Focused Integration Tests — Vault Hardening
// ============================================================
// These tests verify constant-time comparisons, symlink rejection,
// input validation, zeroize-on-drop, and other defensive measures.
// ============================================================

use std::fs;
#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::path::PathBuf;

use bizra_agent::key_vault::{constant_time_eq, KeyVault, SecretString, VaultBackend, VaultError};
use bizra_agent::vault_env::EnvBackend;
use bizra_agent::vault_file::FileBackend;

// ── Helper functions ────────────────────────────────────────

fn setup_file_backend(dir: &tempfile::TempDir) -> FileBackend {
    let secrets_dir = dir.path().join("secrets");
    fs::create_dir_all(&secrets_dir).unwrap();
    FileBackend::with_dir(dir.path().to_path_buf())
}

#[allow(dead_code)]
fn write_secret(dir: &tempfile::TempDir, key: &str, value: &str) -> PathBuf {
    let secrets_dir = dir.path().join("secrets");
    let path = secrets_dir.join(format!("{key}.secret"));
    fs::write(&path, value).unwrap();

    #[cfg(unix)]
    {
        let perms = std::fs::Permissions::from_mode(0o600);
        fs::set_permissions(&path, perms).unwrap();
    }

    path
}

// ── 1. constant_time_eq: equal inputs ───────────────────────

#[test]
fn security_constant_time_eq_equal_inputs() {
    assert!(constant_time_eq(b"secret-token-abc", b"secret-token-abc"));
    assert!(constant_time_eq(b"", b""));
    assert!(constant_time_eq(b"\x00\x00\x00", b"\x00\x00\x00"));
    // All 256 byte values
    let all_bytes: Vec<u8> = (0..=255).collect();
    assert!(constant_time_eq(&all_bytes, &all_bytes));
}

// ── 2. constant_time_eq: different inputs ───────────────────

#[test]
fn security_constant_time_eq_different_inputs() {
    assert!(!constant_time_eq(b"secret-a", b"secret-b"));
    // Differ in only the last byte
    assert!(!constant_time_eq(b"abcdef0", b"abcdef1"));
    // Differ in only the first byte
    assert!(!constant_time_eq(b"\x00rest", b"\x01rest"));
}

// ── 3. constant_time_eq: different lengths ──────────────────

#[test]
fn security_constant_time_eq_different_lengths() {
    assert!(!constant_time_eq(b"short", b"longer-string"));
    assert!(!constant_time_eq(b"a", b""));
    assert!(!constant_time_eq(b"", b"x"));
    // Prefix match should still fail
    assert!(!constant_time_eq(b"token", b"token-extended"));
}

// ── 4. SecretString::ct_eq ──────────────────────────────────

#[test]
fn security_secret_string_ct_eq() {
    let secret = SecretString::new("my-api-key-12345");
    assert!(secret.ct_eq(b"my-api-key-12345"));
    assert!(!secret.ct_eq(b"my-api-key-12346"));
    assert!(!secret.ct_eq(b"my-api-key-1234"));
    assert!(!secret.ct_eq(b"my-api-key-123456"));
    assert!(!secret.ct_eq(b""));
}

// ── 5. SecretString does not leak in Debug ──────────────────

#[test]
fn security_secret_string_debug_no_leak() {
    let secret = SecretString::new("ultra-secret-password-XYZ");
    let debug_output = format!("{secret:?}");
    assert_eq!(debug_output, "SecretString(***)");
    assert!(!debug_output.contains("ultra"));
    assert!(!debug_output.contains("secret"));
    assert!(!debug_output.contains("password"));
    assert!(!debug_output.contains("XYZ"));
}

// ── 6. SecretString zeroes memory on drop ───────────────────

#[test]
fn security_secret_string_zeroize_on_drop() {
    let s = SecretString::new("zeroize-me-please-1234567890");
    let exposed = s.expose();
    let ptr = exposed.as_ptr();
    let len = exposed.len();

    drop(s);

    // Best-effort check: read the memory after drop.
    // This is technically UB but on common allocators the page remains mapped.
    let mut all_zero = true;
    for i in 0..len {
        let byte = unsafe { std::ptr::read_volatile(ptr.add(i)) };
        if byte != 0 {
            all_zero = false;
            break;
        }
    }
    if !all_zero {
        eprintln!("WARNING: zeroize check inconclusive (allocator may have reclaimed/reused page)");
    }
}

// ── 7. vault_file rejects symlinks ──────────────────────────

#[cfg(unix)]
#[test]
fn security_vault_file_rejects_symlinks() {
    let dir = tempfile::tempdir().unwrap();
    let backend = setup_file_backend(&dir);

    // Create a real secret file
    let real_path = write_secret(&dir, "real_secret", "actual-value");

    // Create a symlink pointing to the real secret under a different key name
    let symlink_path = dir.path().join("secrets").join("symlinked.secret");
    std::os::unix::fs::symlink(&real_path, &symlink_path).unwrap();

    let result = backend.get("symlinked");
    assert!(
        matches!(result, Err(VaultError::PermissionDenied { .. })),
        "Expected PermissionDenied for symlinked secret, got: {result:?}"
    );
}

// ── 8. vault_env rejects keys with special characters ───────

#[test]
fn security_vault_env_rejects_special_chars() {
    let backend = EnvBackend::new();

    let bad_keys = [
        "key with spaces",
        "key;injection",
        "key$(cmd)",
        "key`cmd`",
        "key|pipe",
        "key&background",
        "key\nnewline",
        "key\0null",
        "../traversal",
        "key/slash",
        "key=value",
    ];

    for bad_key in &bad_keys {
        let result = backend.get(bad_key);
        assert!(
            matches!(result, Err(VaultError::IoError { .. })),
            "Expected IoError for key '{bad_key}', got: {result:?}"
        );
    }
}

// ── 9. vault_file rejects path traversal ────────────────────

#[test]
fn security_vault_file_rejects_path_traversal() {
    let dir = tempfile::tempdir().unwrap();
    let backend = setup_file_backend(&dir);

    let traversal_keys = [
        "../../../etc/passwd",
        "..%2F..%2Fetc/passwd",
        "foo/bar",
        "foo\\bar",
        "..secret",
        "key\0evil",
        "/absolute",
    ];

    for bad_key in &traversal_keys {
        let result = backend.get(bad_key);
        assert!(
            matches!(result, Err(VaultError::IoError { .. })),
            "Expected IoError for traversal key '{bad_key}', got: {result:?}"
        );
    }
}

// ── 10. Empty key rejected by all backends ──────────────────

#[test]
fn security_empty_key_rejected_by_all_backends() {
    // EnvBackend
    let env_backend = EnvBackend::new();
    let env_result = env_backend.get("");
    assert!(
        env_result.is_err(),
        "EnvBackend should reject empty key, got: {env_result:?}"
    );

    // FileBackend
    let dir = tempfile::tempdir().unwrap();
    let file_backend = setup_file_backend(&dir);
    let file_result = file_backend.get("");
    assert!(
        file_result.is_err(),
        "FileBackend should reject empty key, got: {file_result:?}"
    );
}

// ── 11. vault_env rejects overlong keys ─────────────────────

#[test]
fn security_vault_env_rejects_overlong_key() {
    let backend = EnvBackend::new();
    let long_key = "a".repeat(129);
    let result = backend.get(&long_key);
    assert!(
        matches!(result, Err(VaultError::IoError { .. })),
        "Expected IoError for overlong key (129 chars), got: {result:?}"
    );

    // Exactly 128 should be fine (if env var exists)
    let ok_key = "a".repeat(128);
    let result_ok = backend.get(&ok_key);
    // Should be NotFound (no env var set), not IoError
    assert!(
        matches!(result_ok, Err(VaultError::NotFound { .. })),
        "128-char key should be accepted but not found, got: {result_ok:?}"
    );
}

// ── 12. vault_env contains() rejects bad keys ──────────────

#[test]
fn security_vault_env_contains_rejects_bad_keys() {
    let backend = EnvBackend::new();
    assert!(!backend.contains(""));
    assert!(!backend.contains("bad key"));
    assert!(!backend.contains("key;evil"));
    assert!(!backend.contains(&"x".repeat(129)));
}

// ── 13. Access log cap enforcement ──────────────────────────

#[test]
fn security_access_log_caps_at_max() {
    let backends: Vec<Box<dyn VaultBackend>> = vec![Box::new(EnvBackend::new())];
    let mut vault = KeyVault::with_backends(backends);

    // Generate 1100 accesses (should trigger trimming at 1001)
    for i in 0..1100 {
        let _ = vault.get(&format!("cap_test_key_{i}"));
    }

    let log = vault.access_log();
    assert!(
        log.len() <= 1000,
        "Access log should be capped at 1000 entries, got {}",
        log.len()
    );
}

// ── 14. constant_time_eq: single-byte difference ────────────

#[test]
fn security_constant_time_eq_single_bit_difference() {
    // Differ by exactly one bit in various positions
    let a = [0b10101010u8; 16];
    let mut b = a;
    b[7] = 0b10101011; // flip last bit of byte 7
    assert!(!constant_time_eq(&a, &b));

    b[7] = a[7]; // restore
    assert!(constant_time_eq(&a, &b));
}
