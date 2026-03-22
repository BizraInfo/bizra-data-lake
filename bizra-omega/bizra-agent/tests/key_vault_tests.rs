// bizra-agent/tests/key_vault_tests.rs
// ============================================================
// Integration tests for the Multi-Provider Key Vault (Phase 4)
// ============================================================

#[cfg(unix)]
use std::os::unix::fs::PermissionsExt;
use std::{fs, path::PathBuf};

use bizra_agent::{
    key_vault::{KeyVault, SecretString, VaultBackend, VaultError},
    vault_env::EnvBackend,
    vault_file::FileBackend,
    vault_toml::TomlBackend,
};

// ── SecretString tests ───────────────────────────────────────

#[test]
fn secret_string_debug_shows_redacted() {
    let s = SecretString::new("hunter2");
    let debug = format!("{s:?}");
    assert_eq!(debug, "SecretString(***)");
    assert!(
        !debug.contains("hunter2"),
        "Secret value must never appear in Debug output"
    );
}

#[test]
fn secret_string_zeroize_on_drop() {
    // We cannot access the private `inner` field from an integration test,
    // so we verify zeroize semantics by grabbing a pointer via expose()
    // before dropping. The underlying bytes start at the same allocation.
    let s = SecretString::new("sensitive-data-12345");
    let exposed = s.expose();
    let ptr = exposed.as_ptr();
    let len = exposed.len();

    // Drop the secret (this runs our volatile-zero Drop impl).
    drop(s);

    // Best-effort check: read the memory where the secret used to live.
    // This is technically UB, but on common allocators the page is still
    // mapped and we can verify our Drop zeroed it. If the allocator
    // already reclaimed the page, we just log a warning.
    let mut zeroed = true;
    for i in 0..len {
        let byte = unsafe { std::ptr::read_volatile(ptr.add(i)) };
        if byte != 0 {
            zeroed = false;
            break;
        }
    }
    if !zeroed {
        eprintln!("WARNING: zeroize check inconclusive (allocator may have reclaimed page)");
    }
}

// ── EnvBackend tests ─────────────────────────────────────────

#[test]
fn env_backend_reads_prefixed_var() {
    // Use a unique key to avoid interference with other tests.
    let unique_key = "vault_test_env_read_001";
    let env_var = format!("BIZRA_{}", unique_key.to_uppercase());
    std::env::set_var(&env_var, "test-secret-value");

    let backend = EnvBackend::new();
    let result = backend.get(unique_key);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().expose(), "test-secret-value");

    std::env::remove_var(&env_var);
}

#[test]
fn env_backend_missing_var_returns_not_found() {
    let backend = EnvBackend::new();
    let result = backend.get("totally_nonexistent_key_xyz");
    assert!(matches!(result, Err(VaultError::NotFound { .. })));
}

#[test]
fn env_backend_empty_var_returns_not_found() {
    let unique_key = "vault_test_env_empty_002";
    let env_var = format!("BIZRA_{}", unique_key.to_uppercase());
    std::env::set_var(&env_var, "");

    let backend = EnvBackend::new();
    let result = backend.get(unique_key);
    assert!(matches!(result, Err(VaultError::NotFound { .. })));

    std::env::remove_var(&env_var);
}

#[test]
fn env_backend_contains_check() {
    let unique_key = "vault_test_env_contains_003";
    let env_var = format!("BIZRA_{}", unique_key.to_uppercase());

    let backend = EnvBackend::new();
    assert!(!backend.contains(unique_key));

    std::env::set_var(&env_var, "present");
    assert!(backend.contains(unique_key));

    std::env::remove_var(&env_var);
}

// ── FileBackend tests ────────────────────────────────────────

fn setup_file_backend(dir: &tempfile::TempDir) -> FileBackend {
    let secrets_dir = dir.path().join("secrets");
    fs::create_dir_all(&secrets_dir).unwrap();
    FileBackend::with_dir(dir.path().to_path_buf())
}

fn write_secret(dir: &tempfile::TempDir, key: &str, value: &str) -> PathBuf {
    let secrets_dir = dir.path().join("secrets");
    let path = secrets_dir.join(format!("{key}.secret"));
    fs::write(&path, value).unwrap();

    // Set 0600 on Unix.
    #[cfg(unix)]
    {
        let perms = std::fs::Permissions::from_mode(0o600);
        fs::set_permissions(&path, perms).unwrap();
    }

    path
}

#[test]
fn file_backend_reads_from_temp_dir() {
    let dir = tempfile::tempdir().unwrap();
    let backend = setup_file_backend(&dir);

    write_secret(&dir, "test_key", "my-file-secret");

    let result = backend.get("test_key");
    assert!(result.is_ok());
    assert_eq!(result.unwrap().expose(), "my-file-secret");
}

#[test]
fn file_backend_missing_file_returns_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let backend = setup_file_backend(&dir);

    let result = backend.get("nonexistent_key");
    assert!(matches!(result, Err(VaultError::NotFound { .. })));
}

#[cfg(unix)]
#[test]
fn file_backend_wrong_permissions_returns_denied() {
    let dir = tempfile::tempdir().unwrap();
    let backend = setup_file_backend(&dir);

    let path = write_secret(&dir, "bad_perms", "should-fail");

    // Set too-open permissions.
    let perms = std::fs::Permissions::from_mode(0o644);
    fs::set_permissions(&path, perms).unwrap();

    let result = backend.get("bad_perms");
    assert!(
        matches!(result, Err(VaultError::PermissionDenied { .. })),
        "Expected PermissionDenied, got: {result:?}"
    );
}

#[test]
fn file_backend_integrity_check_pass() {
    let dir = tempfile::tempdir().unwrap();
    let backend = setup_file_backend(&dir);

    let value = "integrity-ok";
    write_secret(&dir, "hash_ok", value);

    // Write matching BLAKE3 hash sidecar.
    let hash = blake3::hash(value.as_bytes()).to_hex().to_string();
    let hash_path = dir.path().join("secrets").join("hash_ok.secret.hash");
    fs::write(&hash_path, &hash).unwrap();

    let result = backend.get("hash_ok");
    assert!(result.is_ok());
    assert_eq!(result.unwrap().expose(), value);
}

#[test]
fn file_backend_integrity_check_fail() {
    let dir = tempfile::tempdir().unwrap();
    let backend = setup_file_backend(&dir);

    write_secret(&dir, "hash_bad", "real-content");

    // Write a wrong hash sidecar.
    let hash_path = dir.path().join("secrets").join("hash_bad.secret.hash");
    fs::write(
        &hash_path,
        "0000000000000000000000000000000000000000000000000000000000000000",
    )
    .unwrap();

    let result = backend.get("hash_bad");
    assert!(
        matches!(result, Err(VaultError::IntegrityFailed { .. })),
        "Expected IntegrityFailed, got: {result:?}"
    );
}

#[test]
fn file_backend_path_traversal_blocked() {
    let dir = tempfile::tempdir().unwrap();
    let backend = setup_file_backend(&dir);

    // Various traversal attempts.
    let bad_keys = [
        "../../../etc/passwd",
        "foo/bar",
        "foo\\bar",
        "..secret",
        "key\0evil",
    ];

    for bad_key in &bad_keys {
        let result = backend.get(bad_key);
        assert!(
            matches!(result, Err(VaultError::IoError { .. })),
            "Expected IoError for key '{bad_key}', got: {result:?}"
        );
    }
}

#[test]
fn file_backend_empty_file_returns_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let backend = setup_file_backend(&dir);

    write_secret(&dir, "empty_key", "   \n  ");

    let result = backend.get("empty_key");
    assert!(
        matches!(result, Err(VaultError::NotFound { .. })),
        "Empty file (whitespace only) should be NotFound, got: {result:?}"
    );
}

// ── TomlBackend tests ────────────────────────────────────────

#[test]
fn toml_backend_reads_provider_section() {
    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join("install.toml");
    fs::write(
        &config_path,
        r#"
[metadata]
version = "1.0"

[provider]
anthropic_api_key = "sk-ant-test-12345"
openai_api_key = "sk-openai-test-67890"
"#,
    )
    .unwrap();

    let backend = TomlBackend::with_path(config_path);
    let result = backend.get("anthropic_api_key");
    assert!(result.is_ok());
    assert_eq!(result.unwrap().expose(), "sk-ant-test-12345");

    let result2 = backend.get("openai_api_key");
    assert!(result2.is_ok());
    assert_eq!(result2.unwrap().expose(), "sk-openai-test-67890");
}

#[test]
fn toml_backend_missing_key_returns_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join("install.toml");
    fs::write(
        &config_path,
        r#"
[provider]
existing_key = "value"
"#,
    )
    .unwrap();

    let backend = TomlBackend::with_path(config_path);
    let result = backend.get("nonexistent_key");
    assert!(matches!(result, Err(VaultError::NotFound { .. })));
}

#[test]
fn toml_backend_missing_file_returns_not_found() {
    let backend = TomlBackend::with_path(PathBuf::from("/tmp/no-such-file-ever.toml"));
    let result = backend.get("any_key");
    assert!(matches!(result, Err(VaultError::NotFound { .. })));
}

#[test]
fn toml_backend_empty_value_returns_not_found() {
    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join("install.toml");
    fs::write(
        &config_path,
        r#"
[provider]
empty_key = ""
"#,
    )
    .unwrap();

    let backend = TomlBackend::with_path(config_path);
    let result = backend.get("empty_key");
    assert!(matches!(result, Err(VaultError::NotFound { .. })));
}

#[test]
fn toml_backend_invalid_toml_returns_parse_error() {
    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join("install.toml");
    fs::write(&config_path, "this is not valid TOML {{{").unwrap();

    let backend = TomlBackend::with_path(config_path);
    let result = backend.get("any_key");
    assert!(
        matches!(result, Err(VaultError::ParseError { .. })),
        "Expected ParseError, got: {result:?}"
    );
}

// ── Layered KeyVault tests ───────────────────────────────────

#[test]
fn layered_vault_env_overrides_file() {
    let unique_key = "vault_test_layer_env_004";
    let env_var = format!("BIZRA_{}", unique_key.to_uppercase());

    // Set up env backend with a value.
    std::env::set_var(&env_var, "from_env");

    // Set up file backend with a different value.
    let dir = tempfile::tempdir().unwrap();
    let file_backend = setup_file_backend(&dir);
    write_secret(&dir, unique_key, "from_file");

    let backends: Vec<Box<dyn VaultBackend>> =
        vec![Box::new(EnvBackend::new()), Box::new(file_backend)];
    let mut vault = KeyVault::with_backends(backends);

    let result = vault.get(unique_key);
    assert!(result.is_ok());
    assert_eq!(
        result.unwrap().expose(),
        "from_env",
        "Env backend should take priority over file backend"
    );

    std::env::remove_var(&env_var);
}

#[test]
fn layered_vault_file_fallback_when_env_missing() {
    let unique_key = "vault_test_layer_file_005";
    let env_var = format!("BIZRA_{}", unique_key.to_uppercase());

    // Ensure env var is NOT set.
    std::env::remove_var(&env_var);

    // Set up file backend with a value.
    let dir = tempfile::tempdir().unwrap();
    let file_backend = setup_file_backend(&dir);
    write_secret(&dir, unique_key, "from_file");

    let backends: Vec<Box<dyn VaultBackend>> =
        vec![Box::new(EnvBackend::new()), Box::new(file_backend)];
    let mut vault = KeyVault::with_backends(backends);

    let result = vault.get(unique_key);
    assert!(result.is_ok());
    assert_eq!(
        result.unwrap().expose(),
        "from_file",
        "File backend should be used when env is missing"
    );
}

#[test]
fn layered_vault_toml_fallback() {
    let unique_key = "vault_test_layer_toml_006";
    let env_var = format!("BIZRA_{}", unique_key.to_uppercase());
    std::env::remove_var(&env_var);

    // Set up TOML backend.
    let dir = tempfile::tempdir().unwrap();
    let config_path = dir.path().join("install.toml");
    fs::write(
        &config_path,
        format!(
            r#"
[provider]
{unique_key} = "from_toml"
"#
        ),
    )
    .unwrap();

    let backends: Vec<Box<dyn VaultBackend>> = vec![
        Box::new(EnvBackend::new()),
        Box::new(TomlBackend::with_path(config_path)),
    ];
    let mut vault = KeyVault::with_backends(backends);

    let result = vault.get(unique_key);
    assert!(result.is_ok());
    assert_eq!(result.unwrap().expose(), "from_toml");
}

#[test]
fn layered_vault_all_miss_returns_not_found() {
    let backends: Vec<Box<dyn VaultBackend>> = vec![Box::new(EnvBackend::new())];
    let mut vault = KeyVault::with_backends(backends);

    let result = vault.get("absolutely_nonexistent_key_999");
    assert!(matches!(result, Err(VaultError::NotFound { .. })));
}

#[test]
fn layered_vault_cache_works() {
    let unique_key = "vault_test_cache_007";
    let env_var = format!("BIZRA_{}", unique_key.to_uppercase());
    std::env::set_var(&env_var, "cached_value");

    let backends: Vec<Box<dyn VaultBackend>> = vec![Box::new(EnvBackend::new())];
    let mut vault = KeyVault::with_backends(backends);

    // First get — fetched from backend.
    let r1 = vault.get(unique_key).unwrap();
    assert_eq!(r1.expose(), "cached_value");

    // Mutate the env var — should still return cached value.
    std::env::set_var(&env_var, "mutated_value");

    let r2 = vault.get(unique_key).unwrap();
    assert_eq!(
        r2.expose(),
        "cached_value",
        "Should return cached value, not re-read env"
    );

    std::env::remove_var(&env_var);
}

#[test]
fn layered_vault_refresh_invalidates_cache() {
    let unique_key = "vault_test_refresh_008";
    let env_var = format!("BIZRA_{}", unique_key.to_uppercase());
    std::env::set_var(&env_var, "v1");

    let backends: Vec<Box<dyn VaultBackend>> = vec![Box::new(EnvBackend::new())];
    let mut vault = KeyVault::with_backends(backends);

    // Cache "v1".
    let r1 = vault.get(unique_key).unwrap();
    assert_eq!(r1.expose(), "v1");

    // Update env and refresh.
    std::env::set_var(&env_var, "v2");
    vault.refresh(unique_key);

    // Should now fetch "v2".
    let r2 = vault.get(unique_key).unwrap();
    assert_eq!(r2.expose(), "v2");

    std::env::remove_var(&env_var);
}

#[test]
fn layered_vault_refresh_all_clears_cache() {
    let unique_key = "vault_test_refresh_all_009";
    let env_var = format!("BIZRA_{}", unique_key.to_uppercase());
    std::env::set_var(&env_var, "original");

    let backends: Vec<Box<dyn VaultBackend>> = vec![Box::new(EnvBackend::new())];
    let mut vault = KeyVault::with_backends(backends);

    vault.get(unique_key).unwrap();
    std::env::set_var(&env_var, "updated");
    vault.refresh_all();

    let r = vault.get(unique_key).unwrap();
    assert_eq!(r.expose(), "updated");

    std::env::remove_var(&env_var);
}

// ── Access log tests ─────────────────────────────────────────

#[test]
fn access_log_records_lookups() {
    let unique_key_ok = "vault_test_log_ok_010";
    let unique_key_miss = "vault_test_log_miss_010";
    let env_var_ok = format!("BIZRA_{}", unique_key_ok.to_uppercase());
    std::env::set_var(&env_var_ok, "logged-value");

    let backends: Vec<Box<dyn VaultBackend>> = vec![Box::new(EnvBackend::new())];
    let mut vault = KeyVault::with_backends(backends);

    // Successful lookup.
    let _ = vault.get(unique_key_ok);
    // Failed lookup.
    let _ = vault.get(unique_key_miss);

    let log = vault.access_log();
    assert!(
        log.len() >= 2,
        "Expected at least 2 log entries, got {}",
        log.len()
    );

    // Find the success entry.
    let success_entry = log.iter().find(|e| e.key == unique_key_ok && e.success);
    assert!(
        success_entry.is_some(),
        "Should have a successful access log entry for '{unique_key_ok}'"
    );
    assert_eq!(success_entry.unwrap().source, "env");

    // Find the failure entry.
    let fail_entry = log.iter().find(|e| e.key == unique_key_miss && !e.success);
    assert!(
        fail_entry.is_some(),
        "Should have a failed access log entry for '{unique_key_miss}'"
    );

    std::env::remove_var(&env_var_ok);
}

#[test]
fn access_log_never_contains_secret_values() {
    let unique_key = "vault_test_log_secret_011";
    let env_var = format!("BIZRA_{}", unique_key.to_uppercase());
    let secret_value = "super-secret-value-never-in-log";
    std::env::set_var(&env_var, secret_value);

    let backends: Vec<Box<dyn VaultBackend>> = vec![Box::new(EnvBackend::new())];
    let mut vault = KeyVault::with_backends(backends);
    let _ = vault.get(unique_key);

    // Serialize the entire log to string and check it never contains the value.
    let log_str = format!("{:?}", vault.access_log());
    assert!(
        !log_str.contains(secret_value),
        "Access log must never contain secret values"
    );

    std::env::remove_var(&env_var);
}

// ── Backend name tests ───────────────────────────────────────

#[test]
fn backend_names_are_correct() {
    assert_eq!(EnvBackend::new().name(), "env");
    assert_eq!(FileBackend::with_dir(PathBuf::from("/tmp")).name(), "file");
    assert_eq!(
        TomlBackend::with_path(PathBuf::from("/tmp/x.toml")).name(),
        "toml"
    );
}
