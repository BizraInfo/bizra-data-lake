# Phase 4: Multi-Provider Key Vault

## Sprint 3 — Alpha-100 Action Infrastructure (Secret Management)

> Standing on Giants: Diffie & Hellman (public key cryptography, 1976) · Lamport (BFT + trust boundaries, 1982) · Daemen & Rijmen (AES block cipher, 2001) · Al-Ghazali (Ihsan — never expose what harms, 1095)
> artifact: `bizra-node/src/action_executor.rs:352-353`, `bizra-agent/src/hash_namespace.rs`

---

## 1. Context

Sprint 2's `ActionExecutor::call_bridge()` reads a single env var:
```rust
let token = std::env::var("BIZRA_BRIDGE_TOKEN")
    .map_err(|_| ActionError::new("MISSING_BRIDGE_TOKEN", "BIZRA_BRIDGE_TOKEN not set"))?;
```

`filedfs/llm_bridge.js` reads API keys from env vars loaded by `loadProviderEnv()`:
```javascript
const key = process.env.ANTHROPIC_API_KEY || process.env.OPENAI_API_KEY;
```

**What exists:** Scattered env var reads across Rust and JS. No unified secret storage.
**What's missing:** A `KeyVault` abstraction with pluggable backends, key rotation, and zero-plaintext-in-memory guarantees.

---

## 2. Functional Requirements

### FR-1: KeyVault Trait
- Define a `KeyVault` trait with `get(key_name) -> Result<SecretString, VaultError>`
- `SecretString` type: wraps `String`, implements `Zeroize` on drop, no `Display`/`Debug` leak
- Pluggable: multiple backends implement the same trait

### FR-2: Environment Variable Backend
- Reads from `std::env::var(key_name)`
- Key name mapping: `BIZRA_BRIDGE_TOKEN` → vault key `"bridge_token"`
- Prefix convention: all BIZRA secrets use `BIZRA_` prefix in env
- This is the default backend (backward compatible with Sprint 1+2)

### FR-3: File-Based Backend
- Reads secrets from `$BIZRA_VAULT_DIR/secrets/<key_name>.secret` (one file per secret)
- Files contain raw secret value (no JSON wrapping)
- File permissions enforced: `0600` (owner read/write only) on Unix
- Directory: `$BIZRA_VAULT_DIR` env var, default `~/.bizra/vault/`
- Optional: BLAKE3 integrity hash in `<key_name>.secret.hash` sidecar file

### FR-4: TOML Config Backend
- Reads from `provider.env` or `install.toml` (Sprint 1 installer output)
- Parses `[provider]` section for API keys
- Lower priority than env vars and file backend

### FR-5: Layered Resolution
Resolution order (first match wins):
1. Environment variable (`BIZRA_<UPPER_KEY>`)
2. File-based secret (`$BIZRA_VAULT_DIR/secrets/<key>.secret`)
3. TOML config (`install.toml` → `[provider].<key>`)
4. Error: `VaultError::NotFound`

### FR-6: Key Rotation Support
- `KeyVault::refresh(key_name)` re-reads from backend (invalidates cache)
- Optional TTL cache: secrets cached for N seconds, auto-refresh on expiry
- Default TTL: 300 seconds (5 minutes)

### FR-7: Audit Trail
- Every `get()` call logs: key name (NOT value), backend source, timestamp
- Failed lookups logged at WARN level
- Log format compatible with audit JSONL (Phase 3 audit hook)

### FR-8: Security Invariants
- `SecretString` NEVER appears in `Debug`, `Display`, `serde::Serialize`
- Memory zeroed on drop (`zeroize` crate or manual)
- No secret written to stdout, stderr, or any log at any verbosity level
- `KeyVault` is `!Send` + `!Sync` if holding plaintext (single-thread safety)

---

## 3. Pseudocode

### 3.1 Types (`bizra-agent/src/key_vault.rs`)

```pseudocode
-- Secret string with zeroize-on-drop semantics
STRUCT SecretString:
    inner: Vec<u8>

IMPL SecretString:
    FUNCTION new(s: &str) -> Self:
        Self { inner: s.as_bytes().to_vec() }

    FUNCTION expose(&self) -> &str:
        str::from_utf8(&self.inner).unwrap_or("")

    FUNCTION len(&self) -> usize:
        self.inner.len()

    FUNCTION is_empty(&self) -> bool:
        self.inner.is_empty()

IMPL Drop FOR SecretString:
    FUNCTION drop(&mut self):
        -- Overwrite memory with zeros before dealloc
        FOR byte IN &mut self.inner:
            *byte = 0
        -- Compiler fence to prevent optimization
        compiler_fence(Ordering::SeqCst)

IMPL Debug FOR SecretString:
    FUNCTION fmt(&self, f) -> fmt::Result:
        write!(f, "SecretString(***)")  -- NEVER expose value

-- Vault errors
ENUM VaultError:
    NotFound { key: String }
    IoError { key: String, source: String }
    PermissionDenied { key: String, path: String }
    IntegrityFailed { key: String, expected: String, actual: String }
    ParseError { source: String }

-- Vault backend trait
TRAIT VaultBackend:
    FUNCTION get(&self, key: &str) -> Result<SecretString, VaultError>
    FUNCTION contains(&self, key: &str) -> bool
    FUNCTION name(&self) -> &str  -- "env", "file", "toml"

-- Layered vault (combines multiple backends)
STRUCT KeyVault:
    backends: Vec<Box<dyn VaultBackend>>
    cache: HashMap<String, CachedSecret>
    cache_ttl_secs: u64
    access_log: Vec<VaultAccessEntry>

STRUCT CachedSecret:
    secret: SecretString
    fetched_at: u64
    source: String

STRUCT VaultAccessEntry:
    key: String       -- key name (NOT value)
    source: String    -- which backend resolved it
    timestamp: u64
    success: bool
```

### 3.2 Layered Resolution

```pseudocode
IMPL KeyVault:
    FUNCTION new() -> Self:
        backends = vec![
            Box::new(EnvBackend::new()),
            Box::new(FileBackend::new()),
            Box::new(TomlBackend::new()),
        ]
        Self { backends, cache: HashMap::new(), cache_ttl_secs: 300, access_log: Vec::new() }

    FUNCTION get(&mut self, key: &str) -> Result<SecretString, VaultError>:
        -- Check cache first
        IF let Some(cached) = self.cache.get(key):
            now = current_time_secs()
            IF now - cached.fetched_at < self.cache_ttl_secs:
                self.log_access(key, &cached.source, true)
                RETURN Ok(cached.secret.clone())
            -- Cache expired, fall through to backends

        -- Try each backend in priority order (FR-5)
        FOR backend IN &self.backends:
            MATCH backend.get(key):
                Ok(secret):
                    self.cache.insert(key.to_string(), CachedSecret {
                        secret: secret.clone(),
                        fetched_at: current_time_secs(),
                        source: backend.name().to_string(),
                    })
                    self.log_access(key, backend.name(), true)
                    RETURN Ok(secret)
                Err(VaultError::NotFound { .. }):
                    CONTINUE  -- Try next backend
                Err(other):
                    -- Non-NotFound errors are real failures
                    self.log_access(key, backend.name(), false)
                    LOG_WARN("Vault backend '" + backend.name() + "' error for '" + key + "': " + other)
                    CONTINUE  -- Try next backend despite error

        self.log_access(key, "none", false)
        RETURN Err(VaultError::NotFound { key: key.to_string() })

    FUNCTION refresh(&mut self, key: &str):
        self.cache.remove(key)

    FUNCTION refresh_all(&mut self):
        self.cache.clear()

    FUNCTION log_access(&mut self, key: &str, source: &str, success: bool):
        self.access_log.push(VaultAccessEntry {
            key: key.to_string(),
            source: source.to_string(),
            timestamp: current_time_secs(),
            success,
        })
        -- Trim log to last 1000 entries
        IF self.access_log.len() > 1000:
            self.access_log.drain(0..500)

    FUNCTION access_log(&self) -> &[VaultAccessEntry]:
        &self.access_log
```

### 3.3 Environment Variable Backend

```pseudocode
STRUCT EnvBackend:
    prefix: String  -- default: "BIZRA_"

IMPL EnvBackend:
    FUNCTION new() -> Self:
        Self { prefix: "BIZRA_".to_string() }

IMPL VaultBackend FOR EnvBackend:
    FUNCTION get(&self, key: &str) -> Result<SecretString, VaultError>:
        -- Convert vault key to env var name
        -- "bridge_token" → "BIZRA_BRIDGE_TOKEN"
        env_key = self.prefix.clone() + key.to_uppercase().as_str()
        MATCH std::env::var(&env_key):
            Ok(value):
                IF value.is_empty():
                    RETURN Err(VaultError::NotFound { key: key.to_string() })
                RETURN Ok(SecretString::new(&value))
            Err(_):
                RETURN Err(VaultError::NotFound { key: key.to_string() })

    FUNCTION contains(&self, key: &str) -> bool:
        env_key = self.prefix.clone() + key.to_uppercase().as_str()
        std::env::var(&env_key).is_ok()

    FUNCTION name(&self) -> &str:
        "env"
```

### 3.4 File-Based Backend

```pseudocode
STRUCT FileBackend:
    vault_dir: PathBuf  -- $BIZRA_VAULT_DIR or ~/.bizra/vault/

IMPL FileBackend:
    FUNCTION new() -> Self:
        dir = std::env::var("BIZRA_VAULT_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| home_dir().join(".bizra/vault"))
        Self { vault_dir: dir }

IMPL VaultBackend FOR FileBackend:
    FUNCTION get(&self, key: &str) -> Result<SecretString, VaultError>:
        -- Sanitize key name (prevent path traversal)
        safe_key = key.replace(['/', '\\', '..', '\0'], "")
        IF safe_key != key:
            RETURN Err(VaultError::IoError { key, source: "Invalid key name" })

        secret_path = self.vault_dir.join("secrets").join(safe_key + ".secret")

        IF NOT secret_path.exists():
            RETURN Err(VaultError::NotFound { key: key.to_string() })

        -- Check file permissions (Unix only)
        #[cfg(unix)]
        {
            metadata = fs::metadata(&secret_path)?
            mode = metadata.permissions().mode() & 0o777
            IF mode != 0o600:
                RETURN Err(VaultError::PermissionDenied {
                    key: key.to_string(),
                    path: format!("mode {:o}, expected 600", mode),
                })
        }

        content = fs::read_to_string(&secret_path)?
        secret = content.trim().to_string()

        -- Optional integrity check
        hash_path = secret_path.with_extension("secret.hash")
        IF hash_path.exists():
            expected_hash = fs::read_to_string(&hash_path)?.trim().to_string()
            actual_hash = blake3::hash(secret.as_bytes()).to_hex().to_string()
            IF expected_hash != actual_hash:
                RETURN Err(VaultError::IntegrityFailed {
                    key: key.to_string(),
                    expected: expected_hash,
                    actual: actual_hash,
                })

        IF secret.is_empty():
            RETURN Err(VaultError::NotFound { key: key.to_string() })

        RETURN Ok(SecretString::new(&secret))

    FUNCTION contains(&self, key: &str) -> bool:
        safe_key = key.replace(['/', '\\', '..', '\0'], "")
        self.vault_dir.join("secrets").join(safe_key + ".secret").exists()

    FUNCTION name(&self) -> &str:
        "file"
```

### 3.5 TOML Config Backend

```pseudocode
STRUCT TomlBackend:
    config_path: PathBuf

IMPL TomlBackend:
    FUNCTION new() -> Self:
        -- Look for install.toml in standard locations
        path = std::env::var("BIZRA_CONFIG_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("install.toml"))
        Self { config_path: path }

IMPL VaultBackend FOR TomlBackend:
    FUNCTION get(&self, key: &str) -> Result<SecretString, VaultError>:
        IF NOT self.config_path.exists():
            RETURN Err(VaultError::NotFound { key: key.to_string() })

        content = fs::read_to_string(&self.config_path)?
        toml_value = toml::from_str(&content)?

        -- Look in [provider] section
        value = toml_value
            .get("provider")
            .and_then(|p| p.get(key))
            .and_then(|v| v.as_str())

        MATCH value:
            Some(s) IF NOT s.is_empty():
                RETURN Ok(SecretString::new(s))
            _:
                RETURN Err(VaultError::NotFound { key: key.to_string() })

    FUNCTION contains(&self, key: &str) -> bool:
        self.get(key).is_ok()

    FUNCTION name(&self) -> &str:
        "toml"
```

### 3.6 Integration into ActionExecutor

```pseudocode
-- Replace direct env::var in action_executor.rs:call_bridge()

-- BEFORE (Sprint 2):
let token = std::env::var("BIZRA_BRIDGE_TOKEN")
    .map_err(|_| ActionError::new("MISSING_BRIDGE_TOKEN", "..."))?;

-- AFTER (Sprint 3):
-- ActionExecutor now holds KeyVault
STRUCT ActionExecutor:
    ... (existing fields)
    key_vault: KeyVault

FUNCTION call_bridge(&mut self, method, params):
    secret = self.key_vault.get("bridge_token")
        .map_err(|e| ActionError::new("MISSING_BRIDGE_TOKEN",
            format!("Bridge token not found: {:?}", e).as_str()))?;
    token = secret.expose().to_string()
    -- Rest of call_bridge unchanged
```

---

## 4. File Inventory

| File | Action | ~Lines | Purpose |
|------|--------|--------|---------|
| `bizra-omega/bizra-agent/src/key_vault.rs` | CREATE | ~250 | KeyVault trait + SecretString + layered resolver |
| `bizra-omega/bizra-agent/src/vault_env.rs` | CREATE | ~40 | EnvBackend implementation |
| `bizra-omega/bizra-agent/src/vault_file.rs` | CREATE | ~70 | FileBackend implementation |
| `bizra-omega/bizra-agent/src/vault_toml.rs` | CREATE | ~50 | TomlBackend implementation |
| `bizra-omega/bizra-agent/src/lib.rs` | MODIFY | +4 | Add vault module declarations + re-exports |
| `bizra-omega/bizra-agent/Cargo.toml` | MODIFY | +1 | Add `toml` dependency (if not present) |
| `bizra-omega/bizra-node/src/action_executor.rs` | MODIFY | +10 | Replace `env::var` with `key_vault.get()` |
| `bizra-omega/bizra-agent/tests/key_vault_tests.rs` | CREATE | ~150 | Backend + integration tests |

---

## 5. TDD Anchors

```
TEST secret_string_zeroize_on_drop
  → Create SecretString with known content
  → Drop it
  → Expect: memory zeroed (verify via unsafe raw pointer inspection or miri)

TEST secret_string_debug_redacted
  → format!("{:?}", SecretString::new("hunter2"))
  → Expect: "SecretString(***)" — no plaintext

TEST env_backend_reads_prefixed_var
  → Set BIZRA_BRIDGE_TOKEN="test123"
  → EnvBackend::get("bridge_token")
  → Expect: Ok(SecretString("test123"))

TEST env_backend_missing_var
  → Unset BIZRA_NONEXISTENT
  → EnvBackend::get("nonexistent")
  → Expect: Err(VaultError::NotFound)

TEST file_backend_reads_secret
  → Write "mysecret" to $VAULT_DIR/secrets/test_key.secret with 0600 perms
  → FileBackend::get("test_key")
  → Expect: Ok(SecretString("mysecret"))

TEST file_backend_wrong_permissions
  → Write secret file with 0644 perms
  → FileBackend::get("test_key")
  → Expect: Err(VaultError::PermissionDenied)

TEST file_backend_integrity_check_pass
  → Write secret + matching .secret.hash sidecar
  → FileBackend::get("test_key")
  → Expect: Ok(...)

TEST file_backend_integrity_check_fail
  → Write secret + mismatched .secret.hash
  → FileBackend::get("test_key")
  → Expect: Err(VaultError::IntegrityFailed)

TEST file_backend_path_traversal_blocked
  → FileBackend::get("../../../etc/passwd")
  → Expect: Err(VaultError::IoError) — sanitized key doesn't match

TEST toml_backend_reads_provider_section
  → Write install.toml with [provider] anthropic_api_key = "sk-..."
  → TomlBackend::get("anthropic_api_key")
  → Expect: Ok(SecretString("sk-..."))

TEST toml_backend_missing_key
  → TomlBackend::get("nonexistent")
  → Expect: Err(VaultError::NotFound)

TEST layered_vault_env_overrides_file
  → Set BIZRA_TEST_KEY="from_env"
  → Write "from_file" to vault file
  → KeyVault::get("test_key")
  → Expect: Ok(SecretString("from_env")) — env has higher priority

TEST layered_vault_file_fallback
  → Unset BIZRA_TEST_KEY
  → Write "from_file" to vault file
  → KeyVault::get("test_key")
  → Expect: Ok(SecretString("from_file"))

TEST layered_vault_cache_ttl
  → Set BIZRA_TEST_KEY="v1"
  → vault.get("test_key") → "v1"
  → Set BIZRA_TEST_KEY="v2" (env mutation)
  → vault.get("test_key") → still "v1" (cached)
  → Advance time past TTL
  → vault.get("test_key") → "v2"

TEST layered_vault_refresh
  → vault.get("test_key") → cached
  → vault.refresh("test_key")
  → vault.get("test_key") → re-fetched from backend

TEST access_log_records_lookups
  → vault.get("key1") — success
  → vault.get("key2") — not found
  → vault.access_log() has 2 entries with correct key names and success flags
```

---

## 6. Integration Points

| From | To | Contract |
|------|----|----------|
| `action_executor.rs::call_bridge()` | `key_vault.rs::KeyVault::get("bridge_token")` | Returns `SecretString` |
| `filedfs/llm_bridge.js` | `BIZRA_ANTHROPIC_API_KEY` env var | JS reads env; vault sets env on start |
| `installer (Sprint 1)` | `install.toml [provider]` section | TOML backend reads this |
| `~/.bizra/vault/secrets/*.secret` | `vault_file.rs::FileBackend` | One file per secret, 0600 perms |
| `*.secret.hash` sidecar | `vault_file.rs` integrity check | BLAKE3 hash verification |

---

## 7. Edge Cases

- **Concurrent access:** `KeyVault` is not `Send`/`Sync`. Each thread/task creates its own instance with its own cache. Safe but duplicates reads — acceptable for Sprint 3 scope.
- **Secret file encoding:** UTF-8 only. Binary secrets not supported (use base64 encoding in file).
- **Empty secret file:** Treated as `NotFound` (empty string is not a valid secret).
- **TOML parse failure:** Returns `VaultError::ParseError`, falls through to next backend.
- **Windows permissions:** File permission check is `#[cfg(unix)]` only. On Windows, rely on NTFS ACLs — no enforcement in Sprint 3.
- **Key rotation during active connection:** `call_bridge()` reads token per-call. If token rotates between calls, next call uses new token. No session stickiness.

---

## 8. Non-Goals (Deferred)

- **Encrypted-at-rest secrets** — Sprint 3 stores plaintext in files; AES-256-GCM encryption is Sprint 4
- **Hardware Security Module (HSM)** — Pluggable trait allows future HSM backend, not implemented in Sprint 3
- **Cloud vault integration** — AWS Secrets Manager, HashiCorp Vault backends are Sprint 5+
- **Key generation/rotation automation** — Sprint 3 is read-only; key lifecycle management is Sprint 4
- **Multi-tenant isolation** — Single vault per node; per-user vault separation is Sprint 5+
