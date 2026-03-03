# Phase 56.03: High — Infrastructure Secrets + Service Binding

> Standing on Giants: Lampson (access control, 1971) · NIST SP 800-190 (container security) · CIS Kubernetes Benchmark

## F8: K8s NODE_SECRET Not Consumed + Default File Permissions

### Current State

File: `bizra-omega/k8s/deployment.yaml:26-33`

```yaml
stringData:
  NODE_SECRET: "REPLACE_WITH_SECRET_MANAGER_REFERENCE"
```

The secret is defined but the Rust API binary (`main.rs:164-179`) never reads
`NODE_SECRET`. Instead, it auto-generates identity keys with default permissions:

```rust
fn load_or_create_identity_bytes() -> anyhow::Result<[u8; 32]> {
    let identity_file = identity_dir.join("identity.key");
    if identity_file.exists() {
        // reads existing
    } else {
        std::fs::create_dir_all(&identity_dir)?;
        // writes new key — no explicit chmod
    }
}
```

Consequences:
- Identity churn on every pod restart (no persistent key)
- Key file has umask-default permissions (possibly world-readable on shared volumes)
- The K8s Secret placeholder is never wired to the app

### Required Behavior

1. If `NODE_SECRET` env var is set, use it as the identity seed (deterministic key from secret)
2. If `NODE_SECRET` is not set, auto-generate but restrict file permissions to `0600`
3. K8s deployment must mount the secret as env var into the container

### Pseudocode

```rust
fn load_or_create_identity_bytes() -> anyhow::Result<[u8; 32]> {
    // Priority 1: env var (K8s secret mount)
    if let Ok(hex_key) = std::env::var("NODE_SECRET") {
        let bytes = hex::decode(hex_key.trim())?;
        let arr: [u8; 32] = bytes.try_into()
            .map_err(|_| anyhow!("NODE_SECRET must be 64 hex chars (32 bytes)"))?;
        tracing::info!("Identity loaded from NODE_SECRET env var");
        return Ok(arr);
    }

    // Priority 2: file on disk
    let identity_file = identity_dir.join("identity.key");
    if identity_file.exists() {
        let hex_key = std::fs::read_to_string(&identity_file)?;
        // ... existing parse logic ...
        return Ok(secret_array);
    }

    // Priority 3: auto-generate with restricted permissions
    std::fs::create_dir_all(&identity_dir)?;
    let identity = NodeIdentity::generate();
    let secret_bytes = identity.secret_bytes();
    let hex_key = hex::encode(&secret_bytes);

    // Write with 0600 permissions (owner read/write only)
    use std::os::unix::fs::OpenOptionsExt;
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .mode(0o600)
        .open(&identity_file)?;
    file.write_all(hex_key.as_bytes())?;

    tracing::warn!("Auto-generated identity key at {:?} — consider setting NODE_SECRET", identity_file);
    Ok(secret_bytes)
}
```

K8s deployment fix:
```yaml
containers:
  - name: bizra-api
    env:
      - name: NODE_SECRET
        valueFrom:
          secretKeyRef:
            name: bizra-secrets
            key: NODE_SECRET
```

### Files Modified

| File | Change |
|------|--------|
| `bizra-omega/bizra-api/src/main.rs` | Read `NODE_SECRET` env var; set file perms to `0600` |
| `bizra-omega/k8s/deployment.yaml` | Wire secret to env var via `secretKeyRef` |

### TDD Anchors

```rust
#[test]
fn test_identity_from_env_var() {
    std::env::set_var("NODE_SECRET", "a1b2...64hex...");
    let bytes = load_or_create_identity_bytes().unwrap();
    assert_eq!(bytes.len(), 32);
    std::env::remove_var("NODE_SECRET");
}

#[test]
fn test_identity_file_has_restricted_permissions() {
    let dir = tempdir().unwrap();
    // load_or_create_identity_bytes with dir override
    let metadata = std::fs::metadata(dir.path().join("identity.key")).unwrap();
    let mode = metadata.permissions().mode();
    assert_eq!(mode & 0o777, 0o600);
}
```

---

## F11: Infrastructure Defaults Expose Services Without Auth

### Current State

Multiple files expose services on `0.0.0.0` with weak or no authentication:

| File | Issue |
|------|-------|
| `deploy/node0/node0-manifest.yaml:208` | Redis `bind 0.0.0.0` with no `requirepass` |
| `deploy/elite-compose.yaml:135` | Grafana `GF_SECURITY_ADMIN_PASSWORD=admin` fallback |
| `deploy/node0/systemd-services/bizra-inference.service:46` | Ollama bound to `0.0.0.0` |
| `deploy/node0/systemd-services/bizra-api.service:64-65` | API `--host 0.0.0.0` |
| `deploy/node0/systemd-services/bizra-dashboard.service:45` | Dashboard `--host 0.0.0.0` |

### Required Behavior

1. All services bind to `127.0.0.1` by default
2. Redis MUST have `requirepass` set (via env var)
3. Grafana default password MUST NOT be `admin` — force change on first login
4. Systemd services use env file for host binding, defaulting to localhost

### Pseudocode

```yaml
# node0-manifest.yaml — Redis section:
command: >
  redis-server
  --bind 127.0.0.1
  --requirepass ${REDIS_PASSWORD}

# elite-compose.yaml — Grafana:
environment:
  GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_ADMIN_PASSWORD:?GRAFANA_ADMIN_PASSWORD must be set}
  GF_USERS_DEFAULT_THEME: dark

# systemd services — default to localhost:
# bizra-api.service:
ExecStart=/mnt/c/bizra-genesis-node/target/release/api_server \
    --host ${BIZRA_API_HOST:-127.0.0.1} \
    --port 3001

# bizra-inference.service:
ExecStart=ollama serve
Environment="OLLAMA_HOST=127.0.0.1:11434"

# bizra-dashboard.service:
ExecStart=... --host ${BIZRA_DASHBOARD_HOST:-127.0.0.1}
```

### Files Modified

| File | Change |
|------|--------|
| `deploy/node0/node0-manifest.yaml` | Redis: `--bind 127.0.0.1 --requirepass` |
| `deploy/elite-compose.yaml` | Grafana: `${GRAFANA_ADMIN_PASSWORD:?}` (fail if unset) |
| `deploy/node0/systemd-services/bizra-inference.service` | `OLLAMA_HOST=127.0.0.1:11434` |
| `deploy/node0/systemd-services/bizra-api.service` | `--host ${BIZRA_API_HOST:-127.0.0.1}` |
| `deploy/node0/systemd-services/bizra-dashboard.service` | `--host ${BIZRA_DASHBOARD_HOST:-127.0.0.1}` |
| `deploy/node0/.env.example` (new) | Document all required secrets |

### TDD Anchors

```bash
# tests/integration/test_infra_binding.sh (new, or pytest subprocess)

test_redis_not_bound_to_all_interfaces() {
    # Parse compose/manifest for bind address
    grep -q "127.0.0.1" deploy/node0/node0-manifest.yaml
}

test_grafana_no_default_admin() {
    # Ensure no literal "admin" password in compose
    ! grep -q 'ADMIN_PASSWORD.*admin' deploy/elite-compose.yaml
}

test_systemd_services_default_localhost() {
    for svc in bizra-api bizra-dashboard bizra-inference; do
        grep -q '127.0.0.1' deploy/node0/systemd-services/${svc}.service
    done
}
```

```python
# tests/integration/test_infra_defaults.py

def test_redis_manifest_has_requirepass():
    manifest = Path("deploy/node0/node0-manifest.yaml").read_text()
    assert "requirepass" in manifest

def test_systemd_api_defaults_to_localhost():
    service = Path("deploy/node0/systemd-services/bizra-api.service").read_text()
    assert "127.0.0.1" in service or "BIZRA_API_HOST:-127.0.0.1" in service
```
