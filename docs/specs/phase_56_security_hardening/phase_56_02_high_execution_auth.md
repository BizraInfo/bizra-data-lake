# Phase 56.02: High — Execution Sandboxes, Auth Middleware, Rate Limiting, Command Safety

> Standing on Giants: Saltzer & Schroeder (fail-closed default, 1975) · Tanenbaum (token bucket, 1981) · Thompson (Reflections on Trusting Trust, 1984) · OWASP (injection prevention)

## F3: ZPK Manifest Bypass — Unsigned Policy Fields

### Current State

File: `core/zpk/kernel.py:445-477`

The worker binary hash is signed and verified. But `version`, `policy_version`,
`ihsan_policy`, and `worker_uri` are read from the unsigned JSON manifest and
directly control policy gate decisions:

```python
artifact = WorkerArtifact(
    version=str(manifest.get("version", "unknown")),           # unsigned
    policy_version=int(manifest.get("policy_version", 1)),     # unsigned — gates policy
    ihsan_policy=float(manifest.get("ihsan_policy", 0.95)),    # unsigned — gates threshold
)
```

An attacker who can modify the manifest (but lacks the signing key) can set
`policy_version=999` to bypass version constraints, or `ihsan_policy=0.0` to
disable the Ihsan quality gate — while the worker binary itself still passes
signature verification.

### Required Behavior

The signed digest MUST cover all policy-relevant fields, not just the worker hash.
Compute a canonical digest over a deterministic serialization of:
`worker_hash + version + policy_version + ihsan_policy`

### Pseudocode

```
FUNCTION compute_manifest_digest(manifest: dict) -> str:
    """Canonical digest covering all policy-relevant fields."""
    canonical = json.dumps({
        "worker_hash": manifest["worker_hash"],
        "version": manifest["version"],
        "policy_version": manifest["policy_version"],
        "ihsan_policy": manifest["ihsan_policy"],
    }, sort_keys=True, separators=(",", ":"))
    RETURN hashlib.blake2b(canonical.encode(), digest_size=32).hexdigest()

# In _fetch_and_verify():
    # Existing: verify worker binary hash
    worker_hash = hex_digest(worker_bytes)
    IF NOT verify_digest_match(worker_hash, expected_hash):
        RETURN None, receipt(error="digest_mismatch")

    # NEW: verify manifest digest covers policy fields
    manifest_digest = compute_manifest_digest(manifest)
    manifest_signature = str(manifest.get("manifest_signature", ""))
    IF NOT manifest_signature:
        # Backward compat: fall back to worker_hash-only signature with warning
        LOG.warning("manifest uses legacy worker-hash-only signature")
        signature = str(manifest.get("worker_signature", ""))
        signature_ok = verify_signature(worker_hash, signature, self.release_public_key_hex)
    ELSE:
        signature_ok = verify_signature(manifest_digest, manifest_signature, self.release_public_key_hex)

    IF NOT signature_ok:
        RETURN None, receipt(error="signature_invalid")
```

### Files Modified

| File | Change |
|------|--------|
| `core/zpk/kernel.py` | Add `compute_manifest_digest()`, update `_fetch_and_verify()` |

### TDD Anchors

```python
def test_manifest_signature_covers_policy_fields():
    """Manifest with modified policy_version fails verification."""
    manifest = valid_manifest()
    manifest["policy_version"] = 999  # tampered
    artifact, receipt = await kernel._fetch_and_verify(manifest_uri)
    assert receipt.signature_ok is False

def test_manifest_signature_covers_ihsan_policy():
    """Manifest with lowered ihsan_policy fails verification."""
    manifest = valid_manifest()
    manifest["ihsan_policy"] = 0.0  # tampered
    artifact, receipt = await kernel._fetch_and_verify(manifest_uri)
    assert receipt.signature_ok is False

def test_legacy_manifest_still_accepted_with_warning(caplog):
    """Old manifests without manifest_signature fall back to worker_hash only."""
    manifest = legacy_manifest_no_manifest_signature()
    artifact, receipt = await kernel._fetch_and_verify(manifest_uri)
    assert receipt.signature_ok is True
    assert "legacy worker-hash-only" in caplog.text
```

---

## F4: ZPK Sync Worker Timeout Gap

### Current State

File: `core/zpk/kernel.py:689-693`

```python
result = entrypoint(context)              # sync — runs forever if it hangs
if inspect.isawaitable(result):
    await asyncio.wait_for(result, timeout=self.config.worker_timeout_seconds)
```

Only awaitables get a timeout. A sync entrypoint that enters an infinite loop
(`while True: pass`) will hang the executor forever.

### Required Behavior

Wrap sync entrypoint execution in a `concurrent.futures.ProcessPoolExecutor`
with a timeout equal to `self.config.worker_timeout_seconds`.

### Pseudocode

```
IMPORT concurrent.futures

# Replace lines 689-693:
result = entrypoint(context)
IF inspect.isawaitable(result):
    await asyncio.wait_for(result, timeout=self.config.worker_timeout_seconds)
ELSE IF result IS None OR result IS a primitive:
    # Sync entrypoint already returned — no timeout needed
    PASS
ELSE:
    # Shouldn't reach here, but handle gracefully
    PASS

# For the case where entrypoint itself may hang (sync infinite loop):
# Move the call into a process pool:
ASYNC FUNCTION _execute_with_timeout(entrypoint, context, timeout_seconds):
    loop = asyncio.get_running_loop()
    executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)
    TRY:
        result = await asyncio.wait_for(
            loop.run_in_executor(executor, entrypoint, context),
            timeout=timeout_seconds
        )
        RETURN result
    EXCEPT asyncio.TimeoutError:
        RAISE RuntimeError(f"worker execution timed out after {timeout_seconds}s")
    FINALLY:
        executor.shutdown(wait=False, cancel_futures=True)
```

### Files Modified

| File | Change |
|------|--------|
| `core/zpk/kernel.py` | Add `_execute_with_timeout()`, replace inline entrypoint call |

### TDD Anchors

```python
def test_sync_infinite_loop_times_out():
    """Sync entrypoint with `while True: pass` is killed after timeout."""
    def infinite_worker(ctx):
        while True:
            pass
    kernel = ZPKKernel(config=ZPKConfig(worker_timeout_seconds=2))
    receipt = await kernel._execute_worker(artifact_with(infinite_worker))
    assert receipt.exit_code == 1
    assert "timed out" in receipt.health["last_error"]

def test_normal_sync_entrypoint_completes():
    """Fast sync entrypoint returns normally within timeout."""
    def fast_worker(ctx):
        return {"ok": True}
    receipt = await kernel._execute_worker(artifact_with(fast_worker))
    assert receipt.exit_code == 0

def test_async_entrypoint_still_uses_asyncio_timeout():
    """Async entrypoint timeout path is unchanged."""
    async def slow_async(ctx):
        await asyncio.sleep(999)
    receipt = await kernel._execute_worker(artifact_with(slow_async))
    assert receipt.exit_code == 1
```

---

## F5: RLM Sandbox No Execution Time Limit

### Current State

File: `core/inference/rlm_bridge.py:322-324`

```python
compiled = compile(stripped, "<rlm-sandbox>", "exec")
exec(compiled, globals_ns, locals_ns)   # no time limit
```

The AST allowlist permits `While` and `For` loops (lines 67-68). A crafted
program like `while True: pass` will hang the process indefinitely.

### Required Behavior

Add a wall-clock timeout guard around `exec()`. On Unix (WSL/Linux), use
`signal.SIGALRM`. Cap execution at `MAX_SANDBOX_SECONDS` (default: 10).

### Pseudocode

```
IMPORT signal

MAX_SANDBOX_SECONDS = int(os.environ.get("BIZRA_RLM_TIMEOUT", "10"))

CLASS SandboxTimeout(Exception):
    pass

FUNCTION _alarm_handler(signum, frame):
    RAISE SandboxTimeout(f"sandbox execution exceeded {MAX_SANDBOX_SECONDS}s")

# In _execute_code():
    compiled = compile(stripped, "<rlm-sandbox>", "exec")

    old_handler = signal.signal(signal.SIGALRM, _alarm_handler)
    signal.alarm(MAX_SANDBOX_SECONDS)
    TRY:
        exec(compiled, globals_ns, locals_ns)
    EXCEPT SandboxTimeout AS e:
        output = f"[SANDBOX_TIMEOUT] {e}"
    FINALLY:
        signal.alarm(0)  # cancel alarm
        signal.signal(signal.SIGALRM, old_handler)  # restore
```

### Files Modified

| File | Change |
|------|--------|
| `core/inference/rlm_bridge.py` | Add `signal.SIGALRM` guard around `exec()` |

### TDD Anchors

```python
@pytest.mark.skipif(not hasattr(signal, "SIGALRM"), reason="Unix only")
def test_sandbox_infinite_loop_times_out():
    """while True: pass is killed by SIGALRM."""
    bridge = RLMSandbox()
    state, output = bridge._execute_code("while True: pass")
    assert "SANDBOX_TIMEOUT" in output

def test_sandbox_normal_code_completes():
    """Normal code runs within timeout."""
    bridge = RLMSandbox()
    state, output = bridge._execute_code("x = 2 + 2")
    assert state.variables["x"] == 4

def test_sandbox_timeout_configurable():
    """BIZRA_RLM_TIMEOUT env var controls timeout."""
    os.environ["BIZRA_RLM_TIMEOUT"] = "1"
    bridge = RLMSandbox()
    state, output = bridge._execute_code("import time; time.sleep(5)")
    assert "SANDBOX_TIMEOUT" in output
```

---

## F6: Auth Middleware Fail-Open

### Current State

File: `core/auth/middleware.py:176-179`

```python
if _global_middleware is None:
    logger.warning("Auth middleware not initialized — anonymous access allowed")
    return None   # ← fails open
```

If middleware was never initialized (deployment misconfiguration, missing init call),
every `Depends(get_current_user)` route silently allows anonymous access.

### Required Behavior

Default to fail-closed (401). Only allow anonymous access when explicitly opted in
via `BIZRA_AUTH_ALLOW_ANONYMOUS=true` environment variable.

### Pseudocode

```
FUNCTION get_current_user(authorization, x_api_key):
    IF _global_middleware IS None:
        allow_anon = os.environ.get("BIZRA_AUTH_ALLOW_ANONYMOUS", "").lower() == "true"
        IF allow_anon:
            logger.warning("Auth disabled via BIZRA_AUTH_ALLOW_ANONYMOUS — anonymous access")
            RETURN None
        ELSE:
            logger.error("Auth middleware not initialized and BIZRA_AUTH_ALLOW_ANONYMOUS not set")
            RAISE HTTPException(
                status_code=503,
                detail="Authentication service unavailable",
            )

    user = _global_middleware.authenticate(authorization, x_api_key)
    IF user IS None:
        RAISE HTTPException(status_code=401, ...)
    RETURN user
```

### Files Modified

| File | Change |
|------|--------|
| `core/auth/middleware.py` | Change `return None` to env-var-gated 503 |

### TDD Anchors

```python
def test_uninitialized_middleware_returns_503(monkeypatch):
    """Without BIZRA_AUTH_ALLOW_ANONYMOUS, uninitialized middleware = 503."""
    monkeypatch.delenv("BIZRA_AUTH_ALLOW_ANONYMOUS", raising=False)
    _global_middleware = None
    with pytest.raises(HTTPException) as exc_info:
        get_current_user(authorization=None, x_api_key=None)
    assert exc_info.value.status_code == 503

def test_uninitialized_middleware_allows_anon_when_opted_in(monkeypatch):
    """With BIZRA_AUTH_ALLOW_ANONYMOUS=true, returns None (anonymous)."""
    monkeypatch.setenv("BIZRA_AUTH_ALLOW_ANONYMOUS", "true")
    result = get_current_user(authorization=None, x_api_key=None)
    assert result is None

def test_initialized_middleware_rejects_bad_creds():
    """Initialized middleware with wrong creds returns 401."""
    ...
```

---

## F7: API Identity Endpoints Unauthenticated (Rust)

### Current State

File: `bizra-omega/bizra-api/src/lib.rs:40-45`

```rust
.route("/identity/generate", post(handlers::identity::generate))
.route("/identity/sign", post(handlers::identity::sign_message))
.route("/identity/verify", post(handlers::identity::verify_signature))
```

All identity endpoints are public. Also: `CorsLayer::permissive()` (line 74).
Any network client can generate a new node identity or sign arbitrary messages.

### Required Behavior

1. `identity/generate` and `identity/sign` MUST require a bearer token (`BIZRA_API_TOKEN`)
2. `identity/verify` can remain public (read-only verification)
3. CORS must restrict origins — not `permissive()` in production

### Pseudocode

```rust
// New: auth middleware extractor
pub async fn require_api_token(
    headers: HeaderMap,
    State(state): State<Arc<AppState>>,
) -> Result<(), ApiError> {
    let expected = state.api_token();  // from env BIZRA_API_TOKEN
    if expected.is_empty() {
        return Ok(());  // token not configured = dev mode (log warning)
    }
    let provided = headers
        .get("authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "));
    match provided {
        Some(token) if token == expected => Ok(()),
        _ => Err(ApiError::Unauthorized),
    }
}

// Route changes:
.route("/identity/generate", post(handlers::identity::generate)
    .route_layer(middleware::from_fn_with_state(state.clone(), require_api_token)))
.route("/identity/sign", post(handlers::identity::sign_message)
    .route_layer(middleware::from_fn_with_state(state.clone(), require_api_token)))
.route("/identity/verify", post(handlers::identity::verify_signature))  // public

// CORS:
let cors = if cfg!(debug_assertions) {
    CorsLayer::permissive()
} else {
    CorsLayer::new()
        .allow_origin(["http://localhost:5173".parse().unwrap()])
        .allow_methods([Method::GET, Method::POST])
        .allow_headers(Any)
};
```

### Files Modified

| File | Change |
|------|--------|
| `bizra-omega/bizra-api/src/middleware/auth.rs` | New file: bearer token extractor |
| `bizra-omega/bizra-api/src/lib.rs` | Add auth layer to identity/generate and identity/sign; restrict CORS |
| `bizra-omega/bizra-api/src/state.rs` | Add `api_token()` method reading `BIZRA_API_TOKEN` env var |

### TDD Anchors

```rust
#[tokio::test]
async fn test_identity_generate_requires_token() {
    let app = test_app_with_token("secret");
    let res = app.post("/api/v1/identity/generate").send().await;
    assert_eq!(res.status(), 401);
}

#[tokio::test]
async fn test_identity_generate_accepts_valid_token() {
    let app = test_app_with_token("secret");
    let res = app.post("/api/v1/identity/generate")
        .header("Authorization", "Bearer secret")
        .send().await;
    assert_eq!(res.status(), 200);
}

#[tokio::test]
async fn test_identity_verify_is_public() {
    let app = test_app_with_token("secret");
    let res = app.post("/api/v1/identity/verify")
        .json(&verify_request())
        .send().await;
    assert_ne!(res.status(), 401);
}
```

---

## F9: Rate Limiter Ineffective Against Bursts (Rust)

### Current State

File: `bizra-omega/bizra-api/src/middleware/rate_limit.rs:16-53`

Uses lifetime average (`total_requests / windows_elapsed`) — not per-window count.
A burst of 10,000 requests in 1 second passes if lifetime average is still below
threshold.

### Required Behavior

Replace with a proper sliding-window or token-bucket algorithm per client IP.

### Pseudocode

```rust
use dashmap::DashMap;
use std::time::Instant;

struct TokenBucket {
    tokens: f64,
    last_refill: Instant,
    capacity: f64,       // MAX_REQUESTS_PER_WINDOW
    refill_rate: f64,    // capacity / WINDOW_SECS
}

impl TokenBucket {
    fn try_consume(&mut self) -> bool {
        let now = Instant::now();
        let elapsed = now.duration_since(self.last_refill).as_secs_f64();
        self.tokens = (self.tokens + elapsed * self.refill_rate).min(self.capacity);
        self.last_refill = now;
        if self.tokens >= 1.0 {
            self.tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

// Global state: DashMap<IpAddr, TokenBucket>
// In middleware:
pub async fn rate_limiter(
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    State(state): State<Arc<AppState>>,
    request: Request<Body>,
    next: Next,
) -> Response {
    let ip = addr.ip();
    let mut bucket = state.rate_limits.entry(ip).or_insert_with(|| {
        TokenBucket::new(MAX_REQUESTS_PER_WINDOW, WINDOW_SECS)
    });
    if !bucket.try_consume() {
        return (StatusCode::TOO_MANY_REQUESTS, "Rate limit exceeded").into_response();
    }
    next.run(request).await
}
```

### Files Modified

| File | Change |
|------|--------|
| `bizra-omega/bizra-api/src/middleware/rate_limit.rs` | Rewrite: per-IP token bucket via `DashMap` |
| `bizra-omega/bizra-api/Cargo.toml` | Add `dashmap` dependency |
| `bizra-omega/bizra-api/src/state.rs` | Add `rate_limits: DashMap<IpAddr, TokenBucket>` to AppState |

### TDD Anchors

```rust
#[test]
fn test_token_bucket_allows_within_limit() {
    let mut bucket = TokenBucket::new(10.0, 60);
    for _ in 0..10 { assert!(bucket.try_consume()); }
}

#[test]
fn test_token_bucket_rejects_burst_over_limit() {
    let mut bucket = TokenBucket::new(10.0, 60);
    for _ in 0..10 { bucket.try_consume(); }
    assert!(!bucket.try_consume());  // 11th rejected
}

#[test]
fn test_token_bucket_refills_over_time() {
    let mut bucket = TokenBucket::new(10.0, 60);
    for _ in 0..10 { bucket.try_consume(); }
    // Simulate time passing
    bucket.last_refill -= Duration::from_secs(6);
    assert!(bucket.try_consume());  // refilled 1 token
}
```

---

## F10: Command Safety Bypass via Whitespace

### Current State

File: `core/benchmark/guardrails.py:534-542`
File: `core/sovereign/tiered_verification.py:88-103, 140-148`

Both use literal substring matching:

```python
BLOCKED_PATTERNS = ["rm -rf", "curl | bash", ...]
for pattern in KNOWN_DANGEROUS_PATTERNS:
    if pattern in content_lower:   # literal match
```

Bypassed by: `rm    -rf`, `rm\t-rf`, `curl  |  bash`, `curl\t|\tbash`.

### Required Behavior

Normalize whitespace before pattern matching. Replace runs of whitespace
(spaces, tabs, newlines) with a single space before checking against patterns.

### Pseudocode

```
IMPORT re

FUNCTION normalize_whitespace(text: str) -> str:
    """Collapse all whitespace runs to single space for pattern matching."""
    RETURN re.sub(r'\s+', ' ', text.strip())

# In tier_1_precheck():
    content_normalized = normalize_whitespace(content.lower())
    FOR pattern IN KNOWN_DANGEROUS_PATTERNS:
        IF pattern IN content_normalized:
            RETURN TierResult(decision=BLOCK, ...)

# In guardrails.check():
    args_str = normalize_whitespace(json.dumps(tool_args).lower())
    FOR pattern IN BLOCKED_PATTERNS:
        IF pattern IN args_str:
            RETURN GuardrailResult.FAILED
```

### Files Modified

| File | Change |
|------|--------|
| `core/sovereign/tiered_verification.py` | Add `normalize_whitespace()`, use in `tier_1_precheck()` |
| `core/benchmark/guardrails.py` | Add `normalize_whitespace()`, use in `check()` |

### TDD Anchors

```python
@pytest.mark.parametrize("payload", [
    "rm    -rf /",
    "rm\t-rf /",
    "rm  \t  -rf /",
    "curl  |  bash",
    "curl\t|\tbash",
])
def test_dangerous_patterns_detected_through_whitespace(payload):
    result = tier_1_precheck(action_type="execute", content=payload)
    assert result.decision == TierDecision.BLOCK

def test_safe_content_not_blocked():
    result = tier_1_precheck(action_type="execute", content="remove file from index")
    assert result.decision == TierDecision.PASS
```
