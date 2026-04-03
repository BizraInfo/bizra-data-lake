---
paths:
  - "src/**/*.rs"
  - "crates/**/*.rs"
---

# Rust Safety Rules

Security and safety patterns for BIZRA Rust code.

## Memory Safety

### Avoid Unsafe
- Never use `unsafe` without explicit approval and documentation
- If `unsafe` is required, isolate it in a well-tested module
- Document safety invariants in comments

### Lifetimes
- Prefer owned types over references when ownership is clear
- Use `Arc<T>` for shared ownership across threads
- Use `Cow<'_, T>` for efficient clone-on-write patterns

## Concurrency Safety

### Thread Safety
- Use `Mutex<T>` or `RwLock<T>` for shared mutable state
- Prefer `RwLock` when reads outnumber writes
- Avoid holding locks across await points

```rust
// Bad - lock held across await
let guard = lock.lock().await;
some_async_operation().await; // Lock still held!
drop(guard);

// Good - release lock before await
let data = {
    let guard = lock.lock().await;
    guard.clone()
};
some_async_operation().await;
```

### Deadlock Prevention
- Always acquire locks in consistent order
- Use `try_lock()` with timeout for defensive locking
- Prefer message passing over shared state

## Input Validation

### User Input
- Validate ALL user input before processing
- Use strong typing (`NewType` pattern) for validated data
- Sanitize strings that will be used in commands or queries

```rust
// Good - validated type
struct UserId(String);

impl UserId {
    pub fn new(s: &str) -> Result<Self, ValidationError> {
        if !s.chars().all(|c| c.is_alphanumeric() || c == '_') {
            return Err(ValidationError::InvalidCharacters);
        }
        if s.len() > 64 {
            return Err(ValidationError::TooLong);
        }
        Ok(Self(s.to_string()))
    }
}
```

### External Data
- Deserialize with validation (`serde` with custom validators)
- Set reasonable limits on sizes (strings, arrays, nested depth)
- Handle malformed data gracefully

## Cryptographic Safety

### Hashing
- Use SHA-256 for integrity hashes (receipts)
- Use constant-time comparison for secret comparisons
- Never roll your own crypto

```rust
use sha2::{Sha256, Digest};

fn compute_integrity_hash(data: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(data.as_bytes());
    hex::encode(hasher.finalize())
}
```

### Secrets
- Never log secrets (API keys, passwords, tokens)
- Use `secrecy` crate for sensitive data
- Clear secrets from memory when no longer needed

```rust
use secrecy::{Secret, ExposeSecret};

struct Config {
    api_key: Secret<String>,
}

// Logging won't expose the secret
tracing::info!("Config loaded: {:?}", config); // api_key shows as "***"

// Explicit exposure when needed
let key = config.api_key.expose_secret();
```

## Network Safety

### TLS
- Always use TLS for external connections
- Use `rediss://` (not `redis://`) for Redis
- Validate certificates (don't skip verification)

### Timeouts
- Set timeouts on all network operations
- Use reasonable defaults (30s for operations, 5s for connects)
- Handle timeout errors explicitly

```rust
// Good - timeout wrapper
tokio::time::timeout(
    Duration::from_secs(30),
    external_api.call(&request)
).await??
```

## BIZRA Security Patterns

### Receipt Integrity
- Always compute integrity hash after receipt creation
- Verify hashes when reading receipts
- Reject receipts with invalid hashes

### FATE Escalation
- Always escalate security-related failures to FATE
- Use appropriate escalation levels:
  - `Low`: Minor issues, informational
  - `Medium`: Requires attention
  - `High`: Security concern, needs review
  - `Critical`: Immediate human intervention

### SAT Validation
- Never bypass SAT consensus checks
- Log all SAT rejections with full context
- Emit rejection receipts for audit trail
