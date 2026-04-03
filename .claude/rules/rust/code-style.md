---
paths:
  - "src/**/*.rs"
  - "crates/**/*.rs"
  - "Cargo.toml"
  - "Cargo.lock"
---

# Rust Code Style Rules

Rules for Rust code in BIZRA's production engine.

## Formatting

### Mandatory Tools
- Run `cargo fmt` before committing
- Run `cargo clippy --all-targets -- -D warnings` (treat warnings as errors)
- All code must pass `cargo check`

### Naming Conventions
- Types: `PascalCase` (e.g., `IhsanScore`, `SapeProbe`)
- Functions/methods: `snake_case` (e.g., `validate_receipt`, `emit_evidence`)
- Constants: `SCREAMING_SNAKE_CASE` (e.g., `IHSAN_THRESHOLD`, `MAX_PROBE_TIMEOUT`)
- Modules: `snake_case` (e.g., `sape_engine`, `fate_handler`)

### Imports
- Group imports: std, external crates, local modules
- Use explicit imports over glob imports
- Keep imports sorted alphabetically within groups

```rust
// Good
use std::collections::HashMap;
use std::sync::Arc;

use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use crate::receipts::Receipt;
use crate::sape::ProbeResult;
```

## Error Handling

### Result Types
- Define domain-specific error types using `thiserror`
- Never use `unwrap()` in production code (use `expect()` with message or propagate)
- Use `?` operator for error propagation
- Always log errors before propagating

```rust
// Good - fail-closed pattern
if !validation.consensus_reached {
    tracing::error!("SAT consensus failed: {:?}", validation.rejections);
    let escalation = fate.escalate_rejection(&context).await?;
    receipts.emit_rejection(&task, &escalation).await?;
    return Err(BizraError::ConsensusFailure(validation));
}
```

### Panic Prevention
- Never panic in library code
- Use `debug_assert!` for development-only checks
- Handle all `Option::None` cases explicitly

## Async Patterns

### Tokio Usage
- Use `#[tokio::main]` for binaries
- Use `#[tokio::test]` for async tests
- Prefer `tokio::spawn` for CPU-bound tasks
- Use `tokio::select!` for concurrent operations

### Tracing
- Use `tracing` crate for logging (not `println!`)
- Add `#[tracing::instrument]` to async functions
- Include structured fields in spans

```rust
#[tracing::instrument(skip(self), fields(task_id = %task.id))]
async fn execute_task(&self, task: Task) -> Result<Response> {
    tracing::info!("Starting task execution");
    // ...
}
```

## BIZRA-Specific Patterns

### Receipt Emission
```rust
let receipt = Receipt {
    receipt_id: generate_receipt_id(),
    timestamp: Utc::now().to_rfc3339(),
    task_summary: task.summary.clone(),
    rejection_codes: vec![],
    escalation_level: EscalationLevel::None,
    integrity_hash: compute_hash(&task, &result),
};
receipts.emit(receipt).await?;
```

### Ihsān Validation
```rust
// Always check against threshold
if ihsan_score.total() < IHSAN_THRESHOLD {
    return Err(BizraError::IhsanGateFailure {
        score: ihsan_score.total(),
        threshold: IHSAN_THRESHOLD,
        failing_dimensions: ihsan_score.failing_dimensions(),
    });
}
```

### SAPE Probing
```rust
// Run all probes, collect failures
let probe_results = sape.run_all_probes(&context).await?;
let failures: Vec<_> = probe_results
    .iter()
    .filter(|p| !p.passed)
    .collect();

if !failures.is_empty() {
    tracing::warn!("SAPE failures: {:?}", failures);
    // Don't proceed silently
}
```

## Testing

- Use `#[cfg(test)]` modules for unit tests
- Name test functions descriptively: `test_ihsan_gate_rejects_below_threshold`
- Use `proptest` for property-based testing where applicable
- Mock external services (Redis, Neo4j) in tests
