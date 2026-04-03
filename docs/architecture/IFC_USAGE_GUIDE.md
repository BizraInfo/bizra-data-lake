# IFC (Information Flow Control) Usage Guide

## Overview

The IFC module (`src/ifc.rs`) provides systematic taint tracking for BIZRA's dual-agentic pipeline, replacing ad-hoc pattern-based redaction with formal information flow control.

**Existing Pattern (fate.rs:117-130)**: Ad-hoc field-name pattern matching
```rust
// Old approach: checks field names for "password", "secret", "key"
let sanitized_v = if k.to_lowercase().contains("password") {
    "[REDACTED]".to_string()
} else { v.clone() };
```

**New Pattern (ifc.rs)**: Systematic per-value taint labels with two-dimensional lattice

## Security Lattices

### Secrecy Lattice (Public < Internal < Confidential < Secret)
- **Public**: Can be shared externally (logs, API responses)
- **Internal**: BIZRA-internal only (agent state, performance metrics)
- **Confidential**: Restricted to specific agents (API keys, credentials)
- **Secret**: Highest sensitivity (cryptographic material, private keys)

### Integrity Lattice (Untrusted < Validated < Attested < Sovereign)
- **Untrusted**: User input, external sources
- **Validated**: Basic validation passed (format checks, schema validation)
- **Attested**: SAT consensus reached (3/5 guardian approval)
- **Sovereign**: Kernel-signed (cryptographic proof of origin)

## Core Concepts

### TaintLabel
Every data value has a label tracking its secrecy and integrity:

```rust
use bizra::ifc::{TaintLabel, SecrecyLevel, IntegrityLevel};

let label = TaintLabel::new(
    SecrecyLevel::Confidential,  // Secrecy dimension
    IntegrityLevel::Validated,   // Integrity dimension
    "user_12345".to_string()     // Origin tracking
);
```

### TaintContext
Pipeline boundary tracking:

```rust
use bizra::ifc::TaintContext;

let mut ctx = TaintContext::new("http_request");

// Label incoming data as Untrusted
ctx.taint("user_input", TaintLabel::new(
    SecrecyLevel::Internal,
    IntegrityLevel::Untrusted,
    "user_12345".into()
));

// After SAT validation, promote integrity
ctx.promote("user_input", IntegrityLevel::Attested)?;
```

## Integration Points

### 1. HTTP Request Boundary (src/http.rs)

```rust
use bizra::ifc::{TaintContext, TaintLabel, SecrecyLevel, IntegrityLevel};

async fn handle_request(req: Request<Body>) -> Result<Response<Body>, BridgeError> {
    let mut taint_ctx = TaintContext::new("http_ingress");

    // Label all incoming data as Untrusted/Internal
    for (key, value) in req.headers() {
        taint_ctx.taint(
            key.as_str(),
            TaintLabel::new(
                SecrecyLevel::Internal,
                IntegrityLevel::Untrusted,
                "http_client".into()
            )
        );
    }

    // Parse body and label
    let body_str = hyper::body::to_bytes(req.into_body()).await?;
    taint_ctx.taint(
        "request_body",
        TaintLabel::new(
            SecrecyLevel::Internal,
            IntegrityLevel::Untrusted,
            "http_client".into()
        )
    );

    // ... execute request ...
}
```

### 2. SAT Validation Boundary (src/sat.rs)

```rust
// After SAT consensus, promote integrity
async fn sat_validate(request: &DualAgenticRequest, ctx: &mut TaintContext)
    -> Result<SATValidation, SATError>
{
    let validation = run_sat_consensus(request).await?;

    if validation.consensus_reached {
        // SAT approved: promote to Attested
        for field in request.fields() {
            ctx.promote(field, IntegrityLevel::Attested)?;
        }
    } else {
        // SAT rejected: maintain Untrusted
        tracing::warn!("SAT consensus failed, data remains Untrusted");
    }

    Ok(validation)
}
```

### 3. MCP Tool Boundary (src/mcp.rs)

```rust
// Before invoking MCP tools, check secrecy
async fn invoke_mcp_tool(
    tool_name: &str,
    params: &HashMap<String, String>,
    ctx: &TaintContext
) -> Result<Value, McpError>
{
    // Verify no Secret/Confidential data leaks to external tools
    let param_keys: Vec<&str> = params.keys().map(|s| s.as_str()).collect();
    ctx.validate_output(&param_keys)?;

    // Safe to invoke external tool
    let result = call_mcp_server(tool_name, params).await?;
    Ok(result)
}
```

### 4. Response Egress (src/bridge.rs)

```rust
// Before returning response, validate no secrets leak
async fn finalize_response(
    response: &DualAgenticResponse,
    ctx: &TaintContext
) -> Result<(), BridgeError>
{
    // Verify all output fields are Public
    ctx.validate_output(&["result", "summary", "reasoning"])?;

    // If validation fails, explicit declassification required
    // (Must be audited and logged)

    Ok(())
}
```

### 5. FATE Escalation (src/fate.rs)

Replace existing redaction (lines 117-130) with:

```rust
use crate::ifc::{TaintContext, SecrecyLevel};

// Before emitting FATE receipt
let mut taint_ctx = TaintContext::new("fate_escalation");

for (k, v) in context {
    let secrecy = if k.contains("password") || k.contains("secret") {
        SecrecyLevel::Confidential
    } else {
        SecrecyLevel::Internal
    };

    taint_ctx.taint(k, TaintLabel::new(
        secrecy,
        IntegrityLevel::Validated,
        "fate_engine".into()
    ));
}

// Declassify for audit (with explicit reason)
taint_ctx.declassify(
    "rejection_context",
    SecrecyLevel::Internal,
    "FATE audit trail requirement"
);

// Emit receipt with audit log
let receipt = RejectionReceipt {
    // ... fields ...
    taint_audit: taint_ctx.audit_log().to_vec(),
};
```

## Policy Enforcement

### Automatic Checks

```rust
// ✅ ALLOWED: Public → Secret (information gain)
ctx.check_flow("public_data", SecrecyLevel::Secret)?;

// ❌ BLOCKED: Secret → Public (information leak)
ctx.check_flow("api_key", SecrecyLevel::Public)?;
// Returns: IFCViolation::SecrecyViolation

// ❌ BLOCKED: Downgrade integrity
ctx.promote("validated_data", IntegrityLevel::Untrusted)?;
// Returns: IFCViolation::IntegrityViolation
```

### Explicit Declassification

When intentional declassification is required (e.g., error reporting), use explicit `declassify` with audit trail:

```rust
// Example: Error message needs external visibility
ctx.declassify(
    "error_details",
    SecrecyLevel::Public,
    "User-facing error message - SAT approved disclosure"
);

// Audit log entry created automatically:
// {
//   "timestamp": "2026-02-14T...",
//   "field": "error_details",
//   "from_secrecy": "Internal",
//   "to_secrecy": "Public",
//   "reason": "User-facing error message - SAT approved disclosure",
//   "actor": "fate_engine"
// }
```

### Context Merging

When combining data from multiple sources (e.g., PAT agents collaborating):

```rust
let mut agent1_ctx = TaintContext::new("pat_strategic_visionary");
agent1_ctx.taint("plan", TaintLabel::new(
    SecrecyLevel::Internal,
    IntegrityLevel::Validated,
    "pat_agent_1".into()
));

let mut agent2_ctx = TaintContext::new("pat_implementation_specialist");
agent2_ctx.taint("plan", TaintLabel::new(
    SecrecyLevel::Confidential,  // More restrictive
    IntegrityLevel::Untrusted,   // Less trusted
    "pat_agent_2".into()
));

// Merge: takes MORE restrictive label
agent1_ctx.merge(&agent2_ctx);

let merged_label = agent1_ctx.get_label("plan");
assert_eq!(merged_label.secrecy, SecrecyLevel::Confidential); // Max secrecy
assert_eq!(merged_label.integrity, IntegrityLevel::Untrusted); // Min integrity
```

## Receipt Integration

Embed IFC audit logs in receipts for full traceability:

```rust
use crate::receipts::ExecutionReceipt;

let receipt = ExecutionReceipt {
    receipt_id: uuid::Uuid::new_v4().to_string(),
    timestamp: Utc::now(),
    task_summary: "User request execution".into(),
    taint_audit: taint_ctx.audit_log().to_vec(), // ← IFC audit log
    // ... other fields ...
};
```

## Migration Strategy

### Phase 1: Parallel Operation
1. Keep existing pattern-based redaction in `fate.rs`
2. Add IFC tracking to new boundaries (HTTP, MCP)
3. Log IFC violations without blocking

### Phase 2: Enforcement
1. Enable blocking for IFC violations at HTTP egress
2. Require explicit declassification for FATE context
3. Emit IFC audit logs in receipts

### Phase 3: Full Adoption
1. Remove pattern-based redaction from `fate.rs`
2. Enforce IFC at all pipeline boundaries
3. Integrate with SAT guardian policies

## Testing

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_api_key_protection() {
        let mut ctx = TaintContext::new("test");

        // User provides API key
        ctx.taint("api_key", TaintLabel::new(
            SecrecyLevel::Confidential,
            IntegrityLevel::Validated,
            "user".into()
        ));

        // Attempt to include in public response
        let result = ctx.validate_output(&["api_key"]);

        // Should block
        assert!(result.is_err());
        match result {
            Err(IFCViolation::SecrecyViolation { from_level, to_level, .. }) => {
                assert_eq!(from_level, SecrecyLevel::Confidential);
                assert_eq!(to_level, SecrecyLevel::Public);
            }
            _ => panic!("Expected SecrecyViolation"),
        }
    }
}
```

## Performance Considerations

- **O(1) label lookup**: HashMap-based
- **O(n) validation**: Linear scan of output fields
- **Memory**: ~200 bytes per label
- **Overhead**: <1ms per boundary check

For high-throughput scenarios, use batch validation:

```rust
// Validate all fields in one pass
let all_keys: Vec<&str> = response_fields.keys().collect();
ctx.validate_output(&all_keys)?;
```

## Security Properties

### Non-Interference
Secret data cannot flow to Public context without explicit declassification and audit trail.

### Fail-Closed
IFC violations return `Result::Err` — execution blocks by default.

### Audit Trail
Every declassification creates immutable audit entry with timestamp, reason, and actor.

### Composability
Contexts merge conservatively — combined secrecy is max(s1, s2), combined integrity is min(i1, i2).

## See Also

- Constitution: `constitution/ihsan_v1.yaml` (safety dimension)
- FATE Engine: `src/fate.rs` (escalation handling)
- SAT Guardians: `src/sat.rs` (validation enforcement)
- Receipts: `src/receipts.rs` (audit log integration)
