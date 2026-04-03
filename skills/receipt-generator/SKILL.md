---
name: Receipt Generator
description: Automatically generates BIZRA-compliant evidence receipts for operations
keywords: [receipt, evidence, audit, integrity, sha256]
user-invocable: false
disable-model-invocation: false
---

# Receipt Generator Skill

## Purpose

This skill enables Claude to automatically generate BIZRA-compliant receipt evidence for any operation, ensuring the receipt-native architecture is maintained throughout development workflows.

## When to Use

Claude should invoke this skill when:
- Completing a build operation
- Finishing a test run
- Performing validation checks
- Creating git commits
- Making configuration changes
- Running deployment operations
- Executing any significant system operation

## Receipt Schema

All receipts follow the canonical schema from `src/receipts.rs`:

```json
{
  "receipt_id": "unique-identifier",
  "timestamp": "RFC3339-format",
  "task_summary": "what-was-done",
  "rejection_codes": [],
  "escalation_level": "None|Low|Medium|High|Critical",
  "integrity_hash": "SHA-256-hash"
}
```

## Capabilities

### 1. Generate Build Receipts

For cargo build operations:
```json
{
  "receipt_id": "build-1234567890",
  "timestamp": "2026-01-20T10:30:00Z",
  "task_summary": "Rust Elite engine built successfully (release mode)",
  "build_info": {
    "mode": "release",
    "compiler_version": "rustc 1.90.0",
    "warnings": 0,
    "binary_size_bytes": 8388608
  },
  "rejection_codes": [],
  "escalation_level": "None",
  "integrity_hash": "abc123..."
}
```

### 2. Generate Test Receipts

For test executions:
```json
{
  "receipt_id": "test-1234567890",
  "timestamp": "2026-01-20T10:35:00Z",
  "task_summary": "Rust test suite execution completed",
  "test_summary": {
    "total": 45,
    "passed": 45,
    "failed": 0,
    "ignored": 0,
    "execution_time_ms": 2340
  },
  "critical_tests": {
    "pat_sat": "pass",
    "ihsan": "pass",
    "sape": "pass",
    "receipts": "pass"
  },
  "rejection_codes": [],
  "escalation_level": "None",
  "integrity_hash": "def456..."
}
```

### 3. Generate Validation Receipts

For Ihsān/SAPE validation:
```json
{
  "receipt_id": "validation-1234567890",
  "timestamp": "2026-01-20T10:40:00Z",
  "task_summary": "Ihsān constitution validation completed",
  "validation_status": "pass",
  "constitution_hash": "sha256...",
  "thresholds": {
    "production": 0.99,
    "ci": 0.90,
    "dev": 0.80
  },
  "dimensions": 8,
  "weight_sum": 1.0,
  "rejection_codes": [],
  "escalation_level": "None",
  "integrity_hash": "ghi789..."
}
```

### 4. Generate Commit Receipts

For git commits:
```json
{
  "receipt_id": "commit-abc123de",
  "timestamp": "2026-01-20T10:45:00Z",
  "commit_hash": "abc123def456",
  "commit_message": "feat(rust): add new SAPE probe",
  "files_changed": 5,
  "lines_added": 120,
  "lines_deleted": 10,
  "branch": "feature/new-probe",
  "author": "Developer Name",
  "co_authored": "Claude Opus 4.5",
  "rejection_codes": [],
  "escalation_level": "None",
  "integrity_hash": "jkl012..."
}
```

## Integrity Hash Calculation

The integrity hash is calculated as:
```bash
echo -n "${receipt_id}${timestamp}${task_summary}" | sha256sum
```

This ensures:
- Receipt immutability
- Tamper detection
- Content verification
- Audit trail integrity

## File Storage

Receipts are stored in:
```
docs/evidence/receipts/
├── build-YYYYMMDD-HHMMSS.json
├── test-YYYYMMDD-HHMMSS.json
├── validation-YYYYMMDD-HHMMSS.json
├── commit-YYYYMMDD-HHMMSS.json
└── [operation]-[timestamp].json
```

**Storage rules**:
- Append-only (never delete or modify)
- One file per receipt
- Timestamped filenames for chronology
- JSON format for machine readability

## BIZRA Integration

### Receipt-First Development

**Before** considering an operation complete:
1. Gather operation metrics/results
2. Generate receipt with this skill
3. Calculate integrity hash
4. Write to `docs/evidence/receipts/`
5. Report receipt location to user

**Never** proceed without receipt:
```
✅ CORRECT:
Operation → Generate Receipt → Mark Complete

❌ WRONG:
Operation → Mark Complete (no receipt)
```

### Fail-Closed Requirements

Generate failure receipts for failed operations:
```json
{
  "receipt_id": "build-failed-1234567890",
  "timestamp": "2026-01-20T11:00:00Z",
  "task_summary": "Rust build failed with clippy errors",
  "rejection_codes": ["CLIPPY_ERROR", "BUILD_FAILED"],
  "escalation_level": "High",
  "error_details": {
    "command": "cargo build --release",
    "exit_code": 101,
    "errors": [
      "error: unused variable `x` in src/main.rs:42"
    ]
  },
  "integrity_hash": "error123..."
}
```

### Evidence-Driven Workflow

Receipts enable:
- Audit trail reconstruction
- Success/failure trend analysis
- Performance tracking over time
- Compliance demonstration
- Root cause analysis

## Example Usage

### Context Triggers

When Claude sees any of these patterns:
```
"cargo build completed successfully"
"All tests passed"
"Ihsān validation complete"
"Committed with hash abc123"
"Deployment finished"
```

Claude should **automatically**:
1. Invoke this skill
2. Generate appropriate receipt
3. Store in `docs/evidence/receipts/`
4. Report receipt location to user

### User-Requested

User says:
- "Generate a receipt for this build"
- "Create evidence for the test run"
- "Document this operation"
- "Record this as a receipt"

Claude should:
1. Use this skill
2. Include all relevant operation details
3. Calculate integrity hash
4. Save receipt
5. Confirm location

## Template

Basic receipt template:
```json
{
  "receipt_id": "${operation}-${timestamp_unix}",
  "timestamp": "${timestamp_rfc3339}",
  "task_summary": "${brief_description}",
  "rejection_codes": [],
  "escalation_level": "None",
  "integrity_hash": "${calculated_hash}"
}
```

Extended with operation-specific fields:
- `build_info` for builds
- `test_summary` for tests
- `validation_status` for validations
- `commit_hash` for commits
- Custom fields as needed

## Receipt Types

| Type | Prefix | When Generated |
|------|--------|----------------|
| Build | `build-` | After cargo/npm build |
| Test | `test-` | After test suite runs |
| Validation | `validation-` | After Ihsān/SAPE checks |
| Commit | `commit-` | After git commit |
| Deploy | `deploy-` | After deployment |
| Evidence | `evidence-` | Meta-receipts for validation |

## Quality Checks

Before saving receipt, verify:
- [ ] All required fields present
- [ ] receipt_id is unique
- [ ] timestamp in RFC3339 format
- [ ] integrity_hash calculated correctly
- [ ] escalation_level is valid enum
- [ ] File doesn't already exist

## Tools Required

- **Write**: Save receipt to file
- **Bash**: Calculate SHA-256 hash, get timestamp
- **Read**: Check for existing receipts (optional)

## Performance

- Receipt generation: <100ms
- Hash calculation: <10ms
- File write: <50ms
- Total overhead: <200ms per receipt

## Error Handling

If receipt generation fails:
1. Log the error
2. Create a minimal receipt with error details
3. Set escalation_level to "Critical"
4. Report to user immediately

Never silently skip receipt generation.

---

**Skill Philosophy**: "Every operation leaves a trail. Generate receipts automatically, consistently, and correctly."

## Usage Pattern

```
User: "Build the Rust components"

Claude:
1. Runs: cargo build --release
2. Observes: Build successful
3. Invokes: Receipt Generator Skill
4. Generates: build-20260120-103000.json
5. Reports: "✓ Build complete. Receipt: docs/evidence/receipts/build-20260120-103000.json"
```

**This happens automatically** - users don't need to request receipts explicitly.
