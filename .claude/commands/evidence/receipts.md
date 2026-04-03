---
allowed-tools: Bash(ls:*), Bash(cat:*), Bash(jq:*), Bash(sha256sum:*), Bash(find:*)
description: Validate and analyze receipt evidence
argument-hint: [count|validate|recent|stats]
---

# Receipt Evidence Management

## Receipt System Overview

**Receipt-Native Architecture**:
- All decisions emit structured receipts
- Append-only storage in `docs/evidence/receipts/`
- SHA-256 integrity hashing
- Defined schema in `src/receipts.rs`

## Current Receipt Status

- Receipt directory: !`ls -lh docs/evidence/receipts/ 2>/dev/null | tail -1 || echo "Directory not found"`
- Total receipts: !`find docs/evidence/receipts/ -name "*.json" -o -name "*.jsonl" | wc -l 2>/dev/null || echo "0"`
- Latest receipt: !`ls -t docs/evidence/receipts/*.{json,jsonl} 2>/dev/null | head -1 | xargs basename || echo "None"`
- Total size: !`du -sh docs/evidence/receipts/ 2>/dev/null | cut -f1 || echo "0B"`

## Command: **$1**

## Your Task

### For "count" - Receipt Inventory
```bash
# Count receipts by type
echo "Receipt Inventory:"
echo "==================="

total=$(find docs/evidence/receipts/ -name "*.json" -o -name "*.jsonl" 2>/dev/null | wc -l)
echo "Total receipts: $total"

# Count by prefix
for type in build test deploy validation ihsan sape fate; do
    count=$(find docs/evidence/receipts/ -name "${type}*.json" 2>/dev/null | wc -l)
    if [ $count -gt 0 ]; then
        echo "  ${type}: $count"
    fi
done

# Size analysis
echo ""
echo "Storage:"
du -sh docs/evidence/receipts/ 2>/dev/null || echo "  0B"

# Recent activity
echo ""
echo "Recent activity (last 24h):"
find docs/evidence/receipts/ -name "*.json" -mtime -1 2>/dev/null | wc -l
```

### For "validate" - Receipt Schema Validation
```bash
echo "Validating receipt schema..."
echo "============================"

# Check each receipt
validated=0
failed=0

for receipt in docs/evidence/receipts/*.json; do
    if [ -f "$receipt" ]; then
        # Validate JSON syntax
        if jq empty "$receipt" 2>/dev/null; then
            # Check required fields
            has_id=$(jq -e '.receipt_id' "$receipt" >/dev/null 2>&1 && echo "1" || echo "0")
            has_timestamp=$(jq -e '.timestamp' "$receipt" >/dev/null 2>&1 && echo "1" || echo "0")
            has_hash=$(jq -e '.integrity_hash' "$receipt" >/dev/null 2>&1 && echo "1" || echo "0")

            if [ "$has_id" = "1" ] && [ "$has_timestamp" = "1" ] && [ "$has_hash" = "1" ]; then
                ((validated++))
            else
                echo "❌ Missing required fields: $(basename $receipt)"
                ((failed++))
            fi
        else
            echo "❌ Invalid JSON: $(basename $receipt)"
            ((failed++))
        fi
    fi
done

echo ""
echo "Results:"
echo "  ✓ Valid: $validated"
echo "  ❌ Failed: $failed"

# Fail-closed if validation failures
if [ $failed -gt 0 ]; then
    echo ""
    echo "🛑 FAIL-CLOSED: Receipt validation failed"
    exit 2
fi
```

### For "recent" - Recent Receipts
```bash
echo "Recent Receipts (last 10):"
echo "=========================="

ls -lt docs/evidence/receipts/*.json 2>/dev/null | head -10 | while read -r line; do
    receipt=$(echo "$line" | awk '{print $NF}')
    if [ -f "$receipt" ]; then
        receipt_id=$(jq -r '.receipt_id' "$receipt" 2>/dev/null || echo "unknown")
        timestamp=$(jq -r '.timestamp' "$receipt" 2>/dev/null || echo "unknown")
        task=$(jq -r '.task_summary // .test_summary // .validation_status // "unknown"' "$receipt" 2>/dev/null)

        echo ""
        echo "File: $(basename $receipt)"
        echo "  ID: $receipt_id"
        echo "  Time: $timestamp"
        echo "  Task: $task"
    fi
done
```

### For "stats" - Statistical Analysis
```bash
echo "Receipt Statistics:"
echo "==================="

# Creation timeline
echo ""
echo "Receipts by date:"
find docs/evidence/receipts/ -name "*.json" -printf '%TY-%Tm-%Td\n' 2>/dev/null | \
  sort | uniq -c | tail -7

# Success vs failure rate
echo ""
echo "Success/Failure Analysis:"

total=0
success=0
failed=0

for receipt in docs/evidence/receipts/*.json; do
    if [ -f "$receipt" ]; then
        ((total++))

        # Check for success indicators
        if jq -e '.success == true or .validation_status == "pass" or .test_summary.failed == 0' "$receipt" >/dev/null 2>&1; then
            ((success++))
        elif jq -e '.success == false or .validation_status == "fail" or .test_summary.failed > 0 or .rejection_codes' "$receipt" >/dev/null 2>&1; then
            ((failed++))
        fi
    fi
done

echo "  Total: $total"
echo "  Success: $success ($(( success * 100 / total ))%)"
echo "  Failed: $failed ($(( failed * 100 / total ))%)"

# Escalation analysis
echo ""
echo "FATE Escalations:"
escalated=$(grep -l "escalation_level" docs/evidence/receipts/*.json 2>/dev/null | wc -l)
echo "  Receipts with escalations: $escalated"

if [ $escalated -gt 0 ]; then
    echo "  By level:"
    for level in Low Medium High Critical; do
        count=$(grep -l "\"escalation_level\": \"$level\"" docs/evidence/receipts/*.json 2>/dev/null | wc -l)
        if [ $count -gt 0 ]; then
            echo "    $level: $count"
        fi
    done
fi

# Integrity analysis
echo ""
echo "Integrity Checks:"
integrity_count=$(grep -l "integrity_hash" docs/evidence/receipts/*.json 2>/dev/null | wc -l)
echo "  Receipts with integrity hash: $integrity_count / $total"
```

## Receipt Schema Requirements

**Required fields** (from `src/receipts.rs`):
```
- receipt_id: string (unique identifier)
- timestamp: string (RFC3339 format)
- task_summary: string (or test_summary, validation_status, etc.)
- rejection_codes: array (if applicable)
- escalation_level: enum (None|Low|Medium|High|Critical)
- integrity_hash: string (SHA-256)
```

## Validation Checklist

### Schema Compliance
- [ ] All receipts have valid JSON syntax
- [ ] receipt_id present and unique
- [ ] timestamp in RFC3339 format
- [ ] integrity_hash present (SHA-256)
- [ ] Appropriate summary field exists

### Integrity Verification
- [ ] Hash matches content
- [ ] No duplicate receipt_ids
- [ ] Timestamps are chronological
- [ ] Append-only (no deletions)

### Content Analysis
- [ ] Success/failure rates reasonable
- [ ] Escalation levels appropriate
- [ ] Rejection codes documented
- [ ] Task summaries descriptive

## Receipt Schema Guard

**Files that require coordinated updates**:
1. `src/receipts.rs` - Rust struct definition
2. `core/fate.py` - Python equivalent
3. Tests in `tests/` - Schema validation tests
4. Evidence docs in `docs/execution/` - Documentation
5. `CLAUDE.md` - Developer guide

**Changing receipt schema?**
- Update all 5 locations above
- Maintain backward compatibility
- Add migration logic if needed
- Update receipt validation tests
- Document changes in CHANGELOG

## Fail-Closed Requirements

**BLOCK** if:
- Receipts have invalid JSON
- Required fields missing
- Integrity hashes invalid
- Duplicate receipt_ids found
- Validation failure rate >10%

**WARN** but allow:
- No receipts found (new installation)
- Some receipts missing optional fields
- Old receipts with previous schema versions

## Evidence Generation

Create receipt validation receipt (meta-receipt):
```json
{
  "receipt_id": "receipt-validation-$(date +%s)",
  "timestamp": "$(date -Iseconds)",
  "validation_summary": {
    "total_receipts": 0,
    "validated": 0,
    "failed": 0,
    "missing_fields": 0
  },
  "integrity_status": "pass|fail",
  "schema_compliance": "pass|fail",
  "storage_size_mb": 0.0
}
```

Save to: `docs/evidence/receipts/receipt-validation-$(date +%Y%m%d-%H%M%S).json`

## Report Format

```
## Receipt Evidence Report

**Command**: $1
**Status**: ✅ VALID | ❌ INVALID
**Total Receipts**: X

### Inventory
- Build receipts: X
- Test receipts: X
- Validation receipts: X
- FATE receipts: X

### Schema Compliance
- Valid JSON: X/X
- Required fields: X/X
- Integrity hashes: X/X

### Success Rates
- Overall: X%
- Build: X%
- Test: X%
- Validation: X%

### Escalations (if any)
- Low: X
- Medium: X
- High: X
- Critical: X

### Storage
- Total size: X MB
- Oldest receipt: YYYY-MM-DD
- Newest receipt: YYYY-MM-DD

### Receipt
- Location: docs/evidence/receipts/receipt-validation-YYYYMMDD-HHMMSS.json
```

---

**Receipt Philosophy**: "All decisions emit structured receipts. Append-only. Integrity-protected. Schema-stable. Receipt-first development."
