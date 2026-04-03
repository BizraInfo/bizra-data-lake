---
allowed-tools: Bash(python*:*), Bash(cat:*), Bash(grep:*)
description: Validate Ihsān (إحسان) score and ethical compliance
---

# Ihsān (إحسان) Excellence Score Validation

## Constitution Status

- Constitution file: !`ls -lh constitution/ihsan_v1.yaml`
- Last modified: !`stat -c %y constitution/ihsan_v1.yaml 2>/dev/null || stat -f "%Sm" constitution/ihsan_v1.yaml`
- Integrity: !`sha256sum constitution/ihsan_v1.yaml | cut -d' ' -f1`

## Current Constitution

!`cat constitution/ihsan_v1.yaml`

## Your Task

### 1. Constitution Validation

**Verify YAML syntax**:
```bash
python3 -c "
import yaml
with open('constitution/ihsan_v1.yaml', 'r') as f:
    constitution = yaml.safe_load(f)
    print('✓ YAML syntax valid')
    print(f'Dimensions found: {len(constitution.get(\"dimensions\", {}))}')
"
```

**Check required fields**:
```bash
python3 << 'EOF'
import yaml

with open('constitution/ihsan_v1.yaml', 'r') as f:
    const = yaml.safe_load(f)

# Required dimensions
required = [
    'correctness', 'safety', 'user_benefit', 'efficiency',
    'auditability', 'anti_centralization', 'robustness', 'adl_fairness'
]

dimensions = const.get('dimensions', {})
missing = [d for d in required if d not in dimensions]

if missing:
    print(f'❌ MISSING DIMENSIONS: {missing}')
    exit(1)

print('✓ All 8 required dimensions present')

# Verify weights sum to 1.0
weights = [dimensions[d].get('weight', 0) for d in required]
total = sum(weights)

print(f'Weight sum: {total:.2f}')
if abs(total - 1.0) > 0.01:
    print(f'❌ FAIL-CLOSED: Weights must sum to 1.0 (got {total})')
    exit(2)

print('✓ Weights sum correctly to 1.0')

# Show weight distribution
print('\nWeight Distribution:')
for dim in required:
    weight = dimensions[dim].get('weight', 0)
    print(f'  {dim}: {weight:.2f}')
EOF
```

### 2. Threshold Validation

**Current thresholds**:
```bash
python3 << 'EOF'
import yaml

with open('constitution/ihsan_v1.yaml', 'r') as f:
    const = yaml.safe_load(f)

thresholds = const.get('thresholds', {})
print('Thresholds:')
for env, threshold in thresholds.items():
    print(f'  {env}: {threshold}')

# Verify production threshold
prod_threshold = thresholds.get('production', 0)
if prod_threshold < 0.99:
    print(f'\n❌ FAIL-CLOSED: Production threshold MUST be ≥0.99 (got {prod_threshold})')
    exit(2)

print('\n✓ Production threshold meets requirement (≥0.99)')
EOF
```

### 3. Dimension Analysis

**Analyze each dimension**:
```bash
python3 << 'EOF'
import yaml

with open('constitution/ihsan_v1.yaml', 'r') as f:
    const = yaml.safe_load(f)

dimensions = const.get('dimensions', {})

print('Dimension Details:\n')
for dim_name, dim_data in dimensions.items():
    weight = dim_data.get('weight', 0)
    desc = dim_data.get('description', 'No description')

    print(f'{dim_name.upper()} (weight: {weight:.2f})')
    print(f'  Description: {desc}')
    print(f'  Metrics: {", ".join(dim_data.get("metrics", []))}')
    print()
EOF
```

### 4. Cross-Reference with Code

**Check Rust implementation**:
```bash
# Verify Ihsān implementation in Rust
grep -n "IHSAN_THRESHOLD" src/ihsan.rs || echo "⚠️ Threshold constant not found"
grep -n "0.99" src/ihsan.rs || echo "⚠️ Default threshold not found"

# Check dimension names match
python3 << 'EOF'
import yaml
import re

# Load constitution
with open('constitution/ihsan_v1.yaml', 'r') as f:
    dimensions = list(yaml.safe_load(f).get('dimensions', {}).keys())

# Check Rust code
try:
    with open('src/ihsan.rs', 'r') as f:
        rust_code = f.read()

    missing = []
    for dim in dimensions:
        if dim not in rust_code:
            missing.append(dim)

    if missing:
        print(f'⚠️ Dimensions missing from Rust code: {missing}')
    else:
        print('✓ All dimensions referenced in Rust code')
except FileNotFoundError:
    print('⚠️ src/ihsan.rs not found')
EOF
```

**Check Python implementation**:
```bash
# Verify Python kernel loads constitution
grep -n "ihsan_v1.yaml" core/*.py || echo "⚠️ Constitution not loaded in Python"
```

### 5. Historical Validation

**Check for unauthorized modifications**:
```bash
# Show recent changes to constitution
echo "Recent modifications:"
git log -5 --oneline -- constitution/ihsan_v1.yaml

# Show who last modified
git log -1 --format="%an (%ae) at %ai" -- constitution/ihsan_v1.yaml

# Check for uncommitted changes
if git diff constitution/ihsan_v1.yaml | grep -q '^+'; then
    echo "⚠️ UNCOMMITTED CHANGES to constitution detected:"
    git diff constitution/ihsan_v1.yaml
fi
```

## Validation Results

### Critical Checks (MUST PASS)

- [ ] YAML syntax is valid
- [ ] All 8 dimensions present (correctness, safety, user_benefit, efficiency, auditability, anti_centralization, robustness, adl_fairness)
- [ ] Weights sum to 1.0 (±0.01)
- [ ] Production threshold ≥ 0.99
- [ ] Constitution referenced in Rust code
- [ ] No unauthorized modifications

### Weight Distribution Audit

| Dimension | Weight | Priority |
|-----------|--------|----------|
| correctness | 0.22 | Highest |
| safety | 0.22 | Highest |
| user_benefit | 0.14 | High |
| efficiency | 0.12 | Medium |
| auditability | 0.12 | Medium |
| anti_centralization | 0.08 | Low |
| robustness | 0.06 | Low |
| adl_fairness | 0.04 | Low |

## Fail-Closed Requirements

**BLOCK** and require human review if:
- Weights don't sum to 1.0
- Production threshold < 0.99
- Missing required dimensions
- YAML syntax errors
- Unauthorized modifications detected

## Evidence Generation

Create Ihsān validation receipt:
```json
{
  "receipt_id": "ihsan-validation-$(date +%s)",
  "timestamp": "$(date -Iseconds)",
  "constitution_hash": "$(sha256sum constitution/ihsan_v1.yaml | cut -d' ' -f1)",
  "validation_status": "pass|fail",
  "thresholds": {
    "production": 0.99,
    "ci": 0.90,
    "dev": 0.80
  },
  "dimensions": 8,
  "weight_sum": 1.0,
  "integrity_check": "pass"
}
```

Save to: `docs/evidence/receipts/ihsan-validation-$(date +%Y%m%d-%H%M%S).json`

## Report Format

```
## Ihsān (إحسان) Validation Report

**Status**: ✅ VALID | ❌ INVALID
**Constitution Hash**: [SHA-256]
**Last Modified**: [timestamp]

### Constitution Integrity
- [ ] YAML valid
- [ ] 8 dimensions present
- [ ] Weights sum to 1.0
- [ ] Threshold ≥ 0.99

### Cross-References
- [ ] Rust implementation matches
- [ ] Python kernel references it
- [ ] No unauthorized changes

### Weight Distribution
[Table showing all 8 dimensions and weights]

### Receipt
- Location: docs/evidence/receipts/ihsan-validation-YYYYMMDD-HHMMSS.json
```

---

**Constitutional Principle**: "The constitution is executable code. Changes require updating Rust implementation, Python references, tests, and documentation."
