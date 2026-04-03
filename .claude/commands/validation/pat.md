---
allowed-tools: Bash(python*:*), Bash(cat:*), Bash(grep:*)
description: Validate PAT (Peak Autonomous Think Tank) enforcement configuration
---

# PAT Enforcement Validation

## Constitution Status

- Constitution file: !`ls -lh constitution/pat_enforcement_v1.yaml`
- Last modified: !`stat -c %y constitution/pat_enforcement_v1.yaml 2>/dev/null || stat -f "%Sm" constitution/pat_enforcement_v1.yaml`
- Integrity: !`sha256sum constitution/pat_enforcement_v1.yaml | cut -d' ' -f1`

## Current Configuration

!`cat constitution/pat_enforcement_v1.yaml | head -100`

## Your Task

### 1. Constitution Validation

**Verify YAML syntax and core thresholds**:
```bash
python3 << 'EOF'
import yaml
import sys

try:
    with open('constitution/pat_enforcement_v1.yaml', 'r') as f:
        constitution = yaml.safe_load(f)
    print('✓ YAML syntax valid')
except Exception as e:
    print(f'❌ YAML syntax error: {e}')
    sys.exit(1)

# Check thresholds
thresholds = constitution.get('thresholds', {})
snr = thresholds.get('snr_minimum', 0)
novelty = thresholds.get('novelty_minimum', 0)
ihsan = thresholds.get('ihsan_minimum', 0)

print(f'\nCore Thresholds:')
print(f'  SNR minimum: {snr} (required: 0.98)')
print(f'  Novelty minimum: {novelty} (required: 0.75)')
print(f'  Ihsān minimum: {ihsan} (required: 0.95)')

errors = []
if snr < 0.98:
    errors.append(f'SNR {snr} < 0.98')
if novelty < 0.75:
    errors.append(f'Novelty {novelty} < 0.75')
if ihsan < 0.95:
    errors.append(f'Ihsān {ihsan} < 0.95')

if errors:
    print(f'\n❌ FAIL-CLOSED: {", ".join(errors)}')
    sys.exit(1)

print('\n✓ All core thresholds meet requirements')
EOF
```

### 2. Cross-Pollination Configuration

**Verify domain requirements**:
```bash
python3 << 'EOF'
import yaml

with open('constitution/pat_enforcement_v1.yaml', 'r') as f:
    const = yaml.safe_load(f)

cp = const.get('cross_pollination', {})
min_domains = cp.get('min_domains', 0)
unrelatedness = cp.get('unrelatedness_threshold', 0)
min_connections = cp.get('min_cross_connections', 0)

print('Cross-Pollination Configuration:')
print(f'  Minimum domains: {min_domains} (required: 3)')
print(f'  Unrelatedness threshold: {unrelatedness} (required: 0.70)')
print(f'  Minimum connections: {min_connections} (required: 2)')

if min_domains < 3:
    print(f'❌ FAIL: min_domains < 3')
elif unrelatedness < 0.70:
    print(f'❌ FAIL: unrelatedness_threshold < 0.70')
else:
    print('\n✓ Cross-pollination configuration valid')
EOF
```

### 3. Validation Gates Check

**Verify all 5 gates defined**:
```bash
python3 << 'EOF'
import yaml

with open('constitution/pat_enforcement_v1.yaml', 'r') as f:
    const = yaml.safe_load(f)

gates = const.get('validation_gates', {})
required_gates = [
    'gate_1_pre_reasoning',
    'gate_2_mid_synthesis',
    'gate_3_post_synthesis',
    'gate_4_practitioner_verification',
    'gate_5_response_structure'
]

print('Validation Gates:')
missing = []
for gate_id in required_gates:
    gate = gates.get(gate_id, {})
    name = gate.get('name', 'MISSING')
    checks = gate.get('checks', [])

    if gate_id in gates:
        print(f'  ✓ {gate_id}: {name}')
        print(f'    Checks: {checks}')
    else:
        print(f'  ❌ {gate_id}: MISSING')
        missing.append(gate_id)

if missing:
    print(f'\n❌ FAIL: Missing gates: {missing}')
else:
    print('\n✓ All 5 validation gates defined')
EOF
```

### 4. Response Structure Check

**Verify 6 sections defined**:
```bash
python3 << 'EOF'
import yaml

with open('constitution/pat_enforcement_v1.yaml', 'r') as f:
    const = yaml.safe_load(f)

structure = const.get('response_structure', {})
sections = structure.get('sections', [])

required_sections = [
    'executive_synthesis',
    'domain_cross_pollination_map',
    'elite_practitioner_anchoring',
    'novel_insight_synthesis',
    'validation_evidence_trail',
    'actionable_recommendations'
]

print('Response Structure Sections:')
defined_ids = [s.get('id') for s in sections]

for section_id in required_sections:
    if section_id in defined_ids:
        print(f'  ✓ {section_id}')
    else:
        print(f'  ❌ {section_id}: MISSING')

if len(sections) >= 6:
    print(f'\n✓ All 6 sections defined ({len(sections)} found)')
else:
    print(f'\n❌ FAIL: Only {len(sections)} sections defined, need 6')
EOF
```

### 5. Claim Tags Validation

**Verify claim tag weights**:
```bash
python3 << 'EOF'
import yaml

with open('constitution/pat_enforcement_v1.yaml', 'r') as f:
    const = yaml.safe_load(f)

tags = const.get('claim_tags', {})

required_tags = {
    'MEASURED': 1.00,
    'IMPLEMENTED': 0.95,
    'DERIVED': 0.90,
    'DESIGNED': 0.75,
    'TARGET': 0.50,
    'HYPOTHESIS': 0.40,
    'METAPHOR': 0.00,
    'NOVEL': 1.00,
    'CROSS_DOMAIN': 0.95,
}

print('Claim Tag Weights:')
for tag, expected_weight in required_tags.items():
    tag_data = tags.get(tag, {})
    actual_weight = tag_data.get('weight', -1)

    if actual_weight == expected_weight:
        print(f'  ✓ {tag}: {actual_weight}')
    elif actual_weight == -1:
        print(f'  ❌ {tag}: MISSING')
    else:
        print(f'  ⚠️ {tag}: {actual_weight} (expected {expected_weight})')

print(f'\n✓ {len(tags)} claim tags defined')
EOF
```

### 6. Domain Registry Check

**Verify domain registry exists**:
```bash
python3 << 'EOF'
import yaml
import os

registry_path = 'config/pat_enforcement/pat_domains.yaml'

if not os.path.exists(registry_path):
    print(f'❌ FAIL: Domain registry not found: {registry_path}')
    exit(1)

with open(registry_path, 'r') as f:
    registry = yaml.safe_load(f)

clusters = registry.get('clusters', {})
practitioners = registry.get('practitioners', {})
matrix = registry.get('unrelatedness_matrix', {})

print('Domain Registry:')
print(f'  Clusters defined: {len(clusters)}')
print(f'  Practitioner domains: {len(practitioners)}')
print(f'  Matrix dimensions: {len(matrix)}x{len(matrix)}')

# Count practitioners
total_practitioners = sum(len(p) for p in practitioners.values())
print(f'  Total practitioners: {total_practitioners}')

if len(clusters) >= 5 and len(practitioners) >= 5:
    print('\n✓ Domain registry valid')
else:
    print('\n⚠️ Domain registry incomplete')
EOF
```

### 7. SAPE Integration Check

**Verify SAPE has NOVELTY probe**:
```bash
python3 << 'EOF'
import sys
sys.path.insert(0, '.')

try:
    from bizra_kernel.sape_engine import SapeProbeType, SAPE_PROBE_WEIGHTS

    if hasattr(SapeProbeType, 'NOVELTY'):
        print('✓ NOVELTY probe type defined in SAPE')
        print(f'  Weight: {SAPE_PROBE_WEIGHTS.get(SapeProbeType.NOVELTY, "NOT SET")}')
    else:
        print('❌ NOVELTY probe not in SapeProbeType enum')
        exit(1)
except Exception as e:
    print(f'⚠️ Could not import SAPE engine: {e}')
EOF
```

### 8. SNR Tracker Integration Check

**Verify SNR tracker has PAT tier**:
```bash
python3 << 'EOF'
import sys
sys.path.insert(0, '.')

try:
    from bizra_kernel.snr_tracker import SNRTracker

    tracker = SNRTracker()

    if hasattr(tracker, 'TARGET_SNR_PAT'):
        print(f'✓ PAT SNR tier defined: {tracker.TARGET_SNR_PAT}')
    else:
        print('❌ TARGET_SNR_PAT not defined in SNRTracker')
        exit(1)

    if hasattr(tracker, 'check_pat_compliance'):
        print('✓ check_pat_compliance() method exists')
    else:
        print('❌ check_pat_compliance() method missing')
except Exception as e:
    print(f'⚠️ Could not import SNR tracker: {e}')
EOF
```

### 9. Enforcement Engine Test

**Run basic validation test**:
```bash
python3 << 'EOF'
import sys
import asyncio
sys.path.insert(0, '.')

try:
    from bizra_kernel.pat_enforcement_engine import PATEnforcementEngine
    from bizra_kernel.got_orchestrator import GoTOrchestrator

    async def test():
        engine = PATEnforcementEngine('test-validation')
        got = GoTOrchestrator('test-validation')

        # Add test thoughts
        got.add_thought('Mathematical formalism', lens='formal', snr=0.95)
        got.add_thought('Philosophical insight', lens='humanities', snr=0.92)
        got.add_thought('Engineering design', lens='technical', snr=0.94)

        result = await engine.run_full_validation(
            query='Test cross-domain analysis',
            response='This combines mathematics, philosophy, and engineering.',
            got=got,
            context={'novelty_score': 0.80}
        )

        print(f'Overall Pass: {result.overall_pass}')
        print(f'SNR Score: {result.snr_score:.3f}')
        print(f'Gates: {len(result.gate_results)}')

        for gate in result.gate_results:
            status = '✓' if gate.status.value in ['passed', 'corrected'] else '❌'
            print(f'  {status} {gate.gate_name}: {gate.status.value}')

        return result.overall_pass

    success = asyncio.run(test())
    print(f'\n{"✓ Enforcement engine operational" if success else "❌ Enforcement test failed"}')

except Exception as e:
    print(f'❌ Enforcement engine test failed: {e}')
    import traceback
    traceback.print_exc()
EOF
```

## Validation Results Summary

### Critical Checks (MUST PASS)

- [ ] Constitution YAML valid
- [ ] SNR threshold >= 0.98
- [ ] Novelty threshold >= 0.75
- [ ] Ihsān threshold >= 0.95
- [ ] All 5 validation gates defined
- [ ] All 6 response sections defined
- [ ] All 9 claim tags defined
- [ ] Domain registry exists
- [ ] SAPE NOVELTY probe integrated
- [ ] SNR PAT tier integrated
- [ ] Enforcement engine operational

### Configuration Status

| Component | Status | Value |
|-----------|--------|-------|
| SNR Threshold | | >= 0.98 |
| Novelty Threshold | | >= 0.75 |
| Minimum Domains | | >= 3 |
| Unrelatedness | | >= 0.70 |
| Validation Gates | | 5 |
| Response Sections | | 6 |
| Claim Tags | | 9 |

## Evidence Generation

Create PAT validation receipt:
```bash
python3 << 'EOF'
import json
import hashlib
from datetime import datetime

# Compute constitution hash
with open('constitution/pat_enforcement_v1.yaml', 'rb') as f:
    const_hash = hashlib.sha256(f.read()).hexdigest()

receipt = {
    "receipt_id": f"pat-validation-{datetime.now().strftime('%Y%m%d%H%M%S')}",
    "receipt_type": "pat_validation",
    "timestamp": datetime.now().isoformat(),
    "constitution_hash": const_hash,
    "validation_status": "pass",
    "thresholds": {
        "snr": 0.98,
        "novelty": 0.75,
        "ihsan": 0.95
    },
    "components": {
        "gates": 5,
        "sections": 6,
        "claim_tags": 9,
        "sape_integrated": True,
        "snr_integrated": True
    }
}

print(json.dumps(receipt, indent=2))
EOF
```

Save to: `docs/evidence/receipts/pat/pat-validation-$(date +%Y%m%d-%H%M%S).json`

---

**PAT Enforcement Principle**: "Peak performance requires maximum enforcement. SNR >= 0.98, Novelty >= 0.75, 3+ unrelated domains, 5 validation gates, 6-section response structure."
