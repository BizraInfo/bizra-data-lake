---
allowed-tools: Bash(python*:*), Bash(cargo:test)
description: Run SAPE (Symbolic-Abstraction Probe Elevation) validation
---

# SAPE Probe Validation

## SAPE System Overview

**9-Probe Verification System**:
1. threat_scan - Security threat detection
2. compliance - Policy compliance checking
3. bias - Bias and fairness analysis
4. user_benefit - User value assessment
5. correctness - Logical correctness validation
6. safety - Safety constraint verification
7. groundedness - Factual grounding check
8. relevance - Task relevance analysis
9. fluency - Output quality assessment

## Current SAPE Status

- Rust SAPE engine: !`ls -lh src/sape.rs`
- Python SAPE planning: !`ls -lh core/sape.py`
- Redis elevations: !`redis-cli -u ${SYNAPSE_URL} --tls --cacert config/redis/ca-cert.pem keys "bizra:sape:elevation:*" 2>/dev/null | wc -l || echo "Redis not accessible"`

## Your Task

### 1. Probe Implementation Validation

**Verify all 9 probes exist in Rust**:
```bash
echo "Checking Rust SAPE implementation..."

for probe in threat_scan compliance bias user_benefit correctness safety groundedness relevance fluency; do
    if grep -q "fn probe_${probe}" src/sape.rs; then
        echo "✓ ${probe} probe implemented"
    else
        echo "❌ ${probe} probe MISSING"
    fi
done
```

**Verify Python SAPE planning**:
```bash
echo "Checking Python SAPE planning..."

python3 << 'EOF'
try:
    from core.sape import SapeEngine
    print('✓ SapeEngine imported successfully')

    # Check for probe methods
    engine = object.__new__(SapeEngine)
    probes = ['threat_scan', 'compliance', 'bias', 'user_benefit',
              'correctness', 'safety', 'groundedness', 'relevance', 'fluency']

    for probe in probes:
        if hasattr(SapeEngine, f'probe_{probe}'):
            print(f'✓ probe_{probe} method exists')
        else:
            print(f'❌ probe_{probe} method MISSING')
except Exception as e:
    print(f'❌ Error: {e}')
EOF
```

### 2. Probe Execution Tests

**Run Rust SAPE tests**:
```bash
# Test individual probes
cargo test sape:: --no-fail-fast

# Test probe elevation (>3 repetitions)
cargo test sape::test_elevation

# Test high-stakes probes with Neo4j
cargo test sape::test_graph_evidence
```

**Run Python SAPE tests**:
```bash
# Test SAPE planning logic
pytest tests/test_sape.py -v

# Test pattern elevation
pytest tests/test_sape.py::test_probe_elevation -v
```

### 3. Pattern Elevation Analysis

**Check current elevated patterns**:
```bash
python3 << 'EOF'
import os
import redis

try:
    synapse_url = os.getenv('SYNAPSE_URL', 'redis://localhost:6379')
    if synapse_url.startswith('rediss://'):
        # TLS connection
        r = redis.from_url(
            synapse_url,
            ssl_cert_reqs='required',
            ssl_ca_certs='config/redis/ca-cert.pem'
        )
    else:
        r = redis.from_url(synapse_url)

    # Get all elevation keys
    keys = r.keys('bizra:sape:elevation:*')
    print(f'Elevated patterns: {len(keys)}')

    # Show top 5 elevated patterns
    for i, key in enumerate(keys[:5]):
        pattern_hash = key.decode().split(':')[-1]
        occurrences = r.get(key)
        print(f'  {i+1}. {pattern_hash}: {occurrences} occurrences')

except Exception as e:
    print(f'⚠️ Cannot access Redis: {e}')
EOF
```

### 4. Neo4j Graph Evidence Validation

**Check Neo4j connectivity for high-stakes probes**:
```bash
python3 << 'EOF'
import os

try:
    from neo4j import GraphDatabase

    uri = os.getenv('NEO4J_URI', 'bolt://localhost:7687')
    user = os.getenv('NEO4J_USER', 'neo4j')
    password = os.getenv('NEO4J_PASSWORD', '')

    driver = GraphDatabase.driver(uri, auth=(user, password))

    with driver.session() as session:
        result = session.run('MATCH (n) RETURN count(n) as count')
        count = result.single()['count']
        print(f'✓ Neo4j connected: {count} nodes in wisdom graph')

    driver.close()

except Exception as e:
    print(f'⚠️ Neo4j not accessible: {e}')
    print('   High-stakes SAPE probes may fail')
EOF
```

### 5. Probe Performance Analysis

**Measure probe execution time**:
```bash
cargo test sape:: -- --nocapture 2>&1 | \
  grep -E "(test.*probe|finished in)" | \
  tail -10
```

**Target performance**:
- Individual probe: <100ms
- Full 9-probe suite: <1s
- With Neo4j evidence: <2s

### 6. SAPE Configuration Validation

**Check SAPE cache TTL**:
```bash
echo "SAPE_CACHE_TTL=${SAPE_CACHE_TTL:-3600} seconds (default: 3600)"
```

**Verify SAPE in request flow**:
```bash
# Check SAPE is invoked in request flow
grep -n "sape.probe" src/bridge.rs || echo "⚠️ SAPE not in request flow"
grep -n "sape_probing" src/http.rs || echo "⚠️ SAPE not in HTTP handler"
```

## Validation Results

### Critical Checks (MUST PASS)

- [ ] All 9 probes implemented in Rust
- [ ] All 9 probes implemented in Python
- [ ] Probe tests pass in Rust
- [ ] Probe tests pass in Python
- [ ] Pattern elevation works (>3 repetitions)
- [ ] Redis connectivity for elevations
- [ ] Neo4j connectivity for high-stakes probes (optional but recommended)

### Probe Inventory

| Probe | Rust | Python | Tests | Performance |
|-------|------|--------|-------|-------------|
| threat_scan | ✓ | ✓ | PASS | <100ms |
| compliance | ✓ | ✓ | PASS | <100ms |
| bias | ✓ | ✓ | PASS | <100ms |
| user_benefit | ✓ | ✓ | PASS | <100ms |
| correctness | ✓ | ✓ | PASS | <100ms |
| safety | ✓ | ✓ | PASS | <100ms |
| groundedness | ✓ | ✓ | PASS | <100ms |
| relevance | ✓ | ✓ | PASS | <100ms |
| fluency | ✓ | ✓ | PASS | <100ms |

### Elevated Patterns

```
Total patterns: [count]
Top patterns by occurrences:
1. [hash]: [count] occurrences
2. [hash]: [count] occurrences
...
```

## Fail-Closed Requirements

**BLOCK** if:
- Any probe is missing from implementation
- Probe tests fail in either Rust or Python
- Pattern elevation broken
- Redis inaccessible (can't store elevations)

**WARN** but allow:
- Neo4j inaccessible (degrades to non-graph evidence)
- Probe performance >100ms (investigate optimization)

## Evidence Generation

Create SAPE validation receipt:
```json
{
  "receipt_id": "sape-validation-$(date +%s)",
  "timestamp": "$(date -Iseconds)",
  "probes": {
    "total": 9,
    "implemented_rust": 9,
    "implemented_python": 9,
    "tests_passed": 9
  },
  "pattern_elevation": {
    "redis_connected": true,
    "elevated_patterns": 0,
    "threshold": 3
  },
  "neo4j": {
    "connected": true,
    "nodes": 0
  },
  "performance": {
    "average_probe_ms": 50,
    "full_suite_ms": 450
  }
}
```

Save to: `docs/evidence/receipts/sape-validation-$(date +%Y%m%d-%H%M%S).json`

## Report Format

```
## SAPE Validation Report

**Status**: ✅ OPERATIONAL | ❌ DEGRADED
**Probes**: 9/9 implemented
**Tests**: X/X passed
**Elevations**: X patterns elevated

### Probe Status
[Table with all 9 probes and their status]

### Pattern Elevation
- Redis: ✓ Connected
- Patterns: X elevated
- Threshold: >3 repetitions

### High-Stakes Evidence
- Neo4j: ✓ Connected (optional)
- Graph nodes: X

### Performance
- Average probe: Xms (target: <100ms)
- Full suite: Xms (target: <1s)

### Receipt
- Location: docs/evidence/receipts/sape-validation-YYYYMMDD-HHMMSS.json
```

---

**SAPE Philosophy**: "Auto-elevate repeated patterns (>3 occurrences) into optimized kernel shortcuts. High-stakes probes require graph evidence."
