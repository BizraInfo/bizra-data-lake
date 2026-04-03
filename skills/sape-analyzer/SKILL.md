---
name: SAPE Probe Analyzer
description: Analyzes SAPE (Symbolic-Abstraction Probe Elevation) probe results and provides recommendations
keywords: [sape, probe, validation, pattern-elevation, verification]
user-invocable: false
disable-model-invocation: false
---

# SAPE Probe Analyzer Skill

## Purpose

This skill enables Claude to automatically analyze SAPE probe results, identify probe failures, recommend fixes, and track pattern elevation. It ensures the 9-probe verification system operates correctly and efficiently.

## When to Use

Claude should invoke this skill when:
- SAPE probe results are available
- Probe failures detected
- Pattern elevation threshold reached (>3 occurrences)
- Analyzing probe performance
- Investigating validation failures
- Optimizing probe execution

## SAPE 9-Probe System

1. **threat_scan** - Security threat detection
2. **compliance** - Policy compliance checking
3. **bias** - Bias and fairness analysis
4. **user_benefit** - User value assessment
5. **correctness** - Logical correctness validation
6. **safety** - Safety constraint verification
7. **groundedness** - Factual grounding check
8. **relevance** - Task relevance analysis
9. **fluency** - Output quality assessment

## Capabilities

### 1. Probe Result Analysis

Analyze probe execution results:
```json
{
  "probe_execution": {
    "threat_scan": {
      "passed": true,
      "score": 0.98,
      "execution_time_ms": 45,
      "evidence": []
    },
    "compliance": {
      "passed": true,
      "score": 0.97,
      "execution_time_ms": 52,
      "evidence": ["GDPR_COMPLIANT", "HIPAA_COMPLIANT"]
    },
    "bias": {
      "passed": false,
      "score": 0.82,
      "execution_time_ms": 67,
      "evidence": ["DETECTED_GENDER_BIAS"],
      "recommendation": "Review language for gender-neutral terms"
    }
  }
}
```

**Analysis output**:
```
✅ PASSED: 8/9 probes
❌ FAILED: bias (score: 0.82, threshold: 0.90)

Recommendations:
- Review bias probe failure: DETECTED_GENDER_BIAS
- Consider rephrasing output for gender neutrality
- Re-run probe after corrections
```

### 2. Pattern Elevation Detection

Track probe patterns and auto-elevate:
```json
{
  "pattern_hash": "abc123def456",
  "occurrences": 5,
  "first_seen": "2026-01-20T10:00:00Z",
  "last_seen": "2026-01-20T10:30:00Z",
  "probe_results": {
    "threat_scan": "pass",
    "compliance": "pass",
    "bias": "pass"
  },
  "elevation_status": "ELEVATED",
  "elevation_timestamp": "2026-01-20T10:30:00Z"
}
```

**When pattern seen >3 times**:
1. Calculate pattern hash
2. Store in Redis (`bizra:sape:elevation:{hash}`)
3. Create optimized shortcut
4. Skip full probes for this pattern
5. Report elevation to logs

### 3. Probe Performance Analysis

Identify slow probes:
```
Probe Performance Analysis:
===========================
threat_scan:    45ms ✓ (target: <100ms)
compliance:     52ms ✓ (target: <100ms)
bias:           67ms ✓ (target: <100ms)
user_benefit:   89ms ✓ (target: <100ms)
correctness:   120ms ⚠️ (target: <100ms)
safety:         43ms ✓ (target: <100ms)
groundedness:  156ms ⚠️ (target: <100ms)
relevance:      38ms ✓ (target: <100ms)
fluency:        41ms ✓ (target: <100ms)

Recommendations:
- Optimize correctness probe (120ms > 100ms target)
- Optimize groundedness probe (156ms > 100ms target)
- Consider caching for repeated patterns
```

### 4. Failure Root Cause Analysis

For probe failures:
```
Probe Failure: bias
Score: 0.82 (threshold: 0.90)
Evidence: DETECTED_GENDER_BIAS

Root Cause Analysis:
1. Input text: "The developer and his team..."
2. Bias detected: Gender assumption ("his")
3. Fix: Use gender-neutral language ("their")
4. Expected score after fix: >0.90

Recommended Actions:
- Replace "his" with "their"
- Review entire output for similar patterns
- Re-run bias probe
- Document pattern for future prevention
```

### 5. Cross-Probe Correlation

Identify related failures:
```
Correlation Analysis:
====================
bias FAILED (0.82)
  ↓ may affect
safety WARNING (0.91 - near threshold)
  ↓ related to
user_benefit PASSED (0.95)

Pattern: Bias failures often correlate with safety warnings
Recommendation: When bias fails, double-check safety probe
```

## Redis Integration

### Pattern Storage

Elevated patterns stored in Redis:
```
Key: bizra:sape:elevation:abc123def456
Value: {
  "pattern_hash": "abc123...",
  "occurrences": 5,
  "probe_results": {...},
  "optimized_shortcut": {...}
}
TTL: 3600 seconds (configurable via SAPE_CACHE_TTL)
```

### Elevation Queries

Check if pattern is elevated:
```bash
redis-cli -u $SYNAPSE_URL GET bizra:sape:elevation:${pattern_hash}
```

### Pattern Cleanup

Remove stale elevations:
```bash
# Patterns expire after SAPE_CACHE_TTL (default: 1 hour)
# Manual cleanup:
redis-cli -u $SYNAPSE_URL DEL bizra:sape:elevation:${pattern_hash}
```

## Neo4j Graph Evidence

For high-stakes probes, store evidence in Neo4j:

```cypher
CREATE (p:ProbeExecution {
  probe_name: 'threat_scan',
  timestamp: timestamp(),
  score: 0.98,
  passed: true
})

CREATE (e:Evidence {
  type: 'NO_THREATS_DETECTED',
  confidence: 0.99
})

CREATE (p)-[:HAS_EVIDENCE]->(e)
```

Query evidence:
```cypher
MATCH (p:ProbeExecution)-[:HAS_EVIDENCE]->(e:Evidence)
WHERE p.probe_name = 'threat_scan'
  AND p.timestamp > timestamp() - 86400000
RETURN p, e
ORDER BY p.timestamp DESC
LIMIT 10
```

## BIZRA Integration

### SAPE in Request Flow

```
User Request
  ↓
SAT Pre-Validation
  ↓
SAPE Probing ← [Use this skill here]
  ├─ Check pattern elevation
  ├─ Run 9 probes (or use shortcut)
  ├─ Analyze results
  ├─ Store in Redis/Neo4j
  └─ Elevate if >3 occurrences
  ↓
Ihsān Gate (score ≥ 0.99)
  ↓
PAT Execution
```

### Fail-Closed Enforcement

**BLOCK if**:
- Any critical probe fails
- threat_scan score < 0.90
- safety score < 0.90
- correctness score < 0.90

**WARN but allow**:
- Non-critical probes slightly below threshold
- Performance exceeds 100ms (investigate)
- Pattern elevation cache miss

### Evidence-Driven Workflow

Every SAPE run generates:
1. Probe execution receipt
2. Pattern hash (for elevation)
3. Redis storage entry
4. Neo4j evidence (high-stakes)
5. Performance metrics

## Example Usage

### Automatic Invocation

When Claude sees probe results in command output:
```
SAPE Probe Results:
threat_scan: PASS (0.98)
compliance: PASS (0.97)
bias: FAIL (0.82)
user_benefit: PASS (0.95)
...
```

Claude **automatically**:
1. Invokes SAPE Analyzer Skill
2. Analyzes which probes failed
3. Identifies root causes
4. Provides recommendations
5. Updates pattern elevation

### User-Requested

User says:
- "Analyze the SAPE probe results"
- "Why did the bias probe fail?"
- "Optimize SAPE performance"
- "Check pattern elevation status"

Claude should:
1. Use this skill
2. Provide detailed analysis
3. Show recommendations
4. Report pattern elevation status

## Analysis Report Template

```markdown
## SAPE Analysis Report

**Timestamp**: 2026-01-20T10:30:00Z
**Total Probes**: 9
**Passed**: 8
**Failed**: 1

### Probe Results

| Probe | Status | Score | Time | Evidence |
|-------|--------|-------|------|----------|
| threat_scan | ✅ PASS | 0.98 | 45ms | - |
| compliance | ✅ PASS | 0.97 | 52ms | GDPR, HIPAA |
| bias | ❌ FAIL | 0.82 | 67ms | GENDER_BIAS |
| ... | ... | ... | ... | ... |

### Failures

**bias** (score: 0.82, threshold: 0.90)
- Evidence: DETECTED_GENDER_BIAS
- Root cause: Gender assumptions in text
- Fix: Use gender-neutral language
- Re-run required: Yes

### Pattern Elevation

- Pattern hash: abc123def456
- Occurrences: 5
- Status: ELEVATED
- Shortcut active: Yes

### Performance

- Average probe time: 65ms ✓
- Slowest probe: groundedness (156ms) ⚠️
- Total execution: 585ms

### Recommendations

1. Fix bias probe failure (gender-neutral language)
2. Optimize groundedness probe (<100ms target)
3. Re-run probes after fixes
4. Monitor pattern elevation effectiveness
```

## Tools Required

- **Read**: Access probe result files
- **Bash**: Query Redis, calculate hashes, parse JSON
- **Grep**: Search for probe patterns
- **Write**: Save analysis reports (optional)

## Performance

- Analysis: <50ms
- Pattern lookup: <10ms (Redis)
- Graph query: <100ms (Neo4j)
- Total overhead: <200ms

## Quality Checks

Before reporting analysis:
- [ ] All 9 probes analyzed
- [ ] Failures identified with root causes
- [ ] Recommendations provided
- [ ] Pattern elevation checked
- [ ] Performance metrics calculated

---

**Skill Philosophy**: "SAPE probes are the immune system. Analyze failures quickly, elevate patterns efficiently, ensure security rigorously."

## Usage Pattern

```
User: "Run SAPE validation on this output"

Claude:
1. Executes: SAPE 9-probe suite
2. Observes: bias probe failed (0.82)
3. Invokes: SAPE Analyzer Skill
4. Analyzes: Root cause = gender assumptions
5. Reports: "SAPE: 8/9 passed. Bias failed (0.82) - use gender-neutral language"
6. Recommends: Specific fixes
7. Updates: Pattern elevation if applicable
```

**This happens automatically** - users get immediate analysis with actionable recommendations.
