---
paths:
  - "src/sape*.rs"
  - "core/sape*.py"
  - "bizra_kernel/sape*.py"
  - "scripts/sape*.py"
---

# SAPE Validation Rules

Rules for BIZRA's Symbolic-Abstraction Probe Elevation system.

## 9-Probe System

### Required Probes

| Probe | Purpose | Threshold |
|-------|---------|-----------|
| threat_scan | Security threat detection | 0.90 |
| compliance | Policy compliance | 0.90 |
| bias | Fairness and bias detection | 0.90 |
| user_benefit | User value assessment | 0.85 |
| correctness | Logical correctness | 0.90 |
| safety | Safety constraint compliance | 0.95 |
| groundedness | Factual accuracy | 0.85 |
| relevance | Task relevance | 0.85 |
| fluency | Output quality | 0.80 |

### Probe Order
1. **Critical first**: threat_scan, safety (block early)
2. **Compliance**: compliance, bias
3. **Quality**: correctness, groundedness, relevance
4. **User-facing**: user_benefit, fluency

## Implementation Patterns

### Probe Structure
```python
@dataclass
class ProbeResult:
    probe_name: str
    passed: bool
    score: float  # 0.0 to 1.0
    execution_time_ms: int
    evidence: list[str]
    recommendation: str | None = None
```

### Running Probes
```rust
async fn run_sape_probes(&self, context: &ProbeContext) -> Result<Vec<ProbeResult>> {
    let probes = [
        self.probe_threat_scan(context),
        self.probe_compliance(context),
        self.probe_bias(context),
        self.probe_user_benefit(context),
        self.probe_correctness(context),
        self.probe_safety(context),
        self.probe_groundedness(context),
        self.probe_relevance(context),
        self.probe_fluency(context),
    ];

    let results = futures::future::join_all(probes).await;

    // Check for critical failures early
    for result in &results {
        if !result.passed && is_critical_probe(&result.probe_name) {
            return Err(BizraError::CriticalProbeFailure(result.clone()));
        }
    }

    Ok(results)
}
```

## Pattern Elevation

### Elevation Trigger
Patterns with >3 occurrences should be elevated to optimized shortcuts.

```python
async def check_elevation(pattern_hash: str, probe_results: list[ProbeResult]):
    key = f"bizra:sape:elevation:{pattern_hash}"
    count = await redis.incr(key)

    if count > 3:
        # Elevate pattern
        shortcut = create_optimized_shortcut(probe_results)
        await redis.set(f"{key}:shortcut", json.dumps(shortcut))
        logger.info(f"Pattern elevated: {pattern_hash}")
```

### Cache TTL
- Elevation cache: 3600 seconds (configurable via `SAPE_CACHE_TTL`)
- Clear cache on constitution changes

## Performance Requirements

### Timing Targets
- Individual probe: < 100ms
- Full probe suite: < 500ms
- Pattern lookup: < 10ms

### Monitoring
Track these metrics:
- `bizra_sape_probe_duration_seconds{probe}`
- `bizra_sape_probe_passed{probe}`
- `bizra_sape_elevation_hits`
- `bizra_sape_elevation_misses`

## Error Handling

### Probe Failures
```python
# Don't fail silently
for result in probe_results:
    if not result.passed:
        logger.warning(
            f"SAPE probe failed",
            extra={
                "probe": result.probe_name,
                "score": result.score,
                "evidence": result.evidence
            }
        )

        # Emit evidence
        await emit_probe_failure_receipt(result)

        # Escalate if critical
        if is_critical_probe(result.probe_name):
            await fate.escalate(EscalationLevel.HIGH, f"Critical probe failure: {result.probe_name}")
```

### Timeout Handling
```rust
// Wrap probe with timeout
let result = tokio::time::timeout(
    Duration::from_millis(PROBE_TIMEOUT_MS),
    self.probe_threat_scan(context)
).await;

match result {
    Ok(probe_result) => probe_result,
    Err(_) => ProbeResult::timeout("threat_scan"),
}
```

## Neo4j Graph Evidence

For high-stakes probes, store evidence in Neo4j:

```cypher
CREATE (p:ProbeExecution {
    probe_name: 'threat_scan',
    timestamp: timestamp(),
    score: 0.98,
    passed: true,
    task_id: $task_id
})

CREATE (e:Evidence {
    type: 'NO_THREATS_DETECTED',
    confidence: 0.99,
    details: $details
})

CREATE (p)-[:HAS_EVIDENCE]->(e)
```

## Testing Requirements

- Test each probe individually
- Test probe order execution
- Test pattern elevation trigger
- Test timeout handling
- Test critical vs non-critical failure handling
- Test graph evidence storage
- Mock LLM calls in tests
