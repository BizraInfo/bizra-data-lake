---
allowed-tools: Bash(python*:*), Bash(cargo:*), Bash(docker:*), Read, Write, Edit, Grep, Glob, WebSearch, WebFetch
description: Peak Performance Mode - Maximum quality output with all systems engaged
argument-hint: [task-description]
---

# Peak - Peak Performance Mode

## Overview

Peak mode engages **ALL BIZRA systems** at maximum capacity for the highest possible output quality. Use for critical, high-stakes tasks where excellence is non-negotiable.

## Peak Mode Activation Checklist

```
┌─────────────────────────────────────────────────────────────────┐
│                    PEAK PERFORMANCE MODE                        │
├─────────────────────────────────────────────────────────────────┤
│  ◉ Giants Protocol     - Prior art search complete              │
│  ◉ Maestro Trinity     - Multi-domain analysis active           │
│  ◉ Graph of Thoughts   - Non-linear reasoning engaged           │
│  ◉ SNR Routing         - T0 Elite tier confirmed                │
│  ◉ Data Lake           - All M1-M6 tiers queried                │
│  ◉ SAPE 9-Probe        - Full validation suite active           │
│  ◉ Ihsān Gate          - 0.95 threshold enforced                │
│  ◉ SAT Consensus       - 3/5 guardian approval required         │
│  ◉ Swarm Intelligence  - Multi-agent collaboration enabled      │
│  ◉ Receipt Generation  - Full evidence chain active             │
├─────────────────────────────────────────────────────────────────┤
│  Status: PEAK MODE ACTIVE                                       │
└─────────────────────────────────────────────────────────────────┘
```

## Current System Status

Check status manually:
```bash
# Elite Engine (Rust)
curl -s http://localhost:8080/health 2>/dev/null | jq -r '.status // "Not running"' || echo "Not running"

# Kernel (Python)
curl -s http://localhost:8010/health 2>/dev/null | jq -r '.status // "Not running"' || echo "Not running"

# Docker Services
docker compose ps --format '{{.Name}}: {{.Status}}' 2>/dev/null | grep -c running || echo "0"

# Data Lake MCP (if running)
curl -sk https://localhost:8443/health 2>/dev/null | jq -r '.status // "Not accessible"' || echo "Not accessible"

# Neo4j (Wisdom)
curl -s http://localhost:7474 2>/dev/null && echo "Running" || echo "Not running"
```

## Peak Mode Protocol

### Phase 0: System Readiness

**Verify all systems operational**:

```bash
echo "=== PEAK MODE SYSTEM CHECK ==="

# Check Rust Elite Engine
if curl -s http://localhost:8080/health | grep -q "ok"; then
    echo "✓ Elite Engine: READY"
else
    echo "✗ Elite Engine: NOT READY"
fi

# Check Python Kernel
if curl -s http://localhost:8010/health | grep -q "ok"; then
    echo "✓ Python Kernel: READY"
else
    echo "✗ Python Kernel: NOT READY"
fi

# Check Docker Services
running=$(docker compose ps --format '{{.Status}}' 2>/dev/null | grep -c running)
total=$(docker compose ps --format '{{.Status}}' 2>/dev/null | wc -l)
echo "✓ Docker Services: ${running}/${total} running"

# Check Data Lake
if curl -sk https://localhost:8443/health | grep -q "ok"; then
    echo "✓ Data Lake MCP: READY"
else
    echo "⚠ Data Lake MCP: DEGRADED (continuing without)"
fi

echo "=== SYSTEM CHECK COMPLETE ==="
```

### Phase 1: Giants Protocol (Prior Art)

**Before any implementation, search for existing solutions**:

1. Search internal codebase for related patterns
2. Check BIZRA ecosystem repositories
3. Review industry best practices
4. Consider academic foundations

**Document findings before proceeding**.

### Phase 2: SNR Classification

**Verify task is T0 (Elite) worthy**:

```python
def verify_t0_eligibility(task):
    """Task must meet T0 criteria for Peak mode"""
    criteria = {
        "specificity": task.has_clear_requirements,
        "measurability": task.has_success_criteria,
        "scope_clarity": task.has_defined_boundaries,
        "context": task.has_sufficient_context,
        "precision": task.uses_technical_terms_correctly
    }

    if not all(criteria.values()):
        return False, "Task does not meet T0 criteria"

    return True, "T0 Elite confirmed"
```

If not T0, **elevate to T0** by gathering missing information.

### Phase 3: Maestro Trinity Analysis

**Analyze across all three domains**:

| Domain | Analysis Required | Key Questions |
|--------|-------------------|---------------|
| **Ideology** | Ethics, Philosophy | Ihsān dimensions? Sovereignty impact? |
| **AI/ML** | Technical approach | Which models? Which agents? |
| **Blockchain** | Decentralization | Receipts? Consensus? Graph updates? |

**Resolve any cross-domain conflicts**.

### Phase 4: Graph of Thoughts Reasoning

**For complex problems, build reasoning graph**:

```
[Q] Root Question
    │
    ├── [A1] Technical Analysis
    │   ├── [E1.1] Evidence: ...
    │   └── [E1.2] Evidence: ...
    │
    ├── [A2] Ethical Analysis
    │   ├── [E2.1] Evidence: ...
    │   └── [E2.2] Contradiction: ...
    │
    └── [A3] Practical Analysis
        ├── [E3.1] Evidence: ...
        └── [S1] Synthesis → [D] Decision
```

### Phase 5: Data Lake Query

**Query all relevant memory tiers**:

```bash
# Query unified memory
python3 << 'EOF'
import asyncio
from core.unified_memory import get_unified_memory

async def peak_query(task):
    memory = await get_unified_memory()

    # Query all tiers
    results = {
        "M1_session": await memory.query(task, tier="M1"),
        "M3_semantic": await memory.query(task, tier="M3"),
        "M4_procedural": await memory.query(task, tier="M4"),
        "M6_sovereign": await memory.query_sovereign(task)
    }

    return results

# Execute
results = asyncio.run(peak_query("{task}"))
for tier, data in results.items():
    print(f"{tier}: {len(data)} results")
EOF
```

### Phase 6: Swarm Activation (if needed)

**For complex tasks, activate multi-agent swarm**:

```python
swarm_config = {
    "mode": "collaborative",
    "agents": [
        "MasterReasoner",   # Strategy
        "DataAnalyzer",     # Analysis
        "EthicsGuardian",   # Validation
        "CreativeSynthesizer"  # Generation
    ],
    "consensus_required": True
}
```

### Phase 7: Full SAPE Validation

**All 9 probes MUST pass**:

| Probe | Threshold | Critical |
|-------|-----------|----------|
| threat_scan | 0.95 | YES |
| compliance | 0.95 | YES |
| bias | 0.90 | YES |
| user_benefit | 0.85 | YES |
| correctness | 0.95 | YES |
| safety | 0.95 | YES |
| groundedness | 0.85 | NO |
| relevance | 0.80 | NO |
| fluency | 0.80 | NO |

### Phase 8: SAT Consensus

**Require 3/5 guardian approval**:

```
SAT Guardians:
  ◉ PoiVerifier      - Impact verification
  ◉ ResourceAllocator - Efficiency check
  ◉ RiskGuardian     - Security assessment
  ◉ GovernanceEngine - Policy compliance
  ◉ EvidenceEngine   - Audit trail

Required: 3/5 approval for Peak mode execution
```

### Phase 9: Ihsān Gate

**Enforce 0.95 threshold across all 8 dimensions**:

```yaml
ihsan_dimensions:
  correctness: 0.22        # Is it right?
  safety: 0.22             # Is it safe?
  user_benefit: 0.14       # Does it help?
  efficiency: 0.12         # Is it optimal?
  auditability: 0.12       # Can it be reviewed?
  anti_centralization: 0.08 # Does it decentralize?
  robustness: 0.06         # Is it resilient?
  adl_fairness: 0.04       # Is it fair?

threshold: 0.95            # REQUIRED for Peak mode
```

### Phase 10: Execution & Evidence

**Execute with full receipt chain**:

```json
{
  "receipt_id": "peak-{timestamp}",
  "mode": "PEAK",
  "systems_engaged": [
    "giants", "maestro", "got", "snr",
    "lake", "swarm", "sape", "sat", "ihsan"
  ],
  "execution": {...},
  "evidence_chain": [...]
}
```

## Peak Mode Template

### Task: [User's Task]

---

#### Phase 0: System Readiness

| System | Status | Ready |
|--------|--------|-------|
| Elite Engine | ... | ✓/✗ |
| Python Kernel | ... | ✓/✗ |
| Docker Services | X/Y | ✓/✗ |
| Data Lake | ... | ✓/⚠ |
| Neo4j | ... | ✓/⚠ |

---

#### Phase 1: Giants (Prior Art)

| Source | Patterns Found | Applicable |
|--------|----------------|------------|
| Internal | X | Y |
| Ecosystem | X | Y |
| Industry | X | Y |

---

#### Phase 2: SNR Classification

**Tier**: T0 Elite
**SNR Score**: 0.XX
**Confirmation**: [criteria met]

---

#### Phase 3: Maestro Trinity

| Domain | Analysis | Constraints |
|--------|----------|-------------|
| Ideology | ... | ... |
| AI/ML | ... | ... |
| Blockchain | ... | ... |

---

#### Phase 4: GoT Reasoning

```
[Graph visualization]
```

---

#### Phase 5: Data Lake Query

| Tier | Results | Top Relevance |
|------|---------|---------------|
| M1 | X | 0.XX |
| M3 | X | 0.XX |
| M4 | X | 0.XX |
| M6 | X | 0.XX |

---

#### Phase 6: Swarm (if used)

| Agent | Role | Status |
|-------|------|--------|
| ... | ... | ... |

---

#### Phase 7: SAPE Validation

| Probe | Score | Pass |
|-------|-------|------|
| threat_scan | 0.XX | ✓ |
| compliance | 0.XX | ✓ |
| bias | 0.XX | ✓ |
| user_benefit | 0.XX | ✓ |
| correctness | 0.XX | ✓ |
| safety | 0.XX | ✓ |
| groundedness | 0.XX | ✓ |
| relevance | 0.XX | ✓ |
| fluency | 0.XX | ✓ |

---

#### Phase 8: SAT Consensus

| Guardian | Vote | Reason |
|----------|------|--------|
| PoiVerifier | APPROVE | ... |
| ResourceAllocator | APPROVE | ... |
| RiskGuardian | APPROVE | ... |
| GovernanceEngine | ... | ... |
| EvidenceEngine | ... | ... |

**Consensus**: 3/5 ✓

---

#### Phase 9: Ihsān Gate

| Dimension | Weight | Score |
|-----------|--------|-------|
| correctness | 0.22 | 0.XX |
| safety | 0.22 | 0.XX |
| user_benefit | 0.14 | 0.XX |
| efficiency | 0.12 | 0.XX |
| auditability | 0.12 | 0.XX |
| anti_centralization | 0.08 | 0.XX |
| robustness | 0.06 | 0.XX |
| adl_fairness | 0.04 | 0.XX |

**Total Score**: 0.XX (threshold: 0.95) ✓

---

#### Phase 10: Execution

[Execution output]

---

## Evidence Generation

Generate Peak mode receipt:

```json
{
  "receipt_id": "peak-$(date +%s)",
  "timestamp": "$(date -Iseconds)",
  "mode": "PEAK",
  "task_summary": "[task]",
  "systems_engaged": {
    "giants": {"prior_art_found": 0},
    "maestro": {"domains_analyzed": 3},
    "got": {"nodes": 0, "edges": 0},
    "snr": {"tier": "T0", "score": 0.0},
    "lake": {"tiers_queried": 4, "results": 0},
    "swarm": {"agents": 0, "mode": null},
    "sape": {"probes_passed": 9, "probes_total": 9},
    "sat": {"approvals": 3, "required": 3},
    "ihsan": {"score": 0.0, "threshold": 0.95}
  },
  "execution": {
    "status": "SUCCESS",
    "duration_ms": 0,
    "output_summary": ""
  },
  "evidence_chain": [],
  "integrity_hash": ""
}
```

## Report Format

```
## Peak Performance Report

**Task**: [task]
**Mode**: PEAK (All Systems Engaged)
**Timestamp**: [ISO timestamp]

### System Status

| System | Status |
|--------|--------|
| Elite Engine | ✓ |
| Python Kernel | ✓ |
| Docker (X/Y) | ✓ |
| Data Lake | ✓ |

### Protocol Execution

| Phase | System | Status | Key Output |
|-------|--------|--------|------------|
| 1 | Giants | ✓ | X prior art found |
| 2 | SNR | ✓ | T0 confirmed |
| 3 | Maestro | ✓ | 3 domains analyzed |
| 4 | GoT | ✓ | X nodes, Y edges |
| 5 | Lake | ✓ | X results from 4 tiers |
| 6 | Swarm | ✓/- | X agents / skipped |
| 7 | SAPE | ✓ | 9/9 probes passed |
| 8 | SAT | ✓ | 3/5 consensus |
| 9 | Ihsān | ✓ | 0.XX >= 0.95 |
| 10 | Execute | ✓ | Complete |

### Output

[Execution result]

### Evidence Chain

1. giants-[ts] → 2. snr-[ts] → 3. maestro-[ts] → ...

### Receipt
- ID: peak-[timestamp]
- Location: docs/evidence/receipts/
```

---

**Peak Philosophy**: "Excellence is not an accident. It's the result of systematic activation of every available capability. Peak mode leaves nothing to chance."

---

## PAT Peak Mode (Advanced)

PAT Peak Mode extends standard Peak with **5 Validation Gates**, **Cross-Domain Synthesis**, **Novelty Boosting**, and **Elite Practitioner Anchoring**.

### PAT Mode Tiers

```
┌─────────────────────────────────────────────────────────────────┐
│                    PAT PEAK MODE TIERS                          │
├─────────────────────────────────────────────────────────────────┤
│  STANDARD      │ SNR >= 0.980  │ 3 domains  │ Novelty >= 0.75   │
│  ELEVATED      │ SNR >= 0.985  │ 4 domains  │ Novelty >= 0.80   │
│  SOVEREIGN     │ SNR >= 0.990  │ 5 domains  │ Novelty >= 0.85   │
│  TRANSCENDENT  │ SNR >= 0.995  │ 6 domains  │ Novelty >= 0.90   │
├─────────────────────────────────────────────────────────────────┤
│  Current Tier: STANDARD                                         │
└─────────────────────────────────────────────────────────────────┘
```

### PAT System Status

Check PAT components:
```bash
python3 << 'EOF'
import sys
sys.path.insert(0, '.')

try:
    from bizra_kernel.pat_unified_orchestrator import PATUnifiedOrchestrator
    from bizra_kernel.pat_telemetry import PATTelemetry
    from bizra_kernel.pat_citation_validator import PATCitationValidator

    print("✓ PAT Unified Orchestrator: LOADED")
    print("✓ PAT Telemetry System: LOADED")
    print("✓ PAT Citation Validator: LOADED")

    # Check telemetry
    telemetry = PATTelemetry()
    report = telemetry.generate_report(3600)
    print(f"  - SNR Average: {report.snr_average:.4f}")
    print(f"  - Novelty Average: {report.novelty_average:.4f}")

except ImportError as e:
    print(f"⚠ PAT Components: PARTIAL ({e})")
except Exception as e:
    print(f"✗ PAT Status: ERROR ({e})")
EOF
```

### PAT 5-Gate Validation Protocol

| Gate | Phase | Checks | Threshold | Correction |
|------|-------|--------|-----------|------------|
| **Gate 1** | Pre-Reasoning | Domains ≥3, Unrelatedness ≥0.70 | 0.70 | Expand domains |
| **Gate 2** | Mid-Synthesis | Running SNR ≥0.95, Contradictions | 0.95 | Prune nodes |
| **Gate 3** | Post-Synthesis | SNR ≥0.98, Novelty ≥0.75, Coverage | 0.98/0.75 | Additional synthesis |
| **Gate 4** | Practitioner | 3+/domain, top_1% tier, relevance | 3/0.60 | Fetch practitioners |
| **Gate 5** | Response | 6 sections, claim tags | 100% | Reformat |

### PAT 6-Section Response Structure

Every PAT Peak response MUST include:

```markdown
## Executive Synthesis
- [MEASURED] Key finding 1
- [IMPLEMENTED] Key finding 2
- [NOVEL] Novel insight

## Domain Cross-Pollination Map
**Domains:** A, B, C
**Unrelatedness:** 0.XX

## Elite Practitioner Anchoring
| Domain | Practitioner | Tier | Relevance |
| ------ | ------------ | ---- | --------- |
| ... | ... | top_1% | 0.XX |

## Novel Insight Synthesis
**Novelty Score:** 0.XX (threshold: 0.75)

## Validation Evidence Trail
| Gate | Status | Score |
| ---- | ------ | ----- |
**Receipt ID:** `pat-XXXXXX`

## Actionable Recommendations
### What We Know
### What We Assume
### What We Should Test Next
```

### PAT Claim Tags

| Tag | Weight | Use When |
|-----|--------|----------|
| `[MEASURED]` | 1.00 | Empirically verified data |
| `[IMPLEMENTED]` | 0.95 | Working code exists |
| `[DERIVED]` | 0.90 | Logically derived from facts |
| `[NOVEL]` | 1.00 | Cross-domain synthesis (≥0.75 novelty) |
| `[CROSS_DOMAIN]` | 0.95 | Multi-domain connection |
| `[DESIGNED]` | 0.75 | Specification only |
| `[TARGET]` | 0.50 | Aspiration/goal |
| `[HYPOTHESIS]` | 0.40 | Requires testing |
| `[METAPHOR]` | 0.00 | Figurative only |

### PAT Telemetry Commands

Check live metrics:
```bash
python3 << 'EOF'
import sys
sys.path.insert(0, '.')
from bizra_kernel.pat_telemetry import PATTelemetry

telemetry = PATTelemetry()
report = telemetry.generate_report(3600)

print(f"=== PAT TELEMETRY (Last Hour) ===")
print(f"SNR:     avg={report.snr_average:.4f}, min={report.snr_min:.4f}, max={report.snr_max:.4f}")
print(f"Novelty: avg={report.novelty_average:.4f}, min={report.novelty_min:.4f}, max={report.novelty_max:.4f}")
print(f"Domains: avg={report.domains_average:.1f}, max={report.domains_max}")
print(f"Status: {report.overall_status}")
EOF
```

### PAT Citation Validation

Validate elite practitioner citations:
```bash
python3 << 'EOF'
import sys
sys.path.insert(0, '.')
from bizra_kernel.pat_citation_validator import PATCitationValidator

validator = PATCitationValidator()
test_content = """
According to Terence Tao, mathematical foundations require careful analysis.
Using Leslie Lamport's distributed consensus approach ensures safety.
Derek Parfit's ethical framework guides decision-making.
"""

result = validator.validate_response(test_content)
print(validator.generate_citation_report(result))
EOF
```

### PAT Peak Full Orchestration

Execute complete PAT pipeline:
```bash
python3 << 'EOF'
import sys
import asyncio
sys.path.insert(0, '.')

async def pat_peak(query, response):
    from bizra_kernel.pat_unified_orchestrator import PATUnifiedOrchestrator

    orchestrator = PATUnifiedOrchestrator("pat-peak-session")
    result = await orchestrator.orchestrate(query, response, {})

    print(f"=== PAT PEAK RESULT ===")
    print(f"Mode: {result.mode.value}")
    print(f"SNR: {result.snr_score:.4f}")
    print(f"Novelty: {result.novelty_score:.4f}")
    print(f"Gates: {result.gates_passed}/{result.gates_total}")
    print(f"Status: {'PASS' if result.overall_pass else 'FAIL'}")
    print(f"Receipt: {result.receipt_id}")

asyncio.run(pat_peak("Example query", "Example response"))
EOF
```

---

**PAT Philosophy**: "Peak Autonomous Think Tank - SNR >= 0.98, 3+ unrelated domains synthesize novel insights, elite practitioners anchor every claim. Maximum enforcement, maximum excellence."
