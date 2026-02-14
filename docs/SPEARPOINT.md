# Spearpoint — Recursive Discovery & Verification Engine

> **Version:** 1.1.0
> **Package:** `core/spearpoint/`
> **Tests:** 161 passing (`tests/core/spearpoint/`)
> **Bridge:** Available as `rdve_research` skill via [Desktop Bridge](DESKTOP_BRIDGE.md)

## Overview

Spearpoint is the autonomous research engine of BIZRA. It runs a Generator-Verifier bicameral loop that produces, evaluates, and refines hypotheses using 15 thinking patterns extracted from 3,819 top-tier AI papers.

```
┌─────────────────────────────────────────────────────┐
│  RDVE Skill (invoke_skill "rdve_research")          │
│  core/spearpoint/rdve_skill.py                      │
└─────────────────────┬───────────────────────────────┘
                      │
┌─────────────────────▼───────────────────────────────┐
│  SpearpointOrchestrator                              │
│  core/spearpoint/orchestrator.py                     │
│  Routes missions to AutoResearcher or AutoEvaluator  │
├──────────────────┬──────────────────────────────────┤
│                  │                                   │
│  IMPROVE path    │  REPRODUCE path                   │
│  AutoResearcher  │  AutoEvaluator                    │
│  (generate)      │  (falsify)                        │
│       │          │       │                           │
│  PatternSelector │  [run experiments]                │
│  SciReasoning    │  [check reproducibility]          │
│  Bridge          │  verdict: APPROVED | REJECTED     │
│  (15 patterns)   │                                   │
└──────────────────┴──────────────────────────────────┘
         │
┌────────▼────────────────────────────────────────────┐
│  RecursiveLoop — OODA Heartbeat                      │
│  Circuit breaker: 3 consecutive rejections → backoff │
│  Pattern rotation on stagnation                      │
└─────────────────────────────────────────────────────┘
```

---

## Quick Start

### Via Desktop Bridge (recommended)

```bash
# Start bridge
PYTHONPATH=/mnt/c/BIZRA-DATA-LAKE python -m core.bridges.desktop_bridge

# Invoke RDVE statistics
python3 -c "
import os, socket, json, time, uuid
s = socket.socket(); s.connect(('127.0.0.1', 9742))
token = os.environ['BIZRA_BRIDGE_TOKEN']
msg = json.dumps({
    'jsonrpc': '2.0', 'method': 'invoke_skill',
    'params': {'skill': 'rdve_research', 'inputs': {'operation': 'statistics'}},
    'id': 1,
    'headers': {
        'X-BIZRA-TOKEN': token,
        'X-BIZRA-TS': int(time.time() * 1000),
        'X-BIZRA-NONCE': uuid.uuid4().hex,
    }
}).encode() + b'\n'
s.sendall(msg); print(s.recv(4096).decode()); s.close()
"
```

### Via Python API

```python
from core.spearpoint import SpearpointOrchestrator, SpearpointConfig, MissionType

config = SpearpointConfig()
orch = SpearpointOrchestrator(config=config)

# Run an improvement mission
mission = await orch.run_mission(mission_type=MissionType.IMPROVE)
print(f"Mission {mission.mission_id}: {mission.success}")

# Research with a specific pattern
result = await orch.research_pattern(pattern_id="P01", claim_context="optimize attention")
print(f"Pattern P01 result: {result}")
```

---

## The 15 Thinking Patterns

Source: Li et al. (2025) "[Sci-Reasoning: A Dataset Decoding AI Innovation Patterns](https://arxiv.org/abs/2601.04577)" — 3,819 Oral/Spotlight papers from NeurIPS, ICML, ICLR.

| ID | Pattern | Category | Description |
|----|---------|----------|-------------|
| P01 | Prior Work Extraction | Foundation | Extract and build upon established methods |
| P02 | Gap Identification | Foundation | Identify limitations in existing approaches |
| P03 | Constraint Specification | Foundation | Formalize problem requirements |
| P04 | Empirical Derivation | Methodology | Derive methods from experimental data |
| P05 | Theoretical Derivation | Methodology | Derive from formal mathematical analysis |
| P06 | Analogical Derivation | Methodology | Adapt methods by analogy from related domains |
| P07 | Component Integration | Architecture | Combine existing components into new systems |
| P08 | Workflow Design | Architecture | Design multi-step processing pipelines |
| P09 | Assumption Relaxation | Innovation | Remove limiting assumptions to generalize |
| P10 | Representation Shift | Innovation | Replace fundamental data primitives |
| P11 | Cross-Domain Synthesis | Innovation | Import solutions from distant fields |
| P12 | Complementary Integration | Synthesis | Merge complementary approaches |
| P13 | Support Extension | Synthesis | Extend a base method with supporting evidence |
| P14 | Generalization | Abstraction | Abstract specific methods to broader contexts |
| P15 | Hierarchical Structuring | Abstraction | Organize solutions in layered hierarchies |

### Pattern Selection

The `PatternStrategySelector` in `core/spearpoint/pattern_selector.py` uses Thompson Sampling (Beta distributions) with four strategies:

| Strategy | When Used | Behavior |
|----------|-----------|----------|
| **Explore** | High uncertainty | Sample from Beta priors to discover new patterns |
| **Exploit** | High confidence | Select the pattern with highest success rate |
| **Complement** | After success | Choose a complementary pattern from a different category |
| **Rotate** | After circuit breaker | Force a different category to break stagnation |

---

## Mission Types

### IMPROVE — Generate Better Solutions

The AutoResearcher generates hypotheses using Sci-Reasoning patterns, then evaluates them.

```python
mission = await orch.run_mission(
    mission_type=MissionType.IMPROVE,
    observation={"snr_score": 0.91, "latency_ms": 120}
)
```

Flow: PatternSelector → SciReasoningBridge → AutoResearcher → hypothesis → evaluation

### REPRODUCE — Falsify Claims

The AutoEvaluator attempts to reproduce and falsify a claim.

```python
mission = await orch.run_mission(
    mission_type=MissionType.REPRODUCE,
    claim="System achieves <100ms P99 latency"
)
```

Flow: claim → AutoEvaluator → experiment → verdict (APPROVED/REJECTED)

---

## RDVE Skill Handler

The `RDVESkillHandler` (`core/spearpoint/rdve_skill.py`) is a thin bridge adapter that wraps `SpearpointOrchestrator` for invocation through the Desktop Bridge.

### Registration

Registered automatically when the Desktop Bridge lazy-loads its SkillRouter:

```python
# In desktop_bridge.py:_get_skill_router()
from core.spearpoint.rdve_skill import register_rdve_skill
register_rdve_skill(router)  # Registers as "rdve_research", agent "rdve-researcher"
```

### 4 Operations

| Operation | Required | Optional | Returns |
|-----------|----------|----------|---------|
| `research_pattern` | `pattern_id` | `claim_context`, `top_k` | Mission result with pattern data |
| `reproduce` | `claim` | `proposed_change` | Mission result with verdict |
| `improve` | — | `observation`, `top_k` | Mission result with improvement |
| `statistics` | — | — | RDVE metadata, invocation count |

### Examples

**Research with Cross-Domain Synthesis (P11):**
```json
{
  "method": "invoke_skill",
  "params": {
    "skill": "rdve_research",
    "inputs": {
      "operation": "research_pattern",
      "pattern_id": "P11",
      "claim_context": "apply graph neural networks to time series forecasting"
    }
  }
}
```

**Reproduce a performance claim:**
```json
{
  "method": "invoke_skill",
  "params": {
    "skill": "rdve_research",
    "inputs": {
      "operation": "reproduce",
      "claim": "Attention mechanism achieves 95% accuracy on MMLU"
    }
  }
}
```

---

## Circuit Breaker

Located in `core/spearpoint/recursive_loop.py`. Prevents runaway research loops.

| Parameter | Value | Effect |
|-----------|-------|--------|
| **Trip threshold** | 3 consecutive rejections | Pauses research loop |
| **Backoff** | Exponential (2x, capped at 60s) | Delays next attempt |
| **Recovery** | First approved mission | Resets breaker |
| **Pattern rotation** | On trip | Forces different pattern category |

### Prometheus Alerts

| Alert | Severity | Trigger |
|-------|----------|---------|
| `BizraRDVECircuitBreakerTripped` | warning | Breaker tripped in last 10m |
| `BizraRDVERejectionRateHigh` | warning | >50% rejection rate over 15m |
| `BizraRDVEInactive` | info | No missions in 30m |

---

## Sci-Reasoning Data

Static data assets in `data/sci_reasoning/`:

| File | Size | Contents |
|------|------|----------|
| `classified_papers.json` | 1.7 MB | 3,291 papers with pattern classifications |
| `pattern_taxonomy.json` | 25 KB | Taxonomy of 15 patterns with metadata |
| `prior_works/` | varies | Paper lineage graphs (citation chains) |

Loaded by `SciReasoningBridge` (`core/bridges/sci_reasoning_bridge.py`). If files are missing, the bridge returns empty data gracefully.

---

## Monitoring

### Prometheus Recording Rules

| Rule | Expression | Interval |
|------|-----------|----------|
| `bizra:rdve_missions:rate5m` | `rate(rdve_missions_total[5m])` | 60s |
| `bizra:rdve_success_rate:avg5m` | approved / total (5m avg) | 60s |
| `bizra:rdve_pattern_usage:rate5m` | `rate(rdve_pattern_invocations_total[5m])` | 60s |
| `bizra:rdve_hypotheses:rate5m` | `rate(rdve_hypotheses_evaluated_total[5m])` | 60s |
| `bizra:rdve_breaker_trips:total` | `rdve_circuit_breaker_trips_total` | 60s |

Defined in `deploy/monitoring/prometheus-config.yaml` (group: `bizra.rdve.recording`).

---

## File Map

```
core/spearpoint/
├── __init__.py                 # Public API exports
├── sovereign_spearpoint.py     # Foundation: SNR/Ihsan calc, Z3 verification
├── config.py                   # SpearpointConfig, TierLevel, MissionType
├── orchestrator.py             # SpearpointOrchestrator (mission router)
├── auto_researcher.py          # AutoResearcher (improve path)
├── auto_evaluator.py           # AutoEvaluator (reproduce path)
├── recursive_loop.py           # RecursiveLoop + circuit breaker
├── pattern_selector.py         # Thompson Sampling pattern selection
├── metrics_provider.py         # Prometheus metrics
└── rdve_skill.py               # RDVE bridge adapter (4 operations)

core/bridges/
├── sci_reasoning_bridge.py     # 15 patterns, 3,291 papers, lineage graphs
└── sci_reasoning_patterns.py   # PatternID enum (P01-P15), ThinkingPattern

data/sci_reasoning/
├── classified_papers.json      # Paper classifications
├── pattern_taxonomy.json       # Pattern metadata
└── prior_works/                # Citation lineage graphs
```

---

## Testing

```bash
# Full spearpoint suite (161 tests)
PYTHONPATH=/mnt/c/BIZRA-DATA-LAKE pytest tests/core/spearpoint/ -v

# RDVE skill only (28 tests)
PYTHONPATH=/mnt/c/BIZRA-DATA-LAKE pytest tests/core/spearpoint/test_rdve_skill.py -v

# Sci-Reasoning bridge (47 tests)
PYTHONPATH=/mnt/c/BIZRA-DATA-LAKE pytest tests/core/bridges/test_sci_reasoning_bridge.py -v
```

---

## Standing on Giants

| Researcher | Year | Contribution | Spearpoint Integration |
|-----------|------|-------------|----------------------|
| **Li et al.** | 2025 | Sci-Reasoning: 15 thinking patterns | Pattern router + Thompson Sampling |
| **Lu et al.** | 2024 | AI Scientist: agentic tree search | AutoResearcher generation loop |
| **Gu et al.** | 2024 | CORE-Bench: computational reproducibility | AutoEvaluator falsification |
| **Besta et al.** | 2024 | Graph-of-Thoughts: non-linear reasoning | Research graph topology |
| **Boyd** | 1995 | OODA loop: observe-orient-decide-act | RecursiveLoop heartbeat |
| **Al-Ghazali** | 1095 | Muraqabah: vigilant self-monitoring | Circuit breaker + SNR gating |
