# FATE Gates Reference

Complete specification of the FATE (Fairness, Accuracy, Truthfulness, Excellence) gate system.

## Table of Contents

1. [Overview](#overview)
2. [The Four Gates](#the-four-gates)
3. [Gate Processing](#gate-processing)
4. [Thresholds](#thresholds)
5. [Failure Handling](#failure-handling)
6. [Customization](#customization)
7. [Monitoring](#monitoring)

---

## Overview

FATE gates are the ethical validation layer that every output must pass through before reaching the user. They embody the principle that **excellence is the minimum, not the goal**.

```
┌─────────────────────────────────────────────────────────────┐
│                    FATE GATE SYSTEM                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   Input (Agent Output)                                      │
│          │                                                  │
│          ▼                                                  │
│   ┌─────────────┐                                          │
│   │   Ihsān     │ ═══ Excellence (Quality Score)           │
│   │   إحسان     │     Threshold: ≥ 0.95                    │
│   └──────┬──────┘                                          │
│          │ PASS                                             │
│          ▼                                                  │
│   ┌─────────────┐                                          │
│   │    Adl      │ ═══ Justice (Fairness/Equality)          │
│   │    عدل      │     Threshold: ≤ 0.35 (Gini)             │
│   └──────┬──────┘                                          │
│          │ PASS                                             │
│          ▼                                                  │
│   ┌─────────────┐                                          │
│   │    Harm     │ ═══ Harm Prevention                      │
│   │    ضرر      │     Threshold: ≤ 0.30                    │
│   └──────┬──────┘                                          │
│          │ PASS                                             │
│          ▼                                                  │
│   ┌─────────────┐                                          │
│   │ Confidence  │ ═══ Certainty/Reliability                │
│   │    ثقة      │     Threshold: ≥ 0.80                    │
│   └──────┬──────┘                                          │
│          │ PASS                                             │
│          ▼                                                  │
│   Output (Validated)                                        │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## The Four Gates

### 1. Ihsān (إحسان) — Excellence Gate

**Purpose:** Ensure output quality meets the highest standards.

**Arabic Meaning:** "Excellence" or "doing beautiful"

**Threshold:** ≥ 0.95 (default)

**Calculation:**
```
ihsan_score = (
    accuracy × 0.30 +
    completeness × 0.25 +
    coherence × 0.20 +
    relevance × 0.15 +
    clarity × 0.10
)
```

**Components:**

| Component | Weight | Description |
|-----------|--------|-------------|
| Accuracy | 30% | Factual correctness |
| Completeness | 25% | Addresses all aspects |
| Coherence | 20% | Logical flow |
| Relevance | 15% | On-topic response |
| Clarity | 10% | Clear expression |

**Failure Response:**
- Request improvement from agent
- Escalate to Guardian for review
- Present partial result with warning

---

### 2. Adl (عدل) — Justice Gate

**Purpose:** Ensure fairness and equitable treatment.

**Arabic Meaning:** "Justice" or "fairness"

**Threshold:** ≤ 0.35 (Gini coefficient)

**Calculation:**
```
adl_score = gini_coefficient(resource_distribution)

# Where Gini coefficient:
# 0.00 = Perfect equality
# 1.00 = Perfect inequality
```

**Applies To:**

| Context | Measurement |
|---------|-------------|
| Multi-agent tasks | Workload distribution |
| Resource allocation | Compute distribution |
| Information access | Data availability |
| Recommendations | Bias in suggestions |

**Failure Response:**
- Rebalance resources
- Flag potential bias
- Request diverse perspectives

---

### 3. Harm (ضرر) — Harm Prevention Gate

**Purpose:** Prevent harmful outputs.

**Arabic Meaning:** "Harm" or "damage"

**Threshold:** ≤ 0.30 (harm score)

**Calculation:**
```
harm_score = max(
    physical_harm_potential,
    psychological_harm_potential,
    financial_harm_potential,
    privacy_violation_potential,
    security_risk_potential
)
```

**Categories:**

| Category | Examples |
|----------|----------|
| Physical | Instructions that could cause injury |
| Psychological | Manipulation, harassment |
| Financial | Scams, bad financial advice |
| Privacy | PII exposure, surveillance |
| Security | Vulnerabilities, exploits |

**Failure Response:**
- **Block** output entirely
- Alert Guardian
- Log for audit
- Never partial release

---

### 4. Confidence (ثقة) — Certainty Gate

**Purpose:** Ensure reliability of information.

**Arabic Meaning:** "Confidence" or "trust"

**Threshold:** ≥ 0.80 (confidence score)

**Calculation:**
```
confidence_score = (
    model_confidence × 0.40 +
    source_reliability × 0.30 +
    consistency × 0.20 +
    verification_level × 0.10
)
```

**Components:**

| Component | Weight | Description |
|-----------|--------|-------------|
| Model Confidence | 40% | LLM's self-reported confidence |
| Source Reliability | 30% | Quality of sources cited |
| Consistency | 20% | Agreement across multiple checks |
| Verification | 10% | External verification possible |

**Failure Response:**
- Add uncertainty disclaimer
- Request additional research
- Present as "preliminary"

---

## Gate Processing

### Sequential Processing

Gates are processed in order. A failure at any gate stops processing:

```
Input → Ihsān → Adl → Harm → Confidence → Output
           │      │     │        │
           ▼      ▼     ▼        ▼
         FAIL?  FAIL? FAIL?   FAIL?
           │      │     │        │
           ▼      ▼     ▼        ▼
        Handle  Handle Handle  Handle
```

### Processing Timeline

```
t=0ms     Input received
t=10ms    Ihsān evaluation begins
t=50ms    Ihsān complete → Adl begins
t=80ms    Adl complete → Harm begins
t=120ms   Harm complete → Confidence begins
t=150ms   Confidence complete → Output released
```

### Parallel Evaluation (Optimization)

For performance, gates can evaluate in parallel with early termination:

```
Input → [Ihsān, Adl, Harm, Confidence] → Aggregate → Output
                                              │
                                    Any FAIL? → Block
```

---

## Thresholds

### Default Thresholds

| Gate | Threshold | Direction | Meaning |
|------|-----------|-----------|---------|
| Ihsān | 0.95 | ≥ | Score must be at least 0.95 |
| Adl | 0.35 | ≤ | Gini must be at most 0.35 |
| Harm | 0.30 | ≤ | Harm score must be at most 0.30 |
| Confidence | 0.80 | ≥ | Confidence must be at least 0.80 |

### Threshold Profiles

**Conservative (High Safety):**
```yaml
fate_gates:
  ihsan_threshold: 0.98
  adl_gini_max: 0.25
  harm_threshold: 0.20
  confidence_min: 0.90
```

**Balanced (Default):**
```yaml
fate_gates:
  ihsan_threshold: 0.95
  adl_gini_max: 0.35
  harm_threshold: 0.30
  confidence_min: 0.80
```

**Permissive (Higher Risk Tolerance):**
```yaml
fate_gates:
  ihsan_threshold: 0.90
  adl_gini_max: 0.45
  harm_threshold: 0.40
  confidence_min: 0.70
```

---

## Failure Handling

### By Gate

| Gate | Failure Action |
|------|----------------|
| Ihsān | Retry with improvement prompt |
| Adl | Rebalance and retry |
| Harm | **Block immediately** |
| Confidence | Add disclaimer, continue |

### Escalation Path

```
Gate Failure
     │
     ▼
Automatic Retry (max 2)
     │
     ▼ Still failing
Guardian Review
     │
     ▼ Guardian approves bypass?
Human Confirmation Required
     │
     ▼ Human approves?
Output with Audit Log
```

### Audit Trail

Every gate failure is logged:

```json
{
  "timestamp": "2026-02-05T12:34:56Z",
  "gate": "harm",
  "score": 0.42,
  "threshold": 0.30,
  "action": "blocked",
  "input_hash": "abc123...",
  "agent": "developer",
  "reason": "Potential security vulnerability disclosure"
}
```

---

## Customization

### Per-Task Thresholds

```yaml
# In task definition
task:
  name: "Security Audit"
  fate_overrides:
    harm_threshold: 0.10  # Stricter for security tasks
```

### Context-Aware Adjustment

```rust
fn adjust_thresholds(context: &Context) -> FATEThresholds {
    let mut thresholds = FATEThresholds::default();

    // Stricter for public-facing outputs
    if context.is_public {
        thresholds.harm_threshold = 0.20;
    }

    // More lenient for internal analysis
    if context.is_internal_only {
        thresholds.confidence_min = 0.70;
    }

    thresholds
}
```

### Bypass (Dangerous)

Bypassing gates requires:
1. Guardian explicit approval
2. Human confirmation
3. Audit log entry
4. Time-limited validity

```yaml
# This is logged at CRITICAL level
bypass:
  gate: "confidence"
  reason: "Experimental research output"
  approved_by: "guardian"
  confirmed_by: "human"
  expires: "2026-02-05T14:00:00Z"
```

---

## Monitoring

### Gate Status Command

```bash
/guardian status
```

Output:
```
╔════════════════════════════════════════════════════════════╗
║                    FATE Gates Status                        ║
╠════════════════════════════════════════════════════════════╣
║                                                            ║
║   ✓ Ihsān (Excellence)                                     ║
║     Current: 0.97  │  Threshold: 0.95  │  PASSING          ║
║     ████████████████████░░░░                               ║
║                                                            ║
║   ✓ Adl (Justice)                                          ║
║     Current: 0.28  │  Threshold: 0.35  │  PASSING          ║
║     ██████████░░░░░░░░░░░░░░                               ║
║                                                            ║
║   ✓ Harm (Prevention)                                      ║
║     Current: 0.12  │  Threshold: 0.30  │  PASSING          ║
║     ████░░░░░░░░░░░░░░░░░░░░                               ║
║                                                            ║
║   ✓ Confidence (Certainty)                                 ║
║     Current: 0.91  │  Threshold: 0.80  │  PASSING          ║
║     ██████████████████████░░                               ║
║                                                            ║
╠════════════════════════════════════════════════════════════╣
║   Overall: ALL GATES PASSING                               ║
║   Last Check: 2026-02-05 12:34:56 UTC                      ║
╚════════════════════════════════════════════════════════════╝
```

### Metrics

```
# Prometheus-style metrics

bizra_fate_gate_passes_total{gate="ihsan"} 1542
bizra_fate_gate_failures_total{gate="ihsan"} 23
bizra_fate_gate_score{gate="ihsan"} 0.97

bizra_fate_gate_passes_total{gate="harm"} 1560
bizra_fate_gate_failures_total{gate="harm"} 5
bizra_fate_gate_score{gate="harm"} 0.12
```

---

## Philosophy

### Why FATE?

The FATE gate system ensures that:

1. **Excellence is enforced** — No mediocre outputs
2. **Fairness is measured** — No hidden biases
3. **Harm is prevented** — Safety is non-negotiable
4. **Uncertainty is acknowledged** — Honesty about limitations

### The Arabic Connection

Each gate name comes from Arabic, reflecting BIZRA's cultural roots:

| Gate | Arabic | Deeper Meaning |
|------|--------|----------------|
| Ihsān | إحسان | "Worship as if you see God; if you cannot, know He sees you" |
| Adl | عدل | "Justice even against yourself" |
| Harm | ضرر | "No harm and no reciprocal harm" (Islamic legal maxim) |
| Confidence | ثقة | "Trust but verify" |

---

**FATE gates: Where ethics meets engineering.** ⚖️
