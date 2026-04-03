---
allowed-tools: Bash(python*:*), Read, Grep, Glob
description: Route tasks by Signal-to-Noise tier (T0-T4)
argument-hint: [task-to-route]
---

# SNR - Signal-to-Noise Tier Routing

## Overview

SNR routing classifies tasks by their **Signal-to-Noise Ratio** and routes them to appropriate execution paths. Higher SNR = clearer intent = faster execution. Lower SNR = more clarification needed.

## SNR Tier Definitions

| Tier | SNR Range | Classification | Execution Path |
|------|-----------|----------------|----------------|
| **T0** | >= 0.95 | Elite | Full SAPE + receipts + all validation |
| **T1** | 0.85-0.95 | Standard | Core validation, standard receipts |
| **T2** | 0.70-0.85 | Assisted | Guided workflow, enhanced context |
| **T3** | 0.50-0.70 | Exploratory | Research mode, iterative refinement |
| **T4** | < 0.50 | Clarification | Ask user for more information |

## Current System Status

- SNR Tracker: !`ls -lh bizra_kernel/snr_tracker.py 2>/dev/null || echo "Not found"`
- Task Complexity Analyzer: !`grep -l "complexity" src/*.rs 2>/dev/null | head -1 || echo "Not found"`
- Model Router: !`ls -lh src/model_router.rs 2>/dev/null || echo "Not found"`

## Your Task

### Phase 1: Signal Analysis

Analyze the task for **signal clarity**:

**Signal Indicators** (increase SNR):
- Specific, measurable requirements
- Clear success criteria
- Defined scope/boundaries
- Technical precision
- Referenced files/functions

**Noise Indicators** (decrease SNR):
- Vague language ("improve", "fix", "make better")
- Missing context
- Ambiguous scope
- Undefined terms
- Conflicting requirements

### Phase 2: SNR Calculation

```python
def calculate_snr(task):
    """Calculate Signal-to-Noise Ratio for task"""
    signal_factors = {
        "specificity": 0.0,      # How specific is the request?
        "measurability": 0.0,    # Can success be measured?
        "scope_clarity": 0.0,    # Is scope well-defined?
        "context_provided": 0.0, # Is sufficient context given?
        "technical_precision": 0.0  # Technical terms accurate?
    }

    noise_factors = {
        "ambiguity": 0.0,        # Vague language present?
        "missing_info": 0.0,     # Information gaps?
        "conflicting_req": 0.0,  # Contradictory requirements?
        "scope_creep_risk": 0.0  # Undefined boundaries?
    }

    # Calculate scores (0-1 for each)
    signal_score = sum(signal_factors.values()) / len(signal_factors)
    noise_score = sum(noise_factors.values()) / len(noise_factors)

    # SNR = signal / (signal + noise)
    snr = signal_score / (signal_score + noise_score + 0.001)

    return snr, get_tier(snr)
```

### Phase 3: Tier Assignment

Based on SNR, assign appropriate tier:

#### T0 Elite (SNR >= 0.95)

**Characteristics**:
- Crystal clear requirements
- Specific files/functions named
- Success criteria defined
- No ambiguity

**Execution**:
- Full SAPE 9-probe validation
- Complete receipt generation
- SAT consensus (3/5 required)
- Ihsan gate enforcement
- Neo4j graph evidence

**Example**:
> "Add a new field `created_at: DateTime<Utc>` to the `Receipt` struct in `src/receipts.rs`, update the `emit()` function to populate it, and add a test in `tests/receipts_test.rs`."

#### T1 Standard (SNR 0.85-0.95)

**Characteristics**:
- Clear intent
- Minor details to infer
- Standard patterns apply

**Execution**:
- Core SAPE probes (threat_scan, safety, correctness)
- Standard receipts
- SAT validation (advisory)
- Ihsan gate enforcement

**Example**:
> "Add timestamp tracking to receipts."

#### T2 Assisted (SNR 0.70-0.85)

**Characteristics**:
- Good intent but gaps
- Needs some assumptions
- Multiple valid approaches

**Execution**:
- Guided workflow
- Present options to user
- Enhanced context gathering
- Partial SAPE validation

**Example**:
> "Make the receipt system more robust."

#### T3 Exploratory (SNR 0.50-0.70)

**Characteristics**:
- Vague requirements
- Research needed
- Discovery phase

**Execution**:
- Research mode
- Iterative refinement
- Multiple proposals
- User feedback loops

**Example**:
> "Improve the system."

#### T4 Clarification (SNR < 0.50)

**Characteristics**:
- Insufficient information
- Cannot proceed safely
- High risk of wrong direction

**Execution**:
- Ask clarifying questions
- Do NOT proceed with assumptions
- Gather requirements first

**Example**:
> "Fix it."

### Phase 4: Route Execution

Based on tier, execute appropriate workflow:

```
T0 Elite:
    ├── Full SAPE validation
    ├── Complete receipt generation
    ├── SAT consensus
    └── Elite execution path

T1 Standard:
    ├── Core validation
    ├── Standard receipts
    └── Normal execution

T2 Assisted:
    ├── Present options
    ├── Gather context
    └── Guided execution

T3 Exploratory:
    ├── Research phase
    ├── Multiple proposals
    └── Iterative refinement

T4 Clarification:
    └── Ask questions first
```

## SNR Analysis Template

### Task: [User's Task]

---

#### Signal Analysis

| Factor | Score (0-1) | Evidence |
|--------|-------------|----------|
| Specificity | 0.X | [evidence] |
| Measurability | 0.X | [evidence] |
| Scope Clarity | 0.X | [evidence] |
| Context Provided | 0.X | [evidence] |
| Technical Precision | 0.X | [evidence] |
| **Signal Total** | **X.X/5** | |

#### Noise Analysis

| Factor | Score (0-1) | Evidence |
|--------|-------------|----------|
| Ambiguity | 0.X | [evidence] |
| Missing Info | 0.X | [evidence] |
| Conflicting Requirements | 0.X | [evidence] |
| Scope Creep Risk | 0.X | [evidence] |
| **Noise Total** | **X.X/4** | |

#### SNR Calculation

```
Signal Score: X.XX
Noise Score: X.XX
SNR = Signal / (Signal + Noise) = X.XX
```

#### Tier Assignment

**Assigned Tier**: T[0-4]
**Confidence**: High/Medium/Low

---

#### Execution Path

Based on T[X] tier:
- [ ] [Appropriate action 1]
- [ ] [Appropriate action 2]
- [ ] [Appropriate action 3]

---

## Validation Checks

### SNR Calculation Validity

- [ ] All signal factors evaluated
- [ ] All noise factors evaluated
- [ ] Evidence provided for scores
- [ ] Tier assignment justified

### Execution Path Validity

- [ ] Path matches tier
- [ ] No over-execution (T3 getting T0 treatment)
- [ ] No under-execution (T0 getting T4 treatment)

## Evidence Generation

Generate SNR routing receipt:

```json
{
  "receipt_id": "snr-$(date +%s)",
  "timestamp": "$(date -Iseconds)",
  "task_summary": "[task description]",
  "snr_analysis": {
    "signal_factors": {
      "specificity": 0.0,
      "measurability": 0.0,
      "scope_clarity": 0.0,
      "context_provided": 0.0,
      "technical_precision": 0.0
    },
    "noise_factors": {
      "ambiguity": 0.0,
      "missing_info": 0.0,
      "conflicting_req": 0.0,
      "scope_creep_risk": 0.0
    },
    "signal_score": 0.0,
    "noise_score": 0.0,
    "snr": 0.0
  },
  "tier": "T0|T1|T2|T3|T4",
  "execution_path": [],
  "integrity_hash": ""
}
```

## Report Format

```
## SNR Routing Report

**Task**: [task description]
**Timestamp**: [ISO timestamp]

### SNR Analysis

**Signal Score**: X.XX/5.0
- Specificity: X.X
- Measurability: X.X
- Scope Clarity: X.X
- Context: X.X
- Technical Precision: X.X

**Noise Score**: X.XX/4.0
- Ambiguity: X.X
- Missing Info: X.X
- Conflicts: X.X
- Scope Creep: X.X

### Calculation

```
SNR = X.XX / (X.XX + X.XX) = X.XX
```

### Tier Assignment

**Tier**: T[0-4] [Elite|Standard|Assisted|Exploratory|Clarification]
**Confidence**: [High|Medium|Low]

### Execution Path

Based on T[X]:
1. [Action 1]
2. [Action 2]
3. [Action 3]

### Receipt
- ID: snr-[timestamp]
- Location: docs/evidence/receipts/
```

---

**SNR Philosophy**: "Match execution intensity to task clarity. Elite tasks deserve elite treatment. Vague tasks need clarification, not assumptions."
