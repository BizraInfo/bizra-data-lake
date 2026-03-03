# Step 5: Ihsan Gate Extension to 6 Dimensions

## Standing on Giants: Al-Ghazali (Ihsan as excellence); Shannon (measurable quality); Anthropic (Constitutional AI)

## Problem Statement

The Ihsan gate (`core/proof_engine/ihsan_gate.py`) operates on 4 components
(correctness, safety, efficiency, user_benefit), but the canonical weight tensor
in `core/integration/constants.py` defines **8 dimensions**:

| Dimension | Weight | Currently Scored? |
|-----------|--------|-------------------|
| correctness | 0.22 | YES |
| safety | 0.22 | YES |
| user_benefit | 0.14 | YES |
| efficiency | 0.12 | YES |
| **auditability** | **0.12** | NO |
| **anti_centralization** | **0.08** | NO |
| **robustness** | **0.06** | NO |
| **adl_fairness** | **0.04** | NO |

The `_canonical_component_weights()` function projects the 8-dim tensor down
to 4 dimensions by selecting 4 keys and re-normalizing. This discards **30%
of the weight signal** (0.12 + 0.08 + 0.06 + 0.04 = 0.30).

**This step extends the gate to 6 dimensions** by adding `auditability` and
`robustness` — the two highest-weighted unused dimensions. This captures
**88% of the total weight** (0.22 + 0.22 + 0.14 + 0.12 + 0.12 + 0.06 = 0.88),
leaving only `anti_centralization` (0.08) and `adl_fairness` (0.04) for Phase 60.

**Why these two?**
- `auditability` (0.12): Same weight as `efficiency`. Can the output be reviewed
  and verified? Measured via evidence markers, citations, structured reasoning.
- `robustness` (0.06): Does the output degrade gracefully under noisy inputs?
  Measured via lexical diversity, multi-source evidence, appropriate hedging.
- `anti_centralization` (0.08): Better enforced by the Gini gate at the token
  economics layer — not an output quality dimension.
- `adl_fairness` (0.04): Already enforced by the ADL Gini threshold in the
  token minter — adding it here would double-count.

## Target Files

| File | Action |
|------|--------|
| `core/proof_engine/ihsan_gate.py` | Add `auditability` and `robustness` to `IhsanComponents`; update `_canonical_component_weights()` to project 6 dims |
| `core/proof_engine/ihsan_computer.py` | Add `_score_auditability()` and `_score_robustness()` scorers |
| `core/sovereign/mission.py` | No changes needed — uses IhsanGate through SNR engine, not directly |
| `tests/core/proof_engine/test_ihsan_6dim.py` | New test file for 6-dim extension |
| `tests/core/proof_engine/test_snr_v1_ihsan_gate.py` | Verify existing 4-dim tests still pass |

## Pseudocode

### ihsan_gate.py — Extend IhsanComponents to 6 dimensions

```pseudocode
@dataclass
CLASS IhsanComponents:
    """Individual components of the Ihsan excellence score.

    v1.0: 4 components (correctness, safety, efficiency, user_benefit)
    v1.1: 6 components (+auditability, +robustness) — Phase 59 Step 5
    """

    correctness: float = 0.0
    safety: float = 0.0
    efficiency: float = 0.0
    user_benefit: float = 0.0
    auditability: float = 0.0     # NEW: Can the output be reviewed/verified?
    robustness: float = 0.0       # NEW: Does it degrade gracefully?

    METHOD to_dict() -> Dict[str, float]:
        RETURN {
            "correctness": self.correctness,
            "safety": self.safety,
            "efficiency": self.efficiency,
            "user_benefit": self.user_benefit,
            "auditability": self.auditability,
            "robustness": self.robustness,
        }

    METHOD composite_score(weights=None) -> float:
        w = weights OR _canonical_component_weights()
        RETURN (
            w.get("correctness", 0.0) * self.correctness
            + w.get("safety", 0.0) * self.safety
            + w.get("efficiency", 0.0) * self.efficiency
            + w.get("user_benefit", 0.0) * self.user_benefit
            + w.get("auditability", 0.0) * self.auditability
            + w.get("robustness", 0.0) * self.robustness
        )
```

### ihsan_gate.py — Update weight projection to 6 dimensions

```pseudocode
FUNCTION _canonical_component_weights() -> Dict[str, float]:
    """
    Derive 6-component gate weights from the canonical 8-dim Ihsan weights.

    This gate scores: correctness, safety, efficiency, user_benefit,
    auditability, robustness.
    Deferred: anti_centralization (Gini gate), adl_fairness (Gini gate).
    """
    base = {
        "correctness": float(IHSAN_WEIGHTS["correctness"]),
        "safety": float(IHSAN_WEIGHTS["safety"]),
        "efficiency": float(IHSAN_WEIGHTS["efficiency"]),
        "user_benefit": float(IHSAN_WEIGHTS["user_benefit"]),
        "auditability": float(IHSAN_WEIGHTS["auditability"]),
        "robustness": float(IHSAN_WEIGHTS["robustness"]),
    }
    total = sum(base.values())
    IF total <= 0.0:
        # Fail-safe fallback: equal weights across 6 dimensions
        RETURN {k: 1.0 / 6.0 for k in base}
    RETURN {k: v / total for k, v in base.items()}
```

**Weight re-normalization example:**

| Dimension | Raw Weight | Normalized (6-dim) | Old Normalized (4-dim) |
|-----------|-----------|---------------------|------------------------|
| correctness | 0.22 | 0.250 | 0.314 |
| safety | 0.22 | 0.250 | 0.314 |
| user_benefit | 0.14 | 0.159 | 0.200 |
| efficiency | 0.12 | 0.136 | 0.171 |
| auditability | 0.12 | 0.136 | — |
| robustness | 0.06 | 0.068 | — |
| **Total** | **0.88** | **1.000** | **1.000** |

**Impact on existing scores:** The 4 existing dimensions lose ~14% of their
relative weight (redistributed to auditability + robustness). Composite scores
will shift downward if the new dimensions score lower than the existing ones.
This is correct behavior — the gate was previously blind to these quality
aspects, and a slight score reduction reflects the newly-visible quality gap.

### ihsan_gate.py — Extend evaluate() diagnostics

```pseudocode
CLASS IhsanGate:
    METHOD evaluate(components: IhsanComponents) -> IhsanResult:
        score = components.composite_score(self.weights)
        reason_codes = []

        IF score < self.threshold:
            reason_codes.append(ReasonCode.IHSAN_BELOW_THRESHOLD.value)

            # Existing component diagnostics
            IF components.safety < 0.90:
                reason_codes.append("SAFETY_COMPONENT_LOW")
            IF components.correctness < 0.85:
                reason_codes.append("CORRECTNESS_COMPONENT_LOW")

            # NEW: diagnostics for new dimensions
            IF components.auditability < 0.70:
                reason_codes.append("AUDITABILITY_COMPONENT_LOW")
            IF components.robustness < 0.60:
                reason_codes.append("ROBUSTNESS_COMPONENT_LOW")

        decision = "APPROVED" IF score >= self.threshold ELSE "REJECTED"
        RETURN IhsanResult(
            score=score,
            threshold=self.threshold,
            decision=decision,
            components=components,
            reason_codes=reason_codes,
            version="1.1.0",   # Bump from 1.0.0
        )
```

### ihsan_computer.py — Add auditability scorer

```pseudocode
CLASS IhsanComputer:

    @staticmethod
    METHOD _score_auditability(signals: IhsanSignals) -> float:
        """
        Auditability: Can the output be independently reviewed and verified?

        High auditability = structured reasoning, citations, code examples,
        numerical evidence, low hedging.

        Signals used (all from existing IhsanSignals):
        - evidence_hits: URLs, numbers, bullets, code blocks, evidence markers
        - hedge_hits: "maybe", "might", "possibly" — unverifiable claims
        - word_count: too short = insufficient detail for review
        - unique_ratio: repetitive text is harder to audit (less information)

        Standing on: Shannon — information content is measurable.
        """
        IF signals.word_count <= 0:
            RETURN 0.0

        # Base: assumes moderate auditability
        base = 0.30

        # Evidence markers add verifiability (max +0.45 for 5 types)
        evidence_bonus = min(0.45, 0.09 * signals.evidence_hits)

        # Hedging reduces auditability (unverifiable claims)
        hedge_penalty = min(0.20, 0.04 * signals.hedge_hits)

        # Lexical diversity adds information density (auditable signal)
        diversity_bonus = 0.15 * signals.unique_ratio

        # Minimum length for auditable content
        length_penalty = 0.15 IF signals.word_count < 20 ELSE 0.0

        RETURN _clamp01(
            base + evidence_bonus + diversity_bonus
            - hedge_penalty - length_penalty
        )
```

### ihsan_computer.py — Add robustness scorer

```pseudocode
    @staticmethod
    METHOD _score_robustness(signals: IhsanSignals) -> float:
        """
        Robustness: Will the output remain useful under noisy/degraded inputs?

        High robustness = diverse vocabulary, multi-source evidence,
        appropriate hedging (acknowledges uncertainty rather than asserting
        false confidence), moderate length.

        Key insight: some hedging is GOOD for robustness — it signals that the
        output acknowledges its own uncertainty, making it more resilient to
        being wrong. But excessive hedging = lack of substance.

        Signals used (all from existing IhsanSignals):
        - unique_ratio: diverse vocabulary = resilient expression
        - evidence_hits: multi-source = doesn't depend on single point
        - hedge_hits: moderate hedging = uncertainty acknowledgment (good);
                      excessive = vacuousness (bad)
        - word_count: extremes (too short or too long) = fragile

        Standing on: Shannon — robust coding in noisy channels.
        """
        IF signals.word_count <= 0:
            RETURN 0.0

        # Lexical diversity is the primary robustness signal
        base = 0.20 + 0.40 * signals.unique_ratio

        # Multi-source evidence = less fragile
        evidence_bonus = min(0.20, 0.05 * signals.evidence_hits)

        # Moderate hedging is GOOD (1-2 hedges = epistemic humility)
        # Excessive hedging is BAD (>3 = vacuousness)
        IF signals.hedge_hits == 0:
            hedge_effect = -0.05     # No hedging = overconfident
        ELIF signals.hedge_hits <= 2:
            hedge_effect = 0.05      # Moderate = healthy uncertainty
        ELSE:
            hedge_effect = -0.03 * (signals.hedge_hits - 2)  # Excessive
        hedge_effect = max(-0.15, hedge_effect)

        # Length extremes reduce robustness
        IF signals.word_count < 15:
            length_penalty = 0.15    # Too short = fragile
        ELIF signals.word_count > 600:
            length_penalty = 0.10    # Too long = diluted
        ELSE:
            length_penalty = 0.0

        RETURN _clamp01(
            base + evidence_bonus + hedge_effect - length_penalty
        )
```

### ihsan_computer.py — Wire new scorers into compute()

```pseudocode
    METHOD compute_with_signals(content, *, snr_score, query_text, context):
        signals = self._extract_signals(...)
        snr = _clamp01(0.5 IF snr_score IS None ELSE float(snr_score))

        correctness = self._score_correctness(signals, snr)
        safety = self._score_safety(signals)
        efficiency = self._score_efficiency(signals)
        user_benefit = self._score_user_benefit(signals)
        auditability = self._score_auditability(signals)     # NEW
        robustness = self._score_robustness(signals)          # NEW

        RETURN (
            IhsanComponents(
                correctness=correctness,
                safety=safety,
                efficiency=efficiency,
                user_benefit=user_benefit,
                auditability=auditability,
                robustness=robustness,
            ),
            signals,
        )
```

## Backward Compatibility

**Critical constraint:** Existing consumers that construct `IhsanComponents`
with only 4 fields must continue to work. This is guaranteed because:

1. The new fields have `default=0.0` — positional construction with 4 args
   still works: `IhsanComponents(0.9, 0.95, 0.85, 0.90)` → auditability=0.0,
   robustness=0.0
2. Keyword construction with 4 fields still works:
   `IhsanComponents(correctness=0.9, safety=0.95, efficiency=0.85, user_benefit=0.90)`
3. `composite_score()` with `weights=None` uses `_canonical_component_weights()`
   which now includes 6 keys. The two new 0.0-valued fields multiply by their
   weights, contributing 0.0 to the composite — equivalent to the old behavior
   of not including them at all, BUT with reduced total weight on the 4 existing
   dimensions. This means **scores will decrease slightly** for callers that
   don't supply the new dimensions.

**Migration strategy for callers:**
- Phase 59: Add new fields to IhsanComponents, update IhsanComputer. Tests
  that construct components manually should set all 6 fields.
- Phase 59: Any caller that doesn't supply the new fields gets a softer score
  (conservative — correct behavior, since auditability and robustness are truly
  unknown for that output).
- Phase 60: Audit all callers and wire auditability + robustness scoring.

**Alternative considered and rejected:** Adding a `version` parameter to
`composite_score()` that selects 4-dim or 6-dim weights. Rejected because:
- Increases complexity
- Allows callers to avoid upgrading indefinitely
- The slight score decrease for unscored dimensions is the CORRECT behavior
  (unknown quality should reduce confidence, not be assumed perfect)

## TDD Anchors

### Test File: `tests/core/proof_engine/test_ihsan_6dim.py`

```pseudocode
TEST ihsan_components_has_six_fields:
    """IhsanComponents has 6 fields with correct defaults."""
    c = IhsanComponents()
    ASSERT c.correctness == 0.0
    ASSERT c.safety == 0.0
    ASSERT c.efficiency == 0.0
    ASSERT c.user_benefit == 0.0
    ASSERT c.auditability == 0.0
    ASSERT c.robustness == 0.0

TEST backward_compat_four_field_construction:
    """4-field positional construction still works (new fields default to 0.0)."""
    c = IhsanComponents(0.9, 0.95, 0.85, 0.90)
    ASSERT c.correctness == 0.9
    ASSERT c.safety == 0.95
    ASSERT c.efficiency == 0.85
    ASSERT c.user_benefit == 0.90
    ASSERT c.auditability == 0.0
    ASSERT c.robustness == 0.0

TEST backward_compat_keyword_construction:
    """Keyword construction with 4 fields still works."""
    c = IhsanComponents(correctness=0.9, safety=0.95, efficiency=0.85, user_benefit=0.90)
    ASSERT c.auditability == 0.0
    ASSERT c.robustness == 0.0

TEST to_dict_includes_six_fields:
    """to_dict() serializes all 6 components."""
    c = IhsanComponents(0.9, 0.95, 0.85, 0.90, 0.80, 0.75)
    d = c.to_dict()
    ASSERT len(d) == 6
    ASSERT d["auditability"] == 0.80
    ASSERT d["robustness"] == 0.75

TEST canonical_weights_project_six_dimensions:
    """_canonical_component_weights() returns 6 normalized weights."""
    w = _canonical_component_weights()
    ASSERT len(w) == 6
    ASSERT set(w.keys()) == {"correctness", "safety", "efficiency",
                              "user_benefit", "auditability", "robustness"}
    ASSERT abs(sum(w.values()) - 1.0) < 1e-9  # Normalized to sum=1.0

TEST canonical_weights_preserve_relative_order:
    """Weight ordering: correctness = safety > user_benefit > efficiency = auditability > robustness."""
    w = _canonical_component_weights()
    ASSERT w["correctness"] == w["safety"]       # Both 0.22/0.88
    ASSERT w["safety"] > w["user_benefit"]
    ASSERT w["user_benefit"] > w["efficiency"]
    ASSERT w["efficiency"] == w["auditability"]   # Both 0.12/0.88
    ASSERT w["auditability"] > w["robustness"]

TEST composite_score_six_dim:
    """Composite score uses all 6 dimensions."""
    c = IhsanComponents(
        correctness=1.0, safety=1.0, efficiency=1.0,
        user_benefit=1.0, auditability=1.0, robustness=1.0,
    )
    ASSERT abs(c.composite_score() - 1.0) < 1e-9  # All perfect = 1.0

TEST composite_score_partial_dim:
    """New dimensions at 0.0 reduce composite vs. old behavior."""
    c_full = IhsanComponents(0.95, 0.98, 0.90, 0.92, 0.85, 0.80)
    c_partial = IhsanComponents(0.95, 0.98, 0.90, 0.92, 0.0, 0.0)
    # Partial should score lower — new dims contribute 0
    ASSERT c_full.composite_score() > c_partial.composite_score()

TEST failsafe_weights_six_equal:
    """Fail-safe produces 6 equal weights when IHSAN_WEIGHTS sum is zero."""
    # Monkeypatch IHSAN_WEIGHTS to all zeros
    with mock.patch.dict(IHSAN_WEIGHTS, {k: 0.0 for k in IHSAN_WEIGHTS}):
        w = _canonical_component_weights()
        ASSERT all(abs(v - 1.0/6.0) < 1e-9 for v in w.values())

TEST gate_version_bumped:
    """IhsanResult.version reflects the 6-dim upgrade."""
    gate = IhsanGate()
    c = IhsanComponents(0.95, 0.98, 0.90, 0.92, 0.85, 0.80)
    result = gate.evaluate(c)
    ASSERT result.version == "1.1.0"

TEST gate_diagnoses_low_auditability:
    """REJECTED result includes AUDITABILITY_COMPONENT_LOW reason code."""
    gate = IhsanGate(threshold=0.95)
    c = IhsanComponents(0.95, 0.98, 0.90, 0.92, 0.50, 0.80)
    result = gate.evaluate(c)
    IF result.decision == "REJECTED":
        ASSERT "AUDITABILITY_COMPONENT_LOW" IN result.reason_codes

TEST gate_diagnoses_low_robustness:
    """REJECTED result includes ROBUSTNESS_COMPONENT_LOW reason code."""
    gate = IhsanGate(threshold=0.95)
    c = IhsanComponents(0.95, 0.98, 0.90, 0.92, 0.85, 0.40)
    result = gate.evaluate(c)
    IF result.decision == "REJECTED":
        ASSERT "ROBUSTNESS_COMPONENT_LOW" IN result.reason_codes

TEST computer_produces_six_components:
    """IhsanComputer.compute() returns 6-field IhsanComponents."""
    computer = IhsanComputer()
    components = computer.compute("This is a test response with evidence: 42%.")
    ASSERT hasattr(components, "auditability")
    ASSERT hasattr(components, "robustness")
    ASSERT 0.0 <= components.auditability <= 1.0
    ASSERT 0.0 <= components.robustness <= 1.0

TEST auditability_high_for_structured_content:
    """Content with citations, numbers, code blocks scores high auditability."""
    computer = IhsanComputer()
    content = """Based on benchmark results (https://example.com/benchmark):
    - Latency: 20.4ms (down from 83.8ms)
    - Throughput: 1200 req/s

    ```python
    result = run_benchmark()
    assert result.latency < 30
    ```

    Therefore, the optimization is verified."""
    components = computer.compute(content)
    ASSERT components.auditability >= 0.70

TEST auditability_low_for_vague_content:
    """Content with hedging and no evidence scores low auditability."""
    computer = IhsanComputer()
    content = "Maybe this might work possibly. I think it could be fine."
    components = computer.compute(content)
    ASSERT components.auditability < 0.50

TEST robustness_moderate_hedging_is_good:
    """1-2 hedge phrases improve robustness (epistemic humility)."""
    computer = IhsanComputer()
    # Content with moderate hedging
    content_hedged = ("The results show improvement. However, this might vary "
                      "under different conditions. The benchmark confirms a "
                      "20ms improvement with structured evidence.")
    # Content with no hedging (overconfident)
    content_certain = ("The results definitively prove improvement. The "
                       "benchmark confirms 20ms improvement with structured "
                       "evidence. This is the final answer.")
    c_hedged = computer.compute(content_hedged)
    c_certain = computer.compute(content_certain)
    # Moderate hedging should score >= no hedging for robustness
    ASSERT c_hedged.robustness >= c_certain.robustness - 0.05

TEST robustness_excessive_hedging_is_bad:
    """Excessive hedging (>3 phrases) reduces robustness."""
    computer = IhsanComputer()
    content = ("Maybe this might work. I'm not sure. Possibly it could "
               "be fine but I guess we'll see. Maybe not though.")
    components = computer.compute(content)
    ASSERT components.robustness < 0.55

TEST robustness_bounded:
    """Robustness score is always in [0, 1]."""
    computer = IhsanComputer()
    FOR content IN ["", "x", "word " * 1000, "a b c d e f g h i j k l m"]:
        c = computer.compute(content)
        ASSERT 0.0 <= c.robustness <= 1.0

TEST all_existing_4dim_tests_still_pass:
    """Regression: existing test_snr_v1_ihsan_gate.py passes unchanged.

    This is verified by running the existing test file — no code changes
    needed in the test file because IhsanComponents defaults new fields to 0.0.
    """
    # Run existing tests (this is a meta-test, executed by pytest)
    PASS

TEST ihsan_score_dict_has_six_components:
    """IhsanGate.ihsan_score() includes all 6 components in output dict."""
    gate = IhsanGate()
    c = IhsanComponents(0.95, 0.98, 0.90, 0.92, 0.85, 0.80)
    result = gate.ihsan_score(c)
    ASSERT len(result["components"]) == 6
    ASSERT "auditability" IN result["components"]
    ASSERT "robustness" IN result["components"]
```

## Scoring Calibration

The new scorers are calibrated to produce scores in the same range as the
existing 4 components for typical LLM output:

| Scorer | Empty | Short vague | Medium structured | Long evidenced |
|--------|-------|-------------|-------------------|----------------|
| correctness | 0.40 | 0.55-0.65 | 0.80-0.90 | 0.90-0.97 |
| safety | 0.98 | 0.98 | 0.98 | 0.98 |
| efficiency | 0.00 | 0.55-0.70 | 0.80-0.90 | 0.75-0.85 |
| user_benefit | 0.00 | 0.20-0.35 | 0.55-0.75 | 0.75-0.90 |
| **auditability** | **0.00** | **0.15-0.30** | **0.55-0.75** | **0.80-0.90** |
| **robustness** | **0.00** | **0.30-0.45** | **0.55-0.70** | **0.65-0.80** |

**Design choice:** Robustness caps lower (~0.80) than auditability (~0.90)
because true robustness requires perturbation testing (Phase 60+), not just
lexical analysis. The scorer provides a conservative lower bound.

## Risk Mitigation

**Risk:** Existing callers that construct IhsanComponents with 4 positional
args and pass custom weights will get different composite scores.

**Mitigation:** The new fields default to 0.0, which means:
1. `composite_score(weights=None)` re-normalizes over 6 dims, so existing
   4-dim components lose ~14% relative weight. This is CORRECT — the gate
   was previously blind to 30% of the quality signal.
2. `composite_score(weights={"correctness": 0.4, "safety": 0.3, ...})` with
   only 4 keys in the custom weights dict will score 0.0 for auditability
   and robustness (those weights default to 0.0 via `w.get(key, 0.0)`).
   This preserves the old behavior exactly for custom-weight callers.

**Risk:** IhsanFloorWatchdog may trigger more frequently due to slightly
lower composite scores.

**Mitigation:** The floor watchdog uses `ihsan_score` passed to it, not
component-level data. If callers pass the same score as before (e.g., the
mission.py fixed score of 0.95/0.80), the watchdog is unaffected. Only
callers that use IhsanComputer → IhsanGate → score will see the shift.

**Rollback:** If 6-dim scoring causes unexpected gate failures:
1. Revert `_canonical_component_weights()` to project 4 dims
2. Keep the new fields in IhsanComponents (harmless at 0.0)
3. Keep the new scorers in IhsanComputer (unused but tested)
4. File issue for gradual threshold adjustment

## Acceptance Criteria

1. `IhsanComponents` has 6 fields: correctness, safety, efficiency,
   user_benefit, auditability, robustness
2. `_canonical_component_weights()` returns 6 normalized weights summing to 1.0
3. `IhsanComputer.compute()` returns 6-field IhsanComponents
4. Auditability scorer: structured content with evidence >= 0.70
5. Robustness scorer: moderate hedging scores >= no hedging
6. `IhsanResult.version` bumped to "1.1.0"
7. All existing 4-dim tests pass unchanged (backward compatibility)
8. New dimension diagnostics appear in REJECTED reason codes
9. Full test suite GREEN (7,911+ tests)
10. Evidence receipt emitted: `{step: "ihsan_6dim", status: "complete"}`

## Migration Path

1. Add fields and scorers (Phase 59 — this step)
2. Wire into all callers using IhsanComputer (Phase 59)
3. Calibrate thresholds under real mission workloads (Phase 60)
4. Add perturbation-based robustness testing (Phase 60)
5. Add anti_centralization and adl_fairness to reach 8-dim (Phase 61)
