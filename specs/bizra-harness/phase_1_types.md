# Phase 1 — Harness Types

> Standing on Giants: Hoare (design by contract, 1969) · Liskov (substitution
> principle, 1987) · Al-Ghazali (Ihsan ethics, 1095)

## Overview

Define the shared vocabulary: config, result, verdict, and pillar-level
types. All types are frozen dataclasses — immutable after construction.
Thresholds are imported, never defined here.

## File: `core/harness/types.py`

```pseudocode
IMPORTS:
    from core.integration.constants import (
        UNIFIED_IHSAN_THRESHOLD,    # 0.95
        UNIFIED_SNR_THRESHOLD,      # 0.85
        SNR_THRESHOLD_T0_ELITE,     # 0.98
        SNR_THRESHOLD_T1_HIGH,      # 0.95
    )
    from core.benchmark.guardrails import GuardrailResult
    from core.proof_engine.bench import BenchResult
    from core.proof_engine.evidence_ledger import Receipt

# ── Enums ──────────────────────────────────────────────────────────

ENUM Verdict(str, Enum):
    """Final harness determination."""
    PASS          = "pass"           # All gates met, no regressions
    FAIL          = "fail"           # At least one gate violated
    INCONCLUSIVE  = "inconclusive"   # Could not determine (partial run)

ENUM PillarName(str, Enum):
    """The 8 smoke pillars + 2 harness-specific pillars."""
    RUNTIME_BOOT       = "runtime_boot"
    TOKEN_SYSTEM       = "token_system"
    EVIDENCE_CHAIN     = "evidence_chain"
    SNR_CHECK          = "snr_check"
    SPEARPOINT         = "spearpoint"
    OPPORTUNITY        = "opportunity"
    CLI                = "cli"
    FULL_STACK         = "full_stack"
    # Harness-specific extensions:
    GUARDRAILS         = "guardrails"
    BENCHMARK          = "benchmark"

ENUM RunMode(str, Enum):
    """How the harness executes."""
    SMOKE       = "smoke"        # Fast: boot checks + SNR only (~5s)
    STANDARD    = "standard"     # All 10 pillars, mocked inference (~15s)
    FULL        = "full"         # All pillars + real inference (~120s)
    BENCHMARK   = "benchmark"    # BDL loop focus (variable duration)

# ── Pillar Result ──────────────────────────────────────────────────

@dataclass(frozen=True)
CLASS PillarResult:
    """Outcome of a single pillar evaluation."""
    pillar:       PillarName
    passed:       bool
    duration_ms:  float
    score:        Optional[float]    = None   # 0.0-1.0 if applicable
    error:        Optional[str]      = None   # failure reason
    details:      dict[str, Any]     = field(default_factory=dict)

    PROPERTY is_scored -> bool:
        return self.score is not None

# ── Harness Config ─────────────────────────────────────────────────

@dataclass(frozen=True)
CLASS HarnessConfig:
    """Configuration for a harness run. Immutable."""
    mode:              RunMode         = RunMode.STANDARD
    claim:             str             = "System meets constitutional thresholds"
    scenario_id:       Optional[str]   = None    # from ScenarioLibrary
    # Gate thresholds — imported from constants, overridable for testing
    snr_floor:         float           = UNIFIED_SNR_THRESHOLD       # 0.85
    ihsan_floor:       float           = UNIFIED_IHSAN_THRESHOLD     # 0.95
    regression_tolerance: float        = 0.02    # max allowed drop from baseline
    # Execution limits
    timeout_seconds:   float           = 60.0    # per-pillar timeout
    max_bdl_cycles:    int             = 3       # BDL loop cap (benchmark mode)
    # Baseline
    compare_baseline:  bool            = True    # compare against stored baseline
    update_baseline:   bool            = False   # seal this run as new baseline
    # Output
    output_dir:        Optional[Path]  = None    # report destination
    emit_receipt:      bool            = True    # produce evidence receipt

    METHOD validate(self) -> list[str]:
        """Return list of validation errors (empty = valid)."""
        errors = []
        IF self.snr_floor < 0.0 OR self.snr_floor > 1.0:
            errors.append("snr_floor must be in [0.0, 1.0]")
        IF self.ihsan_floor < 0.0 OR self.ihsan_floor > 1.0:
            errors.append("ihsan_floor must be in [0.0, 1.0]")
        IF self.regression_tolerance < 0.0:
            errors.append("regression_tolerance must be >= 0.0")
        IF self.timeout_seconds <= 0:
            errors.append("timeout_seconds must be > 0")
        RETURN errors

# ── Regression Report ──────────────────────────────────────────────

@dataclass(frozen=True)
CLASS RegressionReport:
    """Comparison of current run against stored baseline."""
    baseline_run_id:     str
    baseline_snr:        float
    current_snr:         float
    snr_delta:           float           # current - baseline (negative = regression)
    baseline_ihsan:      float
    current_ihsan:       float
    ihsan_delta:         float
    regressed_pillars:   list[PillarName]
    tolerance:           float           # from config
    is_regression:       bool            # True if any delta < -tolerance

# ── Harness Result ─────────────────────────────────────────────────

@dataclass(frozen=True)
CLASS HarnessResult:
    """Unified output of a complete harness run."""
    run_id:            str               # UUID4
    timestamp:         datetime           # UTC
    config:            HarnessConfig
    verdict:           Verdict
    # Aggregate scores
    snr_score:         float             # 0.0-1.0 composite
    ihsan_score:       float             # 0.0-1.0 composite
    tier:              str               # TierPolicy label from resolve_tier()
    # Per-pillar breakdown
    pillars:           dict[PillarName, PillarResult]
    total_duration_ms: float
    # Gate results
    guardrail_results: list[GuardrailResult]
    all_gates_passed:  bool
    # Benchmark (optional, only in BENCHMARK mode)
    bench_result:      Optional[BenchResult]     = None
    # Regression (optional, only if compare_baseline=True)
    regression:        Optional[RegressionReport] = None
    # Evidence
    receipt:           Optional[Receipt]          = None

    PROPERTY passed_count -> int:
        return sum(1 for p in self.pillars.values() if p.passed)

    PROPERTY failed_count -> int:
        return sum(1 for p in self.pillars.values() if not p.passed)

    PROPERTY pillar_summary -> dict[str, bool]:
        """Flat dict for JSON serialization: {pillar_name: passed}."""
        return {p.value: r.passed for p, r in self.pillars.items()}

    METHOD to_dict(self) -> dict[str, Any]:
        """JSON-serializable representation."""
        return {
            "run_id":           self.run_id,
            "timestamp":        self.timestamp.isoformat(),
            "verdict":          self.verdict.value,
            "snr_score":        round(self.snr_score, 4),
            "ihsan_score":      round(self.ihsan_score, 4),
            "tier":             self.tier,
            "pillars":          self.pillar_summary,
            "passed":           self.passed_count,
            "failed":           self.failed_count,
            "total_duration_ms": round(self.total_duration_ms, 1),
            "all_gates_passed": self.all_gates_passed,
            "is_regression":    self.regression.is_regression
                                if self.regression else None,
            "mode":             self.config.mode.value,
            "claim":            self.config.claim,
        }
```

## TDD Anchors

```python
# test_types.py — Phase 1 validation

def test_verdict_enum_values():
    assert len(Verdict) == 3
    assert all(isinstance(v.value, str) for v in Verdict)

def test_pillar_name_count():
    """8 smoke pillars + 2 harness extensions = 10."""
    assert len(PillarName) == 10

def test_harness_config_defaults():
    cfg = HarnessConfig()
    assert cfg.snr_floor == 0.85
    assert cfg.ihsan_floor == 0.95
    assert cfg.mode == RunMode.STANDARD

def test_harness_config_validate_happy():
    cfg = HarnessConfig()
    assert cfg.validate() == []

def test_harness_config_validate_bad_snr():
    cfg = HarnessConfig(snr_floor=1.5)
    errors = cfg.validate()
    assert any("snr_floor" in e for e in errors)

def test_pillar_result_immutable():
    pr = PillarResult(pillar=PillarName.SNR_CHECK, passed=True, duration_ms=10.0)
    with pytest.raises(FrozenInstanceError):
        pr.passed = False

def test_harness_result_to_dict_keys():
    result = _make_minimal_result()   # fixture
    d = result.to_dict()
    assert "run_id" in d
    assert "verdict" in d
    assert "pillars" in d
    assert isinstance(d["pillars"], dict)

def test_regression_report_delta():
    rr = RegressionReport(
        baseline_run_id="b1", baseline_snr=0.92, current_snr=0.89,
        snr_delta=-0.03, baseline_ihsan=0.96, current_ihsan=0.96,
        ihsan_delta=0.0, regressed_pillars=[PillarName.SNR_CHECK],
        tolerance=0.02, is_regression=True,
    )
    assert rr.is_regression is True
    assert rr.snr_delta < -rr.tolerance
```

## Invariants

1. `HarnessResult.verdict == PASS` implies `all_gates_passed is True`
2. `HarnessResult.verdict == PASS` implies `snr_score >= config.snr_floor`
3. `HarnessResult.verdict == PASS` implies `ihsan_score >= config.ihsan_floor`
4. `HarnessResult.verdict == PASS` AND `config.compare_baseline` implies
   `regression.is_regression is False`
5. All scores are clamped to `[0.0, 1.0]`
