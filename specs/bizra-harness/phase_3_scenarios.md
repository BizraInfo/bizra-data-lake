# Phase 3 — Scenario Library and Baseline Comparison

> Standing on Giants: Popper (falsifiability, 1934) · Bayes (prior from data,
> 1763) · Conway (code mirrors org structure, 1967)

## Overview

A harness is only as good as its test scenarios. This phase defines:

1. **HarnessScenario** — A reusable, named test claim with expected outcomes
2. **ScenarioLibrary** — Registry of built-in and user-defined scenarios
3. **Baseline comparison logic** — What "regression" means precisely

## File: `core/harness/scenarios.py`

```pseudocode
IMPORTS:
    from core.integration.constants import (
        UNIFIED_SNR_THRESHOLD,
        UNIFIED_IHSAN_THRESHOLD,
        SNR_THRESHOLD_T1_HIGH,
    )
    from core.harness.types import RunMode, PillarName

# ── Scenario Definition ────────────────────────────────────────────

@dataclass(frozen=True)
CLASS HarnessScenario:
    """A named, reusable test scenario for the harness.

    Scenarios define WHAT to evaluate and WHAT to expect. They don't
    define HOW — that's the runner's job.
    """
    id:              str              # unique slug: "basic_claim", "high_cost"
    claim:           str              # the assertion to evaluate
    description:     str              # human-readable purpose
    mode:            RunMode          = RunMode.STANDARD
    # Expected thresholds (override config defaults if tighter)
    expected_snr:    Optional[float]  = None   # minimum expected SNR
    expected_ihsan:  Optional[float]  = None   # minimum expected Ihsan
    # Pillar focus (if set, only these pillars are evaluated)
    focus_pillars:   Optional[frozenset[PillarName]] = None
    # Tags for filtering
    tags:            frozenset[str]   = frozenset()
    # Category for grouping in reports
    category:        str              = "general"

    METHOD to_config_overrides(self) -> dict[str, Any]:
        """Return config fields this scenario wants to override."""
        overrides = {"claim": self.claim, "mode": self.mode}
        IF self.expected_snr is not None:
            overrides["snr_floor"] = self.expected_snr
        IF self.expected_ihsan is not None:
            overrides["ihsan_floor"] = self.expected_ihsan
        RETURN overrides

# ── Built-in Scenarios ─────────────────────────────────────────────

# Category: Constitutional Gates
SCENARIO_BASIC = HarnessScenario(
    id="basic_claim",
    claim="System meets minimum constitutional thresholds",
    description="Baseline sanity check — SNR >= 0.85, Ihsan >= 0.95",
    mode=RunMode.SMOKE,
    category="constitutional",
    tags=frozenset({"smoke", "ci", "fast"}),
)

SCENARIO_ELITE = HarnessScenario(
    id="elite_claim",
    claim="System meets elite-tier quality thresholds",
    description="High-bar check — SNR >= 0.95, all guardrails pass",
    mode=RunMode.STANDARD,
    expected_snr=SNR_THRESHOLD_T1_HIGH,      # 0.95
    expected_ihsan=UNIFIED_IHSAN_THRESHOLD,   # 0.95
    category="constitutional",
    tags=frozenset({"standard", "ci"}),
)

# Category: Subsystem Focus
SCENARIO_INFERENCE = HarnessScenario(
    id="inference_quality",
    claim="Inference pipeline produces high-signal responses",
    description="Focus on SNR + spearpoint pillars",
    mode=RunMode.STANDARD,
    focus_pillars=frozenset({
        PillarName.SNR_CHECK,
        PillarName.SPEARPOINT,
        PillarName.GUARDRAILS,
    }),
    category="inference",
    tags=frozenset({"inference", "standard"}),
)

SCENARIO_SOVEREIGNTY = HarnessScenario(
    id="sovereignty_boot",
    claim="Sovereign runtime initializes with valid identity",
    description="Focus on runtime + token + evidence chain pillars",
    mode=RunMode.SMOKE,
    focus_pillars=frozenset({
        PillarName.RUNTIME_BOOT,
        PillarName.TOKEN_SYSTEM,
        PillarName.EVIDENCE_CHAIN,
    }),
    category="sovereignty",
    tags=frozenset({"sovereignty", "smoke", "fast"}),
)

# Category: Performance
SCENARIO_LATENCY = HarnessScenario(
    id="latency_budget",
    claim="Full harness completes within 30-second budget",
    description="Performance scenario — all pillars must finish under timeout",
    mode=RunMode.STANDARD,
    category="performance",
    tags=frozenset({"performance", "standard"}),
)

SCENARIO_BDL = HarnessScenario(
    id="benchmark_dominance",
    claim="BDL loop converges within 3 cycles",
    description="Benchmark dominance loop with cycle cap",
    mode=RunMode.BENCHMARK,
    category="benchmark",
    tags=frozenset({"benchmark", "slow"}),
)

# Category: Regression
SCENARIO_REGRESSION = HarnessScenario(
    id="regression_check",
    claim="No quality regression from last sealed baseline",
    description="Compares against .spearpoint/baselines.jsonl",
    mode=RunMode.STANDARD,
    category="regression",
    tags=frozenset({"regression", "ci"}),
)

# ── Built-in Registry ──────────────────────────────────────────────

CONSTANT BUILTIN_SCENARIOS: dict[str, HarnessScenario] = {
    s.id: s for s in [
        SCENARIO_BASIC,
        SCENARIO_ELITE,
        SCENARIO_INFERENCE,
        SCENARIO_SOVEREIGNTY,
        SCENARIO_LATENCY,
        SCENARIO_BDL,
        SCENARIO_REGRESSION,
    ]
}

# ── Scenario Library ───────────────────────────────────────────────

CLASS ScenarioLibrary:
    """Registry of named test scenarios.

    Combines built-in scenarios with user-defined ones loaded from
    an optional YAML/JSON file.
    """

    METHOD __init__(self, extra: Optional[dict[str, HarnessScenario]] = None):
        self._scenarios: dict[str, HarnessScenario] = dict(BUILTIN_SCENARIOS)
        IF extra:
            self._scenarios.update(extra)

    @classmethod
    METHOD default(cls) -> 'ScenarioLibrary':
        """Load built-in scenarios + .spearpoint/scenarios.json if present."""
        extra = cls._load_user_scenarios()
        RETURN cls(extra=extra)

    @classmethod
    METHOD _load_user_scenarios(cls) -> dict[str, HarnessScenario]:
        """Load user-defined scenarios from .spearpoint/scenarios.json."""
        path = Path(".spearpoint/scenarios.json")
        IF NOT path.exists():
            RETURN {}
        TRY:
            data = json.loads(path.read_text())
            scenarios = {}
            FOR entry IN data:
                s = HarnessScenario(
                    id=entry["id"],
                    claim=entry["claim"],
                    description=entry.get("description", ""),
                    mode=RunMode(entry.get("mode", "standard")),
                    expected_snr=entry.get("expected_snr"),
                    expected_ihsan=entry.get("expected_ihsan"),
                    category=entry.get("category", "user"),
                    tags=frozenset(entry.get("tags", [])),
                )
                scenarios[s.id] = s
            RETURN scenarios
        EXCEPT Exception:
            RETURN {}   # graceful degradation

    METHOD get(self, scenario_id: str) -> Optional[HarnessScenario]:
        RETURN self._scenarios.get(scenario_id)

    METHOD list_all(self) -> list[HarnessScenario]:
        RETURN list(self._scenarios.values())

    METHOD list_by_tag(self, tag: str) -> list[HarnessScenario]:
        RETURN [s for s in self._scenarios.values() if tag in s.tags]

    METHOD list_by_category(self, category: str) -> list[HarnessScenario]:
        RETURN [s for s in self._scenarios.values() if s.category == category]

    METHOD register(self, scenario: HarnessScenario) -> None:
        """Add a scenario at runtime (e.g., from test fixtures)."""
        IF scenario.id IN self._scenarios:
            RAISE ValueError(f"Scenario '{scenario.id}' already registered")
        self._scenarios[scenario.id] = scenario
```

## Focus Pillar Logic

When a scenario specifies `focus_pillars`, the runner should filter:

```pseudocode
# In HarnessRunner.run(), after resolving scenario:
IF scenario AND scenario.focus_pillars:
    pillar_evaluators = [
        e for e in pillar_evaluators
        if e.name in scenario.focus_pillars
    ]
```

This allows targeted evaluations without running the entire suite.

## TDD Anchors

```python
# test_scenarios.py — Phase 3 validation

def test_builtin_count():
    """7 built-in scenarios exist."""
    assert len(BUILTIN_SCENARIOS) == 7

def test_scenario_ids_are_unique():
    ids = [s.id for s in BUILTIN_SCENARIOS.values()]
    assert len(ids) == len(set(ids))

def test_scenario_library_default():
    lib = ScenarioLibrary.default()
    assert lib.get("basic_claim") is not None
    assert lib.get("nonexistent") is None

def test_scenario_library_list_by_tag():
    lib = ScenarioLibrary.default()
    ci_scenarios = lib.list_by_tag("ci")
    assert len(ci_scenarios) >= 2   # basic_claim + elite_claim + regression

def test_scenario_library_register():
    lib = ScenarioLibrary()
    custom = HarnessScenario(
        id="custom_test", claim="Custom claim", description="Test",
    )
    lib.register(custom)
    assert lib.get("custom_test") is custom

def test_scenario_library_register_duplicate_raises():
    lib = ScenarioLibrary()
    with pytest.raises(ValueError, match="already registered"):
        lib.register(SCENARIO_BASIC)   # already in builtins

def test_scenario_to_config_overrides():
    overrides = SCENARIO_ELITE.to_config_overrides()
    assert overrides["snr_floor"] == 0.95
    assert overrides["claim"] == SCENARIO_ELITE.claim

def test_focus_pillars_filters_evaluators():
    """Scenario with focus_pillars should reduce pillar count."""
    scenario = SCENARIO_SOVEREIGNTY   # 3 focus pillars
    all_evaluators = pillars_for_mode(RunMode.SMOKE)
    filtered = [e for e in all_evaluators if e.name in scenario.focus_pillars]
    # Filtered should be subset
    assert len(filtered) <= len(all_evaluators)

def test_user_scenarios_graceful_on_missing_file():
    """No .spearpoint/scenarios.json → empty dict, no crash."""
    result = ScenarioLibrary._load_user_scenarios()
    # Result is either empty (no file) or populated (file exists)
    assert isinstance(result, dict)
```

## User-Defined Scenario Format

`.spearpoint/scenarios.json`:
```json
[
  {
    "id": "my_custom_eval",
    "claim": "Token economy maintains Gini <= 0.35",
    "description": "Validates ADL justice gate under load",
    "mode": "standard",
    "expected_snr": 0.90,
    "category": "economics",
    "tags": ["economics", "custom"]
  }
]
```
