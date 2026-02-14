# Phase 20.1: Simulation Types & Battlefield Registry

## Context

This spec defines the data types and battlefield management layer for the
True Spearpoint Benchmark Dominance Loop simulation. The simulation exercises
the full 5-phase recursive loop (Evaluate -> Ablate -> Architect -> Submit ->
Analyze) with realistic metrics, convergence tracking, and SOTA detection.

**Existing infrastructure (zero new dependencies):**

| Module | Status | Usage |
|--------|--------|-------|
| `core/benchmark/clear_framework.py` | Implemented | CLEARFramework, CLEARMetrics |
| `core/benchmark/dominance_loop.py` | Implemented | BenchmarkDominanceLoop, LoopState |
| `core/benchmark/ablation_engine.py` | Implemented | AblationEngine, AblationStudy |
| `core/benchmark/moe_router.py` | Implemented | MoERouter, ExpertTier |
| `core/benchmark/guardrails.py` | Implemented | GuardrailSuite |
| `core/proof_engine/snr.py` | Implemented | SNREngine, SNRTrace |
| `core/proof_engine/receipt.py` | Implemented | ReceiptBuilder, Ed25519Signer |
| `core/proof_engine/evidence_ledger.py` | Implemented | EvidenceLedger |
| `core/spearpoint/auto_evaluator.py` | Implemented | AutoEvaluator |
| `core/spearpoint/metrics_provider.py` | Implemented | MetricsProvider |
| `core/integration/constants.py` | Implemented | All thresholds |

Standing on Giants:
- Shannon (1948): SNR as convergence signal
- Boyd (1995): OODA loop phases
- Pareto (1896): Multi-objective efficiency frontier
- Deming (1950): PDCA continuous improvement
- Goldratt (1984): Theory of Constraints (bottleneck identification)
- HAL (2025): Holistic Agent Leaderboard methodology

---

## Module 1: Simulation Data Types

### File: `core/spearpoint/simulation_types.py` (~120 lines)

```
FROM enum IMPORT Enum
FROM dataclasses IMPORT dataclass, field
FROM typing IMPORT Dict, List, Optional, Tuple
FROM datetime IMPORT datetime, timezone

# -------------------------------------------------------------------
# Enums
# -------------------------------------------------------------------

CLASS BattlefieldId(Enum):
    """Target benchmarks for the 2026 campaign."""
    HLE = "HLE"                         # Humanity's Last Exam
    SWE_BENCH = "SWE_bench_Verified"    # Agentic coding
    ARC_AGI_2 = "ARC_AGI_2"            # Abstract reasoning
    AGENT_BEATS = "AgentBeats"          # Enterprise agentic tasks
    CUSTOM = "CUSTOM"                   # User-defined benchmark


CLASS CyclePhase(Enum):
    """Phases within a single dominance iteration."""
    EVALUATE = "evaluate"
    ABLATE = "ablate"
    ARCHITECT = "architect"
    SUBMIT = "submit"
    ANALYZE = "analyze"


CLASS DominanceStatus(Enum):
    """Overall simulation status."""
    INITIALIZING = "initializing"
    RUNNING = "running"
    SOTA_PARTIAL = "sota_partial"       # SOTA on some but not all targets
    DOMINATED = "dominated"             # SOTA on all targets
    BUDGET_EXHAUSTED = "budget_exhausted"
    PATIENCE_EXHAUSTED = "patience_exhausted"
    FAILED = "failed"


# -------------------------------------------------------------------
# Metric Snapshots
# -------------------------------------------------------------------

@dataclass
CLASS CLEARSnapshot:
    """Immutable capture of CLEAR metrics at a point in time."""
    cna: float              # Cost-Normalized Accuracy
    cps: float              # Cost Per Success (USD)
    scr: float              # SLA Compliance Rate (0-100%)
    pas: float              # Policy Adherence Score (0-1)
    pass_at_k: float        # Pass@8 consistency (0-1)
    pareto_efficient: bool  # On the Pareto frontier?
    raw_efficacy: float     # Raw efficacy dimension (0-1)
    raw_cost: float         # Raw cost dimension (0-1)
    raw_latency: float      # Raw latency dimension (0-1)
    raw_assurance: float    # Raw assurance dimension (0-1)
    raw_reliability: float  # Raw reliability dimension (0-1)
    weighted_score: float   # Weighted composite (0-1)

    METHOD delta(self, previous: "CLEARSnapshot") -> Dict[str, float]:
        """Compute deltas against previous snapshot."""
        RETURN {
            "cna": self.cna - previous.cna,
            "cps": self.cps - previous.cps,
            "pass_at_k": self.pass_at_k - previous.pass_at_k,
            "weighted_score": self.weighted_score - previous.weighted_score,
        }


@dataclass
CLASS AblationFinding:
    """Single weak module identified by ablation."""
    name: str                   # Module/component name
    category: str               # reasoning | memory | routing | tool | verifier
    contribution: float         # Impact on CNA when removed (negative = critical)
    essential: bool             # True if contribution > 10%
    harmful: bool               # True if contribution < -5%


@dataclass
CLASS ArchitectureUpgrade:
    """Record of an architecture change."""
    module_name: str            # What was upgraded
    old_version: str            # Previous version
    new_version: str            # New version
    technique: str              # MoE | MIRAS | SequentialAttention | ZScorer | Ensemble
    expected_cna_gain: float    # Estimated CNA improvement


@dataclass
CLASS BattlefieldResult:
    """Result of submitting to a single benchmark."""
    battlefield_id: str         # Benchmark name
    score: float                # Benchmark-specific score
    rank: int                   # Current rank on leaderboard
    sota_score: float           # Current SOTA score
    achieved_sota: bool         # score > sota_score
    cost_usd: float             # Total cost of submission
    latency_p99_ms: float       # p99 latency across tasks
    integrity_valid: bool       # Anti-gaming validation passed


# -------------------------------------------------------------------
# Iteration State
# -------------------------------------------------------------------

@dataclass
CLASS IterationRecord:
    """Complete record of one dominance loop iteration."""
    iteration: int
    phase_timings: Dict[str, float]     # phase_name -> duration_seconds
    clear_snapshot: CLEARSnapshot
    ablation_findings: List[AblationFinding]
    upgrades: List[ArchitectureUpgrade]
    battlefield_results: List[BattlefieldResult]
    snr: float
    ihsan_score: float
    battlefields_won: List[str]
    cost_total_usd: float
    timestamp: str = field(default_factory=LAMBDA: datetime.now(timezone.utc).isoformat())


@dataclass
CLASS SimulationState:
    """Mutable state across the entire simulation."""
    status: DominanceStatus = DominanceStatus.INITIALIZING
    iteration: int = 0
    snr: float = 0.0
    cna: float = 0.0
    ihsan_score: float = 0.0
    battlefields_won: List[str] = field(default_factory=list)
    iterations: List[IterationRecord] = field(default_factory=list)
    budget_remaining_usd: float = 100.0     # Default campaign budget
    patience: int = 5                        # Regressions before termination
    regressions_consecutive: int = 0
    architecture_version: str = "9.0.0"
    pareto_frontier: List[Tuple[float, float]] = field(default_factory=list)
    # Pareto frontier: list of (accuracy, 1/cost) points

    METHOD is_terminated(self) -> bool:
        RETURN self.status IN (
            DominanceStatus.DOMINATED,
            DominanceStatus.BUDGET_EXHAUSTED,
            DominanceStatus.PATIENCE_EXHAUSTED,
            DominanceStatus.FAILED,
        )
```

### Design Decisions

| Decision | Rationale |
|----------|-----------|
| Immutable CLEARSnapshot | Prevents accidental mutation between iterations |
| BattlefieldId enum | Typesafe benchmark references, extensible via CUSTOM |
| Budget tracking in USD | CLEAR principle: cost is a first-class dimension |
| Pareto frontier as (accuracy, 1/cost) | Standard Pareto representation for multi-objective |
| Patience counter | Prevents infinite loops when no progress (Goldratt) |

---

## Module 2: Battlefield Registry

### File: `core/spearpoint/battlefield_registry.py` (~130 lines)

```
FROM dataclasses IMPORT dataclass, field
FROM typing IMPORT Callable, Dict, List, Optional
FROM .simulation_types IMPORT BattlefieldId, BattlefieldResult


@dataclass
CLASS BattlefieldSpec:
    """Specification for a registered benchmark battlefield."""
    id: str                         # Unique benchmark identifier
    display_name: str               # Human-readable name
    sota_score: float               # Current known SOTA score
    task_count: int                 # Number of tasks in the benchmark
    scoring_fn: str                 # "accuracy" | "pass_at_k" | "kami"
    cost_per_task_usd: float        # Estimated cost per task
    max_latency_ms: float           # SLA: max acceptable p99 latency
    integrity_checks: List[str]     # Required ABC checks for this benchmark
    leaderboard_url: str = ""       # Public leaderboard URL


CLASS BattlefieldRegistry:
    """
    Registry of target benchmarks for campaign execution.

    Manages SOTA tracking, task suite loading, and submission protocol
    for each registered battlefield.

    Standing on: HAL (2025) — holistic benchmark methodology
    """

    CONSTRUCTOR():
        self._battlefields: Dict[str, BattlefieldSpec] = {}
        self._register_defaults()

    METHOD _register_defaults(self) -> None:
        """Register the 2026 target battlefields with current SOTA."""
        self.register(BattlefieldSpec(
            id=BattlefieldId.HLE.value,
            display_name="Humanity's Last Exam",
            sota_score=0.542,           # As of Jan 2026
            task_count=3000,
            scoring_fn="accuracy",
            cost_per_task_usd=0.12,
            max_latency_ms=30000,       # 30s per task
            integrity_checks=["LEAKAGE_SCAN", "SEED_SWEEP", "NULL_MODEL"],
        ))
        self.register(BattlefieldSpec(
            id=BattlefieldId.SWE_BENCH.value,
            display_name="SWE-bench Verified",
            sota_score=0.524,           # As of Jan 2026
            task_count=500,
            scoring_fn="pass_at_k",
            cost_per_task_usd=0.35,
            max_latency_ms=60000,       # 60s for code generation
            integrity_checks=["LEAKAGE_SCAN", "TOOL_SANDBOX", "REGRESSION"],
        ))
        self.register(BattlefieldSpec(
            id=BattlefieldId.ARC_AGI_2.value,
            display_name="ARC-AGI-2",
            sota_score=0.821,           # As of Jan 2026
            task_count=400,
            scoring_fn="accuracy",
            cost_per_task_usd=0.08,
            max_latency_ms=10000,       # 10s per task
            integrity_checks=["LEAKAGE_SCAN", "SEED_SWEEP", "PROMPT_INJECTION"],
        ))

    METHOD register(self, spec: BattlefieldSpec) -> None:
        """Register or update a battlefield spec."""
        self._battlefields[spec.id] = spec

    METHOD get(self, battlefield_id: str) -> Optional[BattlefieldSpec]:
        """Get battlefield spec by ID."""
        RETURN self._battlefields.get(battlefield_id)

    METHOD list_ids(self) -> List[str]:
        """List all registered battlefield IDs."""
        RETURN sorted(self._battlefields.keys())

    METHOD estimated_campaign_cost(self, battlefield_ids: List[str]) -> float:
        """Estimate total campaign cost across battlefields."""
        total = 0.0
        FOR bid IN battlefield_ids:
            spec = self._battlefields.get(bid)
            IF spec:
                total += spec.task_count * spec.cost_per_task_usd
        RETURN total

    METHOD update_sota(self, battlefield_id: str, new_sota: float) -> None:
        """Update SOTA score after a successful campaign submission."""
        spec = self._battlefields.get(battlefield_id)
        IF spec AND new_sota > spec.sota_score:
            spec.sota_score = new_sota

    METHOD check_sota(self, battlefield_id: str, score: float) -> bool:
        """Check if a score beats the current SOTA."""
        spec = self._battlefields.get(battlefield_id)
        IF spec IS None:
            RETURN False
        RETURN score > spec.sota_score
```

### TDD Anchors

```
test_default_battlefields_registered
    registry = BattlefieldRegistry()
    ASSERT len(registry.list_ids()) == 3
    ASSERT "HLE" IN registry.list_ids()
    ASSERT "SWE_bench_Verified" IN registry.list_ids()
    ASSERT "ARC_AGI_2" IN registry.list_ids()

test_get_battlefield_spec
    registry = BattlefieldRegistry()
    spec = registry.get("HLE")
    ASSERT spec IS NOT None
    ASSERT spec.sota_score > 0
    ASSERT spec.task_count == 3000

test_custom_battlefield_registration
    registry = BattlefieldRegistry()
    registry.register(BattlefieldSpec(
        id="AgentBeats", display_name="AgentBeats 2026",
        sota_score=0.0, task_count=1000, scoring_fn="kami",
        cost_per_task_usd=0.15, max_latency_ms=20000,
        integrity_checks=["LEAKAGE_SCAN"]
    ))
    ASSERT "AgentBeats" IN registry.list_ids()

test_campaign_cost_estimation
    registry = BattlefieldRegistry()
    cost = registry.estimated_campaign_cost(["HLE", "SWE_bench_Verified"])
    ASSERT cost > 0
    # HLE: 3000 * 0.12 = 360 + SWE: 500 * 0.35 = 175 = 535
    ASSERT abs(cost - 535.0) < 0.01

test_sota_update
    registry = BattlefieldRegistry()
    old_sota = registry.get("HLE").sota_score
    registry.update_sota("HLE", old_sota + 0.05)
    ASSERT registry.get("HLE").sota_score == old_sota + 0.05

test_check_sota_true
    registry = BattlefieldRegistry()
    ASSERT registry.check_sota("HLE", 999.0) IS True

test_check_sota_false
    registry = BattlefieldRegistry()
    ASSERT registry.check_sota("HLE", 0.0) IS False

test_check_sota_unknown_battlefield
    registry = BattlefieldRegistry()
    ASSERT registry.check_sota("NONEXISTENT", 1.0) IS False
```

---

## Implementation Order

| Step | File | Lines | Dependencies |
|------|------|-------|-------------|
| 1 | `core/spearpoint/simulation_types.py` | ~120 | None (stdlib only) |
| 2 | `core/spearpoint/battlefield_registry.py` | ~130 | simulation_types |

**Total: ~250 lines, zero new external dependencies.**
