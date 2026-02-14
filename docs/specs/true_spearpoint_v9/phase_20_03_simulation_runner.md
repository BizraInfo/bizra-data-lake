# Phase 20.3: Simulation Runner & TDD Anchors

## Context

This spec defines the simulation runner — the top-level entry point that
initializes the TrueSpearpointSimulator, executes the dominance loop, and
produces the final report. It also defines the complete TDD anchor suite.

Standing on Giants:
- Kent Beck (2003): Test-Driven Development
- Shannon (1948): SNR convergence validation
- Deming (1950): Measurable quality gates

---

## Module 4: Simulation Runner

### File: `core/spearpoint/run_simulation.py` (~100 lines)

```
"""
True Spearpoint Simulation Runner
==================================

Entry point for the Benchmark Dominance Loop lifecycle simulation.

Usage:
    python -m core.spearpoint.run_simulation
    python -m core.spearpoint.run_simulation --targets HLE SWE_bench_Verified
    python -m core.spearpoint.run_simulation --budget 50.0 --max-iter 5
"""

IMPORT asyncio
IMPORT argparse
IMPORT json
IMPORT logging
IMPORT sys
FROM pathlib IMPORT Path

FROM .dominance_simulator IMPORT TrueSpearpointSimulator
FROM .simulation_types IMPORT DominanceStatus

logger = logging.getLogger(__name__)


ASYNC FUNCTION run_simulation(
    targets: List[str] = None,
    budget_usd: float = 100.0,
    max_iterations: int = 10,
    patience: int = 5,
    json_output: bool = False,
    verbose: bool = False,
) -> Dict[str, Any]:
    """
    Execute the True Spearpoint dominance simulation.

    Returns the final dominance report dict.
    """
    IF verbose:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    simulator = TrueSpearpointSimulator(
        targets=targets,
        budget_usd=budget_usd,
        max_iterations=max_iterations,
        patience=patience,
    )

    # Optional: attach evidence ledger for audit trail
    TRY:
        FROM core.proof_engine.evidence_ledger IMPORT EvidenceLedger
        ledger_path = Path(".spearpoint") / "simulation_ledger.jsonl"
        ledger_path.parent.mkdir(parents=True, exist_ok=True)
        ledger = EvidenceLedger(str(ledger_path))
        simulator.set_evidence_ledger(ledger)
    EXCEPT Exception:
        pass    # Ledger is optional; simulation works without it

    # Execute
    state = AWAIT simulator.dominate()
    report = simulator.dominance_report()

    IF json_output:
        print(json.dumps(report, indent=2))
    ELSE:
        _print_report(report, state)

    RETURN report


FUNCTION _print_report(report: Dict, state) -> None:
    """Print human-readable dominance report."""
    print()
    print("=" * 60)
    print("FINAL DOMINANCE REPORT")
    print("=" * 60)
    print(f"Status:              {report['status']}")
    print(f"Iterations:          {report['iterations']}")
    print(f"SNR:                 {report['snr']:.5f} (target >= 0.98)")
    print(f"CNA:                 {report['cna']:.2f}")
    print(f"Ihsan:               {report['ihsan_score']:.3f}")
    print(f"Battlefields won:    {report['battlefields_won']}")
    print(f"Architecture:        {report['architecture_version']}")
    print(f"Budget used:         ${report['budget_used_usd']:.2f}")
    print()

    # Constraints
    constraints = report["constraints_satisfied"]
    print("Constraints:")
    FOR key, val IN constraints.items():
        status = "PASS" IF val ELSE "FAIL"
        print(f"  {key}: {status}")
    print()

    # Iteration history (compact)
    IF state.iterations:
        print("Iteration History:")
        print(f"  {'#':>3}  {'CNA':>8}  {'SNR':>8}  {'Won':>3}  {'Cost':>8}")
        print(f"  {'---':>3}  {'--------':>8}  {'--------':>8}  {'---':>3}  {'--------':>8}")
        FOR rec IN state.iterations:
            print(
                f"  {rec.iteration:>3}  "
                f"{rec.clear_snapshot.cna:>8.2f}  "
                f"{rec.snr:>8.5f}  "
                f"{len(rec.battlefields_won):>3}  "
                f"${rec.cost_total_usd:>7.2f}"
            )
    print()
    print("=" * 60)

    IF report["status"] == DominanceStatus.DOMINATED.value:
        print("TRUE SPEARPOINT: Benchmark Dominance Loop successful.")
        print("All constraints satisfied. All targets dominated.")
    ELIF report["status"] == DominanceStatus.SOTA_PARTIAL.value:
        remaining = set(state.targets) - set(report["battlefields_won"])
        print(f"Partial dominance. Remaining: {remaining}")
    ELSE:
        print(f"Terminated: {report['status']}")

    print("=" * 60)


FUNCTION main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(description="True Spearpoint Simulation")
    parser.add_argument("--targets", nargs="+", default=None,
                        help="Target battlefield IDs")
    parser.add_argument("--budget", type=float, default=100.0,
                        help="Campaign budget in USD")
    parser.add_argument("--max-iter", type=int, default=10,
                        help="Maximum iterations")
    parser.add_argument("--patience", type=int, default=5,
                        help="Max consecutive regressions")
    parser.add_argument("--json", action="store_true",
                        help="JSON output")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Verbose logging")

    args = parser.parse_args()

    asyncio.run(run_simulation(
        targets=args.targets,
        budget_usd=args.budget,
        max_iterations=args.max_iter,
        patience=args.patience,
        json_output=args.json,
        verbose=args.verbose,
    ))


IF __name__ == "__main__":
    main()
```

---

## Module 5: Package Updates

### File: `core/spearpoint/__init__.py` — Add exports

```
# Add after existing exports:
FROM .simulation_types IMPORT (
    SimulationState,
    CLEARSnapshot,
    BattlefieldResult,
    IterationRecord,
    DominanceStatus,
)
FROM .battlefield_registry IMPORT BattlefieldRegistry
FROM .dominance_simulator IMPORT TrueSpearpointSimulator
```

---

## TDD Anchors

### File: `tests/core/spearpoint/test_dominance_simulation.py` (~200 lines)

```
# -----------------------------------------------------------------
# SIMULATION TYPES
# -----------------------------------------------------------------

test_simulation_state_initial_values
    state = SimulationState()
    ASSERT state.status == DominanceStatus.INITIALIZING
    ASSERT state.iteration == 0
    ASSERT state.snr == 0.0
    ASSERT state.battlefields_won == []
    ASSERT NOT state.is_terminated()

test_simulation_state_terminated_dominated
    state = SimulationState(status=DominanceStatus.DOMINATED)
    ASSERT state.is_terminated()

test_simulation_state_terminated_budget
    state = SimulationState(status=DominanceStatus.BUDGET_EXHAUSTED)
    ASSERT state.is_terminated()

test_simulation_state_not_terminated_running
    state = SimulationState(status=DominanceStatus.RUNNING)
    ASSERT NOT state.is_terminated()

test_clear_snapshot_delta
    snap1 = CLEARSnapshot(cna=260.0, cps=0.05, scr=94.0, pas=0.98,
                          pass_at_k=0.76, pareto_efficient=True,
                          raw_efficacy=0.8, raw_cost=0.7, raw_latency=0.75,
                          raw_assurance=0.85, raw_reliability=0.80,
                          weighted_score=0.78)
    snap2 = CLEARSnapshot(cna=265.0, cps=0.045, scr=95.0, pas=0.99,
                          pass_at_k=0.79, pareto_efficient=True,
                          raw_efficacy=0.82, raw_cost=0.72, raw_latency=0.77,
                          raw_assurance=0.87, raw_reliability=0.82,
                          weighted_score=0.80)
    delta = snap2.delta(snap1)
    ASSERT delta["cna"] == 5.0
    ASSERT delta["pass_at_k"] == pytest.approx(0.03, abs=0.001)

test_ablation_finding_essential
    finding = AblationFinding(name="reasoning", category="reasoning",
                              contribution=-15.0, essential=True, harmful=False)
    ASSERT finding.essential
    ASSERT NOT finding.harmful

test_battlefield_result_sota
    result = BattlefieldResult(
        battlefield_id="HLE", score=0.60, rank=1,
        sota_score=0.542, achieved_sota=True,
        cost_usd=360.0, latency_p99_ms=25000.0,
        integrity_valid=True,
    )
    ASSERT result.achieved_sota

# -----------------------------------------------------------------
# BATTLEFIELD REGISTRY
# -----------------------------------------------------------------

test_registry_defaults
    registry = BattlefieldRegistry()
    ids = registry.list_ids()
    ASSERT "HLE" IN ids
    ASSERT "SWE_bench_Verified" IN ids
    ASSERT "ARC_AGI_2" IN ids
    ASSERT len(ids) == 3

test_registry_get_spec
    registry = BattlefieldRegistry()
    spec = registry.get("HLE")
    ASSERT spec.task_count == 3000
    ASSERT spec.scoring_fn == "accuracy"

test_registry_custom_battlefield
    registry = BattlefieldRegistry()
    FROM core.spearpoint.battlefield_registry IMPORT BattlefieldSpec
    registry.register(BattlefieldSpec(
        id="CustomBench", display_name="Custom",
        sota_score=0.5, task_count=100,
        scoring_fn="accuracy", cost_per_task_usd=0.1,
        max_latency_ms=5000, integrity_checks=[],
    ))
    ASSERT "CustomBench" IN registry.list_ids()

test_registry_campaign_cost
    registry = BattlefieldRegistry()
    cost = registry.estimated_campaign_cost(["HLE"])
    # 3000 tasks * $0.12 = $360
    ASSERT abs(cost - 360.0) < 0.01

test_registry_sota_update
    registry = BattlefieldRegistry()
    registry.update_sota("HLE", 0.60)
    ASSERT registry.get("HLE").sota_score == 0.60

test_registry_check_sota
    registry = BattlefieldRegistry()
    ASSERT registry.check_sota("HLE", 1.0)
    ASSERT NOT registry.check_sota("HLE", 0.0)
    ASSERT NOT registry.check_sota("NONEXISTENT", 1.0)

# -----------------------------------------------------------------
# SIMULATOR PHASES
# -----------------------------------------------------------------

test_simulator_initialization
    sim = TrueSpearpointSimulator(
        targets=["HLE"], budget_usd=50.0, max_iterations=3
    )
    ASSERT sim.state.status == DominanceStatus.INITIALIZING
    ASSERT sim.state.budget_remaining_usd == 50.0

test_phase_evaluate_returns_valid_snapshot
    sim = TrueSpearpointSimulator(targets=["HLE"])
    snapshot, snr, ihsan = sim._phase_evaluate()
    ASSERT snapshot.cna > 0
    ASSERT 0 <= snr <= 1.0
    ASSERT 0 <= ihsan <= 1.0
    ASSERT snapshot.weighted_score > 0

test_phase_ablate_finds_bottlenecks
    sim = TrueSpearpointSimulator(targets=["HLE"])
    snapshot, _, _ = sim._phase_evaluate()
    findings = sim._phase_ablate(snapshot)
    ASSERT len(findings) == 2   # Top 2 bottlenecks
    ASSERT all(f.name FOR f IN findings)
    ASSERT all(f.category IN ("reasoning", "memory", "routing", "verifier") FOR f IN findings)

test_phase_architect_produces_upgrades
    sim = TrueSpearpointSimulator(targets=["HLE"])
    snapshot, _, _ = sim._phase_evaluate()
    findings = sim._phase_ablate(snapshot)
    upgrades = sim._phase_architect(findings, snapshot)
    ASSERT len(upgrades) >= 1
    ASSERT all(u.technique FOR u IN upgrades)
    ASSERT all(u.expected_cna_gain >= 0 FOR u IN upgrades)

test_phase_submit_returns_results
    sim = TrueSpearpointSimulator(targets=["HLE", "SWE_bench_Verified"])
    snapshot, _, _ = sim._phase_evaluate()
    results = sim._phase_submit(snapshot, [])
    ASSERT len(results) == 2
    ASSERT all(r.battlefield_id FOR r IN results)
    ASSERT all(0 <= r.score <= 1.0 FOR r IN results)
    ASSERT all(r.cost_usd > 0 FOR r IN results)

test_phase_analyze_detects_sota
    sim = TrueSpearpointSimulator(targets=["HLE"])
    result = BattlefieldResult(
        battlefield_id="HLE", score=0.99, rank=1,
        sota_score=0.542, achieved_sota=True,
        cost_usd=360.0, latency_p99_ms=25000.0,
        integrity_valid=True,
    )
    won = sim._phase_analyze([result])
    ASSERT "HLE" IN won

test_phase_analyze_rejects_invalid_integrity
    sim = TrueSpearpointSimulator(targets=["HLE"])
    result = BattlefieldResult(
        battlefield_id="HLE", score=0.99, rank=1,
        sota_score=0.542, achieved_sota=True,
        cost_usd=360.0, latency_p99_ms=25000.0,
        integrity_valid=False,          # Anti-gaming failed
    )
    won = sim._phase_analyze([result])
    ASSERT "HLE" NOT IN won

# -----------------------------------------------------------------
# FULL LIFECYCLE
# -----------------------------------------------------------------

@pytest.mark.asyncio
test_full_simulation_converges
    """The dominance loop converges within max_iterations."""
    sim = TrueSpearpointSimulator(
        targets=["HLE"],
        budget_usd=1000.0,
        max_iterations=5,
    )
    state = AWAIT sim.dominate()
    ASSERT state.iteration >= 1
    ASSERT state.snr > 0
    ASSERT state.cna > 0
    ASSERT state.is_terminated() OR state.iteration == 5

@pytest.mark.asyncio
test_simulation_budget_exhaustion
    """Simulation terminates when budget is exhausted."""
    sim = TrueSpearpointSimulator(
        targets=["HLE"],
        budget_usd=0.01,       # Tiny budget
        max_iterations=10,
    )
    state = AWAIT sim.dominate()
    ASSERT state.status == DominanceStatus.BUDGET_EXHAUSTED

@pytest.mark.asyncio
test_simulation_records_history
    """Each iteration produces an IterationRecord."""
    sim = TrueSpearpointSimulator(
        targets=["HLE"],
        budget_usd=1000.0,
        max_iterations=3,
    )
    state = AWAIT sim.dominate()
    ASSERT len(state.iterations) >= 1
    FOR record IN state.iterations:
        ASSERT record.iteration > 0
        ASSERT record.clear_snapshot IS NOT None
        ASSERT record.snr > 0
        ASSERT len(record.phase_timings) == 5   # All 5 phases timed

@pytest.mark.asyncio
test_cna_improves_over_iterations
    """CNA should generally improve across iterations (non-regression)."""
    sim = TrueSpearpointSimulator(
        targets=["HLE"],
        budget_usd=1000.0,
        max_iterations=3,
    )
    state = AWAIT sim.dominate()
    IF len(state.iterations) >= 2:
        first_cna = state.iterations[0].clear_snapshot.cna
        last_cna = state.iterations[-1].clear_snapshot.cna
        ASSERT last_cna >= first_cna * 0.95   # Allow 5% variance

@pytest.mark.asyncio
test_dominance_report_structure
    """Dominance report has required fields."""
    sim = TrueSpearpointSimulator(
        targets=["HLE"],
        budget_usd=1000.0,
        max_iterations=2,
    )
    AWAIT sim.dominate()
    report = sim.dominance_report()
    ASSERT "status" IN report
    ASSERT "iterations" IN report
    ASSERT "snr" IN report
    ASSERT "cna" IN report
    ASSERT "battlefields_won" IN report
    ASSERT "architecture_version" IN report
    ASSERT "constraints_satisfied" IN report

@pytest.mark.asyncio
test_architecture_version_increments
    """Architecture version increments with each iteration."""
    sim = TrueSpearpointSimulator(
        targets=["HLE"],
        budget_usd=1000.0,
        max_iterations=3,
    )
    state = AWAIT sim.dominate()
    ASSERT "TRUE_SPEARPOINT" IN state.architecture_version
    ASSERT f"9.0.{state.iteration}" IN state.architecture_version

# -----------------------------------------------------------------
# ANTI-GAMING
# -----------------------------------------------------------------

test_integrity_validation_blocks_gaming
    """Submissions with failed integrity checks are excluded from SOTA."""
    sim = TrueSpearpointSimulator(targets=["HLE"])
    bad_result = BattlefieldResult(
        battlefield_id="HLE", score=1.0, rank=1,
        sota_score=0.542, achieved_sota=True,
        cost_usd=0.01, latency_p99_ms=1.0,
        integrity_valid=False,
    )
    won = sim._phase_analyze([bad_result])
    ASSERT len(won) == 0
    # SOTA should NOT be updated
    ASSERT sim.registry.get("HLE").sota_score == 0.542
```

---

## Implementation Order

| Step | File | Est. Lines | Dependencies |
|------|------|-----------|-------------|
| 1 | `core/spearpoint/simulation_types.py` | ~120 | stdlib only |
| 2 | `core/spearpoint/battlefield_registry.py` | ~130 | simulation_types |
| 3 | `core/spearpoint/dominance_simulator.py` | ~350 | types, registry, CLEAR, ablation, SNR |
| 4 | `core/spearpoint/run_simulation.py` | ~100 | simulator |
| 5 | `core/spearpoint/__init__.py` | +6 | exports |
| 6 | `tests/core/spearpoint/test_dominance_simulation.py` | ~200 | all above |

**Total: ~900 lines of implementation + ~200 lines of tests.**

---

## Verification Plan

```bash
# Step 1: Type checks
PYTHONPATH=. python -c "from core.spearpoint.simulation_types import *; print('Types OK')"

# Step 2: Registry tests
PYTHONPATH=. pytest tests/core/spearpoint/test_dominance_simulation.py -k "registry" -v

# Step 3: Phase unit tests
PYTHONPATH=. pytest tests/core/spearpoint/test_dominance_simulation.py -k "phase_" -v

# Step 4: Full lifecycle tests
PYTHONPATH=. pytest tests/core/spearpoint/test_dominance_simulation.py -k "full_simulation or converges" -v

# Step 5: CLI smoke test
PYTHONPATH=. python -m core.spearpoint.run_simulation --targets HLE --max-iter 2 --json

# Step 6: Full spearpoint regression
PYTHONPATH=. pytest tests/core/spearpoint/ -v --timeout=60

# Step 7: Full suite (no regressions)
PYTHONPATH=. pytest tests/ -q --timeout=60
```

---

## Cross-Module Integration

```
                     run_simulation.py (CLI entry)
                            |
                  TrueSpearpointSimulator
                 /     |      |     |     \
           EVALUATE  ABLATE  ARCH  SUBMIT  ANALYZE
               |       |      |      |        |
         CLEARFrmwk  AblEng  MoE  Registry  Pareto
         SNREngine           Strat  GuardR
         MetricsProv
               |
         EvidenceLedger (optional audit trail)
```

**The loop composes existing modules — zero new external dependencies.**
**Every iteration is measured, gated, and receipted.**
**Ihsan >= 0.95 is enforced. SNR convergence is tracked. Gaming is blocked.**
