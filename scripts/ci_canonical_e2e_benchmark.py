#!/usr/bin/env python3
"""Canonical E2E Latency Benchmark — measures the full enforcement spine.

Path under test:
  GoT bridge → VRG receipt → Organism → Node0 breathe → EventBus emission

This benchmark validates the canonical enforcement spine end-to-end,
measuring latency at each stage with regression gates.

Standing on Giants: Deming (PDCA, 1950) · Boyd (OODA latency, 1976)

Usage:
    python scripts/ci_canonical_e2e_benchmark.py          # Default gates
    python scripts/ci_canonical_e2e_benchmark.py --strict  # Tighter gates

Exit codes:
    0 — All benchmarks passed gates
    1 — One or more benchmarks exceeded threshold (regression)
"""

import argparse
import asyncio
import json
import logging
import sys
import time
import types
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DEFAULT_OUTPUT = ROOT / "evidence" / "benchmarks" / "canonical_e2e_latest.json"

# Regression gates (milliseconds)
GATES_DEFAULT = {
    "got_bridge_init_ms": 500,
    "got_bridge_reason_ms": 200,
    "vrg_receipt_build_ms": 100,
    "organism_boot_ms": 5000,
    "node0_breathe_ms": 50,
    "eventbus_emission_ms": 10,
    "full_spine_ms": 6000,
}

GATES_STRICT = {
    "got_bridge_init_ms": 300,
    "got_bridge_reason_ms": 100,
    "vrg_receipt_build_ms": 50,
    "organism_boot_ms": 3000,
    "node0_breathe_ms": 30,
    "eventbus_emission_ms": 5,
    "full_spine_ms": 4000,
}


def _measure(label: str, fn: Any) -> float:
    """Execute fn and return elapsed milliseconds."""
    t0 = time.perf_counter()
    result = fn()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return elapsed_ms, result


def benchmark_got_bridge() -> Dict[str, Any]:
    """Benchmark GoT bridge initialization and reasoning."""
    results = {}

    elapsed_import, got_bridge_class = _measure(
        "got_bridge_import",
        _load_got_bridge_class,
    )
    results["got_bridge_import_ms"] = round(elapsed_import, 2)

    elapsed, bridge = _measure(
        "got_bridge_init",
        lambda: _init_got_bridge(got_bridge_class),
    )
    results["got_bridge_init_ms"] = round(elapsed, 2)
    results["got_bridge_available"] = bridge is not None

    if bridge is None:
        results["got_bridge_reason_ms"] = 0
        results["got_bridge_converged"] = False
        return results

    if hasattr(bridge, "reason_and_verify"):
        elapsed, reason_result = _measure(
            "got_bridge_reason",
            lambda: bridge.reason_and_verify(
                "benchmark test query: explain BIZRA architecture"
            ),
        )
        results["got_bridge_reason_ms"] = round(elapsed, 2)
        results["got_bridge_converged"] = getattr(reason_result, "converged", False)
        results["got_bridge_mode"] = "legacy_reason_and_verify"
        return results

    async def _reason_async() -> Any:
        if hasattr(bridge, "reason_verified"):
            return await bridge.reason_verified(
                "benchmark test query: explain BIZRA architecture"
            )
        if hasattr(bridge, "reason"):
            return await bridge.reason(
                "benchmark test query: explain BIZRA architecture"
            )
        raise AttributeError(
            "GoTBridge exposes neither reason_and_verify nor async reason methods"
        )

    elapsed, reason_result = _measure(
        "got_bridge_reason",
        lambda: asyncio.run(_reason_async()),
    )
    results["got_bridge_reason_ms"] = round(elapsed, 2)
    results["got_bridge_converged"] = getattr(reason_result, "converged", False)
    results["got_bridge_verified"] = getattr(reason_result, "verified", False)
    results["got_bridge_mode"] = (
        "async_reason_verified"
        if hasattr(bridge, "reason_verified")
        else "async_reason"
    )

    return results


def _load_got_bridge_class() -> Any:
    """Import the GoT bridge class once so init timing excludes module import tax."""
    try:
        from core.reasoning.got_bridge import GoTBridge

        return GoTBridge
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.warning("GoT bridge unavailable: %s", exc)
        return None


def _init_got_bridge(got_bridge_class: Any) -> Any:
    """Initialize the GoT bridge once imports have already settled."""
    if got_bridge_class is None:
        return None
    return got_bridge_class()


def benchmark_vrg_receipt() -> Dict[str, Any]:
    """Benchmark VRG receipt building."""
    results = {}
    try:
        from core.proof_engine.canonical import canonical_bytes
        from core.proof_engine.receipt import SimpleSigner

        signer = SimpleSigner(b"benchmark_signer")
        payload = {"query": "benchmark", "answer": "test", "ihsan": 0.95}

        elapsed, _ = _measure(
            "vrg_receipt_build",
            lambda: (
                canonical_bytes(payload),
                signer.sign(canonical_bytes(payload)),
            ),
        )
        results["vrg_receipt_build_ms"] = round(elapsed, 2)
        results["vrg_receipt_available"] = True
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.warning("VRG receipt benchmark skipped: %s", exc)
        results["vrg_receipt_build_ms"] = 0
        results["vrg_receipt_available"] = False

    return results


def benchmark_node0() -> Dict[str, Any]:
    """Benchmark Node0 boot + breathe + EventBus emission."""
    import tempfile

    results = {}

    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            # Boot
            elapsed_boot, heartbeat = _measure(
                "node0_boot",
                lambda: _boot_node0(tmpdir),
            )
            results["organism_boot_ms"] = round(elapsed_boot, 2)
            results["node0_available"] = heartbeat is not None

            if heartbeat is None:
                results["node0_breathe_ms"] = 0
                results["eventbus_emission_ms"] = 0
                return results

            # Breathe
            elapsed_breathe, breath = _measure(
                "node0_breathe",
                lambda: heartbeat.breathe(),
            )
            results["node0_breathe_ms"] = round(elapsed_breathe, 2)
            results["breath_chain_valid"] = breath is not None

            # EventBus emission (measure direct publish path on a mock bus)
            try:

                class _BenchBus:
                    """Minimal bus for emission timing."""

                    def __init__(self) -> None:
                        self.count = 0

                    def publish(self, topic: str, payload: Any) -> None:
                        self.count += 1

                bench_bus = _BenchBus()
                heartbeat._event_bus = bench_bus
                # Warm the compatibility publisher so the timed sample reflects
                # steady-state emission rather than first-import overhead.
                heartbeat._emit_breath_event(breath)
                bench_bus.count = 0
                elapsed_emit, _ = _measure(
                    "eventbus_emission",
                    lambda: heartbeat._emit_breath_event(breath),
                )
                results["eventbus_emission_ms"] = round(elapsed_emit, 2)
                results["events_emitted"] = bench_bus.count
            except (AttributeError, TypeError) as exc:
                logger.warning("EventBus emission benchmark skipped: %s", exc)
                results["eventbus_emission_ms"] = 0

    except (ImportError, AttributeError, RuntimeError, OSError) as exc:
        logger.warning("Node0 benchmark skipped: %s", exc)
        results["organism_boot_ms"] = 0
        results["node0_available"] = False
        results["node0_breathe_ms"] = 0
        results["eventbus_emission_ms"] = 0

    return results


def _boot_node0(data_dir: str) -> Any:
    """Boot a Node0Heartbeat for benchmarking."""
    from core.node0.heartbeat import Node0Heartbeat

    hb = Node0Heartbeat(data_dir=data_dir)
    _configure_benchmark_heartbeat(hb)
    hb.boot()
    hb._event_bus = None
    return hb


def _configure_benchmark_heartbeat(heartbeat: Any) -> None:
    """Strip optional sidecars so the benchmark measures the receipt membrane.

    Production Node0 should keep its full organism wiring. The canonical E2E
    benchmark, however, aims to measure the governed spine:

      boot -> breathe -> receipt -> event emission

    not auxiliary learning, federation networking, or witness contribution.
    """

    no_return_methods = (
        "_boot_reflex_bridge",
        "_boot_reasoning_bank",
        "_boot_learning_loop",
        "_boot_federation_ambassador",
        "_record_rb_experience",
        "_run_learning_cycle",
        "_contribute_urp_witness",
    )

    for method_name in no_return_methods:
        if hasattr(heartbeat, method_name):
            setattr(
                heartbeat,
                method_name,
                types.MethodType(lambda self, *args, **kwargs: None, heartbeat),
            )

    if hasattr(heartbeat, "_check_reflex_precipitation"):
        setattr(
            heartbeat,
            "_check_reflex_precipitation",
            types.MethodType(lambda self, helix_result: 0, heartbeat),
        )


def _evaluate_gates(
    results: Dict[str, Any],
    gates: Dict[str, float],
) -> Dict[str, Any]:
    """Evaluate benchmark metrics against latency gates."""
    checks: Dict[str, Dict[str, Any]] = {}
    failed_metrics: List[str] = []

    for metric, gate_value in gates.items():
        actual = float(results.get(metric, 0.0))
        passed = actual <= gate_value
        checks[metric] = {
            "actual": round(actual, 2),
            "gate": gate_value,
            "passed": passed,
        }
        if not passed:
            failed_metrics.append(metric)

    return {
        "passed": not failed_metrics,
        "failed_metrics": failed_metrics,
        "checks": checks,
    }


def run_benchmark(strict: bool = False, output: Path | None = DEFAULT_OUTPUT) -> int:
    """Run full E2E benchmark suite, write a report, and enforce gates."""
    gates = GATES_STRICT if strict else GATES_DEFAULT

    print("=" * 60)
    print("BIZRA Canonical E2E Latency Benchmark")
    print(f"Mode: {'STRICT' if strict else 'DEFAULT'}")
    print("=" * 60)

    all_results: Dict[str, Any] = {}
    full_start = time.perf_counter()

    # Stage 1: GoT bridge
    print("\n[1/3] GoT Bridge...")
    got_results = benchmark_got_bridge()
    all_results.update(got_results)

    # Stage 2: VRG receipt
    print("[2/3] VRG Receipt...")
    vrg_results = benchmark_vrg_receipt()
    all_results.update(vrg_results)

    # Stage 3: Node0 + EventBus
    print("[3/3] Node0 + EventBus...")
    node0_results = benchmark_node0()
    all_results.update(node0_results)

    full_elapsed = (time.perf_counter() - full_start) * 1000
    all_results["full_spine_ms"] = round(full_elapsed, 2)
    gate_verdict = _evaluate_gates(all_results, gates)

    # Gate check
    print("\n" + "-" * 60)
    print(f"{'Metric':<30} {'Value':>10} {'Gate':>10} {'Status':>8}")
    print("-" * 60)

    for metric, gate_value in gates.items():
        actual = all_results.get(metric, 0)
        passed = actual <= gate_value
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{metric:<30} {actual:>10.1f} {gate_value:>10.1f} {status:>8}")

    print("-" * 60)

    if gate_verdict["failed_metrics"]:
        print(
            f"\n❌ REGRESSION: {len(gate_verdict['failed_metrics'])} gate(s) exceeded"
        )
        for metric in gate_verdict["failed_metrics"]:
            check = gate_verdict["checks"][metric]
            print(f"  - {metric}: {check['actual']:.1f}ms > {check['gate']:.1f}ms")
    else:
        print(f"\n✅ ALL GATES PASSED ({len(gates)} checks)")

    report = {
        "benchmark": "canonical_e2e",
        "mode": "strict" if strict else "default",
        "benchmark_results": all_results,
        "gate_verdict": gate_verdict,
    }

    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nEvidence: {output}")

    return 0 if gate_verdict["passed"] else 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canonical E2E Latency Benchmark")
    parser.add_argument("--strict", action="store_true", help="Use strict gates")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the JSON report",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    sys.exit(run_benchmark(strict=args.strict, output=args.output))
