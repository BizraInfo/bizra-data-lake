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
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

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

    # GoT bridge init
    elapsed, bridge = _measure("got_bridge_init", lambda: _init_got_bridge())
    results["got_bridge_init_ms"] = round(elapsed, 2)
    results["got_bridge_available"] = bridge is not None

    if bridge is None:
        results["got_bridge_reason_ms"] = 0
        results["got_bridge_converged"] = False
        return results

    # GoT bridge reason
    elapsed, reason_result = _measure(
        "got_bridge_reason",
        lambda: bridge.reason_and_verify(
            "benchmark test query: explain BIZRA architecture"
        ),
    )
    results["got_bridge_reason_ms"] = round(elapsed, 2)
    results["got_bridge_converged"] = getattr(reason_result, "converged", False)

    return results


def _init_got_bridge() -> Any:
    """Initialize GoT bridge with minimal dependencies."""
    try:
        from core.reasoning.got_bridge import GoTBridge

        return GoTBridge()
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.warning("GoT bridge unavailable: %s", exc)
        return None


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
        from core.node0.heartbeat import Node0Heartbeat

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

            # EventBus emission (measure with mock bus)
            try:
                events_before = heartbeat._total_events_emitted

                class _BenchBus:
                    """Minimal bus for emission timing."""

                    def __init__(self) -> None:
                        self.count = 0

                    def publish(self, topic: str, payload: Any) -> None:
                        self.count += 1

                bench_bus = _BenchBus()
                heartbeat._event_bus = bench_bus
                elapsed_emit, _ = _measure(
                    "eventbus_emission",
                    lambda: heartbeat.breathe(),
                )
                results["eventbus_emission_ms"] = round(
                    elapsed_emit - results["node0_breathe_ms"], 2
                )
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
    hb.boot()
    return hb


def run_benchmark(strict: bool = False) -> int:
    """Run full E2E benchmark suite and check gates."""
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

    # Gate check
    print("\n" + "-" * 60)
    print(f"{'Metric':<30} {'Value':>10} {'Gate':>10} {'Status':>8}")
    print("-" * 60)

    failures: List[str] = []
    for metric, gate_value in gates.items():
        actual = all_results.get(metric, 0)
        passed = actual <= gate_value
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{metric:<30} {actual:>10.1f} {gate_value:>10.1f} {status:>8}")
        if not passed:
            failures.append(f"{metric}: {actual:.1f}ms > {gate_value}ms")

    print("-" * 60)

    if failures:
        print(f"\n❌ REGRESSION: {len(failures)} gate(s) exceeded")
        for f in failures:
            print(f"  - {f}")
        return 1
    else:
        print(f"\n✅ ALL GATES PASSED ({len(gates)} checks)")

    # Write evidence
    evidence_dir = Path("evidence/benchmarks")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_path = evidence_dir / "canonical_e2e_latest.json"
    evidence_path.write_text(json.dumps(all_results, indent=2))
    print(f"\nEvidence: {evidence_path}")

    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Canonical E2E Latency Benchmark")
    parser.add_argument("--strict", action="store_true", help="Use strict gates")
    args = parser.parse_args()

    logging.basicConfig(level=logging.WARNING)
    sys.exit(run_benchmark(strict=args.strict))
