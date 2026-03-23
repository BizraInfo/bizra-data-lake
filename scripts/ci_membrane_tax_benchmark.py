#!/usr/bin/env python3
"""CMN Membrane Tax Benchmark.

Builds on the canonical E2E benchmark and separates governance overhead from
task work so CMN claims can be framed as systems evidence rather than rhetoric.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

try:
    import resource
except ImportError:  # pragma: no cover - platform guard
    resource = None  # type: ignore[assignment]

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.ci_canonical_e2e_benchmark import (
    benchmark_got_bridge,
    benchmark_node0,
    benchmark_vrg_receipt,
)

logger = logging.getLogger(__name__)
DEFAULT_OUTPUT = ROOT / "docs" / "evidence-pack" / "cmn_membrane_tax_report.json"

GATES_DEFAULT = {
    "governance_tax_ms": 250.0,
    "governance_tax_ratio": 0.35,
    "rss_growth_mb": 512.0,
}

GATES_STRICT = {
    "governance_tax_ms": 100.0,
    "governance_tax_ratio": 0.20,
    "rss_growth_mb": 256.0,
}

NON_NEGATIVE_METRICS = (
    "vrg_receipt_build_ms",
    "eventbus_emission_ms",
    "got_bridge_reason_ms",
    "node0_breathe_ms",
)


def _rss_mb() -> float:
    if resource is None:
        return 0.0
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(float(usage) / 1024.0, 2)


def _measure_stage(label: str, fn: Any) -> dict[str, Any]:
    rss_before = _rss_mb()
    started = time.perf_counter()
    result = fn()
    elapsed_ms = (time.perf_counter() - started) * 1000
    rss_after = _rss_mb()
    return {
        "label": label,
        "elapsed_ms": round(elapsed_ms, 2),
        "rss_before_mb": rss_before,
        "rss_after_mb": rss_after,
        "rss_growth_mb": round(max(rss_after - rss_before, 0.0), 2),
        "result": result,
    }


def _benchmark_got_bridge_compatible() -> dict[str, Any]:
    """Benchmark GoT bridge across legacy and async bridge interfaces."""
    try:
        return benchmark_got_bridge()
    except AttributeError as exc:
        logger.warning(
            "Canonical GoT benchmark interface mismatch, falling back to async bridge benchmark: %s",
            exc,
        )

    try:
        from core.reasoning.got_bridge import GoTBridge
    except (ImportError, AttributeError, RuntimeError) as exc:
        logger.warning("GoT bridge unavailable for membrane tax benchmark: %s", exc)
        return {
            "got_bridge_init_ms": 0.0,
            "got_bridge_available": False,
            "got_bridge_reason_ms": 0.0,
            "got_bridge_converged": False,
            "got_bridge_mode": "unavailable",
        }

    started = time.perf_counter()
    bridge = GoTBridge()
    init_ms = (time.perf_counter() - started) * 1000
    results = {
        "got_bridge_init_ms": round(init_ms, 2),
        "got_bridge_available": bridge is not None,
        "got_bridge_reason_ms": 0.0,
        "got_bridge_converged": False,
        "got_bridge_mode": "async_reason",
    }

    if bridge is None:
        return results

    async def _reason() -> Any:
        if hasattr(bridge, "reason_verified"):
            results["got_bridge_mode"] = "async_reason_verified"
            return await bridge.reason_verified(
                "benchmark test query: explain BIZRA architecture"
            )
        if hasattr(bridge, "reason"):
            return await bridge.reason(
                "benchmark test query: explain BIZRA architecture"
            )
        raise AttributeError("GoTBridge exposes neither reason_verified nor reason")

    try:
        started = time.perf_counter()
        reason_result = asyncio.run(_reason())
        results["got_bridge_reason_ms"] = round(
            (time.perf_counter() - started) * 1000,
            2,
        )
        results["got_bridge_converged"] = bool(
            getattr(reason_result, "converged", False)
        )
        results["got_bridge_verified"] = bool(getattr(reason_result, "verified", False))
    except (
        asyncio.CancelledError,
        AttributeError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        logger.warning("Async GoT bridge benchmark degraded: %s", exc)

    return results


def _normalize_benchmark_results(
    results: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, float]]]:
    normalized = dict(results)
    clamped_negative_metrics: dict[str, float] = {}

    for metric in NON_NEGATIVE_METRICS:
        value = float(normalized.get(metric, 0.0))
        if value < 0:
            clamped_negative_metrics[metric] = round(value, 4)
            normalized[metric] = 0.0

    return normalized, {"clamped_negative_metrics": clamped_negative_metrics}


def compute_membrane_tax(report: dict[str, Any]) -> dict[str, float]:
    results = report["benchmark_results"]
    governance_ms = round(
        float(results.get("vrg_receipt_build_ms", 0.0))
        + float(results.get("eventbus_emission_ms", 0.0)),
        2,
    )
    work_ms = round(
        float(results.get("got_bridge_reason_ms", 0.0))
        + float(results.get("node0_breathe_ms", 0.0)),
        2,
    )
    total = governance_ms + work_ms
    ratio = round(governance_ms / total, 4) if total > 0 else 0.0
    rss_growth_mb = round(
        sum(float(stage.get("rss_growth_mb", 0.0)) for stage in report["stages"]),
        2,
    )
    return {
        "governance_tax_ms": governance_ms,
        "task_work_ms": work_ms,
        "governance_tax_ratio": ratio,
        "rss_growth_mb": rss_growth_mb,
    }


def evaluate_report(report: dict[str, Any], *, strict: bool = False) -> dict[str, Any]:
    gates = GATES_STRICT if strict else GATES_DEFAULT
    summary = report["membrane_tax"]
    checks = {}
    failed = []
    for metric, gate in gates.items():
        actual = float(summary.get(metric, 0.0))
        passed = actual <= gate
        checks[metric] = {
            "actual": round(actual, 4),
            "gate": gate,
            "passed": passed,
        }
        if not passed:
            failed.append(metric)
    return {
        "strict": strict,
        "passed": not failed,
        "failed_metrics": failed,
        "checks": checks,
    }


def build_report(*, strict: bool = False) -> dict[str, Any]:
    stages = [
        _measure_stage("got_bridge", _benchmark_got_bridge_compatible),
        _measure_stage("vrg_receipt", benchmark_vrg_receipt),
        _measure_stage("node0", benchmark_node0),
    ]

    raw_benchmark_results: dict[str, Any] = {}
    for stage in stages:
        raw_benchmark_results.update(stage["result"])

    benchmark_results, benchmark_sanity = _normalize_benchmark_results(
        raw_benchmark_results
    )

    report = {
        "benchmark": "cmn_membrane_tax",
        "mode": "strict" if strict else "default",
        "stages": stages,
        "raw_benchmark_results": raw_benchmark_results,
        "benchmark_results": benchmark_results,
        "benchmark_sanity": benchmark_sanity,
    }
    report["membrane_tax"] = compute_membrane_tax(report)
    report["gate_verdict"] = evaluate_report(report, strict=strict)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the CMN membrane tax benchmark.")
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Apply strict governance tax gates",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the JSON report",
    )
    args = parser.parse_args()

    report = build_report(strict=args.strict)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["gate_verdict"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
