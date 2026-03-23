#!/usr/bin/env python3
"""
CPU-Only Universality Baseline Benchmark
=========================================

Standing on Giants:
  البذرة §5 / Planning Principle §5 — Floor before ceiling
  Shannon (1948) — SNR on the universality baseline
  Deming (1950) — measure, then improve

This script measures the canonical mission path on CPU-only hardware
with no GPU, no external LLM, and no network dependency. The result
is the FLOOR proof: if Node0 works here, it works anywhere.

Usage:
    python scripts/ops/canonical_cpu_baseline.py [--json] [--warmup N]
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List


def _measure_boot(data_dir: Path) -> Dict[str, Any]:
    """Measure Node0 boot time (genesis ceremony)."""
    from core.node0.heartbeat import Node0Heartbeat

    start = time.perf_counter()
    hb = Node0Heartbeat(data_dir=data_dir, node_id="cpu-baseline-node")
    hb.boot()
    elapsed_ms = (time.perf_counter() - start) * 1000
    return {"operation": "boot", "duration_ms": round(elapsed_ms, 2)}


def _measure_cold_breathe(data_dir: Path) -> Dict[str, Any]:
    """Measure cold breathe (no pending receipts)."""
    from core.node0.heartbeat import Node0Heartbeat

    hb = Node0Heartbeat(data_dir=data_dir, node_id="cpu-baseline-node")
    hb.boot()

    start = time.perf_counter()
    receipt = hb.breathe()
    elapsed_ms = (time.perf_counter() - start) * 1000
    return {
        "operation": "cold_breathe",
        "duration_ms": round(elapsed_ms, 2),
        "ihsan_composite": round(receipt.ihsan_composite, 4),
    }


def _measure_warm_breathe(data_dir: Path, n: int = 5) -> Dict[str, Any]:
    """Measure warm breathe (repeated, measures cache warmth)."""
    from core.node0.heartbeat import Node0Heartbeat
    from core.sovereign.helix3 import Helix3Scheduler

    helix = Helix3Scheduler()
    hb = Node0Heartbeat(data_dir=data_dir, node_id="cpu-baseline-node")
    hb._helix3 = helix
    hb.boot()

    # Warm up
    for _ in range(2):
        hb.breathe()

    # Measure
    durations: List[float] = []
    for _ in range(n):
        start = time.perf_counter()
        hb.breathe()
        durations.append((time.perf_counter() - start) * 1000)

    return {
        "operation": "warm_breathe",
        "samples": n,
        "mean_ms": round(statistics.mean(durations), 2),
        "p50_ms": round(statistics.median(durations), 2),
        "p95_ms": round(
            sorted(durations)[int(n * 0.95)] if n >= 20 else max(durations), 2
        ),
        "min_ms": round(min(durations), 2),
        "max_ms": round(max(durations), 2),
    }


def _measure_mission_ingest(data_dir: Path, n: int = 10) -> Dict[str, Any]:
    """Measure mission ingest + breathe (the canonical path)."""
    from core.node0.heartbeat import Node0Heartbeat
    from core.sovereign.helix3 import Helix3Scheduler

    helix = Helix3Scheduler()
    hb = Node0Heartbeat(data_dir=data_dir, node_id="cpu-baseline-node")
    hb._helix3 = helix
    hb.boot()

    durations: List[float] = []
    composites: List[float] = []
    for i in range(n):
        # Ingest a mission receipt
        hb.ingest_mission_receipt(
            {
                "mission_id": f"bench-{i}",
                "description": f"CPU baseline mission #{i}",
                "ihsan_score": 0.96,
                "snr_score": 0.92,
                "fate_verdict": "approved",
                "gate_passed": True,
                "rewarded": True,
                "reward_amount": 1.0,
            }
        )

        start = time.perf_counter()
        receipt = hb.breathe()
        durations.append((time.perf_counter() - start) * 1000)
        composites.append(receipt.ihsan_composite)

    return {
        "operation": "mission_ingest_breathe",
        "samples": n,
        "mean_ms": round(statistics.mean(durations), 2),
        "p50_ms": round(statistics.median(durations), 2),
        "min_ms": round(min(durations), 2),
        "max_ms": round(max(durations), 2),
        "mean_ihsan": round(statistics.mean(composites), 4),
    }


def _measure_fate_rejection(data_dir: Path) -> Dict[str, Any]:
    """Measure that rejected receipts have zero economic effect."""
    from core.node0.heartbeat import Node0Heartbeat
    from core.sovereign.helix3 import Helix3Scheduler

    helix = Helix3Scheduler()
    hb = Node0Heartbeat(data_dir=data_dir, node_id="cpu-baseline-node")
    hb._helix3 = helix
    hb.boot()

    # Ingest 3 approved + 5 rejected
    for i in range(3):
        hb.ingest_mission_receipt(
            {
                "mission_id": f"ok-{i}",
                "description": f"Approved mission #{i}",
                "ihsan_score": 0.96,
                "fate_verdict": "approved",
                "gate_passed": True,
            }
        )
    for i in range(5):
        hb.ingest_mission_receipt(
            {
                "mission_id": f"rej-{i}",
                "description": f"Rejected mission #{i}",
                "ihsan_score": 0.2,
                "fate_verdict": "rejected",
                "gate_passed": False,
            }
        )

    start = time.perf_counter()
    receipt = hb.breathe()
    elapsed_ms = (time.perf_counter() - start) * 1000

    composite_clean = receipt.ihsan_composite >= 0.90
    return {
        "operation": "fate_rejection_proof",
        "duration_ms": round(elapsed_ms, 2),
        "ihsan_composite": round(receipt.ihsan_composite, 4),
        "composite_clean": composite_clean,
        "approved_count": receipt.helix_result.get("approved_count", 0),
        "rejected_count": receipt.helix_result.get("rejected_count", 0),
        "reflexes_precipitated": receipt.reflexes_precipitated,
        "seed_minted": receipt.seed_minted,
        "pass": composite_clean
        and receipt.reflexes_precipitated == 0
        and receipt.seed_minted == 0.0,
    }


def _measure_chain_integrity(data_dir: Path, n: int = 20) -> Dict[str, Any]:
    """Measure chain integrity over N breaths."""
    from core.node0.heartbeat import Node0Heartbeat

    hb = Node0Heartbeat(data_dir=data_dir, node_id="cpu-baseline-node")
    hb.boot()

    prev_hash = hb.chain_hash
    chain_valid = True
    for _ in range(n):
        receipt = hb.breathe()
        if receipt.prev_chain_hash != prev_hash:
            chain_valid = False
            break
        prev_hash = receipt.chain_hash

    return {
        "operation": "chain_integrity",
        "breaths": n,
        "chain_valid": chain_valid,
        "final_tick": hb.tick_number,
        "pass": chain_valid,
    }


def run_baseline(warmup: int = 2) -> Dict[str, Any]:
    """Run the full CPU baseline benchmark suite."""
    import tempfile

    results: List[Dict[str, Any]] = []

    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir) / "node0_state"
        data_dir.mkdir()

        # Warmup imports
        for _ in range(warmup):
            pass  # noqa: F811

        results.append(_measure_boot(data_dir))
        results.append(_measure_cold_breathe(data_dir))
        results.append(_measure_warm_breathe(data_dir, n=20))
        results.append(_measure_mission_ingest(data_dir, n=10))
        results.append(_measure_fate_rejection(data_dir))
        results.append(_measure_chain_integrity(data_dir, n=20))

    all_pass = all(r.get("pass", True) for r in results)
    return {
        "benchmark": "cpu_universality_baseline",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "platform": "cpu_only",
        "results": results,
        "all_pass": all_pass,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="CPU-only universality baseline")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--warmup", type=int, default=2, help="Warmup iterations")
    args = parser.parse_args()

    report = run_baseline(warmup=args.warmup)

    if args.json:
        print(json.dumps(report, indent=2))
    else:
        print("═" * 60)
        print("  BIZRA Node0 — CPU Universality Baseline")
        print("  Floor before ceiling (Planning Principle §5)")
        print("═" * 60)
        for r in report["results"]:
            op = r["operation"]
            if "duration_ms" in r:
                status = "✅" if r.get("pass", True) else "❌"
                print(f"  {status} {op:30s}  {r['duration_ms']:8.2f} ms")
            elif "mean_ms" in r:
                print(f"  ✅ {op:30s}  {r['mean_ms']:8.2f} ms (mean, n={r['samples']})")
            else:
                status = "✅" if r.get("pass", True) else "❌"
                print(
                    f"  {status} {op:30s}  PASS"
                    if r.get("pass")
                    else f"  {status} {op:30s}  FAIL"
                )
        print("─" * 60)
        verdict = "PASS ✅" if report["all_pass"] else "FAIL ❌"
        print(f"  Overall: {verdict}")
        print("═" * 60)

    sys.exit(0 if report["all_pass"] else 1)


if __name__ == "__main__":
    main()
