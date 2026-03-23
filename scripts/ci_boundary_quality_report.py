#!/usr/bin/env python3
"""Capture a runtime boundary-quality snapshot for CI and release scorecards."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "docs" / "evidence-pack" / "boundary_quality_report.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


class FailingInference:
    """Deterministic failing inference used to probe the boundary plane."""

    async def infer(self, prompt: str, **kwargs: Any) -> str:
        raise RuntimeError("ci boundary quality probe forced degradation")


async def capture_boundary_quality_report(
    *,
    persistence_dir: Path,
) -> dict[str, Any]:
    """Run a degraded mission/tick path and return the boundary-quality snapshot."""
    from core.node0.heartbeat import Node0Heartbeat
    from core.sovereign.organism import SovereignOrganism

    with patch.object(
        Node0Heartbeat,
        "_boot_federation_ambassador",
        autospec=True,
        return_value=None,
    ):
        organism = await SovereignOrganism.boot(
            inference=FailingInference(),
            persistence_dir=persistence_dir,
        )
    try:
        mission_receipt = await organism.mission("ci boundary quality probe")
        breath = await organism.tick()
        helix = dict(getattr(breath, "helix_result", {}) or {})
        node0 = dict((organism.stats or {}).get("node0", {}) or {})

        boundary_signal = {
            "boundary_error_receipts": int(
                helix.get("boundary_error_receipts", 0)
                or getattr(breath, "boundary_error_receipts", 0)
                or 0
            ),
            "boundary_halts": int(
                helix.get("boundary_halts", 0)
                or getattr(breath, "boundary_halts", 0)
                or 0
            ),
            "boundary_rejections": int(
                helix.get("boundary_rejections", 0)
                or getattr(breath, "boundary_rejections", 0)
                or 0
            ),
            "boundary_degradations": int(
                helix.get("boundary_degradations", 0)
                or getattr(breath, "boundary_degradations", 0)
                or 0
            ),
            "boundary_retries": int(
                helix.get("boundary_retries", 0)
                or getattr(breath, "boundary_retries", 0)
                or 0
            ),
            "pre_boundary_ihsan_composite": float(
                helix.get(
                    "pre_boundary_ihsan_composite",
                    getattr(breath, "ihsan_composite", 0.0),
                )
                or 0.0
            ),
            "post_boundary_ihsan_composite": float(
                getattr(breath, "ihsan_composite", 0.0) or 0.0
            ),
            "boundary_quality_multiplier": float(
                helix.get("boundary_quality_multiplier", 1.0) or 1.0
            ),
        }
        gate_checks = {
            "degradation_receipt_emitted": bool(
                (mission_receipt.metadata or {}).get("degradation_receipts")
            ),
            "boundary_receipt_count_positive": boundary_signal[
                "boundary_error_receipts"
            ]
            >= 1,
            "boundary_multiplier_applied": boundary_signal[
                "boundary_quality_multiplier"
            ]
            < 1.0,
            "pre_boundary_not_below_post_boundary": boundary_signal[
                "pre_boundary_ihsan_composite"
            ]
            >= boundary_signal["post_boundary_ihsan_composite"],
            "penalty_is_bounded": 0.5
            <= boundary_signal["boundary_quality_multiplier"]
            <= 1.0,
        }

        return {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "probe": {
                "mission_id": mission_receipt.mission_id,
                "system": mission_receipt.system,
                "fate_verdict": mission_receipt.fate_verdict,
                "gate_passed": mission_receipt.gate_passed,
            },
            "boundary_signal": boundary_signal,
            "node0": {
                "total_boundary_error_receipts": int(
                    node0.get("total_boundary_error_receipts", 0) or 0
                ),
                "last_breath_boundary_error_receipts": int(
                    node0.get("last_breath_boundary_error_receipts", 0) or 0
                ),
                "last_breath_boundary_retries": int(
                    node0.get("last_breath_boundary_retries", 0) or 0
                ),
                "last_breath_boundary_degradations": int(
                    node0.get("last_breath_boundary_degradations", 0) or 0
                ),
            },
            "gate_verdict": {
                "passed": all(gate_checks.values()),
                "checks": gate_checks,
            },
        }
    finally:
        await organism.shutdown()


def render_boundary_summary(report: dict[str, Any]) -> str:
    """Render a concise markdown summary of the boundary-quality snapshot."""
    signal = report["boundary_signal"]
    verdict = report["gate_verdict"]
    lines = [
        "## Boundary Quality Probe",
        "",
        f"- Verdict: {'PASS' if verdict['passed'] else 'FAIL'}",
        "",
        "| Metric | Value |",
        "| --- | ---: |",
        f"| boundary_error_receipts | {signal['boundary_error_receipts']} |",
        f"| boundary_degradations | {signal['boundary_degradations']} |",
        f"| boundary_retries | {signal['boundary_retries']} |",
        (
            f"| pre_boundary_ihsan_composite | "
            f"{signal['pre_boundary_ihsan_composite']:.4f} |"
        ),
        (
            f"| post_boundary_ihsan_composite | "
            f"{signal['post_boundary_ihsan_composite']:.4f} |"
        ),
        (
            f"| boundary_quality_multiplier | "
            f"{signal['boundary_quality_multiplier']:.4f} |"
        ),
        "",
    ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Capture boundary quality report.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Path to write the JSON report",
    )
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="bizra-boundary-quality-") as tmp_dir:
        report = asyncio.run(
            capture_boundary_quality_report(
                persistence_dir=Path(tmp_dir) / "sovereign-boundary-quality",
            )
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(render_boundary_summary(report), end="")

    return 0 if report["gate_verdict"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
