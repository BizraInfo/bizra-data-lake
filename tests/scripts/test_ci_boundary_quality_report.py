from __future__ import annotations

import asyncio
import json
from pathlib import Path

from scripts.ci_boundary_quality_report import (
    capture_boundary_quality_report,
    render_boundary_summary,
)


def test_capture_boundary_quality_report_emits_penalty_signal(tmp_path: Path) -> None:
    report = asyncio.run(
        capture_boundary_quality_report(
            persistence_dir=tmp_path / "sovereign-boundary-quality",
        )
    )

    assert report["gate_verdict"]["passed"] is True
    signal = report["boundary_signal"]
    assert signal["boundary_error_receipts"] >= 1
    assert signal["boundary_quality_multiplier"] < 1.0
    assert (
        signal["pre_boundary_ihsan_composite"]
        >= signal["post_boundary_ihsan_composite"]
    )


def test_render_boundary_summary_includes_key_metrics(tmp_path: Path) -> None:
    report = {
        "boundary_signal": {
            "boundary_error_receipts": 2,
            "boundary_degradations": 1,
            "boundary_retries": 1,
            "pre_boundary_ihsan_composite": 0.95,
            "post_boundary_ihsan_composite": 0.91,
            "boundary_quality_multiplier": 0.96,
        },
        "gate_verdict": {
            "passed": True,
        },
    }

    markdown = render_boundary_summary(report)

    assert "## Boundary Quality Probe" in markdown
    assert "boundary_quality_multiplier" in markdown
    assert "0.9600" in markdown


def test_report_is_json_serializable(tmp_path: Path) -> None:
    report = asyncio.run(
        capture_boundary_quality_report(
            persistence_dir=tmp_path / "sovereign-boundary-quality",
        )
    )
    payload = json.dumps(report)
    assert "boundary_signal" in payload
