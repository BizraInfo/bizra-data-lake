#!/usr/bin/env python3
"""
Fail CI when the live FastAPI `/v1/*` surface drifts from the reviewed route
exposure contract.

This keeps exposure decisions in code review instead of buried in docs.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.sovereign.api import create_fastapi_app
from core.sovereign.api_exposure_policy import (
    RouteExposure,
    summarize_api_exposure,
    validate_api_exposure_policy,
)


def _runtime(state_dir: Path) -> MagicMock:
    runtime = MagicMock()
    runtime.config = SimpleNamespace(state_dir=state_dir)
    runtime.metrics = MagicMock(to_prometheus=lambda include_help=False: "")
    runtime.status.return_value = {
        "health": {
            "status": "healthy",
            "strict_gate": {"enabled": False, "passed": True},
        },
        "identity": {"version": "ci-gate"},
        "state": {"running": True},
        "autonomous": {"running": False},
        "pat_sat": {
            "negotiation_receipt_chain": {
                "verified_end_to_end": False,
                "chain_valid": None,
                "total_negotiation_receipts": 0,
                "latest_sequence": None,
                "latest_entry_hash": None,
                "latest_receipt_id": None,
            }
        },
    }
    runtime.query = AsyncMock(
        return_value=SimpleNamespace(
            query_id="q-ci",
            success=True,
            response="ok",
            snr_score=0.9,
            ihsan_score=0.95,
            processing_time_ms=1.0,
            graph_hash=None,
        )
    )
    runtime._orchestrator = None
    runtime._node_signer = None
    runtime._evidence_ledger = None
    return runtime


def main() -> int:
    os.environ.setdefault(
        "BIZRA_USERSTORE_MASTER_SECRET",
        "ci-api-exposure-contract-master-secret",
    )

    with TemporaryDirectory() as tmp:
        app = create_fastapi_app(_runtime(Path(tmp)))

    report = validate_api_exposure_policy(app)
    if not report.ok:
        print("[API-EXPOSURE-GATE] FAILED")
        print(report.format_issues())
        return 1

    summary = summarize_api_exposure(app)
    total_routes = sum(summary.values())
    print("[API-EXPOSURE-GATE] PASS")
    print(f"Validated {total_routes} route bindings.")
    for exposure in (
        RouteExposure.AUTHENTICATED,
        RouteExposure.PUBLIC,
        RouteExposure.BOOTSTRAP_PUBLIC,
    ):
        print(f"  - {exposure.value}: {summary.get(exposure, 0)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
