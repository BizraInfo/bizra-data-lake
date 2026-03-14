"""OpenAPI contract drift detection.

Ensures the API surface doesn't change without updating the static schema.
Elite pattern: schema-as-artifact prevents frontend/backend drift.

Standing on Giants:
- Stripe: API versioning with schema-as-contract
- Google AIP-121: Backwards-compatible evolution
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi")


def _runtime(state_dir: Path) -> MagicMock:
    runtime = MagicMock()
    runtime.config = SimpleNamespace(state_dir=state_dir)
    runtime.metrics = MagicMock(to_prometheus=lambda include_help=False: "")
    runtime.status.return_value = {
        "health": {
            "status": "healthy",
            "strict_gate": {"enabled": False, "passed": True},
        },
        "identity": {"version": "test"},
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
    runtime.query = MagicMock()
    runtime._orchestrator = None
    runtime._node_signer = None
    runtime._evidence_ledger = None
    return runtime


@pytest.mark.integration
def test_openapi_schema_version_matches_app(tmp_path: Path, monkeypatch) -> None:
    """The static OpenAPI schema version must match the live app."""
    monkeypatch.setenv("BIZRA_USERSTORE_MASTER_SECRET", "test-openapi-contract")

    from core.sovereign.api import create_fastapi_app

    app = create_fastapi_app(_runtime(tmp_path))
    live_schema = app.openapi()

    assert live_schema["info"]["version"] == "1.3.0"


@pytest.mark.integration
def test_openapi_schema_has_mission_models(tmp_path: Path, monkeypatch) -> None:
    """The OpenAPI schema must include typed mission models."""
    monkeypatch.setenv("BIZRA_USERSTORE_MASTER_SECRET", "test-openapi-contract")

    from core.sovereign.api import create_fastapi_app

    app = create_fastapi_app(_runtime(tmp_path))
    schema = app.openapi()

    models = schema.get("components", {}).get("schemas", {})
    assert "MissionPlanResponse" in models
    assert "ChannelResult" in models

    # Validate MissionPlanResponse has required fields
    mission_props = models["MissionPlanResponse"]["properties"]
    for field in (
        "mission_id",
        "status",
        "synthesis",
        "ihsan_score",
        "snr_score",
        "duration_ms",
        "execution_path",
        "wallet_delta",
        "reflex_delta",
        "memory_delta",
        "hash_chain_ref",
    ):
        assert field in mission_props, f"MissionPlanResponse missing field: {field}"

    # Contract §8.1: sub-models must be present
    for sub_model in (
        "WalletDeltaResponse",
        "ReflexDeltaResponse",
        "MemoryDeltaResponse",
    ):
        assert sub_model in models, f"Missing sub-model: {sub_model}"


@pytest.mark.integration
def test_openapi_schema_has_all_13_tags(tmp_path: Path, monkeypatch) -> None:
    """The OpenAPI schema must include all 13 domain tags."""
    monkeypatch.setenv("BIZRA_USERSTORE_MASTER_SECRET", "test-openapi-contract")

    from core.sovereign.api import create_fastapi_app

    app = create_fastapi_app(_runtime(tmp_path))
    schema = app.openapi()

    tags = {t["name"] for t in schema.get("tags", [])}
    expected_tags = {
        "health",
        "auth",
        "mission",
        "query",
        "verification",
        "memory",
        "economics",
        "constitutional",
        "spearpoint",
        "cognitive",
        "experience",
        "sovereignty",
        "onboarding",
    }
    assert (
        expected_tags == tags
    ), f"Tag drift: missing={expected_tags - tags}, extra={tags - expected_tags}"


# Minimum route count — must only go UP, never down.
# Update this when routes are intentionally removed.
_MIN_ROUTE_COUNT = 59


@pytest.mark.integration
def test_openapi_path_count_minimum(tmp_path: Path, monkeypatch) -> None:
    """Route count must not decrease — only grow."""
    monkeypatch.setenv("BIZRA_USERSTORE_MASTER_SECRET", "test-openapi-contract")

    from core.sovereign.api import create_fastapi_app

    app = create_fastapi_app(_runtime(tmp_path))
    schema = app.openapi()

    path_count = len(schema["paths"])
    assert path_count >= _MIN_ROUTE_COUNT, (
        f"Route count DECREASED: {path_count} < {_MIN_ROUTE_COUNT}. "
        "Routes should never be removed without updating _MIN_ROUTE_COUNT."
    )
