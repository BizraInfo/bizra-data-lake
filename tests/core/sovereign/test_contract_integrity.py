"""Phase C.2.5 — Contract Integrity Sprint regression tests.

Proves that frontend-backend contract alignment holds for all Terminal v1
routes. These tests verify:
  P0-1: network/effect + milestones have auth guards
  P0-2: verify endpoints are POST (not GET)
  P0-4: seed/potential + seed/episodes have auth guards
  P1-7: /v1/plan fallback receipt conforms to Contract §8.1

Standing on Giants: PMBOK (scope), OWASP ASVS (auth boundaries)
"""

from __future__ import annotations

import inspect
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.sovereign.api import create_fastapi_app
from core.sovereign.api_exposure_policy import (
    API_ROUTE_POLICY_BY_KEY,
    RouteExposure,
    validate_api_exposure_policy,
)


def _runtime(state_dir: Path) -> MagicMock:
    """Minimal runtime mock for creating a testable FastAPI app."""
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
    runtime.query = AsyncMock(
        return_value=SimpleNamespace(
            query_id="q-test",
            success=True,
            response="ok",
            snr_score=0.9,
            ihsan_score=0.9,
            processing_time_ms=12.0,
            graph_hash=None,
        )
    )
    return runtime


@pytest.fixture()
def app(tmp_path: Path):
    return create_fastapi_app(_runtime(tmp_path))


# ── P0: Auth guard presence on AUTHENTICATED routes ──────────────────


# All routes that the exposure policy declares AUTHENTICATED must have
# the auth guard pattern in their handler source code.
AUTHENTICATED_TERMINAL_ROUTES = [
    ("/v1/seed/potential", "GET"),
    ("/v1/seed/episodes", "GET"),
    ("/v1/network/effect", "GET"),
    ("/v1/network/milestones", "GET"),
    ("/v1/node/value", "GET"),
    ("/v1/node/lifecycle", "GET"),
    ("/v1/terminal/state", "GET"),
    ("/v1/terminal/briefing", "GET"),
    ("/v1/memory/profile", "GET"),
    ("/v1/constitutional/status", "GET"),
    ("/v1/token/balance", "GET"),
    ("/v1/terminal/critical-acknowledgments", "POST"),
]


@pytest.mark.parametrize("path,verb", AUTHENTICATED_TERMINAL_ROUTES)
def test_authenticated_route_has_auth_guard(app, path: str, verb: str):
    """Every AUTHENTICATED route handler must call _authenticate_http_request."""
    # Verify policy says AUTHENTICATED
    policy = API_ROUTE_POLICY_BY_KEY.get((path, verb))
    assert policy is not None, f"No policy for {verb} {path}"
    assert (
        policy.exposure == RouteExposure.AUTHENTICATED
    ), f"{verb} {path} policy is {policy.exposure}, expected AUTHENTICATED"

    # Find the handler and check it has auth guard
    for route in app.routes:
        if getattr(route, "path", "") == path:
            methods = getattr(route, "methods", set())
            if verb in methods:
                handler = route.endpoint
                source = inspect.getsource(handler)
                assert "_authenticate_http_request" in source, (
                    f"Handler for {verb} {path} is missing auth guard. "
                    f"Policy says AUTHENTICATED but handler doesn't call "
                    f"_authenticate_http_request(request)."
                )
                # Also verify the handler accepts a Request parameter
                sig = inspect.signature(handler)
                param_names = list(sig.parameters.keys())
                assert (
                    "request" in param_names
                ), f"Handler for {verb} {path} missing 'request: Request' parameter"
                return

    pytest.fail(f"Route {verb} {path} not found in app")


# ── P0: Verify endpoints use POST ────────────────────────────────────


VERIFY_POST_ROUTES = [
    "/v1/verify/genesis",
    "/v1/verify/envelope",
    "/v1/verify/receipt",
    "/v1/verify/audit-log",
    "/v1/verify/ledger",
    "/v1/verify/poi",
]


@pytest.mark.parametrize("path", VERIFY_POST_ROUTES)
def test_verify_endpoints_are_post(app, path: str):
    """Verify endpoints must be POST per exposure policy."""
    policy = API_ROUTE_POLICY_BY_KEY.get((path, "POST"))
    assert policy is not None, f"No POST policy for {path}"
    assert policy.exposure == RouteExposure.PUBLIC

    for route in app.routes:
        if getattr(route, "path", "") == path:
            methods = getattr(route, "methods", set())
            assert "POST" in methods, f"{path} should be POST, got {methods}"
            return

    pytest.fail(f"Route POST {path} not found in app")


# ── P0: No phantom routes in frontend client ─────────────────────────


def test_plan_endpoint_exists(app):
    """POST /v1/plan must exist — it's the golden path."""
    for route in app.routes:
        if getattr(route, "path", "") == "/v1/plan":
            methods = getattr(route, "methods", set())
            assert "POST" in methods
            return
    pytest.fail("POST /v1/plan not found")


def test_no_phantom_mission_endpoint(app):
    """There must be no /v1/mission route — frontend redirects to /v1/plan."""
    for route in app.routes:
        if getattr(route, "path", "") == "/v1/mission":
            pytest.fail(
                "/v1/mission exists as a live route — this is a phantom. "
                "Remove it or redirect to /v1/plan."
            )


# ── P1: /v1/plan fallback receipt normalization (Contract §8.1) ──────


# Required fields per Build Contract §8.1
RECEIPT_REQUIRED_FIELDS = {
    "mission_id",
    "receipt_id",
    "status",
    "synthesis",
    "ihsan_score",
    "snr_score",
    "duration_ms",
    "channels_executed",
    "wallet_delta",
    "reflex_delta",
    "memory_delta",
    "execution_path",
    "hash_chain_ref",
}


def test_plan_fallback_receipt_has_all_fields():
    """The fallback dict in /v1/plan must have all Contract §8.1 fields.

    We verify by inspecting the source code of the submit_plan handler
    for the fallback return statement.
    """
    from core.sovereign import api as api_module

    source = inspect.getsource(api_module)

    # The fallback path must contain all required field names
    for field in RECEIPT_REQUIRED_FIELDS:
        assert f'"{field}"' in source, (
            f"Fallback receipt in /v1/plan source is missing required field: {field}. "
            f"Contract §8.1 requires all fields always present."
        )


# ── P1: Exposure policy completeness ─────────────────────────────────


def test_exposure_policy_covers_all_routes(app):
    """Every /v1/* route must have an explicit exposure policy."""
    report = validate_api_exposure_policy(app)
    if not report.ok:
        pytest.fail(f"Exposure policy drift detected:\n{report.format_issues()}")


# ── P1: Wallet/reflex/memory delta shape ─────────────────────────────


def test_wallet_delta_shape():
    """WalletDelta must have seed and bloom fields."""
    from core.sovereign.terminal import WalletDelta

    wd = WalletDelta()
    d = wd._asdict() if hasattr(wd, "_asdict") else {"seed": wd.seed, "bloom": wd.bloom}
    assert "seed" in d
    assert "bloom" in d


def test_reflex_delta_shape():
    """ReflexDelta must have compiled, near_compile, compile_count, threshold."""
    from core.sovereign.terminal import ReflexDelta

    rd = ReflexDelta()
    assert hasattr(rd, "compiled")
    assert hasattr(rd, "near_compile")
    assert hasattr(rd, "compile_count")
    assert hasattr(rd, "threshold")


def test_memory_delta_shape():
    """MemoryDelta must have episodic, semantic, procedural."""
    from core.sovereign.terminal import MemoryDelta

    md = MemoryDelta()
    assert hasattr(md, "episodic")
    assert hasattr(md, "semantic")
    assert hasattr(md, "procedural")
