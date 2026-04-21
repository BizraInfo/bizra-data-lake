"""Cross-layer wire-contract guard for /v1/auth/register and /v1/auth/login.

Sprint A.2 (2026-04-21) — Auth Contract Alignment.

The backend (``core/sovereign/api.py``) is the sealed source of truth:
- ``/v1/auth/register`` requires ``{username, email, password, accept_covenant}``
  and returns ``{user: {...}, tokens: {access_token, refresh_token,
  token_type, expires_in}}``.
- ``/v1/auth/login`` requires ``{username, password}`` and returns
  ``{user_id, username, tokens: {...}}``.

This guard pins the exact wire shape so that the frontend TypeScript types
at ``frontend/src/types.ts`` (``RegisterResponse`` / ``LoginResponse`` /
``AuthTokens`` / ``UserProfile``) stay bound to the backend contract. If
either side drifts, this test fails loudly.

Prior drift that this guard prevents: before Sprint A.2, the frontend sent
``{username, password, name}`` to ``/v1/auth/register`` (no email, no
covenant) and expected a flat ``{token, node_id, expires_at}`` response.
Registration would have crashed at runtime against the real backend.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi", reason="fastapi not installed in CI base env")
pytest.importorskip("starlette", reason="starlette not installed in CI base env")

from starlette.testclient import TestClient  # noqa: E402

from core.sovereign.api import create_fastapi_app  # noqa: E402
from core.sovereign.runtime_types import RuntimeMetrics  # noqa: E402


def _runtime_for_auth(tmp_state_dir: Path) -> MagicMock:
    """Minimal runtime mock for FastAPI auth-endpoint integration."""
    runtime = MagicMock()
    runtime.config = SimpleNamespace(state_dir=tmp_state_dir)
    runtime.metrics = RuntimeMetrics(
        queries_processed=0,
        queries_succeeded=0,
        current_snr_score=0.91,
        current_ihsan_score=0.96,
        avg_query_time_ms=0.0,
    )
    runtime.status.return_value = {
        "health": {"status": "healthy"},
        "identity": {"version": "test"},
        "state": {"running": True},
    }
    return runtime


@pytest.fixture
def auth_client(tmp_path: Path) -> TestClient:
    """Boot a FastAPI app with a fresh auth DB under tmp_path."""
    runtime = _runtime_for_auth(tmp_path)
    app = create_fastapi_app(runtime)
    return TestClient(app)


# Canonical shape expectations — MUST match frontend/src/types.ts exactly.
# If either shape changes, one side of the contract has drifted.
_AUTH_TOKENS_KEYS = {"access_token", "refresh_token", "token_type", "expires_in"}
_USER_PROFILE_KEYS = {
    "user_id",
    "username",
    "email",
    "api_key",
    "namespace",
    "covenant_accepted",
    "created_at",
}
_REGISTER_TOP_KEYS = {"user", "tokens"}
_LOGIN_TOP_KEYS = {"user_id", "username", "tokens"}


class TestRegisterWireContract:
    """POST /v1/auth/register response must match RegisterResponse in types.ts."""

    def test_register_success_returns_nested_user_and_tokens(
        self, auth_client: TestClient
    ) -> None:
        resp = auth_client.post(
            "/v1/auth/register",
            json={
                "username": "contract_user",
                "email": "contract@bizra.ai",
                "password": "s3cretpass-123",
                "accept_covenant": True,
            },
        )
        assert resp.status_code == 200, (
            f"register failed: {resp.status_code} {resp.text}"
        )
        body = resp.json()

        # Top-level shape: exactly {user, tokens}
        assert set(body.keys()) == _REGISTER_TOP_KEYS, (
            f"RegisterResponse top-level keys drifted. "
            f"Expected {_REGISTER_TOP_KEYS}, got {set(body.keys())}. "
            f"Update frontend/src/types.ts::RegisterResponse to match."
        )

        # user shape
        user = body["user"]
        assert set(user.keys()) == _USER_PROFILE_KEYS, (
            f"UserProfile keys drifted. "
            f"Expected {_USER_PROFILE_KEYS}, got {set(user.keys())}. "
            f"Update frontend/src/types.ts::UserProfile."
        )
        assert user["username"] == "contract_user"
        assert user["email"] == "contract@bizra.ai"
        assert user["covenant_accepted"] is True

        # tokens shape
        tokens = body["tokens"]
        assert set(tokens.keys()) == _AUTH_TOKENS_KEYS, (
            f"AuthTokens keys drifted. "
            f"Expected {_AUTH_TOKENS_KEYS}, got {set(tokens.keys())}. "
            f"Update frontend/src/types.ts::AuthTokens."
        )
        assert isinstance(tokens["access_token"], str) and tokens["access_token"]
        assert isinstance(tokens["refresh_token"], str) and tokens["refresh_token"]
        assert isinstance(tokens["expires_in"], int)

    def test_register_rejects_missing_covenant(self, auth_client: TestClient) -> None:
        """Backend requires accept_covenant=True; omission or False → 400."""
        resp = auth_client.post(
            "/v1/auth/register",
            json={
                "username": "no_covenant",
                "email": "nc@bizra.ai",
                "password": "s3cretpass-123",
                "accept_covenant": False,
            },
        )
        assert resp.status_code == 400
        assert "covenant" in resp.json().get("error", "").lower()

    def test_register_rejects_missing_email(self, auth_client: TestClient) -> None:
        """Frontend drift pre-A.2 sent {username, password, name} — must fail."""
        resp = auth_client.post(
            "/v1/auth/register",
            json={
                "username": "no_email",
                "password": "s3cretpass-123",
                "name": "drift-shape-pre-a2",
            },
        )
        # FastAPI 422 for validation error on missing required field
        assert resp.status_code in (400, 422), (
            f"Expected rejection of pre-A.2 flat payload; got {resp.status_code}."
        )


class TestLoginWireContract:
    """POST /v1/auth/login response must match LoginResponse in types.ts."""

    def test_login_success_returns_flat_user_id_and_nested_tokens(
        self, auth_client: TestClient
    ) -> None:
        # Register first so we have credentials to log in with.
        reg = auth_client.post(
            "/v1/auth/register",
            json={
                "username": "login_user",
                "email": "login@bizra.ai",
                "password": "s3cretpass-xyz",
                "accept_covenant": True,
            },
        )
        assert reg.status_code == 200, reg.text

        resp = auth_client.post(
            "/v1/auth/login",
            json={"username": "login_user", "password": "s3cretpass-xyz"},
        )
        assert resp.status_code == 200, f"login failed: {resp.text}"
        body = resp.json()

        # Top-level shape: exactly {user_id, username, tokens}
        assert set(body.keys()) == _LOGIN_TOP_KEYS, (
            f"LoginResponse top-level keys drifted. "
            f"Expected {_LOGIN_TOP_KEYS}, got {set(body.keys())}. "
            f"Update frontend/src/types.ts::LoginResponse."
        )
        assert body["username"] == "login_user"
        assert isinstance(body["user_id"], str) and body["user_id"]

        tokens = body["tokens"]
        assert set(tokens.keys()) == _AUTH_TOKENS_KEYS

    def test_login_rejects_wrong_password(self, auth_client: TestClient) -> None:
        auth_client.post(
            "/v1/auth/register",
            json={
                "username": "pw_user",
                "email": "pw@bizra.ai",
                "password": "correct-pass-123",
                "accept_covenant": True,
            },
        )
        resp = auth_client.post(
            "/v1/auth/login",
            json={"username": "pw_user", "password": "wrong-pass"},
        )
        assert resp.status_code == 401
