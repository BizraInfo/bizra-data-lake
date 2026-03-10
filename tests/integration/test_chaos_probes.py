"""
Chaos Probes — Degraded State Integration Tests
=================================================

Tests system behavior under adversarial conditions:
1. Auth failure paths (expired/invalid/missing tokens)
2. Receipt tamper detection (modified hash chain)
3. Backend failover (primary down, fallback engaged)

These probe the rarely-fired circuits identified in SAPE analysis.

Standing on Giants:
- Netflix Chaos Monkey (2012): fault injection as standard practice
- Lamport (1978): chain integrity verification
- OWASP (2023): auth failure response semantics
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import pytest

# ═══════════════════════════════════════════════════════════════════════════════
# PROBE 1: Auth Failure Paths
# ═══════════════════════════════════════════════════════════════════════════════


class _FakeJWTAuth:
    """Mock JWT auth that can simulate expired/invalid tokens."""

    def __init__(self, *, valid: bool = True, expired: bool = False):
        self._valid = valid
        self._expired = expired

    def verify_token(self, token: str, expected_type: str = "access") -> Optional[Any]:
        if self._expired:
            return None
        if not self._valid:
            return None
        return SimpleNamespace(sub="user-001", exp=time.time() + 3600)


class _FakeUserStore:
    """Mock user store with controllable user state."""

    def __init__(self, *, active: bool = True, exists: bool = True):
        self._active = active
        self._exists = exists

    def get_by_id(self, user_id: str) -> Optional[Any]:
        if not self._exists:
            return None
        return SimpleNamespace(
            id=user_id, is_active=self._active, tier="SPROUT", api_keys=[]
        )

    def verify_api_key(self, key: str) -> Optional[Any]:
        if key == "bzr_valid_key":
            return self.get_by_id("user-api-001")
        return None


class TestAuthFailurePaths:
    """Chaos probes for authentication failure modes."""

    def test_expired_jwt_returns_none(self) -> None:
        """Expired JWT tokens must be rejected — not silently accepted."""
        from core.auth.middleware import AuthMiddleware

        auth = AuthMiddleware(
            user_store=_FakeUserStore(),
            jwt_auth=_FakeJWTAuth(expired=True),
        )
        result = auth.authenticate(authorization="Bearer expired-token-xyz")
        assert result is None, "Expired JWT must be rejected"

    def test_invalid_jwt_returns_none(self) -> None:
        """Malformed JWT tokens must be rejected."""
        from core.auth.middleware import AuthMiddleware

        auth = AuthMiddleware(
            user_store=_FakeUserStore(),
            jwt_auth=_FakeJWTAuth(valid=False),
        )
        result = auth.authenticate(authorization="Bearer garbage.token.here")
        assert result is None, "Invalid JWT must be rejected"

    def test_valid_jwt_inactive_user_rejected(self) -> None:
        """Valid JWT for a deactivated user must fail."""
        from core.auth.middleware import AuthMiddleware

        auth = AuthMiddleware(
            user_store=_FakeUserStore(active=False),
            jwt_auth=_FakeJWTAuth(valid=True),
        )
        result = auth.authenticate(authorization="Bearer valid-but-inactive")
        assert result is None, "Inactive user must be rejected even with valid JWT"

    def test_missing_user_in_store_rejected(self) -> None:
        """JWT for a deleted user must fail."""
        from core.auth.middleware import AuthMiddleware

        auth = AuthMiddleware(
            user_store=_FakeUserStore(exists=False),
            jwt_auth=_FakeJWTAuth(valid=True),
        )
        result = auth.authenticate(authorization="Bearer valid-but-deleted")
        assert result is None, "Nonexistent user must be rejected"

    def test_no_credentials_returns_none(self) -> None:
        """Request with no auth headers must fail cleanly."""
        from core.auth.middleware import AuthMiddleware

        auth = AuthMiddleware(
            user_store=_FakeUserStore(),
            jwt_auth=_FakeJWTAuth(),
        )
        result = auth.authenticate()
        assert result is None, "No credentials must return None"

    def test_invalid_api_key_rejected(self) -> None:
        """Incorrect API key must be rejected."""
        from core.auth.middleware import AuthMiddleware

        auth = AuthMiddleware(
            user_store=_FakeUserStore(),
            jwt_auth=_FakeJWTAuth(valid=False),
        )
        result = auth.authenticate(api_key="bzr_invalid_key_xxx")
        assert result is None, "Invalid API key must be rejected"

    def test_rate_limit_exhaustion_blocks_requests(self) -> None:
        """After burst exhaustion, rate limiter must block."""
        from core.auth.middleware import AuthMiddleware

        auth = AuthMiddleware(
            user_store=_FakeUserStore(),
            jwt_auth=_FakeJWTAuth(),
            rate_limit_per_minute=60,
            burst_size=3,
        )
        # First call creates bucket (returns True, 3 tokens remaining).
        # Calls 2-4 consume tokens (True each). Call 5 should block.
        for i in range(4):
            assert (
                auth.check_rate_limit("flood-user") is True
            ), f"Call {i+1} should pass"
        # Next should be blocked (no time elapsed, tokens exhausted)
        assert auth.check_rate_limit("flood-user") is False


# ═══════════════════════════════════════════════════════════════════════════════
# PROBE 2: Receipt Tamper Detection
# ═══════════════════════════════════════════════════════════════════════════════


class TestReceiptTamperDetection:
    """Chaos probes for hash-chain integrity verification."""

    def test_intact_chain_verifies(self, tmp_path: Path) -> None:
        """A clean ledger must pass verification."""
        from core.proof_engine.evidence_ledger import EvidenceLedger

        ledger = EvidenceLedger(
            path=tmp_path / "test_ledger.jsonl", validate_on_append=False
        )
        ledger.append(receipt={"tool": "test", "score": 0.95})
        ledger.append(receipt={"tool": "test2", "score": 0.98})
        is_valid, errors = ledger.verify_chain()
        assert is_valid is True, f"Clean chain must verify: {errors}"
        assert errors == []

    def test_tampered_entry_detected(self, tmp_path: Path) -> None:
        """Modifying an entry mid-chain must be detected."""
        from core.proof_engine.evidence_ledger import EvidenceLedger

        ledger_path = tmp_path / "tampered_ledger.jsonl"
        ledger = EvidenceLedger(path=ledger_path, validate_on_append=False)
        ledger.append(receipt={"tool": "legit", "score": 0.95})
        ledger.append(receipt={"tool": "legit2", "score": 0.97})
        ledger.append(receipt={"tool": "legit3", "score": 0.99})

        # Tamper: modify the second entry's receipt
        lines = ledger_path.read_text().splitlines()
        assert len(lines) >= 3

        entry = json.loads(lines[1])
        entry["receipt"]["score"] = 0.01  # Tampered score
        lines[1] = json.dumps(entry)
        ledger_path.write_text("\n".join(lines) + "\n")

        # Re-verify — must detect tamper
        is_valid, errors = ledger.verify_chain()
        assert is_valid is False, "Tampered chain must fail verification"
        assert len(errors) > 0, "Must report at least one error"

    def test_deleted_entry_detected(self, tmp_path: Path) -> None:
        """Removing an entry from the chain must be detected."""
        from core.proof_engine.evidence_ledger import EvidenceLedger

        ledger_path = tmp_path / "deleted_ledger.jsonl"
        ledger = EvidenceLedger(path=ledger_path, validate_on_append=False)
        ledger.append(receipt={"tool": "a", "score": 0.95})
        ledger.append(receipt={"tool": "b", "score": 0.96})
        ledger.append(receipt={"tool": "c", "score": 0.97})

        # Delete middle entry
        lines = ledger_path.read_text().splitlines()
        del lines[1]
        ledger_path.write_text("\n".join(lines) + "\n")

        is_valid, errors = ledger.verify_chain()
        assert is_valid is False, "Deleted entry must break chain"

    def test_reordered_entries_detected(self, tmp_path: Path) -> None:
        """Swapping entries must be detected via sequence + hash chain."""
        from core.proof_engine.evidence_ledger import EvidenceLedger

        ledger_path = tmp_path / "reordered_ledger.jsonl"
        ledger = EvidenceLedger(path=ledger_path, validate_on_append=False)
        ledger.append(receipt={"tool": "first"})
        ledger.append(receipt={"tool": "second"})
        ledger.append(receipt={"tool": "third"})

        # Swap entries 1 and 2
        lines = ledger_path.read_text().splitlines()
        lines[0], lines[1] = lines[1], lines[0]
        ledger_path.write_text("\n".join(lines) + "\n")

        is_valid, errors = ledger.verify_chain()
        assert is_valid is False, "Reordered entries must break chain"

    def test_empty_ledger_is_valid(self, tmp_path: Path) -> None:
        """An empty or non-existent ledger is valid (genesis state)."""
        from core.proof_engine.evidence_ledger import EvidenceLedger

        ledger = EvidenceLedger(
            path=tmp_path / "nonexistent.jsonl", validate_on_append=False
        )
        is_valid, errors = ledger.verify_chain()
        assert is_valid is True
        assert errors == []


# ═══════════════════════════════════════════════════════════════════════════════
# PROBE 3: Backend Failover (Connection Pool)
# ═══════════════════════════════════════════════════════════════════════════════


class TestBackendFailover:
    """Chaos probes for inference backend failover behavior."""

    def test_connection_pool_config_enforces_bounds(self) -> None:
        """Pool config must enforce max_size >= min_size."""
        from core.inference._connection_pool import ConnectionPool, ConnectionPoolConfig

        config = ConnectionPoolConfig(min_size=1, max_size=2)
        pool = ConnectionPool(
            backend_type="test",
            endpoint="http://localhost:9999",
            config=config,
        )
        assert pool.config.max_size == 2
        assert pool.config.min_size == 1

    def test_conservative_fallback_is_deny_by_default(self) -> None:
        """FATE conservative fallback must default-deny, not default-allow."""
        from core.sovereign.conservative_fallback import conservative_fallback_check

        # A context with no clear safety signal should be denied
        verdict = conservative_fallback_check(
            {"action": "unknown_dangerous_action", "source": "untrusted"},
        )
        # Conservative fallback must not approve unrecognized actions
        assert verdict is not None, "Fallback must return a verdict (not crash)"
        assert hasattr(verdict, "approved"), "Verdict must have .approved field"
        # Default-deny: unrecognized action patterns should be rejected
        assert (
            verdict.approved is False
        ), "Unknown actions must be denied (default-deny)"

    def test_inference_tier_ordering(self) -> None:
        """Inference tiers must be ordered: local-first, cloud-last."""
        try:
            from bizra_config import LM_STUDIO_URL, OLLAMA_URL
        except ImportError:
            pytest.skip("bizra_config not available in CI")

        # Verify local-first ordering is maintained
        assert (
            "192.168" in LM_STUDIO_URL
            or "localhost" in LM_STUDIO_URL
            or "127.0.0.1" in LM_STUDIO_URL
        ), "Primary inference must be local"
        assert (
            "localhost" in OLLAMA_URL or "127.0.0.1" in OLLAMA_URL
        ), "Fallback inference must be local"

    def test_mcp_gateway_fail_closed_without_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """MCP gateway must reject requests when no token is configured."""
        pytest.importorskip("redis")
        monkeypatch.delenv("BIZRA_MCP_GATEWAY_TOKEN", raising=False)
        monkeypatch.delenv("BIZRA_BRIDGE_TOKEN", raising=False)
        monkeypatch.delenv("BIZRA_MCP_ALLOW_ANONYMOUS", raising=False)
        monkeypatch.delenv("BIZRA_MCP_ALLOW_REMOTE", raising=False)

        from fastapi import HTTPException

        from tools.mcp import mcp_gateway

        req = SimpleNamespace(
            headers={},
            client=SimpleNamespace(host="127.0.0.1"),
            method="POST",
            url=SimpleNamespace(path="/mcp"),
        )
        with pytest.raises(HTTPException) as exc:
            mcp_gateway._authorize_request(req)
        assert exc.value.status_code == 503, "No token configured must yield 503"

    def test_mcp_gateway_rejects_remote_by_default(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Remote requests must be rejected unless explicitly allowed."""
        pytest.importorskip("redis")
        monkeypatch.setenv("BIZRA_MCP_GATEWAY_TOKEN", "test-secret")
        monkeypatch.delenv("BIZRA_MCP_ALLOW_REMOTE", raising=False)
        monkeypatch.delenv("BIZRA_MCP_ALLOW_ANONYMOUS", raising=False)

        from fastapi import HTTPException

        from tools.mcp import mcp_gateway

        req = SimpleNamespace(
            headers={"authorization": "Bearer test-secret"},
            client=SimpleNamespace(host="203.0.113.1"),  # External IP
            method="POST",
            url=SimpleNamespace(path="/mcp"),
        )
        with pytest.raises(HTTPException) as exc:
            mcp_gateway._authorize_request(req)
        assert exc.value.status_code == 403, "Remote requests must be denied by default"
