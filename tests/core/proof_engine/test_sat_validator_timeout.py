"""Timeout boundary tests for the SAT validator."""

from __future__ import annotations

import urllib.error
import urllib.request

from core.proof_engine import sat_validator
from core.proof_engine.sat_validator import SimplePatOutput


def test_sat_validator_uses_configured_http_timeout(monkeypatch):
    captured: dict[str, float] = {}

    def fake_urlopen(request, timeout):
        captured["timeout"] = timeout
        raise urllib.error.URLError("offline")

    monkeypatch.setenv("BIZRA_SAT_TIMEOUT_SECONDS", "7.5")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = sat_validator._call_sat_model(
        SimplePatOutput(
            answer="Evidence-bound answer",
            evidence_refs=["git-log:proof"],
            confidence="high",
        )
    )

    assert result == {}
    assert captured["timeout"] == 7.5


def test_sat_validator_rejects_invalid_timeout(monkeypatch):
    monkeypatch.setenv("BIZRA_SAT_TIMEOUT_SECONDS", "0")

    output = SimplePatOutput(
        answer="Evidence-bound answer",
        evidence_refs=["git-log:proof"],
        confidence="high",
    )

    try:
        sat_validator._call_sat_model(output)
    except ValueError as exc:
        assert "BIZRA_SAT_TIMEOUT_SECONDS" in str(exc)
    else:
        raise AssertionError("expected invalid SAT timeout to fail fast")
