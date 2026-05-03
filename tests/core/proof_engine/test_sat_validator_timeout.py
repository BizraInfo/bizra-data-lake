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
    output = SimplePatOutput(
        answer="Evidence-bound answer",
        evidence_refs=["git-log:proof"],
        confidence="high",
    )

    for invalid_timeout in ("0", "nan", "inf"):
        monkeypatch.setenv("BIZRA_SAT_TIMEOUT_SECONDS", invalid_timeout)

        try:
            sat_validator._call_sat_model(output)
        except ValueError as exc:
            assert "BIZRA_SAT_TIMEOUT_SECONDS" in str(exc)
        else:
            raise AssertionError("expected invalid SAT timeout to fail fast")


def test_sat_validator_rejects_nonfinite_timeout_before_urlopen(monkeypatch):
    def fail_urlopen(request, timeout):
        raise AssertionError("urlopen should not receive non-finite timeout")

    output = SimplePatOutput(
        answer="Evidence-bound answer",
        evidence_refs=["git-log:proof"],
        confidence="high",
    )

    monkeypatch.setenv("BIZRA_SAT_TIMEOUT_SECONDS", "nan")
    monkeypatch.setattr(urllib.request, "urlopen", fail_urlopen)

    try:
        sat_validator._call_sat_model(output)
    except ValueError as exc:
        assert "BIZRA_SAT_TIMEOUT_SECONDS" in str(exc)
    else:
        raise AssertionError("expected invalid SAT timeout to fail fast")
