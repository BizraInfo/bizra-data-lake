"""Timeout boundary tests for the ADK Researcher backend call."""

from __future__ import annotations

import urllib.request

from core.adk.agents import researcher


def test_researcher_uses_configured_pat_timeout(monkeypatch):
    captured: dict[str, float] = {}

    def fake_urlopen(request, timeout):
        captured["timeout"] = timeout
        raise TimeoutError("offline")

    monkeypatch.setenv("BIZRA_PAT_TIMEOUT_SECONDS", "6")
    monkeypatch.setattr(urllib.request, "urlopen", fake_urlopen)

    result = researcher._call_ollama("prompt", "system", "model")

    assert result.startswith("ERROR: Ollama unreachable")
    assert captured["timeout"] == 6.0


def test_researcher_rejects_invalid_pat_timeout(monkeypatch):
    monkeypatch.setenv("BIZRA_PAT_TIMEOUT_SECONDS", "-1")

    try:
        researcher._call_ollama("prompt", "system", "model")
    except ValueError as exc:
        assert "BIZRA_PAT_TIMEOUT_SECONDS" in str(exc)
    else:
        raise AssertionError("expected invalid PAT timeout to fail fast")
