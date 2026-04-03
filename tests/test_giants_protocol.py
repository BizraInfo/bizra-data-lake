from __future__ import annotations

from pathlib import Path

from core.giants_protocol import build_backlog, render_markdown


def test_build_backlog_returns_sorted_scores() -> None:
    backlog = build_backlog("config/giants_protocol_registry.json", top_n=3)
    top = backlog["top"]
    assert len(top) == 3
    assert top[0]["priority_score"] >= top[1]["priority_score"] >= top[2]["priority_score"]


def test_backlog_contains_pilot_or_better() -> None:
    backlog = build_backlog("config/giants_protocol_registry.json", top_n=10)
    statuses = {item["status"] for item in backlog["all"]}
    assert "pilot-ready" in statuses or "production-ready" in statuses


def test_render_markdown_has_expected_headers() -> None:
    backlog = build_backlog("config/giants_protocol_registry.json", top_n=2)
    text = render_markdown(backlog)
    assert "# Giants Protocol Backlog" in text
    assert "score: priority" in text


def test_registry_file_exists() -> None:
    path = Path("config/giants_protocol_registry.json")
    assert path.exists()

