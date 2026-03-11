"""Tests for ReasoningBank → AgentDB adapter."""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from core.memory.adapters.reasoning_bank import ReasoningBankAdapter
from core.memory.types import MemoryKind, RecordState


@pytest.fixture
def rb_dir(tmp_path):
    """Create a realistic ReasoningBank state directory."""
    state_dir = tmp_path / "reasoning_bank"
    state_dir.mkdir()

    # Write sample experiences
    experiences = [
        {
            "id": "exp_001",
            "task": "edit",
            "approach": "edit_py",
            "outcome": {
                "success": True,
                "metrics": {"quality_score": 0.92, "time_ms": 150},
            },
            "context": {"file_type": "python", "tool": "Edit"},
            "ihsan_score": 0.95,
            "timestamp": "2025-01-15T10:00:00+00:00",
        },
        {
            "id": "exp_002",
            "task": "bash",
            "approach": "bash_python3",
            "outcome": {
                "success": False,
                "metrics": {"quality_score": 0.30, "time_ms": 5000},
            },
            "context": {"command": "pytest"},
            "ihsan_score": 0.70,
            "timestamp": "2025-01-15T10:05:00+00:00",
        },
    ]
    exp_file = state_dir / "experiences.jsonl"
    exp_file.write_text("\n".join(json.dumps(e) for e in experiences) + "\n")

    # Write sample patterns
    patterns = {
        "edit_then_test": {
            "triggers": ["edit", "python"],
            "actions": ["test", "lint"],
            "confidence": 0.88,
            "occurrences": 12,
        },
        "search_before_edit": {
            "triggers": ["search", "glob"],
            "actions": ["edit"],
            "confidence": 0.75,
            "occurrences": 6,
        },
    }
    (state_dir / "patterns.json").write_text(json.dumps(patterns))

    # Write sample strategies
    strategies = {
        "edit": {
            "best_approach": "edit_py",
            "score": 0.94,
            "confidence": 0.91,
            "success_rate": 0.89,
            "alternatives": [{"approach": "bash_sed", "score": 0.60}],
        },
    }
    (state_dir / "strategies.json").write_text(json.dumps(strategies))

    return state_dir


def test_adapter_not_available(tmp_path):
    """Adapter reports unavailable when no experiences exist."""
    empty = tmp_path / "empty"
    empty.mkdir()
    adapter = ReasoningBankAdapter(state_dir=empty)
    assert not adapter.available
    assert adapter.export_all() == []


def test_adapter_available(rb_dir):
    """Adapter detects ReasoningBank state."""
    adapter = ReasoningBankAdapter(state_dir=rb_dir)
    assert adapter.available


def test_export_all(rb_dir):
    """Export all records: 2 experiences + 2 patterns + 1 strategy."""
    adapter = ReasoningBankAdapter(state_dir=rb_dir)
    records = adapter.export_all()
    assert len(records) == 5  # 2 exp + 2 pat + 1 strat


def test_experience_to_record(rb_dir):
    """Experience converts to EPISODIC MemoryRecord with correct fields."""
    adapter = ReasoningBankAdapter(state_dir=rb_dir)
    records = adapter.export_experiences()
    assert len(records) == 2

    # First experience: successful edit
    r = records[0]
    assert r.id == "exp_001"
    assert r.kind == MemoryKind.EPISODIC
    assert r.state == RecordState.ACTIVE
    assert r.ihsan_score == 0.95
    assert r.source == "reasoning_bank"
    assert "edit" in r.content
    assert "edit_py" in r.content
    assert "reasoning_bank" in r.tags
    assert r.metadata["success"] is True


def test_failed_experience(rb_dir):
    """Failed experience still converts (low score, not dropped)."""
    adapter = ReasoningBankAdapter(state_dir=rb_dir)
    records = adapter.export_experiences()
    r_fail = records[1]
    assert r_fail.ihsan_score == 0.70
    assert r_fail.metadata["success"] is False


def test_pattern_to_record(rb_dir):
    """Pattern converts to PROCEDURAL MemoryRecord."""
    adapter = ReasoningBankAdapter(state_dir=rb_dir)
    records = adapter.export_patterns()
    assert len(records) == 2

    r = records[0]
    assert r.kind == MemoryKind.PROCEDURAL
    assert r.source == "reasoning_bank"
    assert "Pattern:" in r.content
    assert r.metadata["confidence"] == 0.88
    assert r.metadata["occurrences"] == 12


def test_strategy_to_record(rb_dir):
    """Strategy converts to PROCEDURAL MemoryRecord."""
    adapter = ReasoningBankAdapter(state_dir=rb_dir)
    records = adapter.export_strategies()
    assert len(records) == 1

    r = records[0]
    assert r.id == "rb_strat_edit"
    assert r.kind == MemoryKind.PROCEDURAL
    assert "edit_py" in r.content
    assert r.metadata["best_approach"] == "edit_py"
    assert r.metadata["success_rate"] == 0.89


def test_malformed_jsonl_skipped(tmp_path):
    """Malformed JSONL lines are skipped, not crash."""
    state_dir = tmp_path / "rb"
    state_dir.mkdir()
    exp_file = state_dir / "experiences.jsonl"
    exp_file.write_text(
        '{"id":"ok","task":"x","approach":"y","outcome":{"success":true,"metrics":{}}}\n'
        "NOT JSON\n"
        '{"id":"ok2","task":"z","approach":"w","outcome":{"success":false,"metrics":{}}}\n'
    )
    adapter = ReasoningBankAdapter(state_dir=state_dir)
    records = adapter.export_experiences()
    assert len(records) == 2  # Skipped the malformed line


def test_corrupt_patterns_handled(tmp_path):
    """Corrupt patterns.json returns empty list, not crash."""
    state_dir = tmp_path / "rb"
    state_dir.mkdir()
    (state_dir / "experiences.jsonl").write_text("")
    (state_dir / "patterns.json").write_text("{INVALID")
    adapter = ReasoningBankAdapter(state_dir=state_dir)
    records = adapter.export_patterns()
    assert records == []
