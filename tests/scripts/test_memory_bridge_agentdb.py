"""Tests for the StereoscopicReport -> AgentDB bridge.

Validates that ``ingest_report_to_agent_db`` correctly maps signal nodes
into AgentDB unified memory, including SNR gating and kind mapping.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root and bizra-normalizers are importable.
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_NORMALIZERS_DIR = _PROJECT_ROOT / "bizra-normalizers"
for _p in (_PROJECT_ROOT, _NORMALIZERS_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from memory_bridge import ingest_report_to_agent_db

from core.memory.agent_db import AgentDB
from core.memory.config import MemoryConfig
from core.memory.types import MemoryKind


def _make_report(nodes: list[dict]) -> dict:
    """Build a minimal StereoscopicReport dict."""
    return {
        "report_id": "test-report-001",
        "timestamp": "2026-02-22T00:00:00Z",
        "nodes": nodes,
    }


_TEST_NODES = [
    {
        "kind": "fact",
        "signal": "User prefers dark mode",
        "snr_score": 0.92,
        "confidence": 0.88,
        "providers": ["teach_atom"],
    },
    {
        "kind": "pattern",
        "signal": "User asks questions in short bursts",
        "snr_score": 0.85,
        "confidence": 0.80,
        "providers": ["teach_atom", "conversation"],
    },
    {
        "kind": "emotion",
        "signal": "User expresses frustration with slow responses",
        "snr_score": 0.78,
        "confidence": 0.72,
        "providers": ["conversation"],
    },
    {
        "kind": "goal",
        "signal": "User wants to build a personal knowledge graph",
        "snr_score": 0.95,
        "confidence": 0.91,
        "providers": ["teach_atom"],
    },
    {
        "kind": "relationship",
        "signal": "User collaborates with a team of 5 engineers",
        "snr_score": 0.70,
        "confidence": 0.65,
        "providers": ["teach_atom"],
    },
]


@pytest.fixture()
def agent_db(tmp_path: Path) -> AgentDB:
    """Provide an initialised AgentDB backed by a temporary directory."""
    config = MemoryConfig(data_dir=tmp_path / "agent_db")
    db = AgentDB(config=config)
    db.initialize()
    return db


class TestIngestReportToAgentDB:
    """Suite for ingest_report_to_agent_db."""

    def test_all_nodes_stored(self, agent_db: AgentDB) -> None:
        """All 5 test nodes should be stored when min_snr is 0."""
        report = _make_report(_TEST_NODES)
        result = ingest_report_to_agent_db(report, agent_db, min_snr=0.0)

        assert result["stored"] == 5
        assert result["skipped"] == 0
        assert result["errors"] == 0

    def test_snr_filtering(self, agent_db: AgentDB) -> None:
        """Nodes below min_snr should be skipped."""
        report = _make_report(_TEST_NODES)
        result = ingest_report_to_agent_db(report, agent_db, min_snr=0.80)

        # snr >= 0.80: fact(0.92), pattern(0.85), goal(0.95) = 3 stored
        # snr < 0.80: emotion(0.78), relationship(0.70) = 2 skipped
        assert result["stored"] == 3
        assert result["skipped"] == 2
        assert result["errors"] == 0

    def test_kind_mapping(self, agent_db: AgentDB) -> None:
        """Node kinds should map to the correct MemoryKind values."""
        report = _make_report(_TEST_NODES)
        ingest_report_to_agent_db(report, agent_db)

        # Search by kind to verify mapping.
        semantic = agent_db.search(query="dark mode", kinds=[MemoryKind.SEMANTIC])
        procedural = agent_db.search(
            query="short bursts", kinds=[MemoryKind.PROCEDURAL]
        )
        episodic = agent_db.search(query="frustration", kinds=[MemoryKind.EPISODIC])

        assert len(semantic) >= 1, "fact -> SEMANTIC should be searchable"
        assert len(procedural) >= 1, "pattern -> PROCEDURAL should be searchable"
        assert len(episodic) >= 1, "emotion -> EPISODIC should be searchable"

    def test_search_by_source(self, agent_db: AgentDB) -> None:
        """Records should be retrievable by the provenance source tag."""
        source_tag = "test_compilation_v1"
        report = _make_report(_TEST_NODES)
        ingest_report_to_agent_db(report, agent_db, source=source_tag)

        results = agent_db.search(query="knowledge graph", source=source_tag)
        assert len(results) >= 1
        assert results[0].record.source == source_tag

    def test_metadata_populated(self, agent_db: AgentDB) -> None:
        """Each stored record should carry bridge metadata."""
        report = _make_report(_TEST_NODES[:1])  # single node
        ingest_report_to_agent_db(report, agent_db)

        results = agent_db.search(query="dark mode")
        assert len(results) >= 1

        meta = results[0].record.metadata
        assert meta["bridge"] == "stereoscopic_to_agentdb"
        assert meta["node_kind"] == "fact"
        assert "ihsan_score" in meta
        assert meta["ihsan_score"] <= 1.0

    def test_empty_report(self, agent_db: AgentDB) -> None:
        """An empty report should produce zero stores and zero errors."""
        result = ingest_report_to_agent_db({"nodes": []}, agent_db)
        assert result == {"stored": 0, "skipped": 0, "errors": 0}

    def test_empty_signal_skipped(self, agent_db: AgentDB) -> None:
        """Nodes with blank signal text should be skipped."""
        report = _make_report([{"kind": "fact", "signal": "", "snr_score": 0.9}])
        result = ingest_report_to_agent_db(report, agent_db)

        assert result["stored"] == 0
        assert result["skipped"] == 1

    def test_unknown_kind_defaults_to_semantic(self, agent_db: AgentDB) -> None:
        """An unrecognized node kind should default to SEMANTIC."""
        report = _make_report(
            [{"kind": "alien_concept", "signal": "Something new", "snr_score": 0.9}]
        )
        result = ingest_report_to_agent_db(report, agent_db)
        assert result["stored"] == 1

        results = agent_db.search(query="Something new")
        assert len(results) >= 1
        assert results[0].record.kind == MemoryKind.SEMANTIC

    def test_record_count_matches(self, agent_db: AgentDB) -> None:
        """AgentDB count should increase by the number of stored records."""
        before = agent_db.count
        report = _make_report(_TEST_NODES)
        result = ingest_report_to_agent_db(report, agent_db)
        after = agent_db.count

        assert after - before == result["stored"]
