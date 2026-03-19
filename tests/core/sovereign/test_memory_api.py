from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from starlette.testclient import TestClient

from core.memory.agent_db import AgentDB
from core.memory.config import HNSWConfig, MemoryConfig
from core.memory.types import MemoryKind, MemoryRecord, RecordState
from core.sovereign.api import create_fastapi_app


class _RuntimeStub:
    def __init__(self, tmp_path: Path, agent_db: AgentDB) -> None:
        self.config = SimpleNamespace(state_dir=tmp_path / "state")
        self.metrics = MagicMock(to_prometheus=lambda include_help=False: "")
        self.query = AsyncMock()
        self._agent_db = agent_db
        self._experience_ledger = None
        self._cognitive_fusion = None
        self._orchestrator = None
        self._node_signer = None
        self._evidence_ledger = None

    def status(self) -> dict:
        return {
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


@pytest.fixture()
def agent_db(tmp_path: Path) -> AgentDB:
    config = MemoryConfig(
        data_dir=tmp_path / "agent_db",
        hnsw=HNSWConfig(dimensions=8, max_elements=128),
    )
    config.auto_embed = False
    db = AgentDB(config)
    db.initialize()
    now = datetime.now(timezone.utc)
    db.store_record(
        MemoryRecord(
            id="semantic_1",
            content="Alpha memory for graph retrieval",
            kind=MemoryKind.SEMANTIC,
            state=RecordState.ACTIVE,
            embedding=[0.1] * 8,
            source="claude_flow_db",
            tags=["ops", "alpha"],
            related_ids=["ctx-1"],
            created_at=now,
            updated_at=now,
            last_accessed=now,
        )
    )
    db.store_record(
        MemoryRecord(
            id="procedural_1",
            content="Pattern memory for orchestration",
            kind=MemoryKind.PROCEDURAL,
            state=RecordState.ACTIVE,
            embedding=[0.2] * 8,
            source="claude_flow_project_patterns",
            tags=["pattern"],
            created_at=now,
            updated_at=now,
            last_accessed=now,
        )
    )
    db.store_record(
        MemoryRecord(
            id="archived_1",
            content="Archive-only memory surface",
            kind=MemoryKind.SEMANTIC,
            state=RecordState.ARCHIVED,
            source="claude_flow_db",
            tags=["archive"],
            created_at=now,
            updated_at=now,
            last_accessed=now,
        )
    )
    return db


@pytest.fixture()
def client(tmp_path: Path, monkeypatch, agent_db: AgentDB):
    monkeypatch.setenv("BIZRA_AUTH_ALLOW_ANONYMOUS", "1")
    app = create_fastapi_app(_RuntimeStub(tmp_path, agent_db))
    return TestClient(app, raise_server_exceptions=False)


def test_memory_search_backward_compatible_shape(client: TestClient) -> None:
    response = client.post(
        "/v1/memory/search",
        json={
            "query": "Alpha memory",
            "top_k": 5,
            "min_score": 0.0,
            "source": "claude_flow_db",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] >= 1
    first = payload["results"][0]
    assert "score" in first
    assert "graph_score" in first
    assert "tags" in first
    assert "related_ids" in first


def test_memory_search_supports_kind_tag_and_graph_filters(client: TestClient) -> None:
    response = client.post(
        "/v1/memory/search",
        json={
            "query": "Alpha memory",
            "min_score": 0.0,
            "kinds": ["semantic"],
            "tags": ["ops"],
            "context_ids": ["ctx-1"],
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["results"][0]["graph_score"] > 0


def test_memory_search_include_archived_toggle(client: TestClient) -> None:
    hidden = client.post(
        "/v1/memory/search",
        json={"query": "Archive-only memory", "min_score": 0.0},
    )
    shown = client.post(
        "/v1/memory/search",
        json={
            "query": "Archive-only memory",
            "min_score": 0.0,
            "include_archived": True,
        },
    )

    assert hidden.status_code == 200
    assert hidden.json()["count"] == 0
    assert shown.status_code == 200
    assert shown.json()["count"] == 1


def test_memory_search_rejects_invalid_kind(client: TestClient) -> None:
    response = client.post(
        "/v1/memory/search",
        json={"query": "anything", "kinds": ["not-a-kind"]},
    )

    assert response.status_code == 400


def test_memory_stats_expose_index_health_fields(client: TestClient) -> None:
    response = client.get("/v1/memory/stats")

    assert response.status_code == 200
    payload = response.json()
    assert "fts_row_count" in payload
    assert "indexed_vectors" in payload
    assert "embedding_dimensions" in payload
    assert "vector_backend" in payload
    assert "index_health" in payload
    assert "last_rebuild_at" in payload
