"""Tests for BIZRA NODE0 Server — FastAPI Endpoints."""

import os
import sys
import pytest
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("BIZRA_CONSTITUTION_PATH",
                       str(Path(__file__).parent.parent / "constitution.toml"))

from fastapi.testclient import TestClient
from node0_server import create_app


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("node0_test")
    app = create_app(
        data_dir=tmp,
        ollama_url="http://localhost:99999",  # No real Ollama
    )
    with TestClient(app) as c:
        yield c


class TestRootEndpoint:
    def test_root_returns_200(self, client):
        r = client.get("/")
        assert r.status_code == 200

    def test_root_has_name(self, client):
        data = client.get("/").json()
        assert data["name"] == "BIZRA NODE0"

    def test_root_has_version(self, client):
        data = client.get("/").json()
        assert "5.0.0" in data["version"]

    def test_root_lists_endpoints(self, client):
        data = client.get("/").json()
        assert len(data["endpoints"]) >= 4


class TestMissionEndpoint:
    def test_submit_mission(self, client):
        r = client.post("/mission", json={"input": "Hello NODE0"})
        assert r.status_code == 200

    def test_mission_returns_output(self, client):
        data = client.post("/mission", json={"input": "Test"}).json()
        assert len(data["output"]) > 0

    def test_mission_has_ihsan(self, client):
        data = client.post("/mission", json={"input": "Test"}).json()
        assert 0 <= data["ihsan_composite"] <= 1

    def test_mission_has_dimensions(self, client):
        data = client.post("/mission", json={"input": "Test"}).json()
        assert len(data["ihsan_dimensions"]) == 6

    def test_mission_has_snr(self, client):
        data = client.post("/mission", json={"input": "Test"}).json()
        assert data["snr_normalized"] > 0

    def test_mission_has_receipt(self, client):
        data = client.post("/mission", json={"input": "Test"}).json()
        assert data["receipt_id"] is not None
        assert len(data["receipt_id"]) == 64

    def test_mission_has_signature(self, client):
        data = client.post("/mission", json={"input": "Sign me"}).json()
        assert data.get("signature_hex") is not None

    def test_mission_has_agent_chain(self, client):
        data = client.post("/mission", json={"input": "Explain quantum entanglement in distributed systems"}).json()
        assert len(data["agent_chain"]) >= 1

    def test_mission_has_timing(self, client):
        data = client.post("/mission", json={"input": "Test"}).json()
        assert data["total_ms"] > 0

    def test_mission_has_tier(self, client):
        data = client.post("/mission", json={"input": "Test"}).json()
        assert data["tier"] in ("trivial", "simple", "complex", "sovereign")

    def test_empty_input_rejected(self, client):
        r = client.post("/mission", json={"input": ""})
        assert r.status_code == 422

    def test_mission_bloom_field(self, client):
        data = client.post("/mission", json={"input": "Test bloom"}).json()
        assert isinstance(data["bloom_eligible"], bool)


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_has_node_id(self, client):
        data = client.get("/health").json()
        assert len(data["node_id"]) == 64

    def test_health_has_uptime(self, client):
        data = client.get("/health").json()
        assert data["uptime_seconds"] >= 0

    def test_health_has_agents(self, client):
        data = client.get("/health").json()
        assert data["total_agents"] == 12

    def test_health_tracks_missions(self, client):
        client.post("/mission", json={"input": "health track"})
        data = client.get("/health").json()
        assert data["missions_completed"] > 0


class TestEvidenceEndpoint:
    def test_evidence_returns_200(self, client):
        r = client.get("/evidence")
        assert r.status_code == 200

    def test_evidence_has_chain_validity(self, client):
        data = client.get("/evidence").json()
        assert "chain_valid" in data

    def test_evidence_has_receipts(self, client):
        # Submit a mission first to create evidence
        client.post("/mission", json={"input": "evidence test"})
        data = client.get("/evidence").json()
        assert data["total_receipts"] > 0


class TestIdentityEndpoint:
    def test_identity_returns_200(self, client):
        r = client.get("/identity")
        assert r.status_code == 200

    def test_identity_has_node_id(self, client):
        data = client.get("/identity").json()
        assert len(data["node_id"]) == 64

    def test_identity_has_agents(self, client):
        data = client.get("/identity").json()
        assert len(data["pat_agents"]) == 7
        assert len(data["sat_agents"]) == 5


class TestCacheEndpoint:
    def test_cache_stats_returns_200(self, client):
        r = client.get("/cache/stats")
        assert r.status_code == 200

    def test_cache_has_stats(self, client):
        data = client.get("/cache/stats").json()
        assert "stats" in data
        assert "size" in data
