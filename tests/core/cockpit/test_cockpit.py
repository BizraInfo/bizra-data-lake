"""Tests for Glass Cockpit v0.1."""

import pytest
from fastapi.testclient import TestClient

from core.cockpit.server import app


@pytest.fixture
def client():
    return TestClient(app)


class TestDashboard:
    def test_root_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "Glass Cockpit" in resp.text
        assert "FATE" in resp.text

    def test_html_has_verdict_section(self, client):
        resp = client.get("/")
        assert "Verdict Distribution" in resp.text

    def test_html_has_routing_section(self, client):
        resp = client.get("/")
        assert "Model Routing" in resp.text

    def test_html_has_health_section(self, client):
        resp = client.get("/")
        assert "Runtime Health" in resp.text


class TestApiEndpoints:
    def test_fate_api(self, client):
        resp = client.get("/api/fate")
        assert resp.status_code == 200
        data = resp.json()
        assert "total" in data
        assert "verdicts" in data

    def test_health_api(self, client):
        resp = client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert "gpu" in data
        assert "ollama" in data

    def test_routing_api(self, client):
        resp = client.get("/api/routing")
        assert resp.status_code == 200
        data = resp.json()
        assert "pat.researcher" in data
        assert "sat.sentinel" in data

    def test_activity_api(self, client):
        resp = client.get("/api/activity")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
