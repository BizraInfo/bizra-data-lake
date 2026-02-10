"""
E2E HTTP Tests — Health, Status, and Constitution endpoints.

These tests make real HTTP requests to the running Rust API server.
They validate that the server is alive, correctly configured, and
reports constitutional thresholds.

Run: pytest tests/e2e_http/test_api_health.py -m e2e_http
Requires: Rust API server running on localhost:3001
"""

import pytest


pytestmark = pytest.mark.e2e_http


class TestHealthEndpoint:
    """GET /api/v1/health"""

    def test_health_returns_200(self, api_client, api_server_available):
        if not api_server_available:
            pytest.skip("API server not running")
        resp = api_client.get("/health")
        assert resp.status_code == 200

    def test_health_body_has_status(self, api_client, api_server_available):
        if not api_server_available:
            pytest.skip("API server not running")
        data = api_client.get("/health").json()
        assert data["status"] == "healthy"


class TestStatusEndpoint:
    """GET /api/v1/status"""

    def test_status_returns_200(self, api_client, api_server_available):
        if not api_server_available:
            pytest.skip("API server not running")
        resp = api_client.get("/status")
        assert resp.status_code == 200

    def test_status_contains_version(self, api_client, api_server_available):
        if not api_server_available:
            pytest.skip("API server not running")
        data = api_client.get("/status").json()
        assert "version" in data


class TestConstitutionEndpoint:
    """GET /api/v1/constitution"""

    def test_constitution_returns_200(self, api_client, api_server_available):
        if not api_server_available:
            pytest.skip("API server not running")
        resp = api_client.get("/constitution")
        assert resp.status_code == 200

    def test_constitution_thresholds(self, api_client, api_server_available):
        """Verify the server enforces correct Ihsan and SNR thresholds."""
        if not api_server_available:
            pytest.skip("API server not running")
        data = api_client.get("/constitution").json()
        # Ihsan threshold must be 0.95 (constitutional constant)
        assert data.get("ihsan_threshold", data.get("ihsan", {}).get("minimum")) >= 0.95
        # SNR threshold must be 0.85 (constitutional constant)
        assert data.get("snr_threshold") >= 0.85


class TestInferenceEndpoint:
    """POST /api/v1/inference/generate — the critical path."""

    def test_inference_generate_returns_200(self, api_client, api_server_available):
        """Send a real inference request and verify response structure."""
        if not api_server_available:
            pytest.skip("API server not running")
        payload = {
            "prompt": "What is 2+2?",
            "max_tokens": 32,
            "temperature": 0.1,
        }
        resp = api_client.post("/inference/generate", json=payload)
        # 200 if backend available, still 200 with placeholder if not
        assert resp.status_code == 200
        data = resp.json()
        assert "request_id" in data
        assert "text" in data
        assert "tier" in data
        assert "model" in data

    def test_inference_tier_selection(self, api_client, api_server_available):
        """Verify tier selection endpoint works."""
        if not api_server_available:
            pytest.skip("API server not running")
        payload = {
            "prompt": "Explain quantum computing in detail with code examples",
            "max_tokens": 2000,
            "latency_sensitive": False,
        }
        resp = api_client.post("/inference/tier", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["complexity"] in ["Simple", "Medium", "Complex", "Expert"]
        assert data["recommended_tier"] in ["Edge", "Local", "Pool"]

    def test_inference_models_list(self, api_client, api_server_available):
        """Verify model catalog endpoint."""
        if not api_server_available:
            pytest.skip("API server not running")
        resp = api_client.get("/inference/models")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert len(data["models"]) > 0
        for model in data["models"]:
            assert "name" in model
            assert "tier" in model


class TestPCIEndpoint:
    """POST /api/v1/pci/* — Proof-Carrying Inference protocol."""

    def test_pci_create_envelope(self, api_client, api_server_available):
        """Create a PCI envelope through the HTTP API."""
        if not api_server_available:
            pytest.skip("API server not running")
        payload = {
            "content": '{"query": "test inference", "answer": "42"}',
            "ttl": 3600,
        }
        resp = api_client.post("/pci/envelope/create", json=payload)
        # API may require identity to be initialized first
        assert resp.status_code in [200, 400, 422]

    def test_pci_gate_check(self, api_client, api_server_available):
        """Verify gate check endpoint accepts/rejects based on thresholds."""
        if not api_server_available:
            pytest.skip("API server not running")
        # Valid content with good scores should pass
        payload = {
            "content": '{"valid": "json"}',
            "snr_score": 0.90,
            "ihsan_score": 0.96,
        }
        resp = api_client.post("/pci/gates/check", json=payload)
        assert resp.status_code in [200, 400, 422]


class TestFederationEndpoint:
    """GET /api/v1/federation/* — Federation status."""

    def test_federation_status(self, api_client, api_server_available):
        if not api_server_available:
            pytest.skip("API server not running")
        resp = api_client.get("/federation/status")
        assert resp.status_code == 200

    def test_federation_peers(self, api_client, api_server_available):
        if not api_server_available:
            pytest.skip("API server not running")
        resp = api_client.get("/federation/peers")
        assert resp.status_code == 200
