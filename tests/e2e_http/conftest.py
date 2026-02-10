"""
E2E HTTP Integration Test Fixtures

Provides fixtures for testing against the Rust API server (HTTP)
and the PyO3 inference bridge (direct Python↔Rust).

Usage:
    # Run only when API server is up (docker-compose or manual)
    pytest tests/e2e_http/ -m e2e_http

    # Run PyO3 bridge tests (no server needed, just maturin develop)
    pytest tests/e2e_http/ -m pyo3_bridge
"""

import os
import subprocess
import time

import httpx
import pytest

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

API_BASE_URL = os.getenv("BIZRA_API_URL", "http://localhost:3001/api/v1")
API_STARTUP_TIMEOUT = int(os.getenv("BIZRA_API_STARTUP_TIMEOUT", "30"))


# ---------------------------------------------------------------------------
# Markers
# ---------------------------------------------------------------------------


def pytest_configure(config):
    config.addinivalue_line("markers", "e2e_http: requires running Rust API server")
    config.addinivalue_line("markers", "pyo3_bridge: requires PyO3 bindings (maturin develop)")


# ---------------------------------------------------------------------------
# HTTP Client Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def api_base_url():
    """Base URL for the Rust API server."""
    return API_BASE_URL


@pytest.fixture(scope="session")
def api_client(api_base_url):
    """HTTP client configured for the Rust API server."""
    client = httpx.Client(base_url=api_base_url, timeout=30.0)
    yield client
    client.close()


@pytest.fixture(scope="session")
def api_server_available(api_base_url):
    """Check if the API server is reachable. Skip if not."""
    try:
        resp = httpx.get(f"{api_base_url}/health", timeout=5.0)
        return resp.status_code == 200
    except (httpx.ConnectError, httpx.ConnectTimeout):
        return False


# ---------------------------------------------------------------------------
# PyO3 Bridge Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def bizra_module():
    """Import the PyO3 bizra module. Skip if not built."""
    try:
        import bizra
        return bizra
    except ImportError:
        pytest.skip("PyO3 bizra module not built (run: cd bizra-omega/bizra-python && maturin develop --release)")


@pytest.fixture(scope="session")
def pyo3_identity(bizra_module):
    """Generate a test NodeIdentity via PyO3."""
    return bizra_module.NodeIdentity()


@pytest.fixture(scope="session")
def pyo3_constitution(bizra_module):
    """Create default Constitution via PyO3."""
    return bizra_module.Constitution()


@pytest.fixture(scope="session")
def pyo3_gateway(bizra_module, pyo3_identity, pyo3_constitution):
    """Create an InferenceGateway via PyO3 (no backends registered)."""
    return bizra_module.InferenceGateway(pyo3_identity, pyo3_constitution)
