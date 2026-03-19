import os

import httpx
import pytest

BASE = os.environ.get("BIZRA_CONFORMANCE_BASE", "http://127.0.0.1")
API_KEY = os.environ.get("BIZRA_API_KEY", "")


@pytest.mark.parametrize(
    "path",
    [
        "8011/health",
        "8012/health",
        "8013/health",
        "8014/health",
        "8090/health",
    ],
)
def test_health(path):
    r = httpx.get(f"{BASE}:{path}", timeout=5)
    assert r.status_code == 200
    assert r.json().get("ok") is True


def test_node_plan():
    assert API_KEY, "BIZRA_API_KEY must be set for authenticated conformance checks"
    r = httpx.post(
        f"{BASE}:8090/v1/plan",
        json={"text": "Draft an email to John", "context": {}},
        headers={"x-bizra-api-key": API_KEY},
        timeout=10,
    )
    assert r.status_code == 200
    data = r.json()
    assert "macro_state" in data
    assert "steps" in data and isinstance(data["steps"], list)
    assert "snr" in data
