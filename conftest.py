from __future__ import annotations


MARKERS = [
    "smoke: T0 - runs on every save (< 30 sec)",
    "unit: T1 - runs on every commit (< 2 min)",
    "contract: T2 - runs on merge to main (< 5 min)",
    "genesis_gate: T4 - runs on release candidate only",
    "e2e_http: marks tests that need a running API server",
]


def pytest_configure(config) -> None:
    """Register repository-level test tier markers.

    This supplements tests/conftest.py without editing it, which keeps
    strict-marker mode happy for the versioned test workflow.
    """
    for marker in MARKERS:
        config.addinivalue_line("markers", marker)
