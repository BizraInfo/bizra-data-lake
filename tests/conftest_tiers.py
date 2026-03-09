# conftest_tiers.py — Drop into your project root conftest.py or import from it
# BIZRA Test Tier Markers
# Usage: pytest -m "smoke" / pytest -m "contract" / pytest -m "not slow"



def pytest_configure(config):
    """Register BIZRA test tier markers."""
    config.addinivalue_line("markers", "smoke: T0 — runs on every save (< 30 sec)")
    config.addinivalue_line("markers", "unit: T1 — runs on every commit (< 2 min)")
    config.addinivalue_line("markers", "contract: T2 — runs on merge to main (< 5 min)")
    config.addinivalue_line("markers", "slow: T3 — runs on version lock only (20+ min)")
    config.addinivalue_line("markers", "genesis_gate: T4 — runs on release candidate only")
    config.addinivalue_line("markers", "requires_ollama: needs Ollama runtime")
    config.addinivalue_line("markers", "requires_gpu: needs GPU acceleration")
    config.addinivalue_line("markers", "requires_network: needs internet access")


# ============================================================================
# SMOKE TEST LIST (T0) — These files run on EVERY save
# Keep this list small. < 50 tests. < 30 seconds.
# ============================================================================
SMOKE_TEST_FILES = [
    "tests/core/integration/test_constants_integrity.py",
    "tests/core/sovereign/test_api_exposure_policy.py",
    "tests/core/constitutional/test_ticker_smoke.py",
    "tests/core/token/test_bloom_smoke.py",
    "tests/core/auth/test_middleware_smoke.py",
]

# ============================================================================
# CONTRACT TEST LIST (T2) — These files run on every merge to main
# ~800-1200 tests. < 5 minutes.
# ============================================================================
CONTRACT_TEST_FILES = [
    "tests/core/sovereign/test_api_exposure_policy.py",
    "tests/integration/test_contract_integrity.py",
    "tests/core/sovereign/test_terminal.py",
    "tests/core/orchestration/test_learning_loop.py",
    "tests/core/test_learning_loop_bridges.py",
    "tests/core/test_living_ecosystem.py",
    "tests/core/constitutional/test_ticker.py",
    "tests/core/proof_engine/test_poi_engine.py",
    "tests/core/token/test_bloom.py",
]
