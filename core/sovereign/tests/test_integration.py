"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║   ████████╗███████╗███████╗████████╗    ███████╗██╗   ██╗██╗████████╗███████╗║
║   ╚══██╔══╝██╔════╝██╔════╝╚══██╔══╝    ██╔════╝██║   ██║██║╚══██╔══╝██╔════╝║
║      ██║   █████╗  ███████╗   ██║       ███████╗██║   ██║██║   ██║   █████╗  ║
║      ██║   ██╔══╝  ╚════██║   ██║       ╚════██║██║   ██║██║   ██║   ██╔══╝  ║
║      ██║   ███████╗███████║   ██║       ███████║╚██████╔╝██║   ██║   ███████╗║
║      ╚═╝   ╚══════╝╚══════╝   ╚═╝       ╚══════╝ ╚═════╝ ╚═╝   ╚═╝   ╚══════╝║
║                                                                              ║
║                    SOVEREIGN ENGINE INTEGRATION TESTS                        ║
║              Comprehensive verification of all components                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import asyncio
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional

# Add parent to path for imports
sys.path.insert(0, str(__file__).rsplit("/", 3)[0])


# =============================================================================
# TEST FRAMEWORK (Minimal, no pytest dependency)
# =============================================================================


@dataclass
class TestResult:
    """Result of a single test."""

    name: str
    passed: bool
    duration_ms: float
    error: Optional[str] = None
    details: Dict[str, Any] = None

    def __post_init__(self):
        if self.details is None:
            self.details = {}


class TestSuite:
    """Minimal test suite runner."""

    def __init__(self, name: str):
        self.name = name
        self.results: List[TestResult] = []

    def run_test(self, name: str, test_fn):
        """Run a single test."""
        start = time.perf_counter()
        try:
            if asyncio.iscoroutinefunction(test_fn):
                asyncio.get_event_loop().run_until_complete(test_fn())
            else:
                test_fn()
            duration = (time.perf_counter() - start) * 1000
            self.results.append(TestResult(name, True, duration))
            print(f"  ✓ {name} ({duration:.1f}ms)")
        except AssertionError as e:
            duration = (time.perf_counter() - start) * 1000
            self.results.append(TestResult(name, False, duration, str(e)))
            print(f"  ✗ {name}: {e}")
        except Exception as e:
            duration = (time.perf_counter() - start) * 1000
            self.results.append(TestResult(name, False, duration, str(e)))
            print(f"  ✗ {name}: {type(e).__name__}: {e}")

    def summary(self) -> Dict[str, Any]:
        """Get test summary."""
        passed = sum(1 for r in self.results if r.passed)
        failed = len(self.results) - passed
        total_time = sum(r.duration_ms for r in self.results)

        return {
            "suite": self.name,
            "total": len(self.results),
            "passed": passed,
            "failed": failed,
            "pass_rate": (
                f"{passed / len(self.results) * 100:.1f}%" if self.results else "N/A"
            ),
            "total_time_ms": total_time,
        }


# =============================================================================
# AUTONOMY MODULE TESTS
# =============================================================================


def test_autonomy_module():
    """Test the autonomy module."""
    suite = TestSuite("Autonomy Module")
    print("\n▶ Testing Autonomy Module")

    def test_imports():
        from core.sovereign.autonomy import (
            AutonomousLoop,
            DecisionGate,
        )

        assert AutonomousLoop is not None
        assert DecisionGate is not None

    def test_system_metrics():
        from core.sovereign.autonomy import SystemMetrics

        metrics = SystemMetrics(
            snr_score=0.92,
            ihsan_score=0.94,
            latency_ms=150,
            error_rate=0.02,
        )
        assert metrics.health_score() > 0.8
        assert metrics.is_healthy(0.8)
        assert not metrics.is_healthy(0.99)

    def test_decision_candidate():
        from core.sovereign.autonomy import DecisionCandidate, DecisionType

        candidate = DecisionCandidate(
            decision_type=DecisionType.CORRECTIVE,
            action="test_action",
            expected_impact=0.5,
            risk_score=0.2,
            confidence=0.9,
        )
        assert candidate.id is not None
        assert candidate.action == "test_action"

    async def test_decision_gate():
        from core.sovereign.autonomy import (
            DecisionCandidate,
            DecisionGate,
            DecisionType,
            GateResult,
            SystemMetrics,
        )

        gate = DecisionGate(ihsan_threshold=0.95)
        candidate = DecisionCandidate(
            decision_type=DecisionType.ROUTINE,
            action="routine_task",
            risk_score=0.1,
            confidence=0.8,
        )
        metrics = SystemMetrics(snr_score=0.9, ihsan_score=0.9)
        result = await gate.evaluate(candidate, metrics)
        assert result == GateResult.PASS

    async def test_decision_gate_rejection():
        from core.sovereign.autonomy import (
            DecisionCandidate,
            DecisionGate,
            DecisionType,
            GateResult,
            SystemMetrics,
        )

        gate = DecisionGate(ihsan_threshold=0.95)
        candidate = DecisionCandidate(
            decision_type=DecisionType.ROUTINE,
            action="risky_task",
            risk_score=0.9,  # Too risky
            confidence=0.3,  # Too low
        )
        metrics = SystemMetrics(snr_score=0.9, ihsan_score=0.9)
        result = await gate.evaluate(candidate, metrics)
        assert result == GateResult.REJECT

    def test_autonomous_loop_creation():
        from core.sovereign.autonomy import LoopState, create_autonomous_loop

        loop = create_autonomous_loop(snr_threshold=0.95, ihsan_threshold=0.95)
        assert loop.state == LoopState.IDLE
        assert loop.cycle_count == 0
        status = loop.status()
        assert not status["running"]

    async def test_autonomous_loop_cycle():
        from core.sovereign.autonomy import create_autonomous_loop

        loop = create_autonomous_loop()
        result = await loop.run_cycle()
        assert result["cycle"] == 1
        assert "health" in result

    suite.run_test("Import all components", test_imports)
    suite.run_test("SystemMetrics health calculation", test_system_metrics)
    suite.run_test("DecisionCandidate creation", test_decision_candidate)
    suite.run_test("DecisionGate approval", test_decision_gate)
    suite.run_test("DecisionGate rejection", test_decision_gate_rejection)
    suite.run_test("AutonomousLoop creation", test_autonomous_loop_creation)
    suite.run_test("AutonomousLoop cycle execution", test_autonomous_loop_cycle)

    return suite


# =============================================================================
# RUNTIME MODULE TESTS
# =============================================================================


def test_runtime_module():
    """Test the runtime module."""
    suite = TestSuite("Runtime Module")
    print("\n▶ Testing Runtime Module")

    def test_imports():
        from core.sovereign.runtime import (
            RuntimeConfig,
            SovereignRuntime,
        )

        assert SovereignRuntime is not None
        assert RuntimeConfig is not None

    def test_config_defaults():
        from core.sovereign.runtime import RuntimeConfig, RuntimeMode

        config = RuntimeConfig()
        assert config.snr_threshold == 0.95
        assert config.ihsan_threshold == 0.95
        assert config.mode == RuntimeMode.PRODUCTION
        assert config.node_id.startswith("node-")

    def test_metrics():
        from core.sovereign.runtime import RuntimeMetrics

        metrics = RuntimeMetrics()
        assert metrics.success_rate() == 1.0
        assert metrics.cache_hit_rate() == 0.0
        metrics.successful_queries = 9
        metrics.failed_queries = 1
        assert metrics.success_rate() == 0.9

    def test_query_object():
        from core.sovereign.runtime import SovereignQuery

        query = SovereignQuery(
            content="Test query",
            context={"key": "value"},
        )
        assert query.id is not None
        assert query.content == "Test query"
        assert query.require_reasoning

    def test_result_object():
        from core.sovereign.runtime import SovereignResult

        result = SovereignResult(
            query_id="test-123",
            success=True,
            answer="Test answer",
            snr_score=0.96,
            ihsan_score=0.97,
        )
        assert result.meets_ihsan(0.95)
        assert not result.meets_ihsan(0.99)

    async def test_runtime_initialization():
        from core.sovereign.runtime import RuntimeConfig, SovereignRuntime

        config = RuntimeConfig(autonomous_enabled=False)
        runtime = SovereignRuntime(config)
        await runtime.initialize()
        assert runtime._initialized
        status = runtime.status()
        assert status["state"]["initialized"]
        await runtime.shutdown()

    async def test_runtime_query():
        from core.sovereign.runtime import RuntimeConfig, SovereignRuntime

        config = RuntimeConfig(autonomous_enabled=False)
        async with SovereignRuntime.create(config) as runtime:
            result = await runtime.query("What is 2+2?")
            assert result.success
            assert result.answer is not None
            assert result.snr_score > 0
            assert result.total_time_ms > 0

    async def test_runtime_think():
        from core.sovereign.runtime import RuntimeConfig, SovereignRuntime

        config = RuntimeConfig(autonomous_enabled=False)
        async with SovereignRuntime.create(config) as runtime:
            answer = await runtime.think("Hello")
            assert isinstance(answer, str)
            assert len(answer) > 0

    suite.run_test("Import all components", test_imports)
    suite.run_test("RuntimeConfig defaults", test_config_defaults)
    suite.run_test("RuntimeMetrics calculations", test_metrics)
    suite.run_test("SovereignQuery creation", test_query_object)
    suite.run_test("SovereignResult Ihsān check", test_result_object)
    suite.run_test("Runtime initialization", test_runtime_initialization)
    suite.run_test("Runtime query processing", test_runtime_query)
    suite.run_test("Runtime think interface", test_runtime_think)

    return suite


# =============================================================================
# API MODULE TESTS
# =============================================================================


def test_api_module():
    """Test the API module."""
    suite = TestSuite("API Module")
    print("\n▶ Testing API Module")

    def test_imports():
        from core.sovereign.api import (
            QueryRequest,
            SovereignAPIServer,
        )

        assert SovereignAPIServer is not None
        assert QueryRequest is not None

    def test_query_request():
        from core.sovereign.api import QueryRequest

        request = QueryRequest.from_dict(
            {
                "query": "Test query",
                "context": {"key": "value"},
                "max_depth": 5,
            }
        )
        assert request.query == "Test query"
        assert request.max_depth == 5

    def test_query_response():
        from core.sovereign.api import QueryResponse

        response = QueryResponse(
            success=True,
            answer="Test answer",
            snr_score=0.95,
            ihsan_score=0.96,
        )
        data = response.to_dict()
        assert data["success"]
        assert data["quality"]["snr"] == 0.95

    def test_rate_limiter():
        from core.sovereign.api import RateLimiter

        limiter = RateLimiter(requests_per_minute=60, burst_size=5)

        # Should allow first 5 requests (burst)
        for i in range(5):
            assert limiter.check("test-key"), f"Request {i+1} should be allowed"

        # 6th request should be rate limited
        assert not limiter.check("test-key"), "6th request should be limited"

    def test_rate_limiter_different_keys():
        from core.sovereign.api import RateLimiter

        limiter = RateLimiter(requests_per_minute=60, burst_size=2)

        assert limiter.check("key-1")
        assert limiter.check("key-1")
        assert not limiter.check("key-1")  # Limited

        # Different key should have its own bucket
        assert limiter.check("key-2")

    suite.run_test("Import all components", test_imports)
    suite.run_test("QueryRequest parsing", test_query_request)
    suite.run_test("QueryResponse serialization", test_query_response)
    suite.run_test("RateLimiter burst limiting", test_rate_limiter)
    suite.run_test("RateLimiter per-key isolation", test_rate_limiter_different_keys)

    return suite


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


def test_full_integration():
    """Full integration tests."""
    suite = TestSuite("Full Integration")
    print("\n▶ Testing Full Integration")

    async def test_runtime_with_autonomy():
        from core.sovereign.runtime import RuntimeConfig, SovereignRuntime

        config = RuntimeConfig(
            autonomous_enabled=True,
            loop_interval_seconds=0.1,  # Fast for testing
        )
        async with SovereignRuntime.create(config) as runtime:
            # Let autonomous loop run a few cycles
            await asyncio.sleep(0.3)

            status = runtime.status()
            assert status["autonomous"]["running"]

            # Query should still work
            result = await runtime.query("Test during autonomous")
            assert result.success

    async def test_query_caching():
        from core.sovereign.runtime import RuntimeConfig, SovereignRuntime

        config = RuntimeConfig(
            autonomous_enabled=False,
            enable_cache=True,
        )
        async with SovereignRuntime.create(config) as runtime:
            # First query
            result1 = await runtime.query("Cached query test")
            assert not result1.cached

            # Same query should be cached
            result2 = await runtime.query("Cached query test")
            assert result2.cached
            assert result2.total_time_ms < result1.total_time_ms

    async def test_metrics_accumulation():
        from core.sovereign.runtime import RuntimeConfig, SovereignRuntime

        config = RuntimeConfig(autonomous_enabled=False)
        async with SovereignRuntime.create(config) as runtime:
            # Run multiple queries
            for i in range(5):
                await runtime.query(f"Query {i}")

            metrics = runtime.metrics
            assert metrics.total_queries == 5
            assert metrics.successful_queries == 5
            assert metrics.avg_query_time_ms > 0

    async def test_health_status():
        from core.sovereign.runtime import RuntimeConfig, SovereignRuntime

        config = RuntimeConfig(autonomous_enabled=False)
        async with SovereignRuntime.create(config) as runtime:
            status = runtime.status()
            assert status["health"]["status"] in [
                "healthy",
                "degraded",
                "critical",
                "unknown",
            ]

    suite.run_test("Runtime with autonomous loop", test_runtime_with_autonomy)
    suite.run_test("Query caching", test_query_caching)
    suite.run_test("Metrics accumulation", test_metrics_accumulation)
    suite.run_test("Health status reporting", test_health_status)

    return suite


# =============================================================================
# MAIN RUNNER
# =============================================================================


def run_all_tests():
    """Run all test suites."""
    print("=" * 70)
    print("SOVEREIGN ENGINE INTEGRATION TEST SUITE")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")

    suites = []

    # Run test suites
    suites.append(test_autonomy_module())
    suites.append(test_runtime_module())
    suites.append(test_api_module())
    suites.append(test_full_integration())

    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    total_passed = 0
    total_failed = 0
    total_time = 0

    for suite in suites:
        summary = suite.summary()
        total_passed += summary["passed"]
        total_failed += summary["failed"]
        total_time += summary["total_time_ms"]

        status = "✓" if summary["failed"] == 0 else "✗"
        print(
            f"{status} {summary['suite']}: {summary['passed']}/{summary['total']} ({summary['pass_rate']})"
        )

    print("-" * 70)
    total = total_passed + total_failed
    rate = f"{total_passed / total * 100:.1f}%" if total > 0 else "N/A"
    print(f"Total: {total_passed}/{total} passed ({rate})")
    print(f"Time: {total_time:.1f}ms")

    if total_failed == 0:
        print("\n✓ ALL TESTS PASSED - Sovereign Engine verified")
        return 0
    else:
        print(f"\n✗ {total_failed} TESTS FAILED")
        return 1


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    # Setup event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    exit_code = run_all_tests()
    sys.exit(exit_code)
