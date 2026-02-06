# Integration Test Plan

## Phase 5: Apex Integration Testing

**Test Suite**: `tests/integration/test_apex_integration.py`
**Coverage Target**: 90%+
**Lines**: ~400

---

## Test Categories

### Category 1: Unit Tests (Per Module)

| Module | Test File | Tests |
|--------|-----------|-------|
| `social_integration.py` | `test_social_integration.py` | 8 tests |
| `market_integration.py` | `test_market_integration.py` | 10 tests |
| `swarm_integration.py` | `test_swarm_integration.py` | 12 tests |
| `apex_sovereign.py` | `test_apex_sovereign.py` | 15 tests |

### Category 2: Integration Tests

| Scenario | Description | Expected Outcome |
|----------|-------------|------------------|
| Full OODA Cycle | Complete cycle with all phases | All states visited, metrics updated |
| Social → PAT Routing | Trust influences agent selection | Higher trust agents selected |
| Market → Autonomous Action | High SNR signal triggers action | Action executed without approval |
| Swarm Self-Heal | Unhealthy service triggers restart | Service restarted, swarm healthy |
| Degraded Mode | Apex subsystem failure | Graceful fallback, no crash |

### Category 3: Constitutional Tests

| Test | Ihsan Requirement | Expected |
|------|-------------------|----------|
| Action Validation | ≥ 0.95 | Below threshold rejected |
| Autonomy Gating | SNR → Autonomy | Correct level assigned |
| Veto Override | SAT veto | Action blocked |

---

## Test Fixtures

```python
# conftest.py additions

@pytest.fixture
def mock_apex_system():
    """Mock ApexSystem for isolated testing."""
    apex = MagicMock(spec=ApexSystem)
    apex.social = MagicMock(spec=SocialGraph)
    apex.opportunity = MagicMock(spec=OpportunityEngine)
    apex.swarm = MagicMock(spec=SwarmOrchestrator)
    return apex


@pytest.fixture
def apex_sovereign(mock_apex_system):
    """ApexSovereignEntity with mocked Apex."""
    entity = ApexSovereignEntity(node_id="test")
    entity.apex = mock_apex_system
    return entity


@pytest.fixture
def social_bridge():
    """SociallyAwareBridge for testing."""
    return SociallyAwareBridge(node_id="test")


@pytest.fixture
def market_muraqabah():
    """MarketAwareMuraqabah for testing."""
    return MarketAwareMuraqabah(node_id="test")


@pytest.fixture
def hybrid_swarm():
    """HybridSwarmOrchestrator for testing."""
    return HybridSwarmOrchestrator()
```

---

## Test Specifications

### Test Suite: Social Integration

```python
class TestSocialIntegration:
    """Tests for social_integration.py"""

    def test_trust_routing_prefers_trusted_agents(self, social_bridge):
        """Higher trust agents should be selected for tasks."""
        # Setup
        social_bridge.apex.social._relationships["trusted"] = Relationship(trust_score=0.9)
        social_bridge.apex.social._relationships["untrusted"] = Relationship(trust_score=0.3)

        # Execute
        selected = social_bridge.select_agent_for_task(Task(required_capabilities={"reasoning"}))

        # Verify
        assert selected.id == "trusted"

    def test_trust_below_threshold_excluded(self, social_bridge):
        """Agents below trust threshold should be excluded."""
        social_bridge.min_trust_threshold = 0.5
        social_bridge.apex.social._relationships["untrusted"] = Relationship(trust_score=0.2)

        with pytest.raises(NoCapableAgentError):
            social_bridge.select_agent_for_task(Task(required_capabilities={"exclusive"}))

    def test_collaboration_discovery_uses_got(self, social_bridge):
        """Collaboration finding should use Graph-of-Thoughts."""
        # Setup successful collaboration history
        for _ in range(10):
            social_bridge.apex.social.record_interaction(
                peer_id="agent-1",
                interaction_type=InteractionType.COLLABORATION,
                success=True,
                value=100.0
            )

        partners = social_bridge.find_collaboration_partners(Task(requires_collaboration=True))
        assert len(partners) > 0

    async def test_trust_update_on_success(self, social_bridge):
        """Successful task should increase trust."""
        initial = social_bridge.apex.social.get_trust("agent-1")
        await social_bridge.report_task_outcome(
            task=mock_task(),
            agent=MockAgent(id="agent-1"),
            success=True,
            value=50.0
        )
        assert social_bridge.apex.social.get_trust("agent-1") > initial

    async def test_trust_update_on_failure(self, social_bridge):
        """Failed task should decrease trust."""
        social_bridge.apex.social._relationships["agent-1"] = Relationship(trust_score=0.8)
        await social_bridge.report_task_outcome(
            task=mock_task(),
            agent=MockAgent(id="agent-1"),
            success=False,
            value=0.0
        )
        assert social_bridge.apex.social.get_trust("agent-1") < 0.8
```

### Test Suite: Market Integration

```python
class TestMarketIntegration:
    """Tests for market_integration.py"""

    def test_snr_below_threshold_filtered(self, market_muraqabah):
        """Signals below SNR threshold should not create goals."""
        reading = SensorReading(
            domain="financial",
            sensor_type="trading_signal",
            value={"symbol": "TEST/USD"},
            snr=0.60  # Below 0.85 threshold
        )
        goal = market_muraqabah.process_market_reading(reading)
        assert goal is None

    def test_snr_above_threshold_creates_goal(self, market_muraqabah):
        """Signals above SNR threshold should create goals."""
        reading = SensorReading(
            domain="financial",
            sensor_type="trading_signal",
            value={"symbol": "TEST/USD", "type": "buy"},
            snr=0.92
        )
        goal = market_muraqabah.process_market_reading(reading)
        assert goal is not None
        assert goal.domain == "financial"

    def test_snr_to_autonomy_mapping(self, market_muraqabah):
        """SNR scores should map to correct autonomy levels."""
        assert market_muraqabah._snr_to_autonomy(0.99) == AutonomyLevel.AUTOHIGH
        assert market_muraqabah._snr_to_autonomy(0.96) == AutonomyLevel.AUTOMEDIUM
        assert market_muraqabah._snr_to_autonomy(0.91) == AutonomyLevel.AUTOLOW
        assert market_muraqabah._snr_to_autonomy(0.86) == AutonomyLevel.SUGGESTER

    def test_arbitrage_high_urgency(self, market_muraqabah):
        """Arbitrage opportunities should have high urgency."""
        reading = SensorReading(
            domain="financial",
            sensor_type="arbitrage",
            value={"profit_pct": 0.02},
            snr=0.92
        )
        urgency = market_muraqabah._calculate_urgency(reading)
        assert urgency >= 0.9

    def test_stale_data_reduces_snr(self, market_muraqabah):
        """Data older than threshold should have reduced SNR."""
        old_reading = SensorReading(
            domain="financial",
            sensor_type="trading_signal",
            value={"symbol": "TEST/USD"},
            snr=0.95,
            timestamp=datetime.now(timezone.utc) - timedelta(minutes=10)
        )
        # Processing should reduce effective SNR due to staleness
        goal = market_muraqabah.process_market_reading(old_reading)
        if goal:
            assert goal.autonomy_level < AutonomyLevel.AUTOHIGH
```

### Test Suite: Swarm Integration

```python
class TestSwarmIntegration:
    """Tests for swarm_integration.py"""

    async def test_rust_health_check_mapping(self, hybrid_swarm):
        """RustServiceStatus should map to HealthStatus correctly."""
        adapter = RustServiceAdapter("test-svc", mock_lifecycle())

        adapter.rust_lifecycle.check_health = AsyncMock(return_value=RustServiceStatus.RUNNING)
        assert await adapter.health_check() == HealthStatus.HEALTHY

        adapter.rust_lifecycle.check_health = AsyncMock(return_value=RustServiceStatus.FAILED)
        assert await adapter.health_check() == HealthStatus.CRITICAL

    async def test_restart_exponential_backoff(self, hybrid_swarm):
        """Restarts should use exponential backoff."""
        adapter = RustServiceAdapter("test-svc", mock_lifecycle())
        adapter.rust_lifecycle.restart_service = AsyncMock(return_value=False)

        start_times = []
        for _ in range(3):
            start = asyncio.get_event_loop().time()
            await adapter.restart()
            start_times.append(asyncio.get_event_loop().time() - start)

        assert start_times[1] > start_times[0]
        assert start_times[2] > start_times[1]

    async def test_max_restart_attempts(self, hybrid_swarm):
        """Should stop restarting after max attempts."""
        adapter = RustServiceAdapter("test-svc", mock_lifecycle())
        adapter.rust_lifecycle.restart_service = AsyncMock(return_value=False)

        for _ in range(4):
            result = await adapter.restart()

        # 4th attempt should fail without trying
        assert adapter.restart_count == 3
        assert result is False

    async def test_hybrid_scaling_proportional(self, hybrid_swarm):
        """Scaling should affect Python and Rust proportionally."""
        hybrid_swarm._swarms["test"] = Swarm(agents=[
            MockAgent("py-1"), MockAgent("py-2"),
            MockAgent("rust:svc-1")
        ])

        decision = ScalingDecision(
            action=ScalingAction.SCALE_UP,
            current_count=3,
            target_count=6,
            reason="High load"
        )

        await hybrid_swarm.apply_scaling_decision(decision, "test")

        python_count = len([a for a in hybrid_swarm._swarms["test"].agents
                           if not a.id.startswith("rust:")])
        assert python_count >= 4  # ~70% of delta

    async def test_self_heal_restarts_unhealthy(self, hybrid_swarm):
        """Self-heal loop should restart unhealthy services."""
        hybrid_swarm.register_rust_service("unhealthy-svc", AgentConfig(
            agent_type="rust-worker",
            name="unhealthy-svc"
        ))
        hybrid_swarm.rust_adapters["unhealthy-svc"]._last_health = HealthStatus.CRITICAL

        mock_restart = AsyncMock(return_value=True)
        hybrid_swarm.rust_adapters["unhealthy-svc"].restart = mock_restart

        await hybrid_swarm._self_heal_iteration()

        mock_restart.assert_called_once()
```

### Test Suite: Apex Sovereign Entity

```python
class TestApexSovereign:
    """Tests for apex_sovereign.py"""

    async def test_full_ooda_cycle(self, apex_sovereign):
        """Entity should complete all OODA states."""
        states = []
        original_sleep = asyncio.sleep
        asyncio.sleep = AsyncMock()

        for _ in range(8):  # One full cycle
            await apex_sovereign._run_one_state()
            states.append(apex_sovereign.current_state)

        # Should have visited all states
        assert ApexOODAState.OBSERVE in states
        assert ApexOODAState.ACT in states
        assert ApexOODAState.LEARN in states

    async def test_ihsan_filtering(self, apex_sovereign):
        """Goals below Ihsan threshold should be filtered."""
        apex_sovereign.ihsan.validate = Mock(return_value=0.80)

        analysis = {"goals": [ProactiveGoal(description="Test", ihsan_estimate=0.98)]}
        decisions = await apex_sovereign._decide(analysis)

        assert len(decisions) == 0  # Filtered due to actual ihsan < threshold

    async def test_autonomous_action_tracking(self, apex_sovereign):
        """Autonomous actions should be tracked in metrics."""
        decisions = [
            Decision(
                goal=mock_goal(),
                ihsan_score=0.98,
                autonomy_level=AutonomyLevel.AUTOMEDIUM,
                approved=True
            )
        ]

        await apex_sovereign._act(decisions)

        assert apex_sovereign.metrics["autonomous_actions"] >= 1

    async def test_graceful_degradation(self, apex_sovereign):
        """Apex failure should degrade gracefully."""
        apex_sovereign.apex._opportunity = None  # Simulate failure

        # Should not crash
        observations = await apex_sovereign._observe()

        assert "muraqabah" in observations
        assert observations.get("market_signals") == []

    async def test_learning_updates_trust(self, apex_sovereign):
        """Learning phase should update social trust."""
        outcomes = [
            Outcome(
                decision=mock_decision(),
                success=True,
                value=100.0,
                agents_used=[MockAgent("agent-1")]
            )
        ]

        await apex_sovereign._learn(outcomes)

        apex_sovereign.social_bridge.report_task_outcome.assert_called_once()

    def test_ihsan_average_tracking(self, apex_sovereign):
        """Ihsan average should be tracked over time."""
        initial = apex_sovereign.metrics["ihsan_average"]

        # Simulate decisions with high ihsan
        apex_sovereign._update_ihsan_metric([0.98, 0.97, 0.99])

        assert apex_sovereign.metrics["ihsan_average"] > initial
```

---

## Performance Tests

```python
class TestPerformance:
    """Performance tests for Apex integration."""

    @pytest.mark.slow
    async def test_ooda_cycle_under_1_second(self, apex_sovereign):
        """Single OODA cycle should complete in under 1 second."""
        start = time.time()
        await apex_sovereign._run_one_cycle()
        duration = time.time() - start

        assert duration < 1.0

    @pytest.mark.slow
    async def test_100_cycles_stable(self, apex_sovereign):
        """100 cycles should run without memory leak or error."""
        initial_memory = get_memory_usage()

        for _ in range(100):
            await apex_sovereign._run_one_cycle()

        final_memory = get_memory_usage()

        # Memory should not grow significantly (< 10%)
        assert final_memory < initial_memory * 1.1

    @pytest.mark.slow
    async def test_concurrent_observations(self, apex_sovereign):
        """Multiple observations should run concurrently."""
        start = time.time()

        await asyncio.gather(
            apex_sovereign._observe(),
            apex_sovereign._observe(),
            apex_sovereign._observe(),
        )

        duration = time.time() - start

        # Should not take 3x time (parallel execution)
        assert duration < 2.0
```

---

## Test Commands

```bash
# Run all Apex integration tests
pytest tests/integration/test_apex_integration.py -v

# Run with coverage
pytest tests/integration/test_apex_integration.py --cov=core.sovereign --cov-report=html

# Run performance tests
pytest tests/integration/test_apex_integration.py -m slow -v

# Run specific test class
pytest tests/integration/test_apex_integration.py::TestApexSovereign -v
```

---

## Coverage Targets

| Module | Target | Critical Paths |
|--------|--------|----------------|
| `social_integration.py` | 90% | Trust routing, update |
| `market_integration.py` | 90% | SNR filtering, autonomy mapping |
| `swarm_integration.py` | 85% | Health check, scaling |
| `apex_sovereign.py` | 90% | OODA cycle, Ihsan validation |

---

## File Output

**Target**: `tests/integration/test_apex_integration.py`
**Lines**: ~400
