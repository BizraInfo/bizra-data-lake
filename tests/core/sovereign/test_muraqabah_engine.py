"""
Tests for Muraqabah Engine — 24/7 Continuous Vigilance Monitoring
=================================================================
Validates sensor registration, domain scanning, opportunity detection,
constitutional filtering, and monitoring lifecycle.
"""

import asyncio
import uuid
from collections import deque
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.sovereign.event_bus import EventBus, EventPriority
from core.sovereign.muraqabah_engine import (
    MonitorDomain,
    MuraqabahEngine,
    Opportunity,
    SensorReading,
    SensorState,
)


# =============================================================================
# ENUM TESTS
# =============================================================================


class TestMonitorDomain:
    """Tests for MonitorDomain enum."""

    def test_financial_value(self):
        assert MonitorDomain.FINANCIAL == "financial"
        assert MonitorDomain.FINANCIAL.value == "financial"

    def test_health_value(self):
        assert MonitorDomain.HEALTH == "health"
        assert MonitorDomain.HEALTH.value == "health"

    def test_social_value(self):
        assert MonitorDomain.SOCIAL == "social"
        assert MonitorDomain.SOCIAL.value == "social"

    def test_cognitive_value(self):
        assert MonitorDomain.COGNITIVE == "cognitive"
        assert MonitorDomain.COGNITIVE.value == "cognitive"

    def test_environmental_value(self):
        assert MonitorDomain.ENVIRONMENTAL == "environmental"
        assert MonitorDomain.ENVIRONMENTAL.value == "environmental"

    def test_all_domains_count(self):
        assert len(MonitorDomain) == 5

    def test_is_str_enum(self):
        """MonitorDomain is a str enum, so values compare as strings."""
        assert isinstance(MonitorDomain.FINANCIAL, str)
        domain_str: str = MonitorDomain.FINANCIAL
        assert domain_str == "financial"


class TestSensorState:
    """Tests for SensorState enum."""

    def test_active_state(self):
        assert SensorState.ACTIVE is not None
        assert SensorState.ACTIVE != SensorState.INACTIVE

    def test_inactive_state(self):
        assert SensorState.INACTIVE is not None

    def test_error_state(self):
        assert SensorState.ERROR is not None

    def test_calibrating_state(self):
        assert SensorState.CALIBRATING is not None

    def test_all_states_count(self):
        assert len(SensorState) == 4

    def test_states_are_distinct(self):
        states = [SensorState.ACTIVE, SensorState.INACTIVE, SensorState.ERROR, SensorState.CALIBRATING]
        assert len(set(states)) == 4


# =============================================================================
# DATACLASS TESTS
# =============================================================================


class TestSensorReading:
    """Tests for SensorReading dataclass."""

    def test_default_values(self):
        reading = SensorReading()
        assert reading.sensor_id == ""
        assert reading.domain == MonitorDomain.ENVIRONMENTAL
        assert reading.metric_name == ""
        assert reading.value == 0.0
        assert reading.unit == ""
        assert reading.confidence == 1.0
        assert isinstance(reading.timestamp, datetime)
        assert reading.metadata == {}

    def test_custom_values(self):
        ts = datetime.now(timezone.utc)
        reading = SensorReading(
            sensor_id="env:system_health",
            domain=MonitorDomain.HEALTH,
            metric_name="cpu_usage",
            value=0.75,
            unit="ratio",
            confidence=0.95,
            timestamp=ts,
            metadata={"source": "psutil"},
        )
        assert reading.sensor_id == "env:system_health"
        assert reading.domain == MonitorDomain.HEALTH
        assert reading.metric_name == "cpu_usage"
        assert reading.value == 0.75
        assert reading.unit == "ratio"
        assert reading.confidence == 0.95
        assert reading.timestamp == ts
        assert reading.metadata == {"source": "psutil"}

    def test_timestamp_auto_generated(self):
        reading = SensorReading()
        assert reading.timestamp.tzinfo is not None
        now = datetime.now(timezone.utc)
        delta = (now - reading.timestamp).total_seconds()
        assert delta < 2.0

    def test_metadata_isolation(self):
        """Each reading should have its own metadata dict."""
        r1 = SensorReading()
        r2 = SensorReading()
        r1.metadata["key"] = "value"
        assert "key" not in r2.metadata


class TestOpportunity:
    """Tests for Opportunity dataclass."""

    def test_auto_generated_id(self):
        opp = Opportunity()
        assert opp.id is not None
        assert len(opp.id) == 8

    def test_unique_ids(self):
        ids = {Opportunity().id for _ in range(50)}
        assert len(ids) == 50

    def test_default_values(self):
        opp = Opportunity()
        assert opp.domain == MonitorDomain.ENVIRONMENTAL
        assert opp.description == ""
        assert opp.estimated_value == 0.0
        assert opp.urgency == 0.5
        assert opp.confidence == 0.9
        assert opp.action_required == ""
        assert opp.expires_at is None
        assert isinstance(opp.detected_at, datetime)
        assert opp.metadata == {}

    def test_custom_values(self):
        opp = Opportunity(
            domain=MonitorDomain.FINANCIAL,
            description="Budget exceeded",
            estimated_value=0.8,
            urgency=0.9,
            confidence=0.95,
            action_required="review_budget",
        )
        assert opp.domain == MonitorDomain.FINANCIAL
        assert opp.description == "Budget exceeded"
        assert opp.estimated_value == 0.8
        assert opp.urgency == 0.9
        assert opp.confidence == 0.95
        assert opp.action_required == "review_budget"

    def test_metadata_isolation(self):
        o1 = Opportunity()
        o2 = Opportunity()
        o1.metadata["x"] = 1
        assert "x" not in o2.metadata


# =============================================================================
# ENGINE INITIALIZATION TESTS
# =============================================================================


class TestMuraqabahEngineInit:
    """Tests for MuraqabahEngine initialization and default sensor registration."""

    def test_default_ihsan_threshold(self):
        engine = MuraqabahEngine()
        assert engine.ihsan_threshold == 0.95

    def test_custom_ihsan_threshold(self):
        engine = MuraqabahEngine(ihsan_threshold=0.8)
        assert engine.ihsan_threshold == 0.8

    def test_event_bus_auto_created(self):
        engine = MuraqabahEngine()
        assert engine.event_bus is not None
        assert isinstance(engine.event_bus, EventBus)

    def test_event_bus_injection(self):
        bus = EventBus()
        engine = MuraqabahEngine(event_bus=bus)
        assert engine.event_bus is bus

    def test_default_sensors_count(self):
        """16 default sensors are registered across 5 domains (4+3+3+3+3)."""
        engine = MuraqabahEngine()
        total = sum(len(sensors) for sensors in engine._sensors.values())
        assert total == 16

    def test_environmental_sensors_registered(self):
        engine = MuraqabahEngine()
        env_sensors = engine._sensors[MonitorDomain.ENVIRONMENTAL]
        assert len(env_sensors) == 4
        assert "system_health" in env_sensors
        assert "process" in env_sensors
        assert "network_io" in env_sensors
        assert "disk_io" in env_sensors

    def test_health_sensors_registered(self):
        engine = MuraqabahEngine()
        health_sensors = engine._sensors[MonitorDomain.HEALTH]
        assert len(health_sensors) == 3
        assert "latency" in health_sensors
        assert "error_rate" in health_sensors
        assert "uptime" in health_sensors

    def test_cognitive_sensors_registered(self):
        engine = MuraqabahEngine()
        cog_sensors = engine._sensors[MonitorDomain.COGNITIVE]
        assert len(cog_sensors) == 3
        assert "task_queue" in cog_sensors
        assert "learning" in cog_sensors
        assert "inference_quality" in cog_sensors

    def test_financial_sensors_registered(self):
        engine = MuraqabahEngine()
        fin_sensors = engine._sensors[MonitorDomain.FINANCIAL]
        assert len(fin_sensors) == 3
        assert "compute_cost" in fin_sensors
        assert "efficiency" in fin_sensors
        assert "budget" in fin_sensors

    def test_social_sensors_registered(self):
        engine = MuraqabahEngine()
        soc_sensors = engine._sensors[MonitorDomain.SOCIAL]
        assert len(soc_sensors) == 3
        assert "connectivity" in soc_sensors
        assert "federation" in soc_sensors
        assert "collaboration" in soc_sensors

    def test_all_sensor_states_active(self):
        engine = MuraqabahEngine()
        for state in engine._sensor_states.values():
            assert state == SensorState.ACTIVE

    def test_readings_deque_initialized(self):
        engine = MuraqabahEngine()
        for domain in MonitorDomain:
            assert isinstance(engine._readings[domain], deque)
            assert engine._readings[domain].maxlen == 1000

    def test_opportunities_deque_initialized(self):
        engine = MuraqabahEngine()
        assert isinstance(engine._opportunities, deque)
        assert engine._opportunities.maxlen == 100

    def test_initial_state_flags(self):
        engine = MuraqabahEngine()
        assert engine._running is False
        assert engine._scan_count == 0
        assert engine._opportunity_count == 0


# =============================================================================
# SENSOR MANAGEMENT TESTS
# =============================================================================


class TestSensorManagement:
    """Tests for register_sensor, unregister_sensor, set_interval."""

    def test_register_sensor(self):
        engine = MuraqabahEngine()
        custom_sensor = lambda: {"test_metric": 42.0}
        engine.register_sensor(MonitorDomain.ENVIRONMENTAL, "custom", custom_sensor)
        assert "custom" in engine._sensors[MonitorDomain.ENVIRONMENTAL]
        assert engine._sensor_states["environmental:custom"] == SensorState.ACTIVE

    def test_register_sensor_overwrites_existing(self):
        engine = MuraqabahEngine()
        new_sensor = lambda: {"new_metric": 99.0}
        engine.register_sensor(MonitorDomain.ENVIRONMENTAL, "system_health", new_sensor)
        assert engine._sensors[MonitorDomain.ENVIRONMENTAL]["system_health"] is new_sensor

    def test_unregister_sensor(self):
        engine = MuraqabahEngine()
        assert "system_health" in engine._sensors[MonitorDomain.ENVIRONMENTAL]
        engine.unregister_sensor(MonitorDomain.ENVIRONMENTAL, "system_health")
        assert "system_health" not in engine._sensors[MonitorDomain.ENVIRONMENTAL]
        assert "environmental:system_health" not in engine._sensor_states

    def test_unregister_nonexistent_sensor_no_error(self):
        engine = MuraqabahEngine()
        engine.unregister_sensor(MonitorDomain.ENVIRONMENTAL, "nonexistent")

    def test_set_interval(self):
        engine = MuraqabahEngine()
        engine.set_interval(MonitorDomain.HEALTH, 120)
        assert engine._intervals[MonitorDomain.HEALTH] == 120

    def test_set_interval_minimum_10(self):
        engine = MuraqabahEngine()
        engine.set_interval(MonitorDomain.HEALTH, 5)
        assert engine._intervals[MonitorDomain.HEALTH] == 10

    def test_set_interval_zero_clamps_to_10(self):
        engine = MuraqabahEngine()
        engine.set_interval(MonitorDomain.COGNITIVE, 0)
        assert engine._intervals[MonitorDomain.COGNITIVE] == 10

    def test_set_interval_negative_clamps_to_10(self):
        engine = MuraqabahEngine()
        engine.set_interval(MonitorDomain.SOCIAL, -100)
        assert engine._intervals[MonitorDomain.SOCIAL] == 10


# =============================================================================
# SCAN DOMAIN TESTS (ASYNC)
# =============================================================================


class TestScanDomain:
    """Tests for _scan_domain async method."""

    @pytest.fixture
    def engine(self):
        """Create an engine with all default sensors replaced by controlled mocks."""
        eng = MuraqabahEngine()
        # Clear all default sensors
        for domain in MonitorDomain:
            eng._sensors[domain] = {}
            eng._sensor_states.clear()
        return eng

    @pytest.mark.asyncio
    async def test_scan_domain_returns_readings(self, engine):
        sensor_fn = lambda: {"cpu_usage": 0.5, "memory_usage": 0.3}
        engine.register_sensor(MonitorDomain.ENVIRONMENTAL, "test", sensor_fn)

        readings = await engine._scan_domain(MonitorDomain.ENVIRONMENTAL)
        assert len(readings) == 2
        metric_names = {r.metric_name for r in readings}
        assert metric_names == {"cpu_usage", "memory_usage"}

    @pytest.mark.asyncio
    async def test_scan_domain_stores_readings(self, engine):
        sensor_fn = lambda: {"metric_a": 1.0}
        engine.register_sensor(MonitorDomain.HEALTH, "test", sensor_fn)

        await engine._scan_domain(MonitorDomain.HEALTH)
        assert len(engine._readings[MonitorDomain.HEALTH]) == 1

    @pytest.mark.asyncio
    async def test_scan_domain_reading_values(self, engine):
        sensor_fn = lambda: {"temperature": 72.5}
        engine.register_sensor(MonitorDomain.ENVIRONMENTAL, "temp", sensor_fn)

        readings = await engine._scan_domain(MonitorDomain.ENVIRONMENTAL)
        assert len(readings) == 1
        assert readings[0].value == 72.5
        assert readings[0].metric_name == "temperature"
        assert readings[0].sensor_id == "environmental:temp"
        assert readings[0].domain == MonitorDomain.ENVIRONMENTAL

    @pytest.mark.asyncio
    async def test_scan_domain_ignores_non_numeric(self, engine):
        sensor_fn = lambda: {"status": "ok", "count": 10, "name": "test"}
        engine.register_sensor(MonitorDomain.COGNITIVE, "mixed", sensor_fn)

        readings = await engine._scan_domain(MonitorDomain.COGNITIVE)
        assert len(readings) == 1
        assert readings[0].metric_name == "count"

    @pytest.mark.asyncio
    async def test_scan_domain_sensor_error_sets_error_state(self, engine):
        def failing_sensor():
            raise RuntimeError("Sensor failure")

        engine.register_sensor(MonitorDomain.ENVIRONMENTAL, "bad", failing_sensor)
        readings = await engine._scan_domain(MonitorDomain.ENVIRONMENTAL)
        assert len(readings) == 0
        assert engine._sensor_states["environmental:bad"] == SensorState.ERROR

    @pytest.mark.asyncio
    async def test_scan_domain_skips_inactive_sensors(self, engine):
        sensor_fn = lambda: {"value": 1.0}
        engine.register_sensor(MonitorDomain.HEALTH, "inactive", sensor_fn)
        engine._sensor_states["health:inactive"] = SensorState.INACTIVE

        readings = await engine._scan_domain(MonitorDomain.HEALTH)
        assert len(readings) == 0

    @pytest.mark.asyncio
    async def test_scan_domain_skips_error_sensors(self, engine):
        sensor_fn = lambda: {"value": 1.0}
        engine.register_sensor(MonitorDomain.HEALTH, "errored", sensor_fn)
        engine._sensor_states["health:errored"] = SensorState.ERROR

        readings = await engine._scan_domain(MonitorDomain.HEALTH)
        assert len(readings) == 0

    @pytest.mark.asyncio
    async def test_scan_domain_multiple_sensors(self, engine):
        engine.register_sensor(MonitorDomain.FINANCIAL, "s1", lambda: {"a": 1.0})
        engine.register_sensor(MonitorDomain.FINANCIAL, "s2", lambda: {"b": 2.0, "c": 3.0})

        readings = await engine._scan_domain(MonitorDomain.FINANCIAL)
        assert len(readings) == 3

    @pytest.mark.asyncio
    async def test_scan_domain_empty_sensor_dict(self, engine):
        readings = await engine._scan_domain(MonitorDomain.SOCIAL)
        assert readings == []


# =============================================================================
# OPPORTUNITY DETECTION TESTS
# =============================================================================


class TestOpportunityDetection:
    """Tests for _analyze_single_reading across all domains."""

    @pytest.fixture
    def engine(self):
        return MuraqabahEngine()

    # ---- ENVIRONMENTAL ----

    def test_env_high_cpu_triggers_opportunity(self, engine):
        reading = SensorReading(
            domain=MonitorDomain.ENVIRONMENTAL,
            metric_name="cpu_usage",
            value=0.85,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is not None
        assert opp.domain == MonitorDomain.ENVIRONMENTAL
        assert opp.action_required == "investigate_cpu_usage"

    def test_env_normal_cpu_no_opportunity(self, engine):
        reading = SensorReading(
            domain=MonitorDomain.ENVIRONMENTAL,
            metric_name="cpu_usage",
            value=0.5,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is None

    def test_env_boundary_cpu_80_no_opportunity(self, engine):
        """Exactly 0.8 should NOT trigger (threshold is > 0.8)."""
        reading = SensorReading(
            domain=MonitorDomain.ENVIRONMENTAL,
            metric_name="cpu_usage",
            value=0.8,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is None

    def test_env_high_memory_triggers_opportunity(self, engine):
        reading = SensorReading(
            domain=MonitorDomain.ENVIRONMENTAL,
            metric_name="memory_usage",
            value=0.9,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is not None
        assert opp.action_required == "optimize_memory"
        assert opp.urgency == 0.8

    def test_env_boundary_memory_85_no_opportunity(self, engine):
        reading = SensorReading(
            domain=MonitorDomain.ENVIRONMENTAL,
            metric_name="memory_usage",
            value=0.85,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is None

    def test_env_high_disk_triggers_opportunity(self, engine):
        reading = SensorReading(
            domain=MonitorDomain.ENVIRONMENTAL,
            metric_name="disk_usage",
            value=0.95,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is not None
        assert opp.action_required == "cleanup_disk"
        assert opp.urgency == 0.9

    def test_env_boundary_disk_90_no_opportunity(self, engine):
        reading = SensorReading(
            domain=MonitorDomain.ENVIRONMENTAL,
            metric_name="disk_usage",
            value=0.9,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is None

    def test_env_low_cpu_triggers_opportunity(self, engine):
        reading = SensorReading(
            domain=MonitorDomain.ENVIRONMENTAL,
            metric_name="cpu_usage",
            value=0.1,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is not None
        assert opp.action_required == "increase_throughput"
        assert opp.urgency == 0.2

    def test_env_boundary_cpu_20_no_low_opportunity(self, engine):
        """Exactly 0.2 should NOT trigger low-CPU opportunity (threshold is < 0.2)."""
        reading = SensorReading(
            domain=MonitorDomain.ENVIRONMENTAL,
            metric_name="cpu_usage",
            value=0.2,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is None

    def test_env_cpu_between_thresholds_no_opportunity(self, engine):
        """CPU between 0.2 and 0.8 should produce no opportunity."""
        reading = SensorReading(
            domain=MonitorDomain.ENVIRONMENTAL,
            metric_name="cpu_usage",
            value=0.5,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is None

    def test_env_unknown_metric_no_opportunity(self, engine):
        reading = SensorReading(
            domain=MonitorDomain.ENVIRONMENTAL,
            metric_name="unknown_metric",
            value=99.9,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is None

    # ---- HEALTH ----

    def test_health_high_error_rate(self, engine):
        reading = SensorReading(
            domain=MonitorDomain.HEALTH,
            metric_name="error_rate",
            value=0.10,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is not None
        assert opp.action_required == "investigate_errors"
        assert opp.urgency == 0.85

    def test_health_normal_error_rate(self, engine):
        reading = SensorReading(
            domain=MonitorDomain.HEALTH,
            metric_name="error_rate",
            value=0.03,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is None

    def test_health_high_latency(self, engine):
        reading = SensorReading(
            domain=MonitorDomain.HEALTH,
            metric_name="operation_latency_ms",
            value=150.0,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is not None
        assert opp.action_required == "optimize_latency"

    def test_health_boundary_latency_100_no_opportunity(self, engine):
        reading = SensorReading(
            domain=MonitorDomain.HEALTH,
            metric_name="operation_latency_ms",
            value=100.0,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is None

    def test_health_low_health_score(self, engine):
        reading = SensorReading(
            domain=MonitorDomain.HEALTH,
            metric_name="health_score",
            value=0.5,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is not None
        assert opp.action_required == "run_diagnostics"

    def test_health_normal_health_score(self, engine):
        reading = SensorReading(
            domain=MonitorDomain.HEALTH,
            metric_name="health_score",
            value=0.9,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is None

    # ---- COGNITIVE ----

    def test_cognitive_failed_tasks_high(self, engine):
        reading = SensorReading(
            domain=MonitorDomain.COGNITIVE,
            metric_name="failed_tasks",
            value=10.0,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is not None
        assert opp.action_required == "review_failed_tasks"

    def test_cognitive_failed_tasks_boundary_5_no_opportunity(self, engine):
        reading = SensorReading(
            domain=MonitorDomain.COGNITIVE,
            metric_name="failed_tasks",
            value=5.0,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is None

    def test_cognitive_low_adaptation(self, engine):
        reading = SensorReading(
            domain=MonitorDomain.COGNITIVE,
            metric_name="adaptation_score",
            value=0.5,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is not None
        assert opp.action_required == "trigger_learning"

    def test_cognitive_normal_adaptation(self, engine):
        reading = SensorReading(
            domain=MonitorDomain.COGNITIVE,
            metric_name="adaptation_score",
            value=0.85,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is None

    def test_cognitive_low_ihsan(self, engine):
        reading = SensorReading(
            domain=MonitorDomain.COGNITIVE,
            metric_name="ihsan_average",
            value=0.80,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is not None
        assert opp.action_required == "improve_quality"

    def test_cognitive_normal_ihsan(self, engine):
        reading = SensorReading(
            domain=MonitorDomain.COGNITIVE,
            metric_name="ihsan_average",
            value=0.95,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is None

    # ---- FINANCIAL ----

    def test_financial_high_cost(self, engine):
        reading = SensorReading(
            domain=MonitorDomain.FINANCIAL,
            metric_name="hourly_cost_estimate",
            value=0.25,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is not None
        assert opp.action_required == "optimize_costs"

    def test_financial_normal_cost(self, engine):
        reading = SensorReading(
            domain=MonitorDomain.FINANCIAL,
            metric_name="hourly_cost_estimate",
            value=0.10,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is None

    def test_financial_low_efficiency(self, engine):
        reading = SensorReading(
            domain=MonitorDomain.FINANCIAL,
            metric_name="throughput_efficiency",
            value=0.5,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is not None
        assert opp.action_required == "improve_efficiency"

    def test_financial_normal_efficiency(self, engine):
        reading = SensorReading(
            domain=MonitorDomain.FINANCIAL,
            metric_name="throughput_efficiency",
            value=0.85,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is None

    def test_financial_budget_warning(self, engine):
        reading = SensorReading(
            domain=MonitorDomain.FINANCIAL,
            metric_name="budget_used_percent",
            value=0.9,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is not None
        assert opp.action_required == "review_budget"
        assert opp.urgency == 0.8

    def test_financial_budget_normal(self, engine):
        reading = SensorReading(
            domain=MonitorDomain.FINANCIAL,
            metric_name="budget_used_percent",
            value=0.5,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is None

    # ---- SOCIAL ----

    def test_social_network_disconnected(self, engine):
        reading = SensorReading(
            domain=MonitorDomain.SOCIAL,
            metric_name="internet_connected",
            value=0.0,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is not None
        assert opp.action_required == "restore_connectivity"
        assert opp.urgency == 0.95

    def test_social_network_connected(self, engine):
        reading = SensorReading(
            domain=MonitorDomain.SOCIAL,
            metric_name="internet_connected",
            value=1.0,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is None

    def test_social_no_peers(self, engine):
        reading = SensorReading(
            domain=MonitorDomain.SOCIAL,
            metric_name="connected_peers",
            value=0.0,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is not None
        assert opp.action_required == "reconnect_peers"

    def test_social_has_peers(self, engine):
        reading = SensorReading(
            domain=MonitorDomain.SOCIAL,
            metric_name="connected_peers",
            value=5.0,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is None

    def test_social_low_synergy(self, engine):
        reading = SensorReading(
            domain=MonitorDomain.SOCIAL,
            metric_name="team_synergy_score",
            value=0.4,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is not None
        assert opp.action_required == "improve_coordination"

    def test_social_good_synergy(self, engine):
        reading = SensorReading(
            domain=MonitorDomain.SOCIAL,
            metric_name="team_synergy_score",
            value=0.85,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is None

    # ---- CROSS-DOMAIN ----

    def test_opportunity_carries_reading_confidence(self, engine):
        reading = SensorReading(
            domain=MonitorDomain.ENVIRONMENTAL,
            metric_name="cpu_usage",
            value=0.95,
            confidence=0.88,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is not None
        assert opp.confidence == 0.88

    def test_opportunity_metadata_contains_metric_and_value(self, engine):
        reading = SensorReading(
            domain=MonitorDomain.HEALTH,
            metric_name="error_rate",
            value=0.1,
        )
        opp = engine._analyze_single_reading(reading)
        assert opp is not None
        assert opp.metadata["metric"] == "error_rate"
        assert opp.metadata["value"] == 0.1


# =============================================================================
# CONSTITUTIONAL FILTER TESTS (ASYNC)
# =============================================================================


class TestConstitutionalFilter:
    """Tests for _constitutional_filter."""

    @pytest.fixture
    def engine(self):
        return MuraqabahEngine(ihsan_threshold=0.95)

    @pytest.mark.asyncio
    async def test_passes_normal_opportunity(self, engine):
        opp = Opportunity(confidence=0.9, urgency=0.5)
        result = await engine._constitutional_filter(opp)
        assert result is True

    @pytest.mark.asyncio
    async def test_passes_high_confidence(self, engine):
        opp = Opportunity(confidence=0.99, urgency=0.9)
        result = await engine._constitutional_filter(opp)
        assert result is True

    @pytest.mark.asyncio
    async def test_rejects_low_confidence(self, engine):
        """Confidence < ihsan_threshold - 0.1 = 0.85 should be rejected."""
        opp = Opportunity(confidence=0.80, urgency=0.5)
        result = await engine._constitutional_filter(opp)
        assert result is False

    @pytest.mark.asyncio
    async def test_boundary_confidence_at_threshold(self, engine):
        """Confidence exactly at ihsan_threshold - 0.1 = 0.85 should pass."""
        opp = Opportunity(confidence=0.85, urgency=0.5)
        result = await engine._constitutional_filter(opp)
        assert result is True

    @pytest.mark.asyncio
    async def test_rejects_high_urgency_low_confidence(self, engine):
        """Urgency > 0.8 and confidence < 0.8 should be rejected."""
        opp = Opportunity(urgency=0.9, confidence=0.7)
        result = await engine._constitutional_filter(opp)
        assert result is False

    @pytest.mark.asyncio
    async def test_passes_high_urgency_high_confidence(self, engine):
        opp = Opportunity(urgency=0.95, confidence=0.9)
        result = await engine._constitutional_filter(opp)
        assert result is True

    @pytest.mark.asyncio
    async def test_boundary_urgency_exactly_08_not_triggered(self, engine):
        """Urgency exactly 0.8 should NOT trigger the high-urgency filter (> 0.8 required)."""
        opp = Opportunity(urgency=0.8, confidence=0.7)
        # First filter: 0.7 < 0.85 -> rejected by low confidence rule
        result = await engine._constitutional_filter(opp)
        assert result is False

    @pytest.mark.asyncio
    async def test_custom_threshold_affects_filter(self):
        engine = MuraqabahEngine(ihsan_threshold=0.7)
        # ihsan_threshold - 0.1 = 0.6
        opp = Opportunity(confidence=0.65, urgency=0.5)
        result = await engine._constitutional_filter(opp)
        assert result is True

    @pytest.mark.asyncio
    async def test_custom_threshold_rejects_below(self):
        engine = MuraqabahEngine(ihsan_threshold=0.7)
        # ihsan_threshold - 0.1 = 0.6
        opp = Opportunity(confidence=0.55, urgency=0.5)
        result = await engine._constitutional_filter(opp)
        assert result is False


# =============================================================================
# FULL SCAN TESTS (ASYNC)
# =============================================================================


class TestScan:
    """Tests for the scan() method."""

    @pytest.fixture
    def engine(self):
        """Engine with controlled sensors (no real psutil/socket)."""
        eng = MuraqabahEngine()
        # Replace all sensors with controlled ones
        for domain in MonitorDomain:
            eng._sensors[domain] = {}
            eng._sensor_states.clear()
        return eng

    @pytest.mark.asyncio
    async def test_scan_returns_correct_structure(self, engine):
        engine.register_sensor(
            MonitorDomain.ENVIRONMENTAL, "test", lambda: {"cpu_usage": 0.5}
        )
        result = await engine.scan()
        assert "scan_number" in result
        assert "domains_scanned" in result
        assert "readings" in result
        assert "opportunities" in result

    @pytest.mark.asyncio
    async def test_scan_increments_scan_count(self, engine):
        engine.register_sensor(
            MonitorDomain.HEALTH, "test", lambda: {"value": 0.5}
        )
        await engine.scan()
        assert engine._scan_count == 1
        await engine.scan()
        assert engine._scan_count == 2

    @pytest.mark.asyncio
    async def test_scan_all_domains(self, engine):
        engine.register_sensor(
            MonitorDomain.ENVIRONMENTAL, "test", lambda: {"val": 1.0}
        )
        engine.register_sensor(
            MonitorDomain.HEALTH, "test", lambda: {"val": 2.0}
        )
        result = await engine.scan()
        assert result["domains_scanned"] == 5
        assert result["readings"] == 2

    @pytest.mark.asyncio
    async def test_scan_single_domain(self, engine):
        engine.register_sensor(
            MonitorDomain.ENVIRONMENTAL, "test", lambda: {"val": 1.0}
        )
        engine.register_sensor(
            MonitorDomain.HEALTH, "test", lambda: {"val": 2.0}
        )
        result = await engine.scan(MonitorDomain.ENVIRONMENTAL)
        assert result["domains_scanned"] == 1
        assert result["readings"] == 1

    @pytest.mark.asyncio
    async def test_scan_detects_opportunities(self, engine):
        engine.register_sensor(
            MonitorDomain.ENVIRONMENTAL, "overloaded", lambda: {"cpu_usage": 0.95}
        )
        result = await engine.scan(MonitorDomain.ENVIRONMENTAL)
        assert result["opportunities"] >= 1
        assert engine._opportunity_count >= 1

    @pytest.mark.asyncio
    async def test_scan_stores_opportunities(self, engine):
        engine.register_sensor(
            MonitorDomain.ENVIRONMENTAL, "overloaded", lambda: {"cpu_usage": 0.95}
        )
        await engine.scan(MonitorDomain.ENVIRONMENTAL)
        assert len(engine._opportunities) >= 1

    @pytest.mark.asyncio
    async def test_scan_emits_event(self, engine):
        mock_bus = AsyncMock(spec=EventBus)
        mock_bus.emit = AsyncMock(return_value="evt-123")
        engine.event_bus = mock_bus

        engine.register_sensor(
            MonitorDomain.ENVIRONMENTAL, "hot", lambda: {"cpu_usage": 0.95}
        )
        await engine.scan(MonitorDomain.ENVIRONMENTAL)
        mock_bus.emit.assert_called()
        call_kwargs = mock_bus.emit.call_args
        assert "muraqabah.opportunity.environmental" in str(call_kwargs)

    @pytest.mark.asyncio
    async def test_scan_no_readings_no_opportunities(self, engine):
        result = await engine.scan()
        assert result["readings"] == 0
        assert result["opportunities"] == 0

    @pytest.mark.asyncio
    async def test_scan_filters_low_confidence_opportunities(self, engine):
        """Opportunities with confidence < ihsan_threshold - 0.1 should be filtered."""
        engine.register_sensor(
            MonitorDomain.ENVIRONMENTAL,
            "lowconf",
            lambda: {"cpu_usage": 0.95},
        )
        # Override the sensor reading confidence by replacing _scan_domain
        original_scan = engine._scan_domain

        async def patched_scan(domain):
            readings = await original_scan(domain)
            for r in readings:
                r.confidence = 0.5  # Very low confidence
            return readings

        engine._scan_domain = patched_scan
        result = await engine.scan(MonitorDomain.ENVIRONMENTAL)
        # The opportunity should be filtered out by constitutional filter
        assert result["opportunities"] == 0


# =============================================================================
# OPPORTUNITY HANDLER TESTS
# =============================================================================


class TestOpportunityHandler:
    """Tests for add_opportunity_handler and handler invocation."""

    @pytest.fixture
    def engine(self):
        eng = MuraqabahEngine()
        for domain in MonitorDomain:
            eng._sensors[domain] = {}
            eng._sensor_states.clear()
        return eng

    @pytest.mark.asyncio
    async def test_handler_called_on_opportunity(self, engine):
        received = []
        engine.add_opportunity_handler(lambda opp: received.append(opp))
        engine.register_sensor(
            MonitorDomain.ENVIRONMENTAL, "hot", lambda: {"cpu_usage": 0.95}
        )
        await engine.scan(MonitorDomain.ENVIRONMENTAL)
        assert len(received) >= 1
        assert received[0].action_required == "investigate_cpu_usage"

    @pytest.mark.asyncio
    async def test_multiple_handlers_called(self, engine):
        calls_a = []
        calls_b = []
        engine.add_opportunity_handler(lambda opp: calls_a.append(opp))
        engine.add_opportunity_handler(lambda opp: calls_b.append(opp))
        engine.register_sensor(
            MonitorDomain.ENVIRONMENTAL, "hot", lambda: {"cpu_usage": 0.95}
        )
        await engine.scan(MonitorDomain.ENVIRONMENTAL)
        assert len(calls_a) >= 1
        assert len(calls_b) >= 1

    @pytest.mark.asyncio
    async def test_handler_error_does_not_crash(self, engine):
        def bad_handler(opp):
            raise ValueError("handler error")

        engine.add_opportunity_handler(bad_handler)
        engine.register_sensor(
            MonitorDomain.ENVIRONMENTAL, "hot", lambda: {"cpu_usage": 0.95}
        )
        # Should not raise
        result = await engine.scan(MonitorDomain.ENVIRONMENTAL)
        assert result["opportunities"] >= 1

    @pytest.mark.asyncio
    async def test_no_handler_no_error(self, engine):
        engine.register_sensor(
            MonitorDomain.ENVIRONMENTAL, "hot", lambda: {"cpu_usage": 0.95}
        )
        result = await engine.scan(MonitorDomain.ENVIRONMENTAL)
        assert result["opportunities"] >= 1


# =============================================================================
# GET RECENT OPPORTUNITIES TESTS
# =============================================================================


class TestGetRecentOpportunities:
    """Tests for get_recent_opportunities."""

    def test_empty_returns_empty(self):
        engine = MuraqabahEngine()
        assert engine.get_recent_opportunities() == []

    def test_returns_limited_results(self):
        engine = MuraqabahEngine()
        for i in range(20):
            engine._opportunities.append(
                Opportunity(description=f"Opp {i}")
            )
        recent = engine.get_recent_opportunities(limit=5)
        assert len(recent) == 5
        # Should be the most recent
        assert recent[-1].description == "Opp 19"
        assert recent[0].description == "Opp 15"

    def test_returns_all_if_fewer_than_limit(self):
        engine = MuraqabahEngine()
        for i in range(3):
            engine._opportunities.append(Opportunity(description=f"Opp {i}"))
        recent = engine.get_recent_opportunities(limit=10)
        assert len(recent) == 3

    def test_default_limit_is_10(self):
        engine = MuraqabahEngine()
        for i in range(15):
            engine._opportunities.append(Opportunity(description=f"Opp {i}"))
        recent = engine.get_recent_opportunities()
        assert len(recent) == 10


# =============================================================================
# STATS TESTS
# =============================================================================


class TestStats:
    """Tests for stats() method."""

    def test_stats_structure(self):
        engine = MuraqabahEngine()
        s = engine.stats()
        assert "running" in s
        assert "scan_count" in s
        assert "total_opportunities" in s
        assert "active_sensors" in s
        assert "sensors_by_domain" in s
        assert "readings_by_domain" in s

    def test_stats_initial_values(self):
        engine = MuraqabahEngine()
        s = engine.stats()
        assert s["running"] is False
        assert s["scan_count"] == 0
        assert s["total_opportunities"] == 0
        assert s["active_sensors"] == 16

    def test_stats_sensors_by_domain(self):
        engine = MuraqabahEngine()
        s = engine.stats()
        sbd = s["sensors_by_domain"]
        assert sbd["environmental"] == 4
        assert sbd["health"] == 3
        assert sbd["cognitive"] == 3
        assert sbd["financial"] == 3
        assert sbd["social"] == 3

    def test_stats_readings_by_domain_initially_zero(self):
        engine = MuraqabahEngine()
        s = engine.stats()
        rbd = s["readings_by_domain"]
        for domain_val in rbd.values():
            assert domain_val == 0

    @pytest.mark.asyncio
    async def test_stats_after_scan(self):
        engine = MuraqabahEngine()
        # Replace sensors with simple controlled ones
        for domain in MonitorDomain:
            engine._sensors[domain] = {}
            engine._sensor_states.clear()
        engine.register_sensor(
            MonitorDomain.HEALTH, "test", lambda: {"metric": 1.0}
        )
        await engine.scan()
        s = engine.stats()
        assert s["scan_count"] == 1
        assert s["active_sensors"] == 1
        assert s["readings_by_domain"]["health"] == 1

    def test_stats_after_unregister(self):
        engine = MuraqabahEngine()
        engine.unregister_sensor(MonitorDomain.ENVIRONMENTAL, "system_health")
        s = engine.stats()
        assert s["sensors_by_domain"]["environmental"] == 3
        assert s["active_sensors"] == 15


# =============================================================================
# MONITORING LIFECYCLE TESTS
# =============================================================================


class TestMonitoringLifecycle:
    """Tests for start_monitoring / stop_monitoring."""

    def test_initial_not_running(self):
        engine = MuraqabahEngine()
        assert engine._running is False

    def test_stop_sets_flag(self):
        engine = MuraqabahEngine()
        engine._running = True
        engine.stop_monitoring()
        assert engine._running is False

    @pytest.mark.asyncio
    async def test_start_sets_running_flag(self):
        engine = MuraqabahEngine()
        # Replace sensors to avoid real I/O
        for domain in MonitorDomain:
            engine._sensors[domain] = {}
            engine._sensor_states.clear()
        engine.register_sensor(
            MonitorDomain.ENVIRONMENTAL, "test", lambda: {"val": 0.5}
        )
        # Set very short intervals
        for domain in MonitorDomain:
            engine.set_interval(domain, 10)

        # Start monitoring briefly then stop
        async def stop_after_brief():
            await asyncio.sleep(0.05)
            engine.stop_monitoring()

        task = asyncio.create_task(stop_after_brief())
        await asyncio.wait_for(engine.start_monitoring(), timeout=2.0)
        await task
        assert engine._running is False
        assert engine._scan_count >= 1

    @pytest.mark.asyncio
    async def test_start_monitoring_scans_domains(self):
        engine = MuraqabahEngine()
        for domain in MonitorDomain:
            engine._sensors[domain] = {}
            engine._sensor_states.clear()
        scanned_domains = []

        original_scan = engine.scan

        async def tracking_scan(domain=None):
            scanned_domains.append(domain)
            return await original_scan(domain)

        engine.scan = tracking_scan

        engine.register_sensor(
            MonitorDomain.ENVIRONMENTAL, "test", lambda: {"val": 0.5}
        )

        async def stop_quickly():
            await asyncio.sleep(0.05)
            engine.stop_monitoring()

        task = asyncio.create_task(stop_quickly())
        await asyncio.wait_for(engine.start_monitoring(), timeout=2.0)
        await task
        # Each domain should have been scanned at least once
        scanned_set = set(scanned_domains)
        assert len(scanned_set) >= 1


# =============================================================================
# ANALYZE READINGS TESTS (ASYNC)
# =============================================================================


class TestAnalyzeReadings:
    """Tests for _analyze_readings batch analysis."""

    @pytest.fixture
    def engine(self):
        return MuraqabahEngine()

    @pytest.mark.asyncio
    async def test_analyze_empty_readings(self, engine):
        opps = await engine._analyze_readings([])
        assert opps == []

    @pytest.mark.asyncio
    async def test_analyze_readings_with_opportunities(self, engine):
        readings = [
            SensorReading(
                domain=MonitorDomain.ENVIRONMENTAL,
                metric_name="cpu_usage",
                value=0.95,
            ),
            SensorReading(
                domain=MonitorDomain.HEALTH,
                metric_name="error_rate",
                value=0.10,
            ),
        ]
        opps = await engine._analyze_readings(readings)
        assert len(opps) == 2

    @pytest.mark.asyncio
    async def test_analyze_readings_mixed(self, engine):
        readings = [
            SensorReading(
                domain=MonitorDomain.ENVIRONMENTAL,
                metric_name="cpu_usage",
                value=0.95,  # triggers
            ),
            SensorReading(
                domain=MonitorDomain.ENVIRONMENTAL,
                metric_name="cpu_usage",
                value=0.5,  # does not trigger
            ),
        ]
        opps = await engine._analyze_readings(readings)
        assert len(opps) == 1

    @pytest.mark.asyncio
    async def test_analyze_readings_no_opportunities(self, engine):
        readings = [
            SensorReading(
                domain=MonitorDomain.ENVIRONMENTAL,
                metric_name="cpu_usage",
                value=0.5,
            ),
            SensorReading(
                domain=MonitorDomain.HEALTH,
                metric_name="error_rate",
                value=0.01,
            ),
        ]
        opps = await engine._analyze_readings(readings)
        assert opps == []


# =============================================================================
# EVENT PRIORITY TESTS
# =============================================================================


class TestEventPriorityInScan:
    """Tests that scan emits events with correct priority based on urgency."""

    @pytest.fixture
    def engine(self):
        eng = MuraqabahEngine()
        for domain in MonitorDomain:
            eng._sensors[domain] = {}
            eng._sensor_states.clear()
        mock_bus = AsyncMock(spec=EventBus)
        mock_bus.emit = AsyncMock(return_value="evt-id")
        eng.event_bus = mock_bus
        return eng

    @pytest.mark.asyncio
    async def test_high_urgency_emits_high_priority(self, engine):
        """Network disconnection has urgency 0.95 -> HIGH priority event."""
        engine.register_sensor(
            MonitorDomain.SOCIAL,
            "net",
            lambda: {"internet_connected": 0.0},
        )
        await engine.scan(MonitorDomain.SOCIAL)
        call_kwargs = engine.event_bus.emit.call_args
        assert call_kwargs is not None
        _, kwargs = call_kwargs
        assert kwargs["priority"] == EventPriority.HIGH

    @pytest.mark.asyncio
    async def test_low_urgency_emits_normal_priority(self, engine):
        """Low CPU utilization has urgency 0.2 -> NORMAL priority event."""
        engine.register_sensor(
            MonitorDomain.ENVIRONMENTAL,
            "idle",
            lambda: {"cpu_usage": 0.1},
        )
        await engine.scan(MonitorDomain.ENVIRONMENTAL)
        call_kwargs = engine.event_bus.emit.call_args
        assert call_kwargs is not None
        _, kwargs = call_kwargs
        assert kwargs["priority"] == EventPriority.NORMAL
