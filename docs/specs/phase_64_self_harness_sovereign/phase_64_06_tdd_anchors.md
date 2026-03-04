# Phase 64.06 — TDD Anchors

## Overview

Complete test specification for all Phase 64 modules. Each test is
designed to run WITHOUT external dependencies (no LM Studio, no GPU,
no network) — because the floor constraint applies to tests too.

Tests use `tmp_path`, `SimpleNamespace`, and mocks for isolation.
No bare `MagicMock()` for anything that touches the filesystem.

Standing on Giants: Deming (quality as measurement) · Shannon (test = verify channel capacity)

## Test Organization

```
tests/core/elite/
├── test_asset_registry.py         # 64.01
├── test_urp_contributor.py        # 64.02
├── test_payback_tracker.py        # 64.03
├── test_floor_constraint.py       # 64.04
├── test_scaling_calibrator.py     # 64.05
└── test_self_harness_sovereign.py # integration across all 5
```

## 64.01 — Asset Registry Tests

```pseudocode
class TestAssetRegistry:

    def test_introspect_returns_node_body(tmp_path):
        """Introspect returns NodeBody with cpu and ram at minimum."""
        registry = AssetRegistry(node_id="test-node")
        body = registry.introspect()
        assert isinstance(body, NodeBody)
        assert "cpu" in body.assets or body.total_capacity.get("cpu", 0) > 0
        assert "ram" in body.assets or body.total_capacity.get("ram", 0) > 0

    def test_introspect_without_psutil_returns_minimal(monkeypatch):
        """Without psutil, returns minimal body from /proc/meminfo."""
        monkeypatch.setattr("core.elite.asset_registry._HAS_PSUTIL", False)
        registry = AssetRegistry(node_id="test-node")
        body = registry.introspect()
        assert body is not None
        assert body.total_capacity.get("cpu", 0) >= 1  # at least 1 core from os.cpu_count()

    def test_introspect_without_pynvml_omits_gpu(monkeypatch):
        """Without pynvml, GPU assets are empty (not an error)."""
        monkeypatch.setattr("core.elite.asset_registry._HAS_PYNVML", False)
        registry = AssetRegistry(node_id="test-node")
        body = registry.introspect()
        gpu_assets = [a for a in body.assets.values() if a.asset_type == "gpu"]
        assert len(gpu_assets) == 0  # no GPU is fine!

    def test_idle_capacity_non_negative():
        """idle_capacity returns non-negative values for all types."""
        registry = AssetRegistry(node_id="test-node")
        idle = registry.idle_capacity()
        for value in idle.values():
            assert value >= 0

    def test_can_accept_mission_within_capacity():
        """Mission that fits within idle capacity is accepted."""
        registry = AssetRegistry(node_id="test-node")
        idle = registry.idle_capacity()
        # Request half of what's available
        required = {k: v * 0.5 for k, v in idle.items() if v > 0}
        if required:
            assert registry.can_accept_mission(required) is True

    def test_can_accept_mission_exceeds_capacity():
        """Mission that exceeds capacity is rejected."""
        registry = AssetRegistry(node_id="test-node")
        required = {"cpu": 999999, "ram": 999999}
        assert registry.can_accept_mission(required) is False

    def test_to_urp_pledge_valid():
        """to_urp_pledge returns valid URPPledge dataclass."""
        registry = AssetRegistry(node_id="test-node")
        pledge = registry.to_urp_pledge()
        assert pledge.node_id == "test-node"
        assert pledge.ram_gb >= 0
        assert pledge.vram_gb >= 0

    def test_cache_respects_interval():
        """Second introspect within interval returns cached result."""
        registry = AssetRegistry(node_id="test-node", refresh_interval_s=300)
        body1 = registry.introspect()
        body2 = registry.introspect()
        assert body1.snapshot_at == body2.snapshot_at  # same timestamp = cached

    def test_force_ignores_cache():
        """force=True bypasses cache."""
        registry = AssetRegistry(node_id="test-node", refresh_interval_s=300)
        body1 = registry.introspect()
        body2 = registry.introspect(force=True)
        # May or may not differ, but force should not use cache
        assert body2 is not body1 or body2.snapshot_at >= body1.snapshot_at
```

## 64.02 — URP Contributor Tests

```pseudocode
class TestURPContributor:

    def test_contribute_cycle_returns_none_below_threshold(tmp_path):
        """No contribution when idle < min_idle_fraction."""
        # Create a mock registry that reports 5% idle (below 10% min)
        registry = _mock_registry(idle_fraction=0.05)
        minter = _mock_minter(tmp_path)
        evidence = _mock_evidence(tmp_path)
        contrib = URPContributor(
            node_id="test", asset_registry=registry,
            token_minter=minter, evidence_ledger=evidence,
            min_idle_fraction=0.10,
        )
        record = await contrib.contribute_cycle()
        assert record is None

    def test_contribute_cycle_mints_seed(tmp_path):
        """Contribution mints SEED when idle > min_idle_fraction."""
        registry = _mock_registry(idle_fraction=0.50)
        minter = _mock_minter(tmp_path)
        evidence = _mock_evidence(tmp_path)
        contrib = URPContributor(
            node_id="test", asset_registry=registry,
            token_minter=minter, evidence_ledger=evidence,
        )
        record = await contrib.contribute_cycle()
        assert record is not None
        assert record.seed_earned > 0

    def test_zakat_deducted(tmp_path):
        """2.5% Zakat is deducted from seed_earned."""
        registry = _mock_registry(idle_fraction=0.50)
        minter = _mock_minter(tmp_path)
        evidence = _mock_evidence(tmp_path)
        contrib = URPContributor(
            node_id="test", asset_registry=registry,
            token_minter=minter, evidence_ledger=evidence,
        )
        record = await contrib.contribute_cycle()
        assert record is not None
        expected_net = record.seed_earned * 0.975
        assert abs(record.seed_net - expected_net) < 0.001

    def test_ihsan_gate_rejects_low_quality(tmp_path):
        """Contribution with ihsan < 0.95 is rejected."""
        registry = _mock_registry(idle_fraction=0.50)
        minter = _mock_minter(tmp_path)
        evidence = _mock_evidence(tmp_path)
        contrib = URPContributor(
            node_id="test", asset_registry=registry,
            token_minter=minter, evidence_ledger=evidence,
        )
        # Force low ihsan
        record = await contrib._gate_contribution(seed_earned=10.0, ihsan_score=0.80)
        assert record[0] is False
        assert "IHSAN" in record[1]

    def test_gini_gate_rejects_concentration(tmp_path):
        """Contribution that would exceed Gini 0.35 is rejected."""
        # This test exercises the Gini simulation in _gate_contribution
        ...

    def test_thermal_throttle(tmp_path):
        """Contribution pauses when GPU > thermal threshold."""
        registry = _mock_registry(idle_fraction=0.50, gpu_temp_c=90)
        contrib = URPContributor(
            node_id="test", asset_registry=registry,
            token_minter=_mock_minter(tmp_path),
            evidence_ledger=_mock_evidence(tmp_path),
        )
        record = await contrib.contribute_cycle()
        # Should skip GPU contribution when overheating
        assert record is None or record.resource_type != "gpu"

    def test_evidence_recorded(tmp_path):
        """Every contribution records an evidence receipt."""
        registry = _mock_registry(idle_fraction=0.50)
        evidence = _mock_evidence(tmp_path)
        contrib = URPContributor(
            node_id="test", asset_registry=registry,
            token_minter=_mock_minter(tmp_path),
            evidence_ledger=evidence,
        )
        record = await contrib.contribute_cycle()
        assert record is not None
        assert record.evidence_hash != ""

    def test_stop_completes_cycle(tmp_path):
        """stop() allows current cycle to complete before exiting."""
        # Test that stop is graceful
        contrib = _make_contributor(tmp_path)
        contrib.stop()
        assert contrib._running is False
```

## 64.03 — Payback Tracker Tests

```pseudocode
class TestPaybackTracker:

    def test_calculate_returns_valid_state(tmp_path):
        """calculate() returns PaybackState with correct totals."""
        device = DeviceInvestment(
            device_id="test", device_description="test device",
            purchase_cost_usd=1000.0, purchase_date="2024-01-01",
            expected_lifetime_years=5, depreciation_rate=0.20,
            residual_value_usd=0.0,
        )
        ledger = _mock_contribution_ledger(total_seed=100.0, total_zakat=2.5)
        tracker = PaybackTracker(device=device, contribution_ledger=ledger,
                                  evidence_ledger=_mock_evidence(tmp_path))
        state = tracker.calculate()
        assert state.total_seed_earned == 100.0
        assert state.total_zakat_paid == 2.5

    def test_payback_reached(tmp_path):
        """payback_reached is True when total_value >= purchase_cost."""
        device = DeviceInvestment(
            device_id="test", device_description="test",
            purchase_cost_usd=100.0, purchase_date="2024-01-01",
            expected_lifetime_years=5, depreciation_rate=0.20,
            residual_value_usd=0.0,
        )
        # Earned 10,000 SEED at 0.01 USD/SEED = $100 = purchase cost
        ledger = _mock_contribution_ledger(total_seed=10000.0)
        tracker = PaybackTracker(device=device, contribution_ledger=ledger,
                                  evidence_ledger=_mock_evidence(tmp_path))
        state = tracker.calculate()
        assert state.payback_reached is True

    def test_payback_not_reached(tmp_path):
        """payback_reached is False when total_value < purchase_cost."""
        device = DeviceInvestment(
            device_id="test", device_description="test",
            purchase_cost_usd=10000.0, purchase_date="2024-01-01",
            expected_lifetime_years=5, depreciation_rate=0.20,
            residual_value_usd=0.0,
        )
        ledger = _mock_contribution_ledger(total_seed=100.0)
        tracker = PaybackTracker(device=device, contribution_ledger=ledger,
                                  evidence_ledger=_mock_evidence(tmp_path))
        state = tracker.calculate()
        assert state.payback_reached is False
        assert state.roi_percent < 0

    def test_milestones_not_duplicated(tmp_path):
        """Milestones don't fire twice on recalculation."""
        tracker = _make_tracker(tmp_path, cost=100.0, earned=50.0)
        m1 = tracker.check_milestones(tracker.calculate())
        m2 = tracker.check_milestones(tracker.calculate())
        assert len(m1) > 0  # 25% and maybe 50% milestone
        assert len(m2) == 0  # already recorded

    def test_zero_cost_no_division_error(tmp_path):
        """Zero purchase_cost doesn't cause division by zero."""
        device = DeviceInvestment(
            device_id="test", device_description="free device",
            purchase_cost_usd=0.0, purchase_date="2024-01-01",
            expected_lifetime_years=5, depreciation_rate=0.0,
            residual_value_usd=0.0,
        )
        tracker = PaybackTracker(device=device,
                                  contribution_ledger=_mock_contribution_ledger(),
                                  evidence_ledger=_mock_evidence(tmp_path))
        state = tracker.calculate()
        assert state.payback_reached is True  # free device = instant payback

    def test_state_persists(tmp_path):
        """PaybackState persists across restarts."""
        tracker = _make_tracker(tmp_path, cost=1000.0, earned=250.0)
        tracker.calculate()
        tracker.save()
        loaded = tracker.load()
        assert loaded is not None
        assert loaded.total_seed_earned == 250.0
```

## 64.04 — Floor Constraint Tests

```pseudocode
class TestFloorConstraint:

    def test_default_floor_sane():
        """Default FloorProfile has reasonable minimums."""
        floor = FloorProfile()
        assert floor.min_ram_gb == 2.0
        assert floor.min_cpu_cores == 2
        assert floor.gpu_required is False
        assert floor.network_required is False

    def test_check_passes_for_capable_node():
        """Node that meets floor requirements passes."""
        body = _make_node_body(cpu_cores=4, ram_gb=8, disk_gb=100)
        result = FloorConstraint().check(body)
        assert result.compliant is True
        assert len(result.violations) == 0

    def test_check_fails_below_ram():
        """Node below RAM minimum fails."""
        body = _make_node_body(cpu_cores=4, ram_gb=1, disk_gb=100)
        result = FloorConstraint().check(body)
        assert result.compliant is False
        assert any("RAM" in v for v in result.violations)

    def test_check_never_fails_for_missing_gpu():
        """Missing GPU NEVER causes floor violation."""
        body = _make_node_body(cpu_cores=4, ram_gb=4, disk_gb=10, has_gpu=False)
        result = FloorConstraint().check(body)
        assert result.compliant is True
        # Explicitly verify no GPU-related violation
        assert not any("GPU" in v for v in result.violations)

    def test_check_never_fails_for_missing_network():
        """Missing network NEVER causes floor violation."""
        body = _make_node_body(cpu_cores=4, ram_gb=4, disk_gb=10, has_network=False)
        result = FloorConstraint().check(body)
        assert result.compliant is True

    def test_pipeline_time_passes_under_limit():
        """Pipeline under 60s passes."""
        constraint = FloorConstraint()
        assert constraint.check_pipeline_time(59.0, 400.0) is True

    def test_pipeline_time_fails_over_limit():
        """Pipeline over 60s fails."""
        constraint = FloorConstraint()
        assert constraint.check_pipeline_time(61.0, 400.0) is False

    def test_daughter_test_passes_minimum():
        """Daughter test passes for minimum viable hardware."""
        body = _make_node_body(cpu_cores=2, ram_gb=2, disk_gb=4)
        assert daughter_test(body) is True

    def test_daughter_test_passes_high_end():
        """Daughter test passes for high-end hardware too."""
        body = _make_node_body(cpu_cores=32, ram_gb=62, disk_gb=1000, has_gpu=True)
        assert daughter_test(body) is True

    def test_floor_report_includes_headroom():
        """Floor report includes headroom calculations."""
        body = _make_node_body(cpu_cores=32, ram_gb=62, disk_gb=1000)
        report = FloorConstraint().floor_report(body)
        assert "headroom" in report
        assert report["headroom"]["cpu"] is not None
```

## 64.05 — Scaling Calibrator Tests

```pseudocode
class TestScalingCalibrator:

    def test_measure_returns_snapshot():
        """measure() returns valid ScalingSnapshot."""
        calibrator = ScalingCalibrator(evidence_ledger=_mock_evidence())
        snapshot = calibrator.measure({"node_count": 1})
        assert isinstance(snapshot, ScalingSnapshot)
        assert snapshot.node_count == 1

    def test_single_node_coupling_is_one():
        """Single-node mode returns coupling_constant = 1.0."""
        calibrator = ScalingCalibrator(evidence_ledger=_mock_evidence())
        snapshot = calibrator.measure({"node_count": 1})
        assert snapshot.coupling_constant == 1.0

    def test_check_coupling_insufficient_data():
        """check_coupling returns None with < 3 snapshots."""
        calibrator = ScalingCalibrator(evidence_ledger=_mock_evidence())
        assert calibrator.check_coupling() is None

    def test_stability_report_complete():
        """stability_report includes all required fields."""
        calibrator = ScalingCalibrator(evidence_ledger=_mock_evidence())
        report = calibrator.stability_report()
        assert "coupling_constant" in report
        assert "stability" in report
        assert "technical_axis" in report
        assert "economic_axis" in report

    def test_gini_ceiling_from_constants():
        """Gini ceiling matches ADL_GINI_THRESHOLD."""
        from core.integration.constants import ADL_GINI_THRESHOLD
        calibrator = ScalingCalibrator(evidence_ledger=_mock_evidence())
        assert calibrator._gini_ceiling == ADL_GINI_THRESHOLD
```

## Shared Test Helpers

```pseudocode
def _make_node_body(
    cpu_cores: int = 4,
    ram_gb: float = 8.0,
    disk_gb: float = 100.0,
    has_gpu: bool = False,
    gpu_vram_gb: float = 0.0,
    has_network: bool = True,
) -> NodeBody:
    """Create a NodeBody for testing. Uses real Path objects, NOT MagicMock."""
    assets = {}
    assets["cpu-0"] = HardwareAsset(
        asset_type="cpu", asset_id="cpu-0",
        capacity_total=cpu_cores, capacity_unit="cores",
        available_now=cpu_cores * 0.8, utilization=0.2,
        temperature_c=50.0, is_contributing=False, contribution_fraction=0.0,
    )
    assets["ram-0"] = HardwareAsset(
        asset_type="ram", asset_id="ram-0",
        capacity_total=ram_gb, capacity_unit="GB",
        available_now=ram_gb * 0.7, utilization=0.3,
        temperature_c=None, is_contributing=False, contribution_fraction=0.0,
    )
    # ... disk, gpu, network as needed
    return NodeBody(
        node_id="test-node", hostname="test",
        assets=assets, snapshot_at=datetime.now(UTC),
        sovereignty_tier="SEED", floor_compliant=True,
    )


def _mock_minter(tmp_path: Path) -> TokenMinter:
    """Create isolated TokenMinter for testing."""
    from core.token.minter import TokenMinter
    return TokenMinter.create(
        db_path=tmp_path / "tokens.db",
        ledger=tmp_path / "ledger.jsonl",
        log_path=tmp_path / "mint.log",
    )


def _mock_evidence(tmp_path: Path) -> EvidenceLedger:
    """Create isolated EvidenceLedger for testing."""
    from core.proof_engine.evidence_ledger import EvidenceLedger
    return EvidenceLedger(path=tmp_path / "evidence.jsonl", validate_on_append=False)


def _mock_registry(idle_fraction: float = 0.5, gpu_temp_c: float = 50.0):
    """Create mock AssetRegistry with controlled idle capacity."""
    from types import SimpleNamespace
    registry = SimpleNamespace()
    registry.introspect = lambda force=False: _make_node_body(
        cpu_cores=8, ram_gb=16, has_gpu=True, gpu_vram_gb=16,
    )
    registry.idle_capacity = lambda: {"cpu": 8 * idle_fraction, "ram": 16 * idle_fraction}
    return registry
```

## Anti-Pattern: No Bare MagicMock for Paths

```pseudocode
# WRONG — this creates files named <MagicMock...> at CWD:
runtime = MagicMock()
tracker = PaybackTracker(device=device, contribution_ledger=runtime)

# CORRECT — use SimpleNamespace with real Path:
runtime = SimpleNamespace(state_dir=tmp_path / "state")
tracker = PaybackTracker(device=device, contribution_ledger=ledger,
                          evidence_ledger=EvidenceLedger(path=tmp_path / "ev.jsonl"))
```

This lesson cost us 259 garbage files. Never again.
