"""Tests for Phase 64.04 — FloorConstraint (Universality Gate).

The floor is the supreme design constraint.
GPU is NEVER required. Network is NEVER required.
If the pipeline doesn't work on a $200 phone, BIZRA's thesis is falsified.

No bare MagicMock. Uses real dataclasses.
"""

from __future__ import annotations

from datetime import datetime, timezone

from core.elite.asset_registry import HardwareAsset, NodeBody
from core.elite.floor_constraint import (
    FloorCheckResult,
    FloorConstraint,
    FloorProfile,
    daughter_test,
)

# ── Helpers ──────────────────────────────────────────────────


def _make_node_body(
    cpu_cores: int = 4,
    ram_gb: float = 8.0,
    disk_gb: float = 100.0,
    has_gpu: bool = False,
    gpu_vram_gb: float = 0.0,
    has_network: bool = True,
) -> NodeBody:
    """Create a NodeBody for testing. Uses real dataclasses, NOT MagicMock."""
    assets = {}
    assets["cpu-0"] = HardwareAsset(
        asset_type="cpu",
        asset_id="cpu-0",
        capacity_total=float(cpu_cores),
        capacity_unit="cores",
        available_now=float(cpu_cores) * 0.8,
        utilization=0.2,
        temperature_c=50.0,
    )
    assets["ram-0"] = HardwareAsset(
        asset_type="ram",
        asset_id="ram-0",
        capacity_total=ram_gb,
        capacity_unit="GB",
        available_now=ram_gb * 0.7,
        utilization=0.3,
    )
    assets["disk-0"] = HardwareAsset(
        asset_type="disk",
        asset_id="disk-0",
        capacity_total=disk_gb,
        capacity_unit="GB",
        available_now=disk_gb * 0.8,
        utilization=0.2,
    )
    if has_gpu:
        assets["gpu-0"] = HardwareAsset(
            asset_type="gpu",
            asset_id="gpu-0",
            capacity_total=gpu_vram_gb if gpu_vram_gb > 0 else 16.0,
            capacity_unit="GB",
            available_now=(gpu_vram_gb if gpu_vram_gb > 0 else 16.0) * 0.9,
            utilization=0.1,
            temperature_c=65.0,
        )
    if has_network:
        assets["net-0"] = HardwareAsset(
            asset_type="network",
            asset_id="net-0",
            capacity_total=1000.0,
            capacity_unit="MB_transferred",
            available_now=0.0,
            utilization=0.0,
        )
    return NodeBody(
        node_id="test-node",
        hostname="test",
        assets=assets,
        snapshot_at=datetime.now(timezone.utc),
    )


# ── FloorProfile tests ──────────────────────────────────────


class TestFloorProfile:
    """FloorProfile dataclass tests."""

    def test_defaults_are_sane(self):
        """Default FloorProfile has reasonable minimums."""
        floor = FloorProfile()
        assert floor.min_ram_gb == 2.0
        assert floor.min_storage_gb == 4.0
        assert floor.min_cpu_cores == 2
        assert floor.gpu_required is False
        assert floor.network_required is False
        assert floor.max_pipeline_time_s == 60.0
        assert floor.max_memory_usage_mb == 512.0

    def test_frozen(self):
        """FloorProfile is immutable."""
        floor = FloorProfile()
        try:
            floor.min_ram_gb = 999  # type: ignore[misc]
            assert False, "Should have raised"
        except AttributeError:
            pass


# ── FloorConstraint.check() tests ────────────────────────────


class TestFloorConstraintCheck:
    """FloorConstraint.check() tests."""

    def test_passes_for_capable_node(self):
        """Node that meets floor requirements passes."""
        body = _make_node_body(cpu_cores=4, ram_gb=8, disk_gb=100)
        result = FloorConstraint().check(body)
        assert result.compliant is True
        assert len(result.violations) == 0
        assert isinstance(result, FloorCheckResult)

    def test_passes_for_exact_minimum(self):
        """Node at exact floor minimums passes."""
        body = _make_node_body(cpu_cores=2, ram_gb=2.0, disk_gb=4.0)
        result = FloorConstraint().check(body)
        assert result.compliant is True
        assert len(result.violations) == 0

    def test_fails_below_ram(self):
        """Node below RAM minimum fails."""
        body = _make_node_body(cpu_cores=4, ram_gb=1.0, disk_gb=100)
        result = FloorConstraint().check(body)
        assert result.compliant is False
        assert any("RAM" in v for v in result.violations)
        assert result.margin["ram"] < 0

    def test_fails_below_cpu(self):
        """Node below CPU minimum fails."""
        body = _make_node_body(cpu_cores=1, ram_gb=4.0, disk_gb=100)
        result = FloorConstraint().check(body)
        assert result.compliant is False
        assert any("CPU" in v for v in result.violations)
        assert result.margin["cpu"] < 0

    def test_fails_below_disk(self):
        """Node below disk minimum fails."""
        body = _make_node_body(cpu_cores=4, ram_gb=4.0, disk_gb=1.0)
        result = FloorConstraint().check(body)
        assert result.compliant is False
        assert any("Storage" in v or "disk" in v.lower() for v in result.violations)

    def test_never_fails_for_missing_gpu(self):
        """Missing GPU NEVER causes floor violation. THIS IS CONSTITUTIONAL."""
        body = _make_node_body(cpu_cores=4, ram_gb=4, disk_gb=10, has_gpu=False)
        result = FloorConstraint().check(body)
        assert result.compliant is True
        assert not any("GPU" in v for v in result.violations)

    def test_never_fails_for_missing_network(self):
        """Missing network NEVER causes floor violation. Offline-first always."""
        body = _make_node_body(cpu_cores=4, ram_gb=4, disk_gb=10, has_network=False)
        result = FloorConstraint().check(body)
        assert result.compliant is True
        assert not any("network" in v.lower() for v in result.violations)

    def test_margin_positive_for_above_floor(self):
        """Margin is positive when above floor."""
        body = _make_node_body(cpu_cores=8, ram_gb=16, disk_gb=500)
        result = FloorConstraint().check(body)
        assert result.margin["cpu"] > 0
        assert result.margin["ram"] > 0
        assert result.margin["disk"] > 0

    def test_multiple_violations(self):
        """Node with multiple violations reports all of them."""
        body = _make_node_body(cpu_cores=1, ram_gb=1.0, disk_gb=1.0)
        result = FloorConstraint().check(body)
        assert result.compliant is False
        assert len(result.violations) == 3  # cpu + ram + disk


# ── FloorConstraint.check_pipeline_time() tests ──────────────


class TestPipelineTime:
    """Pipeline time/memory floor checks."""

    def test_passes_under_limit(self):
        """Pipeline under 60s and 512MB passes."""
        constraint = FloorConstraint()
        assert constraint.check_pipeline_time(59.0, 400.0) is True

    def test_fails_over_time_limit(self):
        """Pipeline over 60s fails."""
        constraint = FloorConstraint()
        assert constraint.check_pipeline_time(61.0, 400.0) is False

    def test_fails_over_memory_limit(self):
        """Pipeline over 512MB fails."""
        constraint = FloorConstraint()
        assert constraint.check_pipeline_time(30.0, 600.0) is False

    def test_exact_boundary_passes(self):
        """Exact boundary values pass (not strictly greater)."""
        constraint = FloorConstraint()
        assert constraint.check_pipeline_time(60.0, 512.0) is True

    def test_custom_profile_limits(self):
        """Custom FloorProfile with different limits works."""
        profile = FloorProfile(max_pipeline_time_s=30.0, max_memory_usage_mb=256.0)
        constraint = FloorConstraint(floor_profile=profile)
        assert constraint.check_pipeline_time(29.0, 200.0) is True
        assert constraint.check_pipeline_time(31.0, 200.0) is False


# ── FloorConstraint.floor_report() tests ──────────────────────


class TestFloorReport:
    """Floor report generation tests."""

    def test_includes_headroom(self):
        """Floor report includes headroom calculations."""
        body = _make_node_body(cpu_cores=32, ram_gb=62, disk_gb=1000)
        report = FloorConstraint().floor_report(body)
        assert "headroom" in report
        assert "cpu" in report["headroom"]
        assert "ram" in report["headroom"]
        assert "disk" in report["headroom"]

    def test_includes_surplus_for_urp(self):
        """Floor report includes surplus for URP."""
        body = _make_node_body(cpu_cores=32, ram_gb=62, disk_gb=1000)
        report = FloorConstraint().floor_report(body)
        assert "surplus_for_urp" in report
        assert "cpu" in report["surplus_for_urp"]
        assert "ram" in report["surplus_for_urp"]

    def test_gpu_surplus_shown(self):
        """GPU surplus shown when GPU is present."""
        body = _make_node_body(
            cpu_cores=8, ram_gb=16, disk_gb=100, has_gpu=True, gpu_vram_gb=16
        )
        report = FloorConstraint().floor_report(body)
        assert "gpu" in report["surplus_for_urp"]
        assert "VRAM" in report["surplus_for_urp"]["gpu"]

    def test_compliant_report(self):
        """Compliant node report says PASSED."""
        body = _make_node_body(cpu_cores=4, ram_gb=8, disk_gb=100)
        report = FloorConstraint().floor_report(body)
        assert report["compliant"] is True
        assert "PASSED" in report["daughter_test"]

    def test_non_compliant_report(self):
        """Non-compliant node report says FAILED."""
        body = _make_node_body(cpu_cores=1, ram_gb=1, disk_gb=1)
        report = FloorConstraint().floor_report(body)
        assert report["compliant"] is False
        assert "FAILED" in report["daughter_test"]


# ── daughter_test() tests ────────────────────────────────────


class TestDaughterTest:
    """daughter_test() public function tests."""

    def test_passes_minimum_viable(self):
        """Daughter test passes for minimum viable hardware."""
        body = _make_node_body(cpu_cores=2, ram_gb=2, disk_gb=4)
        assert daughter_test(body) is True

    def test_passes_high_end(self):
        """Daughter test passes for high-end hardware too."""
        body = _make_node_body(
            cpu_cores=32, ram_gb=62, disk_gb=1000, has_gpu=True, gpu_vram_gb=16
        )
        assert daughter_test(body) is True

    def test_fails_below_floor(self):
        """Daughter test fails for hardware below floor."""
        body = _make_node_body(cpu_cores=1, ram_gb=1, disk_gb=1)
        assert daughter_test(body) is False


# ── simulate_floor_node() tests ──────────────────────────────


class TestSimulateFloorNode:
    """simulate_floor_node() tests."""

    def test_returns_default_profile(self):
        """simulate_floor_node returns the most constrained profile."""
        constraint = FloorConstraint()
        profile = constraint.simulate_floor_node()
        assert isinstance(profile, FloorProfile)
        assert profile.min_ram_gb == 2.0
        assert profile.min_cpu_cores == 2
        assert profile.gpu_required is False

    def test_floor_node_passes_own_check(self):
        """A node at floor spec passes the floor check."""
        constraint = FloorConstraint()
        floor = constraint.simulate_floor_node()
        body = _make_node_body(
            cpu_cores=floor.min_cpu_cores,
            ram_gb=floor.min_ram_gb,
            disk_gb=floor.min_storage_gb,
            has_gpu=False,
            has_network=False,
        )
        result = constraint.check(body)
        assert result.compliant is True


# ── Import from core.elite tests ─────────────────────────────


class TestImports:
    """Verify lazy imports work from core.elite."""

    def test_import_floor_constraint(self):
        """FloorConstraint importable from core.elite."""
        from core.elite import FloorConstraint as FC

        assert FC is not None

    def test_import_floor_profile(self):
        """FloorProfile importable from core.elite."""
        from core.elite import FloorProfile as FP

        assert FP is not None

    def test_import_daughter_test(self):
        """daughter_test importable from core.elite."""
        from core.elite import daughter_test as dt

        assert callable(dt)

    def test_import_asset_registry(self):
        """AssetRegistry importable from core.elite."""
        from core.elite import AssetRegistry as AR

        assert AR is not None

    def test_import_node_body(self):
        """NodeBody importable from core.elite."""
        from core.elite import NodeBody as NB

        assert NB is not None
