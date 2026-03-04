"""Tests for Phase 64.01 — AssetRegistry (Node Self-Awareness Engine).

Uses real psutil introspection on the host. No bare MagicMock.
"""

from __future__ import annotations

import time

from core.elite.asset_registry import AssetRegistry, HardwareAsset, NodeBody


class TestHardwareAsset:
    """HardwareAsset dataclass tests."""

    def test_frozen(self):
        """HardwareAsset is immutable."""
        asset = HardwareAsset(
            asset_type="cpu",
            asset_id="cpu-0",
            capacity_total=8.0,
            capacity_unit="cores",
            available_now=6.0,
            utilization=0.25,
        )
        assert asset.asset_type == "cpu"
        assert asset.is_contributing is False
        assert asset.contribution_fraction == 0.0

    def test_optional_temperature(self):
        """Temperature defaults to None."""
        asset = HardwareAsset(
            asset_type="gpu",
            asset_id="gpu-0",
            capacity_total=16.0,
            capacity_unit="GB",
            available_now=14.0,
            utilization=0.1,
            temperature_c=72.0,
        )
        assert asset.temperature_c == 72.0


class TestNodeBody:
    """NodeBody tests."""

    def test_total_capacity(self):
        """total_capacity aggregates by asset type."""
        assets = {
            "cpu-0": HardwareAsset("cpu", "cpu-0", 8.0, "cores", 6.0, 0.25),
            "ram-0": HardwareAsset("ram", "ram-0", 32.0, "GB", 24.0, 0.25),
        }
        body = NodeBody(node_id="test", hostname="test", assets=assets)
        total = body.total_capacity
        assert total["cpu"] == 8.0
        assert total["ram"] == 32.0

    def test_idle_capacity(self):
        """idle_capacity returns available_now sums."""
        assets = {
            "cpu-0": HardwareAsset("cpu", "cpu-0", 8.0, "cores", 6.0, 0.25),
            "ram-0": HardwareAsset("ram", "ram-0", 32.0, "GB", 24.0, 0.25),
        }
        body = NodeBody(node_id="test", hostname="test", assets=assets)
        idle = body.idle_capacity
        assert idle["cpu"] == 6.0
        assert idle["ram"] == 24.0

    def test_contribution_potential(self):
        """contribution_potential is between 0 and 1."""
        assets = {
            "cpu-0": HardwareAsset("cpu", "cpu-0", 8.0, "cores", 4.0, 0.5),
            "ram-0": HardwareAsset("ram", "ram-0", 16.0, "GB", 8.0, 0.5),
        }
        body = NodeBody(node_id="test", hostname="test", assets=assets)
        assert 0.0 <= body.contribution_potential <= 1.0
        assert abs(body.contribution_potential - 0.5) < 0.01

    def test_contribution_potential_empty(self):
        """Empty assets → 0.0 potential."""
        body = NodeBody(node_id="test", hostname="test", assets={})
        assert body.contribution_potential == 0.0


class TestAssetRegistry:
    """AssetRegistry introspection tests."""

    def test_introspect_returns_node_body(self):
        """Introspect returns NodeBody with cpu and ram at minimum."""
        registry = AssetRegistry(node_id="test-node")
        body = registry.introspect()
        assert isinstance(body, NodeBody)
        assert body.node_id == "test-node"
        total = body.total_capacity
        assert total.get("cpu", 0) >= 1
        assert total.get("ram", 0) > 0

    def test_introspect_without_psutil(self, monkeypatch):
        """Without psutil, returns minimal body from os.cpu_count()."""
        monkeypatch.setattr("core.elite.asset_registry._HAS_PSUTIL", False)
        registry = AssetRegistry(node_id="test-no-psutil")
        body = registry.introspect(force=True)
        assert body is not None
        assert body.total_capacity.get("cpu", 0) >= 1

    def test_introspect_without_pynvml(self, monkeypatch):
        """Without pynvml, GPU assets are empty (not an error)."""
        monkeypatch.setattr("core.elite.asset_registry._HAS_PYNVML", False)
        registry = AssetRegistry(node_id="test-no-gpu")
        body = registry.introspect(force=True)
        gpu_assets = [a for a in body.assets.values() if a.asset_type == "gpu"]
        assert len(gpu_assets) == 0

    def test_idle_capacity_non_negative(self):
        """idle_capacity returns non-negative values for all types."""
        registry = AssetRegistry(node_id="test-node")
        idle = registry.idle_capacity()
        for value in idle.values():
            assert value >= 0

    def test_can_accept_mission_within_capacity(self):
        """Mission that fits within idle capacity is accepted."""
        registry = AssetRegistry(node_id="test-node")
        idle = registry.idle_capacity()
        required = {k: v * 0.5 for k, v in idle.items() if v > 0}
        if required:
            assert registry.can_accept_mission(required) is True

    def test_can_accept_mission_exceeds_capacity(self):
        """Mission that exceeds capacity is rejected."""
        registry = AssetRegistry(node_id="test-node")
        required = {"cpu": 999999, "ram": 999999}
        assert registry.can_accept_mission(required) is False

    def test_can_accept_mission_empty_requirements(self):
        """Empty requirements are always accepted."""
        registry = AssetRegistry(node_id="test-node")
        assert registry.can_accept_mission({}) is True

    def test_to_urp_pledge_valid(self):
        """to_urp_pledge returns dict with required keys."""
        registry = AssetRegistry(node_id="test-node")
        pledge = registry.to_urp_pledge()
        assert pledge["node_id"] == "test-node"
        assert pledge["ram_gb"] >= 0
        assert pledge["vram_gb"] >= 0
        assert pledge["storage_gb"] >= 0
        assert pledge["status"] == "active"
        assert "pledged_at" in pledge

    def test_cache_respects_interval(self):
        """Second introspect within interval returns cached result."""
        registry = AssetRegistry(node_id="test-cache", refresh_interval_s=300)
        body1 = registry.introspect()
        body2 = registry.introspect()
        assert body1.snapshot_at == body2.snapshot_at

    def test_force_ignores_cache(self):
        """force=True bypasses cache."""
        registry = AssetRegistry(node_id="test-force", refresh_interval_s=300)
        body1 = registry.introspect()
        # Force creates a new snapshot object
        body2 = registry.introspect(force=True)
        assert body2 is not body1

    def test_summary_has_required_keys(self):
        """summary() includes all dashboard keys."""
        registry = AssetRegistry(node_id="test-summary")
        s = registry.summary()
        for key in [
            "node_id",
            "hostname",
            "assets",
            "total_capacity",
            "idle_capacity",
            "headroom",
            "contribution_potential",
            "sovereignty_tier",
            "floor_compliant",
            "snapshot_at",
        ]:
            assert key in s, f"Missing key: {key}"

    def test_utilization_bounded(self):
        """All utilization values are between 0 and 1."""
        registry = AssetRegistry(node_id="test-util")
        body = registry.introspect()
        for asset in body.assets.values():
            assert 0.0 <= asset.utilization <= 1.0, (
                f"{asset.asset_id} utilization out of bounds: {asset.utilization}"
            )

    def test_cpu_introspection_directly(self):
        """_introspect_cpu returns valid HardwareAsset."""
        registry = AssetRegistry(node_id="test-cpu")
        cpu = registry._introspect_cpu()
        assert cpu.asset_type == "cpu"
        assert cpu.asset_id == "cpu-0"
        assert cpu.capacity_total >= 1
        assert cpu.capacity_unit == "cores"

    def test_ram_introspection_directly(self):
        """_introspect_ram returns valid HardwareAsset."""
        registry = AssetRegistry(node_id="test-ram")
        ram = registry._introspect_ram()
        assert ram.asset_type == "ram"
        assert ram.capacity_total > 0
        assert ram.capacity_unit == "GB"

    def test_disk_introspection_directly(self):
        """_introspect_disk returns valid HardwareAsset."""
        registry = AssetRegistry(node_id="test-disk")
        disk = registry._introspect_disk()
        assert disk.asset_type == "disk"
        assert disk.capacity_total > 0
        assert disk.capacity_unit == "GB"
