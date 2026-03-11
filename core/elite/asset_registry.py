"""
BIZRA Asset Registry — Node Self-Awareness Engine.

The node doesn't USE the computer. The node IS the computer.
This module provides hardware introspection so the node knows
its own body: CPU, GPU, RAM, disk, network, and loaded models.

Standing on Giants:
- Shannon (1948): measure your own channel capacity
- Boyd (1976): observe yourself before orienting in the network
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Optional dependencies (graceful degradation) ──────────────

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    _HAS_PSUTIL = False

try:
    import pynvml

    _HAS_PYNVML = True
except ImportError:
    pynvml = None  # type: ignore[assignment]
    _HAS_PYNVML = False


# ── Data model ────────────────────────────────────────────────


@dataclass(frozen=True)
class HardwareAsset:
    """A single hardware resource owned by the node."""

    asset_type: str  # "cpu", "gpu", "ram", "disk", "network", "model"
    asset_id: str
    capacity_total: float
    capacity_unit: str  # "cores", "GB", "Mbps", "params"
    available_now: float
    utilization: float  # 0.0 to 1.0
    temperature_c: Optional[float] = None
    is_contributing: bool = False
    contribution_fraction: float = 0.0


@dataclass
class NodeBody:
    """Complete hardware profile of this node — its physical body."""

    node_id: str
    hostname: str
    assets: Dict[str, HardwareAsset] = field(default_factory=dict)
    snapshot_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    sovereignty_tier: str = "SEED"
    floor_compliant: bool = True

    @property
    def total_capacity(self) -> Dict[str, float]:
        """Aggregate capacity by asset type."""
        result: Dict[str, float] = {}
        for asset in self.assets.values():
            key = asset.asset_type
            result[key] = result.get(key, 0.0) + asset.capacity_total
        return result

    @property
    def idle_capacity(self) -> Dict[str, float]:
        """Capacity available for URP contribution."""
        result: Dict[str, float] = {}
        for asset in self.assets.values():
            key = asset.asset_type
            result[key] = result.get(key, 0.0) + asset.available_now
        return result

    @property
    def contribution_potential(self) -> float:
        """Normalized 0.0-1.0 score of how much this node CAN contribute."""
        total = self.total_capacity
        idle = self.idle_capacity
        if not total:
            return 0.0
        ratios = []
        for key in total:
            if total[key] > 0:
                ratios.append(idle.get(key, 0.0) / total[key])
        return sum(ratios) / len(ratios) if ratios else 0.0


# ── Core class ────────────────────────────────────────────────


class AssetRegistry:
    """The node's self-awareness of its own hardware body.

    This is NOT a monitoring system. It is SELF-AWARENESS.
    The node doesn't monitor external infrastructure —
    it introspects its own organs.
    """

    def __init__(
        self,
        node_id: str = "",
        *,
        refresh_interval_s: float = 30.0,
        contribution_floor: float = 0.10,
        contribution_ceiling: float = 0.80,
        thermal_throttle_c: float = 85.0,
        lm_studio_url: str = "",
        lm_studio_token: str = "",
    ) -> None:
        self._node_id = node_id or _default_node_id()
        self._refresh_interval = refresh_interval_s
        self._contrib_floor = contribution_floor
        self._contrib_ceiling = contribution_ceiling
        self._thermal_throttle = thermal_throttle_c
        self._lm_url = lm_studio_url or os.getenv("LM_STUDIO_URL", "").rstrip(
            "/v1"
        ).rstrip("/")
        self._lm_token = lm_studio_token or os.getenv("LM_API_TOKEN", "")
        self._last_snapshot: Optional[NodeBody] = None
        self._last_refresh: float = 0.0

    # ── Introspection (observe own body) ──────────────────────

    def introspect(self, force: bool = False) -> NodeBody:
        """Full hardware introspection. Returns NodeBody snapshot."""
        now = time.monotonic()
        if (
            not force
            and self._last_snapshot is not None
            and (now - self._last_refresh) <= self._refresh_interval
        ):
            return self._last_snapshot

        assets: Dict[str, HardwareAsset] = {}

        cpu = self._introspect_cpu()
        assets[cpu.asset_id] = cpu

        ram = self._introspect_ram()
        assets[ram.asset_id] = ram

        disk = self._introspect_disk()
        assets[disk.asset_id] = disk

        for gpu in self._introspect_gpu():
            assets[gpu.asset_id] = gpu

        net = self._introspect_network()
        if net is not None:
            assets[net.asset_id] = net

        hostname = os.uname().nodename if hasattr(os, "uname") else "unknown"

        body = NodeBody(
            node_id=self._node_id,
            hostname=hostname,
            assets=assets,
            snapshot_at=datetime.now(timezone.utc),
        )

        self._last_snapshot = body
        self._last_refresh = now
        return body

    def _introspect_cpu(self) -> HardwareAsset:
        """Introspect CPU: cores, utilization."""
        if _HAS_PSUTIL:
            cores = psutil.cpu_count(logical=True) or 1
            usage_pct = psutil.cpu_percent(interval=0) / 100.0
            available = cores * max(0.0, 1.0 - usage_pct)
            temp = None
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    first_key = next(iter(temps))
                    entries = temps[first_key]
                    if entries:
                        temp = entries[0].current
            except (AttributeError, StopIteration, OSError):
                pass
        else:
            cores = os.cpu_count() or 1
            usage_pct = 0.0
            available = float(cores)
            temp = None

        return HardwareAsset(
            asset_type="cpu",
            asset_id="cpu-0",
            capacity_total=float(cores),
            capacity_unit="cores",
            available_now=round(available, 2),
            utilization=round(min(1.0, max(0.0, usage_pct)), 4),
            temperature_c=temp,
        )

    def _introspect_ram(self) -> HardwareAsset:
        """Introspect RAM: total, available."""
        if _HAS_PSUTIL:
            mem = psutil.virtual_memory()
            total_gb = mem.total / (1024**3)
            avail_gb = mem.available / (1024**3)
            usage = mem.percent / 100.0
        else:
            total_gb = _read_proc_meminfo_total_gb()
            avail_gb = total_gb * 0.7  # rough estimate
            usage = 0.3

        return HardwareAsset(
            asset_type="ram",
            asset_id="ram-0",
            capacity_total=round(total_gb, 2),
            capacity_unit="GB",
            available_now=round(avail_gb, 2),
            utilization=round(min(1.0, max(0.0, usage)), 4),
        )

    def _introspect_disk(self) -> HardwareAsset:
        """Introspect disk: total, available."""
        check_path = os.getenv("BIZRA_SOVEREIGN_ROOT", "/")
        if not os.path.exists(check_path):
            check_path = "/"  # fallback if sovereign root unmounted
        if _HAS_PSUTIL:
            try:
                usage = psutil.disk_usage(check_path)
                total_gb = usage.total / (1024**3)
                free_gb = usage.free / (1024**3)
                pct = usage.percent / 100.0
            except (OSError, FileNotFoundError):
                total_gb = 0.0
                free_gb = 0.0
                pct = 0.0
        else:
            try:
                stat = os.statvfs(check_path)
                total_gb = (stat.f_blocks * stat.f_frsize) / (1024**3)
                free_gb = (stat.f_bavail * stat.f_frsize) / (1024**3)
                pct = 1.0 - (free_gb / total_gb) if total_gb > 0 else 0.0
            except (OSError, AttributeError):
                total_gb = 0.0
                free_gb = 0.0
                pct = 0.0

        return HardwareAsset(
            asset_type="disk",
            asset_id="disk-0",
            capacity_total=round(total_gb, 2),
            capacity_unit="GB",
            available_now=round(free_gb, 2),
            utilization=round(min(1.0, max(0.0, pct)), 4),
        )

    def _introspect_gpu(self) -> List[HardwareAsset]:
        """Introspect GPU(s): VRAM, utilization, temperature."""
        if not _HAS_PYNVML:
            return []  # CPU-only node — still sovereign!

        gpus: List[HardwareAsset] = []
        try:
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
                total_gb = mem.total / (1024**3)
                used_gb = mem.used / (1024**3)
                free_gb = total_gb - used_gb
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                    gpu_pct = util.gpu / 100.0
                except Exception:  # noqa: BLE001 — boundary boundary
                    gpu_pct = used_gb / total_gb if total_gb > 0 else 0.0
                try:
                    temp = pynvml.nvmlDeviceGetTemperature(
                        handle, pynvml.NVML_TEMPERATURE_GPU
                    )
                except Exception:  # noqa: BLE001 — boundary boundary
                    temp = None

                gpus.append(
                    HardwareAsset(
                        asset_type="gpu",
                        asset_id=f"gpu-{i}",
                        capacity_total=round(total_gb, 2),
                        capacity_unit="GB",
                        available_now=round(free_gb, 2),
                        utilization=round(min(1.0, max(0.0, gpu_pct)), 4),
                        temperature_c=temp,
                    )
                )
            pynvml.nvmlShutdown()
        except Exception as exc:  # noqa: BLE001 — boundary boundary
            logger.debug("GPU introspection failed (non-fatal): %s", exc)

        return gpus

    def _introspect_network(self) -> Optional[HardwareAsset]:
        """Introspect network bandwidth estimate."""
        if not _HAS_PSUTIL:
            return None
        try:
            counters = psutil.net_io_counters()
            # Report cumulative bytes as a rough capacity indicator
            total_mb = (counters.bytes_sent + counters.bytes_recv) / (1024**2)
            return HardwareAsset(
                asset_type="network",
                asset_id="net-0",
                capacity_total=round(total_mb, 2),
                capacity_unit="MB_transferred",
                available_now=0.0,  # bandwidth is not "stored"
                utilization=0.0,
            )
        except Exception:  # noqa: BLE001 — boundary boundary
            return None

    # ── Sovereign queries ─────────────────────────────────────

    def idle_capacity(self) -> Dict[str, float]:
        """What capacity can this node contribute RIGHT NOW?"""
        body = self.introspect()
        return body.idle_capacity

    def can_accept_mission(self, required: Dict[str, float]) -> bool:
        """Can this node accept a mission with these resource requirements?"""
        idle = self.idle_capacity()
        for resource_type, amount in required.items():
            if idle.get(resource_type, 0.0) < amount:
                return False
        return True

    def to_urp_pledge(self) -> Dict[str, Any]:
        """Convert current idle capacity into a URPPledge-compatible dict.

        Bridges to existing core/genesis/urp.py URPPledge dataclass.
        """
        body = self.introspect()
        idle = body.idle_capacity
        return {
            "node_id": self._node_id,
            "ram_gb": int(idle.get("ram", 0)),
            "vram_gb": int(idle.get("gpu", 0)),
            "storage_gb": int(idle.get("disk", 0)),
            "pledged_at": datetime.now(timezone.utc).isoformat(),
            "status": "active",
        }

    def summary(self) -> Dict[str, Any]:
        """Human-readable summary for dashboard."""
        body = self.introspect()
        total = body.total_capacity
        idle = body.idle_capacity
        headroom: Dict[str, str] = {}
        for key in total:
            if total[key] > 0:
                ratio = idle.get(key, 0.0) / total[key]
                headroom[key] = (
                    f"{ratio:.0%} idle ({idle.get(key, 0):.1f} / {total[key]:.1f} {next((a.capacity_unit for a in body.assets.values() if a.asset_type == key), '')})"
                )
        return {
            "node_id": body.node_id,
            "hostname": body.hostname,
            "assets": len(body.assets),
            "total_capacity": total,
            "idle_capacity": idle,
            "headroom": headroom,
            "contribution_potential": round(body.contribution_potential, 4),
            "sovereignty_tier": body.sovereignty_tier,
            "floor_compliant": body.floor_compliant,
            "snapshot_at": body.snapshot_at.isoformat(),
        }


# ── Helpers ───────────────────────────────────────────────────


def _default_node_id() -> str:
    """Generate a default node ID from hostname."""
    try:
        return f"node-{os.uname().nodename}"
    except AttributeError:
        return "node-unknown"


def _read_proc_meminfo_total_gb() -> float:
    """Read total RAM from /proc/meminfo (Linux fallback)."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    return kb / (1024**2)
    except (OSError, ValueError, IndexError):
        pass
    return 4.0  # safe default
