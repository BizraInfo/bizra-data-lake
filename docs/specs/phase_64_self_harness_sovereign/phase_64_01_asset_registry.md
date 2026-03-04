# Phase 64.01 — Asset Registry

## Purpose

The node must know its own body. The Asset Registry introspects
hardware capabilities and presents them as sovereign resources that
the node OWNS, not external resources it accesses.

Standing on Giants: Shannon (capacity measurement) · Boyd (observe own state)

## Data Model

```pseudocode
@dataclass(frozen=True)
class HardwareAsset:
    """A single hardware resource owned by the node."""
    asset_type: Literal["cpu", "gpu", "ram", "disk", "network", "model"]
    asset_id: str                    # unique within this node
    capacity_total: float            # total capacity (cores, GB, Mbps)
    capacity_unit: str               # "cores", "GB", "Mbps", "params"
    available_now: float             # currently free capacity
    utilization: float               # 0.0 to 1.0
    temperature_c: Optional[float]   # thermal state (GPU/CPU)
    is_contributing: bool            # currently pledged to URP
    contribution_fraction: float     # 0.0 to 1.0 — how much goes to URP


@dataclass
class NodeBody:
    """Complete hardware profile of this node — its physical body."""
    node_id: str
    hostname: str
    assets: Dict[str, HardwareAsset]  # asset_id -> asset
    snapshot_at: datetime
    sovereignty_tier: str             # SEED/SPROUT/TREE/FOREST
    floor_compliant: bool             # passes minimum viable node check

    @property
    def total_capacity(self) -> Dict[str, float]:
        """Aggregate capacity by type."""
        ...

    @property
    def idle_capacity(self) -> Dict[str, float]:
        """Capacity available for URP contribution."""
        ...

    @property
    def contribution_potential(self) -> float:
        """Normalized 0.0-1.0 score of how much this node CAN contribute."""
        ...
```

## Core Class

```pseudocode
class AssetRegistry:
    """The node's self-awareness of its own hardware body.

    This is NOT a monitoring system. It is SELF-AWARENESS.
    The node doesn't monitor external infrastructure —
    it introspects its own organs.
    """

    def __init__(self, node_id: str, *, refresh_interval_s: float = 30.0):
        self._node_id = node_id
        self._refresh_interval = refresh_interval_s
        self._last_snapshot: Optional[NodeBody] = None
        self._last_refresh: float = 0.0

    # ── Introspection (observe own body) ───────────────────────

    def introspect(self, force: bool = False) -> NodeBody:
        """Full hardware introspection. Returns NodeBody snapshot.

        PSEUDOCODE:
        1. IF not force AND cache is fresh (< refresh_interval): return cached
        2. cpu_assets = _introspect_cpu()    # psutil.cpu_count, cpu_percent
        3. ram_asset  = _introspect_ram()    # psutil.virtual_memory
        4. disk_asset = _introspect_disk()   # psutil.disk_usage
        5. gpu_assets = _introspect_gpu()    # nvidia-smi / pynvml if available
        6. net_asset  = _introspect_network() # psutil.net_io_counters
        7. model_assets = _introspect_models() # LM Studio /api/v1/models
        8. Assemble NodeBody with all assets
        9. Compute floor_compliant via FloorConstraint.check(body)
        10. Cache and return
        """
        ...

    def _introspect_cpu(self) -> HardwareAsset:
        """Introspect CPU: cores, utilization, temperature.

        PSEUDOCODE:
        - cores = psutil.cpu_count(logical=True)
        - usage = psutil.cpu_percent(interval=0.1)
        - temp = psutil.sensors_temperatures().get("coretemp", [{}])[0].get("current")
        - available = cores * (1.0 - usage/100.0)
        - Return HardwareAsset(type="cpu", capacity_total=cores, ...)
        """
        ...

    def _introspect_gpu(self) -> List[HardwareAsset]:
        """Introspect GPU(s): VRAM, utilization, temperature.

        PSEUDOCODE:
        - TRY: import pynvml; pynvml.nvmlInit()
        - FOR each device:
            - total_vram = nvmlDeviceGetMemoryInfo.total / 1e9
            - used_vram = nvmlDeviceGetMemoryInfo.used / 1e9
            - utilization = nvmlDeviceGetUtilizationRates.gpu / 100.0
            - temp = nvmlDeviceGetTemperature(NVML_TEMPERATURE_GPU)
            - Return HardwareAsset(type="gpu", ...)
        - EXCEPT (pynvml not available):
            - Return empty list (CPU-only node — still valid!)
        """
        ...

    def _introspect_ram(self) -> HardwareAsset:
        """PSEUDOCODE: psutil.virtual_memory() → total, available, percent."""
        ...

    def _introspect_disk(self) -> HardwareAsset:
        """PSEUDOCODE: psutil.disk_usage('/') or BIZRA_SOVEREIGN_ROOT."""
        ...

    def _introspect_network(self) -> HardwareAsset:
        """PSEUDOCODE: psutil.net_io_counters() → bandwidth estimate."""
        ...

    def _introspect_models(self) -> List[HardwareAsset]:
        """Introspect loaded LLM models as cognitive assets.

        PSEUDOCODE:
        - GET {LM_STUDIO_URL}/api/v1/models
        - FOR each model with loaded_instances:
            - params = estimate_params(model_id)
            - Return HardwareAsset(type="model", capacity_total=params, ...)
        - FALLBACK: check Ollama at localhost:11434/api/tags
        - If neither available: return empty list (CPU-only node)
        """
        ...

    # ── Sovereign queries ──────────────────────────────────────

    def idle_capacity(self) -> Dict[str, float]:
        """What capacity can this node contribute RIGHT NOW?

        Returns {asset_type: idle_units}.
        This is the input to URPContributor.contribute().
        """
        body = self.introspect()
        return body.idle_capacity

    def can_accept_mission(self, required: Dict[str, float]) -> bool:
        """Can this node accept a mission with these requirements?

        PSEUDOCODE:
        - body = introspect()
        - FOR each (resource_type, amount) in required:
            - IF body.idle_capacity[type] < amount: return False
        - Return True
        """
        ...

    def to_urp_pledge(self) -> "URPPledge":
        """Convert current idle capacity into a URPPledge record.

        Bridges to existing core/genesis/urp.py URPPledge dataclass.

        PSEUDOCODE:
        - body = introspect()
        - Return URPPledge(
            node_id=self._node_id,
            ram_gb=int(body.idle_capacity.get("ram", 0)),
            vram_gb=int(body.idle_capacity.get("gpu", 0)),
            storage_gb=int(body.idle_capacity.get("disk", 0)),
            pledged_at=datetime.now(UTC).isoformat(),
            status="active",
          )
        """
        ...
```

## Flow Diagram

```
┌──────────────┐
│   Node Boot  │
└──────┬───────┘
       │
       ▼
┌──────────────┐    psutil     ┌──────────────┐
│  introspect()├──────────────►│  CPU / RAM   │
│              │    pynvml     │  GPU / Disk  │
│              ├──────────────►│  Network     │
│              │    httpx      │  LM Studio   │
│              ├──────────────►│  Ollama      │
└──────┬───────┘               └──────────────┘
       │
       ▼
┌──────────────┐
│   NodeBody   │─── assets, idle_capacity, floor_compliant
└──────┬───────┘
       │
       ├──► URPContributor.contribute(idle_capacity)
       ├──► PaybackTracker.record_contribution(pledge)
       └──► FloorConstraint.check(body)
```

## Configuration

```yaml
# config/asset_registry.yaml
asset_registry:
  refresh_interval_s: 30
  gpu_introspection: true        # false on CPU-only nodes
  model_introspection: true      # false if no LLM backend
  contribution_floor: 0.10       # minimum 10% idle before contributing
  contribution_ceiling: 0.80     # never contribute more than 80%
  thermal_throttle_c: 85         # stop contributing if GPU > 85°C
```

## Dependency on psutil

```pseudocode
# psutil is optional — CPU-only nodes may not have it
try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

# pynvml is optional — GPU nodes only
try:
    import pynvml
    _HAS_PYNVML = True
except ImportError:
    _HAS_PYNVML = False

# Graceful degradation: if neither is available, return
# a minimal NodeBody with estimated values from /proc/meminfo
# and os.cpu_count(). The node is STILL sovereign — it just
# has less self-awareness. Like a human with fewer senses.
```

## TDD Anchors (see phase_64_06 for full tests)

```
TEST: introspect returns NodeBody with at least cpu and ram assets
TEST: introspect without psutil returns minimal body (no crash)
TEST: introspect without pynvml omits GPU assets (no crash)
TEST: idle_capacity returns non-negative values for all types
TEST: can_accept_mission returns False when requirements exceed capacity
TEST: can_accept_mission returns True when requirements fit
TEST: to_urp_pledge produces valid URPPledge dataclass
TEST: thermal_throttle prevents contribution above threshold
TEST: cache respects refresh_interval_s
TEST: concurrent introspect calls don't race (thread safety)
```
