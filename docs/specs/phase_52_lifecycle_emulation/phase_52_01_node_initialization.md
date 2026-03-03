# Phase 52.1: Node Initialization (Phase 0 -- Node Boot)

> Standing on Giants: Lamport (logical clocks and initialization order, 1978) · Shannon (entropy as readiness metric, 1948) · Al-Ghazali (Ihsan as constitutional floor, 1095) · Fowler (canary health checks, 2010)

## 1. Overview

Before Ahmed's node can process any task, it must boot all subsystems in the correct
dependency order, load constitutional constants, and verify that external services
(LM Studio, filesystem, venv) are reachable. A failed health check blocks task
acceptance entirely -- the node never operates in a degraded constitutional state.

---

## 2. Data Flow

```
                  Node Boot Sequence
  ┌────────────────────────────────────────────────┐
  │  1. Load Constitution                          │
  │     core/integration/constants.py              │
  │     IHSAN >= 0.95, SNR >= 0.85, ADL <= 0.35   │
  └──────────────────┬─────────────────────────────┘
                     │
  ┌──────────────────▼─────────────────────────────┐
  │  2. Initialize Subsystems (dependency order)    │
  │     Engram → PAT-7 → RLM → TTRL → SSO →       │
  │     Guardian → ActionBus → ReceiptChain →       │
  │     ReflexLedger                                │
  └──────────────────┬─────────────────────────────┘
                     │
  ┌──────────────────▼─────────────────────────────┐
  │  3. Health Checks                               │
  │     LM Studio ping, token present,              │
  │     venv active, FAISS index, AHK bridge        │
  └──────────────────┬─────────────────────────────┘
                     │
              ┌──────▼──────┐
              │  Node READY  │ ← emits NodeReady event on ActionBus
              └─────────────┘
```

---

## 3. Pseudocode

### 3.1 Constitution Loading

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

from core.integration.constants import (
    ADL_GINI_THRESHOLD,
    FAISS_EMBEDDING_DIM,
    FAISS_SIMILARITY_FLOOR,
    GOT_MAX_HYPOTHESES,
    IHSAN_THRESHOLD,
    IHSAN_WEIGHTS,
    KERNEL_INVARIANTS,
    LMSTUDIO_URL,
    OLLAMA_URL,
    RUNTIME_IHSAN_THRESHOLD,
    SNR_THRESHOLD,
    SNR_THRESHOLD_T0_ELITE,
    STRICT_IHSAN_THRESHOLD,
    TIMESCALE_T1_CYCLE_MS,
    TIMESCALE_T2_CYCLE_SECONDS,
)


@dataclass(frozen=True)
class Constitution:
    """Immutable constitutional thresholds loaded once at boot."""
    ihsan_threshold: float = IHSAN_THRESHOLD           # 0.95
    ihsan_strict: float = STRICT_IHSAN_THRESHOLD       # 0.99
    ihsan_runtime: float = RUNTIME_IHSAN_THRESHOLD     # 1.0
    snr_minimum: float = SNR_THRESHOLD                 # 0.85
    snr_elite: float = SNR_THRESHOLD_T0_ELITE          # 0.98
    adl_gini_max: float = ADL_GINI_THRESHOLD           # 0.35
    ihsan_weights: dict = field(default_factory=lambda: dict(IHSAN_WEIGHTS))
    kernel_invariants: tuple = KERNEL_INVARIANTS
    got_max_hypotheses: int = GOT_MAX_HYPOTHESES       # 5
    faiss_dim: int = FAISS_EMBEDDING_DIM               # 384
    faiss_floor: float = FAISS_SIMILARITY_FLOOR        # 0.35
    t1_cycle_ms: int = TIMESCALE_T1_CYCLE_MS           # 50
    t2_cycle_s: float = TIMESCALE_T2_CYCLE_SECONDS     # 5.0

    def validate(self) -> bool:
        """Verify all invariants hold. Called once at boot, never skipped."""
        assert self.ihsan_threshold >= 0.95, "Ihsan floor violated"
        assert self.snr_minimum >= 0.85, "SNR floor violated"
        assert self.adl_gini_max <= 0.40, "ADL justice gate violated"
        assert len(self.kernel_invariants) == 3, "Kernel invariants incomplete"
        assert abs(sum(self.ihsan_weights.values()) - 1.0) < 1e-6
        return True
```

### 3.2 Subsystem Registry

```python
import asyncio
import logging
from enum import Enum, auto

logger = logging.getLogger("bizra.node")


class SubsystemState(Enum):
    UNINITIALIZED = auto()
    INITIALIZING = auto()
    READY = auto()
    FAILED = auto()


class SubsystemRegistry:
    """Tracks initialization state of all node subsystems."""

    INIT_ORDER: Final[list[str]] = [
        "engram",          # Layer 1: Knowledge (FAISS + episodic memory)
        "pat7",            # Layer 2: Chain of Reasoning (7 PAT agents)
        "rlm",             # Layer 3: Recursive Language Model
        "ttrl",            # Layer 4: Test-Time Reinforcement Learning
        "sso",             # Layer 5: Spectral Stability Optimization
        "guardian",        # Cross-cutting: Guardian Council (FATE gates)
        "action_bus",      # Cross-cutting: Event dispatch
        "receipt_chain",   # Cross-cutting: Cryptographic audit trail
        "reflex_ledger",   # Cross-cutting: Compiled System 1 reflexes
    ]

    def __init__(self) -> None:
        self._states: dict[str, SubsystemState] = {
            name: SubsystemState.UNINITIALIZED for name in self.INIT_ORDER
        }

    def set_state(self, name: str, state: SubsystemState) -> None:
        if name not in self._states:
            raise ValueError(f"Unknown subsystem: {name}")
        self._states[name] = state

    def all_ready(self) -> bool:
        return all(s == SubsystemState.READY for s in self._states.values())

    def failed_subsystems(self) -> list[str]:
        return [n for n, s in self._states.items() if s == SubsystemState.FAILED]
```

### 3.3 BIZRANode Class

```python
@dataclass
class DeviceConfig:
    """Hardware configuration for Ahmed's node."""
    gpu_name: str = "RTX 4070"
    vram_gb: int = 12
    model_name: str = "bizra-7b-moe"
    model_params_b: float = 7.0
    lm_studio_url: str = LMSTUDIO_URL
    ollama_url: str = OLLAMA_URL


class BIZRANode:
    """A single BIZRA node representing one human (Ahmed) with one device.
    Lifecycle: __init__ -> boot() -> accept_task() -> ... -> shutdown()"""

    def __init__(self, profile_id: str, device: DeviceConfig,
                 constitution: Constitution | None = None) -> None:
        self.profile_id = profile_id
        self.device = device
        self.constitution = constitution or Constitution()
        self.subsystems = SubsystemRegistry()
        self._booted = False

        # Subsystem instances (populated during boot)
        self.engram: EngramStore | None = None
        self.pat7: PAT7Pipeline | None = None
        self.rlm: RecursiveLanguageModel | None = None
        self.ttrl: TTRLEngine | None = None
        self.sso: SSOProjector | None = None
        self.guardian: GuardianCouncil | None = None
        self.action_bus: ActionBus | None = None
        self.receipt_chain: ReceiptChain | None = None
        self.reflex_ledger: ReflexLedger | None = None

    async def boot(self) -> bool:
        """Boot all subsystems in dependency order. Returns True on success."""
        logger.info("node.boot profile_id=%s model=%s",
                     self.profile_id, self.device.model_name)
        self.constitution.validate()

        init_map = {
            "engram": self._init_engram, "pat7": self._init_pat7,
            "rlm": self._init_rlm, "ttrl": self._init_ttrl,
            "sso": self._init_sso, "guardian": self._init_guardian,
            "action_bus": self._init_action_bus,
            "receipt_chain": self._init_receipt_chain,
            "reflex_ledger": self._init_reflex_ledger,
        }

        for name in SubsystemRegistry.INIT_ORDER:
            self.subsystems.set_state(name, SubsystemState.INITIALIZING)
            try:
                await init_map[name]()
                self.subsystems.set_state(name, SubsystemState.READY)
            except Exception as exc:
                logger.error("subsystem.%s FAILED: %s", name, exc)
                self.subsystems.set_state(name, SubsystemState.FAILED)
                return False

        if not await self._run_health_checks():
            return False

        self._booted = True
        if self.action_bus:
            await self.action_bus.emit("NodeReady", {"profile_id": self.profile_id})
        return True

    # --- Subsystem init methods (one per subsystem) ---

    async def _init_engram(self) -> None:
        self.engram = EngramStore(
            embedding_dim=self.constitution.faiss_dim,
            similarity_floor=self.constitution.faiss_floor)
        await self.engram.load_or_create()

    async def _init_pat7(self) -> None:
        self.pat7 = PAT7Pipeline(
            constitution=self.constitution, engram=self.engram,
            lm_url=self.device.lm_studio_url)

    async def _init_rlm(self) -> None:
        self.rlm = RecursiveLanguageModel(
            max_depth=self.constitution.got_max_hypotheses,
            lm_url=self.device.lm_studio_url)

    async def _init_ttrl(self) -> None:
        self.ttrl = TTRLEngine(
            model_name=self.device.model_name, vram_gb=self.device.vram_gb)

    async def _init_sso(self) -> None:
        self.sso = SSOProjector()

    async def _init_guardian(self) -> None:
        self.guardian = GuardianCouncil(
            ihsan_threshold=self.constitution.ihsan_threshold,
            snr_minimum=self.constitution.snr_minimum,
            adl_gini_max=self.constitution.adl_gini_max)

    async def _init_action_bus(self) -> None:
        self.action_bus = ActionBus()

    async def _init_receipt_chain(self) -> None:
        self.receipt_chain = ReceiptChain(profile_id=self.profile_id)
        await self.receipt_chain.initialize_genesis()

    async def _init_reflex_ledger(self) -> None:
        self.reflex_ledger = ReflexLedger(engram=self.engram)
        await self.reflex_ledger.load_persisted()
```

### 3.4 Health Checks

```python
import httpx

class HealthCheckResult:
    def __init__(self, name: str, passed: bool, detail: str = "") -> None:
        self.name = name
        self.passed = passed
        self.detail = detail


async def _run_health_checks(self: BIZRANode) -> bool:
    """All checks must pass. No degraded constitutional state permitted."""
    checks: list[HealthCheckResult] = []

    # 1. LM Studio reachable
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{self.device.lm_studio_url}/v1/models")
            checks.append(HealthCheckResult("lm_studio", resp.status_code == 200))
    except httpx.ConnectError:
        checks.append(HealthCheckResult("lm_studio", False, "connection refused"))

    # 2. API token present (never log value)
    from core.integration.constants import LM_API_TOKEN
    checks.append(HealthCheckResult("api_token", bool(LM_API_TOKEN)))

    # 3. Python venv active
    import sys
    venv_ok = hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
    checks.append(HealthCheckResult("venv", venv_ok))

    # 4. FAISS index loadable
    checks.append(HealthCheckResult("faiss_index",
        self.engram is not None and self.engram.is_loaded()))

    # 5. AHK bridge reachable (TCP:9742)
    try:
        reader, writer = await asyncio.wait_for(
            asyncio.open_connection("127.0.0.1", 9742), timeout=2.0)
        writer.close()
        await writer.wait_closed()
        checks.append(HealthCheckResult("ahk_bridge", True))
    except (OSError, asyncio.TimeoutError):
        checks.append(HealthCheckResult("ahk_bridge", False, "port 9742 unreachable"))

    all_passed = all(c.passed for c in checks)
    for c in checks:
        level = logging.INFO if c.passed else logging.ERROR
        logger.log(level, "health.%s %s %s", c.name,
                   "PASS" if c.passed else "FAIL", c.detail)
    return all_passed
```

---

## 4. Error Handling

| Failure | Behavior | Rationale |
|---------|----------|-----------|
| LM Studio unreachable | Boot FAILS | No inference = no reasoning |
| API token missing | Boot FAILS | Cannot authenticate with LM backend |
| FAISS index corrupt | Re-create empty, boot continues | New node has no memories yet |
| AHK bridge down | Boot FAILS for desktop tasks | Cannot actuate without body |
| Subsystem init exception | Mark FAILED, halt boot | Partial boot = constitutional violation |

---

## 5. TDD Anchors

```python
import pytest


class TestNodeInit:
    """Phase 52.1: Node initialization tests."""

    @pytest.fixture
    def constitution(self):
        return Constitution()

    @pytest.fixture
    def device(self):
        return DeviceConfig(gpu_name="RTX 4070", vram_gb=12)

    def test_node_init(self, constitution, device):
        """BIZRANode instantiates with valid config."""
        node = BIZRANode(profile_id="ahmed-dubai-001",
                         device=device, constitution=constitution)
        assert node.profile_id == "ahmed-dubai-001"
        assert node.constitution.ihsan_threshold >= 0.95
        assert not node._booted

    def test_subsystem_order(self):
        """Subsystems initialize in strict dependency order."""
        expected = ["engram", "pat7", "rlm", "ttrl", "sso",
                    "guardian", "action_bus", "receipt_chain", "reflex_ledger"]
        assert SubsystemRegistry.INIT_ORDER == expected

    def test_constitution_loaded(self, constitution):
        """Constitution loads all thresholds from constants.py."""
        assert constitution.ihsan_threshold == 0.95
        assert constitution.snr_minimum == 0.85
        assert constitution.adl_gini_max == 0.35
        assert len(constitution.kernel_invariants) == 3
        assert constitution.validate() is True

    def test_constitution_ihsan_weights_sum(self, constitution):
        """Ihsan dimension weights must sum to exactly 1.0."""
        assert abs(sum(constitution.ihsan_weights.values()) - 1.0) < 1e-6

    @pytest.mark.asyncio
    async def test_health_check_pass(self, monkeypatch):
        """Health check passes when all services reachable."""
        node = BIZRANode(profile_id="test", device=DeviceConfig())
        # ... setup mocks for httpx, asyncio.open_connection
        result = await node.boot()
        assert result is True
        assert node.subsystems.all_ready()

    @pytest.mark.asyncio
    async def test_health_check_fail_lm_studio(self, monkeypatch):
        """Health check fails when LM Studio is unreachable."""
        node = BIZRANode(profile_id="test", device=DeviceConfig())
        # ... mock LM Studio connection failure
        result = await node.boot()
        assert result is False

    def test_subsystem_all_ready(self):
        """all_ready() only True when every subsystem is READY."""
        registry = SubsystemRegistry()
        assert not registry.all_ready()
        for name in registry.INIT_ORDER:
            registry.set_state(name, SubsystemState.READY)
        assert registry.all_ready()

    def test_subsystem_failed_detection(self):
        """failed_subsystems() reports which subsystems failed."""
        registry = SubsystemRegistry()
        registry.set_state("engram", SubsystemState.READY)
        registry.set_state("pat7", SubsystemState.FAILED)
        assert "pat7" in registry.failed_subsystems()

    def test_unknown_subsystem_raises(self):
        """Setting state on unknown subsystem raises ValueError."""
        registry = SubsystemRegistry()
        with pytest.raises(ValueError, match="Unknown subsystem"):
            registry.set_state("nonexistent", SubsystemState.READY)
```
