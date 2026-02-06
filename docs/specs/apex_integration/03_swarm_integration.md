# Swarm Integration Specification

## Phase 3: SwarmOrchestrator ↔ Rust Lifecycle

**Module**: `core/sovereign/swarm_integration.py`
**Dependencies**: `core/apex/swarm_orchestrator.py`, `core/sovereign/rust_lifecycle.py`
**Lines**: ~220

---

## Functional Requirements

### FR-1: Rust Service as Swarm Member

Register Rust services (bizra-omega) as swarm members:
- Health monitoring via rust_lifecycle.py
- Scaling decisions affect Rust service allocation
- Self-healing triggers Rust service restart

### FR-2: Python-Rust Hybrid Swarms

Coordinate swarms with both Python agents and Rust services:
- Python agents for high-level reasoning
- Rust services for performance-critical operations
- Unified health reporting across both

### FR-3: Dynamic Scaling Bridge

Bridge SwarmOrchestrator scaling to Rust lifecycle:
- Scale up: Spawn additional Rust services
- Scale down: Gracefully shutdown Rust services
- Health degradation: Restart unhealthy services

---

## Pseudocode

```
MODULE swarm_integration

IMPORT SwarmOrchestrator, AgentConfig, HealthStatus, ScalingDecision FROM core.apex
IMPORT RustLifecycleManager, RustServiceStatus FROM core.sovereign.rust_lifecycle

# Constants (Hamilton's operations principles)
CONST HEALTH_CHECK_INTERVAL: INT = 30  # seconds
CONST RESTART_BACKOFF_BASE: INT = 5    # seconds
CONST MAX_RESTART_ATTEMPTS: INT = 3
CONST AVAILABILITY_TARGET: FLOAT = 0.999  # Three nines

CLASS RustServiceAdapter:
    """
    Adapts Rust services to SwarmOrchestrator agent protocol.

    Standing on Giants:
    - Hamilton (2007): Operations at scale principles
    - Verma/Borg (2015): Large-scale cluster management
    - Burns/K8s (2016): Container orchestration patterns
    """

    PROPERTY rust_lifecycle: RustLifecycleManager
    PROPERTY service_name: STRING
    PROPERTY restart_count: INT = 0
    PROPERTY last_health: HealthStatus = HealthStatus.UNKNOWN

    CONSTRUCTOR(self, service_name: STRING, lifecycle: RustLifecycleManager):
        self.service_name = service_name
        self.rust_lifecycle = lifecycle

    ASYNC METHOD health_check(self) -> HealthStatus:
        """
        Check Rust service health via lifecycle manager.

        Maps RustServiceStatus to SwarmOrchestrator HealthStatus.
        """
        rust_status = AWAIT self.rust_lifecycle.check_health(self.service_name)

        status_map = {
            RustServiceStatus.RUNNING: HealthStatus.HEALTHY,
            RustServiceStatus.STARTING: HealthStatus.DEGRADED,
            RustServiceStatus.STOPPING: HealthStatus.DEGRADED,
            RustServiceStatus.STOPPED: HealthStatus.UNHEALTHY,
            RustServiceStatus.FAILED: HealthStatus.CRITICAL,
            RustServiceStatus.UNKNOWN: HealthStatus.UNKNOWN,
        }

        self.last_health = status_map.get(rust_status, HealthStatus.UNKNOWN)
        RETURN self.last_health

    ASYNC METHOD restart(self) -> BOOL:
        """
        Restart Rust service with exponential backoff.

        Returns True if restart successful.
        """
        IF self.restart_count >= MAX_RESTART_ATTEMPTS:
            LOG.error(f"Max restart attempts reached for {self.service_name}")
            RETURN FALSE

        # Exponential backoff
        backoff = RESTART_BACKOFF_BASE * (2 ** self.restart_count)
        LOG.info(f"Restarting {self.service_name} after {backoff}s backoff")

        AWAIT asyncio.sleep(backoff)

        success = AWAIT self.rust_lifecycle.restart_service(self.service_name)

        IF success:
            self.restart_count = 0
            RETURN TRUE
        ELSE:
            self.restart_count += 1
            RETURN FALSE


CLASS HybridSwarmOrchestrator EXTENDS SwarmOrchestrator:
    """
    Orchestrates hybrid Python/Rust swarms.

    Manages both Python agents and Rust services as unified swarm.
    """

    PROPERTY rust_lifecycle: RustLifecycleManager
    PROPERTY rust_adapters: DICT[STRING, RustServiceAdapter]

    CONSTRUCTOR(self):
        SUPER().__init__()
        self.rust_lifecycle = RustLifecycleManager()
        self.rust_adapters = {}

    METHOD register_rust_service(self, service_name: STRING, config: AgentConfig):
        """
        Register a Rust service as a swarm member.

        Creates adapter for unified health/scaling management.
        """
        adapter = RustServiceAdapter(
            service_name=service_name,
            lifecycle=self.rust_lifecycle
        )
        self.rust_adapters[service_name] = adapter

        # Register health check with monitor
        self.health_monitor.register_health_check(
            agent_id=f"rust:{service_name}",
            callback=adapter.health_check
        )

        LOG.info(f"Registered Rust service: {service_name}")

    ASYNC METHOD apply_scaling_decision(self, decision: ScalingDecision, swarm_id: STRING):
        """
        Apply scaling decision to hybrid swarm.

        Scales both Python agents and Rust services proportionally.
        """
        swarm = self._swarms.get(swarm_id)
        IF NOT swarm:
            RAISE ValueError(f"Unknown swarm: {swarm_id}")

        # Partition instances by type
        python_agents = [a FOR a IN swarm.agents IF NOT a.id.startswith("rust:")]
        rust_services = [a FOR a IN swarm.agents IF a.id.startswith("rust:")]

        # Calculate scaling ratio
        IF decision.action == ScalingAction.SCALE_UP:
            delta = decision.target_count - decision.current_count
            # Scale Python and Rust proportionally
            python_delta = INT(delta * 0.7)  # 70% Python
            rust_delta = delta - python_delta  # 30% Rust

            AWAIT self._scale_up_python(swarm_id, python_delta)
            AWAIT self._scale_up_rust(swarm_id, rust_delta)

        ELIF decision.action == ScalingAction.SCALE_DOWN:
            delta = decision.current_count - decision.target_count
            # Prefer scaling down Python first (Rust has startup cost)
            python_delta = MIN(delta, LEN(python_agents) - 1)
            rust_delta = delta - python_delta

            AWAIT self._scale_down_python(swarm_id, python_delta)
            AWAIT self._scale_down_rust(swarm_id, rust_delta)

    ASYNC METHOD _scale_up_rust(self, swarm_id: STRING, count: INT):
        """Spawn additional Rust service instances."""
        FOR i IN RANGE(count):
            service_name = f"{swarm_id}-rust-{uuid.uuid4()[:8]}"
            success = AWAIT self.rust_lifecycle.start_service(service_name)

            IF success:
                self.register_rust_service(service_name, AgentConfig(
                    agent_type="rust-worker",
                    name=service_name,
                    capabilities={"inference", "consensus", "pci"}
                ))
            ELSE:
                LOG.error(f"Failed to start Rust service: {service_name}")

    ASYNC METHOD _scale_down_rust(self, swarm_id: STRING, count: INT):
        """Gracefully shutdown Rust service instances."""
        # Get Rust services in this swarm
        rust_services = [
            name FOR name IN self.rust_adapters
            IF name.startswith(swarm_id)
        ]

        # Shutdown oldest first (FIFO)
        FOR service_name IN rust_services[:count]:
            AWAIT self.rust_lifecycle.stop_service(service_name)
            DEL self.rust_adapters[service_name]

    ASYNC METHOD self_heal(self):
        """
        Self-healing loop for hybrid swarm.

        Checks all members and restarts unhealthy ones.
        """
        WHILE self._running:
            # Check all Rust adapters
            FOR service_name, adapter IN self.rust_adapters.items():
                health = AWAIT adapter.health_check()

                IF health IN (HealthStatus.UNHEALTHY, HealthStatus.CRITICAL):
                    LOG.warning(f"Unhealthy Rust service detected: {service_name}")
                    success = AWAIT adapter.restart()

                    IF NOT success:
                        # Replace with new instance
                        LOG.error(f"Restart failed, replacing: {service_name}")
                        AWAIT self._replace_rust_service(service_name)

            # Also check Python agents via parent
            AWAIT SUPER()._check_python_agents()

            AWAIT asyncio.sleep(HEALTH_CHECK_INTERVAL)

    ASYNC METHOD _replace_rust_service(self, old_service: STRING):
        """Replace failed Rust service with new instance."""
        # Capture config before removal
        old_config = self.rust_adapters[old_service]

        # Remove old
        DEL self.rust_adapters[old_service]
        AWAIT self.rust_lifecycle.stop_service(old_service)

        # Start new
        new_service = f"{old_service.split('-')[0]}-rust-{uuid.uuid4()[:8]}"
        AWAIT self.rust_lifecycle.start_service(new_service)
        self.register_rust_service(new_service, AgentConfig(
            agent_type="rust-worker",
            name=new_service
        ))

---

## TDD Anchors

### Test: Rust Service Health Check

```python
async def test_rust_health_check_mapping():
    """RustServiceStatus should map to HealthStatus correctly."""
    adapter = RustServiceAdapter("test-service", mock_lifecycle)

    mock_lifecycle.check_health.return_value = RustServiceStatus.RUNNING
    assert await adapter.health_check() == HealthStatus.HEALTHY

    mock_lifecycle.check_health.return_value = RustServiceStatus.FAILED
    assert await adapter.health_check() == HealthStatus.CRITICAL
```

### Test: Exponential Backoff

```python
async def test_restart_backoff():
    """Restarts should use exponential backoff."""
    adapter = RustServiceAdapter("test-service", mock_lifecycle)
    mock_lifecycle.restart_service.return_value = False

    start_times = []
    for _ in range(3):
        start = time.time()
        await adapter.restart()
        start_times.append(time.time() - start)

    # Each restart should take longer
    assert start_times[1] > start_times[0]
    assert start_times[2] > start_times[1]
```

### Test: Hybrid Scaling

```python
async def test_hybrid_scaling_proportional():
    """Scaling should affect Python and Rust proportionally."""
    orchestrator = HybridSwarmOrchestrator()
    orchestrator._swarms["test"] = Swarm(
        agents=[
            mock_python_agent("py-1"),
            mock_python_agent("py-2"),
            mock_rust_adapter("rust:svc-1"),
        ]
    )

    decision = ScalingDecision(
        action=ScalingAction.SCALE_UP,
        current_count=3,
        target_count=6,
        reason="High load"
    )

    await orchestrator.apply_scaling_decision(decision, "test")

    # Should have added ~2 Python, ~1 Rust
    assert len([a for a in orchestrator._swarms["test"].agents
                if not a.id.startswith("rust:")]) >= 4
```

### Test: Self-Healing

```python
async def test_self_healing_restarts_unhealthy():
    """Self-heal loop should restart unhealthy services."""
    orchestrator = HybridSwarmOrchestrator()
    orchestrator.register_rust_service("unhealthy-svc", mock_config)

    # Simulate unhealthy
    orchestrator.rust_adapters["unhealthy-svc"].last_health = HealthStatus.CRITICAL
    mock_lifecycle.restart_service.return_value = True

    # Run one iteration of self-heal
    await orchestrator._self_heal_iteration()

    # Should have attempted restart
    mock_lifecycle.restart_service.assert_called_once()
```

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                      HybridSwarmOrchestrator                             │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────┐     ┌─────────────────────────────────┐   │
│  │   Python Agents         │     │   Rust Services                 │   │
│  │   ──────────────        │     │   ─────────────                 │   │
│  │   • MasterReasoner      │     │   • bizra-api (Axum)            │   │
│  │   • DataAnalyzer        │     │   • bizra-inference             │   │
│  │   • ExecutionPlanner    │     │   • bizra-federation            │   │
│  │   • EthicsGuardian      │     │                                 │   │
│  └──────────┬──────────────┘     └──────────────┬──────────────────┘   │
│             │                                   │                       │
│             ▼                                   ▼                       │
│  ┌─────────────────────────┐     ┌─────────────────────────────────┐   │
│  │   HealthMonitor         │     │   RustServiceAdapter            │   │
│  │   (unified checks)      │◀───▶│   (lifecycle bridge)            │   │
│  └──────────┬──────────────┘     └──────────────┬──────────────────┘   │
│             │                                   │                       │
│             ▼                                   ▼                       │
│  ┌─────────────────────────┐     ┌─────────────────────────────────┐   │
│  │   ScalingManager        │     │   RustLifecycleManager          │   │
│  │   (scaling decisions)   │────▶│   (start/stop/restart)          │   │
│  └─────────────────────────┘     └─────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Edge Cases

| Case | Handling |
|------|----------|
| Rust service fails to start | Log error, don't add to swarm |
| All Rust services fail | Alert, operate in Python-only mode |
| Rust restart loop | Max 3 attempts, then replace |
| Scale down below minimum | Keep at least 1 of each type |

---

## File Output

**Target**: `core/sovereign/swarm_integration.py`
**Lines**: ~220
**Imports**:
```python
from core.apex import SwarmOrchestrator, AgentConfig, HealthStatus, ScalingDecision
from core.sovereign.rust_lifecycle import RustLifecycleManager, RustServiceStatus
```
