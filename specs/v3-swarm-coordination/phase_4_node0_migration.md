# Phase 4: node0_activate.py Migration — SwarmEngine Wiring

> ADR-004 | V3 Unified Swarm Coordination Engine
> Standing on Giants: Fowler (Strangler Fig, 2004) · Boyd (OODA, 1976) · Beck (incremental change, 1999)

## 4.1 Problem Statement

`scripts/node0_activate.py` contains the production mission pipeline (~1,600 lines).
The inline `_execute_mission()` method runs agents sequentially with direct httpx
calls, computes SNR/Ihsan, and emits proof receipts.

Phase 2 defined `SwarmEngine` as the replacement execution coordinator.
This phase describes **how** to migrate the inline logic to SwarmEngine
without breaking the existing `cmd_mission()` CLI path.

Strategy: **Strangler Fig** — wrap the existing code, don't rewrite it.
SwarmEngine calls the same `_call_agent()` function; the only change is
who manages the loop and what events are emitted.

## 4.2 Requirements

- SwarmEngine replaces the `for a in agents: await _call_agent(a)` loop
- `_call_agent()` signature is unchanged (SwarmEngine wraps it via `call_fn`)
- AutoModelRouter pre-load step moves into SwarmEngine's PRELOADING phase
- Equalizer check moves into SwarmEngine's EQUALIZING phase
- SNR/Ihsan scoring remains in node0_activate (caller responsibility)
- GoT synthesis remains in node0_activate (separate pipeline)
- Proof receipt emission remains in node0_activate (orthogonal concern)
- Feature flag: `SWARM_ENGINE_ENABLED` env var (default: False initially)
- Backward-compatible: when flag is off, old inline path runs unchanged

## 4.3 Migration Pseudocode

```
# ── In node0_activate.py — _execute_mission() ──

IMPORT os FROM os
IMPORT SwarmConfig, SwarmTopology FROM core.swarm.types
IMPORT SwarmEngine FROM core.swarm.engine
IMPORT wire_swarm_to_bus FROM core.swarm.event_bridge

# ── Feature flag ──

_SWARM_ENGINE_ENABLED = os.getenv("SWARM_ENGINE_ENABLED", "").lower() in ("1", "true")


# ── Inside the Node0Orchestrator class ──

ASYNC FUNCTION _execute_mission(self, mission, agents, yaml_config):
    """Execute a mission through PAT agents.

    When SWARM_ENGINE_ENABLED=true, delegates to SwarmEngine.
    Otherwise, runs the existing sequential httpx loop (unchanged).
    """

    IF _SWARM_ENGINE_ENABLED:
        RETURN AWAIT self._execute_via_swarm_engine(
            mission, agents, yaml_config
        )

    # ── Legacy path (unchanged) ──
    # Existing code: for a in agents: result = await _call_agent(a)
    # ... (no changes to this block)


ASYNC FUNCTION _execute_via_swarm_engine(self, mission, agents, yaml_config):
    """New path: SwarmEngine-managed execution."""

    # 1. Build agent specs from PAT_AGENTS dict
    agent_specs = []
    FOR agent_id IN agents:
        pat_data = PAT_AGENTS.get(agent_id, {})
        IF pat_data:
            spec = AgentSpec.from_pat_agent(agent_id, pat_data)
            agent_specs.append(spec)

    # 2. Select topology from config (default: SEQUENTIAL for safety)
    topology_name = yaml_config.get("swarm_topology", "sequential").upper()
    TRY:
        topology = SwarmTopology[topology_name]
    EXCEPT KeyError:
        topology = SwarmTopology.SEQUENTIAL

    # 3. Build SwarmConfig
    config = SwarmConfig(
        topology=topology,
        max_concurrent=yaml_config.get("swarm_max_concurrent", 3),
        preload_models=True,
        equalizer_enabled=True,
    )

    # 4. Create engine with existing AutoModelRouter
    engine = SwarmEngine(
        config=config,
        model_router=self._model_router,
    )

    # 5. Optionally wire to EventBus (if available in sovereign runtime)
    IF hasattr(self, "_event_bus") AND self._event_bus:
        wire_swarm_to_bus(engine, self._event_bus)

    # 6. Define call_fn that wraps existing _call_agent
    ASYNC FUNCTION call_fn(agent_spec: AgentSpec) -> Dict:
        """Adapter: AgentSpec → existing _call_agent(agent_id)."""
        RETURN AWAIT self._call_agent(
            agent_id=agent_spec.id,
            mission=mission,
            config=yaml_config,
        )

    # 7. Execute through engine
    swarm_result = AWAIT engine.execute_mission(
        mission_id=mission["id"],
        agents=agent_specs,
        call_fn=call_fn,
        model_config=yaml_config,
    )

    # 8. Return results in the same format as legacy path
    RETURN swarm_result["results"]
```

## 4.4 Config Surface

New keys in `proactive_config.yaml` (all optional with safe defaults):

```yaml
# Swarm coordination (Phase 4)
swarm_topology: sequential     # sequential | parallel | hierarchical_mesh
swarm_max_concurrent: 3        # Max parallel agents (PARALLEL/MESH only)
swarm_engine_enabled: false    # Override env var from config
```

The env var `SWARM_ENGINE_ENABLED` takes precedence over the YAML key.
This allows CI to force-disable without touching config files.

## 4.5 Rollback Plan

If SwarmEngine causes regressions in production:

1. Set `SWARM_ENGINE_ENABLED=false` (env var or config)
2. Restart Node0 — legacy path runs immediately
3. No code change needed for rollback

The Strangler Fig pattern ensures the old code is preserved intact
alongside the new path. Both paths call the same `_call_agent()`.

## 4.6 TDD Anchors

```
# tests/core/swarm/test_node0_migration.py

TEST test_legacy_path_when_disabled():
    """When SWARM_ENGINE_ENABLED=false, legacy loop runs."""
    WITH patch.dict(os.environ, {"SWARM_ENGINE_ENABLED": "false"}):
        orchestrator = Node0Orchestrator(...)
        result = AWAIT orchestrator._execute_mission(mission, agents, config)
        # Verify: no SwarmEngine was instantiated
        ASSERT result IS NOT None

TEST test_swarm_engine_path_when_enabled():
    """When SWARM_ENGINE_ENABLED=true, SwarmEngine runs."""
    WITH patch.dict(os.environ, {"SWARM_ENGINE_ENABLED": "true"}):
        WITH patch("core.swarm.engine.SwarmEngine.execute_mission") AS mock_exec:
            mock_exec.return_value = {
                "results": [{"agent": "a", "success": True}],
                "topology": "sequential",
                "eq_action": None,
            }
            orchestrator = Node0Orchestrator(...)
            result = AWAIT orchestrator._execute_mission(mission, agents, config)
            ASSERT mock_exec.called

TEST test_swarm_results_format_matches_legacy():
    """SwarmEngine path returns same format as legacy path."""
    # Both paths should return List[Dict] with {agent, success, ...}
    swarm_results = AWAIT engine.execute_mission(...)
    legacy_results = AWAIT legacy_execute(...)
    FOR sr, lr IN zip(swarm_results, legacy_results):
        ASSERT set(sr.keys()) >= {"agent", "success"}
        ASSERT set(lr.keys()) >= {"agent", "success"}

TEST test_topology_selection_from_config():
    """Config key 'swarm_topology' selects the topology."""
    config = {"swarm_topology": "parallel", "swarm_max_concurrent": 2}
    # Verify ParallelStrategy is selected
    engine = _build_engine_from_config(config)
    ASSERT isinstance(engine._strategy, ParallelStrategy)

TEST test_topology_fallback_on_invalid():
    """Invalid topology name falls back to SEQUENTIAL."""
    config = {"swarm_topology": "quantum_mesh"}
    engine = _build_engine_from_config(config)
    ASSERT isinstance(engine._strategy, SequentialStrategy)

TEST test_event_bus_wired_when_available():
    """If orchestrator has _event_bus, bridge is created."""
    bus = EventBus()
    orchestrator = Node0Orchestrator(...)
    orchestrator._event_bus = bus

    WITH patch("core.swarm.event_bridge.wire_swarm_to_bus") AS mock_wire:
        AWAIT orchestrator._execute_via_swarm_engine(mission, agents, config)
        ASSERT mock_wire.called

TEST test_call_fn_adapter():
    """call_fn correctly maps AgentSpec.id → _call_agent(agent_id)."""
    call_log = []
    ASYNC FUNCTION fake_call_agent(agent_id, mission, config):
        call_log.append(agent_id)
        RETURN {"agent": agent_id, "success": True}

    # Build spec and call through adapter
    spec = AgentSpec(id="strategist", role=AgentRole.STRATEGIST, ...)
    result = AWAIT call_fn(spec)
    ASSERT call_log == ["strategist"]
```

## 4.7 Migration Sequence

```
Week 1: Implement SwarmEngine + types (Phases 1-2)
         Tests pass with mocked agents
Week 2: Implement EventBridge (Phase 3)
         Wire SwarmEngine → EventBus in test harness
Week 3: Wire into node0_activate.py (Phase 4)
         Feature flag off — legacy path unchanged
         Feature flag on — SwarmEngine path runs in dev
Week 4: Smoke test with live LM Studio
         Enable for one mission type (manual test)
Week 5: Enable by default in proactive_config.yaml
         Monitor logs for swarm.* events
         Keep legacy code for 2 more sprints then remove
```
