# Phase 2: SwarmEngine — Unified Execution Coordinator

> ADR-004 | V3 Unified Swarm Coordination Engine
> Standing on Giants: Boyd (OODA, 1976) . Amdahl (parallel fraction, 1967) . Deming (PDCA, 1950)

## 2.1 Requirements

- `SwarmEngine` replaces the inline `_execute_mission()` logic in node0_activate.py
- Supports all three topologies: SEQUENTIAL, PARALLEL, HIERARCHICAL_MESH
- Integrates AutoModelRouter for PRELOADING phase
- Integrates EqualizerAgent for EQUALIZING phase
- Emits `SwarmEvent` at each phase transition (observable by event bus)
- Graceful degradation: if any subsystem fails, falls back to sequential direct HTTP
- Must not break existing `cmd_mission()` CLI path

## 2.2 Architecture

```
                    ┌──────────────────────────────────────┐
                    │           SwarmEngine                  │
                    │                                        │
                    │  ┌──────────┐  ┌──────────────────┐   │
                    │  │ Topology │  │ AutoModelRouter   │   │
                    │  │ Strategy │  │ (pre-load/equalize)│  │
                    │  └────┬─────┘  └────────┬─────────┘   │
                    │       │                 │              │
                    │  ┌────▼─────────────────▼──────────┐  │
                    │  │        Phase Runner              │  │
                    │  │  INIT → PRELOAD → EXECUTE →     │  │
                    │  │  SYNTHESIZE → SCORE → EQUALIZE  │  │
                    │  └─────────────────────────────────┘  │
                    │                                        │
                    │  ┌──────────┐  ┌──────────────────┐   │
                    │  │ Event    │  │ Existing direct   │   │
                    │  │ Emitter  │  │ httpx calls       │   │
                    │  └──────────┘  │ (unchanged)       │   │
                    │                └──────────────────┘   │
                    └──────────────────────────────────────┘
```

## 2.3 Pseudocode: `core/swarm/engine.py`

```
IMPORT asyncio
IMPORT logging
IMPORT time
IMPORT Optional, Dict, List, Callable, Any FROM typing

IMPORT SwarmConfig, SwarmTopology, SwarmPhase FROM core.swarm.types
IMPORT SwarmEvent, SwarmEventKind, AgentSpec FROM core.swarm.types

logger = logging.getLogger("SwarmEngine")

# ── Event listener type ──

EventListener = Callable[[SwarmEvent], None]

# ── Topology Strategies ──

CLASS TopologyStrategy:
    """Base class for execution topology."""

    ASYNC FUNCTION execute(
        agents: List[AgentSpec],
        call_fn: Callable[[AgentSpec], Awaitable[Dict]],
    ) -> List[Dict]:
        RAISE NotImplementedError

CLASS SequentialStrategy(TopologyStrategy):
    """Execute agents one at a time (current node0 behavior).

    Standing on Giants: Amdahl — when serial fraction dominates,
    sequential is correct. LM Studio loads one model at a time.
    """

    ASYNC FUNCTION execute(self, agents, call_fn) -> List[Dict]:
        results = []
        FOR agent IN agents:
            result = AWAIT call_fn(agent)
            results.append(result)
        RETURN results

CLASS ParallelStrategy(TopologyStrategy):
    """Execute agents concurrently with bounded parallelism.

    Requires pre-loaded models (AutoModelRouter) to avoid
    LM Studio contention on concurrent model loads.
    """

    FUNCTION __init__(self, max_concurrent: int = 3):
        self._semaphore = asyncio.Semaphore(max_concurrent)

    ASYNC FUNCTION execute(self, agents, call_fn) -> List[Dict]:
        ASYNC FUNCTION _bounded_call(agent):
            ASYNC WITH self._semaphore:
                RETURN AWAIT call_fn(agent)

        tasks = [_bounded_call(a) FOR a IN agents]
        RETURN AWAIT asyncio.gather(*tasks, return_exceptions=True)

CLASS HierarchicalMeshStrategy(TopologyStrategy):
    """Queen-worker pattern: coordinator runs first, then workers in parallel.

    Phase 1: Coordinator agent produces a plan
    Phase 2: Worker agents execute plan items in parallel
    Phase 3: Coordinator synthesizes worker outputs
    """

    FUNCTION __init__(self, max_concurrent: int = 3):
        self._parallel = ParallelStrategy(max_concurrent)

    ASYNC FUNCTION execute(self, agents, call_fn) -> List[Dict]:
        # Separate coordinator from workers
        coordinators = [a FOR a IN agents IF a.role == AgentRole.COORDINATOR]
        workers = [a FOR a IN agents IF a.role != AgentRole.COORDINATOR]

        results = []

        # Phase 1: coordinator plans
        IF coordinators:
            coord_result = AWAIT call_fn(coordinators[0])
            results.append(coord_result)

        # Phase 2: workers execute in parallel
        IF workers:
            worker_results = AWAIT self._parallel.execute(workers, call_fn)
            results.extend(worker_results)

        # Phase 3: coordinator synthesizes (second call with worker context)
        IF coordinators AND workers:
            # Coordinator gets a second pass with worker outputs as context
            # This is injected via call_fn closure, not here
            synth_result = AWAIT call_fn(coordinators[0])
            results.append(synth_result)

        RETURN results


# ── SwarmEngine ──

CLASS SwarmEngine:
    """Unified swarm execution coordinator.

    Replaces inline _execute_mission() with a phased, topology-aware,
    event-emitting execution engine.
    """

    FUNCTION __init__(
        self,
        config: SwarmConfig = None,
        model_router: Optional[AutoModelRouter] = None,
        equalizer: Optional[EqualizerAgent] = None,
    ):
        self._config = config OR SwarmConfig()
        self._model_router = model_router
        self._equalizer = equalizer
        self._listeners: List[EventListener] = []
        self._phase = SwarmPhase.INITIALIZING
        self._swarm_id = ""

        # Select topology strategy
        self._strategy = self._select_strategy()

    FUNCTION _select_strategy(self) -> TopologyStrategy:
        topo = self._config.topology
        IF topo == SwarmTopology.PARALLEL:
            RETURN ParallelStrategy(self._config.max_concurrent)
        IF topo == SwarmTopology.HIERARCHICAL_MESH:
            RETURN HierarchicalMeshStrategy(self._config.max_concurrent)
        RETURN SequentialStrategy()

    FUNCTION on_event(self, listener: EventListener):
        """Register an event listener."""
        self._listeners.append(listener)

    FUNCTION _emit(self, kind: SwarmEventKind, **kwargs):
        evt = SwarmEvent(
            kind=kind,
            swarm_id=self._swarm_id,
            timestamp=time.monotonic(),
            **kwargs,
        )
        FOR listener IN self._listeners:
            TRY:
                listener(evt)
            EXCEPT Exception:
                PASS  # Listeners must not break the engine

    # ── Main execution pipeline ──

    ASYNC FUNCTION execute_mission(
        self,
        mission_id: str,
        agents: List[AgentSpec],
        call_fn: Callable[[AgentSpec], Awaitable[Dict]],
        model_config: Dict = None,
    ) -> Dict:
        """Execute a complete mission through all phases.

        Returns the same result dict structure as _execute_mission() today,
        ensuring backward compatibility with cmd_mission().
        """
        self._swarm_id = mission_id
        self._emit(SwarmEventKind.SWARM_CREATED, data={"agent_count": len(agents)})

        # ── Phase: PRELOADING ──
        self._set_phase(SwarmPhase.PRELOADING)
        IF self._config.preload_models AND self._model_router:
            TRY:
                fleet = AWAIT self._model_router.preload_mission_fleet(
                    agent_ids=[a.id FOR a IN agents],
                    config=model_config OR {},
                )
                loaded = sum(1 FOR v IN fleet.values() IF v)
                logger.info("Pre-loaded %d/%d models", loaded, len(fleet))
                self._emit(SwarmEventKind.MODEL_PRELOADED, data=fleet)
            EXCEPT Exception AS e:
                logger.debug("Pre-load skipped: %s", e)

        # ── Phase: EXECUTING ──
        self._set_phase(SwarmPhase.EXECUTING)

        ASYNC FUNCTION _tracked_call(agent: AgentSpec) -> Dict:
            self._emit(SwarmEventKind.AGENT_STARTED, agent_id=agent.id)
            TRY:
                result = AWAIT call_fn(agent)
                kind = AGENT_COMPLETED IF result.get("success") ELSE AGENT_FAILED
                self._emit(kind, agent_id=agent.id, data=result)
                RETURN result
            EXCEPT Exception AS e:
                self._emit(SwarmEventKind.AGENT_FAILED, agent_id=agent.id,
                           data={"error": str(e)})
                RETURN {"agent": agent.id, "success": False, "error": str(e)}

        results = AWAIT self._strategy.execute(agents, _tracked_call)

        # Flatten exceptions from gather (ParallelStrategy)
        results = [
            r IF isinstance(r, dict) ELSE {"success": False, "error": str(r)}
            FOR r IN results
        ]

        # ── Phase: SCORING ──
        self._set_phase(SwarmPhase.SCORING)
        # Scoring is handled by caller (node0_activate) via _compute_real_snr
        # Engine returns raw results; caller applies scoring pipeline

        # ── Phase: EQUALIZING ──
        self._set_phase(SwarmPhase.EQUALIZING)
        eq_action = None
        IF self._config.equalizer_enabled AND self._model_router:
            TRY:
                successful = sum(1 FOR r IN results IF r.get("success"))
                eq_action = AWAIT self._model_router.check_equalizer(
                    ihsan_score=successful / max(len(results), 1),
                    backlog=len(agents),
                    presence=0,
                )
                IF eq_action:
                    logger.info("Equalizer: %s", eq_action)
                    self._emit(SwarmEventKind.EQUALIZER_ACTION,
                               data={"action": eq_action})
            EXCEPT Exception AS e:
                logger.debug("Equalizer check skipped: %s", e)

        # ── Phase: COMPLETE ──
        self._set_phase(SwarmPhase.COMPLETE)
        self._emit(SwarmEventKind.MISSION_COMPLETE, data={
            "results_count": len(results),
            "eq_action": eq_action,
        })

        RETURN {
            "results": results,
            "topology": self._config.topology.value,
            "eq_action": eq_action,
        }

    FUNCTION _set_phase(self, phase: SwarmPhase):
        old = self._phase
        self._phase = phase
        self._emit(SwarmEventKind.PHASE_CHANGED, data={
            "from": old.value, "to": phase.value,
        })
```

## 2.4 TDD Anchors

```
# tests/core/swarm/test_engine.py

TEST test_sequential_execution_order():
    """Sequential strategy executes agents in order."""
    call_order = []
    ASYNC FUNCTION fake_call(agent):
        call_order.append(agent.id)
        RETURN {"agent": agent.id, "success": True}

    engine = SwarmEngine(config=SwarmConfig(topology=SEQUENTIAL))
    result = AWAIT engine.execute_mission("m1", [spec_a, spec_b], fake_call)
    ASSERT call_order == ["a", "b"]

TEST test_parallel_execution_concurrent():
    """Parallel strategy runs agents concurrently."""
    engine = SwarmEngine(config=SwarmConfig(topology=PARALLEL, max_concurrent=3))
    # Mock agents with sleep — total time < sum of individual times
    result = AWAIT engine.execute_mission("m1", agents, fake_call)
    ASSERT len(result["results"]) == len(agents)

TEST test_hierarchical_mesh_coordinator_first():
    """Hierarchical mesh runs coordinator before workers."""
    call_order = []
    engine = SwarmEngine(config=SwarmConfig(topology=HIERARCHICAL_MESH))
    result = AWAIT engine.execute_mission("m1", agents, tracked_call)
    ASSERT call_order[0] == "coordinator"

TEST test_event_emission():
    """Engine emits events at each phase transition."""
    events = []
    engine = SwarmEngine()
    engine.on_event(LAMBDA evt: events.append(evt))
    AWAIT engine.execute_mission("m1", [agent], fake_call)
    event_kinds = [e.kind FOR e IN events]
    ASSERT SWARM_CREATED IN event_kinds
    ASSERT PHASE_CHANGED IN event_kinds
    ASSERT MISSION_COMPLETE IN event_kinds

TEST test_preload_integration():
    """Engine calls AutoModelRouter.preload_mission_fleet."""
    mock_router = MockAutoModelRouter()
    engine = SwarmEngine(model_router=mock_router)
    AWAIT engine.execute_mission("m1", [agent], fake_call, model_config={})
    ASSERT mock_router.preload_called IS True

TEST test_graceful_degradation_on_preload_failure():
    """Engine continues if pre-load fails."""
    mock_router = FailingRouter()
    engine = SwarmEngine(model_router=mock_router)
    result = AWAIT engine.execute_mission("m1", [agent], fake_call)
    ASSERT len(result["results"]) == 1  # Still executed

TEST test_agent_failure_does_not_halt_swarm():
    """One agent failure doesn't stop the entire swarm."""
    ASYNC FUNCTION failing_call(agent):
        IF agent.id == "bad":
            RAISE Exception("boom")
        RETURN {"agent": agent.id, "success": True}

    engine = SwarmEngine()
    result = AWAIT engine.execute_mission("m1", [good, bad, good2], failing_call)
    successes = sum(1 FOR r IN result["results"] IF r.get("success"))
    ASSERT successes == 2
```

## 2.5 Migration Path

```
# node0_activate.py migration (Phase 3):
#
# BEFORE (current):
#   results = []
#   for a in agents:
#       result_item = await _call_agent(a)
#       results.append(result_item)
#
# AFTER (with SwarmEngine):
#   engine = SwarmEngine(
#       config=SwarmConfig(topology=SEQUENTIAL),
#       model_router=self._model_router,
#   )
#   swarm_result = await engine.execute_mission(
#       mission["id"], agent_specs, _call_agent, self._yaml_config
#   )
#   results = swarm_result["results"]
#
# The call_fn signature stays the same — SwarmEngine wraps it.
```
