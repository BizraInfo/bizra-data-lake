"""SwarmEngine — Unified phased execution coordinator for agent missions.

ADR-004 Phase 2: Replaces inline _execute_mission() with a topology-aware,
event-emitting execution engine supporting SEQUENTIAL, PARALLEL, and
HIERARCHICAL_MESH topologies.

Standing on Giants: Boyd (OODA) . Amdahl (parallel fraction) . Deming (PDCA)
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Dict, List, Optional, Protocol

from core.swarm.types import (
    AgentRole,
    AgentSpec,
    SwarmConfig,
    SwarmEvent,
    SwarmEventKind,
    SwarmPhase,
    SwarmTopology,
)

logger = logging.getLogger("SwarmEngine")

# -- Type aliases --------------------------------------------------------------

EventListener = Callable[[SwarmEvent], None]
AgentCallFn = Callable[[AgentSpec], Awaitable[Dict[str, Any]]]


# -- Model router protocol (optional dependency) ------------------------------


class ModelRouterProtocol(Protocol):
    """Structural type for AutoModelRouter — avoids hard import."""

    async def preload_mission_fleet(
        self,
        agent_ids: List[str],
        config: Dict[str, Any],
    ) -> Dict[str, bool]: ...

    async def check_equalizer(
        self,
        ihsan_score: float,
        backlog: int,
        presence: int,
    ) -> Optional[str]: ...


# -- Topology Strategies -------------------------------------------------------


class TopologyStrategy:
    """Base class for execution topology."""

    async def execute(
        self,
        agents: List[AgentSpec],
        call_fn: AgentCallFn,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError


class SequentialStrategy(TopologyStrategy):
    """Execute agents one at a time.

    Standing on Giants: Amdahl — when serial fraction dominates,
    sequential is correct. LM Studio loads one model at a time.
    """

    async def execute(
        self,
        agents: List[AgentSpec],
        call_fn: AgentCallFn,
    ) -> List[Dict[str, Any]]:
        results: List[Dict[str, Any]] = []
        for agent in agents:
            result = await call_fn(agent)
            results.append(result)
        return results


class ParallelStrategy(TopologyStrategy):
    """Execute agents concurrently with bounded parallelism.

    Requires pre-loaded models to avoid LM Studio contention.
    """

    def __init__(self, max_concurrent: int = 3):
        self._semaphore = asyncio.Semaphore(max_concurrent)

    async def execute(
        self,
        agents: List[AgentSpec],
        call_fn: AgentCallFn,
    ) -> List[Dict[str, Any]]:
        async def _bounded_call(agent: AgentSpec) -> Dict[str, Any]:
            async with self._semaphore:
                return await call_fn(agent)

        tasks = [_bounded_call(a) for a in agents]
        return list(await asyncio.gather(*tasks, return_exceptions=True))


class HierarchicalMeshStrategy(TopologyStrategy):
    """Queen-worker pattern: coordinator plans, workers execute, coordinator synthesizes.

    Phase 1: Coordinator agent produces a plan
    Phase 2: Worker agents execute plan items in parallel
    Phase 3: Coordinator synthesizes worker outputs
    """

    def __init__(self, max_concurrent: int = 3):
        self._parallel = ParallelStrategy(max_concurrent)

    async def execute(
        self,
        agents: List[AgentSpec],
        call_fn: AgentCallFn,
    ) -> List[Dict[str, Any]]:
        coordinators = [a for a in agents if a.role == AgentRole.COORDINATOR]
        workers = [a for a in agents if a.role != AgentRole.COORDINATOR]

        results: List[Dict[str, Any]] = []

        # Phase 1: coordinator plans
        if coordinators:
            coord_result = await call_fn(coordinators[0])
            results.append(coord_result)

        # Phase 2: workers execute in parallel
        if workers:
            worker_results = await self._parallel.execute(workers, call_fn)
            results.extend(worker_results)

        # Phase 3: coordinator synthesizes (second call with worker context)
        if coordinators and workers:
            synth_result = await call_fn(coordinators[0])
            results.append(synth_result)

        return results


# -- Strategy Factory ----------------------------------------------------------

_STRATEGY_MAP = {
    SwarmTopology.SEQUENTIAL: lambda cfg: SequentialStrategy(),
    SwarmTopology.PARALLEL: lambda cfg: ParallelStrategy(cfg.max_concurrent),
    SwarmTopology.HIERARCHICAL_MESH: lambda cfg: HierarchicalMeshStrategy(
        cfg.max_concurrent
    ),
}


# -- SwarmEngine ---------------------------------------------------------------


class SwarmEngine:
    """Unified swarm execution coordinator.

    Replaces inline _execute_mission() with a phased, topology-aware,
    event-emitting execution engine.
    """

    def __init__(
        self,
        config: Optional[SwarmConfig] = None,
        model_router: Optional[ModelRouterProtocol] = None,
    ):
        self._config = config or SwarmConfig()
        self._model_router = model_router
        self._listeners: List[EventListener] = []
        self._phase = SwarmPhase.INITIALIZING
        self._swarm_id = ""
        self._strategy = _STRATEGY_MAP[self._config.topology](self._config)

    # -- Event system ----------------------------------------------------------

    def on_event(self, listener: EventListener) -> None:
        """Register an event listener."""
        self._listeners.append(listener)

    def _emit(
        self,
        kind: SwarmEventKind,
        agent_id: Optional[str] = None,
        data: Optional[Dict[str, Any]] = None,
    ) -> None:
        evt = SwarmEvent(
            kind=kind,
            swarm_id=self._swarm_id,
            agent_id=agent_id,
            data=data or {},
            timestamp=time.monotonic(),
        )
        for listener in self._listeners:
            try:
                listener(evt)
            except Exception:  # noqa: BLE001 — boundary boundary
                pass  # Listeners must not break the engine

    def _set_phase(self, phase: SwarmPhase) -> None:
        old = self._phase
        self._phase = phase
        self._emit(
            SwarmEventKind.PHASE_CHANGED, data={"from": old.value, "to": phase.value}
        )

    @property
    def phase(self) -> SwarmPhase:
        return self._phase

    # -- Main execution pipeline -----------------------------------------------

    async def execute_mission(
        self,
        mission_id: str,
        agents: List[AgentSpec],
        call_fn: AgentCallFn,
        model_config: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a complete mission through all phases.

        Returns a dict with keys: results, topology, eq_action.
        """
        self._swarm_id = mission_id
        self._emit(SwarmEventKind.SWARM_CREATED, data={"agent_count": len(agents)})

        # -- Phase: PRELOADING --
        self._set_phase(SwarmPhase.PRELOADING)
        if self._config.preload_models and self._model_router:
            try:
                fleet = await self._model_router.preload_mission_fleet(
                    agent_ids=[a.id for a in agents],
                    config=model_config or {},
                )
                loaded = sum(1 for v in fleet.values() if v)
                logger.info("Pre-loaded %d/%d models", loaded, len(fleet))
                self._emit(SwarmEventKind.MODEL_PRELOADED, data=fleet)
            except (asyncio.CancelledError, RuntimeError, OSError) as e:  # SEC-003 — async boundary
                logger.debug("Pre-load skipped: %s", e)

        # -- Phase: EXECUTING --
        self._set_phase(SwarmPhase.EXECUTING)

        async def _tracked_call(agent: AgentSpec) -> Dict[str, Any]:
            self._emit(SwarmEventKind.AGENT_STARTED, agent_id=agent.id)
            try:
                result = await call_fn(agent)
                kind = (
                    SwarmEventKind.AGENT_COMPLETED
                    if result.get("success")
                    else SwarmEventKind.AGENT_FAILED
                )
                self._emit(kind, agent_id=agent.id, data=result)
                return result
            except (asyncio.CancelledError, RuntimeError, OSError) as e:  # SEC-003 — async boundary
                self._emit(
                    SwarmEventKind.AGENT_FAILED,
                    agent_id=agent.id,
                    data={"error": str(e)},
                )
                return {"agent": agent.id, "success": False, "error": str(e)}

        results = await self._strategy.execute(agents, _tracked_call)

        # Flatten exceptions from gather (ParallelStrategy)
        results = [
            r if isinstance(r, dict) else {"success": False, "error": str(r)}
            for r in results
        ]

        # -- Phase: SCORING --
        self._set_phase(SwarmPhase.SCORING)
        # Scoring is caller responsibility (node0_activate applies SNR/Ihsan)

        # -- Phase: EQUALIZING --
        self._set_phase(SwarmPhase.EQUALIZING)
        eq_action: Optional[str] = None
        if self._config.equalizer_enabled and self._model_router:
            try:
                successful = sum(1 for r in results if r.get("success"))
                eq_action = await self._model_router.check_equalizer(
                    ihsan_score=successful / max(len(results), 1),
                    backlog=len(agents),
                    presence=0,
                )
                if eq_action:
                    logger.info("Equalizer: %s", eq_action)
                    self._emit(
                        SwarmEventKind.EQUALIZER_ACTION, data={"action": eq_action}
                    )
            except (asyncio.CancelledError, RuntimeError, OSError) as e:  # SEC-003 — async boundary
                logger.debug("Equalizer check skipped: %s", e)

        # -- Phase: COMPLETE --
        self._set_phase(SwarmPhase.COMPLETE)
        self._emit(
            SwarmEventKind.MISSION_COMPLETE,
            data={
                "results_count": len(results),
                "eq_action": eq_action,
            },
        )

        return {
            "results": results,
            "topology": self._config.topology.value,
            "eq_action": eq_action,
        }
