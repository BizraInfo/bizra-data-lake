"""
SpearpointOrchestrator — Thin Mission Router
=============================================

Entry point for spearpoint operations. Routes missions to the
appropriate handler (AutoEvaluator or AutoResearcher) and collects
results for the evidence chain.

Standing on Giants: Boyd (OODA) + Goldratt (Theory of Constraints)
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional, Union

from core.autopoiesis.hypothesis_generator import SystemObservation

from .auto_evaluator import AutoEvaluator
from .auto_researcher import AutoResearcher
from .config import MissionType, SpearpointConfig
from .recursive_loop import LoopMetrics, RecursiveLoop

logger = logging.getLogger(__name__)


@dataclass
class SpearpointMission:
    """A mission for the Spearpoint orchestrator."""

    mission_id: str = field(default_factory=lambda: f"mission_{uuid.uuid4().hex[:12]}")
    mission_type: MissionType = MissionType.REPRODUCE
    input_data: dict[str, Any] = field(default_factory=dict)
    config: Optional[SpearpointConfig] = None
    created_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "mission_type": self.mission_type.value,
            "input_data": self.input_data,
            "created_at": self.created_at,
        }


@dataclass
class MissionResult:
    """Result of a completed mission."""

    mission_id: str
    mission_type: MissionType
    success: bool
    evaluation_results: list[dict[str, Any]] = field(default_factory=list)
    research_results: list[dict[str, Any]] = field(default_factory=list)
    elapsed_ms: float = 0.0
    error: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    def to_dict(self) -> dict[str, Any]:
        return {
            "mission_id": self.mission_id,
            "mission_type": self.mission_type.value,
            "success": self.success,
            "evaluation_count": len(self.evaluation_results),
            "research_count": len(self.research_results),
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
            "timestamp": self.timestamp,
        }


class SpearpointOrchestrator:
    """
    Thin mission router — entry point for spearpoint operations.

    Routes:
      REPRODUCE -> AutoEvaluator (spearpoint.reproduce)
      IMPROVE   -> AutoResearcher (spearpoint.improve)

    Usage:
        config = SpearpointConfig()
        orchestrator = SpearpointOrchestrator(config)

        # Reproduce: verify a claim
        mission = SpearpointMission(
            mission_type=MissionType.REPRODUCE,
            input_data={"claim": "System latency < 100ms"},
        )
        result = orchestrator.execute_mission(mission)

        # Improve: generate and evaluate hypotheses
        mission = SpearpointMission(
            mission_type=MissionType.IMPROVE,
            input_data={"observation": {...}},
        )
        result = orchestrator.execute_mission(mission)
    """

    def __init__(
        self,
        config: Optional[SpearpointConfig] = None,
        evaluator: Optional[AutoEvaluator] = None,
        researcher: Optional[AutoResearcher] = None,
    ):
        self.config = config or SpearpointConfig()
        self.config.ensure_dirs()

        # Wire components
        self._evaluator = evaluator or AutoEvaluator(config=self.config)
        self._researcher = researcher or AutoResearcher(
            evaluator=self._evaluator,
            config=self.config,
        )

        # Mission history
        self._completed_missions: list[MissionResult] = []
        self._loop: Optional[RecursiveLoop] = None

        logger.info("SpearpointOrchestrator initialized")

    def execute_mission(self, mission: SpearpointMission) -> MissionResult:
        """
        Execute a mission by routing to the appropriate handler.

        Args:
            mission: The mission to execute

        Returns:
            MissionResult with all evaluation/research results
        """
        start = time.perf_counter()

        try:
            if mission.mission_type == MissionType.REPRODUCE:
                return self._execute_reproduce(mission, start)
            elif mission.mission_type == MissionType.IMPROVE:
                return self._execute_improve(mission, start)
            else:
                return MissionResult(
                    mission_id=mission.mission_id,
                    mission_type=mission.mission_type,
                    success=False,
                    elapsed_ms=(time.perf_counter() - start) * 1000,
                    error=f"Unknown mission type: {mission.mission_type}",
                )
        except Exception as e:
            logger.error(f"Mission {mission.mission_id} failed: {e}")
            return MissionResult(
                mission_id=mission.mission_id,
                mission_type=mission.mission_type,
                success=False,
                elapsed_ms=(time.perf_counter() - start) * 1000,
                error="Mission execution failed",
            )
        finally:
            # Cap mission history to prevent unbounded growth
            if len(self._completed_missions) > 100:
                self._completed_missions = self._completed_missions[-100:]

    def _execute_reproduce(
        self,
        mission: SpearpointMission,
        start: float,
    ) -> MissionResult:
        """Execute a REPRODUCE mission via AutoEvaluator."""
        claim = mission.input_data.get("claim", "")
        proposed_change = mission.input_data.get("proposed_change", "")
        prompt = mission.input_data.get("prompt", "")
        response = mission.input_data.get("response", "")
        metrics = mission.input_data.get("metrics", {})

        if not claim:
            return MissionResult(
                mission_id=mission.mission_id,
                mission_type=mission.mission_type,
                success=False,
                elapsed_ms=(time.perf_counter() - start) * 1000,
                error="REPRODUCE mission requires 'claim' in input_data",
            )

        result = self._evaluator.evaluate(
            claim=claim,
            proposed_change=proposed_change,
            mission_id=mission.mission_id,
            prompt=prompt,
            response=response,
            metrics=metrics,
        )

        mission_result = MissionResult(
            mission_id=mission.mission_id,
            mission_type=mission.mission_type,
            success=result.verdict.value != "REJECTED",
            evaluation_results=[result.to_dict()],
            elapsed_ms=(time.perf_counter() - start) * 1000,
        )

        self._completed_missions.append(mission_result)
        return mission_result

    def _execute_improve(
        self,
        mission: SpearpointMission,
        start: float,
    ) -> MissionResult:
        """Execute an IMPROVE mission via AutoResearcher."""
        obs_data = mission.input_data.get("observation")
        top_k = mission.input_data.get("top_k", 3)

        observation = None
        if obs_data and isinstance(obs_data, dict):
            observation = SystemObservation.from_dict(obs_data)

        results = self._researcher.research(
            observation=observation,
            mission_id=mission.mission_id,
            top_k=top_k,
        )

        any_approved = any(r.outcome.value == "approved" for r in results)

        mission_result = MissionResult(
            mission_id=mission.mission_id,
            mission_type=mission.mission_type,
            success=any_approved,
            research_results=[r.to_dict() for r in results],
            elapsed_ms=(time.perf_counter() - start) * 1000,
        )

        self._completed_missions.append(mission_result)
        return mission_result

    def reproduce(
        self,
        *,
        claim: str,
        proposed_change: str = "",
        prompt: str = "",
        response: str = "",
        metrics: Optional[dict[str, Any]] = None,
        mission_id: Optional[str] = None,
    ) -> MissionResult:
        """Explicit spearpoint.reproduce handler (evaluation-first path)."""
        mission = SpearpointMission(
            mission_id=mission_id or f"mission_{uuid.uuid4().hex[:12]}",
            mission_type=MissionType.REPRODUCE,
            input_data={
                "claim": claim,
                "proposed_change": proposed_change,
                "prompt": prompt,
                "response": response,
                "metrics": metrics or {},
            },
        )
        return self.execute_mission(mission)

    def improve(
        self,
        *,
        observation: Optional[Union[SystemObservation, dict[str, Any]]] = None,
        top_k: int = 3,
        mission_id: Optional[str] = None,
    ) -> MissionResult:
        """Explicit spearpoint.improve handler (innovation through evaluator gate)."""
        observation_data: Optional[dict[str, Any]] = None
        if isinstance(observation, SystemObservation):
            observation_data = observation.to_dict()
        elif isinstance(observation, dict):
            observation_data = observation

        mission = SpearpointMission(
            mission_id=mission_id or f"mission_{uuid.uuid4().hex[:12]}",
            mission_type=MissionType.IMPROVE,
            input_data={
                "observation": observation_data,
                "top_k": top_k,
            },
        )
        return self.execute_mission(mission)

    def research_pattern(
        self,
        *,
        pattern_id: str,
        claim_context: str = "",
        top_k: int = 3,
        mission_id: Optional[str] = None,
    ) -> MissionResult:
        """Pattern-aware research using Sci-Reasoning thinking patterns.

        Uses the 15 cognitive moves from Li et al. (2025) to seed
        hypothesis generation with proven innovation strategies.

        Standing on: Li et al. (Sci-Reasoning), Boyd (OODA Orient phase)
        """
        import time

        mid = mission_id or f"mission_{uuid.uuid4().hex[:12]}"
        start = time.perf_counter()

        try:
            results = self._researcher.research_with_pattern(
                pattern_id=pattern_id,
                claim_context=claim_context,
                mission_id=mid,
                top_k=top_k,
            )

            any_approved = any(r.outcome.value == "approved" for r in results)

            mission_result = MissionResult(
                mission_id=mid,
                mission_type=MissionType.IMPROVE,
                success=any_approved,
                research_results=[r.to_dict() for r in results],
                elapsed_ms=(time.perf_counter() - start) * 1000,
            )
        except Exception as e:
            logger.error(f"Pattern research mission {mid} failed: {e}")
            mission_result = MissionResult(
                mission_id=mid,
                mission_type=MissionType.IMPROVE,
                success=False,
                elapsed_ms=(time.perf_counter() - start) * 1000,
                error="Pattern research failed",
            )

        self._completed_missions.append(mission_result)
        return mission_result

    async def run_heartbeat(
        self,
        *,
        max_cycles: Optional[int] = None,
        observation_fn: Optional[Any] = None,
    ) -> LoopMetrics:
        """Run the recursive heartbeat loop with centralized breaker logic."""
        if self._loop is None:
            self._loop = RecursiveLoop(
                evaluator=self._evaluator,
                researcher=self._researcher,
                config=self.config,
            )
        return await self._loop.run(
            max_cycles=max_cycles,
            observation_fn=observation_fn,
        )

    def stop_heartbeat(self) -> None:
        """Request graceful stop for the active heartbeat loop."""
        if self._loop is not None:
            self._loop.request_stop()

    @property
    def evaluator(self) -> AutoEvaluator:
        """Access the evaluator directly."""
        return self._evaluator

    @property
    def researcher(self) -> AutoResearcher:
        """Access the researcher directly."""
        return self._researcher

    def get_statistics(self) -> dict[str, Any]:
        """Get orchestrator statistics."""
        return {
            "total_missions": len(self._completed_missions),
            "successful_missions": sum(
                1 for m in self._completed_missions if m.success
            ),
            "evaluator": self._evaluator.get_statistics(),
            "researcher": self._researcher.get_statistics(),
            "heartbeat": (
                self._loop.get_metrics().to_dict() if self._loop is not None else None
            ),
        }

    def get_mission_history(self, limit: int = 10) -> list[dict[str, Any]]:
        """Get recent mission history."""
        return [m.to_dict() for m in self._completed_missions[-limit:]]


__all__ = [
    "SpearpointOrchestrator",
    "SpearpointMission",
    "MissionResult",
]
