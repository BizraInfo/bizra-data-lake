"""
Apex Sovereign Entity — Unified Proactive System
═══════════════════════════════════════════════════════════════════════════════

The pinnacle of BIZRA's autonomous capabilities. Integrates all Apex pillars
(Social, Market, Swarm) into a unified Extended OODA loop with full
constitutional compliance.

Standing on the Shoulders of Giants:
- Boyd (1995): OODA decision cycle
- Shannon (1948): SNR information theory
- Granovetter (1973): Social network dynamics
- Lamport (1982): Distributed consensus
- Al-Ghazali (1058-1111): Muraqabah vigilance
- Anthropic: Constitutional AI principles

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                     APEX SOVEREIGN ENTITY                                │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────────────┐ │
    │  │ SociallyAware   │  │ MarketAware     │  │  HybridSwarm            │ │
    │  │ Bridge          │  │ Muraqabah       │  │  Orchestrator           │ │
    │  └────────┬────────┘  └────────┬────────┘  └───────────┬─────────────┘ │
    │           │                    │                       │               │
    │           └────────────────────┼───────────────────────┘               │
    │                                ▼                                       │
    │              ┌──────────────────────────────────┐                      │
    │              │     Extended OODA Loop           │                      │
    │              │  OBSERVE → PREDICT → COORDINATE  │                      │
    │              │  → ANALYZE → DECIDE → ACT        │                      │
    │              │  → LEARN → REFLECT               │                      │
    │              └──────────────────────────────────┘                      │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

Created: 2026-02-04 | BIZRA Apex Integration v1.0
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from statistics import mean
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from core.apex import ApexSystem
from core.sovereign.autonomy_matrix import AutonomyLevel
from core.sovereign.market_integration import (
    SNR_FLOOR,
    MarketAwareMuraqabah,
    MarketGoal,
    MarketSensorReading,
)
from core.sovereign.social_integration import ScoredAgent, SociallyAwareBridge
from core.sovereign.swarm_integration import HybridSwarmOrchestrator

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD

# Lazy imports for runtime_engines to avoid circular dependencies
if TYPE_CHECKING:
    from core.sovereign.runtime_engines import (
        GiantsRegistry,
        GoTBridge,
        SNRMaximizer,
    )

logger = logging.getLogger(__name__)


# OODA cycle interval
CYCLE_INTERVAL_MS: int = 1000
IHSAN_THRESHOLD: float = UNIFIED_IHSAN_THRESHOLD


class ApexOODAState(str, Enum):
    """Extended OODA states for Apex integration."""

    OBSERVE = "observe"  # Collect all sensor data
    PREDICT = "predict"  # Forecast trends (market)
    COORDINATE = "coordinate"  # Team planning (social)
    ANALYZE = "analyze"  # PAT analysis
    DECIDE = "decide"  # Autonomy-based decision
    ACT = "act"  # Execute via swarm
    LEARN = "learn"  # Update models
    REFLECT = "reflect"  # Metrics and improvement
    SLEEP = "sleep"  # Low-power monitoring


@dataclass
class Observation:
    """Combined observations from all sensors."""

    market_readings: List[MarketSensorReading] = field(default_factory=list)
    swarm_health: Dict[str, Any] = field(default_factory=dict)
    social_metrics: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class Prediction:
    """Predictions based on observations."""

    market_trends: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    workload_forecast: Optional[Dict[str, Any]] = None
    scaling_recommendation: Optional[str] = None
    confidence: float = 0.5
    # Enhanced fields for GoT integration
    got_result: Optional[Any] = None  # GoTResult when available
    reasoning_path: List[Any] = field(default_factory=list)  # ThoughtNodes


@dataclass
class TeamPlan:
    """Coordination plan for the team."""

    task_assignments: List[Dict[str, Any]] = field(default_factory=list)
    collaborations: List[Dict[str, Any]] = field(default_factory=list)
    selected_agents: List[ScoredAgent] = field(default_factory=list)


@dataclass
class Decision:
    """A decision with constitutional compliance."""

    goal: MarketGoal
    ihsan_score: float
    autonomy_level: AutonomyLevel
    requires_approval: bool
    approved: bool = False
    rejection_reason: Optional[str] = None


@dataclass
class Outcome:
    """Result of executing a decision."""

    decision: Decision
    success: bool
    value: float = 0.0
    agents_used: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    error: Optional[str] = None
    # Giants attribution for explainability
    giants_attribution: List[str] = field(default_factory=list)


class ApexSovereignEntity:
    """
    Proactive Sovereign Entity with full Apex integration.

    Combines:
    - Extended OODA loop (Boyd)
    - Social intelligence (Granovetter/PageRank)
    - Market intelligence (Shannon/Markowitz)
    - Swarm orchestration (Borg/K8s)
    - Constitutional AI (Ihsan)

    Usage:
        entity = ApexSovereignEntity(node_id="node-0")
        await entity.start()

        # Entity runs autonomously in background
        # Check status
        status = entity.status()

        # Stop gracefully
        await entity.stop()
    """

    def __init__(
        self,
        node_id: str,
        ihsan_threshold: float = IHSAN_THRESHOLD,
        snr_floor: float = SNR_FLOOR,
        cycle_interval_ms: int = CYCLE_INTERVAL_MS,
    ):
        """
        Initialize the Apex Sovereign Entity.

        Args:
            node_id: Unique identifier for this node
            ihsan_threshold: Constitutional constraint (default 0.95)
            snr_floor: Minimum signal quality (default 0.85)
            cycle_interval_ms: OODA cycle interval in milliseconds
        """
        self.node_id = node_id
        self.ihsan_threshold = ihsan_threshold
        self.snr_floor = snr_floor
        self.cycle_interval_ms = cycle_interval_ms

        # Initialize Apex system
        self.apex = ApexSystem(
            node_id=node_id,
            ihsan_threshold=ihsan_threshold,
            snr_floor=snr_floor,
        )

        # Initialize integrated components
        self.social_bridge = SociallyAwareBridge(
            node_id=node_id,
            ihsan_threshold=ihsan_threshold,
        )
        self.market_muraqabah = MarketAwareMuraqabah(
            node_id=node_id,
            snr_threshold=snr_floor,
            ihsan_threshold=ihsan_threshold,
        )
        self.swarm = HybridSwarmOrchestrator()

        # State
        self.current_state: ApexOODAState = ApexOODAState.SLEEP
        self.cycle_count: int = 0
        self._running: bool = False
        self._loop_task: Optional[asyncio.Task] = None

        # Metrics
        self.metrics: Dict[str, float] = {
            "cycles": 0,
            "actions_taken": 0,
            "autonomous_actions": 0,
            "ihsan_average": 0.0,
            "snr_average": 0.0,
            "success_rate": 0.0,
        }

        # History for averaging
        self._ihsan_history: List[float] = []
        self._snr_history: List[float] = []
        self._success_history: List[bool] = []

        # Lazy-initialized runtime engine components
        self._snr_maximizer: Optional["SNRMaximizer"] = None
        self._got_bridge: Optional["GoTBridge"] = None
        self._giants_registry: Optional["GiantsRegistry"] = None

        logger.info(f"ApexSovereignEntity initialized: {node_id}")

    # ═══════════════════════════════════════════════════════════════════════════
    # Runtime Engine Accessors (Lazy Initialization)
    # Standing on Giants: Shannon (SNR), Besta (GoT), Newton (Attribution)
    # ═══════════════════════════════════════════════════════════════════════════

    @property
    def snr_maximizer(self) -> "SNRMaximizer":
        """
        Get the SNR Maximizer (lazy initialized).

        Standing on Giants: Shannon (1948) - Information theory, SNR
        """
        if self._snr_maximizer is None:
            from core.sovereign.runtime_engines import SNRMaximizer

            self._snr_maximizer = SNRMaximizer(
                snr_floor=self.snr_floor,
                adaptive=True,
            )
            logger.debug("SNRMaximizer initialized (lazy)")
        return self._snr_maximizer

    @property
    def got_bridge(self) -> "GoTBridge":
        """
        Get the Graph-of-Thoughts Bridge (lazy initialized).

        Standing on Giants: Besta (2024) - Graph-of-Thoughts reasoning
        """
        if self._got_bridge is None:
            from core.sovereign.runtime_engines import GoTBridge

            self._got_bridge = GoTBridge(use_rust=True)
            logger.debug("GoTBridge initialized (lazy)")
        return self._got_bridge

    @property
    def giants_registry(self) -> "GiantsRegistry":
        """
        Get the Giants Registry (lazy initialized).

        Standing on Giants: Newton (1675) - Attribution philosophy
        """
        if self._giants_registry is None:
            from core.sovereign.runtime_engines import get_giants_registry

            self._giants_registry = get_giants_registry()
            logger.debug("GiantsRegistry initialized (lazy)")
        return self._giants_registry

    def get_giants_attribution(
        self, method_name: str = "ApexSovereignEntity"
    ) -> List[str]:
        """
        Get Giants attribution for a method or the entity itself.

        Returns list of attribution strings for decisions/outcomes.
        """
        from core.sovereign.runtime_engines import attribute

        return [
            attribute(["Claude Shannon"]),  # SNR filtering
            attribute(["Maciej Besta"]),  # GoT reasoning
            attribute(["John Boyd"]),  # OODA loop
            attribute(["Leslie Lamport"]),  # Distributed consensus
            attribute(["Abu Hamid Al-Ghazali"]),  # Muraqabah/Ihsan
            attribute(["Anthropic"]),  # Constitutional AI
        ]

    async def start(self) -> None:
        """Start the Apex Sovereign Entity."""
        if self._running:
            logger.warning("Entity already running")
            return

        logger.info(f"Starting ApexSovereignEntity: {self.node_id}")

        # Start subsystems
        await self.apex.start()
        await self.swarm.start()

        # Start OODA loop
        self._running = True
        self.current_state = ApexOODAState.OBSERVE
        self._loop_task = asyncio.create_task(self._run_ooda_loop())

        logger.info(
            f"ApexSovereignEntity started: mode=PROACTIVE, "
            f"ihsan_threshold={self.ihsan_threshold}, snr_floor={self.snr_floor}"
        )

    async def stop(self) -> None:
        """Stop the Apex Sovereign Entity gracefully."""
        if not self._running:
            return

        logger.info("Stopping ApexSovereignEntity...")

        self._running = False

        # Cancel OODA loop
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass

        # Stop subsystems
        await self.swarm.stop()
        await self.apex.stop()

        self.current_state = ApexOODAState.SLEEP
        logger.info("ApexSovereignEntity stopped")

    async def _run_ooda_loop(self) -> None:
        """Main OODA loop with Apex integration."""
        observation: Optional[Observation] = None
        prediction: Optional[Prediction] = None
        team_plan: Optional[TeamPlan] = None
        decisions: List[Decision] = []
        outcomes: List[Outcome] = []

        while self._running:
            try:
                self.cycle_count += 1

                # State machine
                if self.current_state == ApexOODAState.OBSERVE:
                    observation = await self._observe()
                    self.current_state = ApexOODAState.PREDICT

                elif self.current_state == ApexOODAState.PREDICT:
                    prediction = await self._predict(observation)
                    self.current_state = ApexOODAState.COORDINATE

                elif self.current_state == ApexOODAState.COORDINATE:
                    team_plan = await self._coordinate(observation, prediction)
                    self.current_state = ApexOODAState.ANALYZE

                elif self.current_state == ApexOODAState.ANALYZE:
                    goals = await self._analyze(observation, prediction, team_plan)
                    self.current_state = ApexOODAState.DECIDE

                elif self.current_state == ApexOODAState.DECIDE:
                    decisions = await self._decide(goals)
                    self.current_state = ApexOODAState.ACT

                elif self.current_state == ApexOODAState.ACT:
                    outcomes = await self._act(decisions, team_plan)
                    self.current_state = ApexOODAState.LEARN

                elif self.current_state == ApexOODAState.LEARN:
                    await self._learn(outcomes)
                    self.current_state = ApexOODAState.REFLECT

                elif self.current_state == ApexOODAState.REFLECT:
                    await self._reflect()
                    self.current_state = ApexOODAState.OBSERVE

                # Throttle loop
                await asyncio.sleep(self.cycle_interval_ms / 1000)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.error(f"OODA cycle error: {e}", exc_info=True)
                await self._handle_cycle_error(e)
                self.current_state = ApexOODAState.OBSERVE

    async def _observe(self) -> Observation:
        """
        OBSERVE phase: Collect all sensor readings.

        Sources:
        - Market sensors (signals, arbitrage)
        - Swarm health (Python agents, Rust services)
        - Social graph metrics

        Integration: Uses SNRMaximizer (Shannon 1948) for signal quality filtering.
        Low-SNR readings are filtered out before further processing.
        """
        observation = Observation()

        # Get market readings
        try:
            raw_readings = await self.market_muraqabah.scan_financial_domain()

            # Apply SNR Maximizer filtering (Shannon 1948)
            filtered_readings = []
            for reading in raw_readings:
                # Process through SNRMaximizer for enhanced signal analysis
                signal = self.snr_maximizer.process(
                    reading.snr_score,
                    source=f"market:{reading.symbol}",
                    channel="market_sensor",
                    metric_type="confidence",
                )

                # Only keep readings that pass SNR threshold
                if self.snr_maximizer.is_acceptable(signal):
                    filtered_readings.append(reading)
                else:
                    logger.debug(
                        f"Signal filtered by SNRMaximizer: {reading.symbol} "
                        f"SNR={signal.snr:.3f} < {self.snr_floor:.3f}"
                    )

            observation.market_readings = filtered_readings

            logger.debug(
                f"SNR Filter: {len(raw_readings)} raw -> {len(filtered_readings)} filtered "
                f"(pass rate: {len(filtered_readings)/max(1, len(raw_readings)):.1%})"
            )

        except Exception as e:
            logger.warning(f"Market scan failed: {e}")
            observation.market_readings = []

        # Get swarm health
        try:
            observation.swarm_health = await self.swarm.check_all_health()
        except Exception as e:
            logger.warning(f"Swarm health check failed: {e}")
            observation.swarm_health = {}

        # Get social metrics
        try:
            observation.social_metrics = self.social_bridge.get_network_metrics()
        except Exception as e:
            logger.warning(f"Social metrics failed: {e}")
            observation.social_metrics = {}

        logger.debug(
            f"Observed: {len(observation.market_readings)} market readings, "
            f"{len(observation.swarm_health)} health checks"
        )

        return observation

    async def _predict(self, observation: Optional[Observation]) -> Prediction:
        """
        PREDICT phase: Forecast trends from observations.

        Uses market analysis for trend prediction.

        Integration: Uses GoTBridge (Besta 2024) for multi-path reasoning
        when complex predictions are needed. Graph-of-Thoughts enables
        exploration of multiple prediction scenarios.
        """
        prediction = Prediction()

        if not observation:
            return prediction

        # Market trend predictions
        for reading in observation.market_readings:
            if reading.snr_score >= self.snr_floor:
                prediction.market_trends[reading.symbol] = {
                    "type": reading.sensor_type.value,
                    "snr": reading.snr_score,
                    "value": reading.value,
                }

        # Workload forecast from swarm health
        if observation.swarm_health:
            healthy = sum(
                1
                for h in observation.swarm_health.values()
                if str(h) == "HealthStatus.HEALTHY"
            )
            total = len(observation.swarm_health)

            if total > 0 and healthy / total < 0.8:
                prediction.scaling_recommendation = "scale_up"
            elif total > 0 and healthy / total > 0.95:
                prediction.workload_forecast = {"status": "optimal"}

        # Enhanced reasoning with Graph-of-Thoughts (Besta 2024)
        # Only trigger GoT for complex predictions with multiple high-SNR signals
        high_snr_count = sum(
            1 for r in observation.market_readings if r.snr_score >= 0.90
        )

        if high_snr_count >= 2:
            try:
                # Construct reasoning goal from observations
                reasoning_goal = (
                    f"Analyze {len(observation.market_readings)} market signals "
                    f"with {high_snr_count} high-confidence readings to predict "
                    f"optimal action strategy"
                )

                # Use GoT for multi-path reasoning
                got_result = await self.got_bridge.reason(
                    goal=reasoning_goal,
                    max_iterations=30,  # Bounded for performance
                )

                if got_result.success and got_result.solution:
                    prediction.got_result = got_result
                    prediction.reasoning_path = got_result.best_path
                    # Boost confidence based on GoT exploration depth
                    depth_bonus = min(0.2, got_result.max_depth_reached * 0.03)
                    prediction.confidence = min(
                        1.0, prediction.confidence + depth_bonus
                    )

                    logger.debug(
                        f"GoT enhanced prediction: explored={got_result.explored_nodes}, "
                        f"depth={got_result.max_depth_reached}, time={got_result.execution_time_ms:.1f}ms"
                    )

            except Exception as e:
                logger.warning(f"GoT reasoning fallback: {e}")
                # Continue with standard prediction if GoT fails

        prediction.confidence = 0.7 if observation.market_readings else 0.3

        return prediction

    async def _coordinate(
        self,
        observation: Optional[Observation],
        prediction: Optional[Prediction],
    ) -> TeamPlan:
        """
        COORDINATE phase: Plan team activities using social intelligence.

        Uses SocialGraph for trust-based task routing.
        """
        team_plan = TeamPlan()

        if not observation:
            return team_plan

        # Find collaboration opportunities
        try:
            task_capabilities = {"reasoning", "execution"}
            collaborations = self.social_bridge.find_collaboration_partners(
                task_capabilities=task_capabilities,
                min_synergy=0.6,
            )

            for collab in collaborations[:5]:  # Top 5
                team_plan.collaborations.append(
                    {
                        "agents": [collab.agent_a, collab.agent_b],
                        "synergy": collab.synergy_score,
                        "tasks": list(collab.recommended_task_types),
                    }
                )

        except Exception as e:
            logger.warning(f"Collaboration discovery failed: {e}")

        return team_plan

    async def _analyze(
        self,
        observation: Optional[Observation],
        prediction: Optional[Prediction],
        team_plan: Optional[TeamPlan],
    ) -> List[MarketGoal]:
        """
        ANALYZE phase: Process observations into goals.

        Converts market readings into actionable goals.
        """
        goals: List[MarketGoal] = []

        if not observation:
            return goals

        # Process market readings into goals
        for reading in observation.market_readings:
            goal = self.market_muraqabah.process_market_reading(reading)
            if goal:
                goals.append(goal)

                # Track SNR
                self._snr_history.append(reading.snr_score)
                if len(self._snr_history) > 100:
                    self._snr_history.pop(0)

        logger.debug(
            f"Analyzed: {len(goals)} goals from {len(observation.market_readings)} readings"
        )
        return goals

    async def _decide(self, goals: List[MarketGoal]) -> List[Decision]:
        """
        DECIDE phase: Make decisions based on goals and autonomy levels.

        Constitutional compliance:
        - All decisions validated against Ihsan
        - Autonomy level determines approval requirement
        """
        decisions: List[Decision] = []

        for goal in goals:
            # Validate Ihsan
            ihsan_score = goal.ihsan_score

            if ihsan_score < self.ihsan_threshold:
                logger.info(
                    f"Goal filtered by Ihsan: {goal.goal_id}, score={ihsan_score:.3f}"
                )
                continue

            # Track Ihsan
            self._ihsan_history.append(ihsan_score)
            if len(self._ihsan_history) > 100:
                self._ihsan_history.pop(0)

            # Determine approval requirement
            requires_approval = goal.autonomy_level <= AutonomyLevel.SUGGESTER

            decision = Decision(
                goal=goal,
                ihsan_score=ihsan_score,
                autonomy_level=goal.autonomy_level,
                requires_approval=requires_approval,
                approved=not requires_approval,  # Auto-approve high autonomy
            )
            decisions.append(decision)

        logger.debug(
            f"Decided: {len(decisions)} decisions, {sum(1 for d in decisions if d.approved)} auto-approved"
        )
        return decisions

    async def _act(
        self,
        decisions: List[Decision],
        team_plan: Optional[TeamPlan],
    ) -> List[Outcome]:
        """
        ACT phase: Execute approved decisions via hybrid swarm.

        Uses SociallyAwareBridge for agent selection.

        Integration: Adds Giants attribution to every outcome for
        explainability and knowledge provenance tracking.
        """
        outcomes: List[Outcome] = []

        # Get Giants attribution once for all outcomes
        giants_attribution = self.get_giants_attribution("_act")

        for decision in decisions:
            if not decision.approved:
                continue

            start_time = time.time()

            try:
                # Select agent using social trust
                selected = self.social_bridge.select_agent_for_task(
                    required_capabilities={"reasoning", "execution"},
                    prefer_diversity=True,
                )

                # Simulate execution (would call actual agent in production)
                success = (
                    decision.ihsan_score >= 0.95 and decision.goal.snr_score >= 0.85
                )
                value = decision.goal.estimated_value if success else 0.0

                execution_time = (time.time() - start_time) * 1000

                outcome = Outcome(
                    decision=decision,
                    success=success,
                    value=value,
                    agents_used=[selected.agent_id],
                    execution_time_ms=execution_time,
                    giants_attribution=giants_attribution,  # Add attribution
                )
                outcomes.append(outcome)

                # Track success
                self._success_history.append(success)
                if len(self._success_history) > 100:
                    self._success_history.pop(0)

                # Update metrics
                self.metrics["actions_taken"] += 1
                if decision.autonomy_level >= AutonomyLevel.AUTOLOW:
                    self.metrics["autonomous_actions"] += 1

            except Exception as e:
                logger.error(f"Action execution failed: {e}")
                outcome = Outcome(
                    decision=decision,
                    success=False,
                    error=str(e),
                    execution_time_ms=(time.time() - start_time) * 1000,
                    giants_attribution=giants_attribution,  # Add attribution even on failure
                )
                outcomes.append(outcome)
                self._success_history.append(False)

        return outcomes

    async def _learn(self, outcomes: List[Outcome]) -> None:
        """
        LEARN phase: Update models based on outcomes.

        Updates social trust scores based on success/failure.
        """
        for outcome in outcomes:
            # Update social trust for agents used
            for agent_id in outcome.agents_used:
                await self.social_bridge.report_task_outcome(
                    agent_id=agent_id,
                    task_id=outcome.decision.goal.goal_id,
                    success=outcome.success,
                    value=outcome.value,
                )

        logger.debug(f"Learned from {len(outcomes)} outcomes")

    async def _reflect(self) -> None:
        """
        REFLECT phase: Evaluate performance and adjust parameters.

        Updates metrics and checks thresholds.
        """
        self.metrics["cycles"] = self.cycle_count

        # Update averages
        if self._ihsan_history:
            self.metrics["ihsan_average"] = mean(self._ihsan_history)

        if self._snr_history:
            self.metrics["snr_average"] = mean(self._snr_history)

        if self._success_history:
            self.metrics["success_rate"] = sum(self._success_history) / len(
                self._success_history
            )

        # Log summary every 10 cycles
        if self.cycle_count % 10 == 0:
            logger.info(
                f"Cycle {self.cycle_count} | "
                f"Actions: {self.metrics['actions_taken']} | "
                f"Autonomous: {self.metrics['autonomous_actions']} | "
                f"Ihsan: {self.metrics['ihsan_average']:.3f} | "
                f"SNR: {self.metrics['snr_average']:.3f} | "
                f"Success: {self.metrics['success_rate']:.2%}"
            )

        # Adjust parameters if Ihsan dropping
        if self.metrics["ihsan_average"] < self.ihsan_threshold - 0.05:
            logger.warning("Ihsan average dropping, tightening SNR floor")
            self.snr_floor = min(self.snr_floor + 0.02, 0.95)

    async def _handle_cycle_error(self, error: Exception) -> None:
        """Handle errors in the OODA cycle."""
        logger.error(f"Handling cycle error: {error}")
        # Reset to stable state
        await asyncio.sleep(1)  # Brief pause before retry

    def status(self) -> Dict[str, Any]:
        """Get entity status including runtime_engines components."""
        status_dict = {
            "node_id": self.node_id,
            "running": self._running,
            "current_state": self.current_state.value,
            "cycle_count": self.cycle_count,
            "ihsan_threshold": self.ihsan_threshold,
            "snr_floor": self.snr_floor,
            "metrics": self.metrics,
            "subsystems": {
                "apex": self.apex.status(),
                "social": self.social_bridge.get_network_metrics(),
                "swarm": self.swarm.get_metrics(),
            },
        }

        # Add runtime_engines status (only if initialized)
        runtime_engines_status = {}
        if self._snr_maximizer is not None:
            runtime_engines_status["snr_maximizer"] = (
                self._snr_maximizer.get_statistics()
            )
        if self._giants_registry is not None:
            runtime_engines_status["giants_registry"] = self._giants_registry.summary()

        if runtime_engines_status:
            status_dict["runtime_engines"] = runtime_engines_status

        # Add Giants attribution
        status_dict["standing_on_giants"] = [
            "Shannon (1948): SNR Maximization",
            "Besta (2024): Graph-of-Thoughts",
            "Boyd (1995): OODA Loop",
            "Lamport (1982): Distributed Consensus",
            "Al-Ghazali (1095): Muraqabah/Ihsan",
            "Anthropic (2022): Constitutional AI",
        ]

        return status_dict


# Convenience function
def create_apex_entity(node_id: str = "node-0") -> ApexSovereignEntity:
    """Create and return an ApexSovereignEntity instance."""
    return ApexSovereignEntity(node_id=node_id)
