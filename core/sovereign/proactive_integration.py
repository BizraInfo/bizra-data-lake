"""
Proactive Sovereign Entity — Unified System Integration
=======================================================
The complete Proactive Sovereign Entity that integrates:
- Node0 Extended OODA Loop
- Dual-Agentic Team (PAT + SAT)
- Proactive Engine (Muraqabah + Autonomy Matrix)
- Constitutional Framework (Ihsan 8D)

"AI that works 24/7, anticipating needs and creating value."

Standing on Giants: Al-Ghazali + John Boyd + Lamport + Anthropic + Malone
"""

import asyncio
import logging
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Dict, List, Optional

from .autonomy import (
    AutonomousLoop,
    DecisionGate,
    SystemMetrics,
)
from .autonomy_matrix import ActionContext, AutonomyLevel, AutonomyMatrix
from .collective_intelligence import (
    CollectiveIntelligence,
)
from .dual_agentic_bridge import DualAgenticBridge
from .enhanced_team_planner import EnhancedTeamPlanner, ProactiveGoal
from .event_bus import Event, EventPriority, get_event_bus
from .muraqabah_engine import MonitorDomain, MuraqabahEngine, Opportunity
from .predictive_monitor import PredictiveMonitor
from .proactive_scheduler import ProactiveScheduler
from .proactive_team import ProactiveTeam
from .state_checkpointer import StateCheckpointer
from .swarm_knowledge_bridge import SwarmKnowledgeBridge, create_swarm_knowledge_bridge
from .team_planner import AgentRole

logger = logging.getLogger(__name__)


class EntityMode(str, Enum):
    """Operating modes of the Proactive Sovereign Entity."""

    REACTIVE = "reactive"  # Traditional request-response
    PROACTIVE_SUGGEST = "proactive_suggest"  # Detect & suggest, require approval
    PROACTIVE_AUTO = "proactive_auto"  # Auto-execute within constraints
    PROACTIVE_PARTNER = "proactive_partner"  # Full proactive partner mode


@dataclass
class EntityConfig:
    """Configuration for the Proactive Sovereign Entity."""

    mode: EntityMode = EntityMode.PROACTIVE_PARTNER
    ihsan_threshold: float = 0.95
    default_autonomy: AutonomyLevel = AutonomyLevel.AUTOLOW
    cycle_interval: float = 5.0
    checkpoint_interval: float = 300.0
    enable_muraqabah: bool = True
    enable_predictions: bool = True
    enable_collective: bool = True
    enable_knowledge_integration: bool = True
    max_concurrent_goals: int = 5


@dataclass
class EntityCycleResult:
    """Result of one entity operation cycle."""

    cycle_number: int = 0
    mode: EntityMode = EntityMode.REACTIVE
    # OODA phases
    observations: int = 0
    predictions: Dict[str, Any] = field(default_factory=dict)
    coordination: Dict[str, Any] = field(default_factory=dict)
    decisions_made: int = 0
    actions_executed: int = 0
    learning: Dict[str, Any] = field(default_factory=dict)
    # Proactive metrics
    opportunities_detected: int = 0
    goals_created: int = 0
    goals_executed: int = 0
    # Team metrics
    consensus_votes: int = 0
    consensus_approved: int = 0
    collective_synergy: float = 0.0
    # Quality
    ihsan_score: float = 0.0
    snr_score: float = 0.0
    health_score: float = 0.0
    # Timing
    duration_ms: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class ProactiveSovereignEntity:
    """
    The Proactive Sovereign Entity - an AI that works 24/7 for users.

    Architecture:
    ┌─────────────────────────────────────────────────────┐
    │           PROACTIVE SOVEREIGN ENTITY                │
    ├─────────────────────────────────────────────────────┤
    │  Extended OODA: OBSERVE → PREDICT → COORDINATE →    │
    │                 ANALYZE → DECIDE → ACT → LEARN      │
    ├─────────────────────────────────────────────────────┤
    │  Muraqabah Engine  │  Autonomy Matrix              │
    │  (24/7 Monitoring) │  (5-Level Control)            │
    ├─────────────────────────────────────────────────────┤
    │  Enhanced Planner  │  Dual-Agentic Bridge          │
    │  (Proactive Goals) │  (PAT + SAT Consensus)        │
    ├─────────────────────────────────────────────────────┤
    │  Collective Intelligence  │  Proactive Scheduler   │
    │  (Team Synergy)          │  (Anticipatory Jobs)    │
    └─────────────────────────────────────────────────────┘
    """

    def __init__(self, config: Optional[EntityConfig] = None):
        self.config = config or EntityConfig()

        # Event bus for component communication
        self.event_bus = get_event_bus()

        # Core OODA loop (extended)
        self.ooda_loop = AutonomousLoop(
            decision_gate=DecisionGate(ihsan_threshold=self.config.ihsan_threshold),
            ihsan_threshold=self.config.ihsan_threshold,
            cycle_interval=self.config.cycle_interval,
        )

        # State persistence
        self.checkpointer = StateCheckpointer(
            auto_interval_seconds=self.config.checkpoint_interval,
        )

        # Autonomy framework
        self.autonomy = AutonomyMatrix(
            default_level=self.config.default_autonomy,
            ihsan_threshold=self.config.ihsan_threshold,
        )

        # Muraqabah monitoring
        self.muraqabah = (
            MuraqabahEngine(
                ihsan_threshold=self.config.ihsan_threshold,
                event_bus=self.event_bus,
            )
            if self.config.enable_muraqabah
            else None
        )

        # Dual-Agentic bridge
        self.bridge = DualAgenticBridge(
            ihsan_threshold=self.config.ihsan_threshold,
        )

        # Collective intelligence
        self.collective = (
            CollectiveIntelligence() if self.config.enable_collective else None
        )

        # Proactive planner
        self.planner = EnhancedTeamPlanner(
            ihsan_threshold=self.config.ihsan_threshold,
            autonomy_matrix=self.autonomy,
            bridge=self.bridge,
            muraqabah=self.muraqabah,
            event_bus=self.event_bus,
        )

        # Predictive monitor
        self.monitor = PredictiveMonitor() if self.config.enable_predictions else None

        # Proactive scheduler
        self.scheduler = ProactiveScheduler()

        # Proactive team coordinator
        self.team = ProactiveTeam(
            ihsan_threshold=self.config.ihsan_threshold,
            event_bus=self.event_bus,
        )

        # Knowledge integration (BIZRA Data Lake + MoMo R&D)
        self.knowledge_bridge: Optional[SwarmKnowledgeBridge] = None
        self._knowledge_initialized = False

        # State
        self._running = False
        self._cycle_count = 0
        self._active_goals: List[ProactiveGoal] = []
        # PERF FIX: Use deque with maxlen for O(1) bounded storage
        self._cycle_results: Deque[EntityCycleResult] = deque(maxlen=1000)

        # Register OODA extensions
        self._register_ooda_extensions()

        # Subscribe to events
        self._setup_event_handlers()

    def _register_ooda_extensions(self) -> None:
        """Register extended OODA phase handlers."""

        # Predictor: Use predictive monitor
        async def predictor(metrics: SystemMetrics, predictions: Dict) -> Dict:
            if self.monitor:
                self.monitor.record("snr_score", metrics.snr_score)
                self.monitor.record("ihsan_score", metrics.ihsan_score)
                self.monitor.record("error_rate", metrics.error_rate)
                self.monitor.record("latency_ms", metrics.latency_ms)

                analyses = self.monitor.analyze_all()
                alerts = self.monitor.check_alerts()

                return {
                    "analyses": {k: v.direction.name for k, v in analyses.items()},
                    "alerts": len(alerts),
                }
            return {}

        # Coordinator: Use autonomy matrix and planner
        async def coordinator(metrics, predictions, coordination) -> Dict:
            # Get ready goals
            ready_goals = self.planner.get_ready_tasks()
            return {
                "ready_tasks": len(ready_goals),
                "active_goals": len(self._active_goals),
            }

        # Learner: Update autonomy thresholds based on outcomes
        async def learner(outcomes, reflection, learning) -> Dict:
            success_rate = reflection.get("success_rate", 1.0)
            if success_rate < 0.7:
                learning["strategy_updates"].append(
                    {
                        "type": "autonomy_adjustment",
                        "action": "reduce autonomy level for risky actions",
                    }
                )
            return {}

        self.ooda_loop.register_predictor(predictor)
        self.ooda_loop.register_coordinator(coordinator)
        self.ooda_loop.register_learner(learner)

    def _setup_event_handlers(self) -> None:
        """Set up event subscriptions."""

        async def handle_opportunity(event: Event):
            """Handle opportunity from Muraqabah."""
            if self.config.mode in (
                EntityMode.PROACTIVE_AUTO,
                EntityMode.PROACTIVE_PARTNER,
            ):
                opp = Opportunity(
                    domain=MonitorDomain(event.payload.get("domain", "environmental")),
                    description=event.payload.get("description", ""),
                    estimated_value=event.payload.get("value", 0.5),
                    urgency=event.payload.get("urgency", 0.5),
                )
                goal = await self.planner.handle_opportunity(opp)
                if goal:
                    self._active_goals.append(goal)
                    await self._maybe_execute_goal(goal)

        async def handle_alert(event: Event):
            """Handle predictive alert."""
            severity = event.payload.get("severity", "INFO")
            if severity in ("WARNING", "CRITICAL"):
                logger.warning(f"Predictive alert: {event.payload.get('message')}")

        self.event_bus.subscribe("muraqabah.opportunity.*", handle_opportunity)
        self.event_bus.subscribe("proactive.alert.*", handle_alert)

    async def _maybe_execute_goal(self, goal: ProactiveGoal) -> None:
        """Execute goal if autonomy allows."""
        if len(self._active_goals) > self.config.max_concurrent_goals:
            self.planner.queue_goal(goal)
            return

        context = ActionContext(
            action_type=f"proactive_goal_{goal.domain.value}",
            description=goal.description,
            risk_score=1.0 - goal.constitutional_score,
            ihsan_score=goal.constitutional_score,
            is_emergency=goal.urgency > 0.9,
        )

        decision = self.autonomy.determine_autonomy(context)

        if decision.can_execute:
            asyncio.create_task(self._execute_goal_with_tracking(goal))
        elif self.config.mode == EntityMode.PROACTIVE_SUGGEST:
            await self.event_bus.emit(
                topic="proactive.suggestion",
                payload={
                    "goal_id": goal.id,
                    "description": goal.description,
                    "urgency": goal.urgency,
                    "requires_approval": True,
                },
                priority=EventPriority.HIGH,
            )

    async def _execute_goal_with_tracking(self, goal: ProactiveGoal) -> None:
        """Execute a goal with proper tracking."""
        try:
            result = await self.planner.execute_autonomously(goal)
            logger.info(
                f"Goal {goal.id} execution: "
                f"{'SUCCESS' if result.success else 'FAILED'} "
                f"(tasks: {result.tasks_completed}/{result.tasks_completed + result.tasks_failed})"
            )
        finally:
            if goal in self._active_goals:
                self._active_goals.remove(goal)

    async def run_cycle(self) -> EntityCycleResult:
        """Execute one complete entity operation cycle."""
        self._cycle_count += 1
        start_time = datetime.now(timezone.utc)

        result = EntityCycleResult(
            cycle_number=self._cycle_count,
            mode=self.config.mode,
        )

        try:
            # 1. Run extended OODA cycle
            ooda_result = await self.ooda_loop.run_cycle(extended=True)
            result.observations = len(self.ooda_loop.observations)
            result.predictions = ooda_result.get("predictions", {})
            result.coordination = ooda_result.get("coordination", {})
            result.decisions_made = ooda_result.get("approved", 0)
            result.actions_executed = ooda_result.get("executed", 0)
            result.learning = ooda_result.get("learning", {})
            result.health_score = ooda_result.get("health", 0)

            # 2. Run Muraqabah scan (if enabled)
            if self.muraqabah and self.config.mode != EntityMode.REACTIVE:
                scan_result = await self.muraqabah.scan()
                result.opportunities_detected = scan_result.get("opportunities", 0)

            # 3. Process goal queue
            while len(self._active_goals) < self.config.max_concurrent_goals:
                next_goal = self.planner.get_next_goal()
                if not next_goal:
                    break
                self._active_goals.append(next_goal)
                await self._maybe_execute_goal(next_goal)
                result.goals_created += 1

            result.goals_executed = len(self._active_goals)

            # 4. Get bridge stats
            bridge_stats = self.bridge.stats()
            result.consensus_votes = bridge_stats.get("total_proposals", 0)
            result.consensus_approved = bridge_stats.get("approved", 0)

            # 5. Get collective intelligence stats
            if self.collective:
                ci_stats = self.collective.stats()
                result.collective_synergy = ci_stats.get("average_synergy", 0)

            # 6. Calculate quality scores
            if self.ooda_loop.observations:
                latest = self.ooda_loop.observations[-1]
                result.ihsan_score = latest.ihsan_score
                result.snr_score = latest.snr_score

        except Exception as e:
            logger.error(f"Entity cycle error: {e}")

        finally:
            result.duration_ms = (
                datetime.now(timezone.utc) - start_time
            ).total_seconds() * 1000
            # PERF FIX: deque with maxlen auto-discards oldest (O(1))
            self._cycle_results.append(result)

        return result

    async def start(self) -> None:
        """Start the Proactive Sovereign Entity."""
        self._running = True

        # Initialize knowledge integration (BIZRA Data Lake + MoMo R&D)
        if self.config.enable_knowledge_integration and not self._knowledge_initialized:
            try:
                self.knowledge_bridge = await create_swarm_knowledge_bridge(
                    ihsan_threshold=self.config.ihsan_threshold
                )
                self._knowledge_initialized = True
                knowledge_status = "BIZRA Data Lake + MoMo R&D connected"
            except Exception as e:
                logger.warning(f"Knowledge integration unavailable: {e}")
                knowledge_status = "Unavailable (standalone mode)"
        else:
            knowledge_status = (
                "Disabled"
                if not self.config.enable_knowledge_integration
                else "Already initialized"
            )

        logger.info(f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║              PROACTIVE SOVEREIGN ENTITY INITIALIZED                          ║
╠══════════════════════════════════════════════════════════════════════════════╣
║  Mode: {self.config.mode.value:<20}                                         ║
║  Autonomy: {self.config.default_autonomy.name:<15} (5-level matrix active)  ║
║  Monitoring: {'24/7 Muraqabah active' if self.muraqabah else 'Disabled':<30}║
║  Team: Dual-Agentic coordination active                                      ║
║  Knowledge: {knowledge_status:<30}                                          ║
║  Ihsan Threshold: {self.config.ihsan_threshold:<10}                         ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")

        # Start background tasks
        tasks = []

        # Event bus
        tasks.append(asyncio.create_task(self.event_bus.start()))

        # Scheduler
        tasks.append(asyncio.create_task(self.scheduler.start()))

        # Muraqabah monitoring
        if self.muraqabah:
            tasks.append(asyncio.create_task(self.muraqabah.start_monitoring()))

        # Auto-checkpoint
        tasks.append(
            asyncio.create_task(self.checkpointer.auto_checkpoint_loop(self._get_state))
        )

        # Main loop
        while self._running:
            try:
                result = await self.run_cycle()

                if self._cycle_count % 10 == 0:
                    logger.info(
                        f"Cycle {result.cycle_number}: "
                        f"proactive={result.opportunities_detected}, "
                        f"decisions={result.decisions_made}, "
                        f"synergy={result.collective_synergy:.2f}, "
                        f"ihsan={result.ihsan_score:.3f}"
                    )

            except Exception as e:
                logger.error(f"Main loop error: {e}")

            await asyncio.sleep(self.config.cycle_interval)

        # Cleanup
        for task in tasks:
            task.cancel()

    def stop(self) -> None:
        """Stop the Proactive Sovereign Entity."""
        self._running = False
        self.event_bus.stop()
        self.scheduler.stop()
        if self.muraqabah:
            self.muraqabah.stop_monitoring()
        self.checkpointer.stop()
        self.ooda_loop.stop()
        logger.info("Proactive Sovereign Entity stopped")

    def _get_state(self) -> Dict[str, Any]:
        """Get current state for checkpointing."""
        return {
            "cycle_count": self._cycle_count,
            "mode": self.config.mode.value,
            "active_goals": len(self._active_goals),
            "ooda_status": self.ooda_loop.status(),
            "autonomy_stats": self.autonomy.stats(),
            "bridge_stats": self.bridge.stats(),
            "scheduler_stats": self.scheduler.stats(),
            "knowledge_stats": (
                self.knowledge_bridge.stats() if self.knowledge_bridge else None
            ),
        }

    async def restore(self, checkpoint_id: Optional[str] = None) -> bool:
        """Restore from checkpoint."""
        checkpoint = await self.checkpointer.restore(checkpoint_id)
        if checkpoint:
            self._cycle_count = checkpoint.state.get("cycle_count", 0)
            logger.info(f"Restored from checkpoint {checkpoint.id}")
            return True
        return False

    async def query_knowledge(
        self,
        query: str,
        agent_role: AgentRole = AgentRole.MASTER_REASONER,
        max_results: int = 10,
    ) -> Dict[str, Any]:
        """
        Query the integrated knowledge base for agent use.

        This provides access to:
        - BIZRA Data Lake (vector embeddings, graphs, documents)
        - MoMo's 3-year R&D context
        - Session state and patterns
        - Standing on Giants attribution

        Args:
            query: Natural language query
            agent_role: Agent making the request (affects access)
            max_results: Maximum results to return

        Returns:
            Knowledge results with SNR scores
        """
        if not self.knowledge_bridge:
            return {
                "error": "Knowledge integration not initialized",
                "results": [],
            }

        result = await self.knowledge_bridge.query_for_agent(
            role=agent_role,
            query=query,
            max_results=max_results,
        )

        return {
            "query_id": result.query_id,
            "results": result.results,
            "sources": result.sources_consulted,
            "snr_score": result.snr_score,
            "latency_ms": result.latency_ms,
            "from_cache": result.from_cache,
        }

    def get_momo_context(self) -> Dict[str, Any]:
        """
        Get MoMo's R&D context for agents.

        Returns summary of 3 years of research including:
        - User identity and investment hours
        - Ihsan score and genesis status
        - Standing on Giants attribution chain
        """
        if not self.knowledge_bridge:
            return {}
        return self.knowledge_bridge.get_momo_context()

    def stats(self) -> Dict[str, Any]:
        """Get comprehensive entity statistics."""
        return {
            "running": self._running,
            "mode": self.config.mode.value,
            "cycle_count": self._cycle_count,
            "active_goals": len(self._active_goals),
            "ooda": self.ooda_loop.status(),
            "autonomy": self.autonomy.stats(),
            "bridge": self.bridge.stats(),
            "planner": self.planner.stats(),
            "scheduler": self.scheduler.stats(),
            "muraqabah": self.muraqabah.stats() if self.muraqabah else None,
            "monitor": self.monitor.stats() if self.monitor else None,
            "collective": self.collective.stats() if self.collective else None,
            "checkpointer": self.checkpointer.stats(),
            "knowledge": (
                self.knowledge_bridge.stats() if self.knowledge_bridge else None
            ),
        }


# Factory function
def create_proactive_entity(
    mode: EntityMode = EntityMode.PROACTIVE_PARTNER,
    ihsan_threshold: float = 0.95,
    autonomy_level: AutonomyLevel = AutonomyLevel.AUTOLOW,
) -> ProactiveSovereignEntity:
    """Create a configured Proactive Sovereign Entity."""
    config = EntityConfig(
        mode=mode,
        ihsan_threshold=ihsan_threshold,
        default_autonomy=autonomy_level,
    )
    return ProactiveSovereignEntity(config)


__all__ = [
    "EntityConfig",
    "EntityCycleResult",
    "EntityMode",
    "ProactiveSovereignEntity",
    "create_proactive_entity",
]
