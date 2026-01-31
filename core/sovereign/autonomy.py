"""
BIZRA Sovereign Engine - Autonomous Loop Module
================================================
Implements the OBSERVE → ANALYZE → DECIDE → ACT → REFLECT cycle
with DecisionGate for Ihsān-compliant autonomous operation.

Author: BIZRA Sovereign Engine
Version: 1.0.0
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Callable, Awaitable
from datetime import datetime, timedelta
import asyncio
import uuid
import logging
from collections import deque

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class LoopState(Enum):
    """States of the autonomous loop."""
    IDLE = auto()
    OBSERVING = auto()
    ANALYZING = auto()
    PLANNING = auto()
    ACTING = auto()
    REFLECTING = auto()
    ADAPTING = auto()
    PAUSED = auto()
    EMERGENCY = auto()


class DecisionType(Enum):
    """Types of autonomous decisions."""
    ROUTINE = auto()       # Standard operations
    ADAPTIVE = auto()      # Response to changing conditions
    CORRECTIVE = auto()    # Fix detected issues
    PREVENTIVE = auto()    # Prevent predicted issues
    INNOVATIVE = auto()    # Try new approaches
    EMERGENCY = auto()     # Critical interventions


class GateResult(Enum):
    """Results from the decision gate."""
    PASS = auto()      # Decision approved
    REJECT = auto()    # Decision denied
    DEFER = auto()     # Needs more information
    ESCALATE = auto()  # Requires human approval


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class SystemMetrics:
    """Current system health metrics."""
    snr_score: float = 0.0
    ihsan_score: float = 0.0
    latency_ms: float = 0.0
    error_rate: float = 0.0
    throughput: float = 0.0
    memory_usage: float = 0.0
    active_tasks: int = 0
    pending_decisions: int = 0
    timestamp: datetime = field(default_factory=datetime.now)

    def health_score(self) -> float:
        """Calculate overall health score (0-1)."""
        weights = {
            'snr': 0.25,
            'ihsan': 0.25,
            'error': 0.20,
            'latency': 0.15,
            'memory': 0.15
        }

        snr_norm = min(1.0, self.snr_score / 0.95)
        ihsan_norm = min(1.0, self.ihsan_score / 0.95)
        error_norm = max(0.0, 1.0 - self.error_rate * 10)
        latency_norm = max(0.0, 1.0 - self.latency_ms / 5000)
        memory_norm = max(0.0, 1.0 - self.memory_usage)

        return (
            weights['snr'] * snr_norm +
            weights['ihsan'] * ihsan_norm +
            weights['error'] * error_norm +
            weights['latency'] * latency_norm +
            weights['memory'] * memory_norm
        )

    def is_healthy(self, threshold: float = 0.85) -> bool:
        """Check if system is healthy."""
        return self.health_score() >= threshold


@dataclass
class DecisionCandidate:
    """A candidate decision for evaluation."""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    decision_type: DecisionType = DecisionType.ROUTINE
    action: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    expected_impact: float = 0.0
    risk_score: float = 0.0
    confidence: float = 0.0
    rationale: str = ""
    rollback_plan: str = ""
    timeout_seconds: float = 60.0
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class DecisionOutcome:
    """Outcome of an executed decision."""
    decision_id: str = ""
    gate_result: GateResult = GateResult.PASS
    executed: bool = False
    success: bool = False
    actual_impact: float = 0.0
    execution_time_ms: float = 0.0
    error_message: str = ""
    metrics_before: Optional[SystemMetrics] = None
    metrics_after: Optional[SystemMetrics] = None
    completed_at: datetime = field(default_factory=datetime.now)


# =============================================================================
# DECISION GATE
# =============================================================================

class DecisionGate:
    """
    Ihsān-compliant gate for autonomous decisions.
    Ensures all actions meet quality and safety thresholds.
    """

    # Risk thresholds by decision type
    RISK_THRESHOLDS = {
        DecisionType.ROUTINE: 0.3,
        DecisionType.ADAPTIVE: 0.4,
        DecisionType.CORRECTIVE: 0.5,
        DecisionType.PREVENTIVE: 0.4,
        DecisionType.INNOVATIVE: 0.6,
        DecisionType.EMERGENCY: 0.7,
    }

    # Confidence requirements by decision type
    CONFIDENCE_REQUIREMENTS = {
        DecisionType.ROUTINE: 0.7,
        DecisionType.ADAPTIVE: 0.75,
        DecisionType.CORRECTIVE: 0.8,
        DecisionType.PREVENTIVE: 0.75,
        DecisionType.INNOVATIVE: 0.85,
        DecisionType.EMERGENCY: 0.6,
    }

    def __init__(
        self,
        ihsan_threshold: float = 0.95,
        require_rollback: bool = True,
        max_concurrent: int = 5
    ):
        self.ihsan_threshold = ihsan_threshold
        self.require_rollback = require_rollback
        self.max_concurrent = max_concurrent
        self.active_decisions: Dict[str, DecisionCandidate] = {}
        self.decision_history: deque = deque(maxlen=1000)
        self.passed_count = 0
        self.blocked_count = 0

    async def evaluate(
        self,
        decision: DecisionCandidate,
        metrics: SystemMetrics
    ) -> GateResult:
        """Evaluate a decision candidate."""

        # Check concurrent limit
        if len(self.active_decisions) >= self.max_concurrent:
            logger.warning(f"Gate: Too many concurrent decisions")
            return GateResult.DEFER

        # Check risk threshold
        risk_limit = self.RISK_THRESHOLDS.get(decision.decision_type, 0.5)
        if decision.risk_score > risk_limit:
            logger.warning(f"Gate: Risk {decision.risk_score:.2f} > {risk_limit:.2f}")
            self.blocked_count += 1
            return GateResult.REJECT

        # Check confidence requirement
        conf_req = self.CONFIDENCE_REQUIREMENTS.get(decision.decision_type, 0.7)
        if decision.confidence < conf_req:
            logger.info(f"Gate: Confidence {decision.confidence:.2f} < {conf_req:.2f}")
            return GateResult.DEFER

        # Check rollback plan for non-routine decisions
        if self.require_rollback and decision.decision_type != DecisionType.ROUTINE:
            if not decision.rollback_plan:
                logger.warning("Gate: No rollback plan for non-routine decision")
                return GateResult.DEFER

        # Check system health for non-emergency decisions
        if decision.decision_type != DecisionType.EMERGENCY:
            if not metrics.is_healthy(0.7):
                logger.warning("Gate: System unhealthy for non-emergency action")
                return GateResult.DEFER

        # All checks passed
        self.passed_count += 1
        self.active_decisions[decision.id] = decision
        logger.info(f"Gate: PASS for decision {decision.id}")
        return GateResult.PASS

    def complete_decision(self, decision_id: str, outcome: DecisionOutcome):
        """Mark a decision as completed."""
        if decision_id in self.active_decisions:
            del self.active_decisions[decision_id]
        self.decision_history.append(outcome)

    @property
    def approval_rate(self) -> float:
        """Get the approval rate."""
        total = self.passed_count + self.blocked_count
        return self.passed_count / total if total > 0 else 1.0

    def get_stats(self) -> Dict[str, Any]:
        """Get gate statistics."""
        return {
            "passed": self.passed_count,
            "blocked": self.blocked_count,
            "approval_rate": self.approval_rate,
            "active": len(self.active_decisions),
            "history_size": len(self.decision_history)
        }


# =============================================================================
# AUTONOMOUS LOOP
# =============================================================================

class AutonomousLoop:
    """
    The core autonomous operation loop.
    Implements: OBSERVE → ANALYZE → DECIDE → ACT → REFLECT
    """

    def __init__(
        self,
        decision_gate: Optional[DecisionGate] = None,
        snr_threshold: float = 0.95,
        ihsan_threshold: float = 0.95,
        cycle_interval: float = 5.0,
        max_decisions_per_cycle: int = 3
    ):
        self.gate = decision_gate or DecisionGate(ihsan_threshold)
        self.snr_threshold = snr_threshold
        self.ihsan_threshold = ihsan_threshold
        self.cycle_interval = cycle_interval
        self.max_decisions_per_cycle = max_decisions_per_cycle

        # State
        self.state = LoopState.IDLE
        self.cycle_count = 0
        self._running = False
        self._paused = False

        # History
        self.observations: deque = deque(maxlen=1000)
        self.decisions: deque = deque(maxlen=500)
        self.outcomes: deque = deque(maxlen=500)

        # Callbacks
        self._observers: List[Callable[[SystemMetrics], Awaitable[None]]] = []
        self._analyzers: List[Callable[[SystemMetrics], Awaitable[List[DecisionCandidate]]]] = []
        self._executors: Dict[str, Callable[[DecisionCandidate], Awaitable[bool]]] = {}

    # -------------------------------------------------------------------------
    # PHASE 1: OBSERVE
    # -------------------------------------------------------------------------

    async def observe(self) -> SystemMetrics:
        """Gather current system metrics."""
        self.state = LoopState.OBSERVING

        # Default metrics (override with real collectors)
        metrics = SystemMetrics(
            snr_score=0.92,
            ihsan_score=0.94,
            latency_ms=150,
            error_rate=0.02,
            throughput=100,
            memory_usage=0.45,
            active_tasks=len(self.gate.active_decisions),
            pending_decisions=0
        )

        # Call registered observers
        for observer in self._observers:
            try:
                await observer(metrics)
            except Exception as e:
                logger.error(f"Observer error: {e}")

        self.observations.append(metrics)
        return metrics

    # -------------------------------------------------------------------------
    # PHASE 2: ANALYZE
    # -------------------------------------------------------------------------

    async def analyze(self, metrics: SystemMetrics) -> List[DecisionCandidate]:
        """Analyze metrics and generate decision candidates."""
        self.state = LoopState.ANALYZING
        candidates = []

        # Built-in analysis: SNR correction
        if metrics.snr_score < self.snr_threshold - 0.1:
            candidates.append(DecisionCandidate(
                decision_type=DecisionType.CORRECTIVE,
                action="boost_snr",
                parameters={"target": self.snr_threshold},
                expected_impact=0.15,
                risk_score=0.2,
                confidence=0.85,
                rationale=f"SNR {metrics.snr_score:.2f} below threshold",
                rollback_plan="revert_snr_boost"
            ))

        # Built-in analysis: Error rate
        if metrics.error_rate > 0.1:
            candidates.append(DecisionCandidate(
                decision_type=DecisionType.CORRECTIVE,
                action="reduce_errors",
                parameters={"current_rate": metrics.error_rate},
                expected_impact=0.2,
                risk_score=0.3,
                confidence=0.8,
                rationale=f"Error rate {metrics.error_rate:.2%} too high",
                rollback_plan="restore_previous_config"
            ))

        # Call registered analyzers
        for analyzer in self._analyzers:
            try:
                new_candidates = await analyzer(metrics)
                candidates.extend(new_candidates)
            except Exception as e:
                logger.error(f"Analyzer error: {e}")

        # Limit candidates per cycle
        return candidates[:self.max_decisions_per_cycle]

    # -------------------------------------------------------------------------
    # PHASE 3: DECIDE
    # -------------------------------------------------------------------------

    async def decide(
        self,
        candidates: List[DecisionCandidate],
        metrics: SystemMetrics
    ) -> List[DecisionCandidate]:
        """Filter candidates through the decision gate."""
        self.state = LoopState.PLANNING
        approved = []

        for candidate in candidates:
            result = await self.gate.evaluate(candidate, metrics)

            if result == GateResult.PASS:
                approved.append(candidate)
                self.decisions.append(candidate)
            elif result == GateResult.ESCALATE:
                logger.info(f"Decision {candidate.id} escalated for human review")

        return approved

    # -------------------------------------------------------------------------
    # PHASE 4: ACT
    # -------------------------------------------------------------------------

    async def act(
        self,
        decisions: List[DecisionCandidate],
        metrics: SystemMetrics
    ) -> List[DecisionOutcome]:
        """Execute approved decisions."""
        self.state = LoopState.ACTING
        outcomes = []

        for decision in decisions:
            start_time = datetime.now()
            outcome = DecisionOutcome(
                decision_id=decision.id,
                gate_result=GateResult.PASS,
                metrics_before=metrics
            )

            try:
                # Find executor for action
                executor = self._executors.get(decision.action)

                if executor:
                    outcome.success = await asyncio.wait_for(
                        executor(decision),
                        timeout=decision.timeout_seconds
                    )
                else:
                    # Default: log-only execution
                    logger.info(f"Executing: {decision.action} with {decision.parameters}")
                    outcome.success = True

                outcome.executed = True

            except asyncio.TimeoutError:
                outcome.error_message = "Execution timeout"
                outcome.success = False
            except Exception as e:
                outcome.error_message = str(e)
                outcome.success = False
                logger.error(f"Execution error for {decision.id}: {e}")

            outcome.execution_time_ms = (datetime.now() - start_time).total_seconds() * 1000
            outcome.completed_at = datetime.now()

            self.gate.complete_decision(decision.id, outcome)
            self.outcomes.append(outcome)
            outcomes.append(outcome)

        return outcomes

    # -------------------------------------------------------------------------
    # PHASE 5: REFLECT
    # -------------------------------------------------------------------------

    async def reflect(self, outcomes: List[DecisionOutcome]) -> Dict[str, Any]:
        """Analyze outcomes and adapt."""
        self.state = LoopState.REFLECTING

        successful = sum(1 for o in outcomes if o.success)
        total = len(outcomes)

        reflection = {
            "cycle": self.cycle_count,
            "decisions_made": total,
            "successful": successful,
            "success_rate": successful / total if total > 0 else 1.0,
            "avg_execution_time": sum(o.execution_time_ms for o in outcomes) / total if total > 0 else 0,
            "gate_stats": self.gate.get_stats()
        }

        # Adaptation logic
        if reflection["success_rate"] < 0.5 and total >= 3:
            self.state = LoopState.ADAPTING
            logger.warning("Low success rate - entering adaptation mode")

        return reflection

    # -------------------------------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------------------------------

    async def run_cycle(self) -> Dict[str, Any]:
        """Execute one complete cycle."""
        self.cycle_count += 1

        # OBSERVE
        metrics = await self.observe()

        # ANALYZE
        candidates = await self.analyze(metrics)

        # DECIDE
        approved = await self.decide(candidates, metrics)

        # ACT
        outcomes = await self.act(approved, metrics)

        # REFLECT
        reflection = await self.reflect(outcomes)

        return {
            "cycle": self.cycle_count,
            "state": self.state.name,
            "health": metrics.health_score(),
            "candidates": len(candidates),
            "approved": len(approved),
            "executed": len(outcomes),
            "reflection": reflection
        }

    async def run(self):
        """Run the autonomous loop continuously."""
        self._running = True
        logger.info("Autonomous loop started")

        while self._running:
            if self._paused:
                await asyncio.sleep(1)
                continue

            try:
                result = await self.run_cycle()
                logger.debug(f"Cycle {result['cycle']}: {result['executed']} decisions")
            except Exception as e:
                logger.error(f"Cycle error: {e}")
                self.state = LoopState.EMERGENCY

            await asyncio.sleep(self.cycle_interval)

        self.state = LoopState.IDLE
        logger.info("Autonomous loop stopped")

    def start(self) -> asyncio.Task:
        """Start the loop as a background task."""
        return asyncio.create_task(self.run())

    def stop(self):
        """Stop the loop."""
        self._running = False

    def pause(self):
        """Pause the loop."""
        self._paused = True
        self.state = LoopState.PAUSED

    def resume(self):
        """Resume the loop."""
        self._paused = False

    # -------------------------------------------------------------------------
    # REGISTRATION
    # -------------------------------------------------------------------------

    def register_observer(self, observer: Callable[[SystemMetrics], Awaitable[None]]):
        """Register a metrics observer."""
        self._observers.append(observer)

    def register_analyzer(self, analyzer: Callable[[SystemMetrics], Awaitable[List[DecisionCandidate]]]):
        """Register a decision analyzer."""
        self._analyzers.append(analyzer)

    def register_executor(self, action: str, executor: Callable[[DecisionCandidate], Awaitable[bool]]):
        """Register an action executor."""
        self._executors[action] = executor

    # -------------------------------------------------------------------------
    # STATUS
    # -------------------------------------------------------------------------

    def status(self) -> Dict[str, Any]:
        """Get current loop status."""
        return {
            "state": self.state.name,
            "cycle": self.cycle_count,
            "running": self._running,
            "paused": self._paused,
            "observations": len(self.observations),
            "decisions": len(self.decisions),
            "outcomes": len(self.outcomes),
            "gate": self.gate.get_stats()
        }


# =============================================================================
# FACTORY FUNCTION
# =============================================================================

def create_autonomous_loop(
    snr_threshold: float = 0.95,
    ihsan_threshold: float = 0.95,
    cycle_interval: float = 5.0
) -> AutonomousLoop:
    """Create a configured autonomous loop."""
    gate = DecisionGate(ihsan_threshold=ihsan_threshold)
    return AutonomousLoop(
        decision_gate=gate,
        snr_threshold=snr_threshold,
        ihsan_threshold=ihsan_threshold,
        cycle_interval=cycle_interval
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    'LoopState',
    'DecisionType',
    'GateResult',
    'SystemMetrics',
    'DecisionCandidate',
    'DecisionOutcome',
    'DecisionGate',
    'AutonomousLoop',
    'create_autonomous_loop',
]
