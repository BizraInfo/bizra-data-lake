"""
Autopoietic Loop Engine - Self-Improvement Cycle with FATE Verification
=========================================================================

Core engine for BIZRA's autopoietic (self-creating/self-improving) system.
Implements a continuous improvement loop with formal verification gates.

Cycle Phases:
    DORMANT -> OBSERVING -> HYPOTHESIZING -> VALIDATING -> IMPLEMENTING ->
    INTEGRATING -> REFLECTING -> DORMANT

Each improvement hypothesis must pass Z3 FATE gate verification before
shadow deployment, ensuring constitutional compliance is maintained.

Standing on Giants:
- Maturana & Varela (1972): Autopoiesis - self-creating systems theory
- Deming (1986): PDCA continuous improvement cycle
- Anthropic (2025): Constitutional AI constraints

Genesis Strict Synthesis v2.2.2
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum, auto
from pathlib import Path
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
    STRICT_IHSAN_THRESHOLD,
)

logger = logging.getLogger(__name__)


# =============================================================================
# AUTOPOIETIC STATE MACHINE
# =============================================================================


class AutopoieticState(Enum):
    """
    States of the autopoietic improvement loop.

    The loop progresses: DORMANT -> OBSERVING -> ... -> REFLECTING -> DORMANT
    Emergency states bypass normal flow for safety.
    """
    DORMANT = "dormant"              # Not running, awaiting activation
    OBSERVING = "observing"          # Collecting system metrics
    HYPOTHESIZING = "hypothesizing"  # Generating improvement candidates
    VALIDATING = "validating"        # Z3 FATE gate verification
    IMPLEMENTING = "implementing"    # Shadow deployment of validated hypothesis
    INTEGRATING = "integrating"      # Learning from outcome, consolidating
    REFLECTING = "reflecting"        # Meta-analysis of loop performance
    EMERGENCY_ROLLBACK = "emergency_rollback"  # Safety-triggered rollback
    HALTED = "halted"               # Human intervention required


class HypothesisCategory(Enum):
    """Categories of improvement hypotheses."""
    PERFORMANCE = "performance"           # Latency, throughput optimization
    QUALITY = "quality"                   # SNR, Ihsan improvement
    EFFICIENCY = "efficiency"             # Resource utilization
    ROBUSTNESS = "robustness"             # Error handling, fault tolerance
    CAPABILITY = "capability"             # New features or abilities
    STRUCTURAL = "structural"             # Architecture changes (requires approval)


class RiskLevel(Enum):
    """Risk levels for hypotheses."""
    NEGLIGIBLE = 1    # No observable impact on failure
    LOW = 2           # Minor degradation, auto-recoverable
    MODERATE = 3      # Noticeable impact, requires monitoring
    HIGH = 4          # Significant impact, requires human approval
    CRITICAL = 5      # System-wide impact, forbidden without council approval


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class SystemObservation:
    """Snapshot of system metrics at observation time."""
    observation_id: str
    timestamp: datetime

    # Core metrics
    ihsan_score: float
    snr_score: float
    latency_p50_ms: float
    latency_p99_ms: float
    error_rate: float
    throughput_qps: float

    # Resource metrics
    cpu_usage: float
    memory_usage: float
    gpu_usage: Optional[float] = None

    # Quality metrics
    agent_performance: Dict[str, float] = field(default_factory=dict)
    constitutional_compliance: float = 1.0

    # Trend indicators
    trend_direction: str = "stable"  # improving, degrading, stable
    anomalies_detected: List[str] = field(default_factory=list)

    # Raw sensor data
    sensor_readings: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "observation_id": self.observation_id,
            "timestamp": self.timestamp.isoformat(),
            "ihsan_score": self.ihsan_score,
            "snr_score": self.snr_score,
            "latency_p50_ms": self.latency_p50_ms,
            "latency_p99_ms": self.latency_p99_ms,
            "error_rate": self.error_rate,
            "throughput_qps": self.throughput_qps,
            "cpu_usage": self.cpu_usage,
            "memory_usage": self.memory_usage,
            "trend_direction": self.trend_direction,
            "anomalies": self.anomalies_detected,
        }


@dataclass
class ActivationGuardrails:
    """Guardrails that must pass before autopoiesis can activate."""
    enabled: bool = True
    require_live_sensors: bool = True
    allow_mock_sensors: bool = False
    require_fate_gate: bool = True
    allow_mock_fate_gate: bool = False
    min_observations: int = 3
    max_anomalies: int = 0
    max_error_rate: float = 0.02
    max_latency_p99_ms: float = 200.0
    min_throughput_qps: float = 1.0
    min_ihsan_score: float = UNIFIED_IHSAN_THRESHOLD
    min_snr_score: float = UNIFIED_SNR_THRESHOLD
    require_stable_or_improving: bool = True


@dataclass
class ActivationReport:
    """Outcome of activation guardrail evaluation."""
    activated: bool
    reasons: List[str] = field(default_factory=list)
    observation: Optional[SystemObservation] = None
    fate_proof_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "activated": self.activated,
            "reasons": self.reasons,
            "observation": self.observation.to_dict() if self.observation else None,
            "fate_proof_id": self.fate_proof_id,
        }


@dataclass
class Hypothesis:
    """
    An improvement hypothesis generated during the HYPOTHESIZING phase.

    Represents a potential change that could improve system behavior,
    with associated risk assessment and reversibility plan.
    """
    id: str
    description: str
    category: HypothesisCategory
    predicted_improvement: float  # Expected improvement ratio (e.g., 0.05 = 5%)

    # Change specification
    required_changes: List[Dict[str, Any]]  # List of changes to apply
    affected_components: List[str]          # Components that will be modified

    # Risk and safety
    risk_level: RiskLevel
    reversibility_plan: Dict[str, Any]      # How to undo if needed
    ihsan_impact_estimate: float            # Expected Ihsan delta (-/+)
    snr_impact_estimate: float              # Expected SNR delta (-/+)

    # Validation requirements
    requires_human_approval: bool = False
    requires_shadow_deployment: bool = True
    min_observation_duration_s: int = 300   # 5 minutes default

    # Metadata
    source_observation_id: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = 0.5

    def __post_init__(self):
        # Auto-require human approval for high-risk or structural changes
        if self.risk_level.value >= RiskLevel.HIGH.value:
            self.requires_human_approval = True
        if self.category == HypothesisCategory.STRUCTURAL:
            self.requires_human_approval = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "description": self.description,
            "category": self.category.value,
            "predicted_improvement": self.predicted_improvement,
            "risk_level": self.risk_level.value,
            "ihsan_impact_estimate": self.ihsan_impact_estimate,
            "snr_impact_estimate": self.snr_impact_estimate,
            "requires_approval": self.requires_human_approval,
            "confidence": self.confidence,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class ValidationResult:
    """Result of Z3 FATE gate verification."""
    hypothesis_id: str
    is_valid: bool

    # Z3 verification results
    z3_satisfiable: bool
    z3_proof_id: Optional[str] = None
    z3_generation_time_ms: int = 0

    # Constitutional checks
    ihsan_gate_passed: bool = True
    snr_gate_passed: bool = True
    reversibility_verified: bool = True

    # Violations found
    violations: List[str] = field(default_factory=list)
    counterexample: Optional[str] = None

    # Recommendation
    recommendation: str = "proceed"  # proceed, review, reject

    validated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "is_valid": self.is_valid,
            "z3_satisfiable": self.z3_satisfiable,
            "violations": self.violations,
            "recommendation": self.recommendation,
            "validated_at": self.validated_at.isoformat(),
        }


@dataclass
class ImplementationResult:
    """Result of shadow deployment implementation."""
    hypothesis_id: str
    success: bool

    # Shadow deployment metrics
    shadow_instance_id: Optional[str] = None
    deployment_time_ms: int = 0

    # Observed behavior
    baseline_metrics: Dict[str, float] = field(default_factory=dict)
    shadow_metrics: Dict[str, float] = field(default_factory=dict)
    improvement_observed: float = 0.0

    # Safety checks
    ihsan_maintained: bool = True
    snr_maintained: bool = True
    error_rate_acceptable: bool = True

    # Status
    needs_rollback: bool = False
    rollback_reason: Optional[str] = None

    implemented_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "success": self.success,
            "improvement_observed": self.improvement_observed,
            "ihsan_maintained": self.ihsan_maintained,
            "snr_maintained": self.snr_maintained,
            "needs_rollback": self.needs_rollback,
        }


@dataclass
class IntegrationResult:
    """Result of learning and consolidation phase."""
    hypothesis_id: str
    integrated: bool

    # Learning outcomes
    lesson_learned: str
    parameters_updated: Dict[str, Any] = field(default_factory=dict)

    # Impact summary
    final_improvement: float = 0.0
    ihsan_delta: float = 0.0
    snr_delta: float = 0.0

    # Knowledge capture
    pattern_id: Optional[str] = None
    archived: bool = False

    integrated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "integrated": self.integrated,
            "lesson_learned": self.lesson_learned,
            "final_improvement": self.final_improvement,
            "ihsan_delta": self.ihsan_delta,
            "snr_delta": self.snr_delta,
        }


@dataclass
class AutopoieticResult:
    """Complete result of one autopoietic cycle."""
    cycle_id: str
    state: AutopoieticState

    # Phase results
    observation: Optional[SystemObservation] = None
    hypotheses_generated: int = 0
    hypothesis_validated: Optional[Hypothesis] = None
    validation_result: Optional[ValidationResult] = None
    implementation_result: Optional[ImplementationResult] = None
    integration_result: Optional[IntegrationResult] = None

    # Cycle metrics
    cycle_duration_ms: int = 0
    improvements_attempted: int = 0
    improvements_integrated: int = 0

    # Safety events
    rollbacks_triggered: int = 0
    human_approvals_pending: int = 0

    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "state": self.state.value,
            "observation": self.observation.to_dict() if self.observation else None,
            "hypotheses_generated": self.hypotheses_generated,
            "hypothesis_validated": self.hypothesis_validated.to_dict() if self.hypothesis_validated else None,
            "improvements_integrated": self.improvements_integrated,
            "cycle_duration_ms": self.cycle_duration_ms,
            "rollbacks_triggered": self.rollbacks_triggered,
        }


@dataclass
class AuditLogEntry:
    """Audit log entry for all autopoietic modifications."""
    entry_id: str
    timestamp: datetime
    action: str
    hypothesis_id: Optional[str]

    # Change details
    component_affected: str
    change_type: str
    before_state: Dict[str, Any]
    after_state: Dict[str, Any]

    # Verification
    verified_by: str  # "z3_fate", "human", "auto"
    ihsan_score_at_action: float
    snr_score_at_action: float

    # Outcome
    outcome: str  # "success", "rollback", "pending"
    rollback_triggered: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entry_id": self.entry_id,
            "timestamp": self.timestamp.isoformat(),
            "action": self.action,
            "hypothesis_id": self.hypothesis_id,
            "component_affected": self.component_affected,
            "verified_by": self.verified_by,
            "outcome": self.outcome,
        }


# =============================================================================
# RATE LIMITER
# =============================================================================


class RateLimiter:
    """Rate limiter for autopoietic improvements."""

    def __init__(
        self,
        max_improvements_per_cycle: int = 1,
        min_cycle_interval_s: float = 60.0,
        cooldown_after_rollback_s: float = 300.0,
    ):
        self.max_improvements_per_cycle = max_improvements_per_cycle
        self.min_cycle_interval_s = min_cycle_interval_s
        self.cooldown_after_rollback_s = cooldown_after_rollback_s

        self._last_cycle_time: Optional[float] = None
        self._improvements_this_cycle: int = 0
        self._rollback_cooldown_until: Optional[float] = None

    def can_attempt_improvement(self) -> Tuple[bool, str]:
        """Check if an improvement attempt is allowed."""
        now = time.time()

        # Check rollback cooldown
        if self._rollback_cooldown_until and now < self._rollback_cooldown_until:
            remaining = self._rollback_cooldown_until - now
            return False, f"Rollback cooldown active: {remaining:.0f}s remaining"

        # Check cycle interval
        if self._last_cycle_time:
            elapsed = now - self._last_cycle_time
            if elapsed < self.min_cycle_interval_s:
                remaining = self.min_cycle_interval_s - elapsed
                return False, f"Cycle interval not met: {remaining:.0f}s remaining"

        # Check improvements per cycle
        if self._improvements_this_cycle >= self.max_improvements_per_cycle:
            return False, f"Max improvements ({self.max_improvements_per_cycle}) reached this cycle"

        return True, "Allowed"

    def record_improvement_attempt(self):
        """Record an improvement attempt."""
        self._improvements_this_cycle += 1

    def start_new_cycle(self):
        """Start a new improvement cycle."""
        self._last_cycle_time = time.time()
        self._improvements_this_cycle = 0

    def trigger_rollback_cooldown(self):
        """Trigger cooldown after a rollback."""
        self._rollback_cooldown_until = time.time() + self.cooldown_after_rollback_s
        logger.warning(f"Rollback cooldown activated for {self.cooldown_after_rollback_s}s")


# =============================================================================
# ROLLBACK MANAGER
# =============================================================================


class RollbackManager:
    """Manages rollback state and triggers."""

    def __init__(
        self,
        ihsan_floor: float = UNIFIED_IHSAN_THRESHOLD,
        snr_floor: float = UNIFIED_SNR_THRESHOLD,
        error_spike_threshold: float = 0.1,
        max_rollback_history: int = 100,
    ):
        self.ihsan_floor = ihsan_floor
        self.snr_floor = snr_floor
        self.error_spike_threshold = error_spike_threshold

        self._state_stack: Deque[Dict[str, Any]] = deque(maxlen=max_rollback_history)
        self._rollback_history: List[Dict[str, Any]] = []

    def save_state(self, state: Dict[str, Any], hypothesis_id: str):
        """Save state before applying a change."""
        self._state_stack.append({
            "hypothesis_id": hypothesis_id,
            "state": state,
            "saved_at": datetime.now(timezone.utc).isoformat(),
        })

    def should_rollback(
        self,
        current_ihsan: float,
        current_snr: float,
        current_error_rate: float,
        baseline_error_rate: float,
    ) -> Tuple[bool, str]:
        """Check if rollback should be triggered."""
        reasons = []

        # Ihsan drop
        if current_ihsan < self.ihsan_floor:
            reasons.append(f"Ihsan dropped below floor: {current_ihsan:.3f} < {self.ihsan_floor}")

        # SNR drop
        if current_snr < self.snr_floor:
            reasons.append(f"SNR dropped below floor: {current_snr:.3f} < {self.snr_floor}")

        # Error spike
        error_delta = current_error_rate - baseline_error_rate
        if error_delta > self.error_spike_threshold:
            reasons.append(f"Error spike detected: {error_delta:.3f} > {self.error_spike_threshold}")

        if reasons:
            return True, "; ".join(reasons)
        return False, ""

    def get_rollback_state(self, hypothesis_id: str) -> Optional[Dict[str, Any]]:
        """Get saved state for rollback."""
        for entry in reversed(self._state_stack):
            if entry["hypothesis_id"] == hypothesis_id:
                return entry["state"]
        return None

    def record_rollback(self, hypothesis_id: str, reason: str):
        """Record a rollback event."""
        self._rollback_history.append({
            "hypothesis_id": hypothesis_id,
            "reason": reason,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })


# =============================================================================
# HUMAN APPROVAL QUEUE
# =============================================================================


@dataclass
class ApprovalRequest:
    """Request for human approval of a structural change."""
    request_id: str
    hypothesis: Hypothesis
    requested_at: datetime

    # Context
    observation_summary: Dict[str, Any]
    risk_assessment: str
    expected_impact: str

    # Response
    approved: Optional[bool] = None
    approver: Optional[str] = None
    approved_at: Optional[datetime] = None
    rejection_reason: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "request_id": self.request_id,
            "hypothesis_id": self.hypothesis.id,
            "description": self.hypothesis.description,
            "category": self.hypothesis.category.value,
            "risk_level": self.hypothesis.risk_level.value,
            "requested_at": self.requested_at.isoformat(),
            "approved": self.approved,
            "approver": self.approver,
        }


class HumanApprovalQueue:
    """Queue for human approval of high-risk changes."""

    def __init__(self, max_pending: int = 10):
        self.max_pending = max_pending
        self._pending: Dict[str, ApprovalRequest] = {}
        self._completed: List[ApprovalRequest] = []

    def submit_for_approval(
        self,
        hypothesis: Hypothesis,
        observation: SystemObservation,
    ) -> ApprovalRequest:
        """Submit a hypothesis for human approval."""
        request_id = f"approval_{uuid.uuid4().hex[:8]}"

        request = ApprovalRequest(
            request_id=request_id,
            hypothesis=hypothesis,
            requested_at=datetime.now(timezone.utc),
            observation_summary=observation.to_dict(),
            risk_assessment=f"Risk Level: {hypothesis.risk_level.name}",
            expected_impact=f"Predicted improvement: {hypothesis.predicted_improvement:.1%}",
        )

        self._pending[request_id] = request
        logger.info(f"Submitted hypothesis {hypothesis.id} for human approval: {request_id}")

        return request

    def approve(self, request_id: str, approver: str) -> bool:
        """Approve a pending request."""
        if request_id not in self._pending:
            return False

        request = self._pending.pop(request_id)
        request.approved = True
        request.approver = approver
        request.approved_at = datetime.now(timezone.utc)

        self._completed.append(request)
        logger.info(f"Request {request_id} approved by {approver}")

        return True

    def reject(self, request_id: str, approver: str, reason: str) -> bool:
        """Reject a pending request."""
        if request_id not in self._pending:
            return False

        request = self._pending.pop(request_id)
        request.approved = False
        request.approver = approver
        request.approved_at = datetime.now(timezone.utc)
        request.rejection_reason = reason

        self._completed.append(request)
        logger.info(f"Request {request_id} rejected by {approver}: {reason}")

        return True

    def get_pending(self) -> List[ApprovalRequest]:
        """Get all pending approval requests."""
        return list(self._pending.values())

    def check_approval(self, hypothesis_id: str) -> Optional[bool]:
        """Check if a hypothesis has been approved/rejected."""
        for request in self._completed:
            if request.hypothesis.id == hypothesis_id:
                return request.approved
        return None


# =============================================================================
# AUTOPOIETIC LOOP ENGINE
# =============================================================================


class AutopoieticLoop:
    """
    Core autopoietic improvement loop engine.

    Implements continuous self-improvement with safety constraints:
    - FATE gate verification before any change
    - Rate limiting (max 1 improvement per cycle)
    - Automatic rollback on quality degradation
    - Human approval for structural changes
    - Complete audit trail

    Usage:
        from core.sovereign.z3_fate_gate import Z3FATEGate

        fate_gate = Z3FATEGate()
        loop = AutopoieticLoop(fate_gate=fate_gate, ihsan_floor=0.95)

        # Run one cycle
        result = await loop.run_cycle()

        # Check status
        print(loop.get_status())

        # Start continuous improvement
        await loop.start()

        # Stop
        await loop.stop()
    """

    def __init__(
        self,
        fate_gate: Optional[Any] = None,
        ihsan_floor: float = UNIFIED_IHSAN_THRESHOLD,
        snr_floor: float = UNIFIED_SNR_THRESHOLD,
        cycle_interval_s: float = 60.0,
        max_improvements_per_cycle: int = 1,
        audit_log_path: Optional[Path] = None,
        sensor_hub: Optional[Any] = None,
        activation_guardrails: Optional[ActivationGuardrails] = None,
    ):
        """
        Initialize the autopoietic loop.

        Args:
            fate_gate: Z3FATEGate instance for formal verification
            ihsan_floor: Minimum Ihsan score (default: 0.95)
            snr_floor: Minimum SNR score (default: 0.85)
            cycle_interval_s: Seconds between cycles (default: 60)
            max_improvements_per_cycle: Rate limit (default: 1)
            audit_log_path: Path for audit log file
            sensor_hub: MuraqabahSensorHub instance for metrics
            activation_guardrails: Guardrails for activation preflight
        """
        self.ihsan_floor = ihsan_floor
        self.snr_floor = snr_floor
        self.cycle_interval_s = cycle_interval_s

        # External dependencies (lazy-loaded if not provided)
        self._fate_gate = fate_gate
        self._sensor_hub = sensor_hub
        self._activation_guardrails = activation_guardrails or ActivationGuardrails()
        self._activated = False
        self._activation_report: Optional[ActivationReport] = None
        self._activation_failures: int = 0

        # Internal components
        self._rate_limiter = RateLimiter(
            max_improvements_per_cycle=max_improvements_per_cycle,
            min_cycle_interval_s=cycle_interval_s,
        )
        self._rollback_manager = RollbackManager(
            ihsan_floor=ihsan_floor,
            snr_floor=snr_floor,
        )
        self._approval_queue = HumanApprovalQueue()

        # State
        self._state = AutopoieticState.DORMANT
        self._running = False
        self._cycle_count = 0

        # History
        self._observation_history: Deque[SystemObservation] = deque(maxlen=100)
        self._hypothesis_history: List[Hypothesis] = []
        self._audit_log: List[AuditLogEntry] = []
        self._cycle_results: Deque[AutopoieticResult] = deque(maxlen=50)

        # Audit log persistence
        self._audit_log_path = audit_log_path or Path("/tmp/autopoietic_audit.jsonl")

        # Metrics
        self._total_improvements = 0
        self._total_rollbacks = 0
        self._last_improvement_time: Optional[datetime] = None

        logger.info(
            f"AutopoieticLoop initialized: ihsan_floor={ihsan_floor}, "
            f"snr_floor={snr_floor}, cycle_interval={cycle_interval_s}s"
        )

    @property
    def fate_gate(self):
        """Lazy-load Z3FATEGate if not provided."""
        if self._fate_gate is None:
            try:
                from core.sovereign.z3_fate_gate import Z3FATEGate
                self._fate_gate = Z3FATEGate()
            except ImportError:
                logger.warning("Z3FATEGate not available, using mock verification")
                self._fate_gate = MockFATEGate()
        return self._fate_gate

    @property
    def sensor_hub(self):
        """Lazy-load MuraqabahSensorHub if not provided."""
        if self._sensor_hub is None:
            try:
                from core.sovereign.muraqabah_sensors import MuraqabahSensorHub
                self._sensor_hub = MuraqabahSensorHub(snr_threshold=self.snr_floor)
            except ImportError:
                logger.warning("MuraqabahSensorHub not available, using mock sensors")
                self._sensor_hub = MockSensorHub()
        return self._sensor_hub

    # =========================================================================
    # ACTIVATION GUARDRAILS
    # =========================================================================

    def _is_mock_sensor_hub(self) -> bool:
        """Detect if the sensor hub is a mock implementation."""
        return isinstance(self.sensor_hub, MockSensorHub)

    def _is_mock_fate_gate(self) -> bool:
        """Detect if the FATE gate is a mock implementation."""
        return isinstance(self.fate_gate, MockFATEGate)

    async def check_activation(self) -> ActivationReport:
        """
        Evaluate activation guardrails using live metrics and FATE gate.

        Returns:
            ActivationReport with pass/fail reasons and observation snapshot
        """
        guard = self._activation_guardrails

        if not guard.enabled:
            report = ActivationReport(activated=True, reasons=["guardrails_disabled"])
            self._activation_report = report
            return report

        reasons: List[str] = []

        if guard.require_live_sensors and self._is_mock_sensor_hub() and not guard.allow_mock_sensors:
            reasons.append("mock_sensor_hub_detected")

        if guard.require_fate_gate and self._is_mock_fate_gate() and not guard.allow_mock_fate_gate:
            reasons.append("mock_fate_gate_detected")

        observation = await self.observe()
        if observation is None:
            reasons.append("observation_failed")
            return ActivationReport(activated=False, reasons=reasons)

        if len(self._observation_history) < guard.min_observations:
            reasons.append(
                f"insufficient_observations:{len(self._observation_history)}/{guard.min_observations}"
            )

        if observation.ihsan_score < guard.min_ihsan_score:
            reasons.append(f"ihsan_below_floor:{observation.ihsan_score:.3f}")
        if observation.snr_score < guard.min_snr_score:
            reasons.append(f"snr_below_floor:{observation.snr_score:.3f}")
        if observation.error_rate > guard.max_error_rate:
            reasons.append(f"error_rate_high:{observation.error_rate:.3f}")
        if observation.latency_p99_ms > guard.max_latency_p99_ms:
            reasons.append(f"latency_p99_high:{observation.latency_p99_ms:.1f}")
        if observation.throughput_qps < guard.min_throughput_qps:
            reasons.append(f"throughput_low:{observation.throughput_qps:.2f}")
        if guard.require_stable_or_improving and observation.trend_direction == "degrading":
            reasons.append("degrading_trend")
        if len(observation.anomalies_detected) > guard.max_anomalies:
            reasons.append("anomalies_detected")

        fate_proof_id: Optional[str] = None
        if guard.require_fate_gate and not (self._is_mock_fate_gate() and not guard.allow_mock_fate_gate):
            try:
                action_context = {
                    "ihsan": observation.ihsan_score,
                    "snr": observation.snr_score,
                    "risk_level": 0.1,
                    "reversible": True,
                    "human_approved": True,
                    "cost": 0.0,
                    "autonomy_limit": 1.0,
                }
                proof = self.fate_gate.generate_proof(action_context)
                fate_proof_id = getattr(proof, "proof_id", None)
                if not getattr(proof, "satisfiable", False):
                    reasons.append(f"fate_gate_failed:{getattr(proof, 'counterexample', '')}")
            except Exception as e:
                reasons.append(f"fate_gate_exception:{e}")

        activated = len(reasons) == 0
        report = ActivationReport(
            activated=activated,
            reasons=reasons,
            observation=observation,
            fate_proof_id=fate_proof_id,
        )
        self._activation_report = report
        return report

    async def activate(self, force: bool = False) -> ActivationReport:
        """
        Activate the autopoietic loop after guardrails pass.

        Args:
            force: If True, activate even when guardrails fail (logs reasons)
        """
        report = await self.check_activation()
        previous_activated = self._activated
        if report.activated or force:
            if force and not report.activated:
                report.reasons.append("forced_activation")
                report.activated = True
            self._activated = True
            outcome = "activated"
        else:
            self._activated = False
            self._activation_failures += 1
            outcome = "activation_blocked"

        # Audit log
        self._log_audit_entry(
            action="activate",
            hypothesis_id=None,
            component_affected="autopoiesis",
            change_type="activation_guardrails",
            before_state={"activated": previous_activated},
            after_state=report.to_dict(),
            verified_by="fate_gate" if report.fate_proof_id else "guardrails",
            outcome=outcome,
        )

        return report

    def is_active(self) -> bool:
        """Return True if autopoiesis is activated."""
        return self._activated

    # =========================================================================
    # CORE CYCLE
    # =========================================================================

    async def run_cycle(self) -> AutopoieticResult:
        """
        Run one complete autopoietic cycle.

        Phases:
            OBSERVE -> HYPOTHESIZE -> VALIDATE -> IMPLEMENT -> INTEGRATE -> REFLECT

        Returns:
            AutopoieticResult with full cycle outcome
        """
        cycle_id = f"cycle_{uuid.uuid4().hex[:8]}"
        start_time = time.perf_counter_ns()

        result = AutopoieticResult(
            cycle_id=cycle_id,
            state=AutopoieticState.DORMANT,
        )

        if self._activation_guardrails.enabled and not self._activated:
            result.state = AutopoieticState.HALTED
            return self._finalize_result(result, start_time)

        self._rate_limiter.start_new_cycle()
        self._cycle_count += 1

        try:
            # Phase 1: OBSERVE
            result.observation = await self.observe()
            result.state = AutopoieticState.OBSERVING

            if not result.observation:
                result.state = AutopoieticState.DORMANT
                return self._finalize_result(result, start_time)

            # Phase 2: HYPOTHESIZE
            hypotheses = await self.hypothesize(result.observation)
            result.hypotheses_generated = len(hypotheses)
            result.state = AutopoieticState.HYPOTHESIZING

            if not hypotheses:
                # No improvements needed
                result.state = AutopoieticState.REFLECTING
                await self._reflect_on_cycle(result)
                return self._finalize_result(result, start_time)

            # Rate limiting check
            can_proceed, reason = self._rate_limiter.can_attempt_improvement()
            if not can_proceed:
                logger.info(f"Rate limited: {reason}")
                result.state = AutopoieticState.REFLECTING
                await self._reflect_on_cycle(result)
                return self._finalize_result(result, start_time)

            # Select best hypothesis
            best_hypothesis = self._select_best_hypothesis(hypotheses)
            result.hypothesis_validated = best_hypothesis

            # Phase 3: VALIDATE
            validation = await self.validate(best_hypothesis)
            result.validation_result = validation
            result.state = AutopoieticState.VALIDATING

            if not validation.is_valid:
                logger.info(f"Hypothesis {best_hypothesis.id} failed validation: {validation.violations}")
                result.state = AutopoieticState.REFLECTING
                await self._reflect_on_cycle(result)
                return self._finalize_result(result, start_time)

            # Check human approval if needed
            if best_hypothesis.requires_human_approval:
                approval_status = self._approval_queue.check_approval(best_hypothesis.id)
                if approval_status is None:
                    # Submit for approval
                    self._approval_queue.submit_for_approval(best_hypothesis, result.observation)
                    result.human_approvals_pending = 1
                    result.state = AutopoieticState.HALTED
                    return self._finalize_result(result, start_time)
                elif not approval_status:
                    # Rejected
                    logger.info(f"Hypothesis {best_hypothesis.id} was rejected by human review")
                    result.state = AutopoieticState.REFLECTING
                    await self._reflect_on_cycle(result)
                    return self._finalize_result(result, start_time)

            # Phase 4: IMPLEMENT (shadow deployment)
            self._rate_limiter.record_improvement_attempt()
            implementation = await self.implement(best_hypothesis)
            result.implementation_result = implementation
            result.state = AutopoieticState.IMPLEMENTING
            result.improvements_attempted = 1

            if implementation.needs_rollback:
                await self._execute_rollback(best_hypothesis, implementation.rollback_reason or "Unknown")
                result.rollbacks_triggered = 1
                result.state = AutopoieticState.EMERGENCY_ROLLBACK
                self._rate_limiter.trigger_rollback_cooldown()
                return self._finalize_result(result, start_time)

            # Phase 5: INTEGRATE
            integration = await self.integrate(implementation)
            result.integration_result = integration
            result.state = AutopoieticState.INTEGRATING

            if integration.integrated:
                result.improvements_integrated = 1
                self._total_improvements += 1
                self._last_improvement_time = datetime.now(timezone.utc)

            # Phase 6: REFLECT
            result.state = AutopoieticState.REFLECTING
            await self._reflect_on_cycle(result)

        except Exception as e:
            logger.error(f"Autopoietic cycle error: {e}", exc_info=True)
            result.state = AutopoieticState.EMERGENCY_ROLLBACK

        return self._finalize_result(result, start_time)

    def _finalize_result(self, result: AutopoieticResult, start_time_ns: int) -> AutopoieticResult:
        """Finalize and store cycle result."""
        result.completed_at = datetime.now(timezone.utc)
        result.cycle_duration_ms = (time.perf_counter_ns() - start_time_ns) // 1_000_000

        self._cycle_results.append(result)

        # Return to dormant
        if result.state not in (AutopoieticState.HALTED, AutopoieticState.EMERGENCY_ROLLBACK):
            self._state = AutopoieticState.DORMANT
        else:
            self._state = result.state

        return result

    # =========================================================================
    # PHASE IMPLEMENTATIONS
    # =========================================================================

    async def observe(self) -> Optional[SystemObservation]:
        """
        OBSERVE phase: Collect all system metrics.

        Returns:
            SystemObservation snapshot or None if collection failed
        """
        self._state = AutopoieticState.OBSERVING
        observation_id = f"obs_{uuid.uuid4().hex[:8]}"

        try:
            # Collect sensor readings
            readings = await self.sensor_hub.poll_all_sensors()

            # Extract metrics from readings
            metrics = self._extract_metrics_from_readings(readings)

            # Detect trends
            trend = self._detect_trend()

            # Detect anomalies
            anomalies = self._detect_anomalies(metrics)

            observation = SystemObservation(
                observation_id=observation_id,
                timestamp=datetime.now(timezone.utc),
                ihsan_score=metrics.get("ihsan_score", 0.95),
                snr_score=metrics.get("snr_score", 0.90),
                latency_p50_ms=metrics.get("latency_p50_ms", 10.0),
                latency_p99_ms=metrics.get("latency_p99_ms", 50.0),
                error_rate=metrics.get("error_rate", 0.01),
                throughput_qps=metrics.get("throughput_qps", 100.0),
                cpu_usage=metrics.get("cpu_usage", 0.5),
                memory_usage=metrics.get("memory_usage", 0.6),
                gpu_usage=metrics.get("gpu_usage"),
                agent_performance=metrics.get("agent_performance", {}),
                constitutional_compliance=metrics.get("constitutional_compliance", 1.0),
                trend_direction=trend,
                anomalies_detected=anomalies,
                sensor_readings={r.sensor_id: r.value for r in readings} if readings else {},
            )

            self._observation_history.append(observation)

            logger.debug(
                f"Observation {observation_id}: ihsan={observation.ihsan_score:.3f}, "
                f"snr={observation.snr_score:.3f}, trend={trend}"
            )

            return observation

        except Exception as e:
            logger.error(f"Observation failed: {e}")
            return None

    async def hypothesize(self, observation: SystemObservation) -> List[Hypothesis]:
        """
        HYPOTHESIZE phase: Generate improvement candidates.

        Returns:
            List of hypotheses ordered by potential impact
        """
        self._state = AutopoieticState.HYPOTHESIZING
        hypotheses: List[Hypothesis] = []

        # Performance hypotheses based on latency
        if observation.latency_p99_ms > 100:
            hypotheses.append(Hypothesis(
                id=f"hyp_{uuid.uuid4().hex[:8]}",
                description="Optimize inference batching to reduce P99 latency",
                category=HypothesisCategory.PERFORMANCE,
                predicted_improvement=0.15,  # 15% improvement
                required_changes=[{
                    "component": "inference.gateway",
                    "parameter": "batch_size",
                    "action": "increase",
                    "from_value": 8,
                    "to_value": 16,
                }],
                affected_components=["inference.gateway"],
                risk_level=RiskLevel.LOW,
                reversibility_plan={
                    "action": "restore_parameter",
                    "parameter": "batch_size",
                    "restore_value": 8,
                },
                ihsan_impact_estimate=0.0,
                snr_impact_estimate=0.01,
                source_observation_id=observation.observation_id,
                confidence=0.7,
            ))

        # Quality hypotheses based on SNR
        if observation.snr_score < 0.92:
            hypotheses.append(Hypothesis(
                id=f"hyp_{uuid.uuid4().hex[:8]}",
                description="Increase SNR filtering threshold for embeddings",
                category=HypothesisCategory.QUALITY,
                predicted_improvement=0.08,
                required_changes=[{
                    "component": "vector_engine",
                    "parameter": "snr_filter_threshold",
                    "action": "increase",
                    "from_value": 0.80,
                    "to_value": 0.85,
                }],
                affected_components=["vector_engine"],
                risk_level=RiskLevel.LOW,
                reversibility_plan={
                    "action": "restore_parameter",
                    "parameter": "snr_filter_threshold",
                    "restore_value": 0.80,
                },
                ihsan_impact_estimate=0.02,
                snr_impact_estimate=0.05,
                source_observation_id=observation.observation_id,
                confidence=0.65,
            ))

        # Efficiency hypotheses based on resource usage
        if observation.memory_usage > 0.8:
            hypotheses.append(Hypothesis(
                id=f"hyp_{uuid.uuid4().hex[:8]}",
                description="Enable aggressive cache eviction to reduce memory pressure",
                category=HypothesisCategory.EFFICIENCY,
                predicted_improvement=0.10,
                required_changes=[{
                    "component": "cache_manager",
                    "parameter": "eviction_policy",
                    "action": "set",
                    "to_value": "lru_aggressive",
                }],
                affected_components=["cache_manager"],
                risk_level=RiskLevel.MODERATE,
                reversibility_plan={
                    "action": "restore_parameter",
                    "parameter": "eviction_policy",
                    "restore_value": "lru_standard",
                },
                ihsan_impact_estimate=-0.01,  # Slight quality impact
                snr_impact_estimate=0.0,
                source_observation_id=observation.observation_id,
                confidence=0.6,
            ))

        # Error rate hypotheses
        if observation.error_rate > 0.02:
            hypotheses.append(Hypothesis(
                id=f"hyp_{uuid.uuid4().hex[:8]}",
                description="Add retry logic with exponential backoff for transient failures",
                category=HypothesisCategory.ROBUSTNESS,
                predicted_improvement=0.20,
                required_changes=[{
                    "component": "http_client",
                    "parameter": "retry_config",
                    "action": "update",
                    "to_value": {"max_retries": 3, "backoff": "exponential"},
                }],
                affected_components=["http_client"],
                risk_level=RiskLevel.LOW,
                reversibility_plan={
                    "action": "restore_parameter",
                    "parameter": "retry_config",
                    "restore_value": {"max_retries": 0},
                },
                ihsan_impact_estimate=0.01,
                snr_impact_estimate=0.02,
                source_observation_id=observation.observation_id,
                confidence=0.75,
            ))

        # Store hypotheses for history
        self._hypothesis_history.extend(hypotheses)

        logger.info(f"Generated {len(hypotheses)} hypotheses from observation {observation.observation_id}")

        return hypotheses

    async def validate(self, hypothesis: Hypothesis) -> ValidationResult:
        """
        VALIDATE phase: Z3 + FATE gate verification.

        Returns:
            ValidationResult with verification outcome
        """
        self._state = AutopoieticState.VALIDATING

        violations: List[str] = []

        # Build action context for Z3 verification
        action_context = {
            "ihsan": self.ihsan_floor + hypothesis.ihsan_impact_estimate,
            "snr": self.snr_floor + hypothesis.snr_impact_estimate,
            "risk_level": hypothesis.risk_level.value / 5.0,  # Normalize to 0-1
            "reversible": bool(hypothesis.reversibility_plan),
            "human_approved": not hypothesis.requires_human_approval,
            "cost": hypothesis.predicted_improvement,  # Metaphorical cost
            "autonomy_limit": 0.3 if hypothesis.risk_level.value <= 2 else 0.1,
        }

        # Z3 FATE gate verification
        try:
            z3_proof = self.fate_gate.generate_proof(action_context)
            z3_satisfiable = z3_proof.satisfiable
            z3_proof_id = z3_proof.proof_id
            z3_time_ms = z3_proof.generation_time_ms

            if not z3_satisfiable:
                violations.append(f"Z3 FATE gate failed: {z3_proof.counterexample}")
        except Exception as e:
            logger.error(f"Z3 verification error: {e}")
            z3_satisfiable = False
            z3_proof_id = None
            z3_time_ms = 0
            violations.append(f"Z3 verification exception: {str(e)}")

        # Constitutional checks
        ihsan_gate_passed = (self.ihsan_floor + hypothesis.ihsan_impact_estimate) >= self.ihsan_floor
        snr_gate_passed = (self.snr_floor + hypothesis.snr_impact_estimate) >= self.snr_floor
        reversibility_verified = bool(hypothesis.reversibility_plan)

        if not ihsan_gate_passed:
            violations.append(f"Ihsan impact would drop below floor")
        if not snr_gate_passed:
            violations.append(f"SNR impact would drop below floor")
        if not reversibility_verified:
            violations.append("No reversibility plan provided")

        is_valid = z3_satisfiable and ihsan_gate_passed and snr_gate_passed and reversibility_verified

        recommendation = "proceed" if is_valid else ("review" if len(violations) == 1 else "reject")

        result = ValidationResult(
            hypothesis_id=hypothesis.id,
            is_valid=is_valid,
            z3_satisfiable=z3_satisfiable,
            z3_proof_id=z3_proof_id,
            z3_generation_time_ms=z3_time_ms,
            ihsan_gate_passed=ihsan_gate_passed,
            snr_gate_passed=snr_gate_passed,
            reversibility_verified=reversibility_verified,
            violations=violations,
            recommendation=recommendation,
        )

        # Audit log
        self._log_audit_entry(
            action="validate",
            hypothesis_id=hypothesis.id,
            component_affected="fate_gate",
            change_type="verification",
            before_state={"hypothesis": hypothesis.to_dict()},
            after_state={"validation": result.to_dict()},
            verified_by="z3_fate",
            outcome="success" if is_valid else "rejected",
        )

        logger.info(
            f"Validation for {hypothesis.id}: valid={is_valid}, "
            f"z3={z3_satisfiable}, violations={violations}"
        )

        return result

    async def implement(self, hypothesis: Hypothesis) -> ImplementationResult:
        """
        IMPLEMENT phase: Shadow deployment.

        Returns:
            ImplementationResult with deployment outcome
        """
        self._state = AutopoieticState.IMPLEMENTING
        start_time = time.perf_counter_ns()

        # Save state for potential rollback
        baseline_observation = self._observation_history[-1] if self._observation_history else None
        baseline_metrics = baseline_observation.to_dict() if baseline_observation else {}

        self._rollback_manager.save_state(baseline_metrics, hypothesis.id)

        try:
            # Simulate shadow deployment (in production, would deploy to shadow infra)
            shadow_instance_id = f"shadow_{uuid.uuid4().hex[:8]}"

            # Apply changes in shadow mode
            # (In a real implementation, this would modify actual system config)
            await asyncio.sleep(0.1)  # Simulate deployment time

            # Collect shadow metrics
            await asyncio.sleep(hypothesis.min_observation_duration_s / 100)  # Accelerated for demo

            # Simulate shadow metrics (in production, collect from shadow instance)
            shadow_metrics = {
                "ihsan_score": baseline_metrics.get("ihsan_score", 0.95) + hypothesis.ihsan_impact_estimate,
                "snr_score": baseline_metrics.get("snr_score", 0.90) + hypothesis.snr_impact_estimate,
                "latency_p50_ms": baseline_metrics.get("latency_p50_ms", 10.0) * (1 - hypothesis.predicted_improvement),
                "error_rate": baseline_metrics.get("error_rate", 0.01) * 0.9,
            }

            deployment_time_ms = (time.perf_counter_ns() - start_time) // 1_000_000

            # Calculate improvement
            improvement_observed = hypothesis.predicted_improvement * 0.8  # Conservative estimate

            # Check safety conditions
            current_ihsan = shadow_metrics.get("ihsan_score", 0.95)
            current_snr = shadow_metrics.get("snr_score", 0.90)
            current_error_rate = shadow_metrics.get("error_rate", 0.01)
            baseline_error_rate = baseline_metrics.get("error_rate", 0.01)

            needs_rollback, rollback_reason = self._rollback_manager.should_rollback(
                current_ihsan, current_snr, current_error_rate, baseline_error_rate
            )

            result = ImplementationResult(
                hypothesis_id=hypothesis.id,
                success=not needs_rollback,
                shadow_instance_id=shadow_instance_id,
                deployment_time_ms=deployment_time_ms,
                baseline_metrics=baseline_metrics,
                shadow_metrics=shadow_metrics,
                improvement_observed=improvement_observed,
                ihsan_maintained=current_ihsan >= self.ihsan_floor,
                snr_maintained=current_snr >= self.snr_floor,
                error_rate_acceptable=current_error_rate <= baseline_error_rate + 0.05,
                needs_rollback=needs_rollback,
                rollback_reason=rollback_reason if needs_rollback else None,
            )

            # Audit log
            self._log_audit_entry(
                action="implement",
                hypothesis_id=hypothesis.id,
                component_affected=hypothesis.affected_components[0] if hypothesis.affected_components else "unknown",
                change_type="shadow_deployment",
                before_state=baseline_metrics,
                after_state=shadow_metrics,
                verified_by="shadow_test",
                outcome="success" if result.success else "rollback_needed",
            )

            logger.info(
                f"Implementation for {hypothesis.id}: success={result.success}, "
                f"improvement={improvement_observed:.1%}"
            )

            return result

        except Exception as e:
            logger.error(f"Implementation failed: {e}")
            return ImplementationResult(
                hypothesis_id=hypothesis.id,
                success=False,
                needs_rollback=True,
                rollback_reason=str(e),
            )

    async def integrate(self, result: ImplementationResult) -> IntegrationResult:
        """
        INTEGRATE phase: Learn from outcome and consolidate.

        Returns:
            IntegrationResult with learning outcomes
        """
        self._state = AutopoieticState.INTEGRATING

        if not result.success or result.needs_rollback:
            # Failed implementation - learn from failure
            return IntegrationResult(
                hypothesis_id=result.hypothesis_id,
                integrated=False,
                lesson_learned=f"Implementation failed: {result.rollback_reason}",
                final_improvement=0.0,
            )

        # Successful implementation
        ihsan_delta = result.shadow_metrics.get("ihsan_score", 0.95) - result.baseline_metrics.get("ihsan_score", 0.95)
        snr_delta = result.shadow_metrics.get("snr_score", 0.90) - result.baseline_metrics.get("snr_score", 0.90)

        # Archive the pattern for future reference
        pattern_id = hashlib.md5(json.dumps(result.shadow_metrics, sort_keys=True).encode()).hexdigest()[:12]

        integration_result = IntegrationResult(
            hypothesis_id=result.hypothesis_id,
            integrated=True,
            lesson_learned=f"Improvement of {result.improvement_observed:.1%} achieved with stable quality",
            parameters_updated={"shadow_metrics": result.shadow_metrics},
            final_improvement=result.improvement_observed,
            ihsan_delta=ihsan_delta,
            snr_delta=snr_delta,
            pattern_id=pattern_id,
            archived=True,
        )

        # Audit log
        self._log_audit_entry(
            action="integrate",
            hypothesis_id=result.hypothesis_id,
            component_affected="knowledge_base",
            change_type="pattern_archive",
            before_state=result.baseline_metrics,
            after_state=result.shadow_metrics,
            verified_by="integration_engine",
            outcome="success",
        )

        logger.info(
            f"Integration for {result.hypothesis_id}: integrated=True, "
            f"improvement={result.improvement_observed:.1%}, pattern={pattern_id}"
        )

        return integration_result

    # =========================================================================
    # HELPER METHODS
    # =========================================================================

    def _select_best_hypothesis(self, hypotheses: List[Hypothesis]) -> Hypothesis:
        """Select the best hypothesis based on expected value and risk."""
        def score(h: Hypothesis) -> float:
            # Higher improvement, lower risk, higher confidence = better
            risk_penalty = h.risk_level.value * 0.1
            return h.predicted_improvement * h.confidence - risk_penalty

        return max(hypotheses, key=score)

    def _extract_metrics_from_readings(self, readings: List[Any]) -> Dict[str, Any]:
        """Extract metrics from sensor readings."""
        metrics = {
            "ihsan_score": 0.95,
            "snr_score": 0.90,
            "latency_p50_ms": 10.0,
            "latency_p99_ms": 50.0,
            "error_rate": 0.01,
            "throughput_qps": 100.0,
            "cpu_usage": 0.5,
            "memory_usage": 0.6,
            "constitutional_compliance": 1.0,
            "agent_performance": {},
        }

        if readings:
            for reading in readings:
                sensor_id = getattr(reading, 'sensor_id', '')
                value = getattr(reading, 'value', 0)

                if 'ihsan' in sensor_id.lower():
                    metrics["ihsan_score"] = value
                elif 'snr' in sensor_id.lower():
                    metrics["snr_score"] = value
                elif 'latency' in sensor_id.lower():
                    metrics["latency_p50_ms"] = value
                elif 'cpu' in sensor_id.lower():
                    metrics["cpu_usage"] = value
                elif 'memory' in sensor_id.lower():
                    metrics["memory_usage"] = value

        return metrics

    def _detect_trend(self) -> str:
        """Detect trend from observation history."""
        if len(self._observation_history) < 3:
            return "stable"

        recent = list(self._observation_history)[-3:]
        ihsan_values = [o.ihsan_score for o in recent]

        if ihsan_values[-1] > ihsan_values[0] + 0.01:
            return "improving"
        elif ihsan_values[-1] < ihsan_values[0] - 0.01:
            return "degrading"
        return "stable"

    def _detect_anomalies(self, metrics: Dict[str, Any]) -> List[str]:
        """Detect anomalies in current metrics."""
        anomalies = []

        if metrics.get("error_rate", 0) > 0.05:
            anomalies.append("high_error_rate")
        if metrics.get("latency_p99_ms", 0) > 200:
            anomalies.append("high_latency")
        if metrics.get("memory_usage", 0) > 0.9:
            anomalies.append("memory_pressure")
        if metrics.get("ihsan_score", 1.0) < self.ihsan_floor:
            anomalies.append("ihsan_below_floor")
        if metrics.get("snr_score", 1.0) < self.snr_floor:
            anomalies.append("snr_below_floor")

        return anomalies

    async def _execute_rollback(self, hypothesis: Hypothesis, reason: str):
        """Execute rollback of a hypothesis."""
        self._state = AutopoieticState.EMERGENCY_ROLLBACK

        saved_state = self._rollback_manager.get_rollback_state(hypothesis.id)

        if saved_state:
            # In production, would apply saved_state to restore system
            logger.warning(f"Rolling back hypothesis {hypothesis.id}: {reason}")
            self._rollback_manager.record_rollback(hypothesis.id, reason)
            self._total_rollbacks += 1

            # Audit log
            self._log_audit_entry(
                action="rollback",
                hypothesis_id=hypothesis.id,
                component_affected=hypothesis.affected_components[0] if hypothesis.affected_components else "unknown",
                change_type="emergency_rollback",
                before_state={"hypothesis": hypothesis.to_dict()},
                after_state=saved_state,
                verified_by="rollback_manager",
                outcome="rollback",
                rollback_triggered=True,
            )

    async def _reflect_on_cycle(self, result: AutopoieticResult):
        """Reflect on cycle performance and adjust parameters."""
        # Analyze cycle outcome
        if result.improvements_integrated > 0:
            logger.info(f"Cycle {result.cycle_id} successful: {result.improvements_integrated} improvements integrated")
        elif result.rollbacks_triggered > 0:
            logger.warning(f"Cycle {result.cycle_id} had {result.rollbacks_triggered} rollbacks")
        else:
            logger.debug(f"Cycle {result.cycle_id} completed with no changes")

    def _log_audit_entry(
        self,
        action: str,
        hypothesis_id: Optional[str],
        component_affected: str,
        change_type: str,
        before_state: Dict[str, Any],
        after_state: Dict[str, Any],
        verified_by: str,
        outcome: str,
        rollback_triggered: bool = False,
    ):
        """Log an audit entry."""
        entry = AuditLogEntry(
            entry_id=f"audit_{uuid.uuid4().hex[:8]}",
            timestamp=datetime.now(timezone.utc),
            action=action,
            hypothesis_id=hypothesis_id,
            component_affected=component_affected,
            change_type=change_type,
            before_state=before_state,
            after_state=after_state,
            verified_by=verified_by,
            ihsan_score_at_action=self._observation_history[-1].ihsan_score if self._observation_history else 0.95,
            snr_score_at_action=self._observation_history[-1].snr_score if self._observation_history else 0.90,
            outcome=outcome,
            rollback_triggered=rollback_triggered,
        )

        self._audit_log.append(entry)

        # Persist to file
        try:
            with open(self._audit_log_path, 'a') as f:
                f.write(json.dumps(entry.to_dict()) + "\n")
        except Exception as e:
            logger.error(f"Failed to persist audit log: {e}")

    # =========================================================================
    # CONTINUOUS OPERATION
    # =========================================================================

    async def start(self):
        """Start continuous autopoietic improvement loop."""
        logger.info("Starting autopoietic loop")

        activation_report = await self.activate()
        if not activation_report.activated:
            logger.warning(
                "Autopoietic activation blocked: %s",
                ", ".join(activation_report.reasons) or "unknown",
            )
            self._state = AutopoieticState.HALTED
            self._running = False
            return

        self._running = True
        self._state = AutopoieticState.OBSERVING

        while self._running:
            try:
                result = await self.run_cycle()

                if result.state == AutopoieticState.HALTED:
                    logger.warning("Loop halted - human intervention required")
                    break

            except Exception as e:
                logger.error(f"Autopoietic loop error: {e}", exc_info=True)
                self._state = AutopoieticState.EMERGENCY_ROLLBACK

            await asyncio.sleep(self.cycle_interval_s)

        self._state = AutopoieticState.DORMANT
        logger.info("Autopoietic loop stopped")

    async def stop(self):
        """Stop the autopoietic loop."""
        self._running = False
        self._state = AutopoieticState.DORMANT

    def pause(self):
        """Pause the loop (maintains state)."""
        self._running = False

    def resume(self):
        """Resume the loop."""
        self._running = True

    # =========================================================================
    # STATUS AND METRICS
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """Get current status of the autopoietic loop."""
        return {
            "state": self._state.value,
            "running": self._running,
            "activated": self._activated,
            "activation_failures": self._activation_failures,
            "activation_report": self._activation_report.to_dict() if self._activation_report else None,
            "cycle_count": self._cycle_count,
            "ihsan_floor": self.ihsan_floor,
            "snr_floor": self.snr_floor,
            "total_improvements": self._total_improvements,
            "total_rollbacks": self._total_rollbacks,
            "last_improvement": self._last_improvement_time.isoformat() if self._last_improvement_time else None,
            "pending_approvals": len(self._approval_queue.get_pending()),
            "observation_history_size": len(self._observation_history),
            "audit_log_size": len(self._audit_log),
        }

    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """Get pending human approval requests."""
        return [r.to_dict() for r in self._approval_queue.get_pending()]

    def approve_hypothesis(self, request_id: str, approver: str) -> bool:
        """Approve a pending hypothesis."""
        return self._approval_queue.approve(request_id, approver)

    def reject_hypothesis(self, request_id: str, approver: str, reason: str) -> bool:
        """Reject a pending hypothesis."""
        return self._approval_queue.reject(request_id, approver, reason)

    def get_audit_log(self, limit: int = 100) -> List[Dict[str, Any]]:
        """Get recent audit log entries."""
        return [e.to_dict() for e in self._audit_log[-limit:]]

    def get_cycle_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent cycle results."""
        return [r.to_dict() for r in list(self._cycle_results)[-limit:]]


# =============================================================================
# MOCK IMPLEMENTATIONS FOR TESTING
# =============================================================================


class MockFATEGate:
    """Mock FATE gate for testing without Z3."""

    def generate_proof(self, action_context: Dict[str, Any]) -> Any:
        """Generate a mock proof."""
        from dataclasses import dataclass

        @dataclass
        class MockProof:
            proof_id: str = "mock_proof_001"
            satisfiable: bool = True
            generation_time_ms: int = 1
            counterexample: Optional[str] = None

        # Simulate validation
        satisfiable = (
            action_context.get("ihsan", 0) >= UNIFIED_IHSAN_THRESHOLD and
            action_context.get("snr", 0) >= UNIFIED_SNR_THRESHOLD
        )

        return MockProof(satisfiable=satisfiable)


class MockSensorHub:
    """Mock sensor hub for testing."""

    async def poll_all_sensors(self) -> List[Any]:
        """Return mock sensor readings."""
        from dataclasses import dataclass

        @dataclass
        class MockReading:
            sensor_id: str
            value: float
            snr_score: float

        return [
            MockReading("ihsan_compliance", 0.96, 0.95),
            MockReading("snr_quality", 0.91, 0.90),
            MockReading("latency_ms", 15.0, 0.88),
            MockReading("cpu_usage", 0.45, 0.92),
            MockReading("memory_usage", 0.55, 0.90),
        ]


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_autopoietic_loop(
    ihsan_floor: float = UNIFIED_IHSAN_THRESHOLD,
    snr_floor: float = UNIFIED_SNR_THRESHOLD,
    cycle_interval_s: float = 60.0,
    fate_gate: Optional[Any] = None,
    activation_guardrails: Optional[ActivationGuardrails] = None,
) -> AutopoieticLoop:
    """
    Factory function to create an AutopoieticLoop.

    Args:
        ihsan_floor: Minimum Ihsan threshold (default: 0.95)
        snr_floor: Minimum SNR threshold (default: 0.85)
        cycle_interval_s: Seconds between cycles (default: 60)
        fate_gate: Optional Z3FATEGate instance

    Returns:
        Configured AutopoieticLoop instance
    """
    return AutopoieticLoop(
        fate_gate=fate_gate,
        ihsan_floor=ihsan_floor,
        snr_floor=snr_floor,
        cycle_interval_s=cycle_interval_s,
        activation_guardrails=activation_guardrails,
    )


# =============================================================================
# EXPORTS
# =============================================================================


__all__ = [
    # State enum
    "AutopoieticState",
    "HypothesisCategory",
    "RiskLevel",
    # Data classes
    "SystemObservation",
    "ActivationGuardrails",
    "ActivationReport",
    "Hypothesis",
    "ValidationResult",
    "ImplementationResult",
    "IntegrationResult",
    "AutopoieticResult",
    "AuditLogEntry",
    "ApprovalRequest",
    # Components
    "RateLimiter",
    "RollbackManager",
    "HumanApprovalQueue",
    # Main class
    "AutopoieticLoop",
    # Factory
    "create_autopoietic_loop",
]
