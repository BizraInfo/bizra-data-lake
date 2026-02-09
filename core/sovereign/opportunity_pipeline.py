"""
Opportunity Pipeline — BIZRA Proactive Sovereign Entity
========================================================
The "nervous system" that connects detection to action.

Architecture:
    MuraqabahEngine → OpportunityPipeline → BackgroundAgents → AutonomyMatrix → Execution

Standing on Giants:
    - Lamport: Event ordering and distributed coordination
    - Al-Ghazali: Muraqabah (continuous vigilance)
    - Boyd: OODA loop decision cycle
    - Shannon: Signal-to-noise ratio filtering
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Deque, Dict, List, Optional, Tuple

# Core sovereign imports
from .autonomy_matrix import AutonomyLevel
from .event_bus import Event, EventBus

logger = logging.getLogger(__name__)


# =============================================================================
# PIPELINE STAGE DEFINITIONS
# =============================================================================


class PipelineStage(Enum):
    """Stages in the opportunity processing pipeline."""

    DETECTION = auto()  # Muraqabah detected opportunity
    ENRICHMENT = auto()  # Add context from knowledge base
    FILTERING = auto()  # SNR and constitutional filtering
    PLANNING = auto()  # Background agent creates action plan
    APPROVAL = auto()  # Autonomy matrix decision
    EXECUTION = auto()  # Action execution
    REFLECTION = auto()  # Outcome recording and learning


class OpportunityStatus(Enum):
    """Status of an opportunity in the pipeline."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXECUTED = "executed"
    FAILED = "failed"
    DEFERRED = "deferred"


@dataclass
class PipelineOpportunity:
    """An opportunity flowing through the pipeline."""

    id: str
    domain: str
    description: str
    source: str  # Which sensor/agent detected it
    detected_at: float

    # Scores
    snr_score: float = 0.0
    ihsan_score: float = 0.0
    urgency: float = 0.5
    estimated_value: float = 0.0

    # Pipeline state
    stage: PipelineStage = PipelineStage.DETECTION
    status: OpportunityStatus = OpportunityStatus.PENDING
    autonomy_level: AutonomyLevel = AutonomyLevel.OBSERVER

    # Enrichment data
    context: Dict[str, Any] = field(default_factory=dict)
    knowledge_refs: List[str] = field(default_factory=list)

    # Action plan (filled by background agent)
    action_plan: Optional[Dict[str, Any]] = None
    assigned_agent: Optional[str] = None

    # Execution results
    execution_result: Optional[Dict[str, Any]] = None
    executed_at: Optional[float] = None

    # Audit trail
    stage_history: List[Tuple[PipelineStage, float, str]] = field(default_factory=list)
    rejection_reason: Optional[str] = None

    def advance_stage(self, new_stage: PipelineStage, note: str = "") -> None:
        """Record stage transition."""
        self.stage_history.append((self.stage, time.time(), note))
        self.stage = new_stage

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for events and logging."""
        return {
            "id": self.id,
            "domain": self.domain,
            "description": self.description,
            "source": self.source,
            "snr_score": self.snr_score,
            "ihsan_score": self.ihsan_score,
            "urgency": self.urgency,
            "stage": self.stage.name,
            "status": self.status.value,
            "autonomy_level": self.autonomy_level.name,
            "assigned_agent": self.assigned_agent,
        }


# =============================================================================
# PIPELINE FILTERS (Constitutional Guardrails)
# =============================================================================


@dataclass
class FilterResult:
    """Result of a filter check."""

    passed: bool
    reason: str = ""
    score_adjustment: float = 0.0


class ConstitutionalFilter:
    """Base class for pipeline filters."""

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight

    async def check(self, opportunity: PipelineOpportunity) -> FilterResult:
        """Check if opportunity passes this filter."""
        raise NotImplementedError


class SNRFilter(ConstitutionalFilter):
    """Filter opportunities by signal-to-noise ratio."""

    def __init__(self, min_snr: float = 0.85):
        super().__init__("SNR Filter", weight=1.5)
        self.min_snr = min_snr

    async def check(self, opportunity: PipelineOpportunity) -> FilterResult:
        if opportunity.snr_score >= self.min_snr:
            return FilterResult(
                passed=True, reason=f"SNR {opportunity.snr_score:.2f} >= {self.min_snr}"
            )
        return FilterResult(
            passed=False,
            reason=f"SNR {opportunity.snr_score:.2f} < threshold {self.min_snr}",
        )


class IhsanFilter(ConstitutionalFilter):
    """Filter opportunities by Ihsan (excellence) threshold per autonomy level."""

    # Ihsan thresholds per autonomy level (from architecture)
    THRESHOLDS = {
        AutonomyLevel.OBSERVER: 0.0,
        AutonomyLevel.SUGGESTER: 0.95,
        AutonomyLevel.AUTOLOW: 0.97,
        AutonomyLevel.AUTOMEDIUM: 0.98,
        AutonomyLevel.AUTOHIGH: 0.99,
        AutonomyLevel.SOVEREIGN: 1.0,
    }

    def __init__(self):
        super().__init__("Ihsan Filter", weight=2.0)

    async def check(self, opportunity: PipelineOpportunity) -> FilterResult:
        threshold = self.THRESHOLDS.get(opportunity.autonomy_level, 0.95)
        if opportunity.ihsan_score >= threshold:
            return FilterResult(
                passed=True,
                reason=f"Ihsan {opportunity.ihsan_score:.3f} >= {threshold} for {opportunity.autonomy_level.name}",
            )
        return FilterResult(
            passed=False,
            reason=f"Ihsan {opportunity.ihsan_score:.3f} < {threshold} required for {opportunity.autonomy_level.name}",
        )


class DaughterTestFilter(ConstitutionalFilter):
    """
    The 'Daughter Test' — Would you want this action taken for your daughter?
    Implements hard safety constraints.
    """

    # Actions that always require human approval
    SENSITIVE_DOMAINS = {"health", "financial", "legal", "personal_relationships"}
    SENSITIVE_KEYWORDS = {"delete", "cancel", "terminate", "irreversible", "emergency"}

    def __init__(self):
        super().__init__("Daughter Test", weight=3.0)

    async def check(self, opportunity: PipelineOpportunity) -> FilterResult:
        # Check sensitive domains
        if opportunity.domain in self.SENSITIVE_DOMAINS:
            if opportunity.autonomy_level.value > AutonomyLevel.SUGGESTER.value:
                return FilterResult(
                    passed=False,
                    reason=f"Domain '{opportunity.domain}' requires human approval (Daughter Test)",
                )

        # Check sensitive keywords in description
        desc_lower = opportunity.description.lower()
        for keyword in self.SENSITIVE_KEYWORDS:
            if keyword in desc_lower:
                if opportunity.autonomy_level.value > AutonomyLevel.AUTOLOW.value:
                    return FilterResult(
                        passed=False,
                        reason=f"Keyword '{keyword}' requires elevated approval (Daughter Test)",
                    )

        return FilterResult(passed=True, reason="Passed Daughter Test")


class RateLimitFilter(ConstitutionalFilter):
    """Prevent action flooding — limit actions per domain per time window."""

    def __init__(self, max_per_hour: int = 10, max_per_day: int = 50):
        super().__init__("Rate Limit", weight=1.0)
        self.max_per_hour = max_per_hour
        self.max_per_day = max_per_day
        self._hourly_counts: Dict[str, int] = {}
        self._daily_counts: Dict[str, int] = {}
        self._last_reset_hour = time.time()
        self._last_reset_day = time.time()

    async def check(self, opportunity: PipelineOpportunity) -> FilterResult:
        now = time.time()

        # Reset hourly counts
        if now - self._last_reset_hour > 3600:
            self._hourly_counts.clear()
            self._last_reset_hour = now

        # Reset daily counts
        if now - self._last_reset_day > 86400:
            self._daily_counts.clear()
            self._last_reset_day = now

        domain = opportunity.domain
        hourly = self._hourly_counts.get(domain, 0)
        daily = self._daily_counts.get(domain, 0)

        if hourly >= self.max_per_hour:
            return FilterResult(
                passed=False, reason=f"Hourly limit reached for {domain}"
            )
        if daily >= self.max_per_day:
            return FilterResult(
                passed=False, reason=f"Daily limit reached for {domain}"
            )

        return FilterResult(passed=True)

    def record_action(self, domain: str) -> None:
        """Record an executed action for rate limiting."""
        self._hourly_counts[domain] = self._hourly_counts.get(domain, 0) + 1
        self._daily_counts[domain] = self._daily_counts.get(domain, 0) + 1


# =============================================================================
# OPPORTUNITY PIPELINE
# =============================================================================


class OpportunityPipeline:
    """
    The nervous system connecting Muraqabah to execution.

    Flow:
        1. DETECTION: Receive opportunity from MuraqabahEngine
        2. ENRICHMENT: Add context from knowledge base
        3. FILTERING: Apply constitutional guardrails
        4. PLANNING: Background agent creates action plan
        5. APPROVAL: Autonomy matrix decides execution path
        6. EXECUTION: Execute action (auto or with approval)
        7. REFLECTION: Record outcome and learn
    """

    def __init__(
        self,
        event_bus: Optional[EventBus] = None,
        snr_threshold: float = 0.85,
        ihsan_threshold: float = 0.95,
        max_concurrent: int = 10,
    ):
        self.event_bus = event_bus
        self.snr_threshold = snr_threshold
        self.ihsan_threshold = ihsan_threshold
        self.max_concurrent = max_concurrent

        # Pipeline state
        self._running = False
        self._queue: asyncio.Queue[PipelineOpportunity] = asyncio.Queue(maxsize=1000)
        self._active: Dict[str, PipelineOpportunity] = {}
        # PERF FIX: Use deque with maxlen for O(1) bounded storage
        self._completed: Deque[PipelineOpportunity] = deque(maxlen=1000)
        self._pending_approval: Dict[str, PipelineOpportunity] = {}

        # Constitutional filters
        self._filters: List[ConstitutionalFilter] = [
            SNRFilter(min_snr=snr_threshold),
            IhsanFilter(),
            DaughterTestFilter(),
            RateLimitFilter(),
        ]

        # Callbacks for external integration
        self._enrichment_callback: Optional[Callable] = None
        self._planning_callback: Optional[Callable] = None
        self._execution_callback: Optional[Callable] = None
        self._approval_callback: Optional[Callable] = None

        # Metrics
        self._metrics: Dict[str, Any] = {
            "total_received": 0,
            "total_filtered": 0,
            "total_approved": 0,
            "total_rejected": 0,
            "total_executed": 0,
            "total_failed": 0,
            "total_deferred": 0,
            "dropped_opportunities": 0,
            "by_domain": {},
            "by_autonomy": {},
        }

        logger.info(
            "OpportunityPipeline initialized with SNR=%.2f, Ihsan=%.2f",
            snr_threshold,
            ihsan_threshold,
        )

    # -------------------------------------------------------------------------
    # LIFECYCLE
    # -------------------------------------------------------------------------

    async def start(self) -> None:
        """Start the pipeline processing loop."""
        if self._running:
            return

        self._running = True
        logger.info("OpportunityPipeline started")

        # Start worker tasks
        asyncio.create_task(self._process_loop())

        if self.event_bus:
            await self.event_bus.publish(
                Event(topic="pipeline.started", payload={"timestamp": time.time()})
            )

    async def stop(self) -> None:
        """Stop the pipeline gracefully."""
        self._running = False
        logger.info("OpportunityPipeline stopping...")

        # Wait for active items to complete (with timeout)
        timeout = 30
        start = time.time()
        while self._active and (time.time() - start) < timeout:
            await asyncio.sleep(0.5)

        if self.event_bus:
            await self.event_bus.publish(
                Event(
                    topic="pipeline.stopped",
                    payload={"timestamp": time.time(), "metrics": self._metrics},
                )
            )

        logger.info("OpportunityPipeline stopped")

    # -------------------------------------------------------------------------
    # INTAKE
    # -------------------------------------------------------------------------

    async def submit(self, opportunity: PipelineOpportunity) -> str:
        """Submit an opportunity to the pipeline.

        Raises:
            asyncio.QueueFull: If the bounded queue (maxsize=1000) is at capacity.
        """
        self._metrics["total_received"] += 1

        # Track by domain
        domain = opportunity.domain
        if domain not in self._metrics["by_domain"]:
            self._metrics["by_domain"][domain] = {"received": 0, "executed": 0}
        self._metrics["by_domain"][domain]["received"] += 1

        try:
            self._queue.put_nowait(opportunity)
        except asyncio.QueueFull:
            self._metrics["dropped_opportunities"] += 1
            logger.warning(
                "Queue full (%d/%d) — dropped opportunity %s [domain=%s]",
                self._queue.qsize(),
                self._queue.maxsize,
                opportunity.id,
                domain,
            )
            raise

        if self.event_bus:
            await self.event_bus.publish(
                Event(
                    topic="pipeline.opportunity.received", payload=opportunity.to_dict()
                )
            )

        logger.debug("Opportunity %s submitted to pipeline", opportunity.id)
        return opportunity.id

    async def submit_from_muraqabah(
        self,
        domain: str,
        description: str,
        source: str,
        snr_score: float,
        urgency: float = 0.5,
        estimated_value: float = 0.0,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Convenience method to submit from MuraqabahEngine."""
        opportunity = PipelineOpportunity(
            id=f"opp-{uuid.uuid4().hex[:12]}",
            domain=domain,
            description=description,
            source=source,
            detected_at=time.time(),
            snr_score=snr_score,
            urgency=urgency,
            estimated_value=estimated_value,
            context=context or {},
        )
        return await self.submit(opportunity)

    # -------------------------------------------------------------------------
    # PROCESSING LOOP
    # -------------------------------------------------------------------------

    async def _process_loop(self) -> None:
        """Main processing loop."""
        while self._running:
            try:
                # Respect concurrency limit
                while len(self._active) >= self.max_concurrent:
                    await asyncio.sleep(0.1)

                # Get next opportunity (with timeout to check running state)
                try:
                    opportunity = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

                # Process in background task
                self._active[opportunity.id] = opportunity
                asyncio.create_task(self._process_opportunity(opportunity))

            except Exception as e:
                logger.error("Pipeline loop error: %s", e)
                await asyncio.sleep(1.0)

    async def _process_opportunity(self, opp: PipelineOpportunity) -> None:
        """Process a single opportunity through all stages."""
        try:
            # Stage 1: ENRICHMENT
            opp.advance_stage(PipelineStage.ENRICHMENT, "Starting enrichment")
            await self._stage_enrichment(opp)

            # Stage 2: FILTERING
            opp.advance_stage(
                PipelineStage.FILTERING, "Applying constitutional filters"
            )
            passed = await self._stage_filtering(opp)
            if not passed:
                self._finalize(opp, OpportunityStatus.REJECTED)
                return

            # Stage 3: PLANNING
            opp.advance_stage(PipelineStage.PLANNING, "Creating action plan")
            await self._stage_planning(opp)

            # Stage 4: APPROVAL
            opp.advance_stage(PipelineStage.APPROVAL, "Checking autonomy level")
            approved = await self._stage_approval(opp)
            if not approved:
                # Deferred for human approval or rejected
                return

            # Stage 5: EXECUTION
            opp.advance_stage(PipelineStage.EXECUTION, "Executing action")
            success = await self._stage_execution(opp)

            # Stage 6: REFLECTION
            opp.advance_stage(PipelineStage.REFLECTION, "Recording outcome")
            await self._stage_reflection(opp, success)

            self._finalize(
                opp, OpportunityStatus.EXECUTED if success else OpportunityStatus.FAILED
            )

        except Exception as e:
            logger.error("Error processing opportunity %s: %s", opp.id, e)
            opp.rejection_reason = str(e)
            self._finalize(opp, OpportunityStatus.FAILED)

    # -------------------------------------------------------------------------
    # PIPELINE STAGES
    # -------------------------------------------------------------------------

    async def _stage_enrichment(self, opp: PipelineOpportunity) -> None:
        """Enrich opportunity with context from knowledge base."""
        if self._enrichment_callback:
            try:
                enrichment = await self._enrichment_callback(opp)
                if enrichment:
                    opp.context.update(enrichment.get("context", {}))
                    opp.knowledge_refs.extend(enrichment.get("refs", []))
                    # May adjust scores based on knowledge
                    if "ihsan_adjustment" in enrichment:
                        opp.ihsan_score = enrichment["ihsan_adjustment"]
            except Exception as e:
                logger.warning("Enrichment callback failed: %s", e)

        # Default Ihsan score if not set
        if opp.ihsan_score == 0.0:
            opp.ihsan_score = min(0.95, opp.snr_score * 1.05)

    async def _stage_filtering(self, opp: PipelineOpportunity) -> bool:
        """Apply constitutional filters."""
        # PERF FIX #6: Run filters in parallel instead of sequentially
        # Each filter is independent, so we can check them concurrently
        filter_tasks = [f.check(opp) for f in self._filters]
        results = await asyncio.gather(*filter_tasks, return_exceptions=True)

        all_passed = True
        filter_notes = []

        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error("Filter %s error: %s", self._filters[i].name, result)
                all_passed = False
                filter_notes.append(f"{self._filters[i].name}: Error - {result}")
            elif not result.passed:
                all_passed = False
                filter_notes.append(f"{self._filters[i].name}: {result.reason}")
                logger.info(
                    "Opportunity %s filtered by %s: %s",
                    opp.id,
                    self._filters[i].name,
                    result.reason,
                )

        if not all_passed:
            opp.rejection_reason = "; ".join(filter_notes)
            self._metrics["total_filtered"] += 1

        return all_passed

    async def _stage_planning(self, opp: PipelineOpportunity) -> None:
        """Create action plan via background agent."""
        if self._planning_callback:
            try:
                plan = await self._planning_callback(opp)
                if plan:
                    opp.action_plan = plan.get("plan")
                    opp.assigned_agent = plan.get("agent")
                    # Agent may recommend autonomy level
                    if "recommended_autonomy" in plan:
                        opp.autonomy_level = AutonomyLevel[plan["recommended_autonomy"]]
            except Exception as e:
                logger.warning("Planning callback failed: %s", e)

        # Default plan if none provided
        if not opp.action_plan:
            opp.action_plan = {
                "steps": [
                    {"action": "notify_user", "params": {"message": opp.description}}
                ],
                "reversible": True,
                "estimated_duration": 60,
            }

    async def _stage_approval(self, opp: PipelineOpportunity) -> bool:
        """Check autonomy level and get approval if needed."""
        level = opp.autonomy_level

        # Track by autonomy level
        level_name = level.name
        if level_name not in self._metrics["by_autonomy"]:
            self._metrics["by_autonomy"][level_name] = 0
        self._metrics["by_autonomy"][level_name] += 1

        # OBSERVER and SUGGESTER always need human approval
        if level in (AutonomyLevel.OBSERVER, AutonomyLevel.SUGGESTER):
            return await self._request_human_approval(opp)

        # AUTOLOW/AUTOMEDIUM/AUTOHIGH - auto-approve within Ihsan thresholds
        # (already checked in filtering stage)
        if level in (
            AutonomyLevel.AUTOLOW,
            AutonomyLevel.AUTOMEDIUM,
            AutonomyLevel.AUTOHIGH,
        ):
            self._metrics["total_approved"] += 1
            opp.status = OpportunityStatus.APPROVED
            return True

        # SOVEREIGN - full agency (rare, emergencies only)
        if level == AutonomyLevel.SOVEREIGN:
            if opp.ihsan_score >= 1.0:
                self._metrics["total_approved"] += 1
                opp.status = OpportunityStatus.APPROVED
                return True
            else:
                return await self._request_human_approval(opp)

        return False

    async def _request_human_approval(self, opp: PipelineOpportunity) -> bool:
        """Queue opportunity for human approval."""
        opp.status = OpportunityStatus.DEFERRED
        self._pending_approval[opp.id] = opp
        self._metrics["total_deferred"] += 1

        if self.event_bus:
            await self.event_bus.publish(
                Event(topic="pipeline.approval.requested", payload=opp.to_dict())
            )

        # If approval callback provided, use it
        if self._approval_callback:
            try:
                approved = await self._approval_callback(opp)
                if approved:
                    return await self.approve(opp.id)
                else:
                    await self.reject(opp.id, "User rejected")
                    return False
            except Exception as e:
                logger.warning("Approval callback failed: %s", e)

        logger.info("Opportunity %s queued for human approval", opp.id)
        return False  # Will be processed when human approves

    async def _stage_execution(self, opp: PipelineOpportunity) -> bool:
        """Execute the action plan."""
        if self._execution_callback:
            try:
                result = await self._execution_callback(opp)
                opp.execution_result = result
                opp.executed_at = time.time()

                # Record for rate limiting
                for f in self._filters:
                    if isinstance(f, RateLimitFilter):
                        f.record_action(opp.domain)

                self._metrics["total_executed"] += 1
                if opp.domain in self._metrics["by_domain"]:
                    self._metrics["by_domain"][opp.domain]["executed"] += 1

                return result.get("success", True)
            except Exception as e:
                logger.error("Execution failed for %s: %s", opp.id, e)
                opp.execution_result = {"success": False, "error": str(e)}
                self._metrics["total_failed"] += 1
                return False

        # No execution callback - just mark as executed
        opp.execution_result = {"success": True, "note": "No executor configured"}
        opp.executed_at = time.time()
        self._metrics["total_executed"] += 1
        return True

    async def _stage_reflection(self, opp: PipelineOpportunity, success: bool) -> None:
        """Record outcome for learning."""
        if self.event_bus:
            await self.event_bus.publish(
                Event(
                    topic="pipeline.opportunity.completed",
                    payload={
                        **opp.to_dict(),
                        "success": success,
                        "execution_result": opp.execution_result,
                        "duration": time.time() - opp.detected_at,
                    },
                )
            )

    # -------------------------------------------------------------------------
    # HUMAN APPROVAL INTERFACE
    # -------------------------------------------------------------------------

    def get_pending_approvals(self) -> List[Dict[str, Any]]:
        """Get all opportunities pending human approval."""
        return [opp.to_dict() for opp in self._pending_approval.values()]

    async def approve(self, opportunity_id: str) -> bool:
        """Approve a pending opportunity."""
        if opportunity_id not in self._pending_approval:
            return False

        opp = self._pending_approval.pop(opportunity_id)
        opp.status = OpportunityStatus.APPROVED
        self._metrics["total_approved"] += 1
        self._metrics["total_deferred"] -= 1

        # Continue execution
        success = await self._stage_execution(opp)
        await self._stage_reflection(opp, success)
        self._finalize(
            opp, OpportunityStatus.EXECUTED if success else OpportunityStatus.FAILED
        )

        return True

    async def reject(self, opportunity_id: str, reason: str = "") -> bool:
        """Reject a pending opportunity."""
        if opportunity_id not in self._pending_approval:
            return False

        opp = self._pending_approval.pop(opportunity_id)
        opp.status = OpportunityStatus.REJECTED
        opp.rejection_reason = reason or "User rejected"
        self._metrics["total_rejected"] += 1
        self._metrics["total_deferred"] -= 1

        self._finalize(opp, OpportunityStatus.REJECTED)
        return True

    # -------------------------------------------------------------------------
    # CALLBACKS REGISTRATION
    # -------------------------------------------------------------------------

    def set_enrichment_callback(self, callback: Callable) -> None:
        """Set callback for knowledge enrichment stage."""
        self._enrichment_callback = callback

    def set_planning_callback(self, callback: Callable) -> None:
        """Set callback for action planning stage."""
        self._planning_callback = callback

    def set_execution_callback(self, callback: Callable) -> None:
        """Set callback for action execution stage."""
        self._execution_callback = callback

    def set_approval_callback(self, callback: Callable) -> None:
        """Set callback for approval decisions."""
        self._approval_callback = callback

    # -------------------------------------------------------------------------
    # UTILITIES
    # -------------------------------------------------------------------------

    def _finalize(self, opp: PipelineOpportunity, status: OpportunityStatus) -> None:
        """Finalize opportunity processing."""
        opp.status = status

        # Remove from active
        self._active.pop(opp.id, None)

        # PERF FIX: deque with maxlen auto-discards oldest (O(1))
        self._completed.append(opp)

    def get_persistable_state(self) -> Dict[str, Any]:
        """Get serializable state for checkpoint persistence.

        SAFETY-CRITICAL: Rate limiter counts must survive restarts to prevent
        action flooding after crash-restart cycles. Without this, a restart
        resets all hourly/daily counters, allowing unbounded actions.
        """
        # Extract rate limiter state
        rate_limiter_state: Dict[str, Any] = {}
        for f in self._filters:
            if isinstance(f, RateLimitFilter):
                rate_limiter_state = {
                    "hourly_counts": dict(f._hourly_counts),
                    "daily_counts": dict(f._daily_counts),
                    "last_reset_hour": f._last_reset_hour,
                    "last_reset_day": f._last_reset_day,
                }
                break

        return {
            "metrics": dict(self._metrics),
            "rate_limiter": rate_limiter_state,
            "pending_approval_count": len(self._pending_approval),
            "completed_count": len(self._completed),
        }

    def restore_persistable_state(self, state: Dict[str, Any]) -> bool:
        """Restore rate limiter state from persisted checkpoint.

        Returns True if rate limiter was restored.
        """
        rate_state = state.get("rate_limiter", {})
        if not rate_state:
            return False

        for f in self._filters:
            if isinstance(f, RateLimitFilter):
                f._hourly_counts = rate_state.get("hourly_counts", {})
                f._daily_counts = rate_state.get("daily_counts", {})
                f._last_reset_hour = rate_state.get("last_reset_hour", time.time())
                f._last_reset_day = rate_state.get("last_reset_day", time.time())
                logger.info(
                    "Rate limiter restored: %d hourly, %d daily domains",
                    len(f._hourly_counts),
                    len(f._daily_counts),
                )
                return True
        return False

    def stats(self) -> Dict[str, Any]:
        """Get pipeline statistics."""
        maxsize = self._queue.maxsize
        qsize = self._queue.qsize()
        return {
            **self._metrics,
            "queue_size": qsize,
            "queue_maxsize": maxsize,
            "queue_utilization": qsize / maxsize if maxsize > 0 else 0.0,
            "active_count": len(self._active),
            "pending_approval": len(self._pending_approval),
            "completed_count": len(self._completed),
            "running": self._running,
        }

    def get_active_opportunities(self) -> List[Dict[str, Any]]:
        """Get currently processing opportunities."""
        return [opp.to_dict() for opp in self._active.values()]

    def get_completed(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Get recently completed opportunities."""
        # Convert deque slice to list for external use
        return [opp.to_dict() for opp in list(self._completed)[-limit:]]


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


async def create_opportunity_pipeline(
    event_bus: Optional[EventBus] = None,
    snr_threshold: float = 0.85,
    ihsan_threshold: float = 0.95,
    auto_start: bool = True,
) -> OpportunityPipeline:
    """
    Factory function to create and optionally start an OpportunityPipeline.

    Args:
        event_bus: Event bus for publishing pipeline events
        snr_threshold: Minimum SNR score for opportunities
        ihsan_threshold: Base Ihsan threshold (adjusted per autonomy level)
        auto_start: Whether to start the pipeline immediately

    Returns:
        Configured OpportunityPipeline instance
    """
    pipeline = OpportunityPipeline(
        event_bus=event_bus,
        snr_threshold=snr_threshold,
        ihsan_threshold=ihsan_threshold,
    )

    if auto_start:
        await pipeline.start()

    return pipeline


# =============================================================================
# INTEGRATION HELPERS
# =============================================================================


def connect_muraqabah_to_pipeline(
    muraqabah_engine: Any,
    pipeline: OpportunityPipeline,
) -> None:
    """
    Connect MuraqabahEngine to OpportunityPipeline.

    This creates the data flow:
        MuraqabahEngine.publish_opportunity() → OpportunityPipeline.submit()
    """

    async def forward_to_pipeline(event: Event) -> None:
        # RFC-06 FIX: Event uses .topic and .payload (not .event_type / .data)
        if event.topic == "muraqabah.opportunity":
            try:
                await pipeline.submit_from_muraqabah(
                    domain=event.payload.get("domain", "unknown"),
                    description=event.payload.get("description", ""),
                    source=event.payload.get("source", "muraqabah"),
                    snr_score=event.payload.get("snr_score", 0.0),
                    urgency=event.payload.get("urgency", 0.5),
                    estimated_value=event.payload.get("estimated_value", 0.0),
                    context=event.payload.get("context"),
                )
            except asyncio.QueueFull:
                logger.warning(
                    "Pipeline queue full — muraqabah opportunity dropped [domain=%s]",
                    event.payload.get("domain", "unknown"),
                )

    # Subscribe to muraqabah opportunities
    if hasattr(muraqabah_engine, "event_bus") and muraqabah_engine.event_bus:
        muraqabah_engine.event_bus.subscribe(
            "muraqabah.opportunity", forward_to_pipeline
        )


def connect_background_agents_to_pipeline(
    agent_registry: Any,
    pipeline: OpportunityPipeline,
) -> None:
    """
    Connect BackgroundAgentRegistry to OpportunityPipeline for planning.

    This allows background agents to create action plans for opportunities.
    """

    async def plan_with_agents(opp: PipelineOpportunity) -> Dict[str, Any]:
        # Find best agent for this domain
        if hasattr(agent_registry, "get_agent_for_domain"):
            agent = agent_registry.get_agent_for_domain(opp.domain)
            if agent:
                plan = await agent.plan_action(opp, {})
                return {
                    "agent": agent.agent_id,
                    "plan": plan,
                    "recommended_autonomy": agent.autonomy_level.name,
                }
        return {}

    pipeline.set_planning_callback(plan_with_agents)
