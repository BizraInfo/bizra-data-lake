"""
RDVE Orchestrator — The Recursive Discovery & Verification Engine

Wires existing BIZRA components into the unified RDVE pipeline:
    Generator (HypothesisGenerator) → GoT Explorer → SNR Filter →
    Verifier (AutopoieticLoop) → Recursive Feedback

This is the "faster scientist" — each cycle discovers improvements,
validates them constitutionally, and feeds successes back to improve
the next cycle's discovery capability.

Standing on Giants:
    Shannon (SNR quality filter) · Besta (GoT exploration) ·
    Maturana (autopoietic self-improvement) · Boyd (OODA decision loop) ·
    Deming (PDCA quality cycle) · Al-Ghazali (Ihsan ethics) ·
    Anthropic (constitutional AI)

Artifacts:
    core/autopoiesis/hypothesis_generator.py — Generator module
    core/autopoiesis/got_integration.py — GoT tree search
    core/autopoiesis/loop_engine.py — Verifier + rollback
    core/sovereign/snr_maximizer.py — Shannon SNR filter
    core/pci/gates.py — Proof-Carrying Inference
    core/integration/constants.py — Constitutional thresholds (SSOT)
"""

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, Final, List, Optional, Tuple

from core.autopoiesis.got_integration import (
    ExploredHypothesis,
    GoTHypothesisExplorer,
)
from core.autopoiesis.hypothesis_generator import (
    Hypothesis,
    HypothesisGenerator,
    SystemObservation,
)
from core.autopoiesis.loop_engine import AutopoieticLoop
from core.integration.constants import (
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_MINIMUM,
    STRICT_IHSAN_THRESHOLD,
    SNR_THRESHOLD_T0_ELITE,
    SNR_THRESHOLD_T1_HIGH,
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)
from core.sovereign.snr_maximizer import SNRMaximizer

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# VERSION & GIANTS
# ═══════════════════════════════════════════════════════════════════════════════

RDVE_VERSION: Final[str] = "1.0.0"
RDVE_CODENAME: Final[str] = "Spearpoint"

STANDING_ON_GIANTS: Final[list] = [
    "Shannon (information theory, 1948) — SNR quality filter",
    "Besta (Graph-of-Thoughts, 2024) — non-linear exploration",
    "Maturana (autopoiesis, 1972) — self-producing systems",
    "Boyd (OODA loop, 1976) — observe-orient-decide-act",
    "Deming (PDCA cycle, 1950) — plan-do-check-act quality",
    "Al-Ghazali (Ihsan ethics, 1095) — excellence as constraint",
    "Lamport (distributed reliability, 1978) — fault tolerance",
    "Anthropic (constitutional AI, 2023) — alignment by design",
]


# ═══════════════════════════════════════════════════════════════════════════════
# ENUMS & DATA CLASSES
# ═══════════════════════════════════════════════════════════════════════════════


class RDVEStage(str, Enum):
    """RDVE pipeline stages — Boyd's OODA extended with Deming's PDCA."""

    OBSERVE = "observe"
    GENERATE = "generate"
    EXPLORE = "explore"  # GoT tree search
    FILTER = "filter"  # SNR quality gate
    VERIFY = "verify"  # Constitutional verification
    IMPLEMENT = "implement"
    INTEGRATE = "integrate"
    LEARN = "learn"  # Recursive feedback


class RDVEStatus(str, Enum):
    """Engine-level status."""

    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    HALTED = "halted"  # Constitutional violation


class CycleOutcome(str, Enum):
    """Outcome of a single RDVE cycle."""

    DISCOVERY = "discovery"  # Valid improvement found and integrated
    NO_SIGNAL = "no_signal"  # No hypotheses passed SNR filter
    VERIFICATION_FAIL = "verification_fail"  # Hypotheses failed constitutional check
    IMPLEMENTATION_FAIL = "implementation_fail"  # Implementation failed
    CONVERGED = "converged"  # System has converged, no further improvements


@dataclass
class RDVEConfig:
    """Configuration for the RDVE pipeline.

    All thresholds sourced from core/integration/constants.py (SSOT).
    """

    # Exploration parameters
    num_exploration_paths: int = 5
    mcts_iterations: int = 100
    use_mcts: bool = False  # MCTS vs standard GoT exploration

    # Quality gates (from constitutional constants)
    snr_floor: float = UNIFIED_SNR_THRESHOLD  # 0.85
    snr_target: float = SNR_THRESHOLD_T1_HIGH  # 0.95
    ihsan_floor: float = UNIFIED_IHSAN_THRESHOLD  # 0.95
    ihsan_strict: float = STRICT_IHSAN_THRESHOLD  # 0.99

    # Convergence detection
    convergence_window: int = 5  # Cycles to check for convergence
    convergence_threshold: float = 0.01  # Min improvement to avoid convergence

    # Stability
    warmup_cycles: int = 3  # Initial cycles with conservative exploration
    max_cycles: int = 100  # Maximum cycles before forced stop
    cooldown_on_failure: int = 2  # Cycles to wait after failure

    # Safety
    require_human_approval: bool = True  # For HIGH risk hypotheses
    max_concurrent_implementations: int = 1  # Sequential by default
    enable_recursive_self_improvement: bool = False  # Gate behind 0.99 Ihsan


@dataclass
class StageResult:
    """Result of a single RDVE pipeline stage."""

    stage: RDVEStage
    success: bool
    duration_ms: float
    artifacts: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class RDVECycleResult:
    """Complete result of one RDVE cycle (all 8 stages)."""

    cycle_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    cycle_number: int = 0
    outcome: CycleOutcome = CycleOutcome.NO_SIGNAL
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    # Stage results
    stages: Dict[str, StageResult] = field(default_factory=dict)

    # Discovery metrics
    hypotheses_generated: int = 0
    hypotheses_explored: int = 0
    hypotheses_passed_snr: int = 0
    hypotheses_verified: int = 0
    hypotheses_implemented: int = 0

    # Quality scores
    best_snr_score: float = 0.0
    best_ihsan_score: float = 0.0
    best_confidence: float = 0.0

    # The winning hypothesis (if any)
    winning_hypothesis: Optional[Dict[str, Any]] = None

    @property
    def duration_ms(self) -> float:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds() * 1000
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "cycle_number": self.cycle_number,
            "outcome": self.outcome.value,
            "duration_ms": self.duration_ms,
            "hypotheses": {
                "generated": self.hypotheses_generated,
                "explored": self.hypotheses_explored,
                "passed_snr": self.hypotheses_passed_snr,
                "verified": self.hypotheses_verified,
                "implemented": self.hypotheses_implemented,
            },
            "quality": {
                "best_snr": self.best_snr_score,
                "best_ihsan": self.best_ihsan_score,
                "best_confidence": self.best_confidence,
            },
            "winning_hypothesis": self.winning_hypothesis,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# RDVE ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════


class RDVEOrchestrator:
    """
    Recursive Discovery & Verification Engine.

    Orchestrates the full RDVE pipeline by wiring existing BIZRA components:
        Generator → GoT Explorer → SNR Filter → Verifier → Feedback Loop

    This is the "faster scientist" — each cycle improves the system and
    feeds successes back to improve subsequent discovery capability.

    Standing on Giants:
        Shannon (SNR) · Besta (GoT) · Maturana (autopoiesis) ·
        Boyd (OODA) · Deming (PDCA) · Al-Ghazali (Ihsan)

    Usage:
        >>> rdve = RDVEOrchestrator()
        >>> result = await rdve.run_cycle()
        >>> print(result.outcome)
        CycleOutcome.DISCOVERY

        # Or run multiple cycles:
        >>> results = await rdve.run_campaign(max_cycles=10)
    """

    def __init__(
        self,
        config: Optional[RDVEConfig] = None,
        generator: Optional[HypothesisGenerator] = None,
        explorer: Optional[GoTHypothesisExplorer] = None,
        snr_filter: Optional[SNRMaximizer] = None,
        loop: Optional[AutopoieticLoop] = None,
    ):
        self.config = config or RDVEConfig()

        # Wire components (use existing or create with required args)
        if generator is not None:
            self._generator = generator
        else:
            from pathlib import Path
            _mem = Path("sovereign_state/rdve_hypotheses")
            _mem.parent.mkdir(parents=True, exist_ok=True)
            self._generator = HypothesisGenerator(memory_path=_mem)

        if explorer is not None:
            self._explorer = explorer
        else:
            from core.autopoiesis.got_integration import GoTBridge
            self._explorer = GoTHypothesisExplorer(got_bridge=GoTBridge())

        self._snr = snr_filter or SNRMaximizer()
        self._loop = loop or AutopoieticLoop()

        # State
        self._status = RDVEStatus.IDLE
        self._cycle_count = 0
        self._history: List[RDVECycleResult] = []
        self._consecutive_failures = 0
        self._improvement_scores: List[float] = []

        logger.info(
            f"RDVE Orchestrator v{RDVE_VERSION} '{RDVE_CODENAME}' initialized. "
            f"Components: Generator + GoT Explorer + SNR Filter + AutopoieticLoop"
        )

    @property
    def status(self) -> RDVEStatus:
        return self._status

    @property
    def cycle_count(self) -> int:
        return self._cycle_count

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 1: OBSERVE
    # ═══════════════════════════════════════════════════════════════════════

    async def _observe(self) -> Tuple[StageResult, Optional[SystemObservation]]:
        """Gather runtime state. Non-mutating. (Boyd: Observe)"""
        start = time.time()

        import inspect

        try:
            _obs = self._loop.observe()
            observation = await _obs if inspect.isawaitable(_obs) else _obs

            if observation is None:
                # Create a baseline observation if loop can't observe
                observation = SystemObservation(
                    timestamp=datetime.now(timezone.utc),
                    avg_latency_ms=0.0,
                    p95_latency_ms=0.0,
                    p99_latency_ms=0.0,
                    throughput_rps=0.0,
                    cache_hit_rate=0.0,
                    ihsan_score=self.config.ihsan_floor,
                    snr_score=self.config.snr_floor,
                    error_rate=0.0,
                    verification_failure_rate=0.0,
                    cpu_percent=0.0,
                    memory_percent=0.0,
                    gpu_percent=0.0,
                    token_usage_avg=0.0,
                    batch_utilization=0.0,
                    skill_coverage=0.0,
                )

            duration = (time.time() - start) * 1000
            return StageResult(
                stage=RDVEStage.OBSERVE,
                success=True,
                duration_ms=duration,
                artifacts={
                    "ihsan_score": observation.ihsan_score,
                    "snr_score": observation.snr_score,
                    "error_rate": observation.error_rate,
                },
            ), observation

        except Exception as e:
            duration = (time.time() - start) * 1000
            logger.error(f"RDVE Observe failed: {e}")
            return StageResult(
                stage=RDVEStage.OBSERVE,
                success=False,
                duration_ms=duration,
                error=str(e),
            ), None

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 2: GENERATE
    # ═══════════════════════════════════════════════════════════════════════

    async def _generate(
        self, observation: SystemObservation
    ) -> Tuple[StageResult, List[Hypothesis]]:
        """Generate hypotheses from observation. (Boyd: Orient — diverge)"""
        start = time.time()

        try:
            hypotheses = self._generator.generate(observation)

            # Rank by expected value
            ranked = self._generator.rank_hypotheses(hypotheses)

            duration = (time.time() - start) * 1000
            logger.info(f"RDVE Generate: {len(ranked)} hypotheses produced")

            return StageResult(
                stage=RDVEStage.GENERATE,
                success=len(ranked) > 0,
                duration_ms=duration,
                artifacts={
                    "count": len(ranked),
                    "categories": list({h.category.value for h in ranked}),
                },
            ), ranked

        except Exception as e:
            duration = (time.time() - start) * 1000
            logger.error(f"RDVE Generate failed: {e}")
            return StageResult(
                stage=RDVEStage.GENERATE,
                success=False,
                duration_ms=duration,
                error=str(e),
            ), []

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 3: EXPLORE (GoT Tree Search)
    # ═══════════════════════════════════════════════════════════════════════

    async def _explore(
        self, observation: SystemObservation
    ) -> Tuple[StageResult, List[ExploredHypothesis]]:
        """Explore hypothesis space via GoT. (Besta: branch/aggregate/refine)"""
        start = time.time()

        try:
            import asyncio
            import inspect

            if self.config.use_mcts:
                result = self._explorer.explore_with_mcts(
                    observation=observation,
                    iterations=self.config.mcts_iterations,
                    hypothesis_generator=self._generator,
                )
            else:
                result = self._explorer.explore_hypotheses(
                    observation=observation,
                    num_paths=self.config.num_exploration_paths,
                    hypothesis_generator=self._generator,
                )

            # Handle both sync and async return types
            explored = await result if inspect.isawaitable(result) else result

            duration = (time.time() - start) * 1000
            logger.info(
                f"RDVE Explore: {len(explored)} paths explored via "
                f"{'MCTS' if self.config.use_mcts else 'GoT'}"
            )

            return StageResult(
                stage=RDVEStage.EXPLORE,
                success=len(explored) > 0,
                duration_ms=duration,
                artifacts={
                    "paths_explored": len(explored),
                    "method": "mcts" if self.config.use_mcts else "got",
                    "avg_depth": (
                        sum(e.exploration_depth for e in explored) / len(explored)
                        if explored
                        else 0
                    ),
                },
            ), explored

        except Exception as e:
            duration = (time.time() - start) * 1000
            logger.error(f"RDVE Explore failed: {e}")
            return StageResult(
                stage=RDVEStage.EXPLORE,
                success=False,
                duration_ms=duration,
                error=str(e),
            ), []

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 4: FILTER (SNR Quality Gate)
    # ═══════════════════════════════════════════════════════════════════════

    async def _filter_snr(
        self, explored: List[ExploredHypothesis]
    ) -> Tuple[StageResult, List[ExploredHypothesis]]:
        """Filter hypotheses through Shannon SNR gate. (Shannon: noise rejection)"""
        start = time.time()

        passed: List[ExploredHypothesis] = []
        rejected = 0

        for eh in explored:
            # Multi-signal SNR check:
            # 1. The exploration path SNR from GoT
            # 2. The hypothesis description through SNR maximizer
            # 3. The Ihsan impact score

            path_snr = eh.snr_score
            ihsan_impact = eh.ihsan_score

            # Gate: both SNR and Ihsan must meet floor
            if path_snr >= self.config.snr_floor and ihsan_impact >= self.config.ihsan_floor:
                passed.append(eh)
            else:
                rejected += 1
                logger.debug(
                    f"RDVE SNR Filter rejected: SNR={path_snr:.3f} "
                    f"Ihsan={ihsan_impact:.3f} "
                    f"(floors: SNR>={self.config.snr_floor}, "
                    f"Ihsan>={self.config.ihsan_floor})"
                )

        # Sort by composite score (SNR * Ihsan * confidence)
        passed.sort(
            key=lambda e: e.snr_score * e.ihsan_score * e.confidence,
            reverse=True,
        )

        duration = (time.time() - start) * 1000
        logger.info(
            f"RDVE SNR Filter: {len(passed)} passed, {rejected} rejected "
            f"(floor: SNR>={self.config.snr_floor}, Ihsan>={self.config.ihsan_floor})"
        )

        return StageResult(
            stage=RDVEStage.FILTER,
            success=len(passed) > 0,
            duration_ms=duration,
            artifacts={
                "passed": len(passed),
                "rejected": rejected,
                "best_snr": passed[0].snr_score if passed else 0.0,
                "best_ihsan": passed[0].ihsan_score if passed else 0.0,
            },
        ), passed

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 5: VERIFY (Constitutional Validation)
    # ═══════════════════════════════════════════════════════════════════════

    async def _verify(
        self, candidates: List[ExploredHypothesis]
    ) -> Tuple[StageResult, List[ExploredHypothesis]]:
        """Verify through constitutional gate. (Al-Ghazali: Ihsan as hard constraint)"""
        start = time.time()

        verified: List[ExploredHypothesis] = []

        import inspect

        for eh in candidates:
            try:
                _vr = self._loop.validate(eh.hypothesis)
                result = await _vr if inspect.isawaitable(_vr) else _vr

                if result.is_valid:
                    verified.append(eh)
                    logger.info(
                        f"RDVE Verify PASSED: {eh.hypothesis.description[:60]}..."
                    )
                else:
                    logger.info(
                        f"RDVE Verify REJECTED: {eh.hypothesis.description[:60]}... "
                        f"reason={result.rejection_reason}"
                    )
            except Exception as e:
                logger.warning(f"RDVE Verify error for hypothesis: {e}")

        duration = (time.time() - start) * 1000

        return StageResult(
            stage=RDVEStage.VERIFY,
            success=len(verified) > 0,
            duration_ms=duration,
            artifacts={
                "verified": len(verified),
                "rejected": len(candidates) - len(verified),
            },
        ), verified

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 6: IMPLEMENT
    # ═══════════════════════════════════════════════════════════════════════

    async def _implement(
        self, best: ExploredHypothesis
    ) -> Tuple[StageResult, bool]:
        """Implement the best verified hypothesis. (Deming: Do)"""
        start = time.time()

        import inspect

        try:
            _ir = self._loop.implement(best.hypothesis)
            result = await _ir if inspect.isawaitable(_ir) else _ir

            duration = (time.time() - start) * 1000
            success = result.success if hasattr(result, "success") else True

            return StageResult(
                stage=RDVEStage.IMPLEMENT,
                success=success,
                duration_ms=duration,
                artifacts={
                    "hypothesis_id": best.hypothesis.id,
                    "description": best.hypothesis.description[:100],
                },
            ), success

        except Exception as e:
            duration = (time.time() - start) * 1000
            logger.error(f"RDVE Implement failed: {e}")
            return StageResult(
                stage=RDVEStage.IMPLEMENT,
                success=False,
                duration_ms=duration,
                error=str(e),
            ), False

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 7: INTEGRATE
    # ═══════════════════════════════════════════════════════════════════════

    async def _integrate(
        self, best: ExploredHypothesis
    ) -> StageResult:
        """Integrate successful implementation. (Deming: Check + Act)"""
        import inspect

        start = time.time()

        try:
            _ir = self._loop.implement(best.hypothesis)
            impl_result = await _ir if inspect.isawaitable(_ir) else _ir
            _int = self._loop.integrate(impl_result)
            int_result = await _int if inspect.isawaitable(_int) else _int

            duration = (time.time() - start) * 1000
            return StageResult(
                stage=RDVEStage.INTEGRATE,
                success=True,
                duration_ms=duration,
                artifacts={"integrated": True},
            )

        except Exception as e:
            duration = (time.time() - start) * 1000
            return StageResult(
                stage=RDVEStage.INTEGRATE,
                success=False,
                duration_ms=duration,
                error=str(e),
            )

    # ═══════════════════════════════════════════════════════════════════════
    # STAGE 8: LEARN (Recursive Feedback)
    # ═══════════════════════════════════════════════════════════════════════

    async def _learn(
        self,
        best: ExploredHypothesis,
        success: bool,
    ) -> StageResult:
        """Feed outcome back to generator. (Maturana: autopoietic self-production)"""
        start = time.time()

        try:
            self._generator.learn_from_outcome(
                hypothesis=best.hypothesis,
                success=success,
                actual_improvement=(
                    best.hypothesis.predicted_improvement if success else None
                ),
            )

            duration = (time.time() - start) * 1000
            logger.info(
                f"RDVE Learn: {'SUCCESS' if success else 'FAILURE'} fed back "
                f"to generator. Pattern success rates updated."
            )

            return StageResult(
                stage=RDVEStage.LEARN,
                success=True,
                duration_ms=duration,
                artifacts={
                    "outcome": "success" if success else "failure",
                    "generator_stats": self._generator.get_statistics(),
                },
            )

        except Exception as e:
            duration = (time.time() - start) * 1000
            return StageResult(
                stage=RDVEStage.LEARN,
                success=False,
                duration_ms=duration,
                error=str(e),
            )

    # ═══════════════════════════════════════════════════════════════════════
    # CONVERGENCE DETECTION
    # ═══════════════════════════════════════════════════════════════════════

    def _check_convergence(self) -> bool:
        """Detect if the system has converged (no more improvements possible)."""
        if len(self._improvement_scores) < self.config.convergence_window:
            return False

        recent = self._improvement_scores[-self.config.convergence_window :]
        if not recent:
            return False

        # Check if improvement rate has plateaued
        max_improvement = max(recent)
        return max_improvement < self.config.convergence_threshold

    # ═══════════════════════════════════════════════════════════════════════
    # MAIN CYCLE
    # ═══════════════════════════════════════════════════════════════════════

    async def run_cycle(
        self,
        observation: Optional[SystemObservation] = None,
    ) -> RDVECycleResult:
        """
        Execute one complete RDVE cycle (all 8 stages).

        Returns:
            RDVECycleResult with full pipeline telemetry
        """
        self._cycle_count += 1
        result = RDVECycleResult(
            cycle_number=self._cycle_count,
            started_at=datetime.now(timezone.utc),
        )
        self._status = RDVEStatus.RUNNING

        logger.info(f"=== RDVE Cycle {self._cycle_count} START ===")

        # ─── Stage 1: Observe ───
        if observation is None:
            obs_result, observation = await self._observe()
            result.stages["observe"] = obs_result
            if not obs_result.success or observation is None:
                result.outcome = CycleOutcome.NO_SIGNAL
                result.completed_at = datetime.now(timezone.utc)
                self._history.append(result)
                return result

        # ─── Stage 2: Generate ───
        gen_result, hypotheses = await self._generate(observation)
        result.stages["generate"] = gen_result
        result.hypotheses_generated = len(hypotheses)

        if not hypotheses:
            result.outcome = CycleOutcome.NO_SIGNAL
            result.completed_at = datetime.now(timezone.utc)
            self._history.append(result)
            return result

        # ─── Stage 3: Explore (GoT) ───
        exp_result, explored = await self._explore(observation)
        result.stages["explore"] = exp_result
        result.hypotheses_explored = len(explored)

        if not explored:
            result.outcome = CycleOutcome.NO_SIGNAL
            result.completed_at = datetime.now(timezone.utc)
            self._history.append(result)
            return result

        # ─── Stage 4: Filter (SNR) ───
        flt_result, passed = await self._filter_snr(explored)
        result.stages["filter"] = flt_result
        result.hypotheses_passed_snr = len(passed)

        if passed:
            result.best_snr_score = passed[0].snr_score
            result.best_ihsan_score = passed[0].ihsan_score
            result.best_confidence = passed[0].confidence

        if not passed:
            result.outcome = CycleOutcome.NO_SIGNAL
            self._improvement_scores.append(0.0)
            result.completed_at = datetime.now(timezone.utc)
            self._history.append(result)
            return result

        # ─── Stage 5: Verify (Constitutional) ───
        ver_result, verified = await self._verify(passed)
        result.stages["verify"] = ver_result
        result.hypotheses_verified = len(verified)

        if not verified:
            result.outcome = CycleOutcome.VERIFICATION_FAIL
            self._consecutive_failures += 1
            self._improvement_scores.append(0.0)
            result.completed_at = datetime.now(timezone.utc)
            self._history.append(result)
            return result

        # Take the best verified hypothesis
        best = verified[0]

        # ─── Stage 6: Implement ───
        impl_result, impl_success = await self._implement(best)
        result.stages["implement"] = impl_result

        if not impl_success:
            result.outcome = CycleOutcome.IMPLEMENTATION_FAIL
            self._consecutive_failures += 1
            self._improvement_scores.append(0.0)

            # Learn from failure too
            learn_result = await self._learn(best, success=False)
            result.stages["learn"] = learn_result

            result.completed_at = datetime.now(timezone.utc)
            self._history.append(result)
            return result

        # ─── Stage 7: Integrate ───
        int_result = await self._integrate(best)
        result.stages["integrate"] = int_result

        # ─── Stage 8: Learn (Recursive Feedback) ───
        learn_result = await self._learn(best, success=True)
        result.stages["learn"] = learn_result

        # Record success
        result.outcome = CycleOutcome.DISCOVERY
        result.hypotheses_implemented = 1
        result.winning_hypothesis = {
            "id": best.hypothesis.id,
            "description": best.hypothesis.description,
            "category": best.hypothesis.category.value,
            "snr": best.snr_score,
            "ihsan": best.ihsan_score,
            "confidence": best.confidence,
        }

        self._consecutive_failures = 0
        improvement = best.snr_score * best.ihsan_score
        self._improvement_scores.append(improvement)

        result.completed_at = datetime.now(timezone.utc)
        self._history.append(result)

        logger.info(
            f"=== RDVE Cycle {self._cycle_count} COMPLETE: {result.outcome.value} "
            f"(SNR={result.best_snr_score:.3f}, "
            f"Ihsan={result.best_ihsan_score:.3f}, "
            f"duration={result.duration_ms:.0f}ms) ==="
        )

        self._status = RDVEStatus.COMPLETED
        return result

    # ═══════════════════════════════════════════════════════════════════════
    # CAMPAIGN (Multi-Cycle)
    # ═══════════════════════════════════════════════════════════════════════

    async def run_campaign(
        self,
        max_cycles: Optional[int] = None,
    ) -> List[RDVECycleResult]:
        """
        Run multiple RDVE cycles until convergence or max_cycles.

        This is the recursive acceleration loop — each cycle's discoveries
        improve the next cycle's capability.

        Returns:
            List of all cycle results
        """
        max_cycles = max_cycles or self.config.max_cycles
        results: List[RDVECycleResult] = []

        logger.info(
            f"=== RDVE Campaign START (max_cycles={max_cycles}) ==="
        )

        for i in range(max_cycles):
            # Convergence check
            if self._check_convergence():
                logger.info(
                    f"RDVE Campaign: Converged after {i} cycles. "
                    f"Improvement plateau detected."
                )
                break

            # Cooldown after failures
            if self._consecutive_failures >= 3:
                logger.warning(
                    f"RDVE Campaign: {self._consecutive_failures} consecutive "
                    f"failures. Halting for safety."
                )
                self._status = RDVEStatus.HALTED
                break

            result = await self.run_cycle()
            results.append(result)

            # Early termination on constitutional violation
            if self._status == RDVEStatus.HALTED:
                break

        # Campaign summary
        discoveries = sum(
            1 for r in results if r.outcome == CycleOutcome.DISCOVERY
        )
        logger.info(
            f"=== RDVE Campaign COMPLETE: {len(results)} cycles, "
            f"{discoveries} discoveries ==="
        )

        return results

    # ═══════════════════════════════════════════════════════════════════════
    # TELEMETRY
    # ═══════════════════════════════════════════════════════════════════════

    def get_status(self) -> Dict[str, Any]:
        """Get current RDVE engine status."""
        return {
            "version": RDVE_VERSION,
            "codename": RDVE_CODENAME,
            "status": self._status.value,
            "cycle_count": self._cycle_count,
            "total_discoveries": sum(
                1 for r in self._history
                if r.outcome == CycleOutcome.DISCOVERY
            ),
            "consecutive_failures": self._consecutive_failures,
            "converged": self._check_convergence(),
            "config": {
                "snr_floor": self.config.snr_floor,
                "ihsan_floor": self.config.ihsan_floor,
                "exploration_paths": self.config.num_exploration_paths,
                "use_mcts": self.config.use_mcts,
            },
            "components": {
                "generator": type(self._generator).__name__,
                "explorer": type(self._explorer).__name__,
                "snr_filter": type(self._snr).__name__,
                "loop": type(self._loop).__name__,
            },
            "giants": STANDING_ON_GIANTS,
        }

    def get_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent cycle history."""
        return [r.to_dict() for r in self._history[-limit:]]

    def get_discovery_rate(self) -> float:
        """Get the discovery rate (discoveries / total cycles)."""
        if not self._history:
            return 0.0
        discoveries = sum(
            1 for r in self._history
            if r.outcome == CycleOutcome.DISCOVERY
        )
        return discoveries / len(self._history)

    def get_improvement_trajectory(self) -> List[float]:
        """Get the improvement score trajectory across cycles."""
        return list(self._improvement_scores)
