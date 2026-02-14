"""
RecursiveLoop — Continuous Orchestration Heartbeat
===================================================

Sole owner of circuit breaker logic for the research-evaluation loop.
Connects Research <-> Evaluation in a timed loop with:
- Centralized circuit breaker (3 consecutive rejections -> backoff)
- File-locked evidence ledger writes (via AutoEvaluator)
- Configurable interval (default 300s)
- Graceful shutdown via asyncio.Event
- Fail-closed: any exception -> log + continue
- Pattern-aware mode: uses PatternStrategySelector + MetricsProvider
  to route through Sci-Reasoning thinking patterns with real metrics

NO other module owns circuit breaker logic for this loop.

Standing on Giants:
- Nygard (2007): Circuit breaker pattern
- Boyd (1995): OODA loop
- Deming (1950): PDCA cycle
- Li et al. (2025): Sci-Reasoning thinking patterns (pattern-aware mode)
- Thompson (1933): Exploration/exploitation balance
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Optional

from .auto_evaluator import AutoEvaluator
from .auto_researcher import AutoResearcher, ResearchOutcome
from .config import SpearpointConfig

logger = logging.getLogger(__name__)


@dataclass
class LoopMetrics:
    """Metrics for the recursive loop."""

    cycles_completed: int = 0
    total_hypotheses_evaluated: int = 0
    total_approved: int = 0
    total_rejected: int = 0
    total_inconclusive: int = 0
    consecutive_rejections: int = 0
    circuit_breaker_trips: int = 0
    backoff_events: int = 0
    errors: int = 0
    last_cycle_time: str = ""
    total_elapsed_seconds: float = 0.0

    # Pattern-aware mode metrics
    pattern_aware: bool = False
    patterns_tried: int = 0
    last_pattern_id: str = ""
    last_pattern_strategy: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = {
            "cycles_completed": self.cycles_completed,
            "total_hypotheses_evaluated": self.total_hypotheses_evaluated,
            "total_approved": self.total_approved,
            "total_rejected": self.total_rejected,
            "total_inconclusive": self.total_inconclusive,
            "consecutive_rejections": self.consecutive_rejections,
            "circuit_breaker_trips": self.circuit_breaker_trips,
            "backoff_events": self.backoff_events,
            "errors": self.errors,
            "last_cycle_time": self.last_cycle_time,
        }
        if self.pattern_aware:
            d["pattern_aware"] = True
            d["patterns_tried"] = self.patterns_tried
            d["last_pattern_id"] = self.last_pattern_id
            d["last_pattern_strategy"] = self.last_pattern_strategy
        return d


class RecursiveLoop:
    """
    Continuous orchestration heartbeat with centralized circuit breaker.

    The sole owner of circuit breaker logic for the research-evaluation
    loop. AutoResearcher has NO breaker logic — all flow control lives here.

    Usage:
        config = SpearpointConfig()
        evaluator = AutoEvaluator(config=config)
        researcher = AutoResearcher(evaluator=evaluator, config=config)

        loop = RecursiveLoop(
            evaluator=evaluator,
            researcher=researcher,
            config=config,
        )

        # Run continuously until stopped
        await loop.run()

        # Or run a fixed number of cycles
        await loop.run(max_cycles=5)

        # Graceful shutdown
        loop.request_stop()
    """

    def __init__(
        self,
        evaluator: AutoEvaluator,
        researcher: AutoResearcher,
        config: Optional[SpearpointConfig] = None,
        pattern_selector: Optional[Any] = None,
        metrics_provider: Optional[Any] = None,
    ):
        self.config = config or SpearpointConfig()
        self._evaluator = evaluator
        self._researcher = researcher

        # Pattern-aware mode components (optional)
        self._pattern_selector = pattern_selector
        self._metrics_provider = metrics_provider

        # Circuit breaker state (centralized here ONLY)
        self._consecutive_rejections = 0
        self._breaker_threshold = self.config.circuit_breaker_consecutive_rejections
        self._backoff_seconds = self.config.circuit_breaker_backoff_seconds
        self._breaker_open = False

        # Shutdown signal
        self._stop_event = asyncio.Event()

        # Metrics
        self._metrics = LoopMetrics(
            pattern_aware=pattern_selector is not None,
        )
        self._start_time: Optional[float] = None

    async def run(
        self,
        max_cycles: Optional[int] = None,
        observation_fn: Optional[Any] = None,
    ) -> LoopMetrics:
        """
        Run the recursive loop.

        Args:
            max_cycles: Maximum number of cycles (None = unlimited)
            observation_fn: Optional callable returning SystemObservation

        Returns:
            LoopMetrics with final statistics
        """
        self._start_time = time.monotonic()
        cycle = 0

        logger.info(
            f"RecursiveLoop starting: interval={self.config.loop_interval_seconds}s, "
            f"max_cycles={max_cycles or 'unlimited'}, "
            f"breaker_threshold={self._breaker_threshold}"
        )

        while not self._stop_event.is_set():
            # Check max cycles
            if max_cycles is not None and cycle >= max_cycles:
                logger.info(f"Max cycles ({max_cycles}) reached, stopping")
                break

            # Run one cycle (fail-closed: exceptions -> log + continue)
            try:
                await self._run_cycle(observation_fn)
            except Exception as e:
                logger.error(f"Cycle {cycle} error (continuing): {e}")
                self._metrics.errors += 1

            cycle += 1
            self._metrics.cycles_completed = cycle
            self._metrics.last_cycle_time = datetime.now(timezone.utc).isoformat()

            # Wait for interval or stop signal
            if max_cycles is None or cycle < max_cycles:
                try:
                    await asyncio.wait_for(
                        self._stop_event.wait(),
                        timeout=self.config.loop_interval_seconds,
                    )
                    # If we get here, stop was requested
                    break
                except asyncio.TimeoutError:
                    pass  # Normal: interval elapsed, continue

        self._metrics.total_elapsed_seconds = time.monotonic() - self._start_time
        logger.info(
            f"RecursiveLoop stopped after {cycle} cycles, "
            f"{self._metrics.total_elapsed_seconds:.1f}s elapsed"
        )
        return self._metrics

    async def _run_cycle(
        self,
        observation_fn: Optional[Any] = None,
    ) -> None:
        """
        Run one cycle of the research-evaluation loop.

        Steps:
          1. Check circuit breaker
          2. Get observation
          3. Run researcher (generates + evaluates hypotheses)
          4. Process results (update circuit breaker state)
        """
        # Step 1: Check circuit breaker
        if self._breaker_open:
            logger.info(f"Circuit breaker OPEN, backing off {self._backoff_seconds}s")
            self._metrics.backoff_events += 1

            # Wait for backoff period or stop signal
            try:
                await asyncio.wait_for(
                    self._stop_event.wait(),
                    timeout=self._backoff_seconds,
                )
                return  # Stop requested during backoff
            except asyncio.TimeoutError:
                pass  # Backoff complete

            # Reset breaker to half-open (try one more cycle)
            self._breaker_open = False
            self._consecutive_rejections = 0
            logger.info("Circuit breaker reset to CLOSED (half-open test)")

        # Step 2: Get observation
        observation = None
        if observation_fn is not None:
            try:
                obs_result = observation_fn()
                if asyncio.iscoroutine(obs_result):
                    observation = await obs_result
                else:
                    observation = obs_result
            except Exception as e:
                logger.warning(f"Observation function error: {e}")

        # Step 3: Run researcher
        mission_id = f"loop_cycle_{self._metrics.cycles_completed}"
        top_k = min(self.config.max_iterations_per_cycle, 10)

        if self._pattern_selector is not None:
            # Pattern-aware mode: select thinking pattern, route through
            # research_with_pattern() instead of generic research()
            results = self._run_pattern_aware_research(mission_id, top_k)
        else:
            # Classic mode: generic hypothesis generation
            results = self._researcher.research(
                observation=observation,
                mission_id=mission_id,
                top_k=top_k,
            )

        # Step 4: Process results and update circuit breaker
        cycle_approved = 0
        cycle_rejected = 0
        cycle_inconclusive = 0
        for result in results:
            self._metrics.total_hypotheses_evaluated += 1

            if result.outcome == ResearchOutcome.APPROVED:
                self._metrics.total_approved += 1
                self._consecutive_rejections = 0
                cycle_approved += 1

            elif result.outcome == ResearchOutcome.REJECTED:
                self._metrics.total_rejected += 1
                self._consecutive_rejections += 1
                cycle_rejected += 1

            elif result.outcome == ResearchOutcome.INCONCLUSIVE:
                self._metrics.total_inconclusive += 1
                cycle_inconclusive += 1
                # Inconclusive does NOT reset rejection counter
                # but also doesn't increment it

            elif result.outcome == ResearchOutcome.NO_HYPOTHESES:
                pass  # Neutral — no effect on breaker

            elif result.outcome == ResearchOutcome.GATED:
                self._metrics.total_rejected += 1
                self._consecutive_rejections += 1
                cycle_rejected += 1

        # Feed cycle metrics back to provider and pattern selector
        self._record_cycle_feedback(
            cycle_approved,
            cycle_rejected,
            cycle_inconclusive,
            results,
        )

        # Check circuit breaker threshold
        if self._consecutive_rejections >= self._breaker_threshold:
            self._breaker_open = True
            self._metrics.circuit_breaker_trips += 1
            self._metrics.consecutive_rejections = self._consecutive_rejections
            logger.warning(
                f"Circuit breaker TRIPPED: {self._consecutive_rejections} "
                f"consecutive rejections >= {self._breaker_threshold}"
            )

    def _run_pattern_aware_research(
        self,
        mission_id: str,
        top_k: int,
    ) -> list:
        """Run a pattern-aware research cycle using the PatternStrategySelector.

        Selects the optimal thinking pattern based on history and co-occurrence,
        then routes through researcher.research_with_pattern().
        """
        cycle = self._metrics.cycles_completed
        selection = self._pattern_selector.select(cycle=cycle)

        logger.info(
            f"Pattern selection: {selection.pattern_id} "
            f"(strategy={selection.strategy}, reason={selection.reason})"
        )

        self._metrics.last_pattern_id = selection.pattern_id
        self._metrics.last_pattern_strategy = selection.strategy
        self._metrics.patterns_tried += 1

        # Build claim context from selection rationale
        claim_context = (
            f"Cycle {cycle}: {selection.reason}. " f"Strategy: {selection.strategy}."
        )

        # Get real metrics from provider if available
        metrics_kwargs: dict = {}
        if self._metrics_provider is not None:
            snapshot = self._metrics_provider.current_snapshot()
            metrics_kwargs["metrics"] = snapshot.to_clear_metrics()
            metrics_kwargs["ihsan_components"] = snapshot.to_ihsan_components()

        return self._researcher.research_with_pattern(
            pattern_id=selection.pattern_id,
            claim_context=claim_context,
            mission_id=mission_id,
            top_k=top_k,
            **metrics_kwargs,
        )

    def _record_cycle_feedback(
        self,
        approved: int,
        rejected: int,
        inconclusive: int,
        results: list,
    ) -> None:
        """Feed cycle metrics back to provider and pattern selector."""
        # Feed metrics provider
        if self._metrics_provider is not None:
            avg_clear = 0.0
            avg_ihsan = 0.0
            count = 0
            for r in results:
                if hasattr(r, "evaluation") and r.evaluation is not None:
                    avg_clear += getattr(r.evaluation, "clear_score", 0.0)
                    avg_ihsan += getattr(r.evaluation, "ihsan_score", 0.0)
                    count += 1
            if count > 0:
                avg_clear /= count
                avg_ihsan /= count

            self._metrics_provider.record_cycle_metrics(
                approved=approved,
                rejected=rejected,
                inconclusive=inconclusive,
                clear_score=avg_clear,
                ihsan_score=avg_ihsan,
            )

        # Feed pattern selector
        if self._pattern_selector is not None and self._metrics.last_pattern_id:
            self._pattern_selector.record_outcome(
                pattern_id=self._metrics.last_pattern_id,
                approved=approved,
                rejected=rejected,
                inconclusive=inconclusive,
                cycle=self._metrics.cycles_completed,
            )

    def request_stop(self) -> None:
        """Request graceful shutdown of the loop."""
        logger.info("RecursiveLoop stop requested")
        self._stop_event.set()

    @property
    def is_running(self) -> bool:
        """Check if loop is running."""
        return self._start_time is not None and not self._stop_event.is_set()

    @property
    def breaker_open(self) -> bool:
        """Check if circuit breaker is open."""
        return self._breaker_open

    def get_metrics(self) -> LoopMetrics:
        """Get current loop metrics."""
        self._metrics.consecutive_rejections = self._consecutive_rejections
        return self._metrics


__all__ = [
    "RecursiveLoop",
    "LoopMetrics",
]
