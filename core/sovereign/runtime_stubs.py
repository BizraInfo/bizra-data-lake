"""
Runtime Component Stubs — Fallback Implementations
==================================================
Stub implementations for runtime components when real components unavailable.
Provides graceful degradation with sensible defaults.

Standing on Giants: Null Object Pattern + Graceful Degradation
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .runtime_types import (
    LoopStatus,
    ReasoningResult,
    SNRResult,
    ValidationResult,
)

logger = logging.getLogger("sovereign.runtime.stubs")


# =============================================================================
# BASE STUB
# =============================================================================


@dataclass
class ComponentStub:
    """Base class for component stubs."""

    name: str = "stub"
    is_stub: bool = True
    fallback_reason: str = "Component not available"

    def log_fallback(self, operation: str) -> None:
        """Log that a fallback is being used."""
        logger.debug(
            f"{self.name}: Using fallback for {operation} ({self.fallback_reason})"
        )


# =============================================================================
# GRAPH REASONER STUB
# =============================================================================


class GraphReasonerStub(ComponentStub):
    """Stub for GraphOfThoughts reasoner."""

    def __init__(self, reason: str = "GraphOfThoughts not available"):
        super().__init__(name="GraphReasonerStub", fallback_reason=reason)

    async def reason(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None,
        max_depth: int = 5,
    ) -> ReasoningResult:
        """Fallback reasoning - returns simple response."""
        self.log_fallback("reason")
        return ReasoningResult(
            thoughts=[f"Direct response to: {query[:50]}..."],
            conclusion=f"[Stub] Unable to perform deep reasoning. Query: {query}",
            confidence=0.5,
            depth_reached=1,
        )


# =============================================================================
# SNR OPTIMIZER STUB
# =============================================================================


class SNROptimizerStub(ComponentStub):
    """Stub for SNR maximizer."""

    def __init__(self, reason: str = "SNRMaximizer not available"):
        super().__init__(name="SNROptimizerStub", fallback_reason=reason)
        self._default_snr = 0.85  # Return threshold as default

    def optimize(self, text: str) -> SNRResult:
        """Fallback optimization - returns text unchanged."""
        self.log_fallback("optimize")
        return SNRResult(
            original_length=len(text),
            snr_score=self._default_snr,
            meets_threshold=True,
        )

    def calculate_snr(self, text: str) -> float:
        """Calculate SNR for text (stub returns default)."""
        self.log_fallback("calculate_snr")
        return self._default_snr


# =============================================================================
# GUARDIAN STUB
# =============================================================================


class GuardianStub(ComponentStub):
    """Stub for Guardian Council validator."""

    def __init__(self, reason: str = "GuardianCouncil not available"):
        super().__init__(name="GuardianStub", fallback_reason=reason)

    async def validate(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> ValidationResult:
        """Fallback validation - always passes with warning."""
        self.log_fallback("validate")
        return ValidationResult(
            is_valid=True,
            confidence=0.7,
            issues=["[Stub] Validation skipped - Guardian Council unavailable"],
        )

    async def review(
        self,
        content: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Review content (stub implementation)."""
        self.log_fallback("review")
        return {
            "approved": True,
            "confidence": 0.7,
            "reviewers": ["stub_reviewer"],
            "issues": [],
        }


# =============================================================================
# AUTONOMOUS LOOP STUB
# =============================================================================


class AutonomousLoopStub(ComponentStub):
    """Stub for Autonomous OODA Loop."""

    def __init__(self, reason: str = "AutonomousLoop not available"):
        super().__init__(name="AutonomousLoopStub", fallback_reason=reason)
        self._running = False
        self._cycle = 0

    async def start(self) -> None:
        """Start the loop (stub does nothing)."""
        self.log_fallback("start")
        self._running = True
        logger.info("AutonomousLoopStub: Simulated start (no actual loop)")

    def stop(self) -> None:
        """Stop the loop."""
        self.log_fallback("stop")
        self._running = False

    def status(self) -> LoopStatus:
        """Get loop status."""
        return LoopStatus(
            running=self._running,
            cycle=self._cycle,
        )

    async def run_cycle(self, extended: bool = False) -> Dict[str, Any]:
        """Run a single cycle (stub implementation)."""
        self.log_fallback("run_cycle")
        self._cycle += 1
        return {
            "cycle": self._cycle,
            "decisions": 0,
            "actions": 0,
            "health": 1.0,
            "stub": True,
        }


# =============================================================================
# STUB FACTORY
# =============================================================================


class StubFactory:
    """Factory for creating component stubs."""

    @staticmethod
    def create_graph_reasoner(reason: str = "Not configured") -> GraphReasonerStub:
        """Create a graph reasoner stub."""
        return GraphReasonerStub(reason=reason)

    @staticmethod
    def create_snr_optimizer(reason: str = "Not configured") -> SNROptimizerStub:
        """Create an SNR optimizer stub."""
        return SNROptimizerStub(reason=reason)

    @staticmethod
    def create_guardian(reason: str = "Not configured") -> GuardianStub:
        """Create a guardian stub."""
        return GuardianStub(reason=reason)

    @staticmethod
    def create_autonomous_loop(reason: str = "Not configured") -> AutonomousLoopStub:
        """Create an autonomous loop stub."""
        return AutonomousLoopStub(reason=reason)


__all__ = [
    "ComponentStub",
    "GraphReasonerStub",
    "SNROptimizerStub",
    "GuardianStub",
    "AutonomousLoopStub",
    "StubFactory",
]
