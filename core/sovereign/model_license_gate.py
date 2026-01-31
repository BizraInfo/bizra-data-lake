"""
BIZRA Model License Gate - Validate models before inference

Ensures that models have valid CapabilityCards before allowing
inference requests. Part of the constitutional enforcement chain.

"We do not assume. We verify with formal proofs."
"""

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, Protocol

from .capability_card import (
    CapabilityCard,
    ModelTier,
    TaskType,
    IHSAN_THRESHOLD,
    SNR_THRESHOLD,
    verify_capability_card,
)

logger = logging.getLogger(__name__)


class GateResult(Enum):
    """Result of a gate check."""
    PASSED = "passed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class LicenseCheckResult:
    """Result of a license check."""
    allowed: bool
    model_id: str
    tier: Optional[ModelTier]
    reason: Optional[str]
    card: Optional[CapabilityCard]
    gate_name: str = "LICENSE"


class ModelRegistry(Protocol):
    """Protocol for model registry."""

    def get(self, model_id: str) -> Optional[CapabilityCard]:
        """Get a model's capability card."""
        ...

    def has(self, model_id: str) -> bool:
        """Check if a model is registered."""
        ...


class InMemoryRegistry:
    """Simple in-memory registry for development/testing."""

    def __init__(self):
        self._cards: Dict[str, CapabilityCard] = {}

    def register(self, card: CapabilityCard) -> None:
        """Register a capability card."""
        self._cards[card.model_id] = card

    def get(self, model_id: str) -> Optional[CapabilityCard]:
        """Get a model's capability card."""
        return self._cards.get(model_id)

    def has(self, model_id: str) -> bool:
        """Check if a model is registered."""
        return model_id in self._cards

    def revoke(self, model_id: str) -> bool:
        """Revoke a model's registration."""
        if model_id in self._cards:
            self._cards[model_id].revoked = True
            return True
        return False

    def list_all(self) -> list[CapabilityCard]:
        """List all registered cards."""
        return list(self._cards.values())

    def list_valid(self) -> list[CapabilityCard]:
        """List only valid cards."""
        return [c for c in self._cards.values() if c.is_valid()[0]]


class ModelLicenseGate:
    """
    Model License Gate - Validates CapabilityCards before inference.

    Part of the gate chain: SCHEMA → SNR → IHSAN → LICENSE

    Every model must have a valid, non-expired, non-revoked
    CapabilityCard to participate in inference.
    """

    def __init__(self, registry: ModelRegistry):
        """
        Initialize the license gate.

        Args:
            registry: Model registry containing capability cards
        """
        self.registry = registry

    def check(self, model_id: str) -> LicenseCheckResult:
        """
        Check if a model has a valid license.

        Args:
            model_id: The model to check

        Returns:
            LicenseCheckResult with allowed status and reason
        """
        # Check if model is registered
        card = self.registry.get(model_id)

        if card is None:
            logger.warning(f"Model not registered: {model_id}")
            return LicenseCheckResult(
                allowed=False,
                model_id=model_id,
                tier=None,
                reason="Model not registered. Run Constitution Challenge first.",
                card=None,
            )

        # Verify the card
        is_valid, reason = card.is_valid()

        if not is_valid:
            logger.warning(f"Invalid capability card for {model_id}: {reason}")
            return LicenseCheckResult(
                allowed=False,
                model_id=model_id,
                tier=card.tier,
                reason=f"CapabilityCard invalid: {reason}",
                card=card,
            )

        # All checks passed
        logger.debug(f"License check passed for {model_id}")
        return LicenseCheckResult(
            allowed=True,
            model_id=model_id,
            tier=card.tier,
            reason=None,
            card=card,
        )

    def check_for_task(
        self,
        model_id: str,
        task_type: TaskType,
    ) -> LicenseCheckResult:
        """
        Check if a model is licensed for a specific task.

        Args:
            model_id: The model to check
            task_type: The task type to validate

        Returns:
            LicenseCheckResult with allowed status and reason
        """
        # First run basic license check
        result = self.check(model_id)

        if not result.allowed:
            return result

        # Check task support
        assert result.card is not None  # Guaranteed by allowed=True
        if task_type not in result.card.capabilities.tasks_supported:
            return LicenseCheckResult(
                allowed=False,
                model_id=model_id,
                tier=result.card.tier,
                reason=f"Model not licensed for task: {task_type.value}",
                card=result.card,
            )

        return result


class GateChain:
    """
    Constitutional enforcement gate chain.

    Gates are applied in order: SCHEMA → SNR → IHSAN → LICENSE
    If any gate fails, the output is rejected.
    """

    def __init__(self, registry: Optional[ModelRegistry] = None):
        """
        Initialize the gate chain.

        Args:
            registry: Optional model registry for LICENSE gate
        """
        self.registry = registry or InMemoryRegistry()
        self.license_gate = ModelLicenseGate(self.registry)

    def validate_output(self, output: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate an inference output through all gates.

        Args:
            output: Inference output with:
                - content: str
                - model_id: str
                - ihsan_score: float (optional)
                - snr_score: float (optional)
                - schema_valid: bool (optional)
                - capability_card_valid: bool (optional)

        Returns:
            Gate result dictionary
        """
        # SCHEMA gate
        if not output.get("content") or not output.get("model_id"):
            return {
                "passed": False,
                "gate_name": "SCHEMA",
                "score": 0.0,
                "reason": "Invalid output schema: missing required fields",
            }

        # SNR gate
        snr_score = output.get("snr_score")
        if snr_score is None:
            return {
                "passed": False,
                "gate_name": "SNR",
                "score": 0.0,
                "reason": "SNR score not provided",
            }
        if snr_score < SNR_THRESHOLD:
            return {
                "passed": False,
                "gate_name": "SNR",
                "score": snr_score,
                "reason": f"SNR score {snr_score:.3f} below threshold {SNR_THRESHOLD}",
            }

        # IHSAN gate
        ihsan_score = output.get("ihsan_score")
        if ihsan_score is None:
            return {
                "passed": False,
                "gate_name": "IHSAN",
                "score": 0.0,
                "reason": "Ihsān score not provided",
            }
        if ihsan_score < IHSAN_THRESHOLD:
            return {
                "passed": False,
                "gate_name": "IHSAN",
                "score": ihsan_score,
                "reason": f"Ihsān score {ihsan_score:.3f} below threshold {IHSAN_THRESHOLD}",
            }

        # LICENSE gate
        model_id = output["model_id"]
        license_result = self.license_gate.check(model_id)
        if not license_result.allowed:
            return {
                "passed": False,
                "gate_name": "LICENSE",
                "score": 0.0,
                "reason": license_result.reason,
            }

        # All gates passed
        return {
            "passed": True,
            "gate_name": "ALL",
            "score": ihsan_score,
            "reason": None,
        }

    def validate_detailed(self, output: Dict[str, Any]) -> list[Dict[str, Any]]:
        """
        Get detailed results from all gates.

        Args:
            output: Inference output to validate

        Returns:
            List of gate results
        """
        results = []

        # SCHEMA gate
        schema_passed = bool(output.get("content") and output.get("model_id"))
        results.append({
            "gate": "SCHEMA",
            "passed": schema_passed,
            "score": 1.0 if schema_passed else 0.0,
            "reason": None if schema_passed else "Missing required fields",
        })

        # SNR gate
        snr_score = output.get("snr_score", 0.0)
        snr_passed = snr_score >= SNR_THRESHOLD
        results.append({
            "gate": "SNR",
            "passed": snr_passed,
            "score": snr_score,
            "reason": None if snr_passed else f"Below threshold {SNR_THRESHOLD}",
        })

        # IHSAN gate
        ihsan_score = output.get("ihsan_score", 0.0)
        ihsan_passed = ihsan_score >= IHSAN_THRESHOLD
        results.append({
            "gate": "IHSAN",
            "passed": ihsan_passed,
            "score": ihsan_score,
            "reason": None if ihsan_passed else f"Below threshold {IHSAN_THRESHOLD}",
        })

        # LICENSE gate
        model_id = output.get("model_id", "")
        if model_id:
            license_result = self.license_gate.check(model_id)
            results.append({
                "gate": "LICENSE",
                "passed": license_result.allowed,
                "score": 1.0 if license_result.allowed else 0.0,
                "reason": license_result.reason,
            })
        else:
            results.append({
                "gate": "LICENSE",
                "passed": False,
                "score": 0.0,
                "reason": "No model ID provided",
            })

        return results


# Convenience function for creating a gate chain
def create_gate_chain(registry: Optional[ModelRegistry] = None) -> GateChain:
    """
    Create a gate chain for constitutional enforcement.

    Args:
        registry: Optional model registry for LICENSE gate

    Returns:
        Configured GateChain instance
    """
    return GateChain(registry)
