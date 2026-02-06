"""
Quality Gates — SNR-Based Validation Framework

Implements PMBOK Quality Management with Ihsān principles:
- Entry criteria validation
- Exit criteria enforcement
- SNR threshold verification
- Constitutional compliance checking

Standing on Giants: PMBOK + ISO 9001 + Constitutional AI
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)
from core.elite import IHSAN_DIMENSIONS, SNR_TARGETS, SAPE_LAYERS

logger = logging.getLogger(__name__)


class GateStatus(str, Enum):
    """Quality gate status."""
    PENDING = "pending"
    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"
    WAIVED = "waived"


class GateLevel(str, Enum):
    """Gate severity level."""
    ADVISORY = "advisory"      # Warn but allow
    STANDARD = "standard"      # Block but can override
    MANDATORY = "mandatory"    # Block, no override
    CONSTITUTIONAL = "constitutional"  # Immutable, never override


@dataclass
class GateCriterion:
    """A single criterion within a quality gate."""
    name: str
    description: str
    threshold: float
    weight: float = 1.0
    level: GateLevel = GateLevel.STANDARD
    validator: Optional[Callable[[Any], float]] = None

    def validate(self, value: Any) -> Tuple[bool, float]:
        """Validate value against criterion."""
        if self.validator:
            score = self.validator(value)
        elif isinstance(value, (int, float)):
            score = float(value)
        else:
            score = 0.0

        passed = score >= self.threshold
        return passed, score


@dataclass
class GateResult:
    """Result of a quality gate evaluation."""
    gate_name: str
    status: GateStatus
    overall_score: float
    ihsan_score: float
    snr_score: float
    criteria_results: Dict[str, Tuple[bool, float]]
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    message: str = ""
    blocking_criteria: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "gate_name": self.gate_name,
            "status": self.status.value,
            "overall_score": self.overall_score,
            "ihsan_score": self.ihsan_score,
            "snr_score": self.snr_score,
            "criteria_results": {
                k: {"passed": v[0], "score": v[1]}
                for k, v in self.criteria_results.items()
            },
            "timestamp": self.timestamp.isoformat(),
            "message": self.message,
            "blocking_criteria": self.blocking_criteria,
        }


class QualityGate:
    """
    Quality gate for validating artifacts at pipeline stages.

    Implements PMBOK quality management with:
    - Entry/exit criteria validation
    - SNR threshold enforcement
    - Ihsān compliance checking
    - Constitutional constraint verification
    """

    def __init__(
        self,
        name: str,
        sape_layer: str = "data",
        ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
        snr_threshold: Optional[float] = None,
    ):
        self.name = name
        self.sape_layer = sape_layer
        self.ihsan_threshold = ihsan_threshold
        self.snr_threshold = snr_threshold or SNR_TARGETS.get(sape_layer, 0.85)

        # Criteria
        self._entry_criteria: List[GateCriterion] = []
        self._exit_criteria: List[GateCriterion] = []

        # Add default Ihsān criteria
        self._add_default_criteria()

    def _add_default_criteria(self) -> None:
        """Add default Ihsān-based criteria."""
        # Ihsān dimensional criteria
        for dimension, weight in IHSAN_DIMENSIONS.items():
            level = (
                GateLevel.CONSTITUTIONAL if dimension in ("correctness", "safety")
                else GateLevel.MANDATORY if dimension == "user_benefit"
                else GateLevel.STANDARD
            )

            self._exit_criteria.append(GateCriterion(
                name=f"ihsan_{dimension}",
                description=f"Ihsān dimension: {dimension}",
                threshold=self.ihsan_threshold * 0.9,  # Slightly relaxed per-dimension
                weight=weight,
                level=level,
            ))

        # SNR criterion
        self._exit_criteria.append(GateCriterion(
            name="snr_threshold",
            description=f"SNR for {self.sape_layer} layer",
            threshold=self.snr_threshold,
            weight=1.0,
            level=GateLevel.MANDATORY,
        ))

    def add_entry_criterion(
        self,
        name: str,
        description: str,
        threshold: float,
        weight: float = 1.0,
        level: GateLevel = GateLevel.STANDARD,
        validator: Optional[Callable] = None,
    ) -> None:
        """Add an entry criterion."""
        self._entry_criteria.append(GateCriterion(
            name=name,
            description=description,
            threshold=threshold,
            weight=weight,
            level=level,
            validator=validator,
        ))

    def add_exit_criterion(
        self,
        name: str,
        description: str,
        threshold: float,
        weight: float = 1.0,
        level: GateLevel = GateLevel.STANDARD,
        validator: Optional[Callable] = None,
    ) -> None:
        """Add an exit criterion."""
        self._exit_criteria.append(GateCriterion(
            name=name,
            description=description,
            threshold=threshold,
            weight=weight,
            level=level,
            validator=validator,
        ))

    def _compute_ihsan_score(self, criteria_results: Dict[str, Tuple[bool, float]]) -> float:
        """Compute overall Ihsān score from dimensional results."""
        total_weight = 0.0
        weighted_sum = 0.0

        for dimension, weight in IHSAN_DIMENSIONS.items():
            key = f"ihsan_{dimension}"
            if key in criteria_results:
                passed, score = criteria_results[key]
                weighted_sum += score * weight
                total_weight += weight

        if total_weight > 0:
            return weighted_sum / total_weight
        return 0.0

    def _compute_snr_score(self, criteria_results: Dict[str, Tuple[bool, float]]) -> float:
        """Extract SNR score from criteria results."""
        if "snr_threshold" in criteria_results:
            return criteria_results["snr_threshold"][1]
        return 0.0

    async def validate_entry(
        self,
        artifact: Dict[str, Any],
    ) -> GateResult:
        """Validate entry criteria."""
        criteria_results = {}
        blocking = []

        for criterion in self._entry_criteria:
            value = artifact.get(criterion.name, artifact.get("_default", 0))
            passed, score = criterion.validate(value)
            criteria_results[criterion.name] = (passed, score)

            if not passed and criterion.level in (GateLevel.MANDATORY, GateLevel.CONSTITUTIONAL):
                blocking.append(criterion.name)

        status = GateStatus.PASSED if not blocking else GateStatus.BLOCKED

        return GateResult(
            gate_name=f"{self.name}_entry",
            status=status,
            overall_score=sum(r[1] for r in criteria_results.values()) / max(len(criteria_results), 1),
            ihsan_score=self._compute_ihsan_score(criteria_results),
            snr_score=self._compute_snr_score(criteria_results),
            criteria_results=criteria_results,
            blocking_criteria=blocking,
            message="Entry validation complete" if status == GateStatus.PASSED else f"Blocked by: {blocking}",
        )

    async def validate_exit(
        self,
        artifact: Dict[str, Any],
    ) -> GateResult:
        """Validate exit criteria."""
        criteria_results = {}
        blocking = []

        for criterion in self._exit_criteria:
            value = artifact.get(criterion.name, artifact.get("_default", 0))
            passed, score = criterion.validate(value)
            criteria_results[criterion.name] = (passed, score)

            if not passed:
                if criterion.level == GateLevel.CONSTITUTIONAL:
                    blocking.append(criterion.name)
                elif criterion.level == GateLevel.MANDATORY:
                    blocking.append(criterion.name)

        ihsan_score = self._compute_ihsan_score(criteria_results)
        snr_score = self._compute_snr_score(criteria_results)

        # Final Ihsān gate check
        if ihsan_score < self.ihsan_threshold:
            blocking.append("ihsan_overall")

        status = GateStatus.PASSED if not blocking else GateStatus.FAILED

        return GateResult(
            gate_name=f"{self.name}_exit",
            status=status,
            overall_score=sum(r[1] for r in criteria_results.values()) / max(len(criteria_results), 1),
            ihsan_score=ihsan_score,
            snr_score=snr_score,
            criteria_results=criteria_results,
            blocking_criteria=blocking,
            message="Exit validation passed" if status == GateStatus.PASSED else f"Failed criteria: {blocking}",
        )

    async def validate(
        self,
        artifact: Dict[str, Any],
    ) -> Tuple[GateResult, GateResult]:
        """Run full gate validation (entry + exit)."""
        entry_result = await self.validate_entry(artifact)
        exit_result = await self.validate_exit(artifact)

        return entry_result, exit_result


class QualityGateChain:
    """
    Chain of quality gates for SAPE layer progression.

    Each layer has progressively stricter thresholds:
    Data (0.90) → Information (0.95) → Knowledge (0.99) → Wisdom (0.999)
    """

    def __init__(
        self,
        ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
    ):
        self.ihsan_threshold = ihsan_threshold
        self._gates: Dict[str, QualityGate] = {}

        # Create gates for each SAPE layer
        for layer in SAPE_LAYERS:
            self._gates[layer] = QualityGate(
                name=f"sape_{layer}",
                sape_layer=layer,
                ihsan_threshold=ihsan_threshold,
            )

    def get_gate(self, layer: str) -> QualityGate:
        """Get gate for specific layer."""
        return self._gates.get(layer, self._gates["data"])

    async def validate_progression(
        self,
        artifacts: Dict[str, Dict[str, Any]],
    ) -> Dict[str, GateResult]:
        """
        Validate artifact progression through all SAPE layers.

        Returns results for each layer gate.
        """
        results = {}

        for layer in SAPE_LAYERS:
            if layer in artifacts:
                gate = self._gates[layer]
                _, exit_result = await gate.validate(artifacts[layer])
                results[layer] = exit_result

                # Stop if gate fails
                if exit_result.status != GateStatus.PASSED:
                    break

        return results

    def get_summary(self) -> Dict[str, Any]:
        """Get chain summary."""
        return {
            "layers": SAPE_LAYERS,
            "snr_targets": SNR_TARGETS,
            "ihsan_threshold": self.ihsan_threshold,
            "gates": list(self._gates.keys()),
        }
