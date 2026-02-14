"""
SNR Engine — Signal-to-Noise Ratio as Verifiable Function

SNR = SignalMass / (SignalMass + NoiseMass)

Where SignalMass/NoiseMass are auditable totals from:
- Provenance depth + corroboration count
- Constraint satisfiability (Z3 pass/fail)
- Contradiction rate
- Predictive success
- Context-fit score

Every SNR computation produces a trace for audit.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from core.proof_engine import DEFAULT_SNR_POLICY
from core.proof_engine.canonical import blake3_digest, canonical_bytes


@dataclass
class SNRPolicy:
    """
    SNR computation policy.

    Defines weights and thresholds for signal/noise calculation.
    """

    snr_min: float = 0.95
    contradiction_penalty: float = 1.0
    unverifiable_penalty: float = 0.5
    provenance_weight: float = 0.3
    constraint_weight: float = 0.4
    prediction_weight: float = 0.3

    def canonical_bytes(self) -> bytes:
        """Get deterministic byte representation."""
        data = {
            "snr_min": self.snr_min,
            "contradiction_penalty": self.contradiction_penalty,
            "unverifiable_penalty": self.unverifiable_penalty,
            "provenance_weight": self.provenance_weight,
            "constraint_weight": self.constraint_weight,
            "prediction_weight": self.prediction_weight,
        }
        return canonical_bytes(data)

    def digest(self) -> bytes:
        """Compute policy digest."""
        return blake3_digest(self.canonical_bytes())

    def hex_digest(self) -> str:
        """Compute hex-encoded digest."""
        return self.digest().hex()

    @classmethod
    def default(cls) -> "SNRPolicy":
        """Create default policy."""
        return cls(**DEFAULT_SNR_POLICY)


@dataclass
class SNRTrace:
    """
    Audit trace for SNR computation.

    Captures all inputs and intermediate values for verification.
    """

    # Input scores
    provenance_score: float
    constraint_score: float
    prediction_score: float

    # Noise factors
    contradiction_mass: float
    unverifiable_mass: float

    # Computed values
    signal_mass: float
    noise_mass: float
    snr: float

    # Audit metadata
    policy_digest: str
    computation_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "provenance_score": self.provenance_score,
            "constraint_score": self.constraint_score,
            "prediction_score": self.prediction_score,
            "contradiction_mass": self.contradiction_mass,
            "unverifiable_mass": self.unverifiable_mass,
            "signal_mass": self.signal_mass,
            "noise_mass": self.noise_mass,
            "snr": self.snr,
            "policy_digest": self.policy_digest,
            "computation_id": self.computation_id,
            "timestamp": self.timestamp.isoformat(),
        }

    def canonical_bytes(self) -> bytes:
        """Get deterministic byte representation."""
        return canonical_bytes(self.to_dict())

    def digest(self) -> bytes:
        """Compute trace digest."""
        return blake3_digest(self.canonical_bytes())


@dataclass
class SNRInput:
    """
    Input bundle for SNR computation.

    All factors that contribute to signal/noise.
    """

    # Provenance factors
    provenance_depth: int = 0
    corroboration_count: int = 0
    source_trust_score: float = 0.5

    # Constraint factors
    z3_satisfiable: bool = True
    ihsan_score: float = 0.95
    constraint_violations: int = 0

    # Contradiction factors
    contradiction_count: int = 0
    conflicting_sources: int = 0

    # Prediction factors
    prediction_accuracy: float = 0.5
    context_fit_score: float = 0.5

    # Unverifiable factors
    unverifiable_claims: int = 0
    missing_citations: int = 0

    def compute_provenance_score(self) -> float:
        """Compute normalized provenance score."""
        depth_score = min(self.provenance_depth / 5, 1.0)
        corroboration_score = min(self.corroboration_count / 3, 1.0)
        return (
            0.3 * depth_score
            + 0.3 * corroboration_score
            + 0.4 * self.source_trust_score
        )

    def compute_constraint_score(self) -> float:
        """Compute normalized constraint score."""
        z3_score = 1.0 if self.z3_satisfiable else 0.0
        violation_penalty = min(self.constraint_violations * 0.1, 0.5)
        return 0.4 * z3_score + 0.4 * self.ihsan_score + 0.2 * (1.0 - violation_penalty)

    def compute_prediction_score(self) -> float:
        """Compute normalized prediction score."""
        return 0.6 * self.prediction_accuracy + 0.4 * self.context_fit_score

    def compute_contradiction_mass(self) -> float:
        """Compute contradiction noise mass."""
        return self.contradiction_count * 0.2 + self.conflicting_sources * 0.3

    def compute_unverifiable_mass(self) -> float:
        """Compute unverifiable noise mass."""
        return self.unverifiable_claims * 0.15 + self.missing_citations * 0.1


class SNREngine:
    """
    SNR computation engine with full audit trail.

    Computes SNR as a verifiable function with trace.
    """

    def __init__(self, policy: Optional[SNRPolicy] = None):
        self.policy = policy or SNRPolicy.default()
        self._computations: List[SNRTrace] = []
        self._computation_counter = 0

    def compute(
        self,
        inputs: SNRInput,
    ) -> Tuple[float, SNRTrace]:
        """
        Compute SNR with full audit trace.

        Returns (snr_value, trace).
        """
        self._computation_counter += 1
        computation_id = f"snr_{self._computation_counter:08d}"

        # Compute component scores
        provenance_score = inputs.compute_provenance_score()
        constraint_score = inputs.compute_constraint_score()
        prediction_score = inputs.compute_prediction_score()
        contradiction_mass = inputs.compute_contradiction_mass()
        unverifiable_mass = inputs.compute_unverifiable_mass()

        # Compute signal mass (weighted sum of positive factors)
        signal_mass = (
            self.policy.provenance_weight * provenance_score
            + self.policy.constraint_weight * constraint_score
            + self.policy.prediction_weight * prediction_score
        )

        # Compute noise mass (weighted sum of negative factors)
        noise_mass = (
            self.policy.contradiction_penalty * contradiction_mass
            + self.policy.unverifiable_penalty * unverifiable_mass
        )

        # Compute SNR
        total_mass = signal_mass + noise_mass
        if total_mass <= 0:
            snr = 0.0
        else:
            snr = signal_mass / total_mass

        # Clamp to [0, 1]
        snr = max(0.0, min(1.0, snr))

        # Create trace
        trace = SNRTrace(
            provenance_score=provenance_score,
            constraint_score=constraint_score,
            prediction_score=prediction_score,
            contradiction_mass=contradiction_mass,
            unverifiable_mass=unverifiable_mass,
            signal_mass=signal_mass,
            noise_mass=noise_mass,
            snr=snr,
            policy_digest=self.policy.hex_digest(),
            computation_id=computation_id,
        )

        self._computations.append(trace)

        return snr, trace

    def snr_score(
        self,
        inputs: SNRInput,
    ) -> Dict[str, Any]:
        """
        Single authoritative SNR scorer — receipt-compatible output shape.

        Returns the canonical dict matching the receipt.snr schema.
        Parallel to IhsanGate.ihsan_score().

        Standing on: SP-004 (SNR Engine v1 authoritative scorer)
        """
        snr, trace = self.compute(inputs)

        return {
            "score": trace.snr,
            "signal_mass": trace.signal_mass,
            "noise_mass": trace.noise_mass,
            "signal_components": {
                "provenance": trace.provenance_score,
                "constraint": trace.constraint_score,
                "prediction": trace.prediction_score,
            },
            "noise_components": {
                "contradiction": trace.contradiction_mass,
                "unverifiable": trace.unverifiable_mass,
            },
            "claim_tags": {
                "snr": "measured",
                "signal_mass": "measured",
            },
            "trace_id": trace.computation_id,
            "policy_digest": trace.policy_digest,
            "passed": snr >= self.policy.snr_min,
        }

    def check_threshold(
        self,
        inputs: SNRInput,
    ) -> Tuple[bool, float, SNRTrace]:
        """
        Check if SNR meets minimum threshold.

        Returns (passed, snr_value, trace).
        """
        snr, trace = self.compute(inputs)
        passed = snr >= self.policy.snr_min
        return passed, snr, trace

    def compute_simple(
        self,
        provenance_score: float,
        constraint_score: float,
        contradiction_mass: float = 0.0,
        unverifiable_mass: float = 0.0,
        prediction_score: float = 0.5,
    ) -> Tuple[float, SNRTrace]:
        """
        Simplified computation with direct scores.

        For cases where input aggregation is done externally.
        """
        self._computation_counter += 1
        computation_id = f"snr_{self._computation_counter:08d}"

        # Compute signal mass
        signal_mass = (
            self.policy.provenance_weight * provenance_score
            + self.policy.constraint_weight * constraint_score
            + self.policy.prediction_weight * prediction_score
        )

        # Compute noise mass
        noise_mass = (
            self.policy.contradiction_penalty * contradiction_mass
            + self.policy.unverifiable_penalty * unverifiable_mass
        )

        # Compute SNR
        total_mass = signal_mass + noise_mass
        snr = signal_mass / total_mass if total_mass > 0 else 0.0
        snr = max(0.0, min(1.0, snr))

        # Create trace
        trace = SNRTrace(
            provenance_score=provenance_score,
            constraint_score=constraint_score,
            prediction_score=prediction_score,
            contradiction_mass=contradiction_mass,
            unverifiable_mass=unverifiable_mass,
            signal_mass=signal_mass,
            noise_mass=noise_mass,
            snr=snr,
            policy_digest=self.policy.hex_digest(),
            computation_id=computation_id,
        )

        self._computations.append(trace)

        return snr, trace

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        if not self._computations:
            return {
                "total_computations": 0,
                "policy_digest": self.policy.hex_digest(),
            }

        snr_values = [c.snr for c in self._computations]
        passed_count = sum(1 for s in snr_values if s >= self.policy.snr_min)

        return {
            "total_computations": len(self._computations),
            "avg_snr": sum(snr_values) / len(snr_values),
            "min_snr": min(snr_values),
            "max_snr": max(snr_values),
            "pass_rate": passed_count / len(self._computations),
            "policy_digest": self.policy.hex_digest(),
        }

    def get_recent_traces(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent computation traces."""
        return [t.to_dict() for t in self._computations[-limit:]]

    def verify_trace(self, trace: SNRTrace) -> bool:
        """
        Verify a trace is consistent with policy.

        Re-computes SNR and checks for match.
        """
        # Re-compute signal mass
        signal_mass = (
            self.policy.provenance_weight * trace.provenance_score
            + self.policy.constraint_weight * trace.constraint_score
            + self.policy.prediction_weight * trace.prediction_score
        )

        # Re-compute noise mass
        noise_mass = (
            self.policy.contradiction_penalty * trace.contradiction_mass
            + self.policy.unverifiable_penalty * trace.unverifiable_mass
        )

        # Re-compute SNR
        total_mass = signal_mass + noise_mass
        expected_snr = signal_mass / total_mass if total_mass > 0 else 0.0
        expected_snr = max(0.0, min(1.0, expected_snr))

        # Check consistency
        snr_match = abs(trace.snr - expected_snr) < 0.0001
        signal_match = abs(trace.signal_mass - signal_mass) < 0.0001
        noise_match = abs(trace.noise_mass - noise_mass) < 0.0001
        policy_match = trace.policy_digest == self.policy.hex_digest()

        return snr_match and signal_match and noise_match and policy_match
