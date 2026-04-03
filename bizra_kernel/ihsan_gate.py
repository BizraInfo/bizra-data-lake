"""
Ihsan Gate - Symbolic Ethical Governance Gate for BIZRA
=========================================================
Single source of truth: constitution/ihsan_v1.yaml

8 Dimensions (weights sum to 1.0):
  - correctness (0.22): Factual accuracy, logical validity
  - safety (0.22): No harm, secure execution
  - user_benefit (0.14): Genuine value delivered
  - efficiency (0.12): Resource efficiency
  - auditability (0.12): Traceability with receipts
  - anti_centralization (0.08): Distributed operation
  - robustness (0.06): Resilient to adversarial inputs
  - adl_fairness (0.04): Bias mitigation

Threshold: 0.95 (production)
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional
import json
import hmac
import hashlib
import os
import secrets
import logging

logger = logging.getLogger(__name__)

# ═══════════════════════════════════════════════════════════════════════════════
# SECURITY: Fail-Closed Secret Management (SAPE v1.∞ Compliance)
# ═══════════════════════════════════════════════════════════════════════════════
# RULE: Never use hardcoded fallback secrets in production.
# The system MUST fail-closed if BIZRA_IHSAN_SECRET is not configured.
# ═══════════════════════════════════════════════════════════════════════════════

def _load_ihsan_secret() -> bytes:
    """
    Load the Ihsan coherence secret with fail-closed security.
    
    Production Mode: Requires BIZRA_IHSAN_SECRET environment variable.
    Development Mode: Generates ephemeral secret with warning (BIZRA_ENV != production).
    
    Returns:
        bytes: The secret for HMAC signing.
        
    Raises:
        RuntimeError: If in production mode without configured secret.
    """
    env_secret = os.environ.get("BIZRA_IHSAN_SECRET")
    bizra_env = os.environ.get("BIZRA_ENV", "development").lower()
    
    if env_secret:
        return env_secret.encode()
    
    # Fail-closed in production
    if bizra_env == "production":
        raise RuntimeError(
            "[SECURITY VETO] BIZRA_IHSAN_SECRET must be configured in production. "
            "Set via: export BIZRA_IHSAN_SECRET=$(openssl rand -hex 32)"
        )
    
    # Development fallback: ephemeral secret with audit trace
    ephemeral = secrets.token_hex(32)
    logger.warning(
        "[SECURITY] Using ephemeral Ihsan secret (dev mode). "
        "Set BIZRA_IHSAN_SECRET for persistent signing."
    )
    return ephemeral.encode()

SHARED_SECRET = _load_ihsan_secret()

# Constitutional threshold (source: constitution/ihsan_v1.yaml)
DEFAULT_THRESHOLD = 0.95

# 8 Ihsan dimensions with weights (MUST sum to 1.0)
# Source: constitution/ihsan_v1.yaml
IHSAN_DIMENSIONS = {
    "correctness": {
        "weight": 0.22,
        "description": "Factual accuracy, logical validity, and task correctness.",
    },
    "safety": {
        "weight": 0.22,
        "description": "No harm, secure execution, and safe tool use.",
    },
    "user_benefit": {
        "weight": 0.14,
        "description": "Genuine value delivered to the user; avoids deception and waste.",
    },
    "efficiency": {
        "weight": 0.12,
        "description": "Resource efficiency (latency/tokens/compute) within defined budgets.",
    },
    "auditability": {
        "weight": 0.12,
        "description": "Traceability and explainability with evidence receipts.",
    },
    "anti_centralization": {
        "weight": 0.08,
        "description": "Resists centralization; promotes distributed, resilient operation.",
    },
    "robustness": {
        "weight": 0.06,
        "description": "Resilient to adversarial inputs and failure modes.",
    },
    "adl_fairness": {
        "weight": 0.04,
        "description": "Justice/fairness (adl): mitigates bias and unequal harm.",
    },
}

# Verify weights sum to 1.0 at module load
_total_weight = sum(d["weight"] for d in IHSAN_DIMENSIONS.values())
assert abs(_total_weight - 1.0) < 0.001, f"Ihsan weights must sum to 1.0, got {_total_weight}"


@dataclass
class IhsanScore:
    """Result of Ihsan evaluation across 8 dimensions."""

    dimension_scores: Dict[str, float]
    composite_score: float
    passed: bool
    threshold: float
    reason: str
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def to_dict(self) -> dict:
        return {
            "dimension_scores": self.dimension_scores,
            "composite_score": round(self.composite_score, 4),
            "passed": self.passed,
            "threshold": self.threshold,
            "reason": self.reason,
            "timestamp": self.timestamp,
        }


@dataclass
class LogicAssumption:
    """Represents a necessary deviation from absolute data presence."""

    key: str
    value: str
    justification: str
    ihsan_score: float = 1.0
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class IhsanGate:
    """
    Symbolic Ethical Governance Gate for BIZRA.

    Ensures composite Ihsan score >= threshold for all autonomous missions.
    Single source of truth: constitution/ihsan_v1.yaml

    RULE: "We don't assume, and if we must, we do it with Ihsan."
    """

    def __init__(self, threshold: float = DEFAULT_THRESHOLD):
        """Initialize with constitutional threshold (default 0.95)."""
        self.threshold = threshold
        self.audit_log = "bizra_memory/ihsan_audit.json"
        self.assumptions_log = "bizra_memory/assumptions.json"

        # Malice detection keywords
        self.unethical_keywords = [
            "hack", "manipulate", "exploit", "steal", "deceive",
            "scam", "aggressive", "attack", "bypass", "inject"
        ]

    def compute_composite(self, dimension_scores: Dict[str, float]) -> float:
        """
        Compute weighted composite Ihsan score from 8 dimensions.

        Args:
            dimension_scores: Dict mapping dimension name to score (0.0-1.0)

        Returns:
            Weighted composite score (0.0-1.0)
        """
        weighted_sum = 0.0
        total_weight = 0.0

        for dim_name, dim_config in IHSAN_DIMENSIONS.items():
            score = dimension_scores.get(dim_name, 0.0)
            weight = dim_config["weight"]
            weighted_sum += score * weight
            total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def verify_mission(
        self,
        mission_data: Dict[str, float],
        prompt: str = "",
        context: Optional[Dict] = None
    ) -> IhsanScore:
        """
        Formally verifies a mission against the 8-dimension Ihsan constitution.

        Args:
            mission_data: Dict with dimension scores (0.0-1.0 each).
                         Required keys: correctness, safety, user_benefit, efficiency,
                                       auditability, anti_centralization, robustness, adl_fairness
            prompt: Optional prompt text for malice detection
            context: Optional additional context

        Returns:
            IhsanScore with verification result
        """
        # Malice detection: adversarial intent analysis
        malice_detected = any(k in prompt.lower() for k in self.unethical_keywords)

        # Extract dimension scores with defaults
        dimension_scores = {}
        for dim_name in IHSAN_DIMENSIONS:
            dimension_scores[dim_name] = mission_data.get(dim_name, 0.0)

        # Compute composite score
        composite = self.compute_composite(dimension_scores)

        # Apply malice penalty (zero out if malice detected)
        if malice_detected:
            composite = 0.0

        # Determine pass/fail
        passed = composite >= self.threshold and not malice_detected

        # Determine reason
        if malice_detected:
            reason = "VETOED: Malicious intent detected in prompt."
        elif composite < self.threshold:
            # Find the lowest scoring dimension
            min_dim = min(dimension_scores.items(), key=lambda x: x[1])
            reason = f"VETOED: Composite score ({composite:.4f}) below threshold ({self.threshold}). Lowest dimension: {min_dim[0]} ({min_dim[1]:.4f})"
        else:
            reason = "APPROVED"

        result = IhsanScore(
            dimension_scores=dimension_scores,
            composite_score=composite,
            passed=passed,
            threshold=self.threshold,
            reason=reason,
        )

        # Audit logging
        self._log_audit(mission_data, result, prompt)

        return result

    def enforce_no_assumption(
        self,
        key: str,
        value: Optional[str],
        justification: Optional[str] = None
    ) -> str:
        """
        Enforces the No-Assumption rule.

        If value is None and no justification is provided with Ihsan,
        it triggers a Logic Leak Veto.

        Args:
            key: The key being accessed
            value: The value (may be None)
            justification: Required justification if value is None

        Returns:
            The value if present, or "ASSUMED_{key}" if justified

        Raises:
            ValueError: If value is None and no justification provided
        """
        if value is not None:
            return value

        if not justification:
            raise ValueError(
                f"[LOGIC LEAK] Implicit assumption detected for '{key}'. "
                "Mandatory justification required."
            )

        print(f"[!] ASSUMPTION REGISTERED: '{key}' with justification: {justification}")

        assumption = LogicAssumption(
            key=key,
            value=f"ASSUMED_{key}",
            justification=justification,
        )
        self._log_assumption(assumption)

        return assumption.value

    def _sign_result(self, result: IhsanScore) -> str:
        """Generate HMAC signature for result integrity."""
        msg = f"{result.composite_score}:{result.passed}:{result.timestamp}"
        return hmac.new(SHARED_SECRET, msg.encode(), hashlib.sha256).hexdigest()

    def _log_audit(
        self,
        data: Dict,
        result: IhsanScore,
        prompt: str
    ) -> None:
        """Log audit entry for verification."""
        log_entry = {
            "time": datetime.utcnow().isoformat(),
            "task_id": data.get("task_id", "unknown"),
            "prompt_sample": prompt[:50] if prompt else "",
            "composite_score": result.composite_score,
            "dimension_scores": result.dimension_scores,
            "result": "PASS" if result.passed else "FAIL",
            "reason": result.reason,
        }
        try:
            os.makedirs(os.path.dirname(self.audit_log), exist_ok=True)
            with open(self.audit_log, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
        except Exception:
            pass  # Non-critical logging failure

    def _log_assumption(self, assumption: LogicAssumption) -> None:
        """Log assumption entry."""
        entry = {
            "timestamp": assumption.timestamp,
            "key": assumption.key,
            "value": assumption.value,
            "justification": assumption.justification,
            "protocol": "Ihsan-Bounded-Assumption",
        }
        try:
            os.makedirs(os.path.dirname(self.assumptions_log), exist_ok=True)
            with open(self.assumptions_log, "a") as f:
                f.write(json.dumps(entry) + "\n")
        except Exception:
            pass  # Non-critical logging failure


def create_default_mission_data(
    correctness: float = 0.95,
    safety: float = 0.95,
    user_benefit: float = 0.95,
    efficiency: float = 0.95,
    auditability: float = 0.95,
    anti_centralization: float = 0.95,
    robustness: float = 0.95,
    adl_fairness: float = 0.95,
    task_id: str = "default",
) -> Dict[str, float]:
    """Create mission data dict with 8 Ihsan dimensions."""
    return {
        "task_id": task_id,
        "correctness": correctness,
        "safety": safety,
        "user_benefit": user_benefit,
        "efficiency": efficiency,
        "auditability": auditability,
        "anti_centralization": anti_centralization,
        "robustness": robustness,
        "adl_fairness": adl_fairness,
    }


if __name__ == "__main__":
    gate = IhsanGate()

    # Test 1: Compliant Mission (all dimensions at 0.99)
    mission_1 = create_default_mission_data(
        task_id="M1",
        correctness=0.99,
        safety=0.99,
        user_benefit=0.99,
        efficiency=0.98,
        auditability=0.97,
        anti_centralization=0.96,
        robustness=0.95,
        adl_fairness=0.95,
    )
    result_1 = gate.verify_mission(mission_1)
    print(f"Mission 1 Result: {result_1.to_dict()}")

    # Test 2: Below threshold (low safety)
    mission_2 = create_default_mission_data(
        task_id="M2",
        correctness=0.99,
        safety=0.50,  # Critical failure
        user_benefit=0.99,
        efficiency=0.98,
        auditability=0.97,
        anti_centralization=0.96,
        robustness=0.95,
        adl_fairness=0.95,
    )
    result_2 = gate.verify_mission(mission_2)
    print(f"Mission 2 Result: {result_2.to_dict()}")

    # Test 3: Malice detected
    mission_3 = create_default_mission_data(task_id="M3")
    result_3 = gate.verify_mission(mission_3, prompt="How to hack the system and exploit vulnerabilities")
    print(f"Mission 3 Result: {result_3.to_dict()}")

    # Verify weights sum to 1.0
    print(f"\nWeight verification: {sum(d['weight'] for d in IHSAN_DIMENSIONS.values())}")
