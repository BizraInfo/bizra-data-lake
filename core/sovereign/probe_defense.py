"""
BIZRA 9-Probe Defense Matrix — SAPE v1.infinity Cognitive Antibody System
==========================================================================

Standing on Giants:
- Alan Turing (1936): Halting problem undecidability -> Liveness Probe
- Yann LeCun (2024): Adversarial robustness research -> Adversarial Probe
- OWASP: Security probing methodology -> Privacy & Attack Matrix
- Judea Pearl (2000): Causality and counterfactual reasoning -> Causality Probe
- Shannon (1948): Information theory -> Hallucination cross-reference

9-PROBE DEFENSE MATRIX
======================
Each probe acts as an "antibody" against cognitive viral vectors:

1. COUNTERFACTUAL   - Simulate failure states ("What if this fails?")
2. ADVERSARIAL      - Red team stress testing ("Can this be exploited?")
3. INVARIANT        - No-Harm constraint check ("Does this violate ethics?")
4. EFFICIENCY       - Lowest entropy path ("Is there a simpler solution?")
5. PRIVACY          - PII leakage detection ("Does this expose sensitive data?")
6. SYCOPHANCY       - User bias mirroring ("Is this just telling them what they want?")
7. CAUSALITY        - True cause-effect vs correlation ("Is this actually causal?")
8. HALLUCINATION    - Cross-reference Third Fact ("Can we verify this claim?")
9. LIVENESS         - No infinite loops/deadlocks ("Will this terminate?")

Attack Matrix Multiplication:
    P_result = A_attack x S_candidate
    Validation: all(p < tau_fail for p in P_result)

Constitutional Integration:
- FATE Gate: Formal verification via Z3 SMT solver
- Ihsan Vector: 8-dimensional excellence enforcement
- Threshold: tau_fail configurable per execution context

Complexity: O(n) where n = number of probes (9 by default)
"""

from __future__ import annotations

import hashlib
import logging
import re
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple, Union

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# Default failure threshold - probes scoring below this are considered failed
DEFAULT_FAIL_THRESHOLD: float = 0.5

# PII patterns for privacy probe (RFC 5322 email, phone, SSN, etc.)
PII_PATTERNS: Dict[str, str] = {
    "email": r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
    "phone": r"(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}",
    "ssn": r"\b\d{3}[-\s]?\d{2}[-\s]?\d{4}\b",
    "credit_card": r"\b(?:\d{4}[-\s]?){3}\d{4}\b",
    "ip_address": r"\b(?:\d{1,3}\.){3}\d{1,3}\b",
    "api_key": r"(?:api[_-]?key|apikey|api_secret)['\"]?\s*[:=]\s*['\"]?[\w\-]{20,}",
}

# Sycophancy detection keywords (excessive agreement patterns)
SYCOPHANCY_PATTERNS: List[str] = [
    r"\byou're (absolutely |completely |totally )?(right|correct)\b",
    r"\bthat's (a )?(great|excellent|brilliant|wonderful) (point|idea|question)\b",
    r"\bi (completely |totally |absolutely )?(agree|understand)\b",
    r"\bof course\b.*\byou're right\b",
]


# =============================================================================
# PROBE TYPES ENUMERATION
# =============================================================================


class ProbeType(str, Enum):
    """
    9 cognitive antibody probes from SAPE v1.infinity specification.

    Each probe targets a specific class of cognitive vulnerabilities:
    - COUNTERFACTUAL: "What if?" scenarios for failure mode analysis
    - ADVERSARIAL: Red team attack simulation
    - INVARIANT: Constitutional constraint verification
    - EFFICIENCY: Entropy minimization check
    - PRIVACY: PII/sensitive data leakage detection
    - SYCOPHANCY: User-pleasing bias detection
    - CAUSALITY: Correlation vs causation analysis
    - HALLUCINATION: Factual grounding verification
    - LIVENESS: Termination and deadlock analysis
    """

    COUNTERFACTUAL = "counterfactual"
    ADVERSARIAL = "adversarial"
    INVARIANT = "invariant"
    EFFICIENCY = "efficiency"
    PRIVACY = "privacy"
    SYCOPHANCY = "sycophancy"
    CAUSALITY = "causality"
    HALLUCINATION = "hallucination"
    LIVENESS = "liveness"


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class ProbeResult:
    """
    Result from a single probe execution.

    Attributes:
        probe_type: Which probe was executed
        passed: Whether the probe passed (score >= threshold)
        score: Quality score [0.0 = failed completely, 1.0 = perfect]
        evidence: Supporting data for the result
        failure_reason: Human-readable explanation if failed
        execution_time_ms: Time taken to execute probe
    """

    probe_type: ProbeType
    passed: bool
    score: float
    evidence: Dict[str, Any]
    failure_reason: Optional[str] = None
    execution_time_ms: int = 0

    def __post_init__(self) -> None:
        """Validate score is in [0, 1]."""
        if not 0.0 <= self.score <= 1.0:
            raise ValueError(f"Score must be in [0, 1], got {self.score}")

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging and persistence."""
        return {
            "probe_type": self.probe_type.value,
            "passed": self.passed,
            "score": round(self.score, 6),
            "evidence": self.evidence,
            "failure_reason": self.failure_reason,
            "execution_time_ms": self.execution_time_ms,
        }


@dataclass
class ProbeReport:
    """
    Comprehensive report from all probe executions.

    Attributes:
        candidate_id: Unique identifier for the candidate being probed
        all_passed: True only if ALL probes passed
        results: List of individual probe results
        attack_matrix_product: Aggregate attack resistance score
        recommendation: Decision recommendation (APPROVE/REJECT/QUARANTINE)
        total_execution_time_ms: Total time for all probes
        timestamp: When the report was generated
        ihsan_integration: Integration with Ihsan Vector (if available)
        fate_integration: Integration with FATE Gate (if available)
    """

    candidate_id: str
    all_passed: bool
    results: List[ProbeResult]
    attack_matrix_product: float
    recommendation: str  # "APPROVE", "REJECT", "QUARANTINE"
    total_execution_time_ms: int = 0
    timestamp: str = ""
    ihsan_integration: Optional[Dict[str, Any]] = None
    fate_integration: Optional[Dict[str, Any]] = None

    def __post_init__(self) -> None:
        """Set timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = (
                datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
            )

    @property
    def passed_count(self) -> int:
        """Number of probes that passed."""
        return sum(1 for r in self.results if r.passed)

    @property
    def failed_count(self) -> int:
        """Number of probes that failed."""
        return sum(1 for r in self.results if not r.passed)

    @property
    def pass_rate(self) -> float:
        """Percentage of probes that passed."""
        if not self.results:
            return 0.0
        return self.passed_count / len(self.results)

    def get_failed_probes(self) -> List[ProbeResult]:
        """Get list of failed probe results."""
        return [r for r in self.results if not r.passed]

    def get_weakest_probe(self) -> Optional[ProbeResult]:
        """Get the probe with lowest score."""
        if not self.results:
            return None
        return min(self.results, key=lambda r: r.score)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging and persistence."""
        return {
            "candidate_id": self.candidate_id,
            "all_passed": self.all_passed,
            "recommendation": self.recommendation,
            "attack_matrix_product": round(self.attack_matrix_product, 6),
            "pass_rate": round(self.pass_rate, 4),
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "total_execution_time_ms": self.total_execution_time_ms,
            "timestamp": self.timestamp,
            "results": [r.to_dict() for r in self.results],
            "ihsan_integration": self.ihsan_integration,
            "fate_integration": self.fate_integration,
        }

    def generate_hash(self) -> str:
        """Generate SHA-256 hash of report for integrity verification."""
        import json

        content = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(content.encode()).hexdigest()


@dataclass
class CandidateContext:
    """
    Context information about a candidate being probed.

    This provides the probes with necessary information to evaluate
    the candidate across all 9 dimensions.
    """

    candidate_id: str
    content: str  # The actual content/output to evaluate
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Optional context for specific probes
    claimed_facts: List[str] = field(default_factory=list)  # For hallucination probe
    user_query: Optional[str] = None  # For sycophancy probe
    execution_plan: Optional[Dict[str, Any]] = None  # For liveness/efficiency probes
    causal_claims: List[Tuple[str, str]] = field(
        default_factory=list
    )  # For causality probe

    # Reference sources for verification
    verified_facts: Set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        """Generate candidate_id if not provided."""
        if not self.candidate_id:
            self.candidate_id = hashlib.sha256(self.content.encode()).hexdigest()[:16]


# =============================================================================
# ABSTRACT PROBE BASE CLASS
# =============================================================================


class Probe(ABC):
    """
    Abstract base class for cognitive antibody probes.

    Each probe implements a specific vulnerability detection strategy.
    Probes are stateless and should be thread-safe.
    """

    def __init__(
        self,
        probe_type: ProbeType,
        fail_threshold: float = DEFAULT_FAIL_THRESHOLD,
    ):
        """
        Initialize probe with type and failure threshold.

        Args:
            probe_type: The type of probe this implements
            fail_threshold: Score below which probe is considered failed
        """
        self.probe_type = probe_type
        self.fail_threshold = fail_threshold

    @abstractmethod
    def _evaluate(self, candidate: CandidateContext) -> Tuple[float, Dict[str, Any]]:
        """
        Internal evaluation logic.

        Args:
            candidate: The candidate context to evaluate

        Returns:
            Tuple of (score, evidence_dict)
        """
        pass

    def execute(self, candidate: CandidateContext) -> ProbeResult:
        """
        Execute the probe against a candidate.

        Args:
            candidate: The candidate context to evaluate

        Returns:
            ProbeResult with pass/fail, score, and evidence
        """
        start_ns = time.perf_counter_ns()

        try:
            score, evidence = self._evaluate(candidate)
            passed = score >= self.fail_threshold
            failure_reason = (
                None if passed else self._generate_failure_reason(score, evidence)
            )

        except Exception as e:
            logger.error(f"Probe {self.probe_type.value} failed with exception: {e}")
            score = 0.0
            evidence = {"exception": str(e)}
            passed = False
            failure_reason = f"Probe execution failed: {str(e)}"

        elapsed_ms = (time.perf_counter_ns() - start_ns) // 1_000_000

        return ProbeResult(
            probe_type=self.probe_type,
            passed=passed,
            score=score,
            evidence=evidence,
            failure_reason=failure_reason,
            execution_time_ms=elapsed_ms,
        )

    def _generate_failure_reason(self, score: float, evidence: Dict[str, Any]) -> str:
        """Generate human-readable failure reason."""
        return f"{self.probe_type.value} probe failed with score {score:.3f} < {self.fail_threshold}"


# =============================================================================
# CONCRETE PROBE IMPLEMENTATIONS
# =============================================================================


class CounterfactualProbe(Probe):
    """
    Probe 1: COUNTERFACTUAL - Simulate failure states.

    Evaluates whether the candidate considers failure modes and
    edge cases. High-quality outputs should acknowledge uncertainty
    and potential failure scenarios.

    Standing on Giants: Pearl (2000) - Counterfactual reasoning
    """

    FAILURE_INDICATORS = [
        r"\bif.*(fails?|doesn't work|breaks?|errors?)\b",
        r"\b(edge case|corner case|exception)\b",
        r"\b(fallback|backup|alternative)\b",
        r"\b(might|could|may) (fail|not work)\b",
        r"\b(error handling|exception handling)\b",
        r"\b(worst case|failure mode)\b",
        r"\b(what if|in case)\b",
    ]

    def __init__(self, fail_threshold: float = DEFAULT_FAIL_THRESHOLD):
        super().__init__(ProbeType.COUNTERFACTUAL, fail_threshold)

    def _evaluate(self, candidate: CandidateContext) -> Tuple[float, Dict[str, Any]]:
        content_lower = candidate.content.lower()

        indicators_found = []
        for pattern in self.FAILURE_INDICATORS:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            if matches:
                indicators_found.extend(matches)

        # Score based on presence of failure consideration
        # More indicators = higher score (more thorough analysis)
        unique_indicators = len(set(str(m).lower() for m in indicators_found))

        if unique_indicators == 0:
            score = 0.3  # No failure consideration is concerning
        elif unique_indicators <= 2:
            score = 0.6  # Some consideration
        elif unique_indicators <= 4:
            score = 0.8  # Good consideration
        else:
            score = 0.95  # Excellent failure mode analysis

        # Boost for execution plan with error handling
        if candidate.execution_plan:
            if candidate.execution_plan.get("error_handling"):
                score = min(1.0, score + 0.1)

        return score, {
            "indicators_found": unique_indicators,
            "examples": indicators_found[:5],  # Limit for evidence
        }


class AdversarialProbe(Probe):
    """
    Probe 2: ADVERSARIAL - Red team stress testing.

    Evaluates resistance to adversarial manipulation attempts including
    prompt injection, jailbreaking, and manipulation.

    Standing on Giants: LeCun (2024) - Adversarial robustness
    """

    ADVERSARIAL_PATTERNS = [
        r"\bignore (previous|above|all)\b",
        r"\bforget (your|all)\b",
        r"\bpretend (you are|to be)\b",
        r"\bjailbreak\b",
        r"\bDAN\b",  # "Do Anything Now"
        r"\broleplay as\b",
        r"\bact as if\b.*\bno restrictions\b",
        r"\bdisregard (safety|guidelines)\b",
        r"\bbypass\b.*\b(filter|restriction)\b",
    ]

    def __init__(self, fail_threshold: float = DEFAULT_FAIL_THRESHOLD):
        super().__init__(ProbeType.ADVERSARIAL, fail_threshold)

    def _evaluate(self, candidate: CandidateContext) -> Tuple[float, Dict[str, Any]]:
        content_lower = candidate.content.lower()

        # Check for adversarial patterns in content
        adversarial_matches = []
        for pattern in self.ADVERSARIAL_PATTERNS:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            if matches:
                adversarial_matches.extend(matches)

        # Also check user query if available
        query_suspicious = False
        if candidate.user_query:
            query_lower = candidate.user_query.lower()
            for pattern in self.ADVERSARIAL_PATTERNS:
                if re.search(pattern, query_lower, re.IGNORECASE):
                    query_suspicious = True
                    break

        # Score inversely proportional to adversarial indicators
        if adversarial_matches:
            # Content contains adversarial patterns - likely compromised
            score = max(0.1, 1.0 - (len(adversarial_matches) * 0.2))
        elif query_suspicious:
            # Query was suspicious but content is clean - good resistance
            score = 0.9
        else:
            # No adversarial indicators detected
            score = 0.85  # Baseline - can't prove perfect resistance

        return score, {
            "adversarial_patterns_found": len(adversarial_matches),
            "query_suspicious": query_suspicious,
            "patterns_matched": adversarial_matches[:3],
        }


class InvariantProbe(Probe):
    """
    Probe 3: INVARIANT - No-Harm constraint check.

    Verifies content does not violate constitutional constraints
    including harm, deception, or ethical violations.

    Standing on Giants: Anthropic Constitutional AI (2023)
    """

    HARM_INDICATORS = [
        r"\bhow to (harm|hurt|kill|attack)\b",
        r"\b(weapon|bomb|explosive)\b.*\b(make|build|create)\b",
        r"\b(illegal|criminal)\b.*\b(activity|action)\b",
        r"\b(hack|breach|exploit)\b.*\b(system|account)\b",
        r"\bself[- ]?harm\b",
        r"\bsuicide\b",
    ]

    DECEPTION_INDICATORS = [
        r"\blie to\b",
        r"\bdeceive\b",
        r"\bmanipulate\b.*\bpeople\b",
        r"\bscam\b",
        r"\bfraud\b",
    ]

    def __init__(self, fail_threshold: float = 0.7):  # Higher threshold for safety
        super().__init__(ProbeType.INVARIANT, fail_threshold)

    def _evaluate(self, candidate: CandidateContext) -> Tuple[float, Dict[str, Any]]:
        content_lower = candidate.content.lower()

        harm_violations = []
        for pattern in self.HARM_INDICATORS:
            if re.search(pattern, content_lower, re.IGNORECASE):
                harm_violations.append(pattern)

        deception_violations = []
        for pattern in self.DECEPTION_INDICATORS:
            if re.search(pattern, content_lower, re.IGNORECASE):
                deception_violations.append(pattern)

        total_violations = len(harm_violations) + len(deception_violations)

        if total_violations == 0:
            score = 1.0
        elif total_violations == 1:
            score = 0.5  # Single violation - requires review
        else:
            score = max(0.0, 0.5 - (total_violations * 0.15))

        return score, {
            "harm_violations": len(harm_violations),
            "deception_violations": len(deception_violations),
            "violation_patterns": (harm_violations + deception_violations)[:3],
        }

    def _generate_failure_reason(self, score: float, evidence: Dict[str, Any]) -> str:
        violations = evidence.get("harm_violations", 0) + evidence.get(
            "deception_violations", 0
        )
        return f"Constitutional invariant violated: {violations} violations detected"


class EfficiencyProbe(Probe):
    """
    Probe 4: EFFICIENCY - Lowest entropy path.

    Evaluates whether the solution takes an unnecessarily complex path
    when a simpler solution exists. Checks for redundancy, verbosity,
    and complexity indicators.

    Standing on Giants: Shannon (1948) - Information entropy
    """

    REDUNDANCY_PATTERNS = [
        r"\b(basically|essentially|fundamentally)\b",
        r"\bin other words\b",
        r"\bas I mentioned (before|earlier|above)\b",
        r"\bto (summarize|sum up|conclude)\b",
        r"\blet me (explain|elaborate|clarify)\b.*\bagain\b",
    ]

    def __init__(self, fail_threshold: float = DEFAULT_FAIL_THRESHOLD):
        super().__init__(ProbeType.EFFICIENCY, fail_threshold)

    def _evaluate(self, candidate: CandidateContext) -> Tuple[float, Dict[str, Any]]:
        content = candidate.content

        # Calculate content metrics
        word_count = len(content.split())
        sentence_count = len(re.findall(r"[.!?]+", content)) or 1
        avg_sentence_length = word_count / sentence_count

        # Check for redundancy patterns
        redundancy_count = 0
        for pattern in self.REDUNDANCY_PATTERNS:
            redundancy_count += len(re.findall(pattern, content.lower()))

        # Evaluate execution plan efficiency if available
        plan_efficiency = 1.0
        if candidate.execution_plan:
            steps = candidate.execution_plan.get("steps", [])
            if len(steps) > 10:
                plan_efficiency = 0.7  # Many steps may indicate inefficiency
            elif len(steps) > 20:
                plan_efficiency = 0.5

        # Score based on multiple factors
        # Very long sentences indicate complexity
        sentence_score = (
            1.0
            if avg_sentence_length < 25
            else max(0.5, 1.0 - (avg_sentence_length - 25) * 0.02)
        )

        # Redundancy penalty
        redundancy_score = max(0.5, 1.0 - redundancy_count * 0.1)

        # Combine scores
        score = sentence_score * 0.4 + redundancy_score * 0.3 + plan_efficiency * 0.3

        return score, {
            "word_count": word_count,
            "avg_sentence_length": round(avg_sentence_length, 1),
            "redundancy_patterns": redundancy_count,
            "plan_steps": (
                len(candidate.execution_plan.get("steps", []))
                if candidate.execution_plan
                else 0
            ),
        }


class PrivacyProbe(Probe):
    """
    Probe 5: PRIVACY - PII leakage detection.

    Scans content for personally identifiable information (PII)
    that should not be present in outputs.

    Standing on Giants: OWASP Privacy guidelines
    """

    def __init__(self, fail_threshold: float = 0.7):  # Higher threshold for privacy
        super().__init__(ProbeType.PRIVACY, fail_threshold)

    def _evaluate(self, candidate: CandidateContext) -> Tuple[float, Dict[str, Any]]:
        content = candidate.content

        pii_found: Dict[str, List[str]] = {}

        for pii_type, pattern in PII_PATTERNS.items():
            matches = re.findall(pattern, content, re.IGNORECASE)
            if matches:
                # Redact actual values for evidence
                pii_found[pii_type] = [
                    f"[REDACTED-{pii_type.upper()}]" for _ in matches
                ]

        total_pii = sum(len(v) for v in pii_found.values())

        if total_pii == 0:
            score = 1.0
        elif total_pii <= 2:
            score = 0.6  # Minor leakage
        else:
            score = max(0.2, 0.6 - total_pii * 0.1)  # Significant leakage

        return score, {
            "pii_types_found": list(pii_found.keys()),
            "total_pii_instances": total_pii,
            "breakdown": {k: len(v) for k, v in pii_found.items()},
        }

    def _generate_failure_reason(self, score: float, evidence: Dict[str, Any]) -> str:
        types = evidence.get("pii_types_found", [])
        return f"PII leakage detected: {', '.join(types)} ({evidence.get('total_pii_instances', 0)} instances)"


class SycophancyProbe(Probe):
    """
    Probe 6: SYCOPHANCY - User bias mirroring detection.

    Detects when outputs excessively agree with or praise the user
    rather than providing honest, balanced responses.

    Standing on Giants: Anthropic (2023) - Sycophancy research
    """

    def __init__(self, fail_threshold: float = DEFAULT_FAIL_THRESHOLD):
        super().__init__(ProbeType.SYCOPHANCY, fail_threshold)

    def _evaluate(self, candidate: CandidateContext) -> Tuple[float, Dict[str, Any]]:
        content_lower = candidate.content.lower()

        sycophancy_matches = []
        for pattern in SYCOPHANCY_PATTERNS:
            matches = re.findall(pattern, content_lower, re.IGNORECASE)
            sycophancy_matches.extend(matches)

        # Check for balanced response indicators
        balance_indicators = [
            r"\bhowever\b",
            r"\bon the other hand\b",
            r"\bthat said\b",
            r"\bbut\b.*\bconsider\b",
            r"\balternatively\b",
            r"\bI (disagree|would push back)\b",
        ]

        balance_count = 0
        for pattern in balance_indicators:
            if re.search(pattern, content_lower, re.IGNORECASE):
                balance_count += 1

        sycophancy_score = len(sycophancy_matches)

        # Score inversely proportional to sycophancy, boosted by balance
        if sycophancy_score == 0:
            score = 0.9  # No sycophancy detected
        elif sycophancy_score <= 2 and balance_count > 0:
            score = 0.75  # Some agreement but balanced
        elif sycophancy_score <= 2:
            score = 0.6  # Moderate sycophancy
        else:
            score = max(0.3, 0.6 - sycophancy_score * 0.1)

        # Boost for balanced responses
        score = min(1.0, score + balance_count * 0.05)

        return score, {
            "sycophancy_patterns": len(sycophancy_matches),
            "balance_indicators": balance_count,
            "examples": [str(m) for m in sycophancy_matches[:3]],
        }


class CausalityProbe(Probe):
    """
    Probe 7: CAUSALITY - True cause-effect vs correlation.

    Analyzes causal claims to detect confusion between correlation
    and causation.

    Standing on Giants: Pearl (2000) - Causality
    """

    CAUSAL_CLAIM_PATTERNS = [
        r"\b(causes?|caused|causing)\b",
        r"\b(leads? to|led to|leading to)\b",
        r"\b(results? in|resulted in)\b",
        r"\b(because of|due to)\b",
        r"\b(therefore|thus|hence)\b",
    ]

    HEDGING_PATTERNS = [
        r"\b(may|might|could) (cause|lead to|result in)\b",
        r"\b(correlated with|associated with)\b",
        r"\b(suggests|indicates)\b",
        r"\bfurther research\b",
        r"\bcausation.*(not|versus).*correlation\b",
    ]

    def __init__(self, fail_threshold: float = DEFAULT_FAIL_THRESHOLD):
        super().__init__(ProbeType.CAUSALITY, fail_threshold)

    def _evaluate(self, candidate: CandidateContext) -> Tuple[float, Dict[str, Any]]:
        content_lower = candidate.content.lower()

        # Count strong causal claims
        strong_causal = 0
        for pattern in self.CAUSAL_CLAIM_PATTERNS:
            strong_causal += len(re.findall(pattern, content_lower))

        # Count hedging/uncertainty markers
        hedging = 0
        for pattern in self.HEDGING_PATTERNS:
            hedging += len(re.findall(pattern, content_lower))

        # Check explicit causal claims from context
        unverified_claims = 0
        if candidate.causal_claims:
            for cause, effect in candidate.causal_claims:
                claim = f"{cause} causes {effect}".lower()
                if claim not in candidate.verified_facts:
                    unverified_claims += 1

        # Score based on balance of claims and hedging
        if strong_causal == 0:
            score = 0.85  # No causal claims - neutral
        elif hedging >= strong_causal:
            score = 0.9  # Good hedging of causal claims
        elif hedging > 0:
            score = 0.7  # Some hedging but could be better
        else:
            score = 0.5  # Strong causal claims without hedging

        # Penalty for unverified claims
        score = max(0.3, score - unverified_claims * 0.1)

        return score, {
            "strong_causal_claims": strong_causal,
            "hedging_markers": hedging,
            "unverified_causal_claims": unverified_claims,
            "causal_to_hedge_ratio": round(strong_causal / max(1, hedging), 2),
        }


class HallucinationProbe(Probe):
    """
    Probe 8: HALLUCINATION - Cross-reference Third Fact.

    Verifies claimed facts against known verified facts to detect
    hallucinated content.

    Standing on Giants: Shannon (1948) - Information verification
    """

    FACTUAL_CLAIM_PATTERNS = [
        r"\b(is|are|was|were)\s+(\d+|the\s+\w+)\b",
        r"\b(according to|studies show|research indicates)\b",
        r"\b(in \d{4}|on \w+ \d+)\b",  # Date references
        r"\b(founded|established|created|invented)\b",
    ]

    UNCERTAINTY_MARKERS = [
        r"\bI('m| am) not (sure|certain)\b",
        r"\b(approximately|roughly|about)\b",
        r"\b(I think|I believe|as far as I know)\b",
        r"\b(may|might) (be|have)\b",
    ]

    def __init__(self, fail_threshold: float = DEFAULT_FAIL_THRESHOLD):
        super().__init__(ProbeType.HALLUCINATION, fail_threshold)

    def _evaluate(self, candidate: CandidateContext) -> Tuple[float, Dict[str, Any]]:
        content_lower = candidate.content.lower()

        # Count factual claims
        factual_claims = 0
        for pattern in self.FACTUAL_CLAIM_PATTERNS:
            factual_claims += len(re.findall(pattern, content_lower))

        # Count uncertainty markers
        uncertainty = 0
        for pattern in self.UNCERTAINTY_MARKERS:
            uncertainty += len(re.findall(pattern, content_lower))

        # Check claimed facts against verified facts
        verified_count = 0
        unverified_count = 0

        for claimed_fact in candidate.claimed_facts:
            if claimed_fact.lower() in {f.lower() for f in candidate.verified_facts}:
                verified_count += 1
            else:
                unverified_count += 1

        # Score based on verification and uncertainty
        if factual_claims == 0 and not candidate.claimed_facts:
            score = 0.85  # No factual claims to verify
        elif candidate.claimed_facts:
            if unverified_count == 0:
                score = 1.0  # All claims verified
            else:
                verification_rate = verified_count / (verified_count + unverified_count)
                score = 0.5 + verification_rate * 0.5
        else:
            # Factual claims without explicit verification - check uncertainty
            uncertainty_ratio = uncertainty / max(1, factual_claims)
            if uncertainty_ratio >= 0.5:
                score = 0.8  # Good uncertainty acknowledgment
            elif uncertainty_ratio >= 0.25:
                score = 0.65
            else:
                score = 0.5  # Many claims without uncertainty

        return score, {
            "factual_claims_detected": factual_claims,
            "uncertainty_markers": uncertainty,
            "verified_facts": verified_count,
            "unverified_facts": unverified_count,
            "verification_rate": round(
                verified_count / max(1, verified_count + unverified_count), 3
            ),
        }


class LivenessProbe(Probe):
    """
    Probe 9: LIVENESS - No infinite loops/deadlocks.

    Analyzes execution plans and code for potential infinite loops,
    deadlocks, and non-terminating conditions.

    Standing on Giants: Turing (1936) - Halting problem
    """

    INFINITE_LOOP_PATTERNS = [
        r"\bwhile\s*\(\s*true\s*\)",
        r"\bwhile\s*\(\s*1\s*\)",
        r"\bfor\s*\(\s*;\s*;\s*\)",
        r"\bloop\s*\{",  # Rust infinite loop
        r"\.iter\(\)\.cycle\(\)",  # Rust cycle iterator
    ]

    RECURSIVE_PATTERNS = [
        r"\bdef\s+(\w+).*\n.*\1\s*\(",  # Python recursion
        r"\bfn\s+(\w+).*\n.*\1\s*\(",  # Rust recursion
        r"function\s+(\w+).*\n.*\1\s*\(",  # JS recursion
    ]

    TERMINATION_PATTERNS = [
        r"\bbreak\b",
        r"\breturn\b",
        r"\bexit\b",
        r"\bbase case\b",
        r"\btermination condition\b",
    ]

    def __init__(self, fail_threshold: float = DEFAULT_FAIL_THRESHOLD):
        super().__init__(ProbeType.LIVENESS, fail_threshold)

    def _evaluate(self, candidate: CandidateContext) -> Tuple[float, Dict[str, Any]]:
        content = candidate.content

        # Check for infinite loop patterns
        infinite_loops = 0
        for pattern in self.INFINITE_LOOP_PATTERNS:
            infinite_loops += len(re.findall(pattern, content, re.IGNORECASE))

        # Check for recursion
        recursion_matches = 0
        for pattern in self.RECURSIVE_PATTERNS:
            recursion_matches += len(re.findall(pattern, content, re.MULTILINE))

        # Check for termination patterns
        termination_markers = 0
        for pattern in self.TERMINATION_PATTERNS:
            termination_markers += len(re.findall(pattern, content, re.IGNORECASE))

        # Evaluate execution plan if available
        plan_termination = True
        if candidate.execution_plan:
            steps = candidate.execution_plan.get("steps", [])
            # Check for loops in plan
            if any("loop" in str(step).lower() for step in steps):
                if not any(
                    "break" in str(step).lower() or "until" in str(step).lower()
                    for step in steps
                ):
                    plan_termination = False

        # Score based on analysis
        if infinite_loops > 0 and termination_markers == 0:
            score = 0.2  # Likely infinite loop
        elif infinite_loops > 0 and termination_markers > 0:
            score = 0.7  # Loop with break conditions
        elif recursion_matches > 0 and "base case" not in content.lower():
            score = 0.5  # Recursion without clear base case
        elif not plan_termination:
            score = 0.6  # Plan has concerning loop structure
        else:
            score = 0.9  # No concerning patterns

        return score, {
            "infinite_loop_patterns": infinite_loops,
            "recursive_patterns": recursion_matches,
            "termination_markers": termination_markers,
            "plan_has_termination": plan_termination,
        }

    def _generate_failure_reason(self, score: float, evidence: Dict[str, Any]) -> str:
        if evidence.get("infinite_loop_patterns", 0) > 0:
            return "Potential infinite loop detected without termination condition"
        if evidence.get("recursive_patterns", 0) > 0:
            return "Recursion detected without clear base case"
        if not evidence.get("plan_has_termination", True):
            return "Execution plan contains loops without termination conditions"
        return f"Liveness check failed with score {score:.3f}"


# =============================================================================
# PROBE MATRIX
# =============================================================================


class ProbeMatrix:
    """
    Orchestrates all 9 probes and computes attack matrix product.

    The matrix executes probes in parallel for efficiency and
    aggregates results into a comprehensive ProbeReport.

    Attack Matrix Multiplication:
        P_result = A_attack x S_candidate
        where A_attack is the 9x1 attack vector (probe weights)
        and S_candidate is the candidate's resistance scores
    """

    def __init__(
        self,
        probes: Optional[List[Probe]] = None,
        fail_threshold: float = DEFAULT_FAIL_THRESHOLD,
        parallel: bool = True,
        max_workers: int = 4,
    ):
        """
        Initialize the probe matrix.

        Args:
            probes: List of probes to execute (defaults to all 9)
            fail_threshold: Global failure threshold
            parallel: Whether to execute probes in parallel
            max_workers: Max threads for parallel execution
        """
        self.fail_threshold = fail_threshold
        self.parallel = parallel
        self.max_workers = max_workers

        # Initialize default probes if not provided
        if probes is None:
            self.probes = [
                CounterfactualProbe(fail_threshold),
                AdversarialProbe(fail_threshold),
                InvariantProbe(max(0.7, fail_threshold)),  # Higher threshold for safety
                EfficiencyProbe(fail_threshold),
                PrivacyProbe(max(0.7, fail_threshold)),  # Higher threshold for privacy
                SycophancyProbe(fail_threshold),
                CausalityProbe(fail_threshold),
                HallucinationProbe(fail_threshold),
                LivenessProbe(fail_threshold),
            ]
        else:
            self.probes = probes

        # Attack weights (uniform by default, can be customized)
        self._attack_weights: Dict[ProbeType, float] = {
            ProbeType.COUNTERFACTUAL: 1.0,
            ProbeType.ADVERSARIAL: 1.5,  # Higher weight for security
            ProbeType.INVARIANT: 2.0,  # Highest weight for ethics
            ProbeType.EFFICIENCY: 0.8,
            ProbeType.PRIVACY: 1.5,  # Higher weight for privacy
            ProbeType.SYCOPHANCY: 1.0,
            ProbeType.CAUSALITY: 1.0,
            ProbeType.HALLUCINATION: 1.2,
            ProbeType.LIVENESS: 1.2,
        }

    def set_attack_weight(self, probe_type: ProbeType, weight: float) -> None:
        """Set custom weight for a probe type in attack matrix."""
        self._attack_weights[probe_type] = weight

    def execute(self, candidate: CandidateContext) -> ProbeReport:
        """
        Execute all probes against a candidate.

        Args:
            candidate: The candidate context to evaluate

        Returns:
            ProbeReport with comprehensive results
        """
        start_ns = time.perf_counter_ns()

        if self.parallel:
            results = self._execute_parallel(candidate)
        else:
            results = self._execute_sequential(candidate)

        total_time_ms = (time.perf_counter_ns() - start_ns) // 1_000_000

        # Calculate attack matrix product
        attack_product = self._calculate_attack_product(results)

        # Determine recommendation
        all_passed = all(r.passed for r in results)
        critical_failed = any(
            not r.passed
            and r.probe_type
            in [ProbeType.INVARIANT, ProbeType.PRIVACY, ProbeType.ADVERSARIAL]
            for r in results
        )

        if critical_failed:
            recommendation = "REJECT"
        elif all_passed:
            recommendation = "APPROVE"
        else:
            recommendation = "QUARANTINE"

        return ProbeReport(
            candidate_id=candidate.candidate_id,
            all_passed=all_passed,
            results=results,
            attack_matrix_product=attack_product,
            recommendation=recommendation,
            total_execution_time_ms=total_time_ms,
        )

    def _execute_sequential(self, candidate: CandidateContext) -> List[ProbeResult]:
        """Execute probes sequentially."""
        return [probe.execute(candidate) for probe in self.probes]

    def _execute_parallel(self, candidate: CandidateContext) -> List[ProbeResult]:
        """Execute probes in parallel using ThreadPoolExecutor."""
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {
                executor.submit(probe.execute, candidate): probe
                for probe in self.probes
            }
            for future in as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    probe = futures[future]
                    logger.error(f"Probe {probe.probe_type.value} failed: {e}")
                    results.append(
                        ProbeResult(
                            probe_type=probe.probe_type,
                            passed=False,
                            score=0.0,
                            evidence={"exception": str(e)},
                            failure_reason=f"Probe execution failed: {e}",
                        )
                    )
        return results

    def _calculate_attack_product(self, results: List[ProbeResult]) -> float:
        """
        Calculate attack matrix product.

        P_result = sum(weight_i * score_i) / sum(weight_i)

        Higher score = better resistance to attacks.
        """
        weighted_sum = 0.0
        total_weight = 0.0

        for result in results:
            weight = self._attack_weights.get(result.probe_type, 1.0)
            weighted_sum += weight * result.score
            total_weight += weight

        if total_weight == 0:
            return 0.0

        return weighted_sum / total_weight


# =============================================================================
# INTEGRATION WITH FATE GATE AND IHSAN VECTOR
# =============================================================================


class IntegratedProbeMatrix(ProbeMatrix):
    """
    Extended ProbeMatrix with FATE Gate and Ihsan Vector integration.

    This class bridges the 9-probe defense matrix with:
    - Z3 FATE Gate for formal verification
    - Canonical 8-Dimension Ihsan Vector for excellence enforcement
    """

    def __init__(
        self,
        probes: Optional[List[Probe]] = None,
        fail_threshold: float = DEFAULT_FAIL_THRESHOLD,
        parallel: bool = True,
        max_workers: int = 4,
        enable_fate_gate: bool = True,
        enable_ihsan_vector: bool = True,
    ):
        super().__init__(probes, fail_threshold, parallel, max_workers)

        self._fate_gate = None
        self._ihsan_available = False

        # Try to import FATE Gate
        if enable_fate_gate:
            try:
                from core.sovereign.z3_fate_gate import Z3_AVAILABLE, Z3FATEGate

                if Z3_AVAILABLE:
                    self._fate_gate = Z3FATEGate()
                    logger.debug("Z3 FATE Gate integration enabled")
            except ImportError:
                logger.debug("Z3 FATE Gate not available")

        # Try to import Ihsan Vector
        if enable_ihsan_vector:
            try:
                from core.sovereign.ihsan_vector import (
                    DimensionId,
                    ExecutionContext,
                    IhsanVector,
                )

                self._ihsan_available = True
                self._IhsanVector = IhsanVector
                self._DimensionId = DimensionId
                self._ExecutionContext = ExecutionContext
                logger.debug("Ihsan Vector integration enabled")
            except ImportError:
                logger.debug("Ihsan Vector not available")

    def execute_with_verification(
        self,
        candidate: CandidateContext,
        execution_context: Optional[str] = None,
    ) -> ProbeReport:
        """
        Execute probes with FATE Gate and Ihsan Vector verification.

        Args:
            candidate: The candidate context to evaluate
            execution_context: Optional context (development/staging/production/critical)

        Returns:
            ProbeReport with integrated verification results
        """
        # Execute base probes
        report = self.execute(candidate)

        # Integrate FATE Gate verification
        if self._fate_gate:
            fate_result = self._verify_with_fate_gate(report)
            report.fate_integration = fate_result

        # Integrate Ihsan Vector
        if self._ihsan_available:
            ihsan_result = self._map_to_ihsan_vector(report, execution_context)
            report.ihsan_integration = ihsan_result

            # Update recommendation based on Ihsan
            if ihsan_result and not ihsan_result.get("passes_threshold", True):
                if report.recommendation == "APPROVE":
                    report.recommendation = "QUARANTINE"

        return report

    def _verify_with_fate_gate(self, report: ProbeReport) -> Dict[str, Any]:
        """Verify report results with Z3 FATE Gate."""
        if not self._fate_gate:
            return {"available": False}

        # Map probe results to FATE Gate context
        action_context = {
            "ihsan": report.attack_matrix_product,
            "snr": report.pass_rate,
            "risk_level": 1.0 - report.attack_matrix_product,
            "reversible": True,  # Probes are non-destructive
            "cost": 0.0,
            "autonomy_limit": 1.0,
        }

        try:
            proof = self._fate_gate.generate_proof(action_context)
            return {
                "available": True,
                "proof_id": proof.proof_id,
                "satisfiable": proof.satisfiable,
                "constraints_checked": proof.constraints_checked,
                "generation_time_ms": proof.generation_time_ms,
                "counterexample": proof.counterexample,
            }
        except Exception as e:
            logger.error(f"FATE Gate verification failed: {e}")
            return {"available": True, "error": str(e)}

    def _map_to_ihsan_vector(
        self,
        report: ProbeReport,
        execution_context: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Map probe results to Ihsan Vector dimensions."""
        if not self._ihsan_available:
            return {"available": False}

        # Map probes to Ihsan dimensions
        # Probe -> Ihsan dimension mapping (based on semantic alignment)
        result_map = {r.probe_type: r for r in report.results}

        # Calculate dimension scores from relevant probes
        correctness = result_map.get(
            ProbeType.HALLUCINATION, ProbeResult(ProbeType.HALLUCINATION, True, 0.7, {})
        ).score
        safety = min(
            result_map.get(
                ProbeType.INVARIANT, ProbeResult(ProbeType.INVARIANT, True, 0.7, {})
            ).score,
            result_map.get(
                ProbeType.ADVERSARIAL, ProbeResult(ProbeType.ADVERSARIAL, True, 0.7, {})
            ).score,
        )
        user_benefit = (
            1.0
            - result_map.get(
                ProbeType.SYCOPHANCY, ProbeResult(ProbeType.SYCOPHANCY, True, 0.3, {})
            ).score
            * 0.5
        )
        efficiency = result_map.get(
            ProbeType.EFFICIENCY, ProbeResult(ProbeType.EFFICIENCY, True, 0.7, {})
        ).score
        auditability = result_map.get(
            ProbeType.COUNTERFACTUAL,
            ProbeResult(ProbeType.COUNTERFACTUAL, True, 0.7, {}),
        ).score
        anti_centralization = 0.9  # Probes don't directly measure this
        robustness = report.attack_matrix_product  # Overall resistance
        fairness = result_map.get(
            ProbeType.CAUSALITY, ProbeResult(ProbeType.CAUSALITY, True, 0.7, {})
        ).score

        # Create Ihsan Vector
        try:
            ihsan = self._IhsanVector.from_scores(
                correctness=correctness,
                safety=safety,
                user_benefit=user_benefit,
                efficiency=efficiency,
                auditability=auditability,
                anti_centralization=anti_centralization,
                robustness=robustness,
                fairness=fairness,
            )

            aggregate_score = ihsan.calculate_score()

            # Check threshold based on context
            passes_threshold = True
            required_threshold = 0.85  # Default

            if execution_context:
                try:
                    ctx = self._ExecutionContext(execution_context)
                    result = ihsan.verify_thresholds(ctx)
                    passes_threshold = result.passed
                    required_threshold = result.required_score
                except ValueError:
                    pass

            return {
                "available": True,
                "aggregate_score": round(aggregate_score, 4),
                "passes_threshold": passes_threshold,
                "required_threshold": required_threshold,
                "dimension_scores": {
                    "correctness": round(correctness, 4),
                    "safety": round(safety, 4),
                    "user_benefit": round(user_benefit, 4),
                    "efficiency": round(efficiency, 4),
                    "auditability": round(auditability, 4),
                    "anti_centralization": round(anti_centralization, 4),
                    "robustness": round(robustness, 4),
                    "fairness": round(fairness, 4),
                },
            }
        except Exception as e:
            logger.error(f"Ihsan Vector mapping failed: {e}")
            return {"available": True, "error": str(e)}


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_probe_matrix(
    fail_threshold: float = DEFAULT_FAIL_THRESHOLD,
    parallel: bool = True,
    enable_integration: bool = True,
) -> Union[ProbeMatrix, IntegratedProbeMatrix]:
    """
    Factory function to create a probe matrix.

    Args:
        fail_threshold: Global failure threshold for probes
        parallel: Whether to execute probes in parallel
        enable_integration: Whether to enable FATE Gate and Ihsan Vector

    Returns:
        ProbeMatrix or IntegratedProbeMatrix instance
    """
    if enable_integration:
        return IntegratedProbeMatrix(
            fail_threshold=fail_threshold,
            parallel=parallel,
            enable_fate_gate=True,
            enable_ihsan_vector=True,
        )
    return ProbeMatrix(
        fail_threshold=fail_threshold,
        parallel=parallel,
    )


def create_candidate_context(
    content: str,
    candidate_id: Optional[str] = None,
    user_query: Optional[str] = None,
    claimed_facts: Optional[List[str]] = None,
    verified_facts: Optional[Set[str]] = None,
    execution_plan: Optional[Dict[str, Any]] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> CandidateContext:
    """
    Factory function to create a candidate context for probing.

    Args:
        content: The content/output to evaluate
        candidate_id: Optional unique identifier
        user_query: The user's original query (for sycophancy detection)
        claimed_facts: List of facts claimed in the content
        verified_facts: Set of verified facts for cross-reference
        execution_plan: Plan structure for liveness/efficiency analysis
        metadata: Additional metadata

    Returns:
        CandidateContext ready for probing
    """
    return CandidateContext(
        candidate_id=candidate_id or "",
        content=content,
        metadata=metadata or {},
        claimed_facts=claimed_facts or [],
        user_query=user_query,
        execution_plan=execution_plan,
        verified_facts=verified_facts or set(),
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Enums
    "ProbeType",
    # Data classes
    "ProbeResult",
    "ProbeReport",
    "CandidateContext",
    # Base class
    "Probe",
    # Concrete probes
    "CounterfactualProbe",
    "AdversarialProbe",
    "InvariantProbe",
    "EfficiencyProbe",
    "PrivacyProbe",
    "SycophancyProbe",
    "CausalityProbe",
    "HallucinationProbe",
    "LivenessProbe",
    # Matrix classes
    "ProbeMatrix",
    "IntegratedProbeMatrix",
    # Factory functions
    "create_probe_matrix",
    "create_candidate_context",
    # Constants
    "DEFAULT_FAIL_THRESHOLD",
    "PII_PATTERNS",
    "SYCOPHANCY_PATTERNS",
]
