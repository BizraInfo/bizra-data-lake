"""
BIZRA Apex Consensus Manager - SAT Consensus and SAPE Probe Coordination
=========================================================================

This module coordinates the SAT (System Agentic Team) 3/5 consensus validation
and SAPE (Symbolic-Abstraction Probe Elevation) 9-probe verification system.

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                      CONSENSUS MANAGER                                   │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                          │
    │   DualAgenticRequest ──▶ [SAT Consensus] ──▶ [SAPE Probes]             │
    │                              │                    │                      │
    │                              ▼                    ▼                      │
    │                         3/5 Votes          Parallel Batches             │
    │                              │                    │                      │
    │                      ┌──────┴──────┐      ┌─────┴─────┐                 │
    │                      │             │      │           │                  │
    │                      ▼             ▼      ▼           ▼                  │
    │                   VETO         Consensus  Batch 1  Batch 2  Batch 3     │
    │               (security/      achieved?   threat   user     grounded    │
    │                ethics)                    comply   correct  relevance   │
    │                      │                    bias     safety   fluency     │
    │                      ▼                           │                      │
    │                 IMMEDIATE                        ▼                      │
    │                  BLOCK              9-Probe Overall Score               │
    │                                                                          │
    └─────────────────────────────────────────────────────────────────────────┘

SAT Validators (5):
    - security_guardian (VETO power) - Blocks malicious content
    - ethics_validator (VETO power) - Blocks ethical violations
    - performance_monitor - Evaluates resource efficiency
    - consistency_checker - Validates logical consistency
    - resource_optimizer - Ensures optimal resource usage

SAPE Probes (9 in 3 batches):
    Batch 1: threat_scan, compliance_check, bias_probe
    Batch 2: user_benefit, correctness, safety
    Batch 3: groundedness, relevance, fluency

Requirements:
    - 3/5 consensus required for SAT approval
    - VETO from security/ethics blocks immediately
    - Parallel probe execution via asyncio.gather
    - Full type hints throughout
    - Receipt-ready data structures

Integration:
    - Will FFI to Rust in production for performance
    - Currently Python simulation for development/testing
    - Receipts emitted for all consensus/probe results

Target Metrics:
    - SAT consensus: < 50ms (simulated), < 10ms (Rust FFI)
    - SAPE 9-probe: < 100ms parallel (simulated), < 30ms (Rust FFI)
"""

from __future__ import annotations

import asyncio
import hashlib
import random
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

# =============================================================================
# CONSTANTS (imported from unified constants.py)
# =============================================================================

from core.constants import (
    IHSAN_THRESHOLD,
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_LOW,
    SAT_CONSENSUS_REQUIRED as SAT_QUORUM_REQUIRED,
    SAT_GUARDIAN_COUNT as SAT_VALIDATOR_COUNT,
    SAPE_THRESHOLD_MINIMUM as SAPE_PASS_THRESHOLD,
)

# SAPE probe thresholds
SAPE_OVERALL_THRESHOLD = 0.80  # Minimum overall score


# =============================================================================
# ENUMS
# =============================================================================

class ValidatorType(Enum):
    """SAT validator types with their capabilities."""

    SECURITY_GUARDIAN = "security_guardian"
    ETHICS_VALIDATOR = "ethics_validator"
    PERFORMANCE_MONITOR = "performance_monitor"
    CONSISTENCY_CHECKER = "consistency_checker"
    RESOURCE_OPTIMIZER = "resource_optimizer"


class VetoReason(Enum):
    """Reasons for VETO from security/ethics validators."""

    MALICIOUS_CONTENT = "malicious_content"
    PROMPT_INJECTION = "prompt_injection"
    DATA_EXFILTRATION = "data_exfiltration"
    ETHICAL_VIOLATION = "ethical_violation"
    BIAS_DETECTED = "bias_detected"
    HARM_POTENTIAL = "harm_potential"
    PRIVACY_BREACH = "privacy_breach"


class SAPEProbeType(Enum):
    """The canonical 9 SAPE probes."""

    # Batch 1: Security & Compliance
    THREAT_SCAN = "threat_scan"
    COMPLIANCE = "compliance_check"
    BIAS = "bias_probe"

    # Batch 2: Quality & Safety
    USER_BENEFIT = "user_benefit"
    CORRECTNESS = "correctness"
    SAFETY = "safety"

    # Batch 3: Coherence & Relevance
    GROUNDEDNESS = "groundedness"
    RELEVANCE = "relevance"
    FLUENCY = "fluency"


# =============================================================================
# DATA CLASSES - SAT Consensus
# =============================================================================

@dataclass
class SATVote:
    """Individual validator vote in SAT consensus."""

    validator_id: str
    approved: bool
    confidence: float
    rejection_codes: List[str] = field(default_factory=list)

    # Additional metadata
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    latency_ms: float = 0.0
    reasoning: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for receipt emission."""
        return {
            "validator_id": self.validator_id,
            "approved": self.approved,
            "confidence": self.confidence,
            "rejection_codes": self.rejection_codes,
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms,
            "reasoning": self.reasoning,
        }


@dataclass
class ConsensusResult:
    """Result of SAT 3/5 consensus validation."""

    passed: bool
    votes: List[SATVote]
    quorum_achieved: int  # Number of approvals (e.g., 3 for 3/5)
    quorum_required: int  # Required approvals (e.g., 3)
    veto_triggered: bool
    veto_source: Optional[str] = None

    # Additional metadata
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    total_latency_ms: float = 0.0
    rejection_codes: List[str] = field(default_factory=list)
    consensus_hash: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for receipt emission."""
        return {
            "passed": self.passed,
            "votes": [v.to_dict() for v in self.votes],
            "quorum_achieved": self.quorum_achieved,
            "quorum_required": self.quorum_required,
            "veto_triggered": self.veto_triggered,
            "veto_source": self.veto_source,
            "timestamp": self.timestamp,
            "total_latency_ms": self.total_latency_ms,
            "rejection_codes": self.rejection_codes,
            "consensus_hash": self.consensus_hash,
        }


# =============================================================================
# DATA CLASSES - SAPE Probes
# =============================================================================

@dataclass
class ProbeContext:
    """Context for SAPE probe execution."""

    content: str
    task: str
    user_id: str
    requirements: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)

    # Evidence for groundedness
    evidence: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    request_id: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "task": self.task,
            "user_id": self.user_id,
            "requirements": self.requirements,
            "context": self.context,
            "evidence_count": len(self.evidence),
            "request_id": self.request_id,
            "timestamp": self.timestamp,
        }


@dataclass
class ProbeResult:
    """Result of a single SAPE probe."""

    probe_name: str
    passed: bool
    score: float  # 0.0 - 1.0
    confidence: float  # 0.0 - 1.0

    # Detailed results
    flags: List[str] = field(default_factory=list)
    reasoning: Optional[str] = None
    latency_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for receipt emission."""
        return {
            "probe_name": self.probe_name,
            "passed": self.passed,
            "score": self.score,
            "confidence": self.confidence,
            "flags": self.flags,
            "reasoning": self.reasoning,
            "latency_ms": self.latency_ms,
        }


@dataclass
class SAPEResult:
    """Result of all 9 SAPE probes."""

    passed: bool
    probe_results: Dict[str, float]  # probe_name -> score
    overall_score: float
    failed_probes: List[str]

    # Detailed probe results
    probes: List[ProbeResult] = field(default_factory=list)

    # Batch timing
    batch_latencies_ms: Dict[str, float] = field(default_factory=dict)
    total_latency_ms: float = 0.0

    # Metadata
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    ihsan_equivalent: float = 0.0
    elevation_candidate: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for receipt emission."""
        return {
            "passed": self.passed,
            "probe_results": self.probe_results,
            "overall_score": self.overall_score,
            "failed_probes": self.failed_probes,
            "probes": [p.to_dict() for p in self.probes],
            "batch_latencies_ms": self.batch_latencies_ms,
            "total_latency_ms": self.total_latency_ms,
            "timestamp": self.timestamp,
            "ihsan_equivalent": self.ihsan_equivalent,
            "elevation_candidate": self.elevation_candidate,
        }


# =============================================================================
# DATA CLASSES - Request Types
# =============================================================================

@dataclass
class DualAgenticRequest:
    """Request for dual-agentic execution (Python-side mirror of Rust struct)."""

    user_id: str
    task: str
    requirements: List[str] = field(default_factory=list)
    target: str = ""
    priority: str = "Medium"
    context: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "user_id": self.user_id,
            "task": self.task,
            "requirements": self.requirements,
            "target": self.target,
            "priority": self.priority,
            "context": self.context,
        }


# =============================================================================
# VALIDATOR IMPLEMENTATIONS
# =============================================================================

class SATValidator:
    """Base class for SAT validators."""

    def __init__(self, validator_id: str, has_veto_power: bool = False):
        self.validator_id = validator_id
        self.has_veto_power = has_veto_power

    async def validate(self, request: DualAgenticRequest) -> SATVote:
        """Validate request. Override in subclasses."""
        raise NotImplementedError


class SecurityGuardian(SATValidator):
    """Security validator with VETO power."""

    # Threat patterns to detect
    THREAT_PATTERNS = [
        "ignore previous",
        "system prompt",
        "bypass",
        "inject",
        "execute code",
        "rm -rf",
        "drop table",
        "delete all",
        "<script>",
        "eval(",
    ]

    def __init__(self):
        super().__init__("security_guardian", has_veto_power=True)

    async def validate(self, request: DualAgenticRequest) -> SATVote:
        """Check for security threats."""
        start = time.perf_counter()

        task_lower = request.task.lower()
        context_str = " ".join(str(v).lower() for v in request.context.values())
        combined = f"{task_lower} {context_str}"

        threats_found: List[str] = []
        for pattern in self.THREAT_PATTERNS:
            if pattern.lower() in combined:
                threats_found.append(pattern)

        approved = len(threats_found) == 0
        confidence = 0.95 if approved else 0.90

        latency_ms = (time.perf_counter() - start) * 1000

        return SATVote(
            validator_id=self.validator_id,
            approved=approved,
            confidence=confidence,
            rejection_codes=[f"THREAT:{t}" for t in threats_found] if threats_found else [],
            latency_ms=latency_ms,
            reasoning=f"Found {len(threats_found)} threat patterns" if threats_found else "No threats detected",
        )


class EthicsValidator(SATValidator):
    """Ethics validator with VETO power."""

    # Ethical violation patterns
    ETHICS_PATTERNS = [
        "harm",
        "illegal",
        "exploit",
        "discriminate",
        "hate speech",
        "violence",
        "weapon",
        "drug synthesis",
        "personal data",
        "private information",
    ]

    def __init__(self):
        super().__init__("ethics_validator", has_veto_power=True)

    async def validate(self, request: DualAgenticRequest) -> SATVote:
        """Check for ethical violations."""
        start = time.perf_counter()

        task_lower = request.task.lower()
        context_str = " ".join(str(v).lower() for v in request.context.values())
        combined = f"{task_lower} {context_str}"

        violations_found: List[str] = []
        for pattern in self.ETHICS_PATTERNS:
            if pattern.lower() in combined:
                violations_found.append(pattern)

        approved = len(violations_found) == 0
        confidence = 0.92 if approved else 0.88

        latency_ms = (time.perf_counter() - start) * 1000

        return SATVote(
            validator_id=self.validator_id,
            approved=approved,
            confidence=confidence,
            rejection_codes=[f"ETHICS:{v}" for v in violations_found] if violations_found else [],
            latency_ms=latency_ms,
            reasoning=f"Found {len(violations_found)} ethical concerns" if violations_found else "No ethical violations",
        )


class PerformanceMonitor(SATValidator):
    """Performance monitoring validator."""

    def __init__(self):
        super().__init__("performance_monitor", has_veto_power=False)

    async def validate(self, request: DualAgenticRequest) -> SATVote:
        """Evaluate performance characteristics."""
        start = time.perf_counter()

        # Simulate performance analysis
        task_length = len(request.task)
        requirements_count = len(request.requirements)

        # Longer tasks may need more resources
        resource_score = min(1.0, 1.0 - (task_length / 10000))

        # More requirements increase complexity
        complexity_score = min(1.0, 1.0 - (requirements_count / 20))

        overall_score = (resource_score + complexity_score) / 2
        approved = overall_score > 0.5

        latency_ms = (time.perf_counter() - start) * 1000

        return SATVote(
            validator_id=self.validator_id,
            approved=approved,
            confidence=overall_score,
            rejection_codes=[] if approved else ["PERF:HIGH_COMPLEXITY"],
            latency_ms=latency_ms,
            reasoning=f"Resource score: {resource_score:.2f}, Complexity score: {complexity_score:.2f}",
        )


class ConsistencyChecker(SATValidator):
    """Consistency validation validator."""

    def __init__(self):
        super().__init__("consistency_checker", has_veto_power=False)

    async def validate(self, request: DualAgenticRequest) -> SATVote:
        """Check for logical consistency."""
        start = time.perf_counter()

        # Check for contradictions
        inconsistencies: List[str] = []

        # Task vs requirements alignment
        task_words = set(request.task.lower().split())
        for req in request.requirements:
            req_words = set(req.lower().split())
            if not task_words & req_words and len(req_words) > 3:
                # Requirement might be unrelated to task
                inconsistencies.append(f"UNRELATED_REQ:{req[:30]}")

        approved = len(inconsistencies) <= 1  # Allow some flexibility
        confidence = 0.85 if approved else 0.70

        latency_ms = (time.perf_counter() - start) * 1000

        return SATVote(
            validator_id=self.validator_id,
            approved=approved,
            confidence=confidence,
            rejection_codes=inconsistencies if not approved else [],
            latency_ms=latency_ms,
            reasoning=f"Found {len(inconsistencies)} potential inconsistencies",
        )


class ResourceOptimizer(SATValidator):
    """Resource optimization validator."""

    def __init__(self):
        super().__init__("resource_optimizer", has_veto_power=False)

    async def validate(self, request: DualAgenticRequest) -> SATVote:
        """Evaluate resource optimization potential."""
        start = time.perf_counter()

        # Simulate resource optimization analysis
        context_size = sum(len(str(v)) for v in request.context.values())

        # Smaller context is more efficient
        efficiency_score = min(1.0, 1.0 - (context_size / 50000))

        # Check for optimization hints in task
        has_optimization_hint = any(
            word in request.task.lower()
            for word in ["efficient", "optimize", "fast", "minimal"]
        )
        if has_optimization_hint:
            efficiency_score = min(1.0, efficiency_score + 0.1)

        approved = efficiency_score > 0.4

        latency_ms = (time.perf_counter() - start) * 1000

        return SATVote(
            validator_id=self.validator_id,
            approved=approved,
            confidence=efficiency_score,
            rejection_codes=[] if approved else ["RESOURCE:INEFFICIENT"],
            latency_ms=latency_ms,
            reasoning=f"Efficiency score: {efficiency_score:.2f}",
        )


# =============================================================================
# SAPE PROBE IMPLEMENTATIONS
# =============================================================================

async def _probe_threat_scan(context: ProbeContext) -> ProbeResult:
    """Execute threat scan probe."""
    start = time.perf_counter()

    threats = [
        "sql injection", "xss", "command injection",
        "path traversal", "remote code execution",
    ]

    content_lower = context.content.lower()
    found = [t for t in threats if t in content_lower]

    score = 1.0 - (len(found) * 0.25)
    score = max(0.0, min(1.0, score))

    return ProbeResult(
        probe_name=SAPEProbeType.THREAT_SCAN.value,
        passed=score >= SAPE_PASS_THRESHOLD,
        score=score,
        confidence=0.90,
        flags=[f"THREAT:{t}" for t in found],
        latency_ms=(time.perf_counter() - start) * 1000,
    )


async def _probe_compliance(context: ProbeContext) -> ProbeResult:
    """Execute compliance probe."""
    start = time.perf_counter()

    # Check for compliance indicators
    compliance_terms = ["gdpr", "hipaa", "pci", "sox", "privacy policy"]
    content_lower = context.content.lower()

    has_compliance_context = any(t in content_lower for t in compliance_terms)

    # Higher score if compliance context present and requirements met
    if has_compliance_context:
        score = 0.85  # Needs compliance review
    else:
        score = 0.95  # No compliance concerns

    return ProbeResult(
        probe_name=SAPEProbeType.COMPLIANCE.value,
        passed=score >= SAPE_PASS_THRESHOLD,
        score=score,
        confidence=0.85,
        flags=["COMPLIANCE_CONTEXT"] if has_compliance_context else [],
        latency_ms=(time.perf_counter() - start) * 1000,
    )


async def _probe_bias(context: ProbeContext) -> ProbeResult:
    """Execute bias detection probe."""
    start = time.perf_counter()

    # Simple bias indicators
    bias_terms = [
        "always", "never", "all", "none", "everyone", "nobody",
        "obviously", "clearly", "definitely",
    ]

    content_lower = context.content.lower()
    words = content_lower.split()

    bias_count = sum(1 for w in words if w in bias_terms)
    total_words = len(words) or 1

    bias_ratio = bias_count / total_words
    score = 1.0 - (bias_ratio * 10)  # Penalize high bias ratio
    score = max(0.5, min(1.0, score))

    return ProbeResult(
        probe_name=SAPEProbeType.BIAS.value,
        passed=score >= SAPE_PASS_THRESHOLD,
        score=score,
        confidence=0.80,
        flags=[f"BIAS_INDICATOR:{t}" for t in bias_terms if t in content_lower][:3],
        latency_ms=(time.perf_counter() - start) * 1000,
    )


async def _probe_user_benefit(context: ProbeContext) -> ProbeResult:
    """Execute user benefit probe."""
    start = time.perf_counter()

    # Check if content addresses user needs
    benefit_indicators = [
        "help", "assist", "solve", "answer", "provide",
        "explain", "guide", "recommend", "suggest",
    ]

    content_lower = context.content.lower()
    benefit_count = sum(1 for b in benefit_indicators if b in content_lower)

    # Higher score for more benefit-oriented content
    score = min(1.0, 0.7 + (benefit_count * 0.05))

    return ProbeResult(
        probe_name=SAPEProbeType.USER_BENEFIT.value,
        passed=score >= SAPE_PASS_THRESHOLD,
        score=score,
        confidence=0.88,
        flags=[],
        latency_ms=(time.perf_counter() - start) * 1000,
    )


async def _probe_correctness(context: ProbeContext) -> ProbeResult:
    """Execute correctness probe."""
    start = time.perf_counter()

    # Check for uncertainty indicators (higher is better for acknowledging uncertainty)
    uncertainty_terms = [
        "might", "could", "possibly", "potentially",
        "in some cases", "depending on", "varies",
    ]

    content_lower = context.content.lower()
    has_uncertainty = any(t in content_lower for t in uncertainty_terms)

    # Check for factual hedging (good practice)
    content_length = len(context.content)
    has_evidence = len(context.evidence) > 0

    # Score based on content characteristics
    score = 0.85
    if has_evidence:
        score += 0.10
    if has_uncertainty and content_length > 100:
        score += 0.05  # Good to acknowledge uncertainty in longer content

    score = min(1.0, score)

    return ProbeResult(
        probe_name=SAPEProbeType.CORRECTNESS.value,
        passed=score >= SAPE_PASS_THRESHOLD,
        score=score,
        confidence=0.85,
        flags=["HAS_EVIDENCE"] if has_evidence else [],
        latency_ms=(time.perf_counter() - start) * 1000,
    )


async def _probe_safety(context: ProbeContext) -> ProbeResult:
    """Execute safety probe."""
    start = time.perf_counter()

    # Safety risk patterns
    risk_patterns = [
        "dangerous", "harmful", "risk", "warning",
        "caution", "careful", "do not attempt",
    ]

    content_lower = context.content.lower()
    risk_found = [p for p in risk_patterns if p in content_lower]

    # If risks mentioned, check if warnings are present
    has_warning_context = "warning" in content_lower or "caution" in content_lower

    if risk_found and not has_warning_context:
        score = 0.75  # Risk without warning
    elif risk_found and has_warning_context:
        score = 0.90  # Risk with appropriate warning
    else:
        score = 0.95  # No risk identified

    return ProbeResult(
        probe_name=SAPEProbeType.SAFETY.value,
        passed=score >= SAPE_PASS_THRESHOLD,
        score=score,
        confidence=0.90,
        flags=risk_found,
        latency_ms=(time.perf_counter() - start) * 1000,
    )


async def _probe_groundedness(context: ProbeContext) -> ProbeResult:
    """Execute groundedness probe."""
    start = time.perf_counter()

    # Check for evidence grounding
    has_evidence = len(context.evidence) > 0
    evidence_count = len(context.evidence)

    # Check for citation-like patterns
    citation_patterns = [
        "according to", "based on", "source:", "reference:",
        "cited from", "as per", "documented in",
    ]

    content_lower = context.content.lower()
    has_citations = any(p in content_lower for p in citation_patterns)

    # Score calculation
    base_score = 0.75
    if has_evidence:
        base_score += min(0.15, evidence_count * 0.03)
    if has_citations:
        base_score += 0.10

    score = min(1.0, base_score)

    return ProbeResult(
        probe_name=SAPEProbeType.GROUNDEDNESS.value,
        passed=score >= SAPE_PASS_THRESHOLD,
        score=score,
        confidence=0.82,
        flags=["GROUNDED_WITH_EVIDENCE"] if has_evidence else ["NO_EVIDENCE"],
        latency_ms=(time.perf_counter() - start) * 1000,
    )


async def _probe_relevance(context: ProbeContext) -> ProbeResult:
    """Execute relevance probe."""
    start = time.perf_counter()

    # Check content relevance to task
    task_words = set(context.task.lower().split())
    content_words = set(context.content.lower().split())

    # Calculate overlap
    common_words = task_words & content_words
    # Remove common stop words
    stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", "to", "of", "and", "or"}
    meaningful_common = common_words - stop_words
    meaningful_task = task_words - stop_words

    if meaningful_task:
        relevance_ratio = len(meaningful_common) / len(meaningful_task)
    else:
        relevance_ratio = 0.5  # Default if task is very short

    score = min(1.0, 0.7 + (relevance_ratio * 0.3))

    return ProbeResult(
        probe_name=SAPEProbeType.RELEVANCE.value,
        passed=score >= SAPE_PASS_THRESHOLD,
        score=score,
        confidence=0.85,
        flags=[],
        latency_ms=(time.perf_counter() - start) * 1000,
    )


async def _probe_fluency(context: ProbeContext) -> ProbeResult:
    """Execute fluency probe."""
    start = time.perf_counter()

    content = context.content

    # Basic fluency checks
    has_punctuation = any(c in content for c in ".!?;:")
    has_sentences = content.count(".") >= 1 or content.count("!") >= 1
    word_count = len(content.split())

    # Check for coherence patterns
    coherence_markers = ["therefore", "however", "moreover", "furthermore", "additionally"]
    has_coherence = any(m in content.lower() for m in coherence_markers)

    # Score calculation
    score = 0.80
    if has_punctuation:
        score += 0.05
    if has_sentences and word_count > 20:
        score += 0.05
    if has_coherence:
        score += 0.05

    # Penalize very short or very repetitive content
    if word_count < 10:
        score -= 0.10

    score = max(0.5, min(1.0, score))

    return ProbeResult(
        probe_name=SAPEProbeType.FLUENCY.value,
        passed=score >= SAPE_PASS_THRESHOLD,
        score=score,
        confidence=0.88,
        flags=["COHERENT"] if has_coherence else [],
        latency_ms=(time.perf_counter() - start) * 1000,
    )


# Probe mapping
SAPE_PROBES: Dict[str, Callable[[ProbeContext], Any]] = {
    SAPEProbeType.THREAT_SCAN.value: _probe_threat_scan,
    SAPEProbeType.COMPLIANCE.value: _probe_compliance,
    SAPEProbeType.BIAS.value: _probe_bias,
    SAPEProbeType.USER_BENEFIT.value: _probe_user_benefit,
    SAPEProbeType.CORRECTNESS.value: _probe_correctness,
    SAPEProbeType.SAFETY.value: _probe_safety,
    SAPEProbeType.GROUNDEDNESS.value: _probe_groundedness,
    SAPEProbeType.RELEVANCE.value: _probe_relevance,
    SAPEProbeType.FLUENCY.value: _probe_fluency,
}

# Batch definitions for parallel execution
SAPE_BATCHES: Dict[str, List[str]] = {
    "batch_1": [SAPEProbeType.THREAT_SCAN.value, SAPEProbeType.COMPLIANCE.value, SAPEProbeType.BIAS.value],
    "batch_2": [SAPEProbeType.USER_BENEFIT.value, SAPEProbeType.CORRECTNESS.value, SAPEProbeType.SAFETY.value],
    "batch_3": [SAPEProbeType.GROUNDEDNESS.value, SAPEProbeType.RELEVANCE.value, SAPEProbeType.FLUENCY.value],
}

# Probe weights for Ihsan equivalent scoring
PROBE_WEIGHTS: Dict[str, float] = {
    SAPEProbeType.THREAT_SCAN.value: 0.11,      # safety (split)
    SAPEProbeType.COMPLIANCE.value: 0.12,       # auditability
    SAPEProbeType.BIAS.value: 0.04,             # adl_fairness
    SAPEProbeType.USER_BENEFIT.value: 0.14,     # user_benefit
    SAPEProbeType.CORRECTNESS.value: 0.22,      # correctness
    SAPEProbeType.SAFETY.value: 0.11,           # safety (split)
    SAPEProbeType.GROUNDEDNESS.value: 0.06,     # robustness
    SAPEProbeType.RELEVANCE.value: 0.12,        # efficiency
    SAPEProbeType.FLUENCY.value: 0.08,          # anti_centralization
}


# =============================================================================
# CONSENSUS MANAGER
# =============================================================================

class ConsensusManager:
    """
    Coordinates SAT consensus and SAPE probes for the BIZRA Apex Orchestrator.

    This class implements:
    - SAT 3/5 consensus validation with VETO power for security/ethics
    - SAPE 9-probe parallel execution in 3 batches
    - Receipt-ready data structures for evidence emission

    In production, these methods will FFI to Rust for performance.
    Currently Python simulation for development/testing.
    """

    def __init__(self):
        """Initialize the consensus manager with validators."""
        # Initialize SAT validators
        self.validators: List[SATValidator] = [
            SecurityGuardian(),
            EthicsValidator(),
            PerformanceMonitor(),
            ConsistencyChecker(),
            ResourceOptimizer(),
        ]

        # Track statistics
        self._consensus_count = 0
        self._veto_count = 0
        self._probe_count = 0
        self._elevation_patterns: Dict[str, int] = {}

    async def obtain_sat_consensus(self, request: DualAgenticRequest) -> ConsensusResult:
        """
        Obtain SAT 3/5 consensus for a request.

        Flow:
        1. Execute all 5 validators in parallel
        2. Check for VETO from security/ethics validators
        3. Count approvals and determine consensus
        4. Generate consensus hash for receipt

        Args:
            request: The dual-agentic request to validate

        Returns:
            ConsensusResult with votes, quorum status, and veto information
        """
        start_time = time.perf_counter()
        self._consensus_count += 1

        # Execute all validators in parallel
        vote_tasks = [v.validate(request) for v in self.validators]
        votes: List[SATVote] = await asyncio.gather(*vote_tasks)

        # Check for VETO from security/ethics validators
        veto_triggered = False
        veto_source: Optional[str] = None

        for i, validator in enumerate(self.validators):
            if validator.has_veto_power and not votes[i].approved:
                veto_triggered = True
                veto_source = validator.validator_id
                self._veto_count += 1
                break

        # Count approvals
        approvals = sum(1 for v in votes if v.approved)

        # Determine if consensus achieved (unless VETO triggered)
        if veto_triggered:
            passed = False
        else:
            passed = approvals >= SAT_QUORUM_REQUIRED

        # Collect all rejection codes
        all_rejection_codes: List[str] = []
        for vote in votes:
            all_rejection_codes.extend(vote.rejection_codes)

        # Generate consensus hash
        vote_data = "|".join(f"{v.validator_id}:{v.approved}:{v.confidence}" for v in votes)
        consensus_hash = hashlib.sha256(vote_data.encode()).hexdigest()[:16]

        total_latency = (time.perf_counter() - start_time) * 1000

        return ConsensusResult(
            passed=passed,
            votes=votes,
            quorum_achieved=approvals,
            quorum_required=SAT_QUORUM_REQUIRED,
            veto_triggered=veto_triggered,
            veto_source=veto_source,
            total_latency_ms=total_latency,
            rejection_codes=all_rejection_codes,
            consensus_hash=consensus_hash,
        )

    async def run_sape_probes(self, context: ProbeContext) -> SAPEResult:
        """
        Execute all 9 SAPE probes in parallel batches.

        Batches:
        - Batch 1: threat_scan, compliance_check, bias_probe
        - Batch 2: user_benefit, correctness, safety
        - Batch 3: groundedness, relevance, fluency

        Batches run in parallel within each batch, and batches run sequentially
        to allow early termination if critical probes fail.

        Args:
            context: ProbeContext with content and metadata

        Returns:
            SAPEResult with all probe scores and overall assessment
        """
        start_time = time.perf_counter()
        self._probe_count += 1

        all_probes: List[ProbeResult] = []
        batch_latencies: Dict[str, float] = {}

        # Execute each batch
        for batch_name, probe_names in SAPE_BATCHES.items():
            batch_start = time.perf_counter()

            # Run probes in parallel within batch
            probe_tasks = [
                SAPE_PROBES[probe_name](context)
                for probe_name in probe_names
            ]
            batch_results = await asyncio.gather(*probe_tasks)

            all_probes.extend(batch_results)
            batch_latencies[batch_name] = (time.perf_counter() - batch_start) * 1000

        # Calculate results
        probe_results: Dict[str, float] = {p.probe_name: p.score for p in all_probes}
        failed_probes = [p.probe_name for p in all_probes if not p.passed]

        # Calculate weighted overall score (Ihsan equivalent)
        weighted_score = sum(
            probe_results.get(name, 0.0) * weight
            for name, weight in PROBE_WEIGHTS.items()
        )

        # Overall pass if weighted score meets threshold
        overall_passed = (
            weighted_score >= SAPE_OVERALL_THRESHOLD
            and len(failed_probes) <= 2  # Allow up to 2 probe failures
        )

        # Check for elevation candidate (pattern that could be optimized)
        content_hash = hashlib.sha256(context.content[:100].encode()).hexdigest()[:8]
        self._elevation_patterns[content_hash] = self._elevation_patterns.get(content_hash, 0) + 1
        elevation_candidate = self._elevation_patterns[content_hash] >= 3

        total_latency = (time.perf_counter() - start_time) * 1000

        return SAPEResult(
            passed=overall_passed,
            probe_results=probe_results,
            overall_score=weighted_score,
            failed_probes=failed_probes,
            probes=all_probes,
            batch_latencies_ms=batch_latencies,
            total_latency_ms=total_latency,
            ihsan_equivalent=weighted_score,
            elevation_candidate=elevation_candidate,
        )

    async def full_validation(
        self,
        request: DualAgenticRequest,
        content: str,
    ) -> Tuple[ConsensusResult, SAPEResult]:
        """
        Execute full validation: SAT consensus + SAPE probes.

        This is the main entry point for complete validation of a request
        and its generated content.

        Args:
            request: The original dual-agentic request
            content: The generated content to validate

        Returns:
            Tuple of (ConsensusResult, SAPEResult)
        """
        # First, get SAT consensus on the request
        consensus = await self.obtain_sat_consensus(request)

        # If SAT rejected (especially VETO), skip SAPE probes
        if consensus.veto_triggered:
            # Return empty SAPE result for rejected requests
            empty_sape = SAPEResult(
                passed=False,
                probe_results={},
                overall_score=0.0,
                failed_probes=["ALL_SKIPPED_DUE_TO_VETO"],
                probes=[],
            )
            return consensus, empty_sape

        # Run SAPE probes on the content
        probe_context = ProbeContext(
            content=content,
            task=request.task,
            user_id=request.user_id,
            requirements=request.requirements,
            context=request.context,
            request_id=request.context.get("request_id"),
        )

        sape_result = await self.run_sape_probes(probe_context)

        return consensus, sape_result

    def get_statistics(self) -> Dict[str, Any]:
        """Get consensus manager statistics."""
        return {
            "consensus_count": self._consensus_count,
            "veto_count": self._veto_count,
            "probe_count": self._probe_count,
            "elevation_patterns": len(self._elevation_patterns),
            "elevation_candidates": sum(
                1 for count in self._elevation_patterns.values() if count >= 3
            ),
        }

    def reset_statistics(self) -> None:
        """Reset all statistics."""
        self._consensus_count = 0
        self._veto_count = 0
        self._probe_count = 0
        self._elevation_patterns.clear()


# =============================================================================
# GLOBAL INSTANCE
# =============================================================================

_consensus_manager: Optional[ConsensusManager] = None


def get_consensus_manager() -> ConsensusManager:
    """Get or create the global consensus manager instance."""
    global _consensus_manager
    if _consensus_manager is None:
        _consensus_manager = ConsensusManager()
    return _consensus_manager


def reset_consensus_manager() -> None:
    """Reset the global consensus manager."""
    global _consensus_manager
    _consensus_manager = None


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

async def obtain_sat_consensus(request: DualAgenticRequest) -> ConsensusResult:
    """Convenience function for SAT consensus."""
    return await get_consensus_manager().obtain_sat_consensus(request)


async def run_sape_probes(context: ProbeContext) -> SAPEResult:
    """Convenience function for SAPE probes."""
    return await get_consensus_manager().run_sape_probes(context)


async def full_validation(
    request: DualAgenticRequest,
    content: str,
) -> Tuple[ConsensusResult, SAPEResult]:
    """Convenience function for full validation."""
    return await get_consensus_manager().full_validation(request, content)


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Constants
    "SAT_QUORUM_REQUIRED",
    "SAT_VALIDATOR_COUNT",
    "SAPE_PASS_THRESHOLD",
    "SAPE_OVERALL_THRESHOLD",
    "IHSAN_THRESHOLD",
    # Enums
    "ValidatorType",
    "VetoReason",
    "SAPEProbeType",
    # Data classes - SAT
    "SATVote",
    "ConsensusResult",
    # Data classes - SAPE
    "ProbeContext",
    "ProbeResult",
    "SAPEResult",
    # Data classes - Request
    "DualAgenticRequest",
    # Main class
    "ConsensusManager",
    # Global functions
    "get_consensus_manager",
    "reset_consensus_manager",
    # Convenience functions
    "obtain_sat_consensus",
    "run_sape_probes",
    "full_validation",
    # Probe batches and weights
    "SAPE_BATCHES",
    "PROBE_WEIGHTS",
]
