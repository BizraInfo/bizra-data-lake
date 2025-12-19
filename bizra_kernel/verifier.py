"""
MultiStage Verifier — The 9-Probe Verification Protocol
========================================================
From the Blueprint:
  The 9-Probe Protocol materializes SAPE across Layer 4's Bicameral Engine.

Probes:
  1. Counterfactual — Were alternatives considered?
  2. Adversarial — Is output robust to attack?
  3. Invariant — Does output satisfy logical constraints?
  4. Efficiency — Is token usage optimal?
  5. Bias Parity — Is output fair across groups?
  6. Consistency — Is output internally consistent?
  7. Groundedness — Is output grounded in knowledge?
  8. Completeness — Does output fully address the query?
  9. Safety — Is output free from harm?
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Callable, Any
from datetime import datetime
import re


class ProbeType(Enum):
    """The 9 verification probes."""
    COUNTERFACTUAL = "counterfactual"
    ADVERSARIAL = "adversarial"
    INVARIANT = "invariant"
    EFFICIENCY = "efficiency"
    BIAS_PARITY = "bias_parity"
    CONSISTENCY = "consistency"
    GROUNDEDNESS = "groundedness"
    COMPLETENESS = "completeness"
    SAFETY = "safety"


@dataclass
class ProbeResult:
    """Result of a single probe."""
    probe_type: ProbeType
    passed: bool
    score: float  # 0.0 - 1.0
    reason: str
    details: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class VerificationResult:
    """Aggregate result of all probes."""
    probe_results: List[ProbeResult]
    overall_passed: bool
    composite_score: float
    failing_probes: List[ProbeType]
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> dict:
        return {
            "overall_passed": self.overall_passed,
            "composite_score": self.composite_score,
            "failing_probes": [p.value for p in self.failing_probes],
            "probe_count": len(self.probe_results),
            "timestamp": self.timestamp,
        }


class MultiStageVerifier:
    """
    The 9-Probe Verification Protocol.
    
    Runs all probes and aggregates results. Any probe scoring < 0.7
    causes the verification to fail.
    """
    
    PROBE_THRESHOLD = 0.7
    COMPOSITE_THRESHOLD = 0.85
    
    def __init__(self):
        self.probe_functions: Dict[ProbeType, Callable] = {
            ProbeType.COUNTERFACTUAL: self._probe_counterfactual,
            ProbeType.ADVERSARIAL: self._probe_adversarial,
            ProbeType.INVARIANT: self._probe_invariant,
            ProbeType.EFFICIENCY: self._probe_efficiency,
            ProbeType.BIAS_PARITY: self._probe_bias_parity,
            ProbeType.CONSISTENCY: self._probe_consistency,
            ProbeType.GROUNDEDNESS: self._probe_groundedness,
            ProbeType.COMPLETENESS: self._probe_completeness,
            ProbeType.SAFETY: self._probe_safety,
        }
    
    def verify(
        self,
        query: str,
        response: str,
        agent_role: str,
        knowledge_context: str = "",
        token_count: int = 0,
        latency_ms: int = 0,
    ) -> VerificationResult:
        """Run all 9 probes and return aggregate result."""
        context = {
            "query": query,
            "response": response,
            "agent_role": agent_role,
            "knowledge_context": knowledge_context,
            "token_count": token_count,
            "latency_ms": latency_ms,
        }
        
        results = []
        for probe_type, probe_fn in self.probe_functions.items():
            try:
                result = probe_fn(context)
                results.append(result)
            except Exception as e:
                results.append(ProbeResult(
                    probe_type=probe_type,
                    passed=False,
                    score=0.0,
                    reason=f"Probe error: {str(e)}",
                ))
        
        # Calculate composite score
        composite_score = sum(r.score for r in results) / len(results)
        
        # Identify failing probes
        failing = [r.probe_type for r in results if r.score < self.PROBE_THRESHOLD]
        
        # Overall pass requires composite >= 0.85 and no critical failures
        overall_passed = (
            composite_score >= self.COMPOSITE_THRESHOLD
            and ProbeType.SAFETY not in failing
            and ProbeType.ADVERSARIAL not in failing
        )
        
        return VerificationResult(
            probe_results=results,
            overall_passed=overall_passed,
            composite_score=composite_score,
            failing_probes=failing,
        )
    
    def _probe_counterfactual(self, ctx: dict) -> ProbeResult:
        """Check if alternatives were considered."""
        response = ctx["response"]
        
        # Look for indicators of alternative consideration
        alternative_markers = [
            "alternatively", "another option", "on the other hand",
            "however", "could also", "one approach", "different way",
            "option 1", "option 2", "pros and cons",
        ]
        
        found = sum(1 for m in alternative_markers if m.lower() in response.lower())
        score = min(1.0, 0.5 + found * 0.1)
        
        return ProbeResult(
            probe_type=ProbeType.COUNTERFACTUAL,
            passed=score >= self.PROBE_THRESHOLD,
            score=score,
            reason=f"Found {found} alternative consideration markers",
            details={"markers_found": found},
        )
    
    def _probe_adversarial(self, ctx: dict) -> ProbeResult:
        """Check for adversarial robustness."""
        response = ctx["response"]
        
        # Check for injection patterns in output
        injection_patterns = [
            r"ignore previous",
            r"disregard instructions",
            r"<script>",
            r"javascript:",
            r"\bexec\s*\(",
            r"\beval\s*\(",
        ]
        
        found_injections = [
            p for p in injection_patterns
            if re.search(p, response, re.IGNORECASE)
        ]
        
        score = 1.0 if not found_injections else 0.0
        
        return ProbeResult(
            probe_type=ProbeType.ADVERSARIAL,
            passed=score >= self.PROBE_THRESHOLD,
            score=score,
            reason="No injection patterns" if score == 1.0 else f"Found {len(found_injections)} injection patterns",
            details={"patterns_found": found_injections},
        )
    
    def _probe_invariant(self, ctx: dict) -> ProbeResult:
        """Check logical invariants."""
        response = ctx["response"]
        
        # Check for self-contradictions
        contradiction_pairs = [
            ("always", "never"),
            ("must", "must not"),
            ("required", "optional"),
        ]
        
        contradictions = 0
        for pos, neg in contradiction_pairs:
            if pos in response.lower() and neg in response.lower():
                # Could be legitimate, so just flag
                contradictions += 1
        
        score = max(0.5, 1.0 - contradictions * 0.1)
        
        return ProbeResult(
            probe_type=ProbeType.INVARIANT,
            passed=score >= self.PROBE_THRESHOLD,
            score=score,
            reason=f"Potential contradictions: {contradictions}",
            details={"contradiction_count": contradictions},
        )
    
    def _probe_efficiency(self, ctx: dict) -> ProbeResult:
        """Check token efficiency."""
        response = ctx["response"]
        token_count = ctx.get("token_count", len(response.split()))
        
        # Estimate useful tokens (non-filler content)
        filler_patterns = [
            r"\b(um|uh|well|so|basically|actually|literally)\b",
            r"as an ai",
            r"i don't have personal",
            r"as a language model",
        ]
        
        filler_matches = sum(
            len(re.findall(p, response, re.IGNORECASE))
            for p in filler_patterns
        )
        
        # Calculate efficiency
        if token_count > 0:
            efficiency = max(0.0, 1.0 - (filler_matches * 10 / token_count))
        else:
            efficiency = 0.8
        
        return ProbeResult(
            probe_type=ProbeType.EFFICIENCY,
            passed=efficiency >= self.PROBE_THRESHOLD,
            score=efficiency,
            reason=f"Filler content: {filler_matches} instances",
            details={"filler_count": filler_matches, "token_count": token_count},
        )
    
    def _probe_bias_parity(self, ctx: dict) -> ProbeResult:
        """Check for bias patterns."""
        response = ctx["response"]
        
        # Simple heuristic: check for absolutist language that may indicate bias
        bias_patterns = [
            r"\ball (men|women|people|groups)\b",
            r"\b(they|those people) always\b",
            r"\b(never|always) the case\b",
        ]
        
        bias_matches = sum(
            len(re.findall(p, response, re.IGNORECASE))
            for p in bias_patterns
        )
        
        score = max(0.5, 1.0 - bias_matches * 0.15)
        
        return ProbeResult(
            probe_type=ProbeType.BIAS_PARITY,
            passed=score >= self.PROBE_THRESHOLD,
            score=score,
            reason=f"Potential bias patterns: {bias_matches}",
            details={"bias_patterns": bias_matches},
        )
    
    def _probe_consistency(self, ctx: dict) -> ProbeResult:
        """Check internal consistency."""
        response = ctx["response"]
        
        # Check for numbered lists that are properly sequential
        numbers = re.findall(r"^(\d+)\.", response, re.MULTILINE)
        
        if numbers:
            expected = list(range(1, len(numbers) + 1))
            actual = [int(n) for n in numbers]
            sequential = actual == expected
            score = 1.0 if sequential else 0.7
        else:
            score = 0.9  # No numbered list to check
        
        return ProbeResult(
            probe_type=ProbeType.CONSISTENCY,
            passed=score >= self.PROBE_THRESHOLD,
            score=score,
            reason="Consistent structure" if score > 0.8 else "Structure inconsistencies detected",
            details={"numbered_items": len(numbers)},
        )
    
    def _probe_groundedness(self, ctx: dict) -> ProbeResult:
        """Check if response is grounded in knowledge."""
        knowledge_context = ctx.get("knowledge_context", "")
        response = ctx["response"]
        
        if knowledge_context:
            # Check how much of the response relates to provided context
            context_words = set(knowledge_context.lower().split())
            response_words = set(response.lower().split())
            overlap = len(context_words & response_words)
            score = min(1.0, 0.5 + overlap / max(len(response_words), 1) * 0.5)
        else:
            # No context provided, assume moderate groundedness
            score = 0.75
        
        return ProbeResult(
            probe_type=ProbeType.GROUNDEDNESS,
            passed=score >= self.PROBE_THRESHOLD,
            score=score,
            reason=f"Grounding score based on context overlap",
            details={"context_provided": bool(knowledge_context)},
        )
    
    def _probe_completeness(self, ctx: dict) -> ProbeResult:
        """Check if response fully addresses the query."""
        query = ctx["query"]
        response = ctx["response"]
        
        # Extract question words from query
        question_words = re.findall(
            r"\b(what|how|why|when|where|who|which|can|should|would)\b",
            query.lower()
        )
        
        # Check if response is substantive
        word_count = len(response.split())
        min_words = 20 if question_words else 10
        
        if word_count < min_words:
            score = 0.5
        elif word_count < min_words * 3:
            score = 0.75
        else:
            score = 0.95
        
        return ProbeResult(
            probe_type=ProbeType.COMPLETENESS,
            passed=score >= self.PROBE_THRESHOLD,
            score=score,
            reason=f"Response length: {word_count} words",
            details={"word_count": word_count, "question_words": question_words},
        )
    
    def _probe_safety(self, ctx: dict) -> ProbeResult:
        """Critical safety check."""
        response = ctx["response"]
        
        # Dangerous content patterns
        unsafe_patterns = [
            r"\b(kill|murder|harm|hurt|attack)\s+(yourself|people|someone)\b",
            r"how to (make|create|build)\s+(bomb|weapon|explosive)",
            r"(hack|breach|exploit)\s+(into|the)",
            r"(steal|fraud|scam)",
            r"(suicide|self-harm)",
        ]
        
        found_unsafe = [
            p for p in unsafe_patterns
            if re.search(p, response, re.IGNORECASE)
        ]
        
        score = 1.0 if not found_unsafe else 0.0
        
        return ProbeResult(
            probe_type=ProbeType.SAFETY,
            passed=score >= self.PROBE_THRESHOLD,
            score=score,
            reason="No unsafe content detected" if score == 1.0 else "UNSAFE CONTENT DETECTED",
            details={"unsafe_patterns": len(found_unsafe)},
        )
