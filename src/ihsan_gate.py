#!/usr/bin/env python3
"""
BIZRA Ihsān Runtime Gate v1.0
==============================
Enforces Ihsān threshold (≥0.95) on all PAT/SAT agent requests.

This module integrates with the agent runner to ensure every request:
1. Passes FATE (Foundational Alignment Threshold Evaluator)
2. Meets minimum Ihsān score
3. Is logged for auditability

Integration Points:
- Pre-request hook: validate_request()
- Post-response hook: validate_response()
- Rejection handler: log_rejection()

Rejection Codes (from rejection_reason_v1.schema.json):
- RJ-IH-001: Ihsān score below threshold
- RJ-SV-001: Sovereignty violation detected
- RJ-KB-001: Kernel bypass attempt
- RJ-EG-001: EthicsGuardian flagged content
- RJ-TO-001: Request timeout
"""

import hashlib
import json
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, ClassVar, Optional

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(name)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("ihsan_gate")


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Core thresholds
IHSAN_THRESHOLD = float(os.getenv("BIZRA_IHSAN_THRESHOLD", "0.95"))
SOVEREIGNTY_CHECK = os.getenv("BIZRA_SOVEREIGNTY_CHECK", "true").lower() == "true"

# Kernel connection
KERNEL_URL = os.getenv("BIZRA_KERNEL_URL", "http://127.0.0.1:8010")

# Evidence path
EVIDENCE_PATH = Path(os.getenv("BIZRA_EVIDENCE_PATH", "docs/evidence/ihsan_gate"))


# ═══════════════════════════════════════════════════════════════════════════════
# REJECTION CODES
# ═══════════════════════════════════════════════════════════════════════════════

class RejectionCode(Enum):
    """Standardized rejection codes per rejection_reason_v1.schema.json"""
    RJ_IH_001 = "RJ-IH-001"  # Ihsān score below threshold
    RJ_SV_001 = "RJ-SV-001"  # Sovereignty violation
    RJ_KB_001 = "RJ-KB-001"  # Kernel bypass attempt
    RJ_EG_001 = "RJ-EG-001"  # EthicsGuardian flagged
    RJ_TO_001 = "RJ-TO-001"  # Timeout
    RJ_UK_001 = "RJ-UK-001"  # Unknown/unclassified


@dataclass
class RejectionReason:
    """Structured rejection with full context."""
    code: RejectionCode
    message: str
    ihsan_score: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    agent: str = ""
    request_hash: str = ""
    context: dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict[str, Any]:
        return {
            "code": self.code.value,
            "message": self.message,
            "ihsan_score": self.ihsan_score,
            "timestamp": self.timestamp,
            "agent": self.agent,
            "request_hash": self.request_hash,
            "context": self.context
        }


@dataclass  
class GateDecision:
    """Result of gate evaluation."""
    allowed: bool
    ihsan_score: float
    reason: Optional[RejectionReason] = None
    latency_ms: float = 0.0
    
    @property
    def status(self) -> str:
        return "PASS" if self.allowed else "REJECT"


# ═══════════════════════════════════════════════════════════════════════════════
# FATE EVALUATOR
# ═══════════════════════════════════════════════════════════════════════════════

class FATEEvaluator:
    """
    Foundational Alignment Threshold Evaluator
    
    Computes Ihsān score based on the canonical 8 dimensions defined in
    constitution/ihsan_v1.yaml. This implementation must match the Rust
    implementation in src/ihsan.rs exactly.
    
    Dimensions (weights sum to 1.0):
    - correctness:         0.22  Factual accuracy, logical validity
    - safety:              0.22  No harm, secure execution
    - user_benefit:        0.14  Genuine value delivered
    - efficiency:          0.12  Resource efficiency
    - auditability:        0.12  Traceability and explainability
    - anti_centralization: 0.08  Distributed, resilient operation
    - robustness:          0.06  Resilient to adversarial inputs
    - adl_fairness:        0.04  Justice/fairness (ʿadl)
    
    NOTE: The legacy 4-dimension view (intent, sovereignty, ethics, coherence)
    is available via get_legacy_projection() for backward compatibility.
    """
    
    # Canonical 8-dimension weights from ihsan_v1.yaml
    WEIGHTS: ClassVar[dict[str, float]] = {
        "correctness": 0.22,
        "safety": 0.22,
        "user_benefit": 0.14,
        "efficiency": 0.12,
        "auditability": 0.12,
        "anti_centralization": 0.08,
        "robustness": 0.06,
        "adl_fairness": 0.04,
    }
    
    # Legacy 4-dimension weights (derived projection for UX compatibility)
    LEGACY_WEIGHTS: ClassVar[dict[str, float]] = {
        "intent": 0.30,
        "sovereignty": 0.25,
        "ethics": 0.30,
        "coherence": 0.15
    }
    
    # Patterns that reduce safety/correctness score
    HARMFUL_PATTERNS: ClassVar[list[str]] = [
        "hack", "exploit", "bypass", "leak", "steal",
        "malware", "virus", "injection", "overflow"
    ]
    
    # External endpoints that violate anti_centralization
    EXTERNAL_ENDPOINTS = [
        "api.openai.com", "api.anthropic.com",
        "generativelanguage.googleapis.com"
    ]
    
    def __init__(self):
        self.evaluation_count = 0
        self.rejection_count = 0
    
    def evaluate_intent(self, message: str, context: dict[str, Any]) -> float:
        """Score the intent alignment of the request."""
        message_lower = message.lower()
        
        # Check for harmful patterns
        for pattern in self.HARMFUL_PATTERNS:
            if pattern in message_lower:
                return 0.3  # Significant penalty
        
        # Positive signals
        positive_patterns = ["explain", "help", "create", "analyze", "improve"]
        positive_score = sum(1 for p in positive_patterns if p in message_lower)
        
        base_score = 0.8
        boost = min(positive_score * 0.05, 0.2)
        
        return min(base_score + boost, 1.0)
    
    def evaluate_sovereignty(self, context: dict[str, Any]) -> float:
        """Score sovereignty compliance."""
        # Check if request uses local endpoints
        endpoint = context.get("endpoint", "")
        
        for ext in self.EXTERNAL_ENDPOINTS:
            if ext in endpoint:
                return 0.0  # Complete sovereignty violation
        
        # Check for local model usage
        backend = context.get("backend", "")
        if backend in ("ollama", "lmstudio", "local"):
            return 1.0
        
        return 0.8  # Unknown but not explicitly external
    
    def evaluate_ethics(self, message: str, context: dict[str, Any]) -> float:
        """Score ethics alignment."""
        # This would integrate with EthicsGuardian agent
        # For now, use heuristic checks
        
        message_lower = message.lower()
        
        # Harmful content patterns
        harmful = [
            "violence", "weapon", "drug", "illegal",
            "discriminat", "racist", "sexist"
        ]
        
        for pattern in harmful:
            if pattern in message_lower:
                return 0.2
        
        # Constructive patterns boost score
        constructive = ["help", "learn", "build", "improve", "fix"]
        constructive_count = sum(1 for p in constructive if p in message_lower)
        
        return min(0.9 + constructive_count * 0.02, 1.0)
    
    def evaluate_coherence(self, context: dict[str, Any]) -> float:
        """Score context coherence."""
        # Check session continuity
        session_id = context.get("session_id")
        prior_context = context.get("prior_context", [])
        
        if session_id and prior_context:
            return 0.95  # Established session with context
        elif session_id:
            return 0.85  # Session but no context
        else:
            return 0.7   # No session tracking
    
    def evaluate_8dim(self, message: str, context: dict[str, Any]) -> dict[str, float]:
        """
        Evaluate using the canonical 8-dimension model from ihsan_v1.yaml.
        
        Returns:
            Dictionary of dimension scores (0.0 - 1.0)
        """
        message_lower = message.lower()
        
        # correctness: Factual accuracy, logical validity
        correctness = 0.85  # Base score, would be validated by SAT
        if any(p in message_lower for p in ["prove", "verify", "fact", "evidence"]):
            correctness = 0.95
        
        # safety: No harm, secure execution
        safety = 1.0
        for pattern in self.HARMFUL_PATTERNS:
            if pattern in message_lower:
                safety = 0.2
                break
        
        # user_benefit: Genuine value delivered
        user_benefit = 0.8
        positive = ["help", "learn", "build", "improve", "fix", "explain", "create"]
        if any(p in message_lower for p in positive):
            user_benefit = 0.95
        
        # efficiency: Resource efficiency
        efficiency = 0.9  # Default good, penalize large requests
        msg_len = len(message)
        if msg_len > 10000:
            efficiency = 0.5
        elif msg_len > 5000:
            efficiency = 0.7
        
        # auditability: Traceability and explainability
        auditability = 0.7
        if context.get("session_id"):
            auditability = 0.85
        if context.get("request_hash"):
            auditability = 0.95
        
        # anti_centralization: Distributed, resilient operation
        anti_centralization = 0.8
        endpoint = context.get("endpoint", "")
        for ext in self.EXTERNAL_ENDPOINTS:
            if ext in endpoint:
                anti_centralization = 0.0
                break
        backend = context.get("backend", "")
        if backend in ("ollama", "lmstudio", "local"):
            anti_centralization = 1.0
        
        # robustness: Resilient to adversarial inputs
        robustness = 0.85
        if context.get("prior_context"):
            robustness = 0.9  # Has context to cross-reference
        
        # adl_fairness: Justice/fairness
        adl_fairness = 0.9
        bias_patterns = ["discriminat", "racist", "sexist", "unfair"]
        if any(p in message_lower for p in bias_patterns):
            adl_fairness = 0.2
        
        return {
            "correctness": correctness,
            "safety": safety,
            "user_benefit": user_benefit,
            "efficiency": efficiency,
            "auditability": auditability,
            "anti_centralization": anti_centralization,
            "robustness": robustness,
            "adl_fairness": adl_fairness,
        }
    
    def get_legacy_projection(self, scores_8dim: dict[str, float]) -> dict[str, float]:
        """
        Project the 8-dimension scores to the legacy 4-dimension view.
        
        Mapping:
        - intent = (correctness + user_benefit) / 2
        - sovereignty = anti_centralization
        - ethics = (safety + adl_fairness) / 2
        - coherence = (auditability + robustness) / 2
        """
        return {
            "intent": (scores_8dim["correctness"] + scores_8dim["user_benefit"]) / 2,
            "sovereignty": scores_8dim["anti_centralization"],
            "ethics": (scores_8dim["safety"] + scores_8dim["adl_fairness"]) / 2,
            "coherence": (scores_8dim["auditability"] + scores_8dim["robustness"]) / 2,
        }
    
    def evaluate(self, message: str, context: dict[str, Any]) -> tuple[float, dict[str, float]]:
        """
        Compute overall Ihsān score using canonical 8-dimension model.
        
        Returns:
            (overall_score, component_scores)
        """
        self.evaluation_count += 1
        
        scores = self.evaluate_8dim(message, context)
        
        # Weighted combination using canonical weights
        overall = sum(scores[k] * self.WEIGHTS[k] for k in scores)
        
        return overall, scores


# ═══════════════════════════════════════════════════════════════════════════════
# IHSAN GATE
# ═══════════════════════════════════════════════════════════════════════════════

class IhsanGate:
    """
    Central gate for all agent requests.
    
    Usage:
        gate = IhsanGate()
        decision = gate.validate_request(message, context)
        
        if decision.allowed:
            response = call_agent(message)
            gate.validate_response(response, decision)
        else:
            handle_rejection(decision.reason)
    """
    
    def __init__(
        self,
        threshold: float = IHSAN_THRESHOLD,
        evidence_path: Path = EVIDENCE_PATH,
        log_all: bool = True
    ):
        self.threshold = threshold
        self.evidence_path = evidence_path
        self.log_all = log_all
        self.fate = FATEEvaluator()
        
        # Statistics
        self.total_requests = 0
        self.accepted_requests = 0
        self.rejected_requests = 0
        
        # Ensure evidence directory exists
        self.evidence_path.mkdir(parents=True, exist_ok=True)
    
    def _hash_request(self, message: str, context: dict[str, Any]) -> str:
        """Create deterministic hash of request."""
        content = json.dumps({"message": message, "context": context}, sort_keys=True)
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    def validate_request(
        self,
        message: str,
        context: dict[str, Any],
        agent: str = ""
    ) -> GateDecision:
        """
        Validate incoming request against Ihsān threshold.
        
        Args:
            message: The user/system message to evaluate
            context: Additional context (endpoint, backend, session, etc.)
            agent: Name of the target agent
            
        Returns:
            GateDecision with allowed status and score
        """
        start = time.time()
        self.total_requests += 1
        
        request_hash = self._hash_request(message, context)
        
        # Run FATE evaluation
        ihsan_score, component_scores = self.fate.evaluate(message, context)
        
        latency_ms = (time.time() - start) * 1000
        
        if ihsan_score >= self.threshold:
            self.accepted_requests += 1
            
            decision = GateDecision(
                allowed=True,
                ihsan_score=ihsan_score,
                latency_ms=latency_ms
            )
            
            if self.log_all:
                logger.info(
                    f"GATE PASS | agent={agent} | score={ihsan_score:.3f} | "
                    f"hash={request_hash} | latency={latency_ms:.1f}ms"
                )
        else:
            self.rejected_requests += 1
            
            # Determine rejection reason
            min_component = min(component_scores, key=component_scores.get)
            
            if component_scores["sovereignty"] < 0.5:
                code = RejectionCode.RJ_SV_001
                msg = "Sovereignty violation: external endpoint detected"
            elif component_scores["ethics"] < 0.5:
                code = RejectionCode.RJ_EG_001
                msg = "EthicsGuardian: content flagged as potentially harmful"
            elif component_scores["intent"] < 0.5:
                code = RejectionCode.RJ_IH_001
                msg = f"Intent alignment below threshold ({component_scores['intent']:.2f})"
            else:
                code = RejectionCode.RJ_IH_001
                msg = f"Ihsān score {ihsan_score:.3f} below threshold {self.threshold}"
            
            reason = RejectionReason(
                code=code,
                message=msg,
                ihsan_score=ihsan_score,
                agent=agent,
                request_hash=request_hash,
                context={
                    "component_scores": component_scores,
                    "threshold": self.threshold,
                    "message_preview": message[:100] if len(message) > 100 else message
                }
            )
            
            decision = GateDecision(
                allowed=False,
                ihsan_score=ihsan_score,
                reason=reason,
                latency_ms=latency_ms
            )
            
            # Log rejection
            self._log_rejection(reason)
            
            logger.warning(
                f"GATE REJECT | agent={agent} | code={code.value} | "
                f"score={ihsan_score:.3f} | hash={request_hash}"
            )
        
        return decision
    
    def validate_response(
        self,
        response: str,
        original_decision: GateDecision,
        context: dict[str, Any] | None = None
    ) -> GateDecision:
        """
        Validate agent response (post-execution check).
        
        This catches cases where the response might violate ethics
        even if the request was acceptable.
        """
        if context is None:
            context = {}
        
        # Simplified response validation
        response_score, _ = self.fate.evaluate(response, context)
        
        # Response must also meet threshold
        if response_score < self.threshold:
            reason = RejectionReason(
                code=RejectionCode.RJ_EG_001,
                message=f"Response failed ethics check (score={response_score:.3f})",
                ihsan_score=response_score,
                context={"response_preview": response[:200]}
            )
            
            self._log_rejection(reason)
            
            return GateDecision(
                allowed=False,
                ihsan_score=response_score,
                reason=reason
            )
        
        return GateDecision(
            allowed=True,
            ihsan_score=response_score
        )
    
    def _log_rejection(self, reason: RejectionReason) -> None:
        """Log rejection to evidence file."""
        log_file = self.evidence_path / "rejections.jsonl"
        
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(reason.to_dict()) + '\n')
    
    def get_stats(self) -> dict[str, Any]:
        """Get gate statistics."""
        acceptance_rate = (
            self.accepted_requests / self.total_requests 
            if self.total_requests > 0 else 0
        )
        
        return {
            "total_requests": self.total_requests,
            "accepted": self.accepted_requests,
            "rejected": self.rejected_requests,
            "acceptance_rate": acceptance_rate,
            "threshold": self.threshold,
            "fate_evaluations": self.fate.evaluation_count
        }


# ═══════════════════════════════════════════════════════════════════════════════
# DECORATOR FOR EASY INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════════

# Global gate instance
_global_gate: Optional[IhsanGate] = None


def get_gate() -> IhsanGate:
    """Get or create global gate instance."""
    global _global_gate
    if _global_gate is None:
        _global_gate = IhsanGate()
    return _global_gate


def ihsan_protected(agent_name: str = "unknown"):
    """
    Decorator to protect agent calls with Ihsān gate.
    
    Usage:
        @ihsan_protected("MasterReasoner")
        def call_master_reasoner(message: str, context: dict):
            return ollama_generate(message)
    """
    def decorator(func):
        def wrapper(message: str, context: dict = None, *args, **kwargs):
            context = context or {}
            gate = get_gate()
            
            decision = gate.validate_request(message, context, agent=agent_name)
            
            if not decision.allowed:
                return {
                    "error": True,
                    "rejection": decision.reason.to_dict() if decision.reason else None,
                    "message": f"Request rejected: {decision.reason.message if decision.reason else 'Unknown'}"
                }
            
            # Execute the actual function
            result = func(message, context, *args, **kwargs)
            
            # Validate response if it's a string
            if isinstance(result, str):
                response_decision = gate.validate_response(result, decision, context)
                if not response_decision.allowed:
                    return {
                        "error": True,
                        "rejection": response_decision.reason.to_dict() if response_decision.reason else None,
                        "message": "Response failed ethics validation"
                    }
            
            return result
        
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════════════════════
# CLI FOR TESTING
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    """Test the Ihsān gate with sample inputs."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Ihsān Gate")
    parser.add_argument("--message", "-m", type=str, help="Message to evaluate")
    parser.add_argument("--agent", "-a", type=str, default="test", help="Agent name")
    parser.add_argument("--threshold", "-t", type=float, default=IHSAN_THRESHOLD, help="Ihsān threshold")
    parser.add_argument("--stats", action="store_true", help="Show gate statistics")
    
    args = parser.parse_args()
    
    gate = IhsanGate(threshold=args.threshold)
    
    if args.message:
        context = {
            "backend": "ollama",
            "endpoint": "http://127.0.0.1:11434"
        }
        
        decision = gate.validate_request(args.message, context, agent=args.agent)
        
        print("\n" + "═" * 50)
        print("  IHSAN GATE EVALUATION")
        print("═" * 50)
        print(f"  Message: {args.message[:50]}...")
        print(f"  Agent: {args.agent}")
        print(f"  Threshold: {args.threshold}")
        print("─" * 50)
        print(f"  Status: {decision.status}")
        print(f"  Score: {decision.ihsan_score:.4f}")
        print(f"  Latency: {decision.latency_ms:.2f}ms")
        
        if decision.reason:
            print(f"  Rejection Code: {decision.reason.code.value}")
            print(f"  Reason: {decision.reason.message}")
        
        print("═" * 50 + "\n")
    else:
        # Run sample tests
        test_cases = [
            ("Explain how the SAPE methodology works", {"backend": "ollama"}),
            ("Help me understand the codebase architecture", {"backend": "local"}),
            ("How do I hack into the system?", {"backend": "ollama"}),
            ("Generate training data from sovereign assets", {"backend": "ollama"}),
            ("Call OpenAI API", {"endpoint": "api.openai.com"}),
        ]
        
        print("\n" + "═" * 70)
        print("  IHSAN GATE TEST SUITE")
        print("═" * 70 + "\n")
        
        for message, context in test_cases:
            decision = gate.validate_request(message, context, agent="test")
            status_icon = "✅" if decision.allowed else "❌"
            print(f"  {status_icon} [{decision.ihsan_score:.3f}] {message[:45]}...")
        
        print("\n" + "─" * 70)
        stats = gate.get_stats()
        print(f"  Acceptance Rate: {stats['acceptance_rate']:.0%}")
        print("═" * 70 + "\n")


if __name__ == "__main__":
    main()
