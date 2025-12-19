"""
Tension Studio — SAPE Module 7: Contradiction Detection & Resolution
=====================================================================
From the Blueprint (PAB v4.1):
  "Identify and resolve contradictions in reasoning"

This module implements the Tension Studio, which detects logical tensions
between different reasoning paths and synthesizes coherent resolutions.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
from enum import Enum
import re


class TensionType(Enum):
    """Categories of logical tension."""
    LOGICAL_CONTRADICTION = "logical_contradiction"      # A AND NOT A
    TEMPORAL_INCONSISTENCY = "temporal_inconsistency"    # Before AND After conflicts
    RESOURCE_CONFLICT = "resource_conflict"              # Competing resource demands
    VALUE_TRADE_OFF = "value_trade_off"                  # Ethical dimension conflicts
    SCOPE_AMBIGUITY = "scope_ambiguity"                  # Unclear boundaries
    CAUSAL_LOOP = "causal_loop"                          # Circular dependencies
    PRIORITY_CONFLICT = "priority_conflict"              # Competing priorities


@dataclass
class Tension:
    """A detected tension between reasoning elements."""
    tension_id: str
    tension_type: TensionType
    element_a: str
    element_b: str
    description: str
    severity: float  # 0.0 - 1.0
    detected_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    resolved: bool = False
    resolution: Optional[str] = None
    resolution_strategy: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "tension_id": self.tension_id,
            "type": self.tension_type.value,
            "element_a": self.element_a,
            "element_b": self.element_b,
            "description": self.description,
            "severity": self.severity,
            "resolved": self.resolved,
            "resolution": self.resolution,
            "resolution_strategy": self.resolution_strategy,
        }


@dataclass
class ResolutionStrategy:
    """A strategy for resolving a particular type of tension."""
    name: str
    applicable_types: List[TensionType]
    description: str
    priority: int  # Higher = try first


class TensionStudio:
    """
    SAPE Module 7: Tension Studio
    
    Identifies contradictions and synthesizes coherent resolutions
    across multi-agent reasoning outputs.
    """
    
    # Contradiction patterns for automatic detection
    CONTRADICTION_PATTERNS = [
        # Logical opposites
        (r"\balways\b", r"\bnever\b"),
        (r"\bmust\b", r"\bmust not\b"),
        (r"\brequire[ds]?\b", r"\bforbid(?:den|s)?\b"),
        (r"\ball\b", r"\bnone\b"),
        (r"\byes\b", r"\bno\b"),
        (r"\btrue\b", r"\bfalse\b"),
        (r"\benable[ds]?\b", r"\bdisable[ds]?\b"),
        (r"\binclude[ds]?\b", r"\bexclude[ds]?\b"),
        (r"\ballow[eds]?\b", r"\bblock(?:ed|s)?\b"),
        (r"\bapprove[ds]?\b", r"\breject(?:ed|s)?\b"),
    ]
    
    # Temporal conflict patterns
    TEMPORAL_PATTERNS = [
        (r"\bbefore\b", r"\bafter\b"),
        (r"\bfirst\b", r"\blast\b"),
        (r"\binitial\b", r"\bfinal\b"),
        (r"\bstart\b", r"\bend\b"),
    ]
    
    RESOLUTION_STRATEGIES = [
        ResolutionStrategy(
            name="Scope Clarification",
            applicable_types=[TensionType.LOGICAL_CONTRADICTION, TensionType.SCOPE_AMBIGUITY],
            description="Clarify that the contradicting statements apply to different scopes/contexts",
            priority=100,
        ),
        ResolutionStrategy(
            name="Temporal Sequencing",
            applicable_types=[TensionType.TEMPORAL_INCONSISTENCY, TensionType.CAUSAL_LOOP],
            description="Establish clear temporal ordering to resolve sequence conflicts",
            priority=90,
        ),
        ResolutionStrategy(
            name="Priority Ranking",
            applicable_types=[TensionType.PRIORITY_CONFLICT, TensionType.RESOURCE_CONFLICT],
            description="Apply explicit priority ranking to resolve competing demands",
            priority=85,
        ),
        ResolutionStrategy(
            name="Value Balancing",
            applicable_types=[TensionType.VALUE_TRADE_OFF],
            description="Apply Ihsān-weighted composite to balance competing values",
            priority=95,
        ),
        ResolutionStrategy(
            name="Synthesis",
            applicable_types=list(TensionType),  # Applies to all
            description="Synthesize a novel solution that transcends the apparent contradiction",
            priority=50,
        ),
    ]
    
    def __init__(self):
        self.detected_tensions: List[Tension] = []
        self.resolution_history: List[Tuple[Tension, str]] = []
        self._tension_counter = 0
    
    def _next_tension_id(self) -> str:
        self._tension_counter += 1
        return f"TENSION-{self._tension_counter:04d}"
    
    def analyze_text(self, text: str) -> List[Tension]:
        """Analyze text for internal contradictions."""
        detected = []
        text_lower = text.lower()
        
        # Check logical contradiction patterns
        for pattern_a, pattern_b in self.CONTRADICTION_PATTERNS:
            if re.search(pattern_a, text_lower) and re.search(pattern_b, text_lower):
                detected.append(Tension(
                    tension_id=self._next_tension_id(),
                    tension_type=TensionType.LOGICAL_CONTRADICTION,
                    element_a=pattern_a,
                    element_b=pattern_b,
                    description=f"Contradicting terms detected: '{pattern_a}' vs '{pattern_b}'",
                    severity=0.8,
                ))
        
        # Check temporal conflict patterns
        for pattern_a, pattern_b in self.TEMPORAL_PATTERNS:
            if re.search(pattern_a, text_lower) and re.search(pattern_b, text_lower):
                # Only flag if they appear in potentially conflicting context
                detected.append(Tension(
                    tension_id=self._next_tension_id(),
                    tension_type=TensionType.TEMPORAL_INCONSISTENCY,
                    element_a=pattern_a,
                    element_b=pattern_b,
                    description=f"Potential temporal conflict: '{pattern_a}' vs '{pattern_b}'",
                    severity=0.5,  # Lower severity, may be intentional
                ))
        
        self.detected_tensions.extend(detected)
        return detected
    
    def analyze_multi_agent(
        self,
        agent_outputs: Dict[str, str],
    ) -> List[Tension]:
        """
        Analyze outputs from multiple agents for inter-agent tensions.
        
        This is the core function for detecting tensions between PAT agents.
        """
        detected = []
        agent_names = list(agent_outputs.keys())
        
        # Pairwise comparison of agent outputs
        for i, agent_a in enumerate(agent_names):
            for agent_b in agent_names[i+1:]:
                output_a = agent_outputs[agent_a].lower()
                output_b = agent_outputs[agent_b].lower()
                
                # Check for contradicting conclusions
                tensions = self._detect_cross_agent_tensions(
                    agent_a, output_a,
                    agent_b, output_b,
                )
                detected.extend(tensions)
        
        self.detected_tensions.extend(detected)
        return detected
    
    def _detect_cross_agent_tensions(
        self,
        agent_a: str, output_a: str,
        agent_b: str, output_b: str,
    ) -> List[Tension]:
        """Detect tensions between two agent outputs."""
        tensions = []
        
        # Check for opposite recommendations
        for pattern_a, pattern_b in self.CONTRADICTION_PATTERNS:
            if re.search(pattern_a, output_a) and re.search(pattern_b, output_b):
                tensions.append(Tension(
                    tension_id=self._next_tension_id(),
                    tension_type=TensionType.LOGICAL_CONTRADICTION,
                    element_a=f"{agent_a}: {pattern_a}",
                    element_b=f"{agent_b}: {pattern_b}",
                    description=f"Agents {agent_a} and {agent_b} have contradicting positions",
                    severity=0.7,
                ))
        
        # Detect priority conflicts (both agents want resources)
        resource_keywords = ["priority", "critical", "urgent", "must", "essential"]
        a_claims_priority = any(kw in output_a for kw in resource_keywords)
        b_claims_priority = any(kw in output_b for kw in resource_keywords)
        
        if a_claims_priority and b_claims_priority:
            tensions.append(Tension(
                tension_id=self._next_tension_id(),
                tension_type=TensionType.PRIORITY_CONFLICT,
                element_a=f"{agent_a}: claims priority",
                element_b=f"{agent_b}: claims priority",
                description=f"Both {agent_a} and {agent_b} claim priority status",
                severity=0.5,
            ))
        
        return tensions
    
    def resolve_tension(
        self,
        tension: Tension,
        context: Optional[Dict[str, any]] = None,
    ) -> Tension:
        """
        Attempt to resolve a tension using applicable strategies.
        
        Returns the tension with resolution fields populated.
        """
        # Find applicable strategies, sorted by priority
        applicable = [
            s for s in self.RESOLUTION_STRATEGIES
            if tension.tension_type in s.applicable_types
        ]
        applicable.sort(key=lambda s: s.priority, reverse=True)
        
        if not applicable:
            tension.resolution = "No applicable resolution strategy found"
            return tension
        
        strategy = applicable[0]
        
        # Apply resolution based on strategy
        if strategy.name == "Scope Clarification":
            tension.resolution = self._resolve_by_scope(tension, context)
        elif strategy.name == "Temporal Sequencing":
            tension.resolution = self._resolve_by_temporal(tension, context)
        elif strategy.name == "Priority Ranking":
            tension.resolution = self._resolve_by_priority(tension, context)
        elif strategy.name == "Value Balancing":
            tension.resolution = self._resolve_by_value_balance(tension, context)
        else:
            tension.resolution = self._resolve_by_synthesis(tension, context)
        
        tension.resolved = True
        tension.resolution_strategy = strategy.name
        
        self.resolution_history.append((tension, strategy.name))
        return tension
    
    def _resolve_by_scope(
        self,
        tension: Tension,
        context: Optional[Dict] = None,
    ) -> str:
        """Resolve by clarifying that statements apply to different scopes."""
        return (
            f"SCOPE RESOLUTION: The apparent contradiction between "
            f"'{tension.element_a}' and '{tension.element_b}' can be resolved "
            f"by recognizing they apply to different contexts. "
            f"Recommend: Explicitly scope each statement to its applicable domain."
        )
    
    def _resolve_by_temporal(
        self,
        tension: Tension,
        context: Optional[Dict] = None,
    ) -> str:
        """Resolve by establishing temporal ordering."""
        return (
            f"TEMPORAL RESOLUTION: The conflict between "
            f"'{tension.element_a}' and '{tension.element_b}' can be resolved "
            f"by establishing a clear sequence. "
            f"Recommend: Define explicit ordering constraints."
        )
    
    def _resolve_by_priority(
        self,
        tension: Tension,
        context: Optional[Dict] = None,
    ) -> str:
        """Resolve by applying priority ranking."""
        return (
            f"PRIORITY RESOLUTION: The competing demands from "
            f"'{tension.element_a}' and '{tension.element_b}' require prioritization. "
            f"Recommend: Apply Ihsān-weighted ranking (safety > correctness > user_benefit > ...)."
        )
    
    def _resolve_by_value_balance(
        self,
        tension: Tension,
        context: Optional[Dict] = None,
    ) -> str:
        """Resolve by balancing competing values using Ihsān weights."""
        return (
            f"VALUE BALANCE RESOLUTION: The trade-off between "
            f"'{tension.element_a}' and '{tension.element_b}' should be balanced "
            f"using the Ihsān 8-dimension weighted composite. "
            f"Recommend: Calculate IM score for each option and select highest."
        )
    
    def _resolve_by_synthesis(
        self,
        tension: Tension,
        context: Optional[Dict] = None,
    ) -> str:
        """Resolve by synthesizing a novel solution."""
        return (
            f"SYNTHESIS RESOLUTION: The tension between "
            f"'{tension.element_a}' and '{tension.element_b}' may require "
            f"a novel approach that transcends both positions. "
            f"Recommend: Explore Graph-of-Thought to find cross-domain synthesis."
        )
    
    def get_unresolved(self) -> List[Tension]:
        """Get all unresolved tensions."""
        return [t for t in self.detected_tensions if not t.resolved]
    
    def get_statistics(self) -> dict:
        """Get tension detection and resolution statistics."""
        by_type = {}
        for t in self.detected_tensions:
            key = t.tension_type.value
            by_type[key] = by_type.get(key, 0) + 1
        
        return {
            "total_detected": len(self.detected_tensions),
            "resolved": len([t for t in self.detected_tensions if t.resolved]),
            "unresolved": len(self.get_unresolved()),
            "by_type": by_type,
            "resolution_history_count": len(self.resolution_history),
        }


# Convenience function for quick tension check
def quick_tension_check(text: str) -> List[Tension]:
    """Quick check for tensions in a text block."""
    studio = TensionStudio()
    return studio.analyze_text(text)
