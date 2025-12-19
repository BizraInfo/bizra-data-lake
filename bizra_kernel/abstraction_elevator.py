"""
Abstraction Elevator — SAPE Module 6: Pattern Generalization
=============================================================
From the Blueprint (PAB v4.1):
  "Generalize specific solutions to principles"

This module implements the Abstraction Elevator, which identifies
patterns across specific solutions and elevates them to reusable
principles that can be applied to future problems.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Set, Tuple, Any
from enum import Enum
from collections import Counter
import hashlib
import re


class AbstractionLevel(Enum):
    """Levels of abstraction from concrete to universal."""
    INSTANCE = 0       # Specific case
    PATTERN = 1        # Recurring pattern across cases
    PRINCIPLE = 2      # General principle
    AXIOM = 3          # Universal truth
    META = 4           # Meta-level (about principles themselves)


class DomainType(Enum):
    """Knowledge domains for cross-domain generalization."""
    TECHNICAL = "technical"
    ETHICAL = "ethical"
    ECONOMIC = "economic"
    SOCIAL = "social"
    TEMPORAL = "temporal"
    CAUSAL = "causal"


@dataclass
class Instance:
    """A specific instance/solution to be generalized."""
    instance_id: str
    domain: DomainType
    description: str
    key_features: List[str]
    outcome: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class Pattern:
    """A recurring pattern extracted from multiple instances."""
    pattern_id: str
    name: str
    description: str
    abstraction_level: AbstractionLevel
    source_instances: List[str]  # Instance IDs
    key_features: List[str]      # Common features
    applicability: str           # When this pattern applies
    frequency: int               # How many times observed
    confidence: float            # 0.0 - 1.0
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> dict:
        return {
            "pattern_id": self.pattern_id,
            "name": self.name,
            "description": self.description,
            "level": self.abstraction_level.value,
            "source_count": len(self.source_instances),
            "key_features": self.key_features,
            "applicability": self.applicability,
            "frequency": self.frequency,
            "confidence": self.confidence,
        }


@dataclass
class Principle:
    """A general principle derived from patterns."""
    principle_id: str
    name: str
    statement: str              # The principle statement
    abstraction_level: AbstractionLevel
    source_patterns: List[str]  # Pattern IDs
    domains: List[DomainType]   # Applicable domains
    constraints: List[str]      # When the principle holds
    exceptions: List[str]       # Known exceptions
    evidence_strength: float    # 0.0 - 1.0
    ihsan_alignment: float      # Alignment with Ihsān principles
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    
    def to_dict(self) -> dict:
        return {
            "principle_id": self.principle_id,
            "name": self.name,
            "statement": self.statement,
            "level": self.abstraction_level.value,
            "domains": [d.value for d in self.domains],
            "evidence_strength": self.evidence_strength,
            "ihsan_alignment": self.ihsan_alignment,
        }


class AbstractionElevator:
    """
    SAPE Module 6: Abstraction Elevator
    
    Generalizes specific solutions into reusable principles by:
    1. Collecting instances (specific solutions)
    2. Detecting patterns across instances
    3. Elevating patterns to principles
    4. Cross-domain principle transfer
    """
    
    # Minimum instances needed to form a pattern
    PATTERN_THRESHOLD = 3
    # Minimum patterns needed to form a principle
    PRINCIPLE_THRESHOLD = 2
    # Minimum confidence for elevation
    CONFIDENCE_THRESHOLD = 0.7
    
    def __init__(self):
        self.instances: Dict[str, Instance] = {}
        self.patterns: Dict[str, Pattern] = {}
        self.principles: Dict[str, Principle] = {}
        self._instance_counter = 0
        self._pattern_counter = 0
        self._principle_counter = 0
        
        # Pre-registered axioms (foundational principles)
        self._register_axioms()
    
    def _next_instance_id(self) -> str:
        self._instance_counter += 1
        return f"INST-{self._instance_counter:05d}"
    
    def _next_pattern_id(self) -> str:
        self._pattern_counter += 1
        return f"PAT-{self._pattern_counter:05d}"
    
    def _next_principle_id(self) -> str:
        self._principle_counter += 1
        return f"PRIN-{self._principle_counter:04d}"
    
    def _register_axioms(self):
        """Register foundational axioms (from BIZRA Genesis)."""
        axioms = [
            Principle(
                principle_id="AXIOM-001",
                name="Record Immortality",
                statement="Every verified impact is preserved immutably across time",
                abstraction_level=AbstractionLevel.AXIOM,
                source_patterns=[],
                domains=[DomainType.TECHNICAL, DomainType.ETHICAL],
                constraints=["Requires cryptographic immutability"],
                exceptions=[],
                evidence_strength=1.0,
                ihsan_alignment=1.0,
            ),
            Principle(
                principle_id="AXIOM-002",
                name="Hardcoded Ethics",
                statement="Ethical constraints are mathematical laws, not policies",
                abstraction_level=AbstractionLevel.AXIOM,
                source_patterns=[],
                domains=[DomainType.ETHICAL, DomainType.TECHNICAL],
                constraints=["IM >= 0.95 must be enforced at protocol level"],
                exceptions=["Emergency overrides require multi-sig consensus"],
                evidence_strength=1.0,
                ihsan_alignment=1.0,
            ),
            Principle(
                principle_id="AXIOM-003",
                name="No Founder Dependency",
                statement="The system must survive without any single individual",
                abstraction_level=AbstractionLevel.AXIOM,
                source_patterns=[],
                domains=[DomainType.SOCIAL, DomainType.TECHNICAL],
                constraints=["Open source, immutable, DAO-governed"],
                exceptions=[],
                evidence_strength=1.0,
                ihsan_alignment=0.92,
            ),
        ]
        
        for axiom in axioms:
            self.principles[axiom.principle_id] = axiom
    
    def record_instance(
        self,
        domain: DomainType,
        description: str,
        key_features: List[str],
        outcome: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Instance:
        """
        Record a specific instance for pattern detection.
        
        This is the entry point: every solution/case gets recorded
        for future generalization.
        """
        instance = Instance(
            instance_id=self._next_instance_id(),
            domain=domain,
            description=description,
            key_features=key_features,
            outcome=outcome,
            context=context or {},
        )
        self.instances[instance.instance_id] = instance
        
        # Check if new patterns emerge
        self._detect_patterns()
        
        return instance
    
    def _detect_patterns(self) -> List[Pattern]:
        """
        Detect patterns across recorded instances.
        
        Uses feature co-occurrence to identify recurring patterns.
        """
        # Group instances by domain
        by_domain: Dict[DomainType, List[Instance]] = {}
        for inst in self.instances.values():
            if inst.domain not in by_domain:
                by_domain[inst.domain] = []
            by_domain[inst.domain].append(inst)
        
        new_patterns = []
        
        for domain, domain_instances in by_domain.items():
            if len(domain_instances) < self.PATTERN_THRESHOLD:
                continue
            
            # Find common features
            feature_counts = Counter()
            for inst in domain_instances:
                for feature in inst.key_features:
                    feature_counts[feature] += 1
            
            # Features appearing in multiple instances form patterns
            common_features = [
                f for f, count in feature_counts.items()
                if count >= self.PATTERN_THRESHOLD
            ]
            
            if common_features:
                # Check if this pattern already exists
                pattern_key = self._pattern_hash(common_features)
                existing = [
                    p for p in self.patterns.values()
                    if set(p.key_features) == set(common_features)
                ]
                
                if not existing:
                    pattern = Pattern(
                        pattern_id=self._next_pattern_id(),
                        name=f"{domain.value}_pattern_{len(self.patterns)+1}",
                        description=f"Recurring pattern in {domain.value} domain",
                        abstraction_level=AbstractionLevel.PATTERN,
                        source_instances=[i.instance_id for i in domain_instances],
                        key_features=common_features,
                        applicability=f"When solving {domain.value} problems",
                        frequency=len(domain_instances),
                        confidence=len(domain_instances) / len(self.instances),
                    )
                    self.patterns[pattern.pattern_id] = pattern
                    new_patterns.append(pattern)
        
        return new_patterns
    
    def _pattern_hash(self, features: List[str]) -> str:
        """Generate hash for pattern identification."""
        sorted_features = sorted(features)
        return hashlib.sha256("|".join(sorted_features).encode()).hexdigest()[:12]
    
    def elevate_to_principle(
        self,
        pattern_ids: List[str],
        name: str,
        statement: str,
        domains: Optional[List[DomainType]] = None,
    ) -> Optional[Principle]:
        """
        Manually elevate patterns to a principle.
        
        This allows explicit principle creation when automatic
        detection isn't sufficient.
        """
        patterns = [self.patterns.get(pid) for pid in pattern_ids if pid in self.patterns]
        
        if len(patterns) < self.PRINCIPLE_THRESHOLD:
            return None
        
        # Collect domains from patterns if not specified
        if domains is None:
            domains = list(set(
                inst.domain
                for p in patterns
                for iid in p.source_instances
                for inst in [self.instances.get(iid)]
                if inst
            ))
        
        # Calculate evidence strength from pattern confidences
        avg_confidence = sum(p.confidence for p in patterns) / len(patterns)
        
        principle = Principle(
            principle_id=self._next_principle_id(),
            name=name,
            statement=statement,
            abstraction_level=AbstractionLevel.PRINCIPLE,
            source_patterns=pattern_ids,
            domains=domains,
            constraints=[],
            exceptions=[],
            evidence_strength=avg_confidence,
            ihsan_alignment=self._estimate_ihsan_alignment(statement),
        )
        
        self.principles[principle.principle_id] = principle
        return principle
    
    def _estimate_ihsan_alignment(self, statement: str) -> float:
        """Estimate how well a principle aligns with Ihsān values."""
        alignment_keywords = {
            "benefit": 0.1,
            "safety": 0.15,
            "correct": 0.1,
            "fair": 0.1,
            "transparent": 0.08,
            "efficient": 0.07,
            "robust": 0.05,
            "decentralized": 0.05,
            "verified": 0.08,
            "ethical": 0.12,
            "immutable": 0.05,
            "auditable": 0.05,
        }
        
        statement_lower = statement.lower()
        score = 0.5  # Base alignment
        
        for keyword, boost in alignment_keywords.items():
            if keyword in statement_lower:
                score += boost
        
        return min(1.0, score)
    
    def auto_elevate(self) -> List[Principle]:
        """
        Automatically elevate patterns to principles based on thresholds.
        
        Runs the full elevation pipeline and returns new principles.
        """
        new_principles = []
        
        # Group patterns by domain
        by_domain: Dict[DomainType, List[Pattern]] = {}
        for pat in self.patterns.values():
            # Get domain from source instances
            for iid in pat.source_instances:
                inst = self.instances.get(iid)
                if inst:
                    if inst.domain not in by_domain:
                        by_domain[inst.domain] = []
                    by_domain[inst.domain].append(pat)
                    break
        
        for domain, domain_patterns in by_domain.items():
            if len(domain_patterns) < self.PRINCIPLE_THRESHOLD:
                continue
            
            # Find patterns with high confidence
            strong_patterns = [
                p for p in domain_patterns
                if p.confidence >= self.CONFIDENCE_THRESHOLD
            ]
            
            if len(strong_patterns) >= self.PRINCIPLE_THRESHOLD:
                # Synthesize principle from patterns
                common_features = set(strong_patterns[0].key_features)
                for p in strong_patterns[1:]:
                    common_features &= set(p.key_features)
                
                if common_features:
                    statement = self._synthesize_statement(
                        domain, list(common_features)
                    )
                    
                    principle = Principle(
                        principle_id=self._next_principle_id(),
                        name=f"{domain.value}_principle_auto",
                        statement=statement,
                        abstraction_level=AbstractionLevel.PRINCIPLE,
                        source_patterns=[p.pattern_id for p in strong_patterns],
                        domains=[domain],
                        constraints=[],
                        exceptions=[],
                        evidence_strength=sum(p.confidence for p in strong_patterns) / len(strong_patterns),
                        ihsan_alignment=self._estimate_ihsan_alignment(statement),
                    )
                    
                    self.principles[principle.principle_id] = principle
                    new_principles.append(principle)
        
        return new_principles
    
    def _synthesize_statement(
        self,
        domain: DomainType,
        features: List[str],
    ) -> str:
        """Synthesize a principle statement from features."""
        feature_str = ", ".join(features[:3])
        return f"In {domain.value} contexts, ensure {feature_str} for optimal outcomes"
    
    def apply_principle(
        self,
        principle_id: str,
        new_context: Dict[str, Any],
    ) -> Optional[str]:
        """
        Apply a principle to a new context.
        
        Returns actionable guidance based on the principle.
        """
        principle = self.principles.get(principle_id)
        if not principle:
            return None
        
        return (
            f"Based on principle '{principle.name}':\n"
            f"  Statement: {principle.statement}\n"
            f"  Constraints: {', '.join(principle.constraints) or 'None'}\n"
            f"  Exceptions: {', '.join(principle.exceptions) or 'None'}\n"
            f"  Evidence strength: {principle.evidence_strength:.0%}\n"
            f"  Ihsān alignment: {principle.ihsan_alignment:.0%}"
        )
    
    def cross_domain_transfer(
        self,
        source_principle_id: str,
        target_domain: DomainType,
    ) -> Optional[Principle]:
        """
        Transfer a principle from one domain to another.
        
        Creates an analogous principle for the target domain.
        """
        source = self.principles.get(source_principle_id)
        if not source:
            return None
        
        if target_domain in source.domains:
            return source  # Already applies
        
        # Create transferred principle with lower confidence
        transferred = Principle(
            principle_id=self._next_principle_id(),
            name=f"{source.name}_in_{target_domain.value}",
            statement=source.statement.replace(
                source.domains[0].value if source.domains else "source",
                target_domain.value,
            ),
            abstraction_level=source.abstraction_level,
            source_patterns=source.source_patterns,
            domains=[target_domain],
            constraints=source.constraints + ["Transferred from another domain"],
            exceptions=source.exceptions,
            evidence_strength=source.evidence_strength * 0.7,  # Reduced confidence
            ihsan_alignment=source.ihsan_alignment,
        )
        
        self.principles[transferred.principle_id] = transferred
        return transferred
    
    def get_principles_for_domain(
        self,
        domain: DomainType,
        min_strength: float = 0.0,
    ) -> List[Principle]:
        """Get all principles applicable to a domain."""
        return [
            p for p in self.principles.values()
            if domain in p.domains and p.evidence_strength >= min_strength
        ]
    
    def get_axioms(self) -> List[Principle]:
        """Get all registered axioms."""
        return [
            p for p in self.principles.values()
            if p.abstraction_level == AbstractionLevel.AXIOM
        ]
    
    def get_statistics(self) -> dict:
        """Get elevator statistics."""
        level_counts = Counter(
            p.abstraction_level.value for p in self.principles.values()
        )
        domain_counts = Counter(
            d.value
            for p in self.principles.values()
            for d in p.domains
        )
        
        return {
            "total_instances": len(self.instances),
            "total_patterns": len(self.patterns),
            "total_principles": len(self.principles),
            "by_level": dict(level_counts),
            "by_domain": dict(domain_counts),
            "avg_ihsan_alignment": (
                sum(p.ihsan_alignment for p in self.principles.values())
                / max(1, len(self.principles))
            ),
        }


# Convenience function for quick elevation
def quick_elevate(
    instances: List[Tuple[str, List[str], str]],  # (description, features, outcome)
    domain: DomainType = DomainType.TECHNICAL,
) -> List[Principle]:
    """Quick elevation from instances to principles."""
    elevator = AbstractionElevator()
    
    for desc, features, outcome in instances:
        elevator.record_instance(domain, desc, features, outcome)
    
    return elevator.auto_elevate()
