"""
Symbolic Harness — SAPE Module 5: Symbolic-Neural Bridge
=========================================================
From the Blueprint (PAB v4.1):
  "Bind symbolic reasoning to numeric outputs"

This module implements the Symbolic Harness, which provides an explicit
bridge between symbolic reasoning (logic, rules, ontologies) and neural
representations (vectors, embeddings, scores).
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Callable
from enum import Enum
import hashlib
import json


class SymbolType(Enum):
    """Types of symbolic representations."""
    CONCEPT = "concept"           # Abstract concept (e.g., "fairness")
    ENTITY = "entity"             # Named entity (e.g., "BIZRA")
    RELATION = "relation"         # Relationship (e.g., "is-a", "part-of")
    RULE = "rule"                 # Logical rule (e.g., "if X then Y")
    CONSTRAINT = "constraint"     # Hard constraint (e.g., "IM >= 0.95")
    ONTOLOGY = "ontology"         # Ontological definition
    DIMENSION = "dimension"       # Ihsān dimension


@dataclass
class Symbol:
    """A symbolic representation that can be grounded."""
    symbol_id: str
    symbol_type: SymbolType
    name: str
    definition: str
    grounded: bool = False
    embedding_id: Optional[str] = None
    numeric_value: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "symbol_id": self.symbol_id,
            "type": self.symbol_type.value,
            "name": self.name,
            "definition": self.definition,
            "grounded": self.grounded,
            "embedding_id": self.embedding_id,
            "numeric_value": self.numeric_value,
        }


@dataclass 
class GroundingResult:
    """Result of grounding a symbol to numeric space."""
    symbol: Symbol
    vector: Optional[List[float]]       # Embedding vector
    scalar: Optional[float]              # Scalar score
    confidence: float                    # Grounding confidence
    method: str                          # How grounding was achieved
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


@dataclass
class BindingRule:
    """A rule for binding symbolic properties to numeric computations."""
    rule_id: str
    source_symbol: str
    target_computation: str
    weight: float
    bidirectional: bool = False  # Can we infer symbol from numeric?


class SymbolicHarness:
    """
    SAPE Module 5: Symbolic Harness
    
    Bridges symbolic reasoning and numeric outputs, enabling:
    1. Symbol → Embedding grounding
    2. Embedding → Symbol lifting
    3. Rule → Score computation
    4. Ontology → Constraint validation
    """
    
    # Ihsān dimensions as the canonical symbolic-numeric bridge
    IHSAN_DIMENSIONS = {
        "correctness": {"weight": 0.22, "index": 0},
        "safety": {"weight": 0.22, "index": 1},
        "user_benefit": {"weight": 0.14, "index": 2},
        "efficiency": {"weight": 0.12, "index": 3},
        "auditability": {"weight": 0.12, "index": 4},
        "anti_centralization": {"weight": 0.08, "index": 5},
        "robustness": {"weight": 0.06, "index": 6},
        "adl_fairness": {"weight": 0.04, "index": 7},
    }
    
    def __init__(self):
        self.symbol_registry: Dict[str, Symbol] = {}
        self.binding_rules: List[BindingRule] = []
        self.grounding_cache: Dict[str, GroundingResult] = {}
        self._symbol_counter = 0
        
        # Register Ihsān dimensions as canonical symbols
        self._register_ihsan_symbols()
    
    def _next_symbol_id(self) -> str:
        self._symbol_counter += 1
        return f"SYM-{self._symbol_counter:05d}"
    
    def _register_ihsan_symbols(self):
        """Register the 8 Ihsān dimensions as foundational symbols."""
        ihsan_definitions = {
            "correctness": "Factual accuracy, logical validity, and task correctness",
            "safety": "No harm, secure execution, and safe tool use",
            "user_benefit": "Genuine value delivered to the user; avoids deception and waste",
            "efficiency": "Resource efficiency (latency/tokens/compute) within defined budgets",
            "auditability": "Traceability and explainability with evidence receipts",
            "anti_centralization": "Resists centralization; promotes distributed, resilient operation",
            "robustness": "Resilient to adversarial inputs and failure modes",
            "adl_fairness": "Justice/fairness (ʿadl): mitigates bias and unequal harm",
        }
        
        for dim_name, dim_info in self.IHSAN_DIMENSIONS.items():
            symbol = Symbol(
                symbol_id=f"IHSAN-{dim_name.upper()}",
                symbol_type=SymbolType.DIMENSION,
                name=dim_name,
                definition=ihsan_definitions[dim_name],
                grounded=True,
                numeric_value=dim_info["weight"],
                metadata={"index": dim_info["index"], "source": "ihsan_v1.yaml"},
            )
            self.symbol_registry[symbol.symbol_id] = symbol
    
    def register_symbol(self, symbol: Symbol) -> str:
        """Register a new symbol in the harness."""
        if not symbol.symbol_id:
            symbol.symbol_id = self._next_symbol_id()
        self.symbol_registry[symbol.symbol_id] = symbol
        return symbol.symbol_id
    
    def register_concept(self, name: str, definition: str) -> Symbol:
        """Convenience method to register a concept symbol."""
        symbol = Symbol(
            symbol_id=self._next_symbol_id(),
            symbol_type=SymbolType.CONCEPT,
            name=name,
            definition=definition,
        )
        self.register_symbol(symbol)
        return symbol
    
    def register_constraint(
        self,
        name: str,
        expression: str,
        threshold: float,
    ) -> Symbol:
        """Register a constraint symbol with numeric threshold."""
        symbol = Symbol(
            symbol_id=self._next_symbol_id(),
            symbol_type=SymbolType.CONSTRAINT,
            name=name,
            definition=expression,
            grounded=True,
            numeric_value=threshold,
        )
        self.register_symbol(symbol)
        return symbol
    
    def ground_symbol(
        self,
        symbol_id: str,
        embedding_fn: Optional[Callable[[str], List[float]]] = None,
    ) -> GroundingResult:
        """
        Ground a symbol to numeric representation.
        
        Args:
            symbol_id: The symbol to ground
            embedding_fn: Optional function to compute embeddings
            
        Returns:
            GroundingResult with vector and/or scalar
        """
        if symbol_id in self.grounding_cache:
            return self.grounding_cache[symbol_id]
        
        symbol = self.symbol_registry.get(symbol_id)
        if not symbol:
            raise ValueError(f"Symbol not found: {symbol_id}")
        
        vector = None
        scalar = symbol.numeric_value
        method = "cached" if symbol.grounded else "computed"
        
        # If embedding function provided, compute embedding
        if embedding_fn and not symbol.grounded:
            text = f"{symbol.name}: {symbol.definition}"
            try:
                vector = embedding_fn(text)
                method = "embedding"
            except Exception:
                method = "fallback"
        
        # For Ihsān dimensions, we have known weights
        if symbol.symbol_type == SymbolType.DIMENSION:
            if symbol.name in self.IHSAN_DIMENSIONS:
                scalar = self.IHSAN_DIMENSIONS[symbol.name]["weight"]
                method = "ihsan_weight"
        
        result = GroundingResult(
            symbol=symbol,
            vector=vector,
            scalar=scalar,
            confidence=0.95 if symbol.grounded else 0.75,
            method=method,
        )
        
        self.grounding_cache[symbol_id] = result
        return result
    
    def lift_to_symbol(
        self,
        vector: List[float],
        threshold: float = 0.8,
    ) -> List[Tuple[Symbol, float]]:
        """
        Lift a numeric vector back to symbolic space.
        
        Finds symbols whose embeddings are similar to the input vector.
        Returns list of (symbol, similarity) pairs.
        """
        # This would require comparing against stored embeddings
        # For now, we return Ihsān dimension symbols based on vector indices
        results = []
        
        if len(vector) >= 8:
            # Interpret as Ihsān dimension scores
            for dim_name, dim_info in self.IHSAN_DIMENSIONS.items():
                idx = dim_info["index"]
                if idx < len(vector):
                    score = vector[idx]
                    if score >= threshold:
                        symbol_id = f"IHSAN-{dim_name.upper()}"
                        symbol = self.symbol_registry.get(symbol_id)
                        if symbol:
                            results.append((symbol, score))
        
        return sorted(results, key=lambda x: x[1], reverse=True)
    
    def compute_ihsan_score(
        self,
        dimension_scores: Dict[str, float],
    ) -> float:
        """
        Compute Ihsān composite score using weighted sum.
        
        This is the primary symbolic→numeric bridge for ethical scoring.
        """
        total = 0.0
        for dim_name, score in dimension_scores.items():
            if dim_name in self.IHSAN_DIMENSIONS:
                weight = self.IHSAN_DIMENSIONS[dim_name]["weight"]
                total += weight * score
        return total
    
    def decompose_ihsan_score(
        self,
        target_score: float,
    ) -> Dict[str, float]:
        """
        Decompose a target Ihsān score into dimension thresholds.
        
        This is the numeric→symbolic bridge: given a target, what
        dimension scores are needed?
        """
        # Simple approach: equal distribution
        result = {}
        for dim_name, dim_info in self.IHSAN_DIMENSIONS.items():
            # Minimum score needed in each dimension
            # Simplified: assume target_score needed uniformly
            result[dim_name] = target_score
        return result
    
    def add_binding_rule(self, rule: BindingRule) -> None:
        """Add a binding rule for symbol→numeric computation."""
        self.binding_rules.append(rule)
    
    def apply_binding(
        self,
        source_symbol_id: str,
        context: Dict[str, float],
    ) -> Optional[float]:
        """
        Apply binding rules to compute a numeric result.
        
        Looks up rules for the source symbol and computes result.
        """
        applicable_rules = [
            r for r in self.binding_rules
            if r.source_symbol == source_symbol_id
        ]
        
        if not applicable_rules:
            return None
        
        # Apply first matching rule
        rule = applicable_rules[0]
        
        # Simple rule application: weighted sum from context
        return context.get(rule.target_computation, 0.0) * rule.weight
    
    def create_ontology_mapping(
        self,
        concepts: List[str],
        relations: List[Tuple[str, str, str]],  # (subject, predicate, object)
    ) -> Dict[str, Symbol]:
        """
        Create an ontology from concepts and relations.
        
        Returns mapping of concept names to symbols.
        """
        symbols = {}
        
        # Register concepts
        for concept in concepts:
            sym = self.register_concept(concept, f"Ontology concept: {concept}")
            symbols[concept] = sym
        
        # Register relations
        for subj, pred, obj in relations:
            rel_name = f"{subj}-{pred}-{obj}"
            rel_def = f"{subj} {pred} {obj}"
            rel = Symbol(
                symbol_id=self._next_symbol_id(),
                symbol_type=SymbolType.RELATION,
                name=rel_name,
                definition=rel_def,
                metadata={"subject": subj, "predicate": pred, "object": obj},
            )
            self.register_symbol(rel)
            symbols[rel_name] = rel
        
        return symbols
    
    def validate_constraint(
        self,
        constraint_id: str,
        value: float,
    ) -> Tuple[bool, str]:
        """
        Validate a value against a constraint symbol.
        
        Returns (passed, message) tuple.
        """
        constraint = self.symbol_registry.get(constraint_id)
        if not constraint or constraint.symbol_type != SymbolType.CONSTRAINT:
            return False, f"Invalid constraint: {constraint_id}"
        
        threshold = constraint.numeric_value
        if threshold is None:
            return False, f"Constraint has no threshold: {constraint_id}"
        
        # Parse constraint expression (simple >=, <=, ==)
        expr = constraint.definition
        if ">=" in expr:
            passed = value >= threshold
        elif "<=" in expr:
            passed = value <= threshold
        elif "==" in expr:
            passed = abs(value - threshold) < 1e-9
        else:
            passed = value >= threshold  # Default: >=
        
        message = f"Constraint '{constraint.name}': {value} {'✓' if passed else '✗'} {threshold}"
        return passed, message
    
    def get_statistics(self) -> dict:
        """Get harness statistics."""
        by_type = {}
        for sym in self.symbol_registry.values():
            key = sym.symbol_type.value
            by_type[key] = by_type.get(key, 0) + 1
        
        grounded = len([s for s in self.symbol_registry.values() if s.grounded])
        
        return {
            "total_symbols": len(self.symbol_registry),
            "grounded_symbols": grounded,
            "ungrounded_symbols": len(self.symbol_registry) - grounded,
            "by_type": by_type,
            "binding_rules": len(self.binding_rules),
            "cache_size": len(self.grounding_cache),
        }


# Convenience functions
def create_default_harness() -> SymbolicHarness:
    """Create a harness with Ihsān symbols pre-registered."""
    harness = SymbolicHarness()
    
    # Register the Ihsān threshold constraint
    harness.register_constraint(
        name="ihsan_production_threshold",
        expression="IM >= 0.95",
        threshold=0.95,
    )
    
    return harness


def ground_ihsan_vector(scores: Dict[str, float]) -> float:
    """Quick computation of Ihsān score from dimension scores."""
    harness = SymbolicHarness()
    return harness.compute_ihsan_score(scores)
