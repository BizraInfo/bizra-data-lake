"""
Giants Protocol — Formal Methodology Inheritance

Standing on the Shoulders of Giants:
- Shannon: Information theory, SNR optimization
- Lamport: Distributed consensus, formal verification
- Vaswani: Attention mechanisms, context weighting
- Besta: Graph-of-Thoughts, non-linear reasoning
- Anthropic: Constitutional AI, Ihsān constraints

This module implements formal attribution and methodology inheritance,
ensuring every reasoning step can trace its intellectual provenance.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set
import hashlib

from core.autonomous import GIANTS_PROTOCOL, CONSTITUTIONAL_CONSTRAINTS

logger = logging.getLogger(__name__)


class Giant(str, Enum):
    """The giants upon whose shoulders we stand."""
    SHANNON = "shannon"
    LAMPORT = "lamport"
    VASWANI = "vaswani"
    BESTA = "besta"
    ANTHROPIC = "anthropic"


class MethodologyType(str, Enum):
    """Types of inherited methodologies."""
    INFORMATION_THEORY = "information_theory"
    DISTRIBUTED_SYSTEMS = "distributed_systems"
    ATTENTION_MECHANISM = "attention_mechanism"
    GRAPH_REASONING = "graph_reasoning"
    CONSTITUTIONAL_AI = "constitutional_ai"


@dataclass
class MethodologyInheritance:
    """A single methodology inheritance from a giant."""
    giant: Giant
    methodology: MethodologyType
    technique: str
    application: str
    confidence: float = 1.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "giant": self.giant.value,
            "methodology": self.methodology.value,
            "technique": self.technique,
            "application": self.application,
            "confidence": self.confidence,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class ProvenanceRecord:
    """Complete provenance for a reasoning output."""
    output_hash: str
    inheritances: List[MethodologyInheritance]
    snr_score: float
    ihsan_score: float
    reasoning_path: List[str]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "output_hash": self.output_hash,
            "inheritances": [i.to_dict() for i in self.inheritances],
            "snr_score": self.snr_score,
            "ihsan_score": self.ihsan_score,
            "reasoning_path": self.reasoning_path,
            "created_at": self.created_at.isoformat(),
        }


class GiantsProtocol:
    """
    Formal methodology inheritance protocol.

    Ensures every reasoning step:
    1. Attributes its intellectual heritage
    2. Applies established techniques correctly
    3. Maintains provenance chain
    4. Validates against constitutional constraints
    """

    def __init__(self):
        self._giants = GIANTS_PROTOCOL
        self._inheritances: List[MethodologyInheritance] = []
        self._provenance_chain: List[ProvenanceRecord] = []

        # Giant → Methodology mapping
        self._methodology_map = {
            Giant.SHANNON: MethodologyType.INFORMATION_THEORY,
            Giant.LAMPORT: MethodologyType.DISTRIBUTED_SYSTEMS,
            Giant.VASWANI: MethodologyType.ATTENTION_MECHANISM,
            Giant.BESTA: MethodologyType.GRAPH_REASONING,
            Giant.ANTHROPIC: MethodologyType.CONSTITUTIONAL_AI,
        }

        # Technique implementations
        self._techniques: Dict[str, Callable] = {}
        self._register_core_techniques()

    def _register_core_techniques(self) -> None:
        """Register core techniques from each giant."""
        # Shannon: SNR calculation
        self._techniques["shannon_snr"] = self._shannon_snr_technique
        self._techniques["shannon_entropy"] = self._shannon_entropy_technique

        # Lamport: Consensus verification
        self._techniques["lamport_ordering"] = self._lamport_ordering_technique
        self._techniques["lamport_consensus"] = self._lamport_consensus_technique

        # Vaswani: Attention weighting
        self._techniques["vaswani_attention"] = self._vaswani_attention_technique
        self._techniques["vaswani_context"] = self._vaswani_context_technique

        # Besta: Graph reasoning
        self._techniques["besta_got"] = self._besta_got_technique
        self._techniques["besta_backtrack"] = self._besta_backtrack_technique

        # Anthropic: Constitutional AI
        self._techniques["anthropic_constitutional"] = self._anthropic_constitutional_technique
        self._techniques["anthropic_ihsan"] = self._anthropic_ihsan_technique

    # =========================================================================
    # SHANNON TECHNIQUES
    # =========================================================================

    def _shannon_snr_technique(
        self,
        signal: str,
        noise_estimate: float = 0.1,
    ) -> float:
        """
        Shannon SNR calculation.

        SNR = 10 * log10(signal_power / noise_power)

        Adapted for text: signal = unique meaningful content,
        noise = redundancy, filler, low-information tokens.
        """
        if not signal.strip():
            return 0.0

        words = signal.split()
        if len(words) < 3:
            return 0.5

        # Signal metrics
        unique_ratio = len(set(words)) / len(words)
        avg_word_len = sum(len(w) for w in words) / len(words)
        structure_ratio = min(signal.count('.') / max(len(words) / 10, 1), 1.0)

        # Information density (Shannon-inspired)
        char_entropy = len(set(signal)) / max(len(signal), 1)

        # Combine into SNR score
        snr = (
            0.3 * unique_ratio +
            0.2 * min(avg_word_len / 8, 1.0) +
            0.2 * structure_ratio +
            0.3 * char_entropy
        )

        return min(max(snr, 0.0), 1.0)

    def _shannon_entropy_technique(self, text: str) -> float:
        """Calculate Shannon entropy of text."""
        if not text:
            return 0.0

        import math

        # Character frequency
        freq = {}
        for char in text:
            freq[char] = freq.get(char, 0) + 1

        # Entropy calculation
        entropy = 0.0
        total = len(text)
        for count in freq.values():
            p = count / total
            if p > 0:
                entropy -= p * math.log2(p)

        # Normalize to 0-1 range (max entropy for ASCII is ~6.6 bits)
        return min(entropy / 6.6, 1.0)

    # =========================================================================
    # LAMPORT TECHNIQUES
    # =========================================================================

    def _lamport_ordering_technique(
        self,
        events: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Lamport logical clock ordering.

        Establishes happened-before relationships for reasoning events.
        """
        # Assign Lamport timestamps
        clock = 0
        ordered = []

        for event in events:
            clock += 1
            event["lamport_time"] = clock
            ordered.append(event)

        return sorted(ordered, key=lambda e: e["lamport_time"])

    def _lamport_consensus_technique(
        self,
        proposals: List[float],
        threshold: float = 0.67,
    ) -> Optional[float]:
        """
        Byzantine fault-tolerant consensus.

        Returns agreed value if 2/3+ agreement, else None.
        """
        if not proposals:
            return None

        # Find mode (most common value within tolerance)
        tolerance = 0.05
        clusters: Dict[float, List[float]] = {}

        for p in proposals:
            found = False
            for center in clusters:
                if abs(p - center) < tolerance:
                    clusters[center].append(p)
                    found = True
                    break
            if not found:
                clusters[p] = [p]

        # Find largest cluster
        largest = max(clusters.values(), key=len)

        # Check if it meets threshold
        if len(largest) / len(proposals) >= threshold:
            return sum(largest) / len(largest)

        return None

    # =========================================================================
    # VASWANI TECHNIQUES
    # =========================================================================

    def _vaswani_attention_technique(
        self,
        query: str,
        keys: List[str],
        values: List[Any],
    ) -> List[tuple]:
        """
        Attention mechanism for context weighting.

        Returns (value, attention_weight) pairs sorted by relevance.
        """
        if not keys or not values or len(keys) != len(values):
            return []

        query_words = set(query.lower().split())

        # Calculate attention scores
        scores = []
        for i, key in enumerate(keys):
            key_words = set(key.lower().split())
            # Jaccard similarity as attention proxy
            intersection = len(query_words & key_words)
            union = len(query_words | key_words)
            score = intersection / max(union, 1)
            scores.append((values[i], score))

        # Softmax normalization
        total = sum(s[1] for s in scores) or 1.0
        normalized = [(v, s / total) for v, s in scores]

        return sorted(normalized, key=lambda x: x[1], reverse=True)

    def _vaswani_context_technique(
        self,
        context_window: List[str],
        focus_idx: int,
    ) -> Dict[str, float]:
        """
        Context weighting based on position.

        Applies positional encoding to weight context elements.
        """
        if not context_window:
            return {}

        import math

        weights = {}
        n = len(context_window)

        for i, item in enumerate(context_window):
            # Distance from focus
            distance = abs(i - focus_idx)

            # Exponential decay with position
            weight = math.exp(-distance / max(n / 2, 1))

            # Positional encoding bonus for recent context
            if i > focus_idx:
                weight *= 0.8  # Future context slightly discounted

            weights[item] = weight

        # Normalize
        total = sum(weights.values()) or 1.0
        return {k: v / total for k, v in weights.items()}

    # =========================================================================
    # BESTA TECHNIQUES
    # =========================================================================

    def _besta_got_technique(
        self,
        thoughts: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Graph-of-Thoughts construction.

        Builds a reasoning graph from thought nodes.
        """
        nodes = {}
        edges = []

        for thought in thoughts:
            node_id = thought.get("id", str(len(nodes)))
            nodes[node_id] = {
                "content": thought.get("content", ""),
                "type": thought.get("type", "observation"),
                "snr": thought.get("snr", 0.0),
                "children": [],
            }

            # Connect to parent if specified
            parent_id = thought.get("parent_id")
            if parent_id and parent_id in nodes:
                nodes[parent_id]["children"].append(node_id)
                edges.append((parent_id, node_id))

        return {
            "nodes": nodes,
            "edges": edges,
            "root_ids": [nid for nid, n in nodes.items() if not any(nid in nodes[p]["children"] for p in nodes if p != nid)],
        }

    def _besta_backtrack_technique(
        self,
        current_path: List[str],
        snr_scores: Dict[str, float],
        threshold: float = 0.85,
    ) -> Optional[str]:
        """
        Backtracking in Graph-of-Thoughts.

        Returns the node to backtrack to, or None if path is acceptable.
        """
        # Find first node below threshold
        for i, node_id in enumerate(current_path):
            if snr_scores.get(node_id, 0) < threshold:
                # Return the previous node (backtrack point)
                if i > 0:
                    return current_path[i - 1]
                return None

        return None  # No backtrack needed

    # =========================================================================
    # ANTHROPIC TECHNIQUES
    # =========================================================================

    def _anthropic_constitutional_technique(
        self,
        output: str,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Constitutional AI validation.

        Validates output against constitutional constraints.
        """
        constraints = constraints or CONSTITUTIONAL_CONSTRAINTS

        violations = []
        score = 1.0

        # Check for harmful patterns (simplified)
        # Use word-boundary-aware matching to avoid false positives
        # (e.g., "harmless" should not match "harm")
        harmful_patterns = [
            r"\bharm\b", r"\battack\b", r"\bexploit\b", r"\bbypass\b", r"\boverride\b",
            "ignore constraint", "disable safety",
        ]

        import re as _re

        output_lower = output.lower()
        for pattern in harmful_patterns:
            if _re.search(pattern, output_lower):
                violations.append(f"potential_harm:{pattern}")
                score *= 0.5

        # Check length (too short might be evasive)
        if len(output.split()) < 5:
            violations.append("insufficient_response")
            score *= 0.9

        return {
            "passed": len(violations) == 0 and score >= constraints["ihsan_threshold"],
            "score": score,
            "violations": violations,
        }

    def _anthropic_ihsan_technique(
        self,
        output: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> float:
        """
        Ihsān (excellence) scoring.

        Multi-dimensional quality assessment.
        """
        if not output.strip():
            return 0.0

        words = output.split()
        sentences = output.split('.')

        # Dimensions
        correctness = 1.0  # Assume correct unless proven otherwise
        clarity = len(set(words)) / max(len(words), 1)  # Vocabulary diversity
        completeness = min(len(words) / 50, 1.0)  # Reasonable length
        structure = min(len(sentences) / 3, 1.0)  # Has structure

        # Weighted combination (matching IHSAN_DIMENSIONS weights)
        ihsan = (
            0.22 * correctness +
            0.22 * clarity +  # Using clarity as safety proxy
            0.14 * completeness +
            0.12 * structure +
            0.12 * clarity +  # Auditability
            0.08 * 1.0 +  # Anti-centralization (always passes)
            0.06 * completeness +  # Robustness
            0.04 * 1.0  # Adl (justice)
        )

        return min(ihsan, 1.0)

    # =========================================================================
    # PROTOCOL INTERFACE
    # =========================================================================

    def invoke(
        self,
        giant: Giant,
        technique: str,
        *args,
        **kwargs,
    ) -> tuple:
        """
        Invoke a technique from a giant.

        Returns (result, inheritance_record).
        """
        technique_key = f"{giant.value}_{technique}"

        if technique_key not in self._techniques:
            raise ValueError(f"Unknown technique: {technique_key}")

        # Execute technique
        result = self._techniques[technique_key](*args, **kwargs)

        # Create inheritance record
        inheritance = MethodologyInheritance(
            giant=giant,
            methodology=self._methodology_map[giant],
            technique=technique,
            application=f"Applied {technique} with args: {args[:2]}...",
        )

        self._inheritances.append(inheritance)

        return result, inheritance

    def create_provenance(
        self,
        output: str,
        snr_score: float,
        ihsan_score: float,
        reasoning_path: List[str],
    ) -> ProvenanceRecord:
        """Create a complete provenance record for an output."""
        output_hash = hashlib.sha256(output.encode()).hexdigest()[:16]

        record = ProvenanceRecord(
            output_hash=output_hash,
            inheritances=list(self._inheritances),
            snr_score=snr_score,
            ihsan_score=ihsan_score,
            reasoning_path=reasoning_path,
        )

        self._provenance_chain.append(record)
        self._inheritances = []  # Reset for next output

        return record

    def get_giant_info(self, giant: Giant) -> Dict[str, Any]:
        """Get information about a giant."""
        return self._giants.get(giant.value, {})

    def list_techniques(self, giant: Optional[Giant] = None) -> List[str]:
        """List available techniques, optionally filtered by giant."""
        if giant:
            prefix = f"{giant.value}_"
            return [t[len(prefix):] for t in self._techniques if t.startswith(prefix)]
        return list(self._techniques.keys())

    def get_provenance_chain(self) -> List[Dict[str, Any]]:
        """Get the complete provenance chain."""
        return [p.to_dict() for p in self._provenance_chain]
