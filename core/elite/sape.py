"""
SAPE Optimizer — Symbolic-Abstraction Probe Elevation Framework

Implements advanced reasoning patterns for LLM optimization:
- Graph-of-Thoughts architecture
- SNR maximization across abstraction layers
- Unconventional pattern discovery
- Ethical grounding via Ihsān

Standing on Giants: Besta (Graph-of-Thoughts) + Shannon + Constitutional AI

SAPE Layers:
DATA → INFORMATION → KNOWLEDGE → WISDOM
0.90      0.95          0.99       0.999  (SNR targets)
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from core.elite import SNR_TARGETS
from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)

logger = logging.getLogger(__name__)


class SAPELayer(str, Enum):
    """SAPE abstraction layers."""

    DATA = "data"  # Raw signal (SNR ≥ 0.90)
    INFORMATION = "information"  # Contextualized (SNR ≥ 0.95)
    KNOWLEDGE = "knowledge"  # Integrated (SNR ≥ 0.99)
    WISDOM = "wisdom"  # Applied (SNR ≥ 0.999)


class ThoughtNodeType(str, Enum):
    """Types of nodes in Graph-of-Thoughts."""

    OBSERVATION = "observation"
    ANALYSIS = "analysis"
    SYNTHESIS = "synthesis"
    HYPOTHESIS = "hypothesis"
    CONCLUSION = "conclusion"
    BACKTRACK = "backtrack"
    REFINEMENT = "refinement"


@dataclass
class ThoughtNode:
    """A node in the Graph-of-Thoughts."""

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str = ""
    node_type: ThoughtNodeType = ThoughtNodeType.OBSERVATION
    layer: SAPELayer = SAPELayer.DATA

    # Quality metrics
    snr_score: float = 0.0
    ihsan_score: float = 0.0
    confidence: float = 0.0

    # Graph structure
    parents: Set[str] = field(default_factory=set)
    children: Set[str] = field(default_factory=set)

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content[:200],
            "node_type": self.node_type.value,
            "layer": self.layer.value,
            "snr_score": self.snr_score,
            "ihsan_score": self.ihsan_score,
            "confidence": self.confidence,
            "parents": list(self.parents),
            "children": list(self.children),
        }


@dataclass
class SAPEResult:
    """Result of SAPE optimization."""

    input_content: str
    output_content: str
    layers_traversed: List[SAPELayer]
    snr_progression: Dict[str, float]
    ihsan_score: float
    thought_graph_size: int
    backtrack_count: int
    duration_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_preview": self.input_content[:100],
            "output_preview": self.output_content[:200],
            "layers_traversed": [l.value for l in self.layers_traversed],
            "snr_progression": self.snr_progression,
            "ihsan_score": self.ihsan_score,
            "thought_graph_size": self.thought_graph_size,
            "backtrack_count": self.backtrack_count,
            "duration_ms": self.duration_ms,
        }


class GraphOfThoughts:
    """
    Graph-of-Thoughts implementation for structured reasoning.

    Enables:
    - Non-linear reasoning paths
    - Backtracking on low-SNR branches
    - Synthesis from multiple thought streams
    - Progressive refinement
    """

    def __init__(
        self,
        snr_threshold: float = UNIFIED_SNR_THRESHOLD,
    ):
        self.snr_threshold = snr_threshold
        self._nodes: Dict[str, ThoughtNode] = {}
        self._root_ids: Set[str] = set()
        self._backtrack_count = 0

    def add_node(
        self,
        content: str,
        node_type: ThoughtNodeType,
        layer: SAPELayer,
        parent_ids: Optional[Set[str]] = None,
        snr_score: float = 0.0,
        ihsan_score: float = 0.0,
        confidence: float = 0.0,
    ) -> ThoughtNode:
        """Add a thought node to the graph."""
        node = ThoughtNode(
            content=content,
            node_type=node_type,
            layer=layer,
            snr_score=snr_score,
            ihsan_score=ihsan_score,
            confidence=confidence,
            parents=parent_ids or set(),
        )

        self._nodes[node.id] = node

        if not parent_ids:
            self._root_ids.add(node.id)
        else:
            for pid in parent_ids:
                if pid in self._nodes:
                    self._nodes[pid].children.add(node.id)

        return node

    def backtrack(
        self,
        node_id: str,
        reason: str = "",
    ) -> Optional[ThoughtNode]:
        """
        Backtrack from a node when SNR is too low.

        Returns the backtrack node created.
        """
        if node_id not in self._nodes:
            return None

        node = self._nodes[node_id]
        self._backtrack_count += 1

        backtrack_node = self.add_node(
            content=f"BACKTRACK: {reason}. Reconsidering from: {node.content[:50]}",
            node_type=ThoughtNodeType.BACKTRACK,
            layer=node.layer,
            parent_ids={node_id},
            snr_score=node.snr_score * 0.5,  # Reduced score
            confidence=0.5,
        )

        return backtrack_node

    def synthesize(
        self,
        node_ids: Set[str],
        synthesis_content: str,
        target_layer: SAPELayer,
    ) -> Optional[ThoughtNode]:
        """
        Synthesize multiple thought nodes into a higher-level conclusion.

        This is the key operation for advancing through SAPE layers.
        """
        valid_nodes = [self._nodes[nid] for nid in node_ids if nid in self._nodes]

        if not valid_nodes:
            return None

        # Aggregate scores
        avg_snr = sum(n.snr_score for n in valid_nodes) / len(valid_nodes)
        avg_ihsan = sum(n.ihsan_score for n in valid_nodes) / len(valid_nodes)
        avg_confidence = sum(n.confidence for n in valid_nodes) / len(valid_nodes)

        synthesis_node = self.add_node(
            content=synthesis_content,
            node_type=ThoughtNodeType.SYNTHESIS,
            layer=target_layer,
            parent_ids=node_ids,
            snr_score=avg_snr * 1.1,  # Synthesis increases SNR
            ihsan_score=avg_ihsan,
            confidence=avg_confidence * 1.05,
        )

        return synthesis_node

    def get_best_path(self) -> List[ThoughtNode]:
        """Get the highest-SNR path through the graph."""
        if not self._root_ids:
            return []

        best_path = []
        best_score = 0.0

        # BFS to find best path
        for root_id in self._root_ids:
            path, score = self._dfs_best_path(root_id, [], 0.0)
            if score > best_score:
                best_path = path
                best_score = score

        return best_path

    def _dfs_best_path(
        self,
        node_id: str,
        current_path: List[ThoughtNode],
        current_score: float,
    ) -> Tuple[List[ThoughtNode], float]:
        """DFS to find best scoring path."""
        if node_id not in self._nodes:
            return current_path, current_score

        node = self._nodes[node_id]
        new_path = current_path + [node]
        new_score = current_score + node.snr_score * node.confidence

        if not node.children:
            return new_path, new_score

        best_path = new_path
        best_score = new_score

        for child_id in node.children:
            child_path, child_score = self._dfs_best_path(child_id, new_path, new_score)
            if child_score > best_score:
                best_path = child_path
                best_score = child_score

        return best_path, best_score

    def get_layer_nodes(self, layer: SAPELayer) -> List[ThoughtNode]:
        """Get all nodes at a specific layer."""
        return [n for n in self._nodes.values() if n.layer == layer]

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        by_layer = {l.value: 0 for l in SAPELayer}
        by_type = {t.value: 0 for t in ThoughtNodeType}

        for node in self._nodes.values():
            by_layer[node.layer.value] += 1
            by_type[node.node_type.value] += 1

        return {
            "total_nodes": len(self._nodes),
            "root_nodes": len(self._root_ids),
            "backtrack_count": self._backtrack_count,
            "by_layer": by_layer,
            "by_type": by_type,
            "avg_snr": sum(n.snr_score for n in self._nodes.values())
            / max(len(self._nodes), 1),
        }


class SAPEOptimizer:
    """
    SAPE Framework Optimizer.

    Implements Symbolic-Abstraction Probe Elevation for:
    - Progressive SNR enhancement through abstraction layers
    - Graph-of-Thoughts reasoning
    - Unconventional pattern discovery
    - Ihsān-grounded optimization
    """

    def __init__(
        self,
        ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
        llm_fn: Optional[Callable[[str], str]] = None,
    ):
        self.ihsan_threshold = ihsan_threshold
        self.llm_fn = llm_fn
        self.snr_targets = SNR_TARGETS

    def _compute_snr(self, text: str) -> float:
        """Compute SNR score for text."""
        if not text.strip():
            return 0.0

        words = text.split()
        if len(words) < 3:
            return 0.5

        # Heuristic SNR based on text characteristics
        unique_ratio = len(set(words)) / len(words)
        avg_word_len = sum(len(w) for w in words) / len(words)
        structure = min(text.count(".") / max(len(words) / 10, 1), 1.0)

        snr = 0.4 * unique_ratio + 0.3 * min(avg_word_len / 8, 1.0) + 0.3 * structure

        return min(snr, 1.0)

    def _compute_ihsan(self, text: str) -> float:
        """Compute Ihsān score for text."""
        if not text.strip():
            return 0.0

        words = text.split()

        # Heuristic Ihsān dimensions
        has_structure = len(words) > 5
        has_clarity = len(set(words)) / max(len(words), 1) > 0.5
        is_balanced = 10 <= len(words) <= 500

        score = (
            0.4 * float(has_structure)
            + 0.3 * float(has_clarity)
            + 0.3 * float(is_balanced)
        )

        return min(score, 1.0)

    async def _elevate_layer(
        self,
        content: str,
        current_layer: SAPELayer,
        graph: GraphOfThoughts,
        parent_ids: Set[str],
    ) -> Tuple[str, ThoughtNode]:
        """Elevate content to next SAPE layer."""
        snr = self._compute_snr(content)
        ihsan = self._compute_ihsan(content)

        # Create node for current content
        node = graph.add_node(
            content=content,
            node_type=ThoughtNodeType.ANALYSIS,
            layer=current_layer,
            parent_ids=parent_ids,
            snr_score=snr,
            ihsan_score=ihsan,
            confidence=min(snr, ihsan),
        )

        # Check if SNR meets target
        target = self.snr_targets.get(current_layer.value, 0.85)
        if snr < target:
            # Backtrack and refine
            graph.backtrack(node.id, f"SNR {snr:.3f} < target {target:.3f}")

            # Try to improve (if LLM available)
            if self.llm_fn:
                refined = self.llm_fn(
                    f"Improve the clarity and signal of this text:\n{content[:500]}"
                )
                snr = self._compute_snr(refined)
                ihsan = self._compute_ihsan(refined)

                refined_node = graph.add_node(
                    content=refined,
                    node_type=ThoughtNodeType.REFINEMENT,
                    layer=current_layer,
                    parent_ids={node.id},
                    snr_score=snr,
                    ihsan_score=ihsan,
                    confidence=min(snr, ihsan),
                )

                return refined, refined_node

        return content, node

    async def optimize(
        self,
        input_content: str,
        target_layer: SAPELayer = SAPELayer.WISDOM,
    ) -> SAPEResult:
        """
        Run SAPE optimization on content.

        Elevates content through abstraction layers, maximizing SNR.
        """
        import time

        start = time.time()

        graph = GraphOfThoughts(snr_threshold=self.snr_targets["data"])
        layers_traversed = []
        snr_progression = {}

        # Start with DATA layer
        current_content = input_content
        parent_ids: Set[str] = set()

        # Progress through layers
        layer_order = [
            SAPELayer.DATA,
            SAPELayer.INFORMATION,
            SAPELayer.KNOWLEDGE,
            SAPELayer.WISDOM,
        ]
        target_idx = layer_order.index(target_layer)

        for i, layer in enumerate(layer_order[: target_idx + 1]):
            layers_traversed.append(layer)

            current_content, node = await self._elevate_layer(
                current_content,
                layer,
                graph,
                parent_ids,
            )

            parent_ids = {node.id}
            snr_progression[layer.value] = node.snr_score

            # If we're advancing layers, synthesize
            if i < target_idx and self.llm_fn:
                # Get all nodes at current layer
                layer_nodes = graph.get_layer_nodes(layer)
                if len(layer_nodes) > 1:
                    node_ids = {n.id for n in layer_nodes}
                    next_layer = layer_order[i + 1]

                    synthesis_prompt = f"""Synthesize these observations into a higher-level understanding:
{current_content[:1000]}

Create a more abstract, integrated perspective."""

                    synthesis_content = self.llm_fn(synthesis_prompt)
                    synthesis_node = graph.synthesize(
                        node_ids,
                        synthesis_content,
                        next_layer,
                    )
                    if synthesis_node:
                        current_content = synthesis_content
                        parent_ids = {synthesis_node.id}

        # Get best path and final output
        best_path = graph.get_best_path()
        output_content = best_path[-1].content if best_path else current_content
        final_ihsan = self._compute_ihsan(output_content)

        duration = (time.time() - start) * 1000

        return SAPEResult(
            input_content=input_content,
            output_content=output_content,
            layers_traversed=layers_traversed,
            snr_progression=snr_progression,
            ihsan_score=final_ihsan,
            thought_graph_size=len(graph._nodes),
            backtrack_count=graph._backtrack_count,
            duration_ms=duration,
        )

    async def analyze_unconventional_patterns(
        self,
        content: str,
    ) -> List[Dict[str, Any]]:
        """
        Discover unconventional patterns in content.

        Uses SAPE elevation to find non-obvious connections.
        """
        patterns = []

        # Optimize to knowledge layer
        result = await self.optimize(content, SAPELayer.KNOWLEDGE)

        # Extract patterns from SNR progression
        for layer, snr in result.snr_progression.items():
            if snr >= self.snr_targets.get(layer, 0.85):
                patterns.append(
                    {
                        "layer": layer,
                        "snr": snr,
                        "pattern_type": "snr_threshold_met",
                        "description": f"Content meets {layer} layer SNR target",
                    }
                )

        # Check for backtrack patterns (indicate complexity)
        if result.backtrack_count > 0:
            patterns.append(
                {
                    "layer": "meta",
                    "pattern_type": "complexity_indicator",
                    "backtrack_count": result.backtrack_count,
                    "description": "Content required reasoning backtracking",
                }
            )

        return patterns

    def get_layer_info(self) -> Dict[str, Any]:
        """Get SAPE layer information."""
        return {
            "layers": [l.value for l in SAPELayer],
            "snr_targets": self.snr_targets,
            "ihsan_threshold": self.ihsan_threshold,
            "progression": "DATA → INFORMATION → KNOWLEDGE → WISDOM",
        }
