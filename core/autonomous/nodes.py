"""
Reasoning Nodes — Graph-of-Thoughts Architecture

Implements Besta et al.'s Graph-of-Thoughts with:
- Non-linear reasoning paths
- Backtracking on low-SNR branches
- Multi-path synthesis
- Constitutional validation at every node

Standing on Giants: Besta (GoT) + Shannon (SNR) + Anthropic (Constitutional)
"""

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from core.autonomous import CONSTITUTIONAL_CONSTRAINTS, SNR_THRESHOLDS

logger = logging.getLogger(__name__)


class NodeType(str, Enum):
    """Types of reasoning nodes in the graph."""

    OBSERVATION = "observation"  # Raw input processing
    ORIENTATION = "orientation"  # Context establishment
    ANALYSIS = "analysis"  # Pattern extraction
    HYPOTHESIS = "hypothesis"  # Conjecture formation
    SYNTHESIS = "synthesis"  # Integration of multiple paths
    CONCLUSION = "conclusion"  # Final inference
    BACKTRACK = "backtrack"  # Reasoning reversal
    REFINEMENT = "refinement"  # Iterative improvement
    META = "meta"  # Meta-cognitive reflection


class NodeState(str, Enum):
    """State of a reasoning node."""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETED = "completed"
    PRUNED = "pruned"
    BACKTRACKED = "backtracked"


@dataclass
class ReasoningNode:
    """
    A node in the reasoning graph.

    Each node represents a discrete reasoning step with:
    - Content: The thought/inference at this step
    - SNR score: Signal quality (Shannon)
    - Ihsān score: Excellence quality (Anthropic)
    - Provenance: Lineage tracking (Giants Protocol)
    """

    id: str = field(default_factory=lambda: f"node_{uuid.uuid4().hex[:12]}")
    content: str = ""
    node_type: NodeType = NodeType.OBSERVATION
    state: NodeState = NodeState.PENDING

    # Quality metrics
    snr_score: float = 0.0
    ihsan_score: float = 0.0
    confidence: float = 0.0

    # Graph structure
    parents: Set[str] = field(default_factory=set)
    children: Set[str] = field(default_factory=set)
    depth: int = 0

    # Provenance
    technique_used: str = ""
    giant_invoked: str = ""

    # Metadata
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def quality_score(self) -> float:
        """Combined quality score (geometric mean of SNR and Ihsān)."""
        import math

        if self.snr_score <= 0 or self.ihsan_score <= 0:
            return 0.0
        return math.sqrt(self.snr_score * self.ihsan_score)

    @property
    def is_valid(self) -> bool:
        """Check if node meets constitutional thresholds."""
        min_snr = SNR_THRESHOLDS.get(self.node_type.value, 0.85)
        min_ihsan = CONSTITUTIONAL_CONSTRAINTS["ihsan_threshold"]
        return (
            self.snr_score >= min_snr
            and self.ihsan_score >= min_ihsan
            and self.state not in (NodeState.PRUNED, NodeState.BACKTRACKED)
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "content": self.content[:200] if len(self.content) > 200 else self.content,
            "node_type": self.node_type.value,
            "state": self.state.value,
            "snr_score": self.snr_score,
            "ihsan_score": self.ihsan_score,
            "confidence": self.confidence,
            "quality_score": self.quality_score,
            "is_valid": self.is_valid,
            "depth": self.depth,
            "parents": list(self.parents),
            "children": list(self.children),
            "technique_used": self.technique_used,
            "giant_invoked": self.giant_invoked,
        }


@dataclass
class ReasoningPath:
    """A complete path through the reasoning graph."""

    nodes: List[str]
    total_snr: float = 0.0
    total_ihsan: float = 0.0
    backtrack_count: int = 0
    depth: int = 0

    @property
    def average_quality(self) -> float:
        """Average quality score across path."""
        import math

        if self.total_snr <= 0 or self.total_ihsan <= 0:
            return 0.0
        return math.sqrt(self.total_snr * self.total_ihsan)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": self.nodes,
            "total_snr": self.total_snr,
            "total_ihsan": self.total_ihsan,
            "average_quality": self.average_quality,
            "backtrack_count": self.backtrack_count,
            "depth": self.depth,
        }


class ReasoningGraph:
    """
    Graph-of-Thoughts implementation.

    Manages the reasoning graph with:
    - Node creation and linking
    - Path finding (best-first, DFS, BFS)
    - Backtracking on quality degradation
    - Synthesis from multiple paths
    - Pruning of low-quality branches
    """

    def __init__(
        self,
        snr_calculator: Optional[Callable[[str], float]] = None,
        ihsan_calculator: Optional[Callable[[str], float]] = None,
    ):
        self._nodes: Dict[str, ReasoningNode] = {}
        self._root_ids: Set[str] = set()
        self._backtrack_count = 0
        self._synthesis_count = 0

        # Quality calculators
        self._snr_calc = snr_calculator or self._default_snr
        self._ihsan_calc = ihsan_calculator or self._default_ihsan

        # Thresholds
        self._snr_thresholds = SNR_THRESHOLDS
        self._constitutional = CONSTITUTIONAL_CONSTRAINTS

    def _default_snr(self, text: str) -> float:
        """Default SNR calculation (Shannon-inspired)."""
        if not text.strip():
            return 0.0

        words = text.split()
        if len(words) < 3:
            return 0.5

        unique_ratio = len(set(words)) / len(words)
        avg_word_len = sum(len(w) for w in words) / len(words)
        structure = min(text.count(".") / max(len(words) / 10, 1), 1.0)

        return 0.4 * unique_ratio + 0.3 * min(avg_word_len / 8, 1.0) + 0.3 * structure

    def _default_ihsan(self, text: str) -> float:
        """Default Ihsān calculation."""
        if not text.strip():
            return 0.0

        words = text.split()

        has_structure = len(words) > 5
        has_clarity = len(set(words)) / max(len(words), 1) > 0.5
        is_balanced = 10 <= len(words) <= 500

        return (
            0.4 * float(has_structure)
            + 0.3 * float(has_clarity)
            + 0.3 * float(is_balanced)
        )

    # =========================================================================
    # NODE MANAGEMENT
    # =========================================================================

    def add_node(
        self,
        content: str,
        node_type: NodeType = NodeType.OBSERVATION,
        parent_ids: Optional[Set[str]] = None,
        technique: str = "",
        giant: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ReasoningNode:
        """Add a new reasoning node to the graph."""
        # Calculate quality scores
        snr = self._snr_calc(content)
        ihsan = self._ihsan_calc(content)

        # Determine depth
        depth = 0
        if parent_ids:
            depths = [
                self._nodes[pid].depth for pid in parent_ids if pid in self._nodes
            ]
            depth = max(depths) + 1 if depths else 0

        node = ReasoningNode(
            content=content,
            node_type=node_type,
            state=NodeState.ACTIVE,
            snr_score=snr,
            ihsan_score=ihsan,
            confidence=min(snr, ihsan),
            parents=parent_ids or set(),
            depth=depth,
            technique_used=technique,
            giant_invoked=giant,
            metadata=metadata or {},
        )

        self._nodes[node.id] = node

        # Update graph structure
        if not parent_ids:
            self._root_ids.add(node.id)
        else:
            for pid in parent_ids:
                if pid in self._nodes:
                    self._nodes[pid].children.add(node.id)

        # Check for immediate quality issues
        threshold = self._snr_thresholds.get(node_type.value, 0.85)
        if snr < threshold * 0.8:  # Significantly below threshold
            logger.warning(
                f"Low SNR node created: {node.id} ({snr:.3f} < {threshold * 0.8:.3f})"
            )

        return node

    def get_node(self, node_id: str) -> Optional[ReasoningNode]:
        """Get a node by ID."""
        return self._nodes.get(node_id)

    def update_node(
        self,
        node_id: str,
        content: Optional[str] = None,
        state: Optional[NodeState] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[ReasoningNode]:
        """Update a node's content or state."""
        node = self._nodes.get(node_id)
        if not node:
            return None

        if content is not None:
            node.content = content
            node.snr_score = self._snr_calc(content)
            node.ihsan_score = self._ihsan_calc(content)
            node.confidence = min(node.snr_score, node.ihsan_score)

        if state is not None:
            node.state = state

        if metadata is not None:
            node.metadata.update(metadata)

        return node

    # =========================================================================
    # BACKTRACKING (Besta)
    # =========================================================================

    def backtrack(
        self,
        node_id: str,
        reason: str = "",
    ) -> Optional[ReasoningNode]:
        """
        Backtrack from a node.

        Creates a backtrack node and marks the original as backtracked.
        Returns the backtrack node.
        """
        if node_id not in self._nodes:
            return None

        node = self._nodes[node_id]

        # Check backtrack depth limit
        if self._backtrack_count >= self._constitutional["max_backtrack_depth"]:
            logger.warning(f"Max backtrack depth reached: {self._backtrack_count}")
            return None

        self._backtrack_count += 1

        # Mark original node
        node.state = NodeState.BACKTRACKED

        # Create backtrack node
        backtrack_content = (
            f"BACKTRACK: {reason}. Reconsidering: {node.content[:50]}..."
        )
        backtrack_node = self.add_node(
            content=backtrack_content,
            node_type=NodeType.BACKTRACK,
            parent_ids={node_id},
            technique="backtrack",
            giant="besta",
            metadata={"original_node": node_id, "reason": reason},
        )

        return backtrack_node

    def should_backtrack(self, node_id: str) -> Tuple[bool, str]:
        """Check if a node should trigger backtracking."""
        node = self._nodes.get(node_id)
        if not node:
            return False, ""

        threshold = self._snr_thresholds.get(node.node_type.value, 0.85)

        if node.snr_score < threshold:
            return True, f"SNR {node.snr_score:.3f} < threshold {threshold:.3f}"

        if node.ihsan_score < self._constitutional["ihsan_threshold"]:
            return (
                True,
                f"Ihsān {node.ihsan_score:.3f} < threshold {self._constitutional['ihsan_threshold']:.3f}",
            )

        return False, ""

    # =========================================================================
    # SYNTHESIS (Besta)
    # =========================================================================

    def synthesize(
        self,
        node_ids: Set[str],
        synthesis_content: str,
        target_type: NodeType = NodeType.SYNTHESIS,
    ) -> Optional[ReasoningNode]:
        """
        Synthesize multiple nodes into a higher-level understanding.

        This is the key operation for combining reasoning paths.
        """
        valid_nodes = [self._nodes[nid] for nid in node_ids if nid in self._nodes]

        if not valid_nodes:
            return None

        self._synthesis_count += 1

        # Aggregate quality scores
        avg_snr = sum(n.snr_score for n in valid_nodes) / len(valid_nodes)
        avg_ihsan = sum(n.ihsan_score for n in valid_nodes) / len(valid_nodes)

        # Synthesis bonus (integration adds value)
        synthesis_bonus = min(len(valid_nodes) * 0.02, 0.1)

        synthesis_node = self.add_node(
            content=synthesis_content,
            node_type=target_type,
            parent_ids=node_ids,
            technique="synthesis",
            giant="besta",
            metadata={
                "source_nodes": list(node_ids),
                "synthesis_count": self._synthesis_count,
            },
        )

        # Apply synthesis bonus
        synthesis_node.snr_score = min(avg_snr * (1 + synthesis_bonus), 1.0)
        synthesis_node.ihsan_score = min(avg_ihsan * (1 + synthesis_bonus), 1.0)
        synthesis_node.confidence = min(
            synthesis_node.snr_score, synthesis_node.ihsan_score
        )

        return synthesis_node

    # =========================================================================
    # PATH FINDING
    # =========================================================================

    def find_best_path(self) -> ReasoningPath:
        """Find the highest-quality path through the graph."""
        if not self._root_ids:
            return ReasoningPath(nodes=[])

        best_path = ReasoningPath(nodes=[])
        best_score = 0.0

        for root_id in self._root_ids:
            path = self._dfs_best_path(root_id, [], 0.0, 0.0, 0)
            score = path.average_quality
            if score > best_score:
                best_path = path
                best_score = score

        return best_path

    def _dfs_best_path(
        self,
        node_id: str,
        current_nodes: List[str],
        current_snr: float,
        current_ihsan: float,
        backtrack_count: int,
    ) -> ReasoningPath:
        """DFS to find best path."""
        node = self._nodes.get(node_id)
        if not node or node.state in (NodeState.PRUNED, NodeState.BACKTRACKED):
            return ReasoningPath(
                nodes=current_nodes,
                total_snr=current_snr / max(len(current_nodes), 1),
                total_ihsan=current_ihsan / max(len(current_nodes), 1),
                backtrack_count=backtrack_count,
                depth=len(current_nodes),
            )

        new_nodes = current_nodes + [node_id]
        new_snr = current_snr + node.snr_score
        new_ihsan = current_ihsan + node.ihsan_score
        new_backtrack = backtrack_count + (
            1 if node.node_type == NodeType.BACKTRACK else 0
        )

        if not node.children:
            return ReasoningPath(
                nodes=new_nodes,
                total_snr=new_snr / len(new_nodes),
                total_ihsan=new_ihsan / len(new_nodes),
                backtrack_count=new_backtrack,
                depth=len(new_nodes),
            )

        best_path = ReasoningPath(
            nodes=new_nodes,
            total_snr=new_snr / len(new_nodes),
            total_ihsan=new_ihsan / len(new_nodes),
            backtrack_count=new_backtrack,
            depth=len(new_nodes),
        )
        best_score = best_path.average_quality

        for child_id in node.children:
            child_path = self._dfs_best_path(
                child_id, new_nodes, new_snr, new_ihsan, new_backtrack
            )
            if child_path.average_quality > best_score:
                best_path = child_path
                best_score = child_path.average_quality

        return best_path

    def find_all_paths(self, max_paths: int = 10) -> List[ReasoningPath]:
        """Find all paths through the graph (up to max_paths)."""
        paths = []

        for root_id in self._root_ids:
            self._collect_paths(root_id, [], 0.0, 0.0, 0, paths, max_paths)
            if len(paths) >= max_paths:
                break

        return sorted(paths, key=lambda p: p.average_quality, reverse=True)

    def _collect_paths(
        self,
        node_id: str,
        current_nodes: List[str],
        current_snr: float,
        current_ihsan: float,
        backtrack_count: int,
        paths: List[ReasoningPath],
        max_paths: int,
    ) -> None:
        """Collect all paths recursively."""
        if len(paths) >= max_paths:
            return

        node = self._nodes.get(node_id)
        if not node or node.state in (NodeState.PRUNED, NodeState.BACKTRACKED):
            return

        new_nodes = current_nodes + [node_id]
        new_snr = current_snr + node.snr_score
        new_ihsan = current_ihsan + node.ihsan_score
        new_backtrack = backtrack_count + (
            1 if node.node_type == NodeType.BACKTRACK else 0
        )

        if not node.children:
            paths.append(
                ReasoningPath(
                    nodes=new_nodes,
                    total_snr=new_snr / len(new_nodes),
                    total_ihsan=new_ihsan / len(new_nodes),
                    backtrack_count=new_backtrack,
                    depth=len(new_nodes),
                )
            )
            return

        for child_id in node.children:
            self._collect_paths(
                child_id, new_nodes, new_snr, new_ihsan, new_backtrack, paths, max_paths
            )

    # =========================================================================
    # PRUNING
    # =========================================================================

    def prune_low_quality(self, threshold: float = 0.7) -> int:
        """Prune nodes below quality threshold."""
        pruned = 0

        for node in self._nodes.values():
            if node.state == NodeState.ACTIVE and node.quality_score < threshold:
                node.state = NodeState.PRUNED
                pruned += 1

        return pruned

    # =========================================================================
    # STATISTICS
    # =========================================================================

    def get_stats(self) -> Dict[str, Any]:
        """Get graph statistics."""
        nodes = list(self._nodes.values())
        active_nodes = [n for n in nodes if n.state == NodeState.ACTIVE]

        by_type = {}
        by_state = {}
        for node in nodes:
            by_type[node.node_type.value] = by_type.get(node.node_type.value, 0) + 1
            by_state[node.state.value] = by_state.get(node.state.value, 0) + 1

        avg_snr = sum(n.snr_score for n in active_nodes) / max(len(active_nodes), 1)
        avg_ihsan = sum(n.ihsan_score for n in active_nodes) / max(len(active_nodes), 1)

        return {
            "total_nodes": len(nodes),
            "active_nodes": len(active_nodes),
            "root_nodes": len(self._root_ids),
            "backtrack_count": self._backtrack_count,
            "synthesis_count": self._synthesis_count,
            "by_type": by_type,
            "by_state": by_state,
            "avg_snr": avg_snr,
            "avg_ihsan": avg_ihsan,
            "avg_quality": (
                (avg_snr * avg_ihsan) ** 0.5 if avg_snr > 0 and avg_ihsan > 0 else 0
            ),
        }

    def get_layer_nodes(self, node_type: NodeType) -> List[ReasoningNode]:
        """Get all nodes of a specific type."""
        return [n for n in self._nodes.values() if n.node_type == node_type]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize graph to dictionary."""
        return {
            "nodes": {nid: n.to_dict() for nid, n in self._nodes.items()},
            "root_ids": list(self._root_ids),
            "stats": self.get_stats(),
        }
