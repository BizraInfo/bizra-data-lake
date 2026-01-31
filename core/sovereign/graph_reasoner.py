"""
Graph-of-Thoughts Reasoning Engine

Standing on Giants:
- Graph of Thoughts (Besta et al., 2024): "Solving elaborate problems with LLMs"
- Tree of Thoughts (Yao et al., 2023): Deliberate problem solving
- Chain of Thought (Wei et al., 2022): Step-by-step reasoning
- BIZRA ARTE Engine: Symbolic-neural bridge

"The Graph of Thoughts paradigm enables LLMs to pursue and combine
 multiple independent lines of reasoning, moving beyond sequential
 chains to rich, networked cognitive structures."

Architecture:
    ┌─────────────────────────────────────────────────────────────┐
    │                    THOUGHT GRAPH                            │
    │                                                             │
    │         [Hypothesis A]──────┐                               │
    │              │              │                               │
    │              ▼              ▼                               │
    │         [Evidence 1]   [Evidence 2]                         │
    │              │              │                               │
    │              └──────┬───────┘                               │
    │                     ▼                                       │
    │              [Synthesis]────────► [Conclusion]              │
    │                     │                   │                   │
    │                     ▼                   ▼                   │
    │              [Refinement]         [Validation]              │
    │                     │                   │                   │
    │                     └─────────┬─────────┘                   │
    │                               ▼                             │
    │                      [Final Answer]                         │
    │                         (SNR ≥ 0.95)                        │
    └─────────────────────────────────────────────────────────────┘

Reasoning Operations:
- GENERATE: Create new thought nodes
- AGGREGATE: Merge multiple thoughts into synthesis
- REFINE: Improve existing thoughts iteratively
- VALIDATE: Check thoughts against Ihsān constraints
- PRUNE: Remove low-SNR branches
- BACKTRACK: Return to promising unexplored paths
"""

import uuid
import time
import math
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple, Any, Callable
from enum import Enum
from collections import defaultdict
import heapq

logger = logging.getLogger(__name__)


class ThoughtType(Enum):
    """Types of thought nodes in the graph."""
    HYPOTHESIS = "hypothesis"      # Initial conjectures
    EVIDENCE = "evidence"          # Supporting/refuting data
    REASONING = "reasoning"        # Logical deduction steps
    SYNTHESIS = "synthesis"        # Merged conclusions
    REFINEMENT = "refinement"      # Improved versions
    VALIDATION = "validation"      # Quality checks
    CONCLUSION = "conclusion"      # Final answers
    QUESTION = "question"          # Sub-questions to explore
    COUNTERPOINT = "counterpoint"  # Alternative perspectives


class EdgeType(Enum):
    """Types of edges connecting thoughts."""
    SUPPORTS = "supports"          # Evidence supports hypothesis
    REFUTES = "refutes"            # Evidence contradicts
    DERIVES = "derives"            # Logical derivation
    SYNTHESIZES = "synthesizes"    # Aggregation relationship
    REFINES = "refines"            # Improvement relationship
    QUESTIONS = "questions"        # Raises question
    VALIDATES = "validates"        # Quality check relationship


class ReasoningStrategy(Enum):
    """High-level reasoning strategies."""
    BREADTH_FIRST = "breadth_first"    # Explore widely first
    DEPTH_FIRST = "depth_first"        # Explore deeply first
    BEST_FIRST = "best_first"          # Follow highest SNR paths
    BEAM_SEARCH = "beam_search"        # Keep top-k paths
    MCTS = "mcts"                      # Monte Carlo Tree Search
    ADAPTIVE = "adaptive"              # Switch strategies dynamically


@dataclass
class ThoughtNode:
    """A node in the Graph of Thoughts."""
    id: str
    content: str
    thought_type: ThoughtType
    confidence: float = 0.5         # 0-1 confidence score
    snr_score: float = 0.5          # Signal-to-noise ratio
    depth: int = 0                  # Depth in reasoning tree
    created_at: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)

    # Ihsān dimensions
    correctness: float = 0.5
    groundedness: float = 0.5
    coherence: float = 0.5

    @property
    def ihsan_score(self) -> float:
        """Composite Ihsān score (geometric mean)."""
        scores = [
            max(self.correctness, 1e-10),
            max(self.groundedness, 1e-10),
            max(self.coherence, 1e-10),
            max(self.confidence, 1e-10),
        ]
        return math.exp(sum(math.log(s) for s in scores) / len(scores))

    @property
    def passes_ihsan(self) -> bool:
        """Check if thought passes Ihsān threshold."""
        return self.ihsan_score >= 0.75

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "content": self.content[:200] + "..." if len(self.content) > 200 else self.content,
            "type": self.thought_type.value,
            "confidence": self.confidence,
            "snr": self.snr_score,
            "ihsan": self.ihsan_score,
            "depth": self.depth,
        }


@dataclass
class ThoughtEdge:
    """An edge connecting thought nodes."""
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0             # Edge importance
    reasoning: str = ""             # Why this connection exists

    def to_dict(self) -> dict:
        return {
            "source": self.source_id,
            "target": self.target_id,
            "type": self.edge_type.value,
            "weight": self.weight,
        }


@dataclass
class ReasoningPath:
    """A path through the thought graph."""
    nodes: List[str]                # Node IDs in order
    total_snr: float = 0.0
    total_confidence: float = 0.0

    @property
    def length(self) -> int:
        return len(self.nodes)

    @property
    def average_snr(self) -> float:
        return self.total_snr / max(self.length, 1)


class GraphOfThoughts:
    """
    Graph-of-Thoughts Reasoning Engine.

    Implements networked reasoning where multiple thought branches
    can be explored, merged, refined, and validated in parallel.

    Key operations:
    1. GENERATE: Create new thought nodes from prompts/context
    2. AGGREGATE: Merge multiple thoughts into synthesis
    3. REFINE: Iteratively improve thought quality
    4. VALIDATE: Check against Ihsān constraints
    5. SCORE: Compute SNR for ranking
    6. PRUNE: Remove low-quality branches

    Usage:
        graph = GraphOfThoughts()
        root = graph.add_thought("What is the solution?", ThoughtType.QUESTION)

        # Generate hypotheses
        h1 = graph.generate("Hypothesis A", ThoughtType.HYPOTHESIS, parent=root)
        h2 = graph.generate("Hypothesis B", ThoughtType.HYPOTHESIS, parent=root)

        # Add evidence
        e1 = graph.generate("Evidence for A", ThoughtType.EVIDENCE, parent=h1)

        # Synthesize
        synth = graph.aggregate([h1, e1], "Combined conclusion")

        # Get best path
        best = graph.find_best_path(root.id)
    """

    def __init__(
        self,
        strategy: ReasoningStrategy = ReasoningStrategy.BEST_FIRST,
        max_depth: int = 10,
        beam_width: int = 5,
        snr_threshold: float = 0.5,
        ihsan_threshold: float = 0.75,
    ):
        self.strategy = strategy
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.snr_threshold = snr_threshold
        self.ihsan_threshold = ihsan_threshold

        # Graph structure
        self.nodes: Dict[str, ThoughtNode] = {}
        self.edges: List[ThoughtEdge] = []
        self.adjacency: Dict[str, List[str]] = defaultdict(list)  # node -> children
        self.reverse_adj: Dict[str, List[str]] = defaultdict(list)  # node -> parents

        # Root nodes (entry points)
        self.roots: List[str] = []

        # Statistics
        self.stats = {
            "nodes_created": 0,
            "nodes_pruned": 0,
            "edges_created": 0,
            "refinements": 0,
            "aggregations": 0,
        }

    def _generate_id(self) -> str:
        """Generate unique thought ID."""
        return f"thought_{uuid.uuid4().hex[:12]}"

    def add_thought(
        self,
        content: str,
        thought_type: ThoughtType,
        confidence: float = 0.5,
        metadata: Optional[Dict] = None,
        parent_id: Optional[str] = None,
        edge_type: EdgeType = EdgeType.DERIVES,
    ) -> ThoughtNode:
        """Add a new thought node to the graph."""
        node_id = self._generate_id()

        # Determine depth
        depth = 0
        if parent_id and parent_id in self.nodes:
            depth = self.nodes[parent_id].depth + 1

        node = ThoughtNode(
            id=node_id,
            content=content,
            thought_type=thought_type,
            confidence=confidence,
            depth=depth,
            metadata=metadata or {},
        )

        self.nodes[node_id] = node
        self.stats["nodes_created"] += 1

        # Track root nodes
        if parent_id is None:
            self.roots.append(node_id)
        else:
            # Create edge to parent
            self.add_edge(parent_id, node_id, edge_type)

        logger.debug(f"Added thought: {node_id} ({thought_type.value})")
        return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        edge_type: EdgeType,
        weight: float = 1.0,
        reasoning: str = "",
    ) -> ThoughtEdge:
        """Add an edge between thought nodes."""
        edge = ThoughtEdge(
            source_id=source_id,
            target_id=target_id,
            edge_type=edge_type,
            weight=weight,
            reasoning=reasoning,
        )

        self.edges.append(edge)
        self.adjacency[source_id].append(target_id)
        self.reverse_adj[target_id].append(source_id)
        self.stats["edges_created"] += 1

        return edge

    def generate(
        self,
        content: str,
        thought_type: ThoughtType,
        parent: Optional[ThoughtNode] = None,
        edge_type: EdgeType = EdgeType.DERIVES,
        **kwargs,
    ) -> ThoughtNode:
        """Generate a new thought from parent context."""
        parent_id = parent.id if parent else None
        return self.add_thought(
            content=content,
            thought_type=thought_type,
            parent_id=parent_id,
            edge_type=edge_type,
            **kwargs,
        )

    def aggregate(
        self,
        thoughts: List[ThoughtNode],
        synthesis_content: str,
        aggregation_type: ThoughtType = ThoughtType.SYNTHESIS,
    ) -> ThoughtNode:
        """Aggregate multiple thoughts into a synthesis."""
        # Compute aggregated confidence
        avg_confidence = sum(t.confidence for t in thoughts) / len(thoughts)
        avg_snr = sum(t.snr_score for t in thoughts) / len(thoughts)
        max_depth = max(t.depth for t in thoughts)

        synth = ThoughtNode(
            id=self._generate_id(),
            content=synthesis_content,
            thought_type=aggregation_type,
            confidence=avg_confidence * 1.1,  # Synthesis bonus
            snr_score=avg_snr * 1.05,
            depth=max_depth + 1,
        )

        self.nodes[synth.id] = synth
        self.stats["aggregations"] += 1

        # Connect all source thoughts to synthesis
        for thought in thoughts:
            self.add_edge(
                thought.id,
                synth.id,
                EdgeType.SYNTHESIZES,
                weight=thought.confidence,
            )

        logger.debug(f"Aggregated {len(thoughts)} thoughts into {synth.id}")
        return synth

    def refine(
        self,
        thought: ThoughtNode,
        refined_content: str,
        improvement_score: float = 0.1,
    ) -> ThoughtNode:
        """Refine an existing thought with improved content."""
        refined = ThoughtNode(
            id=self._generate_id(),
            content=refined_content,
            thought_type=ThoughtType.REFINEMENT,
            confidence=min(thought.confidence + improvement_score, 1.0),
            snr_score=min(thought.snr_score + improvement_score, 1.0),
            depth=thought.depth + 1,
        )

        self.nodes[refined.id] = refined
        self.stats["refinements"] += 1

        self.add_edge(thought.id, refined.id, EdgeType.REFINES)

        logger.debug(f"Refined {thought.id} -> {refined.id}")
        return refined

    def validate(
        self,
        thought: ThoughtNode,
        validator_fn: Optional[Callable[[str], Tuple[bool, float]]] = None,
    ) -> ThoughtNode:
        """Validate a thought and create validation node."""
        # Default validation: check Ihsān threshold
        if validator_fn:
            is_valid, score = validator_fn(thought.content)
        else:
            is_valid = thought.ihsan_score >= self.ihsan_threshold
            score = thought.ihsan_score

        validation = ThoughtNode(
            id=self._generate_id(),
            content=f"Validation: {'PASS' if is_valid else 'FAIL'} (score: {score:.3f})",
            thought_type=ThoughtType.VALIDATION,
            confidence=score,
            snr_score=score,
            depth=thought.depth + 1,
            metadata={"validated_id": thought.id, "passed": is_valid},
        )

        self.nodes[validation.id] = validation
        self.add_edge(thought.id, validation.id, EdgeType.VALIDATES)

        return validation

    def score_node(self, node: ThoughtNode) -> float:
        """Compute comprehensive SNR score for a node."""
        # Base score from node properties
        base_score = (node.confidence + node.snr_score + node.ihsan_score) / 3

        # Depth penalty (deeper = less certain)
        depth_factor = 1.0 / (1.0 + 0.1 * node.depth)

        # Support factor (more supporting edges = higher score)
        support_count = sum(
            1 for e in self.edges
            if e.target_id == node.id and e.edge_type == EdgeType.SUPPORTS
        )
        support_factor = 1.0 + 0.1 * support_count

        # Refutation penalty
        refute_count = sum(
            1 for e in self.edges
            if e.target_id == node.id and e.edge_type == EdgeType.REFUTES
        )
        refute_factor = 1.0 / (1.0 + 0.2 * refute_count)

        final_score = base_score * depth_factor * support_factor * refute_factor
        node.snr_score = min(final_score, 1.0)

        return node.snr_score

    def prune(self, threshold: Optional[float] = None) -> int:
        """Remove low-SNR nodes from the graph."""
        threshold = threshold or self.snr_threshold
        pruned = 0

        nodes_to_remove = [
            node_id for node_id, node in self.nodes.items()
            if node.snr_score < threshold and node.thought_type != ThoughtType.QUESTION
        ]

        for node_id in nodes_to_remove:
            # Remove from adjacency
            for child in self.adjacency.get(node_id, []):
                self.reverse_adj[child].remove(node_id)
            for parent in self.reverse_adj.get(node_id, []):
                self.adjacency[parent].remove(node_id)

            del self.nodes[node_id]
            del self.adjacency[node_id]
            del self.reverse_adj[node_id]
            pruned += 1

        # Remove orphaned edges
        self.edges = [
            e for e in self.edges
            if e.source_id in self.nodes and e.target_id in self.nodes
        ]

        self.stats["nodes_pruned"] += pruned
        logger.info(f"Pruned {pruned} low-SNR nodes")
        return pruned

    def find_best_path(
        self,
        start_id: Optional[str] = None,
        target_type: ThoughtType = ThoughtType.CONCLUSION,
    ) -> ReasoningPath:
        """Find the highest-SNR path through the graph."""
        if start_id is None:
            start_id = self.roots[0] if self.roots else None

        if not start_id or start_id not in self.nodes:
            return ReasoningPath(nodes=[], total_snr=0, total_confidence=0)

        # Best-first search using priority queue
        # Priority = negative SNR (for max-heap behavior)
        pq = [(-self.nodes[start_id].snr_score, [start_id])]
        visited = set()
        best_path = ReasoningPath(nodes=[start_id], total_snr=self.nodes[start_id].snr_score)

        while pq:
            neg_snr, path = heapq.heappop(pq)
            current_id = path[-1]

            if current_id in visited:
                continue
            visited.add(current_id)

            current_node = self.nodes[current_id]

            # Check if we've reached a conclusion
            if current_node.thought_type == target_type:
                total_snr = sum(self.nodes[n].snr_score for n in path)
                if total_snr > best_path.total_snr:
                    best_path = ReasoningPath(
                        nodes=path,
                        total_snr=total_snr,
                        total_confidence=sum(self.nodes[n].confidence for n in path),
                    )

            # Explore children
            for child_id in self.adjacency.get(current_id, []):
                if child_id not in visited and child_id in self.nodes:
                    child = self.nodes[child_id]
                    new_path = path + [child_id]
                    heapq.heappush(pq, (-child.snr_score, new_path))

        return best_path

    def get_conclusions(self, min_snr: float = 0.7) -> List[ThoughtNode]:
        """Get all conclusion nodes above SNR threshold."""
        return [
            node for node in self.nodes.values()
            if node.thought_type == ThoughtType.CONCLUSION
            and node.snr_score >= min_snr
        ]

    def get_frontier(self) -> List[ThoughtNode]:
        """Get leaf nodes (nodes with no children) - the current reasoning frontier."""
        all_parents = set()
        for children in self.adjacency.values():
            # Collect all nodes that are parents (have children)
            pass

        # Find nodes that have no outgoing edges (no children)
        parent_ids = set(self.adjacency.keys())
        leaf_ids = [
            node_id for node_id in self.nodes.keys()
            if node_id not in parent_ids or not self.adjacency[node_id]
        ]
        return [self.nodes[nid] for nid in leaf_ids if nid in self.nodes]

    def get_leaves(self) -> List[ThoughtNode]:
        """Alias for get_frontier."""
        return self.get_frontier()

    def clear(self):
        """Reset the graph to empty state."""
        self.nodes.clear()
        self.edges.clear()
        self.adjacency.clear()
        self.reverse_adj.clear()
        self.roots.clear()
        # Reset stats but keep structure
        self.stats = {
            "nodes_created": 0,
            "nodes_pruned": 0,
            "edges_created": 0,
            "refinements": 0,
            "aggregations": 0,
        }
        logger.debug("Graph cleared")

    @property
    def thoughts(self) -> Dict[str, ThoughtNode]:
        """Property alias for nodes dict."""
        return self.nodes

    def to_dict(self) -> dict:
        """Serialize graph for inspection."""
        return {
            "nodes": [n.to_dict() for n in self.nodes.values()],
            "edges": [e.to_dict() for e in self.edges],
            "roots": self.roots,
            "stats": self.stats,
            "config": {
                "strategy": self.strategy.value,
                "max_depth": self.max_depth,
                "snr_threshold": self.snr_threshold,
                "ihsan_threshold": self.ihsan_threshold,
            },
        }

    def visualize_ascii(self) -> str:
        """Generate ASCII visualization of the graph."""
        lines = ["Graph of Thoughts", "=" * 50]

        def render_node(node_id: str, indent: int = 0) -> List[str]:
            if node_id not in self.nodes:
                return []

            node = self.nodes[node_id]
            prefix = "  " * indent
            symbol = {
                ThoughtType.QUESTION: "?",
                ThoughtType.HYPOTHESIS: "H",
                ThoughtType.EVIDENCE: "E",
                ThoughtType.REASONING: "R",
                ThoughtType.SYNTHESIS: "S",
                ThoughtType.REFINEMENT: "↑",
                ThoughtType.VALIDATION: "✓",
                ThoughtType.CONCLUSION: "★",
                ThoughtType.COUNTERPOINT: "⚡",
            }.get(node.thought_type, "•")

            result = [
                f"{prefix}[{symbol}] {node.content[:60]}... (SNR: {node.snr_score:.2f})"
            ]

            for child_id in self.adjacency.get(node_id, [])[:3]:  # Limit children
                result.extend(render_node(child_id, indent + 1))

            return result

        for root_id in self.roots[:3]:  # Limit roots
            lines.extend(render_node(root_id))

        lines.append("=" * 50)
        lines.append(f"Nodes: {len(self.nodes)} | Edges: {len(self.edges)}")

        return "\n".join(lines)
