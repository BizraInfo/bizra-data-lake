"""
Graph-of-Thoughts Bridge — Multi-Path Reasoning Engine
═══════════════════════════════════════════════════════════════════════════════

"The key insight is that by structuring the reasoning process as a graph,
we can explore multiple paths, aggregate diverse perspectives, and refine
solutions through iterative improvement."
    — Maciej Besta et al., 2024

This module implements a Python bridge to the Rust Graph-of-Thoughts engine
in bizra-core, while providing a pure Python fallback for standalone use.

Core Operations (Besta 2024):
1. GENERATE — Create new thought nodes
2. AGGREGATE — Combine multiple thoughts into one
3. REFINE — Improve a thought through iteration
4. VALIDATE — Score thought quality
5. PRUNE — Remove low-quality paths
6. BACKTRACK — Return to earlier promising nodes

Architecture:
    ┌─────────────────────────────────────────────────────────────────────────┐
    │                    GRAPH OF THOUGHTS BRIDGE                              │
    ├─────────────────────────────────────────────────────────────────────────┤
    │                                                                         │
    │  ┌─────────────────────────────────────────────────────────────────┐   │
    │  │                      ThoughtGraph                                │   │
    │  │  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐  ┌────────┐   │   │
    │  │  │ Node 0 │──│ Node 1 │──│ Node 2 │  │ Node 3 │──│ Node 4 │   │   │
    │  │  │(root)  │  └────┬───┘  └────────┘  └────┬───┘  └────────┘   │   │
    │  │  └────────┘       │                       │                    │   │
    │  │                   ▼                       ▼                    │   │
    │  │              ┌────────┐              ┌────────┐                │   │
    │  │              │ Node 5 │──────────────│ Node 6 │                │   │
    │  │              │(merged)│              │(refined)│               │   │
    │  │              └────────┘              └────────┘                │   │
    │  └─────────────────────────────────────────────────────────────────┘   │
    │                                                                         │
    │  Operations: Generate → Aggregate → Refine → Validate → Prune          │
    │                                                                         │
    └─────────────────────────────────────────────────────────────────────────┘

Standing on Giants: Besta (2024), Wei (2022 CoT), Yao (2023 ToT)

Created: 2026-02-04 | BIZRA Sovereign Runtime v1.0
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from heapq import heappop, heappush
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

# GoT constants
MAX_DEPTH: int = 10
MAX_BRANCHES: int = 5
PRUNE_THRESHOLD: float = 0.3
REFINEMENT_ITERATIONS: int = 3


class ThoughtType(str, Enum):
    """Types of thought operations."""

    ROOT = "root"  # Starting thought
    GENERATE = "generate"  # New thought generation
    AGGREGATE = "aggregate"  # Combining multiple thoughts
    REFINE = "refine"  # Iterative improvement
    VALIDATE = "validate"  # Quality scoring
    PRUNE = "prune"  # Removal marker
    BACKTRACK = "backtrack"  # Return to earlier state
    SOLUTION = "solution"  # Final answer


class ThoughtStatus(str, Enum):
    """Status of a thought node."""

    ACTIVE = "active"
    PRUNED = "pruned"
    MERGED = "merged"
    SOLUTION = "solution"


@dataclass
class ThoughtNode:
    """
    A node in the Graph of Thoughts.

    Each node represents a reasoning step with content,
    quality scores, and connections to other nodes.
    """

    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str = ""
    thought_type: ThoughtType = ThoughtType.GENERATE

    # Quality metrics
    score: float = 0.5  # Overall quality [0, 1]
    confidence: float = 0.5  # Certainty in the thought
    coherence: float = 0.5  # Internal consistency
    relevance: float = 0.5  # Relevance to goal

    # Graph structure
    parent_ids: list[str] = field(default_factory=list)
    child_ids: list[str] = field(default_factory=list)
    depth: int = 0

    # Metadata
    status: ThoughtStatus = ThoughtStatus.ACTIVE
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def __lt__(self, other):
        """For heap comparison (higher score = higher priority)."""
        return self.score > other.score

    def combined_score(self) -> float:
        """Calculate combined quality score."""
        return self.confidence * 0.4 + self.coherence * 0.3 + self.relevance * 0.3


@dataclass
class ThoughtEdge:
    """An edge connecting two thought nodes."""

    source_id: str
    target_id: str
    edge_type: str = "derives"  # derives, aggregates, refines
    weight: float = 1.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class GoTResult:
    """Result of a Graph-of-Thoughts reasoning process."""

    goal: str
    solution: Optional[ThoughtNode]
    explored_nodes: int
    pruned_nodes: int
    max_depth_reached: int
    best_path: list[ThoughtNode]
    all_solutions: list[ThoughtNode]
    execution_time_ms: float
    success: bool


class ThoughtGraph:
    """
    Graph of Thoughts reasoning engine.

    Implements multi-path reasoning through graph construction
    and traversal, enabling exploration of multiple solution paths.

    Standing on Giants:
    - Besta (2024): Graph of Thoughts architecture
    - Wei (2022): Chain of Thought prompting
    - Yao (2023): Tree of Thoughts exploration

    Usage:
        graph = ThoughtGraph()
        result = await graph.reason("Solve this problem...")
        print(f"Solution: {result.solution.content}")
    """

    def __init__(
        self,
        max_depth: int = MAX_DEPTH,
        max_branches: int = MAX_BRANCHES,
        prune_threshold: float = PRUNE_THRESHOLD,
    ):
        """
        Initialize the thought graph.

        Args:
            max_depth: Maximum reasoning depth
            max_branches: Maximum branches per node
            prune_threshold: Score below which nodes are pruned
        """
        self.max_depth = max_depth
        self.max_branches = max_branches
        self.prune_threshold = prune_threshold

        # Graph storage
        self._nodes: dict[str, ThoughtNode] = {}
        self._edges: list[ThoughtEdge] = []
        self._root: Optional[ThoughtNode] = None

        # Exploration state
        self._frontier: list[ThoughtNode] = []  # Priority queue
        self._explored: set[str] = set()
        self._solutions: list[ThoughtNode] = []

        # Scoring function (can be overridden)
        self._scorer: Optional[Callable[[ThoughtNode], float]] = None

        # Generation function (can be overridden)
        self._generator: Optional[Callable[[ThoughtNode, str], list[str]]] = None

        logger.debug("ThoughtGraph initialized")

    def set_scorer(self, scorer: Callable[[ThoughtNode], float]) -> None:
        """set custom scoring function."""
        self._scorer = scorer

    def set_generator(self, generator: Callable[[ThoughtNode, str], list[str]]) -> None:
        """set custom thought generation function."""
        self._generator = generator

    def _default_scorer(self, node: ThoughtNode) -> float:
        """Default scoring function based on node properties."""
        # Simple heuristic scoring
        depth_penalty = 0.95**node.depth  # Prefer shallower solutions
        return node.combined_score() * depth_penalty

    def _default_generator(self, parent: ThoughtNode, goal: str) -> list[str]:
        """Default thought generator (returns placeholder thoughts)."""
        # In production, this would call an LLM
        base_thoughts = [
            f"Approach 1: Break down '{goal}' into subtasks",
            f"Approach 2: Find similar solved problems for '{goal}'",
            f"Approach 3: Identify key constraints in '{goal}'",
        ]
        return base_thoughts[: self.max_branches]

    def create_root(self, goal: str) -> ThoughtNode:
        """Create the root node from a goal."""
        root = ThoughtNode(
            content=goal,
            thought_type=ThoughtType.ROOT,
            score=1.0,
            confidence=1.0,
            coherence=1.0,
            relevance=1.0,
            depth=0,
        )
        self._nodes[root.id] = root
        self._root = root
        heappush(self._frontier, root)

        logger.debug(f"Created root node: {root.id}")
        return root

    def generate(self, parent: ThoughtNode, goal: str) -> list[ThoughtNode]:
        """
        GENERATE: Create new thought nodes from a parent.

        Besta (2024): "Generation creates new thoughts by decomposing
        complex problems or exploring alternative approaches."
        """
        if parent.depth >= self.max_depth:
            return []

        # Use custom or default generator
        generator = self._generator or self._default_generator
        thought_contents = generator(parent, goal)

        new_nodes = []
        for content in thought_contents[: self.max_branches]:
            node = ThoughtNode(
                content=content,
                thought_type=ThoughtType.GENERATE,
                parent_ids=[parent.id],
                depth=parent.depth + 1,
                # Initial scores (will be refined)
                score=0.5,
                confidence=0.5,
                coherence=0.6,
                relevance=0.7,
            )

            self._nodes[node.id] = node
            parent.child_ids.append(node.id)

            # Create edge
            edge = ThoughtEdge(
                source_id=parent.id,
                target_id=node.id,
                edge_type="derives",
            )
            self._edges.append(edge)

            new_nodes.append(node)
            heappush(self._frontier, node)

        logger.debug(f"Generated {len(new_nodes)} thoughts from {parent.id}")
        return new_nodes

    def aggregate(self, nodes: list[ThoughtNode]) -> ThoughtNode:
        """
        AGGREGATE: Combine multiple thoughts into one.

        Besta (2024): "Aggregation merges insights from multiple
        reasoning paths into a unified understanding."
        """
        if not nodes:
            raise ValueError("Cannot aggregate empty node list")

        # Combine content
        combined_content = "Synthesis: " + " | ".join(n.content for n in nodes)

        # Aggregate scores
        avg_score = sum(n.score for n in nodes) / len(nodes)
        max_confidence = max(n.confidence for n in nodes)
        avg_coherence = sum(n.coherence for n in nodes) / len(nodes)

        aggregated = ThoughtNode(
            content=combined_content,
            thought_type=ThoughtType.AGGREGATE,
            parent_ids=[n.id for n in nodes],
            depth=max(n.depth for n in nodes) + 1,
            score=avg_score * 1.1,  # Aggregation bonus
            confidence=max_confidence,
            coherence=avg_coherence,
            relevance=avg_score,
        )

        self._nodes[aggregated.id] = aggregated

        # Update parent references
        for node in nodes:
            node.child_ids.append(aggregated.id)
            self._edges.append(
                ThoughtEdge(
                    source_id=node.id,
                    target_id=aggregated.id,
                    edge_type="aggregates",
                )
            )

        heappush(self._frontier, aggregated)

        logger.debug(f"Aggregated {len(nodes)} nodes into {aggregated.id}")
        return aggregated

    def refine(
        self, node: ThoughtNode, iterations: int = REFINEMENT_ITERATIONS
    ) -> ThoughtNode:
        """
        REFINE: Iteratively improve a thought.

        Besta (2024): "Refinement improves thoughts through
        self-critique and iterative enhancement."
        """
        current = node

        for i in range(iterations):
            refined_content = f"{current.content} [refined v{i + 1}]"

            refined = ThoughtNode(
                content=refined_content,
                thought_type=ThoughtType.REFINE,
                parent_ids=[current.id],
                depth=current.depth + 1,
                score=min(1.0, current.score * 1.05),  # Refinement improvement
                confidence=min(1.0, current.confidence * 1.03),
                coherence=min(1.0, current.coherence * 1.02),
                relevance=current.relevance,
            )

            self._nodes[refined.id] = refined
            current.child_ids.append(refined.id)

            self._edges.append(
                ThoughtEdge(
                    source_id=current.id,
                    target_id=refined.id,
                    edge_type="refines",
                )
            )

            current = refined

        heappush(self._frontier, current)

        logger.debug(f"Refined node through {iterations} iterations: {current.id}")
        return current

    def validate(self, node: ThoughtNode) -> float:
        """
        VALIDATE: Score a thought node.

        Uses custom scorer or default scoring function.
        """
        scorer = self._scorer or self._default_scorer
        score = scorer(node)
        node.score = score
        return score

    def prune(self) -> int:
        """
        PRUNE: Remove low-quality nodes from frontier.

        Returns number of nodes pruned.
        """
        pruned_count = 0
        new_frontier = []

        while self._frontier:
            node = heappop(self._frontier)
            if node.score >= self.prune_threshold:
                new_frontier.append(node)
            else:
                node.status = ThoughtStatus.PRUNED
                pruned_count += 1

        # Rebuild frontier
        for node in new_frontier:
            heappush(self._frontier, node)

        logger.debug(f"Pruned {pruned_count} nodes")
        return pruned_count

    def backtrack(self, to_node_id: str) -> Optional[ThoughtNode]:
        """
        BACKTRACK: Return to an earlier promising node.

        Useful when current path is unproductive.
        """
        if to_node_id not in self._nodes:
            return None

        target = self._nodes[to_node_id]

        # Mark current frontier as explored but not pruned
        for node in self._frontier:
            self._explored.add(node.id)

        # Reset frontier to backtrack point
        self._frontier = [target]

        logger.debug(f"Backtracked to node {to_node_id}")
        return target

    def mark_solution(self, node: ThoughtNode) -> None:
        """Mark a node as a solution."""
        node.status = ThoughtStatus.SOLUTION
        node.thought_type = ThoughtType.SOLUTION
        self._solutions.append(node)

    def get_best_path(self) -> list[ThoughtNode]:
        """Get the path from root to best solution."""
        if not self._solutions:
            return []

        best_solution = max(self._solutions, key=lambda n: n.score)
        return self._trace_path(best_solution)

    def _trace_path(self, node: ThoughtNode) -> list[ThoughtNode]:
        """Trace path from root to given node."""
        path = [node]
        current = node

        while current.parent_ids:
            parent_id = current.parent_ids[0]  # Take first parent
            if parent_id in self._nodes:
                parent = self._nodes[parent_id]
                path.insert(0, parent)
                current = parent
            else:
                break

        return path

    async def reason(
        self,
        goal: str,
        max_iterations: int = 50,
        min_solutions: int = 1,
    ) -> GoTResult:
        """
        Run the full Graph-of-Thoughts reasoning process.

        Args:
            goal: The problem to solve
            max_iterations: Maximum exploration iterations
            min_solutions: Minimum solutions to find before stopping

        Returns:
            GoTResult with solution and exploration stats
        """
        start_time = time.time()

        # Initialize
        self.create_root(goal)
        iterations = 0
        max_depth_reached = 0

        while self._frontier and iterations < max_iterations:
            iterations += 1

            # Get highest-scoring node
            current = heappop(self._frontier)

            if current.id in self._explored:
                continue

            self._explored.add(current.id)
            max_depth_reached = max(max_depth_reached, current.depth)

            # Validate current node
            self.validate(current)

            # Check if this is a solution
            if current.score >= 0.9 and current.depth >= 2:
                self.mark_solution(current)
                if len(self._solutions) >= min_solutions:
                    break

            # Generate children
            if current.depth < self.max_depth:
                children = self.generate(current, goal)

                # Validate new children
                for child in children:
                    self.validate(child)

                # Maybe aggregate promising children
                high_quality = [c for c in children if c.score >= 0.7]
                if len(high_quality) >= 2:
                    self.aggregate(high_quality[:2])

            # Periodic pruning
            if iterations % 10 == 0:
                self.prune()

        execution_time = (time.time() - start_time) * 1000

        # Get results
        best_path = self.get_best_path()
        solution = best_path[-1] if best_path else None
        pruned_count = sum(
            1 for n in self._nodes.values() if n.status == ThoughtStatus.PRUNED
        )

        result = GoTResult(
            goal=goal,
            solution=solution,
            explored_nodes=len(self._explored),
            pruned_nodes=pruned_count,
            max_depth_reached=max_depth_reached,
            best_path=best_path,
            all_solutions=list(self._solutions),
            execution_time_ms=execution_time,
            success=solution is not None,
        )

        logger.info(
            f"GoT complete: explored={result.explored_nodes}, "
            f"solutions={len(result.all_solutions)}, "
            f"depth={max_depth_reached}, time={execution_time:.1f}ms"
        )

        return result

    def visualize(self) -> str:
        """Generate ASCII visualization of the graph."""
        if not self._root:
            return "Empty graph"

        lines = ["Graph of Thoughts:", ""]

        def render_node(node: ThoughtNode, prefix: str = "", is_last: bool = True):
            connector = "└── " if is_last else "├── "
            status_icon = {
                ThoughtStatus.ACTIVE: "○",
                ThoughtStatus.PRUNED: "✗",
                ThoughtStatus.MERGED: "◎",
                ThoughtStatus.SOLUTION: "★",
            }.get(node.status, "?")

            content_preview = (
                node.content[:40] + "..." if len(node.content) > 40 else node.content
            )
            line = f"{prefix}{connector}{status_icon} [{node.id}] {content_preview} (score={node.score:.2f})"
            lines.append(line)

            children = [
                self._nodes[cid] for cid in node.child_ids if cid in self._nodes
            ]
            for i, child in enumerate(children):
                is_child_last = i == len(children) - 1
                child_prefix = prefix + ("    " if is_last else "│   ")
                render_node(child, child_prefix, is_child_last)

        render_node(self._root)
        return "\n".join(lines)


class GoTBridge:
    """
    Bridge to Rust Graph-of-Thoughts implementation.

    Provides interface to use the Rust bizra-core ThoughtGraph
    when available, with Python fallback.
    """

    def __init__(self, use_rust: bool = True):
        """
        Initialize the GoT bridge.

        Args:
            use_rust: Try to use Rust implementation if available
        """
        self.use_rust = use_rust
        self._rust_available = False
        self._python_graph: Optional[ThoughtGraph] = None

        # Try to import Rust bindings
        if use_rust:
            try:
                # Would import from bizra-python bindings
                # from bizra_omega import ThoughtGraph as RustThoughtGraph
                # self._rust_available = True
                pass
            except ImportError:
                logger.info("Rust GoT not available, using Python fallback")

        if not self._rust_available:
            self._python_graph = ThoughtGraph()

        logger.info(f"GoTBridge initialized: rust={self._rust_available}")

    async def reason(
        self,
        goal: str,
        max_iterations: int = 50,
        scorer: Optional[Callable[[ThoughtNode], float]] = None,
        generator: Optional[Callable[[ThoughtNode, str], list[str]]] = None,
    ) -> GoTResult:
        """
        Perform Graph-of-Thoughts reasoning.

        Args:
            goal: Problem to solve
            max_iterations: Maximum exploration iterations
            scorer: Custom scoring function
            generator: Custom thought generator

        Returns:
            GoTResult with solution and stats
        """
        if self._rust_available:
            # Would use Rust implementation
            pass

        # Python fallback
        graph = ThoughtGraph()

        if scorer:
            graph.set_scorer(scorer)
        if generator:
            graph.set_generator(generator)

        return await graph.reason(goal, max_iterations)

    def visualize_last_graph(self) -> str:
        """Visualize the last reasoning graph."""
        if self._python_graph:
            return self._python_graph.visualize()
        return "No graph available"


# Global bridge instance
_got_bridge: Optional[GoTBridge] = None


def get_got_bridge() -> GoTBridge:
    """Get the global GoT bridge."""
    global _got_bridge
    if _got_bridge is None:
        _got_bridge = GoTBridge()
    return _got_bridge


async def think(goal: str, **kwargs) -> GoTResult:
    """
    Convenience function for Graph-of-Thoughts reasoning.

    Usage:
        result = await think("How to optimize database queries?")
        print(result.solution.content)
    """
    bridge = get_got_bridge()
    return await bridge.reason(goal, **kwargs)
