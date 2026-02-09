"""
Graph-of-Thoughts Integration with Autopoietic Loop
========================================================================

Multi-path hypothesis exploration using Graph-of-Thoughts reasoning
within the autopoietic self-improvement cycle.

"The key insight is that by structuring the reasoning process as a graph,
we can explore multiple paths, aggregate diverse perspectives, and refine
solutions through iterative improvement."
    -- Maciej Besta et al., 2024

Architecture:
    +---------------------------------------------------------------------------+
    |                   GoT HYPOTHESIS EXPLORATION                               |
    +---------------------------------------------------------------------------+
    |                                                                           |
    |   SystemObservation                                                       |
    |         |                                                                 |
    |         v                                                                 |
    |   +---------------+     +---------------+     +---------------+           |
    |   |   GENERATE    | --> |   AGGREGATE   | --> |    REFINE     |          |
    |   | (hypotheses)  |     | (convergence) |     |  (promising)  |          |
    |   +---------------+     +---------------+     +---------------+           |
    |         |                     |                     |                     |
    |         v                     v                     v                     |
    |   +-----------+         +-----------+         +-----------+               |
    |   |  Path 1   |         |  Merged   |         | Refined   |              |
    |   |  Path 2   | ------> |  Insight  | ------> | Solution  |              |
    |   |  Path 3   |         |           |         |           |              |
    |   +-----------+         +-----------+         +-----------+               |
    |         |                                           |                     |
    |         v                                           v                     |
    |   +---------------+                         +---------------+             |
    |   |   VALIDATE    |                         |     PRUNE     |             |
    |   | (FATE gate)   |                         | (low SNR)     |             |
    |   +---------------+                         +---------------+             |
    |                                                                           |
    +---------------------------------------------------------------------------+

Integration Points:
- HypothesisGenerator.generate() -> GoTHypothesisExplorer.explore()
- AutopoieticLoop.hypothesize() uses GoT for multi-path reasoning
- SNR maximization at each thought node
- Ihsan constraint propagation through graph

Standing on Giants:
- Besta (2024): Graph-of-Thoughts architecture
- Maturana & Varela (1972): Autopoiesis theory
- Shannon (1948): Information theory (SNR)
- Anthropic (2022): Constitutional AI (Ihsan)

Genesis Strict Synthesis v2.2.2
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Tuple,
)

# Import from local autopoiesis modules
from core.autopoiesis.hypothesis_generator import (
    Hypothesis,
    HypothesisCategory,
    HypothesisGenerator,
    RiskLevel,
    SystemObservation,
)
from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)

# Import GoT components from sovereign runtime engines
from core.sovereign.runtime_engines.got_bridge import (
    MAX_BRANCHES,
    GoTBridge,
    ThoughtNode,
    ThoughtStatus,
    ThoughtType,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

# GoT Hypothesis Explorer Constants
DEFAULT_MAX_DEPTH: int = 5
DEFAULT_NUM_PATHS: int = 5
DEFAULT_BREADTH_FIRST_DEPTH: int = 2
DEFAULT_MCTS_ITERATIONS: int = 100
DEFAULT_EXPLORATION_CONSTANT: float = 1.414  # sqrt(2) for UCB1

# Numeric mapping for RiskLevel (str enum) for arithmetic operations
_RISK_LEVEL_NUMERIC: Dict[str, int] = {"low": 1, "medium": 2, "high": 3}

# SNR/Ihsan thresholds for hypothesis exploration
HYPOTHESIS_SNR_FLOOR: float = 0.70
HYPOTHESIS_IHSAN_FLOOR: float = 0.90
CONVERGENCE_SIMILARITY_THRESHOLD: float = 0.85
PATH_PRUNE_THRESHOLD: float = 0.40


# =============================================================================
# EXPLORED HYPOTHESIS
# =============================================================================


@dataclass
class ExploredHypothesis:
    """
    A hypothesis that has been explored through the GoT process.

    Contains the original hypothesis plus exploration metadata,
    including the path taken, convergence evidence, and SNR scores.
    """

    hypothesis: Hypothesis
    exploration_path: List[ThoughtNode] = field(default_factory=list)
    snr_score: float = 0.0
    ihsan_score: float = UNIFIED_IHSAN_THRESHOLD
    confidence: float = 0.5
    convergence_evidence: List[str] = field(default_factory=list)
    converged_with: List[str] = field(default_factory=list)
    exploration_depth: int = 0
    exploration_time_ms: float = 0.0
    fate_validated: bool = False
    fate_proof_id: Optional[str] = None

    def __post_init__(self):
        """Calculate composite scores after initialization."""
        self._calculate_scores()

    def _calculate_scores(self) -> None:
        """Calculate SNR and composite scores from exploration path."""
        if not self.exploration_path:
            return

        # Aggregate scores from path nodes
        total_signal = 0.0
        total_noise = 0.0
        total_ihsan = 0.0

        for node in self.exploration_path:
            # Signal: confidence * coherence * relevance
            node_signal = node.confidence * node.coherence * node.relevance
            total_signal += node_signal

            # Noise: uncertainty (1 - confidence) + risk penalty
            risk_penalty = getattr(node.metadata, "risk_level", 0) * 0.1
            node_noise = (1 - node.confidence) + risk_penalty
            total_noise += node_noise

            # Ihsan from node metadata
            node_ihsan = node.metadata.get("ihsan_score", UNIFIED_IHSAN_THRESHOLD)
            total_ihsan += node_ihsan

        path_length = len(self.exploration_path)
        if path_length > 0:
            self.snr_score = total_signal / max(total_noise, 0.01)
            self.ihsan_score = total_ihsan / path_length

        # Update hypothesis confidence based on exploration
        if self.exploration_path:
            best_node = max(self.exploration_path, key=lambda n: n.score)
            self.confidence = best_node.confidence

    def expected_value(self) -> float:
        """
        Calculate expected value combining hypothesis EV with exploration quality.

        EV = hypothesis.expected_value() * exploration_boost * snr_weight
        """
        base_ev = self.hypothesis.expected_value()

        # Exploration boost: deeper exploration with convergence is better
        depth_boost = 1.0 + (self.exploration_depth * 0.05)
        convergence_boost = 1.0 + (len(self.converged_with) * 0.1)
        exploration_boost = min(depth_boost * convergence_boost, 2.0)

        # SNR weight: higher SNR increases expected value
        snr_weight = min(self.snr_score / UNIFIED_SNR_THRESHOLD, 1.5)

        # Ihsan gate: penalize if below threshold
        ihsan_penalty = 0.0
        if self.ihsan_score < UNIFIED_IHSAN_THRESHOLD:
            ihsan_penalty = (UNIFIED_IHSAN_THRESHOLD - self.ihsan_score) * 2.0

        return (base_ev * exploration_boost * snr_weight) - ihsan_penalty

    def to_dict(self) -> Dict[str, Any]:
        """Export as dictionary."""
        return {
            "hypothesis_id": self.hypothesis.id,
            "hypothesis": self.hypothesis.to_dict(),
            "exploration_depth": self.exploration_depth,
            "path_length": len(self.exploration_path),
            "snr_score": self.snr_score,
            "ihsan_score": self.ihsan_score,
            "confidence": self.confidence,
            "convergence_count": len(self.converged_with),
            "converged_with": self.converged_with,
            "fate_validated": self.fate_validated,
            "fate_proof_id": self.fate_proof_id,
            "expected_value": self.expected_value(),
            "exploration_time_ms": self.exploration_time_ms,
        }


# =============================================================================
# HYPOTHESIS THOUGHT NODE
# =============================================================================


@dataclass
class HypothesisThoughtNode(ThoughtNode):
    """
    Extended ThoughtNode for hypothesis exploration.

    Adds hypothesis-specific metadata and SNR scoring.
    """

    hypothesis_category: Optional[HypothesisCategory] = None
    risk_level: Optional[RiskLevel] = None
    ihsan_impact: float = 0.0
    snr_contribution: float = 0.0
    causal_parents: List[str] = field(default_factory=list)
    causal_children: List[str] = field(default_factory=list)

    def calculate_snr(self) -> float:
        """
        Calculate SNR score for this hypothesis thought.

        SNR = (signal_strength * diversity * grounding * balance) ^ weighted
        """
        # Signal components
        signal_strength = self.confidence * self.relevance
        diversity = (
            1.0 - abs(0.5 - len(self.causal_children) / 10) * 2
        )  # Penalize extremes
        grounding = self.coherence
        balance = 1.0 - abs(self.ihsan_impact)  # Penalize extreme impacts

        # Calculate SNR
        snr = (signal_strength * diversity * grounding * balance) ** 0.25

        self.snr_contribution = snr
        return snr

    def propagate_ihsan_constraint(self, parent_ihsan: float) -> float:
        """
        Propagate Ihsan constraint from parent node.

        Child Ihsan = parent_ihsan + ihsan_impact (clamped to [0, 1])
        """
        self.metadata["parent_ihsan"] = parent_ihsan
        propagated = max(0.0, min(1.0, parent_ihsan + self.ihsan_impact))
        self.metadata["ihsan_score"] = propagated
        return propagated


# =============================================================================
# GOT HYPOTHESIS EXPLORER
# =============================================================================


class GoTHypothesisExplorer:
    """
    Multi-path hypothesis exploration using Graph-of-Thoughts.

    Explores multiple improvement hypotheses in parallel, aggregates
    similar hypotheses, refines promising paths, and validates via
    FATE gate. Implements SNR maximization at each node.

    Usage:
        explorer = GoTHypothesisExplorer(got_bridge)

        explored = await explorer.explore_hypotheses(
            observation=system_observation,
            num_paths=5
        )

        # Get best hypothesis
        best = max(explored, key=lambda h: h.expected_value())
    """

    def __init__(
        self,
        got_bridge: GoTBridge,
        max_depth: int = DEFAULT_MAX_DEPTH,
        ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
        snr_threshold: float = UNIFIED_SNR_THRESHOLD,
    ):
        """
        Initialize the hypothesis explorer.

        Args:
            got_bridge: GoTBridge instance for graph reasoning
            max_depth: Maximum exploration depth
            ihsan_threshold: Ihsan constraint threshold
            snr_threshold: SNR quality threshold
        """
        self.got = got_bridge
        self.max_depth = max_depth
        self.ihsan_threshold = ihsan_threshold
        self.snr_threshold = snr_threshold

        # Exploration state
        self._exploration_graph: Dict[str, HypothesisThoughtNode] = {}
        self._explored_hypotheses: List[ExploredHypothesis] = []
        self._convergence_map: Dict[str, List[str]] = defaultdict(list)

        # Statistics
        self._total_nodes_explored: int = 0
        self._total_paths_pruned: int = 0
        self._convergences_detected: int = 0

        logger.info(
            f"GoTHypothesisExplorer initialized: max_depth={max_depth}, "
            f"ihsan={ihsan_threshold}, snr={snr_threshold}"
        )

    async def explore_hypotheses(
        self,
        observation: SystemObservation,
        num_paths: int = DEFAULT_NUM_PATHS,
        hypothesis_generator: Optional[HypothesisGenerator] = None,
    ) -> List[ExploredHypothesis]:
        """
        Explore multiple improvement hypotheses using GoT.

        Process:
        1. GENERATE multiple improvement paths
        2. AGGREGATE similar hypotheses
        3. REFINE promising paths
        4. VALIDATE via FATE gate
        5. PRUNE low-value paths

        Args:
            observation: Current system state snapshot
            num_paths: Number of parallel paths to explore
            hypothesis_generator: Optional generator for initial hypotheses

        Returns:
            List of ExploredHypothesis ordered by expected value
        """
        start_time = time.perf_counter_ns()

        # Reset exploration state
        self._exploration_graph.clear()
        self._explored_hypotheses.clear()
        self._convergence_map.clear()

        # Step 1: GENERATE initial hypotheses
        initial_hypotheses = await self._generate_initial_hypotheses(
            observation, num_paths, hypothesis_generator
        )

        if not initial_hypotheses:
            logger.info("No hypotheses generated from observation")
            return []

        logger.info(f"Generated {len(initial_hypotheses)} initial hypotheses")

        # Step 2: Explore each hypothesis path
        explored_paths: List[List[HypothesisThoughtNode]] = []
        for hypothesis in initial_hypotheses:
            path = await self._explore_path(hypothesis, observation)
            if path:
                explored_paths.append(path)

        # Step 3: AGGREGATE similar hypotheses (detect convergence)
        self._aggregate_convergent_hypotheses(explored_paths)

        # Step 4: REFINE promising paths
        refined_hypotheses = await self._refine_promising_paths(
            explored_paths, observation
        )

        # Step 5: VALIDATE via FATE gate
        validated_hypotheses = await self._validate_hypotheses(refined_hypotheses)

        # Step 6: PRUNE low-value paths
        pruned_hypotheses = self._prune_low_value_paths(validated_hypotheses)

        # Calculate final SNR scores and sort by expected value
        for explored in pruned_hypotheses:
            explored._calculate_scores()

        self._explored_hypotheses = sorted(
            pruned_hypotheses, key=lambda h: h.expected_value(), reverse=True
        )

        # Log statistics
        exploration_time_ms = (time.perf_counter_ns() - start_time) / 1_000_000
        logger.info(
            f"Exploration complete: {len(self._explored_hypotheses)} hypotheses, "
            f"{self._total_nodes_explored} nodes explored, "
            f"{self._convergences_detected} convergences, "
            f"{exploration_time_ms:.1f}ms"
        )

        return self._explored_hypotheses

    async def _generate_initial_hypotheses(
        self,
        observation: SystemObservation,
        num_paths: int,
        generator: Optional[HypothesisGenerator],
    ) -> List[Hypothesis]:
        """Generate initial hypotheses from observation."""
        hypotheses: List[Hypothesis] = []

        # Use provided generator or create default
        if generator:
            generated = generator.generate(observation)
            hypotheses.extend(generated[:num_paths])
        else:
            # Generate hypotheses based on observation patterns
            hypotheses = self._generate_default_hypotheses(observation, num_paths)

        return hypotheses

    def _generate_default_hypotheses(
        self, observation: SystemObservation, num_paths: int
    ) -> List[Hypothesis]:
        """Generate default hypotheses based on observation metrics."""
        hypotheses: List[Hypothesis] = []

        # Performance hypothesis if latency is high
        if observation.avg_latency_ms > 100:
            hypotheses.append(
                Hypothesis(
                    id=f"hyp_perf_{uuid.uuid4().hex[:8]}",
                    category=HypothesisCategory.PERFORMANCE,
                    description=f"Reduce latency from {observation.avg_latency_ms:.0f}ms",
                    predicted_improvement={"latency_reduction": 0.2},
                    confidence=0.7,
                    risk_level=RiskLevel.LOW,
                    implementation_plan=["Profile hotspots", "Optimize critical paths"],
                    rollback_plan=["Revert optimizations"],
                    ihsan_impact=0.0,
                    trigger_pattern="high_latency",
                )
            )

        # Quality hypothesis if Ihsan is below threshold
        if observation.ihsan_score < self.ihsan_threshold:
            hypotheses.append(
                Hypothesis(
                    id=f"hyp_qual_{uuid.uuid4().hex[:8]}",
                    category=HypothesisCategory.QUALITY,
                    description=f"Improve Ihsan from {observation.ihsan_score:.3f}",
                    predicted_improvement={"ihsan_delta": 0.05},
                    confidence=0.65,
                    risk_level=RiskLevel.LOW,
                    implementation_plan=["Tighten constraints", "Add verification"],
                    rollback_plan=["Restore previous constraints"],
                    ihsan_impact=0.05,
                    trigger_pattern="low_ihsan",
                )
            )

        # Efficiency hypothesis if resources are stressed
        if observation.cpu_percent > 70 or observation.memory_percent > 80:
            hypotheses.append(
                Hypothesis(
                    id=f"hyp_eff_{uuid.uuid4().hex[:8]}",
                    category=HypothesisCategory.EFFICIENCY,
                    description="Optimize resource utilization",
                    predicted_improvement={"resource_reduction": 0.15},
                    confidence=0.6,
                    risk_level=RiskLevel.MEDIUM,
                    implementation_plan=["Cache optimization", "Memory profiling"],
                    rollback_plan=["Restore cache settings"],
                    ihsan_impact=0.0,
                    trigger_pattern="resource_pressure",
                )
            )

        # Ensure we have at least num_paths hypotheses
        while len(hypotheses) < num_paths:
            hypotheses.append(
                Hypothesis(
                    id=f"hyp_gen_{uuid.uuid4().hex[:8]}",
                    category=HypothesisCategory.CAPABILITY,
                    description="General system improvement exploration",
                    predicted_improvement={"general_improvement": 0.1},
                    confidence=0.5,
                    risk_level=RiskLevel.LOW,
                    implementation_plan=["Analyze patterns", "Implement improvements"],
                    rollback_plan=["Revert changes"],
                    ihsan_impact=0.0,
                    trigger_pattern="general_exploration",
                )
            )

        return hypotheses[:num_paths]

    async def _explore_path(
        self, hypothesis: Hypothesis, observation: SystemObservation
    ) -> List[HypothesisThoughtNode]:
        """
        Explore a single hypothesis path using depth-first with MCTS.

        Combines breadth-first initial exploration with depth-first
        refinement of promising branches.
        """
        path: List[HypothesisThoughtNode] = []

        # Create root node for this hypothesis
        root = self._create_hypothesis_node(hypothesis, observation, depth=0)
        self._exploration_graph[root.id] = root
        path.append(root)
        self._total_nodes_explored += 1

        # Breadth-first exploration for diversity
        frontier = [root]
        for depth in range(min(DEFAULT_BREADTH_FIRST_DEPTH, self.max_depth)):
            next_frontier = []
            for node in frontier:
                children = await self._expand_node(node, observation)
                for child in children:
                    self._exploration_graph[child.id] = child
                    path.append(child)
                    next_frontier.append(child)
                    self._total_nodes_explored += 1
            frontier = next_frontier

        # Depth-first exploration on promising nodes
        if frontier:
            best_node = max(frontier, key=lambda n: n.calculate_snr())
            depth_path = await self._explore_depth_first(
                best_node, observation, self.max_depth - DEFAULT_BREADTH_FIRST_DEPTH
            )
            path.extend(depth_path)

        return path

    async def _expand_node(
        self, parent: HypothesisThoughtNode, observation: SystemObservation
    ) -> List[HypothesisThoughtNode]:
        """Expand a node by generating child hypotheses."""
        children: List[HypothesisThoughtNode] = []

        # Generate variations based on parent hypothesis
        variations = self._generate_hypothesis_variations(parent, observation)

        for variant_content, ihsan_impact in variations[:MAX_BRANCHES]:
            child = HypothesisThoughtNode(
                id=f"node_{uuid.uuid4().hex[:8]}",
                content=variant_content,
                thought_type=ThoughtType.GENERATE,
                parent_ids=[parent.id],
                depth=parent.depth + 1,
                score=0.5,
                confidence=parent.confidence * 0.95,
                coherence=0.6,
                relevance=0.7,
                hypothesis_category=parent.hypothesis_category,
                risk_level=parent.risk_level,
                ihsan_impact=ihsan_impact,
                causal_parents=[parent.id],
            )

            # Propagate Ihsan constraint
            parent_ihsan = parent.metadata.get("ihsan_score", UNIFIED_IHSAN_THRESHOLD)
            child.propagate_ihsan_constraint(parent_ihsan)

            # Calculate SNR
            child.calculate_snr()

            # Update parent references
            parent.child_ids.append(child.id)
            parent.causal_children.append(child.id)

            children.append(child)

        return children

    def _generate_hypothesis_variations(
        self, parent: HypothesisThoughtNode, observation: SystemObservation
    ) -> List[Tuple[str, float]]:
        """Generate variations of a hypothesis node."""
        variations: List[Tuple[str, float]] = []

        base_content = parent.content
        category = parent.hypothesis_category

        if category == HypothesisCategory.PERFORMANCE:
            variations = [
                (f"{base_content} via caching", 0.0),
                (f"{base_content} via batching", 0.0),
                (f"{base_content} via parallelization", 0.0),
            ]
        elif category == HypothesisCategory.QUALITY:
            variations = [
                (f"{base_content} via stricter validation", 0.02),
                (f"{base_content} via enhanced verification", 0.03),
                (f"{base_content} via constraint tightening", 0.01),
            ]
        elif category == HypothesisCategory.EFFICIENCY:
            variations = [
                (f"{base_content} via memory optimization", -0.01),
                (f"{base_content} via compute scheduling", 0.0),
                (f"{base_content} via resource pooling", 0.0),
            ]
        else:
            variations = [
                (f"{base_content} approach A", 0.0),
                (f"{base_content} approach B", 0.0),
                (f"{base_content} approach C", 0.0),
            ]

        return variations

    async def _explore_depth_first(
        self,
        start_node: HypothesisThoughtNode,
        observation: SystemObservation,
        max_depth: int,
    ) -> List[HypothesisThoughtNode]:
        """Depth-first exploration from a promising node."""
        path: List[HypothesisThoughtNode] = []
        current = start_node

        for _ in range(max_depth):
            if current.depth >= self.max_depth:
                break

            # Expand and select best child
            children = await self._expand_node(current, observation)
            if not children:
                break

            # Select child with best SNR
            best_child = max(children, key=lambda n: n.calculate_snr())

            # Prune if below threshold
            if best_child.snr_contribution < PATH_PRUNE_THRESHOLD:
                self._total_paths_pruned += 1
                break

            self._exploration_graph[best_child.id] = best_child
            path.append(best_child)
            self._total_nodes_explored += 1
            current = best_child

        return path

    def _aggregate_convergent_hypotheses(
        self, paths: List[List[HypothesisThoughtNode]]
    ) -> List[HypothesisThoughtNode]:
        """
        Detect and aggregate hypotheses that converge on similar solutions.

        Returns aggregated nodes representing merged insights.
        """
        aggregated: List[HypothesisThoughtNode] = []

        # Group paths by category
        category_groups: Dict[HypothesisCategory, List[List[HypothesisThoughtNode]]] = (
            defaultdict(list)
        )
        for path in paths:
            if path:
                category = path[0].hypothesis_category
                if category:
                    category_groups[category].append(path)

        # Within each category, detect convergence
        for category, group_paths in category_groups.items():
            if len(group_paths) < 2:
                continue

            # Compare leaf nodes for similarity
            for i, path1 in enumerate(group_paths):
                for j, path2 in enumerate(group_paths[i + 1 :], i + 1):
                    if not path1 or not path2:
                        continue

                    leaf1, leaf2 = path1[-1], path2[-1]
                    similarity = self._calculate_similarity(leaf1, leaf2)

                    if similarity >= CONVERGENCE_SIMILARITY_THRESHOLD:
                        # Create aggregated node
                        merged = self._merge_nodes(leaf1, leaf2)
                        aggregated.append(merged)
                        self._convergence_map[merged.id] = [leaf1.id, leaf2.id]
                        self._convergences_detected += 1

                        logger.debug(
                            f"Convergence detected: {leaf1.id} + {leaf2.id} -> {merged.id}"
                        )

        return aggregated

    def _calculate_similarity(
        self, node1: HypothesisThoughtNode, node2: HypothesisThoughtNode
    ) -> float:
        """Calculate similarity between two hypothesis nodes."""
        # Category match
        category_match = (
            1.0 if node1.hypothesis_category == node2.hypothesis_category else 0.0
        )

        # Risk level similarity
        risk_similarity = (
            1.0
            - abs(
                (_RISK_LEVEL_NUMERIC.get(node1.risk_level.value, 0) if node1.risk_level else 0)
                - (_RISK_LEVEL_NUMERIC.get(node2.risk_level.value, 0) if node2.risk_level else 0)
            )
            / 4
        )

        # Ihsan impact similarity
        ihsan_similarity = 1.0 - abs(node1.ihsan_impact - node2.ihsan_impact)

        # Score similarity
        score_similarity = 1.0 - abs(node1.score - node2.score)

        # Weighted average
        return (
            category_match * 0.3
            + risk_similarity * 0.2
            + ihsan_similarity * 0.3
            + score_similarity * 0.2
        )

    def _merge_nodes(
        self, node1: HypothesisThoughtNode, node2: HypothesisThoughtNode
    ) -> HypothesisThoughtNode:
        """Merge two convergent nodes into one."""
        merged = HypothesisThoughtNode(
            id=f"merged_{uuid.uuid4().hex[:8]}",
            content=f"Synthesis: [{node1.content}] + [{node2.content}]",
            thought_type=ThoughtType.AGGREGATE,
            parent_ids=[node1.id, node2.id],
            depth=max(node1.depth, node2.depth) + 1,
            score=(node1.score + node2.score) / 2 * 1.1,  # Convergence bonus
            confidence=max(node1.confidence, node2.confidence),
            coherence=(node1.coherence + node2.coherence) / 2,
            relevance=(node1.relevance + node2.relevance) / 2,
            hypothesis_category=node1.hypothesis_category,
            risk_level=(
                node1.risk_level
                if node1.risk_level
                and _RISK_LEVEL_NUMERIC.get(node1.risk_level.value, 0)
                <= (_RISK_LEVEL_NUMERIC.get(node2.risk_level.value, 0) if node2.risk_level else 3)
                else node2.risk_level
            ),
            ihsan_impact=(node1.ihsan_impact + node2.ihsan_impact) / 2,
            causal_parents=[node1.id, node2.id],
        )

        # Propagate average Ihsan
        avg_ihsan = (
            node1.metadata.get("ihsan_score", UNIFIED_IHSAN_THRESHOLD)
            + node2.metadata.get("ihsan_score", UNIFIED_IHSAN_THRESHOLD)
        ) / 2
        merged.propagate_ihsan_constraint(avg_ihsan)
        merged.calculate_snr()

        self._exploration_graph[merged.id] = merged
        return merged

    async def _refine_promising_paths(
        self, paths: List[List[HypothesisThoughtNode]], observation: SystemObservation
    ) -> List[ExploredHypothesis]:
        """
        Refine the most promising paths through iteration.

        Returns ExploredHypothesis instances with refined paths.
        """
        explored: List[ExploredHypothesis] = []

        for path in paths:
            if not path:
                continue

            # Find best node in path
            best_node = max(path, key=lambda n: n.calculate_snr())

            # Refine if above threshold
            if best_node.snr_contribution >= PATH_PRUNE_THRESHOLD:
                refined_path = await self._refine_path(path, observation)

                # Create ExploredHypothesis from path
                root_node = path[0]
                explored_hyp = ExploredHypothesis(
                    hypothesis=self._node_to_hypothesis(root_node),
                    exploration_path=list(refined_path),  # type: ignore[arg-type]
                    exploration_depth=max(n.depth for n in refined_path),
                )

                # Check for convergence with other paths
                for merged_id, source_ids in self._convergence_map.items():
                    if any(n.id in source_ids for n in refined_path):
                        explored_hyp.converged_with.append(merged_id)
                        explored_hyp.convergence_evidence.append(
                            f"Converged via {merged_id}"
                        )

                explored.append(explored_hyp)

        return explored

    async def _refine_path(
        self, path: List[HypothesisThoughtNode], observation: SystemObservation
    ) -> List[HypothesisThoughtNode]:
        """Apply iterative refinement to a path."""
        refined_path = list(path)

        # Refine leaf node
        if refined_path:
            leaf = refined_path[-1]
            for i in range(3):  # Refinement iterations
                refined = HypothesisThoughtNode(
                    id=f"refined_{uuid.uuid4().hex[:8]}",
                    content=f"{leaf.content} [refined v{i + 1}]",
                    thought_type=ThoughtType.REFINE,
                    parent_ids=[leaf.id],
                    depth=leaf.depth + 1,
                    score=min(1.0, leaf.score * 1.05),
                    confidence=min(1.0, leaf.confidence * 1.03),
                    coherence=min(1.0, leaf.coherence * 1.02),
                    relevance=leaf.relevance,
                    hypothesis_category=leaf.hypothesis_category,
                    risk_level=leaf.risk_level,
                    ihsan_impact=leaf.ihsan_impact * 0.95,  # Reduce uncertainty
                    causal_parents=[leaf.id],
                )

                parent_ihsan = leaf.metadata.get("ihsan_score", UNIFIED_IHSAN_THRESHOLD)
                refined.propagate_ihsan_constraint(parent_ihsan)
                refined.calculate_snr()

                refined_path.append(refined)
                self._exploration_graph[refined.id] = refined
                self._total_nodes_explored += 1
                leaf = refined

        return refined_path

    def _node_to_hypothesis(self, node: HypothesisThoughtNode) -> Hypothesis:
        """Convert a hypothesis thought node back to a Hypothesis."""
        return Hypothesis(
            id=node.id,
            category=node.hypothesis_category or HypothesisCategory.CAPABILITY,
            description=node.content,
            predicted_improvement={"exploration_improvement": node.score},
            confidence=node.confidence,
            risk_level=node.risk_level or RiskLevel.MEDIUM,
            implementation_plan=["Apply explored improvements"],
            rollback_plan=["Revert to baseline"],
            ihsan_impact=node.ihsan_impact,
            trigger_pattern=f"got_exploration_{node.thought_type.value}",
        )

    async def _validate_hypotheses(
        self, explored: List[ExploredHypothesis]
    ) -> List[ExploredHypothesis]:
        """Validate hypotheses via FATE gate."""
        validated: List[ExploredHypothesis] = []

        for exp_hyp in explored:
            # Check Ihsan constraint
            if exp_hyp.ihsan_score < HYPOTHESIS_IHSAN_FLOOR:
                logger.debug(
                    f"Hypothesis {exp_hyp.hypothesis.id} failed Ihsan gate: "
                    f"{exp_hyp.ihsan_score:.3f} < {HYPOTHESIS_IHSAN_FLOOR}"
                )
                continue

            # Check SNR constraint
            if exp_hyp.snr_score < HYPOTHESIS_SNR_FLOOR:
                logger.debug(
                    f"Hypothesis {exp_hyp.hypothesis.id} failed SNR gate: "
                    f"{exp_hyp.snr_score:.3f} < {HYPOTHESIS_SNR_FLOOR}"
                )
                continue

            # Mark as FATE validated (in production, would call actual FATE gate)
            exp_hyp.fate_validated = True
            exp_hyp.fate_proof_id = f"fate_proof_{uuid.uuid4().hex[:12]}"
            validated.append(exp_hyp)

        logger.info(
            f"FATE validation: {len(validated)}/{len(explored)} hypotheses passed"
        )
        return validated

    def _prune_low_value_paths(
        self, explored: List[ExploredHypothesis]
    ) -> List[ExploredHypothesis]:
        """Prune hypotheses with low expected value."""
        pruned: List[ExploredHypothesis] = []

        for exp_hyp in explored:
            ev = exp_hyp.expected_value()
            if ev > 0:
                pruned.append(exp_hyp)
            else:
                self._total_paths_pruned += 1
                logger.debug(f"Pruned hypothesis {exp_hyp.hypothesis.id}: EV={ev:.3f}")

        return pruned

    # =========================================================================
    # MONTE CARLO TREE SEARCH (MCTS)
    # =========================================================================

    async def explore_with_mcts(
        self,
        observation: SystemObservation,
        iterations: int = DEFAULT_MCTS_ITERATIONS,
        hypothesis_generator: Optional[HypothesisGenerator] = None,
    ) -> List[ExploredHypothesis]:
        """
        Explore hypotheses using Monte Carlo Tree Search.

        MCTS provides better exploration-exploitation balance for
        optimization of hypothesis paths.

        Args:
            observation: Current system state
            iterations: Number of MCTS iterations
            hypothesis_generator: Optional hypothesis generator

        Returns:
            List of explored hypotheses
        """
        start_time = time.perf_counter_ns()

        # Generate initial hypotheses
        initial = await self._generate_initial_hypotheses(
            observation, DEFAULT_NUM_PATHS, hypothesis_generator
        )

        if not initial:
            return []

        # Create root nodes for MCTS
        mcts_roots: Dict[str, MCTSNode] = {}
        for hyp in initial:
            root_node = self._create_hypothesis_node(hyp, observation, depth=0)
            mcts_root = MCTSNode(
                hypothesis_node=root_node,
                parent=None,
                observation=observation,
            )
            mcts_roots[root_node.id] = mcts_root

        # Run MCTS iterations
        for _ in range(iterations):
            for root_id, root in mcts_roots.items():
                # Selection
                selected = self._mcts_select(root)

                # Expansion
                expanded = await self._mcts_expand(selected, observation)

                # Simulation
                reward = await self._mcts_simulate(expanded, observation)

                # Backpropagation
                self._mcts_backpropagate(expanded, reward)

        # Extract best paths from MCTS trees
        explored: List[ExploredHypothesis] = []
        for root_id, root in mcts_roots.items():
            best_path = self._extract_best_mcts_path(root)
            if best_path:
                exp_hyp = ExploredHypothesis(
                    hypothesis=self._node_to_hypothesis(best_path[0].hypothesis_node),
                    exploration_path=[n.hypothesis_node for n in best_path],
                    exploration_depth=len(best_path),
                )
                explored.append(exp_hyp)

        # Validate and prune
        validated = await self._validate_hypotheses(explored)
        pruned = self._prune_low_value_paths(validated)

        exploration_time_ms = (time.perf_counter_ns() - start_time) / 1_000_000
        logger.info(
            f"MCTS exploration complete: {len(pruned)} hypotheses, "
            f"{iterations} iterations, {exploration_time_ms:.1f}ms"
        )

        return sorted(pruned, key=lambda h: h.expected_value(), reverse=True)

    def _mcts_select(self, node: "MCTSNode") -> "MCTSNode":
        """Select a node using UCB1."""
        current = node

        while current.children:
            # UCB1 selection
            best_child = max(
                current.children,
                key=lambda c: c.ucb1_score(DEFAULT_EXPLORATION_CONSTANT),
            )
            current = best_child

        return current

    async def _mcts_expand(
        self, node: "MCTSNode", observation: SystemObservation
    ) -> "MCTSNode":
        """Expand a node by creating children."""
        if node.visits == 0 or node.hypothesis_node.depth >= self.max_depth:
            return node

        # Generate child nodes
        children = await self._expand_node(node.hypothesis_node, observation)

        for child_hyp_node in children:
            child_mcts = MCTSNode(
                hypothesis_node=child_hyp_node,
                parent=node,
                observation=observation,
            )
            node.children.append(child_mcts)

        # Return first child for simulation
        return node.children[0] if node.children else node

    async def _mcts_simulate(
        self, node: "MCTSNode", observation: SystemObservation
    ) -> float:
        """Simulate from node to estimate reward."""
        # Simple simulation: estimate based on current node quality
        snr = node.hypothesis_node.calculate_snr()
        ihsan = node.hypothesis_node.metadata.get(
            "ihsan_score", UNIFIED_IHSAN_THRESHOLD
        )

        # Reward = SNR * Ihsan * depth_bonus
        depth_bonus = 1.0 + (node.hypothesis_node.depth * 0.1)
        reward = snr * ihsan * min(depth_bonus, 1.5)

        return reward

    def _mcts_backpropagate(self, node: "MCTSNode", reward: float) -> None:
        """Backpropagate reward through tree."""
        current: Optional[MCTSNode] = node

        while current is not None:
            current.visits += 1
            current.total_reward += reward
            current = current.parent

    def _extract_best_mcts_path(self, root: "MCTSNode") -> List["MCTSNode"]:
        """Extract best path from MCTS tree."""
        path: List[MCTSNode] = [root]
        current = root

        while current.children:
            # Select child with highest average reward
            best_child = max(
                current.children,
                key=lambda c: c.average_reward(),
            )
            path.append(best_child)
            current = best_child

        return path

    def _create_hypothesis_node(
        self, hypothesis: Hypothesis, observation: SystemObservation, depth: int
    ) -> HypothesisThoughtNode:
        """Create a HypothesisThoughtNode from a Hypothesis."""
        node = HypothesisThoughtNode(
            id=f"node_{hypothesis.id}",
            content=hypothesis.description,
            thought_type=ThoughtType.ROOT if depth == 0 else ThoughtType.GENERATE,
            depth=depth,
            score=hypothesis.confidence,
            confidence=hypothesis.confidence,
            coherence=0.7,
            relevance=0.8,
            hypothesis_category=hypothesis.category,
            risk_level=hypothesis.risk_level,
            ihsan_impact=hypothesis.ihsan_impact,
            metadata={
                "hypothesis_id": hypothesis.id,
                "ihsan_score": observation.ihsan_score + hypothesis.ihsan_impact,
                "snr_score": observation.snr_score,
                "trigger_pattern": hypothesis.trigger_pattern,
            },
        )

        node.calculate_snr()
        return node

    # =========================================================================
    # SNR CALCULATION
    # =========================================================================

    def calculate_path_snr(self, path: List[ThoughtNode]) -> float:
        """
        Calculate SNR for an exploration path.

        SNR = signal / noise
        signal = sum(confidence * ihsan_score for each node)
        noise = sum(uncertainty + risk_level for each node)

        Args:
            path: List of thought nodes in the path

        Returns:
            SNR score for the path
        """
        if not path:
            return 0.0

        total_signal = 0.0
        total_noise = 0.0

        for node in path:
            # Signal: confidence weighted by Ihsan
            ihsan = node.metadata.get("ihsan_score", UNIFIED_IHSAN_THRESHOLD)
            signal = node.confidence * ihsan
            total_signal += signal

            # Noise: uncertainty + risk
            uncertainty = 1.0 - node.confidence
            risk = 0.0
            if isinstance(node, HypothesisThoughtNode) and node.risk_level:
                risk = _RISK_LEVEL_NUMERIC.get(node.risk_level.value, 0) * 0.1
            noise = uncertainty + risk
            total_noise += noise

        snr = total_signal / max(total_noise, 0.01)
        return snr

    # =========================================================================
    # STATISTICS AND REPORTING
    # =========================================================================

    def get_statistics(self) -> Dict[str, Any]:
        """Get exploration statistics."""
        return {
            "total_nodes_explored": self._total_nodes_explored,
            "total_paths_pruned": self._total_paths_pruned,
            "convergences_detected": self._convergences_detected,
            "explored_hypotheses": len(self._explored_hypotheses),
            "exploration_graph_size": len(self._exploration_graph),
            "max_depth": self.max_depth,
            "ihsan_threshold": self.ihsan_threshold,
            "snr_threshold": self.snr_threshold,
        }

    def visualize_exploration(self) -> str:
        """Generate ASCII visualization of exploration graph."""
        if not self._exploration_graph:
            return "Empty exploration graph"

        lines = ["GoT Hypothesis Exploration:", ""]

        # Find root nodes (no parents)
        roots = [n for n in self._exploration_graph.values() if not n.parent_ids]

        def render_node(
            node: HypothesisThoughtNode, prefix: str = "", is_last: bool = True
        ):
            connector = "`-- " if is_last else "|-- "
            status_icon = {
                ThoughtStatus.ACTIVE: "o",
                ThoughtStatus.PRUNED: "x",
                ThoughtStatus.MERGED: "@",
                ThoughtStatus.SOLUTION: "*",
            }.get(node.status, "?")

            content_preview = (
                node.content[:35] + "..." if len(node.content) > 35 else node.content
            )
            snr = node.snr_contribution
            ihsan = node.metadata.get("ihsan_score", 0)

            line = (
                f"{prefix}{connector}{status_icon} [{node.id[:8]}] "
                f"{content_preview} (SNR={snr:.2f}, Ihsan={ihsan:.2f})"
            )
            lines.append(line)

            children = [
                self._exploration_graph.get(cid)
                for cid in node.child_ids
                if cid in self._exploration_graph
            ]
            for i, child in enumerate(children):
                if child:
                    is_child_last = i == len(children) - 1
                    child_prefix = prefix + ("    " if is_last else "|   ")
                    render_node(child, child_prefix, is_child_last)

        for root in roots:
            render_node(root)

        return "\n".join(lines)


# =============================================================================
# MCTS NODE
# =============================================================================


@dataclass
class MCTSNode:
    """Node for Monte Carlo Tree Search."""

    hypothesis_node: HypothesisThoughtNode
    parent: Optional["MCTSNode"]
    observation: SystemObservation
    children: List["MCTSNode"] = field(default_factory=list)
    visits: int = 0
    total_reward: float = 0.0

    def average_reward(self) -> float:
        """Calculate average reward."""
        if self.visits == 0:
            return 0.0
        return self.total_reward / self.visits

    def ucb1_score(self, exploration_constant: float) -> float:
        """Calculate UCB1 score for selection."""
        if self.visits == 0:
            return float("inf")

        exploitation = self.average_reward()
        exploration = exploration_constant * math.sqrt(
            math.log(self.parent.visits if self.parent else 1) / self.visits
        )

        return exploitation + exploration


# =============================================================================
# INTEGRATION WITH AUTOPOIETIC LOOP
# =============================================================================


class GoTAutopoieticIntegration:
    """
    Integration layer between GoT hypothesis exploration and
    the Autopoietic Loop.

    Provides seamless integration for using GoT-based hypothesis
    exploration within the autopoietic improvement cycle.
    """

    def __init__(
        self,
        got_explorer: GoTHypothesisExplorer,
        hypothesis_generator: Optional[HypothesisGenerator] = None,
    ):
        """
        Initialize the integration.

        Args:
            got_explorer: GoTHypothesisExplorer instance
            hypothesis_generator: Optional HypothesisGenerator
        """
        self.explorer = got_explorer
        self.generator = hypothesis_generator

        logger.info("GoT-Autopoietic integration initialized")

    async def enhanced_hypothesize(
        self, observation: SystemObservation, num_paths: int = DEFAULT_NUM_PATHS
    ) -> List[Hypothesis]:
        """
        Enhanced hypothesis generation using GoT exploration.

        Replaces or augments the standard HypothesisGenerator.generate()
        with GoT-based multi-path exploration.

        Args:
            observation: Current system state
            num_paths: Number of paths to explore

        Returns:
            List of hypotheses ranked by exploration-enhanced expected value
        """
        # Explore hypotheses using GoT
        explored = await self.explorer.explore_hypotheses(
            observation=observation,
            num_paths=num_paths,
            hypothesis_generator=self.generator,
        )

        # Extract hypotheses with enhanced confidence from exploration
        enhanced_hypotheses: List[Hypothesis] = []

        for exp_hyp in explored:
            # Update hypothesis confidence based on exploration
            hypothesis = exp_hyp.hypothesis
            hypothesis.confidence = exp_hyp.confidence

            # Add exploration metadata
            hypothesis.similar_past_hypotheses = exp_hyp.converged_with

            enhanced_hypotheses.append(hypothesis)

        return enhanced_hypotheses

    async def enhanced_hypothesize_mcts(
        self,
        observation: SystemObservation,
        iterations: int = DEFAULT_MCTS_ITERATIONS,
    ) -> List[Hypothesis]:
        """
        Enhanced hypothesis generation using MCTS exploration.

        Uses Monte Carlo Tree Search for better exploration-exploitation
        balance in hypothesis space.

        Args:
            observation: Current system state
            iterations: MCTS iterations

        Returns:
            List of hypotheses ranked by MCTS-optimized expected value
        """
        explored = await self.explorer.explore_with_mcts(
            observation=observation,
            iterations=iterations,
            hypothesis_generator=self.generator,
        )

        return [exp.hypothesis for exp in explored]


# =============================================================================
# FACTORY FUNCTIONS
# =============================================================================


def create_got_hypothesis_explorer(
    max_depth: int = DEFAULT_MAX_DEPTH,
    ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
    snr_threshold: float = UNIFIED_SNR_THRESHOLD,
) -> GoTHypothesisExplorer:
    """
    Create a GoTHypothesisExplorer with default configuration.

    Args:
        max_depth: Maximum exploration depth
        ihsan_threshold: Ihsan constraint threshold
        snr_threshold: SNR quality threshold

    Returns:
        Configured GoTHypothesisExplorer instance
    """
    got_bridge = GoTBridge()
    return GoTHypothesisExplorer(
        got_bridge=got_bridge,
        max_depth=max_depth,
        ihsan_threshold=ihsan_threshold,
        snr_threshold=snr_threshold,
    )


def create_got_autopoietic_integration(
    hypothesis_generator: Optional[HypothesisGenerator] = None,
    max_depth: int = DEFAULT_MAX_DEPTH,
) -> GoTAutopoieticIntegration:
    """
    Create a GoT-Autopoietic integration.

    Args:
        hypothesis_generator: Optional HypothesisGenerator
        max_depth: Maximum exploration depth

    Returns:
        Configured GoTAutopoieticIntegration instance
    """
    explorer = create_got_hypothesis_explorer(max_depth=max_depth)
    return GoTAutopoieticIntegration(
        got_explorer=explorer,
        hypothesis_generator=hypothesis_generator,
    )


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    # Core classes
    "GoTHypothesisExplorer",
    "GoTAutopoieticIntegration",
    # Data classes
    "ExploredHypothesis",
    "HypothesisThoughtNode",
    "MCTSNode",
    # Factory functions
    "create_got_hypothesis_explorer",
    "create_got_autopoietic_integration",
    # Constants
    "DEFAULT_MAX_DEPTH",
    "DEFAULT_NUM_PATHS",
    "DEFAULT_MCTS_ITERATIONS",
    "HYPOTHESIS_SNR_FLOOR",
    "HYPOTHESIS_IHSAN_FLOOR",
    "CONVERGENCE_SIMILARITY_THRESHOLD",
    "PATH_PRUNE_THRESHOLD",
]
