"""
SDPO Test-Time Discovery Engine — Exploration and Novel Solution Discovery
===============================================================================

Enables test-time training and discovery of novel solutions:
- Exploration of solution space via SDPO guidance
- Novelty scoring and diversity maintenance
- Continuous improvement through discovered patterns

Standing on Giants: Shannon + SDPO Paper + Exploration-Exploitation
Genesis Strict Synthesis v2.2.2
"""

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)
from core.sdpo.optimization import (
    BIZRAFeedbackGenerator,
    SDPOAdvantage,
    SDPOAdvantageCalculator,
    SDPOFeedback,
)


@dataclass
class DiscoveryConfig:
    """Configuration for test-time discovery."""

    max_exploration_depth: int = 5
    novelty_threshold: float = 0.3  # Min novelty to keep solution
    diversity_weight: float = 0.4  # Weight for diversity in selection
    exploitation_ratio: float = 0.7  # Exploit vs explore balance
    max_solutions_per_query: int = 10
    ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD
    snr_threshold: float = UNIFIED_SNR_THRESHOLD


@dataclass
class ExplorationPath:
    """A single exploration path in solution space."""

    id: str
    depth: int
    solution: str
    parent_id: Optional[str]
    novelty_score: float
    quality_score: float
    advantage: Optional[SDPOAdvantage]
    feedback_applied: bool
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "depth": self.depth,
            "solution_preview": (
                self.solution[:200] + "..."
                if len(self.solution) > 200
                else self.solution
            ),
            "parent_id": self.parent_id,
            "novelty_score": self.novelty_score,
            "quality_score": self.quality_score,
            "advantage": self.advantage.to_dict() if self.advantage else None,
            "feedback_applied": self.feedback_applied,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DiscoveryResult:
    """Result from test-time discovery."""

    query: str
    best_solution: str
    all_paths: List[ExplorationPath]
    total_explorations: int
    novel_discoveries: int
    average_novelty: float
    average_quality: float
    discovery_time_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_preview": (
                self.query[:100] + "..." if len(self.query) > 100 else self.query
            ),
            "best_solution_preview": self.best_solution[:200] + "...",
            "total_explorations": self.total_explorations,
            "novel_discoveries": self.novel_discoveries,
            "average_novelty": self.average_novelty,
            "average_quality": self.average_quality,
            "discovery_time_ms": self.discovery_time_ms,
            "top_paths": [p.to_dict() for p in self.all_paths[:5]],
        }


class NoveltyScorer:
    """
    Scores novelty of solutions relative to archive.

    Uses simple content-based hashing and Jaccard distance for novelty.
    More sophisticated implementations could use embedding similarity.
    """

    def __init__(self, archive_size: int = 1000):
        self._archive: List[Set[str]] = []
        self._archive_hashes: Set[str] = set()
        self._max_size = archive_size

    def score_novelty(self, solution: str) -> float:
        """
        Score novelty of a solution (0-1, higher = more novel).

        Uses Jaccard distance from archived solutions.
        """
        if not self._archive:
            return 1.0  # First solution is maximally novel

        # Tokenize solution
        tokens = set(solution.lower().split())

        # Calculate minimum Jaccard distance from archive
        min_similarity = 1.0
        for archived in self._archive:
            intersection = len(tokens & archived)
            union = len(tokens | archived)
            similarity = intersection / union if union > 0 else 0
            min_similarity = min(min_similarity, similarity)

        # Novelty = inverse of maximum similarity
        return 1.0 - min_similarity

    def add_to_archive(self, solution: str):
        """Add solution to novelty archive."""
        solution_hash = hashlib.md5(
            solution.encode(), usedforsecurity=False
        ).hexdigest()

        if solution_hash in self._archive_hashes:
            return  # Already archived

        # Evict oldest if full
        if len(self._archive) >= self._max_size:
            self._archive.pop(0)
            # Note: hash cleanup would be more complex in production

        tokens = set(solution.lower().split())
        self._archive.append(tokens)
        self._archive_hashes.add(solution_hash)

    def get_archive_size(self) -> int:
        return len(self._archive)


class SDPOTestTimeDiscovery:
    """
    Test-Time Discovery Engine using SDPO.

    Explores solution space at inference time, using SDPO feedback
    to guide exploration toward novel, high-quality solutions.

    Usage:
        discovery = SDPOTestTimeDiscovery(llm_callback=my_llm)

        result = await discovery.discover(
            query="How can we optimize the federation protocol?",
            initial_solutions=["Use sharding", "Add caching"],
        )

        print(f"Best solution: {result.best_solution}")
        print(f"Novel discoveries: {result.novel_discoveries}")
    """

    def __init__(
        self,
        llm_callback: Optional[Callable[[str], str]] = None,
        config: Optional[DiscoveryConfig] = None,
    ):
        self.llm_callback = llm_callback
        self.config = config or DiscoveryConfig()
        self.advantage_calculator = SDPOAdvantageCalculator()
        self.feedback_generator = BIZRAFeedbackGenerator()
        self.novelty_scorer = NoveltyScorer()

        # Discovery state
        self._paths: Dict[str, ExplorationPath] = {}
        self._best_by_query: Dict[str, str] = {}

    async def discover(
        self,
        query: str,
        initial_solutions: Optional[List[str]] = None,
        context: Optional[str] = None,
    ) -> DiscoveryResult:
        """
        Discover solutions via test-time exploration.

        Args:
            query: The problem/query to solve
            initial_solutions: Starting solutions to explore from
            context: Additional context for the query

        Returns:
            DiscoveryResult with best solution and exploration paths
        """
        start = datetime.now(timezone.utc)
        paths: List[ExplorationPath] = []

        # Generate initial solutions if not provided
        if not initial_solutions:
            initial_solutions = await self._generate_initial_solutions(query, context)

        # Create initial paths
        for i, solution in enumerate(initial_solutions):
            path = self._create_path(
                solution=solution,
                parent_id=None,
                depth=0,
            )
            paths.append(path)
            self._paths[path.id] = path

        # Iterative exploration
        for depth in range(1, self.config.max_exploration_depth + 1):
            # Select paths to explore (exploitation vs exploration)
            selected = self._select_paths_to_explore(paths)

            if not selected:
                break

            # Explore each selected path
            new_paths = []
            for parent_path in selected:
                children = await self._explore_path(query, parent_path, context)
                new_paths.extend(children)

            paths.extend(new_paths)

            # Prune to keep top solutions
            if len(paths) > self.config.max_solutions_per_query * 2:
                paths = self._prune_paths(paths)

        # Select best solution
        best_path = self._select_best_path(paths)
        best_solution = best_path.solution if best_path else ""

        # Archive novel solutions
        novel_count = 0
        for path in paths:
            if path.novelty_score >= self.config.novelty_threshold:
                self.novelty_scorer.add_to_archive(path.solution)
                novel_count += 1

        elapsed_ms = (datetime.now(timezone.utc) - start).total_seconds() * 1000

        return DiscoveryResult(
            query=query,
            best_solution=best_solution,
            all_paths=sorted(paths, key=lambda p: p.quality_score, reverse=True),
            total_explorations=len(paths),
            novel_discoveries=novel_count,
            average_novelty=(
                sum(p.novelty_score for p in paths) / len(paths) if paths else 0
            ),
            average_quality=(
                sum(p.quality_score for p in paths) / len(paths) if paths else 0
            ),
            discovery_time_ms=elapsed_ms,
        )

    async def _generate_initial_solutions(
        self,
        query: str,
        context: Optional[str],
    ) -> List[str]:
        """Generate initial solution candidates."""
        if not self.llm_callback:
            # Return placeholder without LLM
            return [f"Solution for: {query[:50]}..."]

        prompt = f"""Generate 3 diverse approaches to solve this problem:

Problem: {query}
{f"Context: {context}" if context else ""}

Provide 3 distinct solutions, separated by "---":"""

        response = self.llm_callback(prompt)
        solutions = [s.strip() for s in response.split("---") if s.strip()]

        return solutions[:3] if solutions else [response]

    async def _explore_path(
        self,
        query: str,
        parent: ExplorationPath,
        context: Optional[str],
    ) -> List[ExplorationPath]:
        """Explore from a parent path to generate child solutions."""
        children = []

        # Quality-based exploration
        quality_check = self._check_quality(parent.solution)

        if not quality_check["passes"]:
            # Generate feedback and apply SDPO correction
            feedback = self.feedback_generator.generate_feedback(quality_check)

            if self.llm_callback:
                improved = await self._apply_sdpo_improvement(
                    query, parent.solution, feedback, context
                )

                advantage = await self.advantage_calculator.calculate_advantages(
                    question=query,
                    failed_attempt=parent.solution,
                    feedback=feedback.text,
                    corrected_attempt=improved,
                )

                child = self._create_path(
                    solution=improved,
                    parent_id=parent.id,
                    depth=parent.depth + 1,
                    advantage=advantage,
                    feedback_applied=True,
                )
                children.append(child)

        # Novelty-based exploration (always try to find novel solutions)
        if self.llm_callback:
            novel_solution = await self._generate_novel_variant(
                query, parent.solution, context
            )

            if novel_solution != parent.solution:
                child = self._create_path(
                    solution=novel_solution,
                    parent_id=parent.id,
                    depth=parent.depth + 1,
                )
                children.append(child)

        return children

    async def _apply_sdpo_improvement(
        self,
        query: str,
        solution: str,
        feedback: SDPOFeedback,
        context: Optional[str],
    ) -> str:
        """Apply SDPO-guided improvement."""
        if not self.llm_callback:
            return solution

        prompt = f"""Original problem: {query}
{f"Context: {context}" if context else ""}

Current solution:
{solution}

Feedback for improvement:
{feedback.text}

Please provide an improved solution that addresses the feedback:"""

        return self.llm_callback(prompt)

    async def _generate_novel_variant(
        self,
        query: str,
        solution: str,
        context: Optional[str],
    ) -> str:
        """Generate a novel variant of a solution."""
        if not self.llm_callback:
            return solution

        prompt = f"""Original problem: {query}
{f"Context: {context}" if context else ""}

Existing solution:
{solution}

Generate a DIFFERENT approach that solves the same problem but uses alternative methods or perspectives:"""

        return self.llm_callback(prompt)

    def _create_path(
        self,
        solution: str,
        parent_id: Optional[str],
        depth: int,
        advantage: Optional[SDPOAdvantage] = None,
        feedback_applied: bool = False,
    ) -> ExplorationPath:
        """Create a new exploration path."""
        path_id = hashlib.md5(
            f"{solution}{parent_id}{depth}".encode(), usedforsecurity=False
        ).hexdigest()[:12]

        novelty = self.novelty_scorer.score_novelty(solution)
        quality = self._estimate_quality(solution)

        return ExplorationPath(
            id=path_id,
            depth=depth,
            solution=solution,
            parent_id=parent_id,
            novelty_score=novelty,
            quality_score=quality,
            advantage=advantage,
            feedback_applied=feedback_applied,
        )

    def _select_paths_to_explore(
        self,
        paths: List[ExplorationPath],
    ) -> List[ExplorationPath]:
        """Select paths for further exploration using exploitation/exploration balance."""
        if not paths:
            return []

        # Score each path
        scores = []
        for path in paths:
            exploit_score = path.quality_score
            explore_score = path.novelty_score

            combined = (
                self.config.exploitation_ratio * exploit_score
                + (1 - self.config.exploitation_ratio) * explore_score
            )
            scores.append((path, combined))

        # Sort by combined score
        scores.sort(key=lambda x: x[1], reverse=True)

        # Select top paths (up to half of max solutions)
        n_select = max(1, self.config.max_solutions_per_query // 2)
        return [path for path, _ in scores[:n_select]]

    def _select_best_path(
        self, paths: List[ExplorationPath]
    ) -> Optional[ExplorationPath]:
        """Select the best overall path."""
        if not paths:
            return None

        # Prioritize quality, with novelty as tiebreaker
        return max(
            paths,
            key=lambda p: (p.quality_score, p.novelty_score),
        )

    def _prune_paths(self, paths: List[ExplorationPath]) -> List[ExplorationPath]:
        """Prune paths to keep diverse, high-quality solutions."""
        # Sort by quality + diversity bonus
        scored = []
        for path in paths:
            diversity_bonus = path.novelty_score * self.config.diversity_weight
            score = path.quality_score + diversity_bonus
            scored.append((path, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        return [path for path, _ in scored[: self.config.max_solutions_per_query]]

    def _check_quality(self, solution: str) -> Dict[str, Any]:
        """Check solution quality."""
        quality = self._estimate_quality(solution)

        return {
            "passes": quality >= self.config.ihsan_threshold,
            "ihsan_score": quality,
            "snr": quality,
        }

    def _estimate_quality(self, solution: str) -> float:
        """Estimate solution quality (heuristic)."""
        # Length-based heuristic (prefer substantial solutions)
        length_score = min(1.0, len(solution) / 500)

        # Coherence heuristic (sentence count)
        sentences = solution.count(".") + solution.count("!") + solution.count("?")
        coherence_score = min(1.0, sentences / 5)

        # Specificity heuristic (presence of specific terms)
        specific_terms = ["because", "therefore", "specifically", "example", "step"]
        specificity = sum(1 for t in specific_terms if t in solution.lower()) / len(
            specific_terms
        )

        return 0.4 * length_score + 0.3 * coherence_score + 0.3 * specificity

    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get discovery engine statistics."""
        return {
            "total_paths_explored": len(self._paths),
            "novelty_archive_size": self.novelty_scorer.get_archive_size(),
            "queries_processed": len(self._best_by_query),
        }
