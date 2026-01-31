"""
Quality Filtering Engine — Perplexity, IFD, and Cluster Complexity

Standing on Giants:
- Perplexity Filtering (Ankner et al., 2024)
- Learning Percentage (Mekala et al., 2024)
- Instruction-Following Difficulty (Li et al., 2023)
- Cluster Complexity (Abbas et al., 2024)

"Data filtering aims to remove low-quality or sensitive samples,
 reducing computational overhead while maintaining performance."
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Tuple
from collections import defaultdict
import logging

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class FilterResult:
    """Result of filtering operation."""
    original_count: int
    filtered_count: int
    removed_indices: List[int]
    scores: Dict[int, float]  # Index -> quality score
    method: str
    threshold: float

    @property
    def retention_rate(self) -> float:
        return self.filtered_count / max(self.original_count, 1)


class PerplexityFilter:
    """
    Perplexity-based quality filtering.

    Perplexity measures how "surprised" a language model is by the text.
    Lower perplexity = more predictable = potentially higher quality.

    Formula: PPL(x) = exp(-1/n * Σ log(p(xi | x1...xi-1)))

    DATA4LLM insight: Medium-high perplexity samples are often most valuable
    (too low = repetitive, too high = noisy).
    """

    def __init__(
        self,
        min_perplexity: float = 10.0,
        max_perplexity: float = 100.0,
        perplexity_fn: Optional[Callable[[str], float]] = None,
    ):
        self.min_perplexity = min_perplexity
        self.max_perplexity = max_perplexity
        self.perplexity_fn = perplexity_fn

    def _estimate_perplexity(self, text: str) -> float:
        """
        Estimate perplexity using heuristics when LLM not available.

        Uses vocabulary diversity and sentence structure as proxies.
        """
        if not text.strip():
            return float('inf')

        words = text.lower().split()
        if len(words) < 3:
            return float('inf')

        # Vocabulary diversity (type-token ratio)
        unique_words = len(set(words))
        ttr = unique_words / len(words)

        # Sentence structure (punctuation density)
        punctuation = sum(1 for c in text if c in '.!?;:')
        punct_density = punctuation / len(words) if words else 0

        # Heuristic perplexity estimate
        # High TTR + moderate punctuation = well-structured text
        estimated_ppl = 50 * (1 / (ttr + 0.1)) * (1 + abs(punct_density - 0.1))

        return max(1.0, min(estimated_ppl, 500.0))

    def compute_perplexity(self, texts: List[str]) -> np.ndarray:
        """Compute perplexity for all texts."""
        if self.perplexity_fn:
            return np.array([self.perplexity_fn(t) for t in texts])
        return np.array([self._estimate_perplexity(t) for t in texts])

    def filter(self, texts: List[str]) -> FilterResult:
        """
        Filter texts by perplexity.

        Keeps texts with perplexity in [min_perplexity, max_perplexity].
        """
        n = len(texts)
        if n == 0:
            return FilterResult(0, 0, [], {}, "perplexity", self.max_perplexity)

        logger.info(f"Computing perplexity for {n} texts...")
        perplexities = self.compute_perplexity(texts)

        # Filter by bounds
        keep_mask = (perplexities >= self.min_perplexity) & (perplexities <= self.max_perplexity)
        removed_indices = [i for i, keep in enumerate(keep_mask) if not keep]

        # Quality score: inverse perplexity (normalized)
        scores = {}
        for i, ppl in enumerate(perplexities):
            if keep_mask[i]:
                # Higher score for medium perplexity
                optimal_ppl = (self.min_perplexity + self.max_perplexity) / 2
                distance_from_optimal = abs(ppl - optimal_ppl) / (self.max_perplexity - self.min_perplexity)
                scores[i] = max(0, 1 - distance_from_optimal)

        logger.info(f"Perplexity filter: {n} -> {sum(keep_mask)} ({len(removed_indices)} removed)")

        return FilterResult(
            original_count=n,
            filtered_count=sum(keep_mask),
            removed_indices=removed_indices,
            scores=scores,
            method="perplexity",
            threshold=self.max_perplexity,
        )


class InstructionFollowingDifficultyFilter:
    """
    Instruction-Following Difficulty (IFD) filtering.

    Measures how much the instruction affects the model's ability to generate the response.

    Formula: IFD(i, r) = PPL(r|i) / PPL(r)

    High IFD = instruction is crucial for the response = high-quality instruction data.
    Low IFD = response is generic regardless of instruction = low-quality.
    """

    def __init__(
        self,
        min_ifd: float = 0.5,
        conditional_ppl_fn: Optional[Callable[[str, str], float]] = None,
        unconditional_ppl_fn: Optional[Callable[[str], float]] = None,
    ):
        self.min_ifd = min_ifd
        self.conditional_ppl_fn = conditional_ppl_fn
        self.unconditional_ppl_fn = unconditional_ppl_fn

    def _estimate_ifd(self, instruction: str, response: str) -> float:
        """Estimate IFD when LLM functions not available."""
        if not instruction.strip() or not response.strip():
            return 0.0

        # Heuristic: Check instruction-response relevance
        instruction_words = set(instruction.lower().split())
        response_words = set(response.lower().split())

        # Overlap indicates instruction influence
        overlap = len(instruction_words & response_words)
        relevance = overlap / max(len(instruction_words), 1)

        # Length ratio
        length_ratio = min(len(response) / max(len(instruction), 1), 10) / 10

        # Response specificity (unique words)
        specificity = len(response_words) / max(len(response.split()), 1)

        # Combine into IFD estimate
        ifd = 0.3 * relevance + 0.4 * length_ratio + 0.3 * specificity

        return min(max(ifd, 0.0), 1.0)

    def compute_ifd(
        self,
        instructions: List[str],
        responses: List[str],
    ) -> np.ndarray:
        """Compute IFD scores for instruction-response pairs."""
        if self.conditional_ppl_fn and self.unconditional_ppl_fn:
            ifds = []
            for inst, resp in zip(instructions, responses):
                ppl_conditional = self.conditional_ppl_fn(inst, resp)
                ppl_unconditional = self.unconditional_ppl_fn(resp)
                if ppl_unconditional > 0:
                    ifd = ppl_conditional / ppl_unconditional
                else:
                    ifd = 0.0
                ifds.append(ifd)
            return np.array(ifds)

        return np.array([
            self._estimate_ifd(inst, resp)
            for inst, resp in zip(instructions, responses)
        ])

    def filter(
        self,
        instructions: List[str],
        responses: List[str],
    ) -> FilterResult:
        """Filter instruction-response pairs by IFD score."""
        n = len(instructions)
        if n == 0:
            return FilterResult(0, 0, [], {}, "ifd", self.min_ifd)

        logger.info(f"Computing IFD for {n} instruction-response pairs...")
        ifd_scores = self.compute_ifd(instructions, responses)

        # Keep high-IFD samples
        keep_mask = ifd_scores >= self.min_ifd
        removed_indices = [i for i, keep in enumerate(keep_mask) if not keep]

        scores = {i: float(ifd_scores[i]) for i in range(n) if keep_mask[i]}

        logger.info(f"IFD filter: {n} -> {sum(keep_mask)} ({len(removed_indices)} removed)")

        return FilterResult(
            original_count=n,
            filtered_count=sum(keep_mask),
            removed_indices=removed_indices,
            scores=scores,
            method="ifd",
            threshold=self.min_ifd,
        )


class ClusterComplexityFilter:
    """
    Cluster-based diversity filtering.

    Algorithm (Abbas et al., 2024):
    1. Embed and cluster documents
    2. Compute cluster complexity = intra_distance × inter_distance
    3. Resample to balance diversity (high complexity) and quality (low intra)

    High complexity clusters contain diverse, informative content.
    Low complexity clusters are repetitive.
    """

    def __init__(
        self,
        n_clusters: int = 100,
        min_cluster_complexity: float = 0.3,
        resampling_temperature: float = 1.0,
    ):
        self.n_clusters = n_clusters
        self.min_cluster_complexity = min_cluster_complexity
        self.resampling_temperature = resampling_temperature

    def _simple_kmeans(
        self,
        embeddings: np.ndarray,
        n_clusters: int,
        max_iter: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simple k-means clustering (no sklearn dependency)."""
        n, d = embeddings.shape
        n_clusters = min(n_clusters, n)

        # Initialize centroids randomly
        indices = np.random.choice(n, n_clusters, replace=False)
        centroids = embeddings[indices].copy()

        labels = np.zeros(n, dtype=int)

        for _ in range(max_iter):
            # Assign to nearest centroid
            distances = np.zeros((n, n_clusters))
            for k in range(n_clusters):
                diff = embeddings - centroids[k]
                distances[:, k] = np.sum(diff ** 2, axis=1)

            new_labels = np.argmin(distances, axis=1)

            # Check convergence
            if np.array_equal(labels, new_labels):
                break
            labels = new_labels

            # Update centroids
            for k in range(n_clusters):
                mask = labels == k
                if np.sum(mask) > 0:
                    centroids[k] = embeddings[mask].mean(axis=0)

        return labels, centroids

    def compute_cluster_complexity(
        self,
        embeddings: np.ndarray,
        labels: np.ndarray,
        centroids: np.ndarray,
    ) -> Dict[int, float]:
        """Compute complexity score for each cluster."""
        n_clusters = len(centroids)
        complexities = {}

        for k in range(n_clusters):
            mask = labels == k
            cluster_points = embeddings[mask]

            if len(cluster_points) < 2:
                complexities[k] = 0.0
                continue

            # Intra-cluster distance (average distance to centroid)
            intra_distances = np.sqrt(np.sum((cluster_points - centroids[k]) ** 2, axis=1))
            avg_intra = np.mean(intra_distances)

            # Inter-cluster distance (distance to other centroids)
            other_centroids = np.array([c for i, c in enumerate(centroids) if i != k])
            if len(other_centroids) > 0:
                inter_distances = np.sqrt(np.sum((centroids[k] - other_centroids) ** 2, axis=1))
                avg_inter = np.mean(inter_distances)
            else:
                avg_inter = 1.0

            # Complexity = intra × inter (high when spread out AND distinct)
            complexities[k] = avg_intra * avg_inter

        # Normalize to [0, 1]
        if complexities:
            max_complexity = max(complexities.values())
            if max_complexity > 0:
                complexities = {k: v / max_complexity for k, v in complexities.items()}

        return complexities

    def filter(
        self,
        texts: List[str],
        embeddings: np.ndarray,
    ) -> FilterResult:
        """
        Filter by cluster complexity.

        Removes samples from low-complexity (repetitive) clusters.
        """
        n = len(texts)
        if n == 0:
            return FilterResult(0, 0, [], {}, "cluster_complexity", self.min_cluster_complexity)

        logger.info(f"Clustering {n} documents into {self.n_clusters} clusters...")
        labels, centroids = self._simple_kmeans(embeddings, min(self.n_clusters, n))

        logger.info("Computing cluster complexities...")
        complexities = self.compute_cluster_complexity(embeddings, labels, centroids)

        # Assign complexity score to each sample
        sample_scores = {i: complexities.get(labels[i], 0.0) for i in range(n)}

        # Filter low-complexity samples
        keep_mask = np.array([sample_scores[i] >= self.min_cluster_complexity for i in range(n)])
        removed_indices = [i for i, keep in enumerate(keep_mask) if not keep]

        scores = {i: sample_scores[i] for i in range(n) if keep_mask[i]}

        logger.info(f"Cluster complexity filter: {n} -> {sum(keep_mask)} ({len(removed_indices)} removed)")

        return FilterResult(
            original_count=n,
            filtered_count=sum(keep_mask),
            removed_indices=removed_indices,
            scores=scores,
            method="cluster_complexity",
            threshold=self.min_cluster_complexity,
        )


class QualityFilter:
    """
    Unified quality filter combining all methods.

    Pipeline:
    1. Perplexity filtering (fluency)
    2. IFD filtering (instruction quality)
    3. Cluster complexity filtering (diversity)

    Outputs quality scores for SNR calculation.
    """

    def __init__(
        self,
        enable_perplexity: bool = True,
        enable_ifd: bool = True,
        enable_cluster: bool = True,
        perplexity_bounds: Tuple[float, float] = (10.0, 100.0),
        min_ifd: float = 0.5,
        min_cluster_complexity: float = 0.3,
        perplexity_fn: Optional[Callable[[str], float]] = None,
    ):
        self.enable_perplexity = enable_perplexity
        self.enable_ifd = enable_ifd
        self.enable_cluster = enable_cluster

        self.perplexity_filter = PerplexityFilter(
            min_perplexity=perplexity_bounds[0],
            max_perplexity=perplexity_bounds[1],
            perplexity_fn=perplexity_fn,
        ) if enable_perplexity else None

        self.ifd_filter = InstructionFollowingDifficultyFilter(
            min_ifd=min_ifd,
        ) if enable_ifd else None

        self.cluster_filter = ClusterComplexityFilter(
            min_cluster_complexity=min_cluster_complexity,
        ) if enable_cluster else None

    def filter(
        self,
        texts: List[str],
        embeddings: Optional[np.ndarray] = None,
        instructions: Optional[List[str]] = None,
        responses: Optional[List[str]] = None,
    ) -> Tuple[List[int], Dict[int, float], Dict[str, FilterResult]]:
        """
        Run quality filtering pipeline.

        Returns:
            - Indices to keep
            - Quality scores for each kept index
            - Results from each filter
        """
        n = len(texts)
        keep_set = set(range(n))
        all_scores: Dict[int, List[float]] = defaultdict(list)
        results: Dict[str, FilterResult] = {}

        # Perplexity filter
        if self.enable_perplexity and self.perplexity_filter:
            result = self.perplexity_filter.filter(texts)
            results["perplexity"] = result
            keep_set &= (set(range(n)) - set(result.removed_indices))
            for idx, score in result.scores.items():
                all_scores[idx].append(score)

        # IFD filter (if instruction-response data provided)
        if self.enable_ifd and self.ifd_filter and instructions and responses:
            result = self.ifd_filter.filter(instructions, responses)
            results["ifd"] = result
            keep_set &= (set(range(n)) - set(result.removed_indices))
            for idx, score in result.scores.items():
                all_scores[idx].append(score)

        # Cluster complexity filter
        if self.enable_cluster and self.cluster_filter and embeddings is not None:
            result = self.cluster_filter.filter(texts, embeddings)
            results["cluster_complexity"] = result
            keep_set &= (set(range(n)) - set(result.removed_indices))
            for idx, score in result.scores.items():
                all_scores[idx].append(score)

        # Aggregate scores (geometric mean for composite quality)
        final_scores = {}
        for idx in keep_set:
            scores_list = all_scores.get(idx, [1.0])
            if scores_list:
                final_scores[idx] = math.exp(sum(math.log(max(s, 1e-10)) for s in scores_list) / len(scores_list))
            else:
                final_scores[idx] = 1.0

        logger.info(f"Quality filtering pipeline: {n} -> {len(keep_set)}")

        return sorted(keep_set), final_scores, results
