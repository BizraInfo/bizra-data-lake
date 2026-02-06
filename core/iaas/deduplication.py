"""
Deduplication Engine — Multi-Layer Near-Duplicate Detection

Standing on Giants:
- MinHash LSH (Broder, 1997) — Jaccard similarity approximation
- SimHash (Charikar, 2002) — Hamming distance fingerprints
- SoftDeDup (He et al., 2024) — Reweighting instead of removal
- SemDeDup (Abbas et al., 2023) — Embedding-based clustering

"Redundant data negatively impacts LLM performance by reducing
 generalization ability and increasing overfitting."
"""

import hashlib
import logging
import math
import struct
from collections import defaultdict
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DeduplicationResult:
    """Result of deduplication operation."""

    original_count: int
    deduplicated_count: int
    removed_indices: Set[int]
    duplicate_groups: List[List[int]]  # Groups of duplicate indices
    method: str
    threshold: float

    @property
    def reduction_rate(self) -> float:
        return 1 - (self.deduplicated_count / max(self.original_count, 1))

    @property
    def uniqueness_rate(self) -> float:
        return self.deduplicated_count / max(self.original_count, 1)


class MinHashDeduplicator:
    """
    MinHash Locality-Sensitive Hashing for near-duplicate detection.

    Algorithm (Broder, 1997):
    1. Shingling: Convert text to n-gram set
    2. MinHash: Apply k hash functions, keep minimum hash per function
    3. LSH Banding: Group similar signatures for efficient lookup
    4. Candidate filtering: Verify Jaccard similarity for candidates

    Time complexity: O(n * k) for signature generation
    Space complexity: O(n * k) for storing signatures
    """

    def __init__(
        self,
        num_permutations: int = 128,
        threshold: float = 0.8,
        ngram_size: int = 5,
        bands: int = 16,
    ):
        self.num_permutations = num_permutations
        self.threshold = threshold
        self.ngram_size = ngram_size
        self.bands = bands
        self.rows_per_band = num_permutations // bands

        # Generate random hash function parameters
        self._max_hash = (1 << 32) - 1
        np.random.seed(42)  # Reproducibility
        self._a = np.random.randint(
            1, self._max_hash, size=num_permutations, dtype=np.uint64
        )
        self._b = np.random.randint(
            0, self._max_hash, size=num_permutations, dtype=np.uint64
        )

    def _shingle(self, text: str) -> Set[str]:
        """Convert text to character n-grams (shingles)."""
        text = text.lower().strip()
        if len(text) < self.ngram_size:
            return {text}
        return {
            text[i : i + self.ngram_size]
            for i in range(len(text) - self.ngram_size + 1)
        }

    def _minhash_signature(self, shingles: Set[str]) -> np.ndarray:
        """Compute MinHash signature for a shingle set."""
        signature = np.full(self.num_permutations, self._max_hash, dtype=np.uint64)

        for shingle in shingles:
            # Hash the shingle
            h = int(hashlib.md5(shingle.encode()).hexdigest()[:8], 16)

            # Apply all permutations
            hashes = (self._a * h + self._b) % self._max_hash
            signature = np.minimum(signature, hashes)

        return signature

    def _jaccard_from_signatures(self, sig1: np.ndarray, sig2: np.ndarray) -> float:
        """Estimate Jaccard similarity from MinHash signatures."""
        return np.mean(sig1 == sig2)

    def _lsh_hash(self, signature: np.ndarray, band_idx: int) -> int:
        """Hash a band of the signature for LSH bucketing."""
        start = band_idx * self.rows_per_band
        end = start + self.rows_per_band
        band = signature[start:end]
        return hash(tuple(band))

    def deduplicate(self, texts: List[str]) -> DeduplicationResult:
        """
        Perform MinHash-based deduplication.

        Returns indices to keep (first occurrence of each duplicate group).
        """
        n = len(texts)
        if n == 0:
            return DeduplicationResult(0, 0, set(), [], "minhash", self.threshold)

        # Step 1: Compute signatures
        logger.info(f"Computing MinHash signatures for {n} documents...")
        signatures = []
        for text in texts:
            shingles = self._shingle(text)
            sig = self._minhash_signature(shingles)
            signatures.append(sig)

        # Step 2: LSH bucketing
        logger.info(f"Performing LSH bucketing with {self.bands} bands...")
        buckets: Dict[Tuple[int, int], List[int]] = defaultdict(list)
        for idx, sig in enumerate(signatures):
            for band_idx in range(self.bands):
                bucket_key = (band_idx, self._lsh_hash(sig, band_idx))
                buckets[bucket_key].append(idx)

        # Step 3: Find candidate pairs
        candidate_pairs: Set[Tuple[int, int]] = set()
        for indices in buckets.values():
            if len(indices) > 1:
                for i in range(len(indices)):
                    for j in range(i + 1, len(indices)):
                        candidate_pairs.add(
                            (min(indices[i], indices[j]), max(indices[i], indices[j]))
                        )

        # Step 4: Verify candidates and build duplicate groups
        logger.info(f"Verifying {len(candidate_pairs)} candidate pairs...")
        union_find = list(range(n))

        def find(x: int) -> int:
            if union_find[x] != x:
                union_find[x] = find(union_find[x])
            return union_find[x]

        def union(x: int, y: int):
            px, py = find(x), find(y)
            if px != py:
                union_find[px] = py

        for i, j in candidate_pairs:
            similarity = self._jaccard_from_signatures(signatures[i], signatures[j])
            if similarity >= self.threshold:
                union(i, j)

        # Step 5: Build duplicate groups and select representatives
        groups: Dict[int, List[int]] = defaultdict(list)
        for idx in range(n):
            groups[find(idx)].append(idx)

        # Keep first occurrence from each group
        keep_indices = {min(group) for group in groups.values()}
        remove_indices = set(range(n)) - keep_indices

        duplicate_groups = [sorted(g) for g in groups.values() if len(g) > 1]

        logger.info(
            f"MinHash deduplication: {n} -> {len(keep_indices)} ({len(remove_indices)} removed)"
        )

        return DeduplicationResult(
            original_count=n,
            deduplicated_count=len(keep_indices),
            removed_indices=remove_indices,
            duplicate_groups=duplicate_groups,
            method="minhash",
            threshold=self.threshold,
        )


class SimHashDeduplicator:
    """
    SimHash fingerprinting for near-duplicate detection.

    Algorithm (Charikar, 2002):
    1. Tokenize text into weighted features (TF-IDF or equal weights)
    2. Hash each token to d-bit vector
    3. Sum weighted hash bits (+w for 1, -w for 0)
    4. Produce fingerprint by taking sign of each dimension
    5. Compare fingerprints using Hamming distance

    Advantages over MinHash:
    - Fixed-size fingerprints (64 or 128 bits)
    - Faster comparison (XOR + popcount)
    - Better for text with minor edits
    """

    def __init__(
        self,
        fingerprint_bits: int = 64,
        hamming_threshold: int = 3,
    ):
        self.fingerprint_bits = fingerprint_bits
        self.hamming_threshold = hamming_threshold

    def _tokenize(self, text: str) -> Dict[str, float]:
        """Tokenize text with equal weights."""
        text = text.lower()
        tokens = text.split()
        # Simple term frequency
        tf: Dict[str, float] = defaultdict(float)
        for token in tokens:
            tf[token] += 1.0
        return dict(tf)

    def _hash_token(self, token: str) -> int:
        """Hash a token to fingerprint_bits integer."""
        h = hashlib.md5(token.encode()).digest()
        # Take first 8 bytes for 64-bit fingerprint
        return struct.unpack("<Q", h[:8])[0] & ((1 << self.fingerprint_bits) - 1)

    def _compute_fingerprint(self, text: str) -> int:
        """Compute SimHash fingerprint for text."""
        weighted_tokens = self._tokenize(text)
        if not weighted_tokens:
            return 0

        # Initialize bit counters
        v = [0.0] * self.fingerprint_bits

        for token, weight in weighted_tokens.items():
            token_hash = self._hash_token(token)
            for i in range(self.fingerprint_bits):
                if token_hash & (1 << i):
                    v[i] += weight
                else:
                    v[i] -= weight

        # Convert to fingerprint
        fingerprint = 0
        for i in range(self.fingerprint_bits):
            if v[i] > 0:
                fingerprint |= 1 << i

        return fingerprint

    def _hamming_distance(self, fp1: int, fp2: int) -> int:
        """Compute Hamming distance between two fingerprints."""
        xor = fp1 ^ fp2
        return bin(xor).count("1")

    def deduplicate(self, texts: List[str]) -> DeduplicationResult:
        """
        Perform SimHash-based deduplication.

        Uses bit manipulation for efficient comparison.
        """
        n = len(texts)
        if n == 0:
            return DeduplicationResult(
                0, 0, set(), [], "simhash", self.hamming_threshold
            )

        # Step 1: Compute fingerprints
        logger.info(f"Computing SimHash fingerprints for {n} documents...")
        fingerprints = [self._compute_fingerprint(text) for text in texts]

        # Step 2: Find duplicates (O(n²) but fast due to bitwise ops)
        # For large n, use LSH on fingerprint bits instead
        union_find = list(range(n))

        def find(x: int) -> int:
            if union_find[x] != x:
                union_find[x] = find(union_find[x])
            return union_find[x]

        def union(x: int, y: int):
            px, py = find(x), find(y)
            if px != py:
                union_find[px] = py

        logger.info("Finding near-duplicates via Hamming distance...")
        for i in range(n):
            for j in range(i + 1, n):
                if (
                    self._hamming_distance(fingerprints[i], fingerprints[j])
                    <= self.hamming_threshold
                ):
                    union(i, j)

        # Step 3: Build groups
        groups: Dict[int, List[int]] = defaultdict(list)
        for idx in range(n):
            groups[find(idx)].append(idx)

        keep_indices = {min(group) for group in groups.values()}
        remove_indices = set(range(n)) - keep_indices
        duplicate_groups = [sorted(g) for g in groups.values() if len(g) > 1]

        logger.info(
            f"SimHash deduplication: {n} -> {len(keep_indices)} ({len(remove_indices)} removed)"
        )

        return DeduplicationResult(
            original_count=n,
            deduplicated_count=len(keep_indices),
            removed_indices=remove_indices,
            duplicate_groups=duplicate_groups,
            method="simhash",
            threshold=self.hamming_threshold,
        )


class SemanticDeduplicator:
    """
    Embedding-based semantic deduplication.

    Algorithm (SemDeDup, Abbas et al., 2023):
    1. Embed all documents using pretrained model
    2. Cluster embeddings using k-means
    3. Within each cluster, compute pairwise cosine similarities
    4. Keep only the most representative document per duplicate group

    Advantages:
    - Catches semantically similar but lexically different duplicates
    - Uses existing BIZRA embedding infrastructure (FAISS)
    """

    def __init__(
        self,
        similarity_threshold: float = 0.95,
        embedding_fn: Optional[Callable[[str], np.ndarray]] = None,
    ):
        self.similarity_threshold = similarity_threshold
        self.embedding_fn = embedding_fn

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Compute cosine similarity between two vectors."""
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))

    def deduplicate(
        self,
        texts: List[str],
        embeddings: Optional[np.ndarray] = None,
    ) -> DeduplicationResult:
        """
        Perform semantic deduplication.

        If embeddings not provided, uses embedding_fn to compute them.
        """
        n = len(texts)
        if n == 0:
            return DeduplicationResult(
                0, 0, set(), [], "semantic", self.similarity_threshold
            )

        # Get embeddings
        if embeddings is None:
            if self.embedding_fn is None:
                raise ValueError("Either embeddings or embedding_fn must be provided")
            logger.info(f"Computing embeddings for {n} documents...")
            embeddings = np.array([self.embedding_fn(text) for text in texts])

        # Find duplicates via pairwise similarity (optimized with matrix ops)
        logger.info("Computing pairwise similarities...")

        # Normalize embeddings
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized = embeddings / norms

        # Compute similarity matrix (batch for memory efficiency)
        union_find = list(range(n))

        def find(x: int) -> int:
            if union_find[x] != x:
                union_find[x] = find(union_find[x])
            return union_find[x]

        def union(x: int, y: int):
            px, py = find(x), find(y)
            if px != py:
                union_find[px] = py

        # Process in batches to avoid memory issues
        batch_size = 1000
        for i in range(0, n, batch_size):
            batch_end = min(i + batch_size, n)
            batch = normalized[i:batch_end]

            # Compute similarities for this batch against all documents
            similarities = batch @ normalized.T

            for local_idx in range(batch_end - i):
                global_idx = i + local_idx
                for j in range(global_idx + 1, n):
                    if similarities[local_idx, j] >= self.similarity_threshold:
                        union(global_idx, j)

        # Build groups
        groups: Dict[int, List[int]] = defaultdict(list)
        for idx in range(n):
            groups[find(idx)].append(idx)

        keep_indices = {min(group) for group in groups.values()}
        remove_indices = set(range(n)) - keep_indices
        duplicate_groups = [sorted(g) for g in groups.values() if len(g) > 1]

        logger.info(
            f"Semantic deduplication: {n} -> {len(keep_indices)} ({len(remove_indices)} removed)"
        )

        return DeduplicationResult(
            original_count=n,
            deduplicated_count=len(keep_indices),
            removed_indices=remove_indices,
            duplicate_groups=duplicate_groups,
            method="semantic",
            threshold=self.similarity_threshold,
        )


class SoftDeDupReweighter:
    """
    SoftDeDup: Reweighting instead of removal.

    Algorithm (He et al., 2024):
    1. Compute n-gram frequency across entire corpus
    2. Calculate "commonness" of each document
    3. Downweight common documents instead of removing

    Advantages:
    - Preserves potentially valuable rare information
    - Gradual quality degradation instead of binary decision
    - Better for training data where some repetition is acceptable
    """

    def __init__(
        self,
        ngram_size: int = 5,
        temperature: float = 1.0,
    ):
        self.ngram_size = ngram_size
        self.temperature = temperature

    def _get_ngrams(self, text: str) -> List[str]:
        """Extract n-grams from text."""
        text = text.lower()
        words = text.split()
        if len(words) < self.ngram_size:
            return [text]
        return [
            " ".join(words[i : i + self.ngram_size])
            for i in range(len(words) - self.ngram_size + 1)
        ]

    def compute_weights(self, texts: List[str]) -> np.ndarray:
        """
        Compute sample weights based on n-gram commonness.

        Returns weights in [0, 1] where lower weight = more common (duplicate-like).
        """
        n = len(texts)
        if n == 0:
            return np.array([])

        # Step 1: Compute global n-gram frequencies
        logger.info(f"Computing n-gram frequencies for {n} documents...")
        ngram_counts: Dict[str, int] = defaultdict(int)
        doc_ngrams: List[List[str]] = []

        for text in texts:
            ngrams = self._get_ngrams(text)
            doc_ngrams.append(ngrams)
            for ng in set(ngrams):  # Count unique per document
                ngram_counts[ng] += 1

        # Step 2: Compute document commonness
        logger.info("Computing document commonness scores...")
        commonness = np.zeros(n)

        for idx, ngrams in enumerate(doc_ngrams):
            if not ngrams:
                commonness[idx] = 0
                continue

            # Geometric mean of n-gram frequencies (log-space for stability)
            log_freqs = [math.log(ngram_counts[ng]) for ng in ngrams]
            commonness[idx] = math.exp(sum(log_freqs) / len(log_freqs))

        # Step 3: Convert commonness to weights (inverse relationship)
        # High commonness -> low weight
        max_commonness = max(commonness) if len(commonness) > 0 else 1.0
        if max_commonness == 0:
            return np.ones(n)

        # Softmax-style normalization with temperature
        normalized = commonness / max_commonness
        weights = 1.0 / (1.0 + np.exp(self.temperature * (normalized - 0.5)))

        logger.info(
            f"SoftDeDup weights: min={weights.min():.3f}, max={weights.max():.3f}, mean={weights.mean():.3f}"
        )

        return weights


class DeduplicationEngine:
    """
    Unified deduplication engine combining all methods.

    Pipeline:
    1. Exact hash deduplication (SHA-256)
    2. MinHash near-duplicate detection
    3. SimHash fingerprint matching
    4. Semantic embedding clustering
    5. SoftDeDup reweighting for survivors

    This multi-layer approach catches:
    - Exact duplicates (Layer 1)
    - Near-duplicates with minor edits (Layer 2-3)
    - Semantically equivalent content (Layer 4)
    - Provides quality weights for remaining (Layer 5)
    """

    def __init__(
        self,
        enable_exact: bool = True,
        enable_minhash: bool = True,
        enable_simhash: bool = True,
        enable_semantic: bool = True,
        enable_softdedup: bool = True,
        minhash_threshold: float = 0.8,
        simhash_threshold: int = 3,
        semantic_threshold: float = 0.95,
        embedding_fn: Optional[Callable[[str], np.ndarray]] = None,
    ):
        self.enable_exact = enable_exact
        self.enable_minhash = enable_minhash
        self.enable_simhash = enable_simhash
        self.enable_semantic = enable_semantic
        self.enable_softdedup = enable_softdedup

        self.minhash = (
            MinHashDeduplicator(threshold=minhash_threshold) if enable_minhash else None
        )
        self.simhash = (
            SimHashDeduplicator(hamming_threshold=simhash_threshold)
            if enable_simhash
            else None
        )
        self.semantic = (
            SemanticDeduplicator(
                similarity_threshold=semantic_threshold, embedding_fn=embedding_fn
            )
            if enable_semantic
            else None
        )
        self.softdedup = SoftDeDupReweighter() if enable_softdedup else None

    def _exact_dedup(self, texts: List[str]) -> DeduplicationResult:
        """Exact SHA-256 hash deduplication."""
        n = len(texts)
        seen: Dict[str, int] = {}
        keep_indices = set()
        duplicate_groups: Dict[str, List[int]] = defaultdict(list)

        for idx, text in enumerate(texts):
            h = hashlib.sha256(text.encode()).hexdigest()
            if h not in seen:
                seen[h] = idx
                keep_indices.add(idx)
            duplicate_groups[h].append(idx)

        remove_indices = set(range(n)) - keep_indices
        groups = [sorted(g) for g in duplicate_groups.values() if len(g) > 1]

        logger.info(
            f"Exact deduplication: {n} -> {len(keep_indices)} ({len(remove_indices)} removed)"
        )

        return DeduplicationResult(
            original_count=n,
            deduplicated_count=len(keep_indices),
            removed_indices=remove_indices,
            duplicate_groups=groups,
            method="exact",
            threshold=1.0,
        )

    def deduplicate(
        self,
        texts: List[str],
        embeddings: Optional[np.ndarray] = None,
    ) -> Tuple[List[str], np.ndarray, Dict[str, DeduplicationResult]]:
        """
        Run full deduplication pipeline.

        Returns:
            - Deduplicated texts
            - Quality weights for each text
            - Results from each deduplication layer
        """
        results: Dict[str, DeduplicationResult] = {}
        current_texts = texts.copy()
        current_indices = list(range(len(texts)))  # Map to original indices
        current_embeddings = embeddings.copy() if embeddings is not None else None

        # Layer 1: Exact deduplication
        if self.enable_exact:
            result = self._exact_dedup(current_texts)
            results["exact"] = result

            keep_mask = [
                i not in result.removed_indices for i in range(len(current_texts))
            ]
            current_texts = [t for t, keep in zip(current_texts, keep_mask) if keep]
            current_indices = [
                idx for idx, keep in zip(current_indices, keep_mask) if keep
            ]
            if current_embeddings is not None:
                current_embeddings = current_embeddings[keep_mask]

        # Layer 2: MinHash
        if self.enable_minhash and self.minhash and len(current_texts) > 1:
            result = self.minhash.deduplicate(current_texts)
            results["minhash"] = result

            keep_mask = [
                i not in result.removed_indices for i in range(len(current_texts))
            ]
            current_texts = [t for t, keep in zip(current_texts, keep_mask) if keep]
            current_indices = [
                idx for idx, keep in zip(current_indices, keep_mask) if keep
            ]
            if current_embeddings is not None:
                current_embeddings = current_embeddings[keep_mask]

        # Layer 3: SimHash
        if self.enable_simhash and self.simhash and len(current_texts) > 1:
            result = self.simhash.deduplicate(current_texts)
            results["simhash"] = result

            keep_mask = [
                i not in result.removed_indices for i in range(len(current_texts))
            ]
            current_texts = [t for t, keep in zip(current_texts, keep_mask) if keep]
            current_indices = [
                idx for idx, keep in zip(current_indices, keep_mask) if keep
            ]
            if current_embeddings is not None:
                current_embeddings = current_embeddings[keep_mask]

        # Layer 4: Semantic
        if self.enable_semantic and self.semantic and len(current_texts) > 1:
            result = self.semantic.deduplicate(current_texts, current_embeddings)
            results["semantic"] = result

            keep_mask = [
                i not in result.removed_indices for i in range(len(current_texts))
            ]
            current_texts = [t for t, keep in zip(current_texts, keep_mask) if keep]
            current_indices = [
                idx for idx, keep in zip(current_indices, keep_mask) if keep
            ]
            if current_embeddings is not None:
                current_embeddings = current_embeddings[keep_mask]

        # Layer 5: SoftDeDup weights
        if self.enable_softdedup and self.softdedup:
            weights = self.softdedup.compute_weights(current_texts)
        else:
            weights = np.ones(len(current_texts))

        logger.info(
            f"Full deduplication pipeline: {len(texts)} -> {len(current_texts)}"
        )
        logger.info(f"  Reduction rate: {1 - len(current_texts)/len(texts):.2%}")

        return current_texts, weights, results
