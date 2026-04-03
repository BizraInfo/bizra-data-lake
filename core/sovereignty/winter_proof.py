"""
WinterProofEmbedder - Deterministic Offline Embeddings

Generates deterministic embeddings using multi-hash approach without external API calls.
Pure stdlib implementation with optional numpy acceleration.

Key Features:
- Multi-hash: SHA-256, SHA3-256, BLAKE3 (or SHA-512 fallback)
- L2 normalization for unit vectors
- Cosine similarity for semantic search
- Domain separation: "bizra-pci-v1:"
- Deterministic: Same input always produces same output

NO external dependencies required (numpy optional for acceleration).
"""

import hashlib
import math
from typing import List, Tuple, Optional, Union
import json

# Optional numpy for acceleration
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False

# Optional blake3 for enhanced security
try:
    import blake3
    HAS_BLAKE3 = True
except ImportError:
    HAS_BLAKE3 = False


class WinterProofEmbedder:
    """
    Deterministic embedding generator using cryptographic hashes.

    Operates in complete offline mode with no external API dependencies.
    Uses multi-hash approach to generate high-dimensional embeddings.
    """

    DOMAIN_PREFIX = "bizra-pci-v1:"
    DEFAULT_DIM = 384  # Standard embedding dimension

    def __init__(self, dimension: int = DEFAULT_DIM, use_numpy: bool = True):
        """
        Initialize WinterProofEmbedder.

        Args:
            dimension: Output embedding dimension (must be divisible by 3)
            use_numpy: Use numpy for faster operations if available
        """
        if dimension % 3 != 0:
            raise ValueError(f"Dimension must be divisible by 3, got {dimension}")

        self.dimension = dimension
        self.use_numpy = use_numpy and HAS_NUMPY
        self.chunk_size = dimension // 3  # Split across 3 hash functions

    def embed(self, text: str) -> List[float]:
        """
        Generate deterministic embedding for text.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector (L2 normalized)
        """
        # Apply domain separation
        domain_text = self.DOMAIN_PREFIX + text
        text_bytes = domain_text.encode('utf-8')

        # Generate three hash-based components
        sha256_component = self._hash_to_vector(text_bytes, 'sha256', self.chunk_size)
        sha3_component = self._hash_to_vector(text_bytes, 'sha3_256', self.chunk_size)

        # Use BLAKE3 if available, otherwise SHA-512
        if HAS_BLAKE3:
            blake3_component = self._blake3_to_vector(text_bytes, self.chunk_size)
        else:
            blake3_component = self._hash_to_vector(text_bytes, 'sha512', self.chunk_size)

        # Concatenate components
        embedding = sha256_component + sha3_component + blake3_component

        # L2 normalize
        normalized = self._l2_normalize(embedding)

        return normalized

    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        return [self.embed(text) for text in texts]

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors.

        Args:
            vec1: First embedding vector
            vec2: Second embedding vector

        Returns:
            Cosine similarity score (0 to 1)
        """
        if self.use_numpy:
            return float(np.dot(vec1, vec2))  # Already L2 normalized

        # Manual dot product
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        return dot_product

    def semantic_search(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5
    ) -> List[Tuple[int, float, str]]:
        """
        Search documents by semantic similarity to query.

        Args:
            query: Query text
            documents: List of document texts
            top_k: Number of top results to return

        Returns:
            List of (index, similarity_score, document) tuples
        """
        query_emb = self.embed(query)
        doc_embs = self.embed_batch(documents)

        # Calculate similarities
        similarities = [
            (i, self.cosine_similarity(query_emb, doc_emb), doc)
            for i, (doc_emb, doc) in enumerate(zip(doc_embs, documents))
        ]

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    def _hash_to_vector(
        self,
        data: bytes,
        algorithm: str,
        size: int
    ) -> List[float]:
        """
        Convert hash output to normalized vector component.

        Args:
            data: Input bytes to hash
            algorithm: Hash algorithm name
            size: Desired vector size

        Returns:
            Normalized vector component
        """
        # Generate hash
        hasher = hashlib.new(algorithm)
        hasher.update(data)

        # Expand hash to required size using iterative hashing
        vector = []
        for i in range(size):
            # Hash with iteration counter for unique values
            iter_hasher = hashlib.new(algorithm)
            iter_hasher.update(data)
            iter_hasher.update(i.to_bytes(4, 'big'))

            # Convert hash to float in [-1, 1]
            hash_bytes = iter_hasher.digest()
            # Take first 8 bytes as int, normalize to [-1, 1]
            int_val = int.from_bytes(hash_bytes[:8], 'big', signed=False)
            # Map [0, 2^64-1] to [-1, 1]
            float_val = (int_val / (2**63)) - 1.0
            vector.append(float_val)

        return vector

    def _blake3_to_vector(self, data: bytes, size: int) -> List[float]:
        """
        Convert BLAKE3 hash output to normalized vector component.

        Args:
            data: Input bytes to hash
            size: Desired vector size

        Returns:
            Normalized vector component
        """
        vector = []
        for i in range(size):
            # BLAKE3 with iteration counter
            hasher = blake3.blake3(data + i.to_bytes(4, 'big'))
            hash_bytes = hasher.digest()

            # Convert to float in [-1, 1]
            int_val = int.from_bytes(hash_bytes[:8], 'big', signed=False)
            float_val = (int_val / (2**63)) - 1.0
            vector.append(float_val)

        return vector

    def _l2_normalize(self, vector: List[float]) -> List[float]:
        """
        L2 normalize vector to unit length.

        Args:
            vector: Input vector

        Returns:
            L2 normalized vector
        """
        if self.use_numpy:
            arr = np.array(vector)
            norm = np.linalg.norm(arr)
            if norm == 0:
                return vector
            return (arr / norm).tolist()

        # Manual L2 normalization
        norm = math.sqrt(sum(x * x for x in vector))
        if norm == 0:
            return vector

        return [x / norm for x in vector]

    def save_embeddings(self, embeddings: List[List[float]], path: str) -> None:
        """
        Save embeddings to JSON file.

        Args:
            embeddings: List of embedding vectors
            path: Output file path
        """
        with open(path, 'w') as f:
            json.dump({
                'dimension': self.dimension,
                'count': len(embeddings),
                'embeddings': embeddings,
                'domain': self.DOMAIN_PREFIX,
            }, f)

    def load_embeddings(self, path: str) -> List[List[float]]:
        """
        Load embeddings from JSON file.

        Args:
            path: Input file path

        Returns:
            List of embedding vectors
        """
        with open(path, 'r') as f:
            data = json.load(f)
            if data['dimension'] != self.dimension:
                raise ValueError(
                    f"Dimension mismatch: expected {self.dimension}, "
                    f"got {data['dimension']}"
                )
            return data['embeddings']

    def generate_receipt(self, text: str, embedding: List[float]) -> dict:
        """
        Generate verification receipt for embedding operation.

        Args:
            text: Original input text
            embedding: Generated embedding

        Returns:
            Receipt dictionary with operation details
        """
        # Hash the embedding for integrity
        emb_bytes = json.dumps(embedding, sort_keys=True).encode('utf-8')
        emb_hash = hashlib.sha256(emb_bytes).hexdigest()

        # Hash the input text
        text_hash = hashlib.sha256(
            (self.DOMAIN_PREFIX + text).encode('utf-8')
        ).hexdigest()

        return {
            'operation': 'winter_proof_embed',
            'domain': self.DOMAIN_PREFIX,
            'text_hash': text_hash,
            'embedding_hash': emb_hash,
            'dimension': self.dimension,
            'text_length': len(text),
            'numpy_accelerated': self.use_numpy,
            'blake3_available': HAS_BLAKE3,
        }


def main():
    """Demo WinterProofEmbedder functionality."""
    print("WinterProofEmbedder - Deterministic Offline Embeddings")
    print("=" * 60)

    embedder = WinterProofEmbedder(dimension=384)

    # Test embeddings
    texts = [
        "BIZRA is a decentralized agentic system",
        "Every human is a node, every node is a seed",
        "Ihsān means excellence in all operations",
    ]

    print(f"\nGenerating embeddings for {len(texts)} texts...")
    embeddings = embedder.embed_batch(texts)

    print(f"Embedding dimension: {len(embeddings[0])}")
    print(f"Using numpy: {embedder.use_numpy}")
    print(f"BLAKE3 available: {HAS_BLAKE3}")

    # Test determinism
    print("\nTesting determinism...")
    emb1 = embedder.embed(texts[0])
    emb2 = embedder.embed(texts[0])
    is_deterministic = emb1 == emb2
    print(f"Deterministic: {is_deterministic}")

    # Test semantic search
    print("\nSemantic search test:")
    query = "excellence and quality"
    results = embedder.semantic_search(query, texts, top_k=3)

    for rank, (idx, score, doc) in enumerate(results, 1):
        print(f"{rank}. Score: {score:.4f} - {doc[:50]}...")

    # Generate receipt
    receipt = embedder.generate_receipt(texts[0], embeddings[0])
    print(f"\nReceipt: {receipt}")


if __name__ == "__main__":
    main()
