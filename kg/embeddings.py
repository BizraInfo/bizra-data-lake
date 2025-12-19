"""
kg/embeddings.py — Sovereignty-by-default embedding interface

Provides pluggable embedding backends with NO network egress by default.
Network-based providers (OpenAI, etc.) are opt-in and require explicit policy.

Usage:
    from kg.embeddings import get_embedder, EmbeddingResult
    
    embedder = get_embedder()  # Returns NullEmbedder by default
    result = embedder.embed("some text")
"""

from __future__ import annotations

import hashlib
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional, Protocol, runtime_checkable


# Default embedding dimension (matches sentence-transformers/all-MiniLM-L6-v2)
DEFAULT_DIMS = 768


@dataclass(frozen=True)
class EmbeddingResult:
    """Result from an embedding operation."""
    vector: List[float]
    model: str
    dims: int
    
    def __post_init__(self):
        if len(self.vector) != self.dims:
            raise ValueError(f"Vector length {len(self.vector)} != declared dims {self.dims}")


@runtime_checkable
class Embedder(Protocol):
    """Protocol for embedding providers."""
    
    @abstractmethod
    def embed(self, text: str) -> EmbeddingResult:
        """Generate embedding for text."""
        ...
    
    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        """Generate embeddings for multiple texts."""
        ...
    
    @property
    @abstractmethod
    def model_name(self) -> str:
        """Return the model identifier."""
        ...
    
    @property
    @abstractmethod
    def dims(self) -> int:
        """Return the embedding dimension."""
        ...


class NullEmbedder:
    """
    Dev-safe embedder: produces deterministic zero vectors.
    
    Use this when:
    - No embedding model is configured
    - Testing without ML dependencies
    - Forcing graph/text-only retrieval
    
    The vector is deterministic based on text hash, enabling consistent testing.
    """
    
    def __init__(self, dims: int = DEFAULT_DIMS):
        self._dims = dims
    
    @property
    def model_name(self) -> str:
        return "null"
    
    @property
    def dims(self) -> int:
        return self._dims
    
    def embed(self, text: str) -> EmbeddingResult:
        # Deterministic vector based on text hash (for consistent testing)
        text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        # Use hash bytes to seed a simple pattern
        seed = int(text_hash[:8], 16) / (16**8)
        vector = [seed * 0.01] * self._dims
        return EmbeddingResult(vector=vector, model=self.model_name, dims=self._dims)
    
    def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        return [self.embed(t) for t in texts]


class LocalEmbedder:
    """
    Local embedding using sentence-transformers (no network egress).
    
    Requires: pip install sentence-transformers
    
    Recommended models (sovereignty-safe):
    - all-MiniLM-L6-v2 (384 dims, fast)
    - all-mpnet-base-v2 (768 dims, better quality)
    - e5-small-v2 (384 dims, good for retrieval)
    """
    
    def __init__(self, model_name: str = "all-mpnet-base-v2"):
        self._model_name = model_name
        self._model = None
        self._dims: Optional[int] = None
    
    def _ensure_loaded(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._model = SentenceTransformer(self._model_name)
                self._dims = self._model.get_sentence_embedding_dimension()
            except ImportError:
                raise RuntimeError(
                    "sentence-transformers not installed. "
                    "Run: pip install sentence-transformers"
                )
    
    @property
    def model_name(self) -> str:
        return self._model_name
    
    @property
    def dims(self) -> int:
        self._ensure_loaded()
        return self._dims or DEFAULT_DIMS
    
    def embed(self, text: str) -> EmbeddingResult:
        self._ensure_loaded()
        vector = self._model.encode(text, convert_to_numpy=True).tolist()
        return EmbeddingResult(vector=vector, model=self._model_name, dims=len(vector))
    
    def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        self._ensure_loaded()
        vectors = self._model.encode(texts, convert_to_numpy=True)
        return [
            EmbeddingResult(vector=v.tolist(), model=self._model_name, dims=len(v))
            for v in vectors
        ]


class OpenAIEmbedder:
    """
    OpenAI embedding API (requires network + API key).
    
    ⚠️ NETWORK EGRESS: This sends data to OpenAI servers.
    Only use when policy explicitly allows external embedding.
    
    Requires: pip install openai
    """
    
    def __init__(
        self,
        model: str = "text-embedding-3-small",
        api_key: Optional[str] = None
    ):
        self._model = model
        self._api_key = api_key or os.environ.get("OPENAI_API_KEY")
        self._client = None
        
        # Known dimensions
        self._dims_map = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }
    
    def _ensure_client(self):
        if self._client is None:
            if not self._api_key:
                raise RuntimeError("OPENAI_API_KEY not set")
            try:
                from openai import OpenAI
                self._client = OpenAI(api_key=self._api_key)
            except ImportError:
                raise RuntimeError("openai not installed. Run: pip install openai")
    
    @property
    def model_name(self) -> str:
        return self._model
    
    @property
    def dims(self) -> int:
        return self._dims_map.get(self._model, 1536)
    
    def embed(self, text: str) -> EmbeddingResult:
        self._ensure_client()
        response = self._client.embeddings.create(
            model=self._model,
            input=text
        )
        vector = response.data[0].embedding
        return EmbeddingResult(vector=vector, model=self._model, dims=len(vector))
    
    def embed_batch(self, texts: List[str]) -> List[EmbeddingResult]:
        self._ensure_client()
        response = self._client.embeddings.create(
            model=self._model,
            input=texts
        )
        return [
            EmbeddingResult(
                vector=item.embedding,
                model=self._model,
                dims=len(item.embedding)
            )
            for item in response.data
        ]


# ══════════════════════════════════════════════════════════════════════════════
# FACTORY
# ══════════════════════════════════════════════════════════════════════════════

def get_embedder(
    provider: Optional[str] = None,
    model: Optional[str] = None
) -> Embedder:
    """
    Get an embedder instance based on configuration.
    
    Environment variables:
    - BIZRA_EMBEDDER: "null" | "local" | "openai" (default: "null")
    - BIZRA_EMBEDDER_MODEL: model name override
    
    Args:
        provider: Override provider selection
        model: Override model selection
    
    Returns:
        Configured Embedder instance
    """
    provider = provider or os.environ.get("BIZRA_EMBEDDER", "null")
    model = model or os.environ.get("BIZRA_EMBEDDER_MODEL")
    
    if provider == "null":
        return NullEmbedder()
    
    elif provider == "local":
        return LocalEmbedder(model_name=model or "all-mpnet-base-v2")
    
    elif provider == "openai":
        return OpenAIEmbedder(model=model or "text-embedding-3-small")
    
    else:
        raise ValueError(f"Unknown embedder provider: {provider}")
