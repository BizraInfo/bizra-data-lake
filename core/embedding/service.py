"""
Embedding Service — Tiered embedding generation with local-first fallback.

Tier 1: sentence-transformers (local GPU/CPU)
Tier 2: Ollama /api/embeddings (local inference server)
Tier 3: Raise EmbeddingUnavailableError (never return zero vectors)

Standing on Giants: Reimers & Gurevych (2019, sentence-BERT)
Artifact: core/embedding/service.py
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)


class EmbeddingUnavailableError(RuntimeError):
    """Raised when no embedding backend is available."""


@dataclass
class EmbeddingConfig:
    """Configuration for the embedding service."""

    # Tier 1: sentence-transformers model
    model_name: str = "all-MiniLM-L6-v2"

    # Tier 2: Ollama
    ollama_url: str = "http://localhost:11434"
    ollama_model: str = "nomic-embed-text"

    # Limits
    max_text_length: int = 512
    request_timeout: float = 10.0

    @classmethod
    def from_env(cls) -> EmbeddingConfig:
        """Load configuration from environment variables."""
        return cls(
            model_name=os.environ.get("BIZRA_EMBED_MODEL", cls.model_name),
            ollama_url=os.environ.get("OLLAMA_URL", cls.ollama_url),
            ollama_model=os.environ.get("BIZRA_OLLAMA_EMBED", cls.ollama_model),
        )


class EmbeddingService:
    """
    Tiered embedding generation with local-first fallback.

    INVARIANT: Never returns a zero vector.
    INVARIANT: Output dimension matches self.dimension.
    """

    def __init__(self, config: Optional[EmbeddingConfig] = None) -> None:
        self.config = config or EmbeddingConfig.from_env()
        self._model: Any = None
        self._dimension: int = 0
        self._tier: str = "none"

    def embed(self, text: str) -> list[float]:
        """
        Generate embedding vector for text input.

        Tries Tier 1 (sentence-transformers), then Tier 2 (Ollama).
        Raises EmbeddingUnavailableError if both fail.
        """
        # Truncate to max length
        if len(text) > self.config.max_text_length:
            text = text[: self.config.max_text_length]

        # Tier 1: sentence-transformers
        try:
            vec = self._embed_local(text)
            self._tier = "local"
            return vec
        except Exception as e:  # noqa: BLE001 — boundary boundary
            logger.debug(f"Tier 1 (sentence-transformers) unavailable: {e}")

        # Tier 2: Ollama
        try:
            vec = self._embed_ollama(text)
            self._tier = "ollama"
            return vec
        except Exception as e:  # noqa: BLE001 — boundary boundary
            logger.debug(f"Tier 2 (Ollama) unavailable: {e}")

        raise EmbeddingUnavailableError(
            "No embedding backend available. "
            "Install sentence-transformers or start Ollama."
        )

    def _embed_local(self, text: str) -> list[float]:
        """Tier 1: sentence-transformers (lazy-loaded)."""
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer
            except ImportError as e:
                raise ImportError("sentence-transformers not installed") from e

            import torch
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = SentenceTransformer(self.config.model_name, device=device)
            self._dimension = self._model.get_sentence_embedding_dimension()
            logger.info(f"Loaded {self.config.model_name} (dim={self._dimension})")

        vector = self._model.encode(text, normalize_embeddings=True)
        return vector.tolist()

    def _embed_ollama(self, text: str) -> list[float]:
        """Tier 2: Ollama /api/embeddings endpoint."""
        import httpx

        response = httpx.post(
            f"{self.config.ollama_url}/api/embeddings",
            json={"model": self.config.ollama_model, "prompt": text},
            timeout=self.config.request_timeout,
        )
        response.raise_for_status()
        data = response.json()
        embedding = data["embedding"]
        if self._dimension == 0:
            self._dimension = len(embedding)
        return embedding

    @property
    def dimension(self) -> int:
        """Current embedding dimension (0 if no model loaded yet)."""
        return self._dimension

    @property
    def active_tier(self) -> str:
        """Which tier was last used: 'local', 'ollama', or 'none'."""
        return self._tier
