"""
Embedding Pipeline — Lazy-loaded vector embedding with tiered fallback.

Priority:
  1. sentence-transformers (local, fast, no network)
  2. Ollama API (local server)
  3. None (records stored without vectors)

Usage:
    from core.memory.embedding import EmbeddingPipeline
    pipeline = EmbeddingPipeline(config)
    vec = pipeline.embed("The Earth orbits the Sun")

Standing on Giants: Reimers & Gurevych (2019) — Sentence-BERT
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Callable, List, Optional

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from .config import MemoryConfig


class EmbeddingPipeline:
    """Lazy-loaded embedding provider with tiered fallback.

    The model is loaded on first use, not at construction time.
    GPU OOM during batch embed triggers automatic CPU fallback.
    """

    def __init__(self, config: MemoryConfig) -> None:
        self._config = config
        self._model = None
        self._backend: str = "none"
        self._loaded = False
        self._load_attempted = False

    @property
    def backend(self) -> str:
        return self._backend

    @property
    def loaded(self) -> bool:
        return self._loaded

    def _ensure_loaded(self) -> None:
        """Lazy-load on first use. Only attempts once."""
        if self._loaded or self._load_attempted:
            return
        self._load_attempted = True

        if not getattr(self._config, "auto_embed", True):
            return

        # Attempt 1: sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer

            device = getattr(self._config, "embed_device", "cpu")
            if device == "auto":
                try:
                    import torch

                    device = "cuda" if torch.cuda.is_available() else "cpu"
                except ImportError:
                    device = "cpu"

            model_name = getattr(self._config, "embed_model", "all-MiniLM-L6-v2")
            self._model = SentenceTransformer(model_name, device=device)
            self._backend = "sentence_transformers"
            self._loaded = True

            test_dim = self._model.get_sentence_embedding_dimension()
            if test_dim != self._config.hnsw.dimensions:
                raise ValueError(
                    f"Embedding dim={test_dim} != HNSW dim={self._config.hnsw.dimensions}. "
                    f"Set HNSWConfig.dimensions={test_dim} or change model."
                )

            logger.info(
                f"Embedding pipeline: {model_name} on {device} (dim={test_dim})"
            )
            return
        except ImportError:
            logger.debug("sentence-transformers not installed, trying Ollama")
        except ValueError:
            raise  # Re-raise dimension mismatch
        except Exception as e:  # noqa: BLE001 — boundary boundary
            logger.warning(f"sentence-transformers load failed: {e}")

        # Attempt 2: Ollama
        try:
            import httpx

            url = getattr(self._config, "ollama_embed_url", "http://localhost:11434")
            resp = httpx.get(f"{url}/api/tags", timeout=2.0)
            if resp.status_code == 200:
                self._backend = "ollama"
                self._loaded = True
                model = getattr(self._config, "ollama_embed_model", "nomic-embed-text")
                logger.info(f"Embedding pipeline: Ollama {model}")
                return
        except (OSError, ValueError):  # SEC-003 — network boundary
            logger.debug("Ollama not available")

        self._backend = "none"
        logger.warning("No embedding backend — records stored without vectors")

    def embed(self, text: str) -> Optional[List[float]]:
        """Embed a single text string. Returns None if unavailable."""
        if not getattr(self._config, "auto_embed", True):
            return None

        self._ensure_loaded()
        if not self._loaded or not text or not text.strip():
            return None

        if self._backend == "sentence_transformers":
            return self._embed_st(text)
        elif self._backend == "ollama":
            return self._embed_ollama(text)
        return None

    def embed_batch(self, texts: List[str]) -> List[Optional[List[float]]]:
        """Embed multiple texts. Single GPU pass for sentence-transformers."""
        if not getattr(self._config, "auto_embed", True):
            return [None] * len(texts)

        self._ensure_loaded()
        if not self._loaded:
            return [None] * len(texts)

        if self._backend == "sentence_transformers":
            return self._embed_st_batch(texts)
        elif self._backend == "ollama":
            return [self._embed_ollama(t) if t and t.strip() else None for t in texts]
        return [None] * len(texts)

    def _embed_st(self, text: str) -> Optional[List[float]]:
        try:
            vec = self._model.encode(text, normalize_embeddings=True)
            return vec.tolist()
        except Exception as e:  # noqa: BLE001 — boundary boundary
            logger.warning(f"ST embed failed: {e}")
            return None

    def _embed_st_batch(
        self, texts: List[str], _cpu_retry: bool = False
    ) -> List[Optional[List[float]]]:
        try:
            valid = [(i, t) for i, t in enumerate(texts) if t and t.strip()]
            if not valid:
                return [None] * len(texts)

            batch_size = getattr(self._config, "embed_batch_size", 64)
            vecs = self._model.encode(
                [t for _, t in valid],
                batch_size=batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
            )

            results: List[Optional[List[float]]] = [None] * len(texts)
            for j, (idx, _) in enumerate(valid):
                results[idx] = vecs[j].tolist()
            return results

        except RuntimeError as e:
            if "out of memory" in str(e).lower() and not _cpu_retry:
                logger.warning("GPU OOM during batch embed — retrying on CPU")
                self._model = self._model.to("cpu")
                return self._embed_st_batch(texts, _cpu_retry=True)
            raise

    def _embed_ollama(self, text: str) -> Optional[List[float]]:
        try:
            import httpx

            url = getattr(self._config, "ollama_embed_url", "http://localhost:11434")
            model = getattr(self._config, "ollama_embed_model", "nomic-embed-text")
            resp = httpx.post(
                f"{url}/api/embed",
                json={"model": model, "input": text},
                timeout=10.0,
            )
            if resp.status_code == 200:
                data = resp.json()
                embeddings = data.get("embeddings", [])
                if embeddings:
                    return embeddings[0]
            return None
        except (OSError, ValueError) as e:  # SEC-003 — network boundary
            logger.warning(f"Ollama embed failed: {e}")
            return None


def create_default_embedding_fn(
    config: MemoryConfig,
) -> Optional[Callable[[str], Optional[List[float]]]]:
    """Factory: create an embedding function for AgentDB.set_embedding_fn()."""
    if not getattr(config, "auto_embed", True):
        return None
    pipeline = EmbeddingPipeline(config)
    return pipeline.embed
