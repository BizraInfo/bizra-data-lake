# Phase 39 — Pseudocode Module 02: Embedding Pipeline Integration

**FR-02** | Priority: 2 | Risk: Medium | New files: 1

---

## Overview

Provide a default embedding function that `AgentDB` auto-loads, so callers
don't need to supply embeddings manually. Lazy-loaded, GPU-aware, with
graceful fallback.

---

## Flow Diagram

```
AgentDB.store("The Earth orbits the Sun")
  │
  ├── embedding=None? AND auto_embed enabled?
  │     YES → call _embedding_fn(content)
  │              │
  │              ├── Try: sentence-transformers (all-MiniLM-L6-v2, dim=768)
  │              ├── Fallback: Ollama nomic-embed-text (dim=768)
  │              └── Fallback: None (store without embedding)
  │
  └── Proceed with store (SQLite + optional HNSW)
```

---

## Config Additions: `core/memory/config.py`

```
IN MemoryConfig dataclass, ADD:

    # Embedding pipeline
    auto_embed: bool = True
    embed_model: str = "all-MiniLM-L6-v2"
    embed_device: str = "cpu"         # "cpu", "cuda", "auto"
    embed_batch_size: int = 64
    ollama_embed_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"
```

---

## Pseudocode: `core/memory/embedding.py`

```
MODULE embedding

IMPORT logging
IMPORT numpy as np
FROM typing IMPORT List, Optional, Callable

LOG = logging.getLogger(__name__)

CLASS EmbeddingPipeline:
    """Lazy-loaded embedding provider with tiered fallback.

    Priority:
      1. sentence-transformers (local, fast, no network)
      2. Ollama API (local server, requires running instance)
      3. None (disabled — records stored without vectors)
    """

    CONSTRUCTOR(config: MemoryConfig):
        self._config = config
        self._model = None           # Lazy-loaded
        self._backend: str = "none"  # "sentence_transformers", "ollama", "none"
        self._loaded = False
        self._load_attempted = False

    PROPERTY backend -> str:
        RETURN self._backend

    PROPERTY dimensions -> int:
        """Return expected embedding dimensions for validation."""
        IF self._backend == "sentence_transformers":
            RETURN 768   # all-MiniLM-L6-v2
        ELIF self._backend == "ollama":
            RETURN 768   # nomic-embed-text
        RETURN self._config.hnsw.dimensions

    METHOD _ensure_loaded():
        """Lazy-load the embedding model on first use."""
        IF self._loaded OR self._load_attempted:
            RETURN

        self._load_attempted = True

        # Attempt 1: sentence-transformers
        TRY:
            FROM sentence_transformers IMPORT SentenceTransformer

            device = self._config.embed_device
            IF device == "auto":
                IMPORT torch
                device = "cuda" IF torch.cuda.is_available() ELSE "cpu"

            self._model = SentenceTransformer(
                self._config.embed_model,
                device=device
            )
            self._backend = "sentence_transformers"
            self._loaded = True

            # Validate dimensions match HNSW config
            test_dim = self._model.get_sentence_embedding_dimension()
            IF test_dim != self._config.hnsw.dimensions:
                RAISE ValueError(
                    f"Embedding model dim={test_dim} != HNSW dim={self._config.hnsw.dimensions}. "
                    f"Set HNSWConfig.dimensions={test_dim} or use a {self._config.hnsw.dimensions}-dim model."
                )

            LOG.info(f"Embedding pipeline: {self._config.embed_model} on {device} (dim={test_dim})")
            RETURN

        EXCEPT ImportError:
            LOG.debug("sentence-transformers not installed, trying Ollama")
        EXCEPT Exception as e:
            LOG.warning(f"sentence-transformers load failed: {e}")

        # Attempt 2: Ollama
        TRY:
            IMPORT httpx
            # Ping Ollama to check availability
            response = httpx.get(
                f"{self._config.ollama_embed_url}/api/tags",
                timeout=2.0
            )
            IF response.status_code == 200:
                self._backend = "ollama"
                self._loaded = True
                LOG.info(f"Embedding pipeline: Ollama {self._config.ollama_embed_model}")
                RETURN

        EXCEPT Exception:
            LOG.debug("Ollama not available")

        # No embedding backend available
        self._backend = "none"
        LOG.warning("No embedding backend available — records will be stored without vectors")

    METHOD embed(text: str) -> Optional[List[float]]:
        """Embed a single text. Returns None if no backend available."""
        IF NOT self._config.auto_embed:
            RETURN None

        self._ensure_loaded()

        IF NOT self._loaded:
            RETURN None

        IF NOT text OR NOT text.strip():
            RETURN None

        IF self._backend == "sentence_transformers":
            RETURN self._embed_st(text)
        ELIF self._backend == "ollama":
            RETURN self._embed_ollama(text)
        RETURN None

    METHOD embed_batch(texts: List[str]) -> List[Optional[List[float]]]:
        """Embed multiple texts efficiently (single GPU pass for ST)."""
        IF NOT self._config.auto_embed:
            RETURN [None] * len(texts)

        self._ensure_loaded()

        IF NOT self._loaded:
            RETURN [None] * len(texts)

        IF self._backend == "sentence_transformers":
            RETURN self._embed_st_batch(texts)
        ELIF self._backend == "ollama":
            # Ollama doesn't support batch natively — loop
            RETURN [self._embed_ollama(t) IF t.strip() ELSE None FOR t IN texts]
        RETURN [None] * len(texts)

    METHOD _embed_st(text: str) -> Optional[List[float]]:
        """sentence-transformers single embed."""
        TRY:
            vec = self._model.encode(text, normalize_embeddings=True)
            RETURN vec.tolist()
        EXCEPT Exception as e:
            LOG.warning(f"ST embed failed: {e}")
            RETURN None

    METHOD _embed_st_batch(texts: List[str]) -> List[Optional[List[float]]]:
        """sentence-transformers batch embed with GPU batching."""
        TRY:
            # Filter empty strings, track indices
            valid_indices = [i FOR i, t IN enumerate(texts) IF t.strip()]
            valid_texts = [texts[i] FOR i IN valid_indices]

            IF NOT valid_texts:
                RETURN [None] * len(texts)

            vecs = self._model.encode(
                valid_texts,
                batch_size=self._config.embed_batch_size,
                normalize_embeddings=True,
                show_progress_bar=False
            )

            results = [None] * len(texts)
            FOR j, idx IN enumerate(valid_indices):
                results[idx] = vecs[j].tolist()

            RETURN results

        EXCEPT RuntimeError as e:
            IF "out of memory" IN str(e).lower():
                LOG.warning("GPU OOM during batch embed — falling back to CPU")
                self._model = self._model.to("cpu")
                RETURN self._embed_st_batch(texts)  # Retry on CPU
            RAISE

    METHOD _embed_ollama(text: str) -> Optional[List[float]]:
        """Ollama API embed."""
        TRY:
            IMPORT httpx
            response = httpx.post(
                f"{self._config.ollama_embed_url}/api/embed",
                json={"model": self._config.ollama_embed_model, "input": text},
                timeout=10.0
            )
            IF response.status_code == 200:
                data = response.json()
                embeddings = data.get("embeddings", [])
                IF embeddings:
                    RETURN embeddings[0]
            RETURN None
        EXCEPT Exception as e:
            LOG.warning(f"Ollama embed failed: {e}")
            RETURN None


FUNCTION create_default_embedding_fn(config: MemoryConfig) -> Optional[Callable]:
    """Factory: create an embedding function for AgentDB.set_embedding_fn()."""
    pipeline = EmbeddingPipeline(config)
    RETURN pipeline.embed
```

---

## Pseudocode: AgentDB integration (`agent_db.py` changes)

```
IN AgentDB.initialize(), AFTER self._query_engine setup, ADD:

    # Wire default embedding function
    IF self._config.auto_embed AND self._embedding_fn IS None:
        FROM .embedding import create_default_embedding_fn
        self._embedding_fn = create_default_embedding_fn(self._config)


ADD METHOD store_batch():

    METHOD store_batch(
        self,
        contents: List[str],
        kind: MemoryKind = MemoryKind.SEMANTIC,
        importance: float = 0.5,
        source: str = "agent",
    ) -> List[MemoryRecord]:
        """Store multiple records with batch embedding (single GPU pass)."""
        self._ensure_initialized()

        # Batch embed
        embeddings = [None] * len(contents)
        IF self._embedding_fn IS NOT None:
            TRY:
                # Check if we have batch capability
                pipeline = getattr(self._embedding_fn, '__self__', None)
                IF pipeline AND hasattr(pipeline, 'embed_batch'):
                    embeddings = pipeline.embed_batch(contents)
                ELSE:
                    embeddings = [self._embedding_fn(c) FOR c IN contents]
            EXCEPT Exception as e:
                LOG.warning(f"Batch embed failed: {e}")

        records = []
        FOR content, embedding IN zip(contents, embeddings):
            record = self.store(
                content=content,
                kind=kind,
                embedding=embedding,
                importance=importance,
                source=source,
            )
            records.append(record)

        RETURN records
```

---

## TDD Anchors

```
TEST test_auto_embed_on_store:
    config = MemoryConfig(auto_embed=True)
    db = AgentDB(config)
    db.initialize()

    record = db.store("The Earth orbits the Sun")
    # If sentence-transformers available, embedding should be populated
    IF HAS_SENTENCE_TRANSFORMERS:
        ASSERT record.embedding IS NOT None
        ASSERT len(record.embedding) == 768

TEST test_auto_embed_disabled:
    config = MemoryConfig(auto_embed=False)
    db = AgentDB(config)
    db.initialize()

    record = db.store("test content")
    # No auto-embed → embedding should be None (no explicit embedding given)
    ASSERT record.embedding IS None

TEST test_embed_dimension_mismatch:
    config = MemoryConfig(embed_model="all-MiniLM-L6-v2")
    config.hnsw.dimensions = 384  # Wrong!
    pipeline = EmbeddingPipeline(config)

    WITH RAISES(ValueError, match="dim"):
        pipeline._ensure_loaded()

TEST test_batch_embed_performance:
    pipeline = EmbeddingPipeline(MemoryConfig())
    texts = ["sentence " + str(i) FOR i IN range(100)]

    start = time.monotonic()
    results = pipeline.embed_batch(texts)
    elapsed = time.monotonic() - start

    ASSERT all(r IS NOT None FOR r IN results)
    ASSERT elapsed < 5.0  # 100 sentences < 5 seconds

TEST test_empty_string_skips_embed:
    pipeline = EmbeddingPipeline(MemoryConfig())
    result = pipeline.embed("")
    ASSERT result IS None

TEST test_ollama_fallback:
    # Mock sentence-transformers as unavailable
    WITH mock.patch.dict(sys.modules, {"sentence_transformers": None}):
        pipeline = EmbeddingPipeline(MemoryConfig())
        pipeline._ensure_loaded()
        ASSERT pipeline.backend IN ("ollama", "none")

TEST test_store_batch_single_gpu_pass:
    db = AgentDB(MemoryConfig())
    db.initialize()

    records = db.store_batch(["alpha", "beta", "gamma"])
    ASSERT len(records) == 3
    ASSERT all(r.id != r2.id FOR r, r2 IN combinations(records, 2))
```

---

## Error Matrix

| Condition | Behavior |
|-----------|----------|
| sentence-transformers not installed | Try Ollama, then None |
| Ollama not running | backend="none", no embeddings |
| GPU OOM on batch | Retry on CPU, log warning |
| Empty text | Return None, don't embed |
| Dim mismatch | ValueError at load time |
| Network timeout to Ollama | Return None for that text |
