"""Tests for FR-02: Embedding Pipeline."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from core.memory.config import MemoryConfig
from core.memory.embedding import EmbeddingPipeline, create_default_embedding_fn


@pytest.fixture
def config(tmp_path: Path) -> MemoryConfig:
    cfg = MemoryConfig(data_dir=tmp_path / "agent_db")
    cfg.auto_embed = True
    return cfg


class TestEmbeddingPipelineInit:
    def test_defaults(self, config):
        pipeline = EmbeddingPipeline(config)
        assert pipeline.backend == "none"
        assert not pipeline.loaded

    def test_auto_embed_disabled(self, config):
        config.auto_embed = False
        pipeline = EmbeddingPipeline(config)
        result = pipeline.embed("hello world")
        assert result is None

    def test_no_backend_available(self, config):
        with patch.dict("sys.modules", {"sentence_transformers": None}):
            # Also make httpx fail for Ollama
            pipeline = EmbeddingPipeline(config)
            # Override ollama URL to unreachable
            config.ollama_embed_url = "http://127.0.0.1:99999"
            pipeline._ensure_loaded()
            assert pipeline.backend in ("ollama", "none")


class TestEmbedSingle:
    def test_empty_string_returns_none(self, config):
        pipeline = EmbeddingPipeline(config)
        pipeline._loaded = True
        pipeline._backend = "sentence_transformers"
        result = pipeline.embed("")
        assert result is None

    def test_whitespace_returns_none(self, config):
        pipeline = EmbeddingPipeline(config)
        pipeline._loaded = True
        pipeline._backend = "sentence_transformers"
        result = pipeline.embed("   ")
        assert result is None

    def test_embed_with_mock_st(self, config):
        """Test embedding with mocked sentence-transformers."""
        import numpy as np

        pipeline = EmbeddingPipeline(config)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(768).astype("float32")
        mock_model.get_sentence_embedding_dimension.return_value = 768
        pipeline._model = mock_model
        pipeline._backend = "sentence_transformers"
        pipeline._loaded = True

        result = pipeline.embed("test sentence")
        assert result is not None
        assert len(result) == 768
        mock_model.encode.assert_called_once()

    def test_embed_with_mock_ollama(self, config):
        """Test embedding with mocked Ollama API."""
        pipeline = EmbeddingPipeline(config)
        pipeline._backend = "ollama"
        pipeline._loaded = True

        with patch("httpx.post") as mock_post:
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {
                "embeddings": [[0.1] * 768]
            }
            mock_post.return_value = mock_resp

            result = pipeline.embed("test")
            assert result is not None
            assert len(result) == 768


class TestEmbedBatch:
    def test_batch_with_mock_st(self, config):
        import numpy as np

        pipeline = EmbeddingPipeline(config)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(3, 768).astype("float32")
        pipeline._model = mock_model
        pipeline._backend = "sentence_transformers"
        pipeline._loaded = True

        results = pipeline.embed_batch(["a", "b", "c"])
        assert len(results) == 3
        assert all(r is not None for r in results)

    def test_batch_skips_empty(self, config):
        import numpy as np

        pipeline = EmbeddingPipeline(config)
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(2, 768).astype("float32")
        pipeline._model = mock_model
        pipeline._backend = "sentence_transformers"
        pipeline._loaded = True

        results = pipeline.embed_batch(["a", "", "c"])
        assert len(results) == 3
        assert results[0] is not None
        assert results[1] is None
        assert results[2] is not None

    def test_batch_disabled(self, config):
        config.auto_embed = False
        pipeline = EmbeddingPipeline(config)
        results = pipeline.embed_batch(["a", "b"])
        assert results == [None, None]


class TestCreateDefaultFn:
    def test_returns_callable(self, config):
        fn = create_default_embedding_fn(config)
        assert fn is not None
        assert callable(fn)

    def test_returns_none_when_disabled(self, config):
        config.auto_embed = False
        fn = create_default_embedding_fn(config)
        assert fn is None
