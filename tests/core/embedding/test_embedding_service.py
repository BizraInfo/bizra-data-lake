"""
Tests for EmbeddingService — tiered embedding generation.

Covers:
- Tier 1 (sentence-transformers) happy path
- Tier 2 (Ollama) fallback
- Error when both tiers unavailable
- Text truncation

Standing on Giants: Reimers & Gurevych (2019, sentence-BERT)
Artifact: core/embedding/service.py
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from core.embedding import EmbeddingConfig, EmbeddingService, EmbeddingUnavailableError


class TestEmbeddingServiceTier1:
    """Tier 1: sentence-transformers (local)."""

    def test_embed_local_produces_nonzero_vector(self):
        """A mocked sentence-transformer model returns a real embedding."""
        import numpy as np

        fake_vector = np.random.randn(384).astype(np.float32)
        fake_vector /= np.linalg.norm(fake_vector)

        mock_model = MagicMock()
        mock_model.encode.return_value = fake_vector
        mock_model.get_sentence_embedding_dimension.return_value = 384

        svc = EmbeddingService(EmbeddingConfig(model_name="test-model"))
        svc._model = mock_model
        svc._dimension = 384

        result = svc.embed("Hello world")

        assert isinstance(result, list)
        assert len(result) == 384
        assert any(v != 0.0 for v in result), "Should not be a zero vector"
        assert svc.active_tier == "local"

    def test_embed_local_sets_dimension(self):
        """After first embed, dimension property reflects model output."""
        import numpy as np

        fake_vector = np.ones(768, dtype=np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_vector
        mock_model.get_sentence_embedding_dimension.return_value = 768

        svc = EmbeddingService()
        svc._model = mock_model
        svc._dimension = 768

        svc.embed("test")
        assert svc.dimension == 768


class TestEmbeddingServiceTier2:
    """Tier 2: Ollama fallback."""

    def test_embed_ollama_fallback(self):
        """When sentence-transformers fails, falls back to Ollama."""
        svc = EmbeddingService(
            EmbeddingConfig(ollama_url="http://localhost:11434")
        )

        fake_embedding = [0.1] * 768

        # Tier 1 raises
        with patch.object(svc, "_embed_local", side_effect=ImportError("no st")):
            with patch.object(svc, "_embed_ollama", return_value=fake_embedding):
                result = svc.embed("test query")

        assert result == fake_embedding
        assert svc.active_tier == "ollama"


class TestEmbeddingServiceFailure:
    """Both tiers fail."""

    def test_raises_when_both_unavailable(self):
        """EmbeddingUnavailableError raised when no backend works."""
        svc = EmbeddingService()

        with patch.object(svc, "_embed_local", side_effect=ImportError("no st")):
            with patch.object(
                svc, "_embed_ollama", side_effect=ConnectionError("no ollama")
            ):
                with pytest.raises(EmbeddingUnavailableError):
                    svc.embed("any text")

    def test_never_returns_zero_vector(self):
        """Verify invariant: service raises rather than returning zero vector."""
        svc = EmbeddingService()

        with patch.object(svc, "_embed_local", side_effect=ImportError("no st")):
            with patch.object(
                svc, "_embed_ollama", side_effect=ConnectionError("no ollama")
            ):
                with pytest.raises(EmbeddingUnavailableError):
                    svc.embed("should not return zeros")


class TestEmbeddingServiceConfig:
    """Configuration and text handling."""

    def test_text_truncated_to_max_length(self):
        """Long texts are truncated before embedding."""
        import numpy as np

        fake_vector = np.ones(384, dtype=np.float32)
        mock_model = MagicMock()
        mock_model.encode.return_value = fake_vector

        config = EmbeddingConfig(max_text_length=10)
        svc = EmbeddingService(config)
        svc._model = mock_model
        svc._dimension = 384

        svc.embed("This text is definitely longer than 10 characters")

        # Verify the model received truncated text
        call_args = mock_model.encode.call_args
        assert len(call_args[0][0]) == 10

    def test_config_from_env_defaults(self):
        """Default config uses expected model and URLs."""
        config = EmbeddingConfig()
        assert config.model_name == "all-MiniLM-L6-v2"
        assert "11434" in config.ollama_url
        assert config.max_text_length == 512

    def test_active_tier_starts_as_none(self):
        """Fresh service has no active tier."""
        svc = EmbeddingService()
        assert svc.active_tier == "none"
        assert svc.dimension == 0
