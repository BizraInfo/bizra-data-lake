"""
Tests for Phase 32 runtime integration — SovereignRuntime uses real embeddings
in _run_cognitive_fusion() with graceful degradation.

Covers:
- Runtime initializes embedding subsystems without crash
- Real embedding flows through to CognitiveFusion
- Quality gate rejection triggers zero-vector fallback
- NTU context enrichment passes through
- Full graceful degradation when all backends unavailable

Standing on Giants: Reimers (sentence-BERT) + Takens (NTU) + Shannon (quality gate)
Artifact: core/sovereign/runtime_core.py :: _run_cognitive_fusion
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest


class TestRuntimeEmbeddingInit:
    """SovereignRuntime._init_embedding_service() works."""

    def test_runtime_boots_with_embedding_fields(self):
        """Runtime initializes Phase 32 fields without crashing."""
        from core.sovereign.runtime_core import SovereignRuntime

        rt = SovereignRuntime()

        # Fields should exist (may be None if deps unavailable)
        assert hasattr(rt, "_embedding_service")
        assert hasattr(rt, "_embedding_gate")
        assert hasattr(rt, "_ntu_adapter")

    def test_embedding_service_initialized_when_available(self):
        """After _init_embedding_service(), service should be set."""
        from core.sovereign.runtime_core import SovereignRuntime

        rt = SovereignRuntime()
        # _init_embedding_service is called in async initialize(), not __init__
        rt._init_embedding_service()

        # core.embedding has no external deps — should always succeed
        assert rt._embedding_service is not None
        assert rt._embedding_gate is not None

    def test_ntu_adapter_initialized_when_numpy_available(self):
        """After _init_embedding_service(), NTUFusionAdapter is set if numpy is installed."""
        from core.sovereign.runtime_core import SovereignRuntime

        rt = SovereignRuntime()
        rt._init_embedding_service()

        try:
            import numpy  # noqa: F401

            assert rt._ntu_adapter is not None
        except ImportError:
            assert rt._ntu_adapter is None


class TestRuntimeCognitiveFusion:
    """_run_cognitive_fusion() uses real embedding pipeline."""

    def _make_runtime_with_mocks(self):
        """Create runtime with controlled embedding and fusion mocks."""
        from core.sovereign.runtime_core import SovereignRuntime

        rt = SovereignRuntime()

        # Mock embedding service
        mock_embed_svc = MagicMock()
        mock_embed_svc.embed.return_value = [0.42] * 768
        rt._embedding_service = mock_embed_svc

        # Mock quality gate (passes)
        mock_gate = MagicMock()
        mock_gate.validate.return_value = MagicMock(passed=True)
        rt._embedding_gate = mock_gate

        # Mock cognitive fusion
        mock_fusion = MagicMock()
        mock_fusion.process.return_value = MagicMock(passes_gate=True, snr_score=0.92)
        rt._cognitive_fusion = mock_fusion

        return rt, mock_embed_svc, mock_gate, mock_fusion

    def test_real_embedding_flows_to_fusion(self):
        """When embedding service works, fusion receives real vectors."""
        rt, mock_svc, mock_gate, mock_fusion = self._make_runtime_with_mocks()

        # Create a SovereignQuery-like object
        query = MagicMock()
        query.text = "What is entropy?"
        query.context = {}

        result = rt._run_cognitive_fusion(query, "thought prompt")

        # Embedding service was called
        mock_svc.embed.assert_called_once_with("What is entropy?")

        # Quality gate validated the embedding
        mock_gate.validate.assert_called_once_with([0.42] * 768)

        # Fusion received the real embedding, not zeros
        call_kwargs = mock_fusion.process.call_args
        embedding_arg = call_kwargs.kwargs.get(
            "query_embedding", call_kwargs[1].get("query_embedding")
        )
        assert embedding_arg == [0.42] * 768

    def test_gate_rejection_triggers_zero_fallback(self):
        """When quality gate rejects embedding, fusion gets zero vector."""
        rt, mock_svc, mock_gate, mock_fusion = self._make_runtime_with_mocks()

        # Gate rejects
        mock_gate.validate.return_value = MagicMock(
            passed=False, reason="embedding_too_uniform"
        )

        query = MagicMock()
        query.text = "degenerate query"
        query.context = {}

        rt._run_cognitive_fusion(query, "prompt")

        # Fusion should receive zero vector (fallback)
        call_kwargs = mock_fusion.process.call_args
        embedding_arg = call_kwargs.kwargs.get(
            "query_embedding", call_kwargs[1].get("query_embedding")
        )
        assert embedding_arg == [0.0] * 768

    def test_embedding_exception_triggers_zero_fallback(self):
        """If embed() throws, fusion still runs with zero vector."""
        rt, mock_svc, mock_gate, mock_fusion = self._make_runtime_with_mocks()

        mock_svc.embed.side_effect = RuntimeError("backend crashed")

        query = MagicMock()
        query.text = "test"
        query.context = {}

        result = rt._run_cognitive_fusion(query, "prompt")

        # Should not crash
        assert result is not None

        # Fusion received zero fallback
        call_kwargs = mock_fusion.process.call_args
        embedding_arg = call_kwargs.kwargs.get(
            "query_embedding", call_kwargs[1].get("query_embedding")
        )
        assert embedding_arg == [0.0] * 768

    def test_ntu_context_enrichment_applied(self):
        """NTU adapter enriches context before fusion receives it."""
        rt, mock_svc, mock_gate, mock_fusion = self._make_runtime_with_mocks()

        mock_ntu = MagicMock()
        mock_ntu.enrich_context.return_value = {
            "ntu_state": {
                "belief": 0.8,
                "entropy": 0.3,
                "potential": 0.7,
                "iteration": 5,
                "pattern": None,
            },
        }
        rt._ntu_adapter = mock_ntu

        query = MagicMock()
        query.text = "test"
        query.context = {"original": True}

        rt._run_cognitive_fusion(query, "prompt")

        # NTU adapter was called with a copy of context
        mock_ntu.enrich_context.assert_called_once()

        # Fusion received enriched context
        call_kwargs = mock_fusion.process.call_args
        context_arg = call_kwargs.kwargs.get("context", call_kwargs[1].get("context"))
        assert "ntu_state" in context_arg

    def test_ntu_exception_does_not_block_fusion(self):
        """If NTU adapter throws, fusion still runs with original context."""
        rt, mock_svc, mock_gate, mock_fusion = self._make_runtime_with_mocks()

        mock_ntu = MagicMock()
        mock_ntu.enrich_context.side_effect = RuntimeError("NTU crashed")
        rt._ntu_adapter = mock_ntu

        query = MagicMock()
        query.text = "test"
        query.context = {"key": "val"}

        result = rt._run_cognitive_fusion(query, "prompt")

        # Should not crash — fusion still runs
        assert result is not None
        mock_fusion.process.assert_called_once()


class TestRuntimeGracefulDegradation:
    """Full degradation: no embedding, no NTU, fusion still runs."""

    def test_runtime_works_when_all_phase32_unavailable(self):
        """Even without embedding/NTU, cognitive fusion runs with zeros."""
        from core.sovereign.runtime_core import SovereignRuntime

        rt = SovereignRuntime()

        # Force all Phase 32 subsystems off
        rt._embedding_service = None
        rt._embedding_gate = None
        rt._ntu_adapter = None

        # Mock fusion only
        mock_fusion = MagicMock()
        mock_fusion.process.return_value = MagicMock(passes_gate=True)
        rt._cognitive_fusion = mock_fusion

        query = MagicMock()
        query.text = "test degraded"
        query.context = {}

        result = rt._run_cognitive_fusion(query, "prompt")

        assert result is not None
        call_kwargs = mock_fusion.process.call_args
        embedding_arg = call_kwargs.kwargs.get(
            "query_embedding", call_kwargs[1].get("query_embedding")
        )
        assert embedding_arg == [0.0] * 768

    def test_fusion_exception_returns_none(self):
        """If fusion itself crashes, _run_cognitive_fusion returns None."""
        from core.sovereign.runtime_core import SovereignRuntime

        rt = SovereignRuntime()
        rt._embedding_service = None
        rt._ntu_adapter = None

        mock_fusion = MagicMock()
        mock_fusion.process.side_effect = RuntimeError("fusion broken")
        rt._cognitive_fusion = mock_fusion

        query = MagicMock()
        query.text = "test"
        query.context = {}

        result = rt._run_cognitive_fusion(query, "prompt")
        assert result is None
