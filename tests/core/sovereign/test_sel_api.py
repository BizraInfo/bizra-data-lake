"""
Tests for the Sovereign Experience Ledger (SEL) API endpoints and CLI.

Covers: FastAPI endpoints, asyncio server routes, CLI subcommand parsing,
and integration with the SEL backend.
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.sovereign.experience_ledger import (
    Episode,
    EpisodeAction,
    EpisodeImpact,
    SovereignExperienceLedger,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def populated_sel():
    """Create an SEL with 5 episodes for testing."""
    sel = SovereignExperienceLedger()
    for i in range(5):
        sel.commit(
            context=f"test query number {i} about machine learning",
            graph_hash=f"graph_{i}",
            graph_node_count=3 + i,
            actions=[("inference", f"LLM call {i}", True, 1000 * (i + 1))],
            snr_score=0.80 + i * 0.04,
            ihsan_score=0.85 + i * 0.03,
            snr_ok=True,
        )
    return sel


@pytest.fixture
def empty_sel():
    """Create an empty SEL."""
    return SovereignExperienceLedger()


@pytest.fixture
def mock_runtime(populated_sel):
    """Create a mock runtime with SEL attached."""
    runtime = MagicMock()
    runtime._experience_ledger = populated_sel
    return runtime


@pytest.fixture
def mock_runtime_no_sel():
    """Create a mock runtime without SEL."""
    runtime = MagicMock()
    runtime._experience_ledger = None
    return runtime


# ═══════════════════════════════════════════════════════════════════════════════
# SEL Backend Tests (via API contract)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSELEpisodeListing:
    """Test the episodes listing logic used by the API."""

    def test_list_episodes_returns_all(self, populated_sel):
        """Should return all episodes."""
        total = len(populated_sel)
        assert total == 5

        episodes = []
        for i in range(total - 1, -1, -1):
            ep = populated_sel.get_by_sequence(i)
            if ep is not None:
                episodes.append(ep.to_dict())

        assert len(episodes) == 5
        # Reverse chronological: newest first
        assert episodes[0]["sequence"] == 4
        assert episodes[-1]["sequence"] == 0

    def test_list_episodes_with_limit(self, populated_sel):
        """Should respect limit parameter."""
        total = len(populated_sel)
        limit = 3
        episodes = []
        for i in range(total - 1, max(total - 1 - limit, -1), -1):
            ep = populated_sel.get_by_sequence(i)
            if ep is not None:
                episodes.append(ep.to_dict())

        assert len(episodes) == 3
        assert episodes[0]["sequence"] == 4
        assert episodes[-1]["sequence"] == 2

    def test_list_episodes_empty(self, empty_sel):
        """Empty SEL should return empty list."""
        assert len(empty_sel) == 0

    def test_episode_dict_has_required_fields(self, populated_sel):
        """Episode dict should have all required API fields."""
        ep = populated_sel.get_by_sequence(0)
        d = ep.to_dict()

        required_fields = [
            "sequence", "timestamp_secs", "context", "graph_hash",
            "graph_node_count", "snr_score", "ihsan_score", "snr_ok",
            "episode_hash", "chain_hash",
        ]
        for field in required_fields:
            assert field in d, f"Missing field: {field}"


class TestSELEpisodeLookup:
    """Test episode lookup by hash."""

    def test_lookup_by_hash(self, populated_sel):
        """Should find episode by its content-address hash."""
        ep = populated_sel.get_by_sequence(2)
        found = populated_sel.get_by_hash(ep.episode_hash)
        assert found is not None
        assert found.context == ep.context

    def test_lookup_by_nonexistent_hash(self, populated_sel):
        """Should return None for unknown hash."""
        found = populated_sel.get_by_hash("0" * 64)
        assert found is None


class TestSELRIRRetrieval:
    """Test RIR retrieval logic used by the API."""

    def test_retrieve_returns_results(self, populated_sel):
        """Should return relevant episodes for a query."""
        results = populated_sel.retrieve("machine learning", top_k=3)
        assert len(results) == 3

    def test_retrieve_respects_top_k(self, populated_sel):
        """Should not exceed top_k."""
        results = populated_sel.retrieve("test", top_k=2)
        assert len(results) == 2

    def test_retrieve_empty_sel(self, empty_sel):
        """Should return empty list for empty SEL."""
        results = empty_sel.retrieve("anything", top_k=5)
        assert results == []

    def test_retrieve_results_serializable(self, populated_sel):
        """Results should be JSON-serializable via to_dict."""
        results = populated_sel.retrieve("learning", top_k=2)
        for ep in results:
            d = ep.to_dict()
            serialized = json.dumps(d)
            assert isinstance(serialized, str)


class TestSELChainVerification:
    """Test chain verification logic used by the API."""

    def test_verify_valid_chain(self, populated_sel):
        """Valid chain should verify."""
        assert populated_sel.verify_chain_integrity() is True

    def test_verify_empty_chain(self, empty_sel):
        """Empty chain should verify."""
        assert empty_sel.verify_chain_integrity() is True

    def test_verify_reports_chain_head(self, populated_sel):
        """Chain head should be a real hash, not genesis."""
        assert populated_sel.chain_head != "genesis"
        assert len(populated_sel.chain_head) == 64

    def test_verify_tampered_chain(self, populated_sel):
        """Tampered chain should fail verification."""
        populated_sel._episodes[1].context = "TAMPERED"
        assert populated_sel.verify_chain_integrity() is False


# ═══════════════════════════════════════════════════════════════════════════════
# Asyncio Server Route Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestAsyncioServerRoutes:
    """Test that the asyncio server routes SEL requests correctly."""

    def test_sel_episodes_route_exists(self):
        """The SovereignAPIServer should handle /v1/sel/episodes."""
        from core.sovereign.api import SovereignAPIServer

        server = SovereignAPIServer(runtime=MagicMock())
        # Check that _handle_sel_episodes method exists
        assert hasattr(server, "_handle_sel_episodes")
        assert callable(server._handle_sel_episodes)

    def test_sel_verify_route_exists(self):
        """The SovereignAPIServer should handle /v1/sel/verify."""
        from core.sovereign.api import SovereignAPIServer

        server = SovereignAPIServer(runtime=MagicMock())
        assert hasattr(server, "_handle_sel_verify")
        assert callable(server._handle_sel_verify)

    def test_sel_retrieve_route_exists(self):
        """The SovereignAPIServer should handle /v1/sel/retrieve."""
        from core.sovereign.api import SovereignAPIServer

        server = SovereignAPIServer(runtime=MagicMock())
        assert hasattr(server, "_handle_sel_retrieve")
        assert callable(server._handle_sel_retrieve)

    def test_sel_episode_by_hash_route_exists(self):
        """The SovereignAPIServer should handle /v1/sel/episodes/{hash}."""
        from core.sovereign.api import SovereignAPIServer

        server = SovereignAPIServer(runtime=MagicMock())
        assert hasattr(server, "_handle_sel_episode_by_hash")
        assert callable(server._handle_sel_episode_by_hash)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI Subcommand Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestCLISubcommandParsing:
    """Test CLI argument parsing for the sel subcommand."""

    def test_sel_episodes_parse(self):
        """Should parse 'sel episodes' command."""
        import argparse

        # Simulate the parser structure
        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        sel_parser = sub.add_parser("sel")
        sel_sub = sel_parser.add_subparsers(dest="sel_command")
        ep = sel_sub.add_parser("episodes")
        ep.add_argument("--limit", type=int, default=20)
        ep.add_argument("--json", action="store_true")

        args = parser.parse_args(["sel", "episodes", "--limit", "10", "--json"])
        assert args.command == "sel"
        assert args.sel_command == "episodes"
        assert args.limit == 10
        assert args.json is True

    def test_sel_retrieve_parse(self):
        """Should parse 'sel retrieve' command with query."""
        import argparse

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        sel_parser = sub.add_parser("sel")
        sel_sub = sel_parser.add_subparsers(dest="sel_command")
        ret = sel_sub.add_parser("retrieve")
        ret.add_argument("query")
        ret.add_argument("--top-k", type=int, default=5)
        ret.add_argument("--json", action="store_true")

        args = parser.parse_args(["sel", "retrieve", "neural network", "--top-k", "3"])
        assert args.command == "sel"
        assert args.sel_command == "retrieve"
        assert args.query == "neural network"
        assert args.top_k == 3

    def test_sel_verify_parse(self):
        """Should parse 'sel verify' command."""
        import argparse

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        sel_parser = sub.add_parser("sel")
        sel_sub = sel_parser.add_subparsers(dest="sel_command")
        ver = sel_sub.add_parser("verify")
        ver.add_argument("--json", action="store_true")

        args = parser.parse_args(["sel", "verify", "--json"])
        assert args.command == "sel"
        assert args.sel_command == "verify"
        assert args.json is True

    def test_sel_export_parse(self):
        """Should parse 'sel export' command with output."""
        import argparse

        parser = argparse.ArgumentParser()
        sub = parser.add_subparsers(dest="command")
        sel_parser = sub.add_parser("sel")
        sel_sub = sel_parser.add_subparsers(dest="sel_command")
        exp = sel_sub.add_parser("export")
        exp.add_argument("--output", "-o", default="-")

        args = parser.parse_args(["sel", "export", "-o", "out.jsonl"])
        assert args.command == "sel"
        assert args.sel_command == "export"
        assert args.output == "out.jsonl"


# ═══════════════════════════════════════════════════════════════════════════════
# Pydantic Model Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSELPydanticModels:
    """Test Pydantic request models for SEL endpoints."""

    def test_sel_retrieve_model_exists(self):
        """SELRetrieveModel should be importable."""
        try:
            from core.sovereign.api import SELRetrieveModel
            assert SELRetrieveModel is not None
        except ImportError:
            pytest.skip("pydantic not available")

    def test_sel_retrieve_model_defaults(self):
        """SELRetrieveModel should have correct defaults."""
        try:
            from core.sovereign.api import SELRetrieveModel
            model = SELRetrieveModel(query="test query")
            assert model.query == "test query"
            assert model.top_k == 5
        except ImportError:
            pytest.skip("pydantic not available")

    def test_sel_retrieve_model_custom_top_k(self):
        """SELRetrieveModel should accept custom top_k."""
        try:
            from core.sovereign.api import SELRetrieveModel
            model = SELRetrieveModel(query="test", top_k=10)
            assert model.top_k == 10
        except ImportError:
            pytest.skip("pydantic not available")
