"""Tests for Phase 46 GoTBridge -- async, mock-based, no live GoT or FAISS.

Standing on Giants: Besta (GoT, 2024) . Shannon (1948) . Johnson (FAISS, 2021)

Test classes:
1. TestGoTBridgeResult        - frozen dataclass, field types
2. TestGoTBridgeInit          - default params from constants
3. TestReason                 - with mock GoT engine, evidence injection, convergence
4. TestReasonWithEvidence     - pre-provided evidence path
5. TestFallbackMode           - no GoT engine returns template result
6. TestSearchIntegration      - mock search engine providing evidence
7. TestConvergenceGate        - SNR threshold enforcement
8. TestEvidenceHelpers        - _evidence_to_facts, _search_for_evidence
9. TestCanonicalSignerGate    - signer fallback gated in canonical mode
"""

from dataclasses import FrozenInstanceError
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.integration.constants import (
    GOT_CONVERGENCE_SNR,
    GOT_MAX_DEPTH,
    GOT_MAX_HYPOTHESES,
)
from core.memory.types import MemoryKind, MemoryRecord, SearchResult
from core.reasoning.got_bridge import (
    PHASE46_GOT_BRIDGE_ENABLED,
    GoTBridge,
    GoTBridgeResult,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_search_result(
    content: str = "evidence text", score: float = 0.9
) -> SearchResult:
    """Build a fake SearchResult for testing."""
    record = MemoryRecord(
        id="test-id",
        content=content,
        kind=MemoryKind.SEMANTIC,
        source="test_source.parquet",
        source_id="chunk_42",
    )
    return SearchResult(record=record, score=score, vector_score=score)


def _make_got_raw(
    conclusion: str = "the answer",
    snr_score: float = 0.95,
    thoughts: list | None = None,
    nodes_pruned: int = 0,
    depth_reached: int = 3,
) -> dict:
    """Build a raw GoT engine result dict."""
    return {
        "conclusion": conclusion,
        "snr_score": snr_score,
        "graph_stats": {"nodes_pruned": nodes_pruned},
        "depth_reached": depth_reached,
        "thoughts": thoughts or ["Hypothesis A is plausible", "Hypothesis B rejected"],
    }


# ===========================================================================
# 1. TestGoTBridgeResult
# ===========================================================================


class TestGoTBridgeResult:
    """GoTBridgeResult is a frozen (immutable) dataclass."""

    def test_frozen_dataclass(self):
        """GoTBridgeResult cannot be mutated after construction."""
        result = GoTBridgeResult(
            answer="test",
            hypotheses_explored=2,
            hypotheses_surviving=1,
            evidence=[],
            snr_score=0.9,
            convergence_path=["step1"],
            reasoning_depth=3,
            converged=True,
        )
        with pytest.raises(FrozenInstanceError):
            result.answer = "changed"  # type: ignore[misc]

    def test_field_types(self):
        """All fields have the documented types."""
        evidence = [_make_search_result()]
        result = GoTBridgeResult(
            answer="an answer",
            hypotheses_explored=5,
            hypotheses_surviving=3,
            evidence=evidence,
            snr_score=0.92,
            convergence_path=["step1", "step2"],
            reasoning_depth=4,
            converged=True,
        )
        assert isinstance(result.answer, str)
        assert isinstance(result.hypotheses_explored, int)
        assert isinstance(result.hypotheses_surviving, int)
        assert isinstance(result.evidence, list)
        assert isinstance(result.snr_score, float)
        assert isinstance(result.convergence_path, list)
        assert isinstance(result.reasoning_depth, int)
        assert isinstance(result.converged, bool)

    def test_evidence_contains_search_results(self):
        """Evidence list holds SearchResult objects."""
        ev = [_make_search_result("chunk A"), _make_search_result("chunk B")]
        result = GoTBridgeResult(
            answer="ok",
            hypotheses_explored=0,
            hypotheses_surviving=0,
            evidence=ev,
            snr_score=0.0,
            convergence_path=[],
            reasoning_depth=0,
            converged=False,
        )
        assert len(result.evidence) == 2
        assert all(isinstance(e, SearchResult) for e in result.evidence)


# ===========================================================================
# 2. TestGoTBridgeInit
# ===========================================================================


class TestGoTBridgeInit:
    """Constructor defaults come from integration constants."""

    def test_defaults_from_constants(self):
        """GoTBridge defaults match GOT_MAX_HYPOTHESES, GOT_CONVERGENCE_SNR, GOT_MAX_DEPTH."""
        bridge = GoTBridge()
        assert bridge._max_hypotheses == GOT_MAX_HYPOTHESES
        assert bridge._convergence_snr == GOT_CONVERGENCE_SNR
        assert bridge._max_depth == GOT_MAX_DEPTH

    def test_custom_params(self):
        """Custom params override defaults."""
        bridge = GoTBridge(
            max_hypotheses=10,
            convergence_snr=0.99,
            max_depth=6,
        )
        assert bridge._max_hypotheses == 10
        assert bridge._convergence_snr == 0.99
        assert bridge._max_depth == 6

    def test_no_engines_by_default(self):
        """Without arguments, both engines are None."""
        bridge = GoTBridge()
        assert bridge._search_engine is None
        assert bridge._got_engine is None

    def test_engines_injected(self):
        """Constructor accepts explicit engine instances."""
        search = MagicMock()
        got = MagicMock()
        bridge = GoTBridge(search_engine=search, got_engine=got)
        assert bridge._search_engine is search
        assert bridge._got_engine is got

    def test_feature_flag_type(self):
        """PHASE46_GOT_BRIDGE_ENABLED is a bool."""
        assert isinstance(PHASE46_GOT_BRIDGE_ENABLED, bool)


# ===========================================================================
# 3. TestReason
# ===========================================================================


class TestReason:
    """reason() with mock GoT engine, evidence injection, convergence."""

    async def test_reason_with_got_engine(self):
        """reason() invokes got.reason() and returns GoTBridgeResult."""
        got = AsyncMock()
        got.reason.return_value = _make_got_raw(snr_score=0.95)

        bridge = GoTBridge(got_engine=got)
        result = await bridge.reason("What is BIZRA?")

        assert isinstance(result, GoTBridgeResult)
        assert result.answer == "the answer"
        got.reason.assert_awaited_once()

    async def test_reason_merges_context_facts(self):
        """Existing context facts are preserved and merged with evidence facts."""
        got = AsyncMock()
        got.reason.return_value = _make_got_raw()

        search = MagicMock()
        search.search.return_value = [_make_search_result("new evidence")]

        bridge = GoTBridge(search_engine=search, got_engine=got)
        await bridge.reason("query", context={"facts": ["existing fact"]})

        # Verify the context passed to GoT contains both old and new facts
        call_args = got.reason.call_args
        ctx = call_args[0][1]  # second positional arg is context
        assert "existing fact" in ctx["facts"]
        assert any("new evidence" in f for f in ctx["facts"])

    async def test_reason_default_context_is_empty(self):
        """reason() with no context defaults to empty dict."""
        got = AsyncMock()
        got.reason.return_value = _make_got_raw()

        bridge = GoTBridge(got_engine=got)
        await bridge.reason("query")

        call_args = got.reason.call_args
        ctx = call_args[0][1]
        assert isinstance(ctx, dict)

    async def test_reason_passes_max_depth(self):
        """reason() passes max_depth as third arg to GoT engine."""
        got = AsyncMock()
        got.reason.return_value = _make_got_raw()

        bridge = GoTBridge(got_engine=got, max_depth=7)
        await bridge.reason("query")

        call_args = got.reason.call_args
        assert call_args[0][2] == 7  # third positional arg is max_depth


# ===========================================================================
# 4. TestReasonWithEvidence
# ===========================================================================


class TestReasonWithEvidence:
    """Pre-provided evidence path (skips FAISS search)."""

    async def test_reason_with_evidence_uses_provided(self):
        """reason_with_evidence() uses caller-supplied evidence, not search."""
        got = AsyncMock()
        got.reason.return_value = _make_got_raw()

        search = MagicMock()
        evidence = [_make_search_result("pre-supplied")]

        bridge = GoTBridge(search_engine=search, got_engine=got)
        result = await bridge.reason_with_evidence("query", evidence=evidence)

        # search.search should NOT have been called
        search.search.assert_not_called()
        assert result.evidence is evidence

    async def test_reason_with_evidence_empty_list(self):
        """Empty evidence list still produces a valid result."""
        got = AsyncMock()
        got.reason.return_value = _make_got_raw()

        bridge = GoTBridge(got_engine=got)
        result = await bridge.reason_with_evidence("query", evidence=[])

        assert isinstance(result, GoTBridgeResult)
        assert result.evidence == []

    async def test_reason_with_evidence_custom_context(self):
        """Custom context is forwarded to GoT engine."""
        got = AsyncMock()
        got.reason.return_value = _make_got_raw()

        bridge = GoTBridge(got_engine=got)
        ctx = {"domain": "security", "constraints": ["no-PII"]}
        await bridge.reason_with_evidence("query", evidence=[], context=ctx)

        call_args = got.reason.call_args
        passed_ctx = call_args[0][1]
        assert passed_ctx["domain"] == "security"
        assert passed_ctx["constraints"] == ["no-PII"]


# ===========================================================================
# 5. TestFallbackMode
# ===========================================================================


class TestFallbackMode:
    """No GoT engine returns template result with converged=False."""

    async def test_no_got_engine_returns_fallback(self):
        """When GoT engine is None and cannot be imported, result is fallback."""
        bridge = GoTBridge(got_engine=None)
        # Patch _get_got_engine to return None (import fails)
        bridge._get_got_engine = MagicMock(return_value=None)

        result = await bridge.reason("What is BIZRA?")

        assert isinstance(result, GoTBridgeResult)
        assert result.converged is False
        assert result.snr_score == 0.0
        assert result.hypotheses_explored == 0
        assert "fallback_no_got_engine" in result.convergence_path

    async def test_fallback_with_evidence(self):
        """Fallback with evidence produces answer mentioning evidence count."""
        bridge = GoTBridge(got_engine=None)
        bridge._get_got_engine = MagicMock(return_value=None)

        ev = [_make_search_result("important finding")]
        result = await bridge.reason_with_evidence("query", evidence=ev)

        assert "1 evidence item" in result.answer
        assert result.converged is False

    async def test_fallback_without_evidence(self):
        """Fallback without evidence returns the query as answer."""
        bridge = GoTBridge(got_engine=None)
        bridge._get_got_engine = MagicMock(return_value=None)

        result = await bridge.reason("my question")

        assert result.answer == "my question"

    async def test_got_engine_raises_uses_fallback(self):
        """If GoT engine raises, fallback is used."""
        got = AsyncMock()
        got.reason.side_effect = RuntimeError("engine crash")

        bridge = GoTBridge(got_engine=got)
        result = await bridge.reason("query")

        assert result.converged is False
        assert "fallback_no_got_engine" in result.convergence_path


# ===========================================================================
# 6. TestSearchIntegration
# ===========================================================================


class TestSearchIntegration:
    """Mock search engine providing evidence to reason()."""

    async def test_search_results_injected_as_evidence(self):
        """Evidence from search engine appears in result.evidence."""
        search = MagicMock()
        search.search.return_value = [
            _make_search_result("chunk A", score=0.9),
            _make_search_result("chunk B", score=0.8),
        ]
        got = AsyncMock()
        got.reason.return_value = _make_got_raw()

        bridge = GoTBridge(search_engine=search, got_engine=got)
        result = await bridge.reason("query")

        assert len(result.evidence) == 2
        search.search.assert_called_once()

    async def test_search_failure_returns_empty_evidence(self):
        """If search raises, evidence is empty and GoT proceeds."""
        search = MagicMock()
        search.search.side_effect = RuntimeError("FAISS broke")

        got = AsyncMock()
        got.reason.return_value = _make_got_raw()

        bridge = GoTBridge(search_engine=search, got_engine=got)
        result = await bridge.reason("query")

        assert result.evidence == []
        got.reason.assert_awaited_once()

    async def test_no_search_engine_yields_empty_evidence(self):
        """Without search engine, evidence is empty list."""
        got = AsyncMock()
        got.reason.return_value = _make_got_raw()

        bridge = GoTBridge(search_engine=None, got_engine=got)
        result = await bridge.reason("query")

        assert result.evidence == []

    async def test_search_top_k_matches_max_hypotheses(self):
        """Search is called with top_k=max_hypotheses."""
        search = MagicMock()
        search.search.return_value = []

        got = AsyncMock()
        got.reason.return_value = _make_got_raw()

        bridge = GoTBridge(search_engine=search, got_engine=got, max_hypotheses=7)
        await bridge.reason("query")

        search.search.assert_called_once_with("query", top_k=7)


# ===========================================================================
# 7. TestConvergenceGate
# ===========================================================================


class TestConvergenceGate:
    """SNR threshold enforcement: converged iff snr_score >= convergence_snr."""

    async def test_converged_when_snr_above_threshold(self):
        """SNR above threshold produces converged=True."""
        got = AsyncMock()
        got.reason.return_value = _make_got_raw(snr_score=GOT_CONVERGENCE_SNR + 0.05)

        bridge = GoTBridge(got_engine=got)
        result = await bridge.reason("query")

        assert result.converged is True
        assert any("CONVERGED" in p for p in result.convergence_path)

    async def test_not_converged_when_snr_below_threshold(self):
        """SNR below threshold produces converged=False."""
        got = AsyncMock()
        got.reason.return_value = _make_got_raw(snr_score=GOT_CONVERGENCE_SNR - 0.10)

        bridge = GoTBridge(got_engine=got)
        result = await bridge.reason("query")

        assert result.converged is False
        assert any("NOT_CONVERGED" in p for p in result.convergence_path)

    async def test_converged_at_exact_threshold(self):
        """SNR exactly at threshold produces converged=True (>=)."""
        got = AsyncMock()
        got.reason.return_value = _make_got_raw(snr_score=GOT_CONVERGENCE_SNR)

        bridge = GoTBridge(got_engine=got)
        result = await bridge.reason("query")

        assert result.converged is True

    async def test_custom_convergence_snr(self):
        """Custom convergence_snr overrides the default threshold."""
        got = AsyncMock()
        got.reason.return_value = _make_got_raw(snr_score=0.50)

        bridge = GoTBridge(got_engine=got, convergence_snr=0.50)
        result = await bridge.reason("query")

        assert result.converged is True

    async def test_convergence_path_records_snr(self):
        """Convergence path includes the actual SNR value."""
        got = AsyncMock()
        got.reason.return_value = _make_got_raw(snr_score=0.88)

        bridge = GoTBridge(got_engine=got)
        result = await bridge.reason("query")

        # Should contain the SNR value formatted
        path_str = " ".join(result.convergence_path)
        assert "0.880" in path_str

    async def test_hypotheses_count(self):
        """hypotheses_explored counts thoughts containing 'Hypothesis'."""
        got = AsyncMock()
        got.reason.return_value = _make_got_raw(
            thoughts=[
                "Hypothesis 1: BIZRA is a seed",
                "Hypothesis 2: BIZRA is a system",
                "Verifying claim 1",
                "hypothesis 3 is weak",
            ],
            nodes_pruned=1,
        )

        bridge = GoTBridge(got_engine=got)
        result = await bridge.reason("query")

        assert result.hypotheses_explored == 3
        # 3 explored - 1 pruned = 2 surviving
        assert result.hypotheses_surviving == 2


# ===========================================================================
# 8. TestEvidenceHelpers
# ===========================================================================


class TestEvidenceHelpers:
    """Static/instance helper methods for evidence handling."""

    def test_evidence_to_facts_format(self):
        """_evidence_to_facts produces '[source] content' strings."""
        ev = [_make_search_result("The FAISS index contains vectors")]
        facts = GoTBridge._evidence_to_facts(ev)

        assert len(facts) == 1
        assert facts[0].startswith("[test_source.parquet]")
        assert "The FAISS index contains vectors" in facts[0]

    def test_evidence_to_facts_truncates_long_content(self):
        """Content longer than 200 chars is truncated."""
        long_content = "x" * 300
        ev = [_make_search_result(long_content)]
        facts = GoTBridge._evidence_to_facts(ev)

        # The content portion should be 200 chars max
        content_part = facts[0].split("] ", 1)[1]
        assert len(content_part) == 200

    def test_evidence_to_facts_multiple(self):
        """Multiple evidence items produce multiple facts."""
        ev = [
            _make_search_result("fact A"),
            _make_search_result("fact B"),
            _make_search_result("fact C"),
        ]
        facts = GoTBridge._evidence_to_facts(ev)
        assert len(facts) == 3

    def test_search_for_evidence_no_engine(self):
        """_search_for_evidence returns [] when no search engine."""
        bridge = GoTBridge(search_engine=None)
        assert bridge._search_for_evidence("query") == []

    def test_search_for_evidence_calls_search(self):
        """_search_for_evidence delegates to search engine."""
        search = MagicMock()
        search.search.return_value = [_make_search_result()]

        bridge = GoTBridge(search_engine=search, max_hypotheses=5)
        results = bridge._search_for_evidence("query")

        assert len(results) == 1
        search.search.assert_called_once_with("query", top_k=5)

    def test_search_for_evidence_handles_exception(self):
        """_search_for_evidence returns [] on search failure."""
        search = MagicMock()
        search.search.side_effect = RuntimeError("broken")

        bridge = GoTBridge(search_engine=search)
        results = bridge._search_for_evidence("query")

        assert results == []

    def test_build_fallback_with_facts(self):
        """_build_fallback_result with facts includes evidence count."""
        ev = [_make_search_result()]
        facts = ["[src] some evidence"]
        result = GoTBridge._build_fallback_result("query", ev, facts)

        assert "1 evidence item" in result.answer
        assert result.converged is False
        assert result.reasoning_depth == 0

    def test_build_fallback_without_facts(self):
        """_build_fallback_result without facts returns query as answer."""
        result = GoTBridge._build_fallback_result("my query", [], [])
        assert result.answer == "my query"


# ── TestCanonicalSignerGate ───────────────────────────────────────────────
# Standing on Giants: Al-Ghazali (intent gate, 1096) — cover the full
# intent surface, not just the edge.


class TestCanonicalSignerGate:
    """Signer fallback behaviour changes under canonical mode."""

    def test_non_canonical_falls_back_to_simple_signer(self):
        """Without canonical_mode, Ed25519 failure falls back to SimpleSigner."""
        with patch(
            "core.proof_engine.receipt.Ed25519Signer.generate",
            side_effect=ImportError("no ed25519"),
        ):
            signer = GoTBridge._resolve_receipt_signer(None, canonical_mode=False)
        # SimpleSigner has a .key_bytes attribute with the default key
        assert hasattr(signer, "sign") or hasattr(signer, "key_bytes")

    def test_canonical_mode_rejects_simple_signer_fallback(self):
        """In canonical mode, Ed25519 failure raises RuntimeError."""
        with patch(
            "core.proof_engine.receipt.Ed25519Signer.generate",
            side_effect=ImportError("no ed25519"),
        ):
            with pytest.raises(RuntimeError, match="canonical mode"):
                GoTBridge._resolve_receipt_signer(None, canonical_mode=True)

    def test_canonical_mode_accepts_ed25519(self):
        """Canonical mode succeeds when Ed25519Signer is available."""
        bridge = GoTBridge(canonical_mode=True)
        # Should not raise — Ed25519Signer.generate() works in test env
        assert bridge._receipt_signer is not None

    def test_explicit_signer_bypasses_gate(self):
        """Providing an explicit signer skips resolution in any mode."""
        custom_signer = MagicMock()
        bridge = GoTBridge(receipt_signer=custom_signer, canonical_mode=True)
        assert bridge._receipt_signer is custom_signer

    def test_canonical_mode_flag_stored(self):
        """The canonical_mode flag is stored on the bridge instance."""
        bridge = GoTBridge(canonical_mode=True)
        assert bridge._canonical_mode is True

        bridge2 = GoTBridge(canonical_mode=False)
        assert bridge2._canonical_mode is False
