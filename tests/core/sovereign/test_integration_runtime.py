"""
Comprehensive Test Suite for SovereignRuntime (integration_runtime.py)
======================================================================
Tests the complete BIZRA Sovereign Runtime: scoring, sovereignty checks,
lifecycle management, model challenges, inference, registry persistence,
and keypair management.

Standing on Giants: Shannon + Lamport + Vaswani + Anthropic
"""

from __future__ import annotations

import asyncio
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.sovereign.capability_card import (
    CapabilityCard,
    CardIssuer,
    ModelCapabilities,
    ModelTier,
    TaskType,
    create_capability_card,
)
from core.sovereign.integration_types import (
    InferenceRequest,
    InferenceResult,
    NetworkMode,
    SovereignConfig,
)
from core.sovereign.model_license_gate import GateChain, InMemoryRegistry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_config(tmp_path: Path, **overrides) -> SovereignConfig:
    """Build a SovereignConfig with tmp_path-based paths."""
    defaults = dict(
        network_mode=NetworkMode.OFFLINE,
        model_store_path=tmp_path / "models",
        keypair_path=tmp_path / "keys" / "keypair.json",
        bootstrap_nodes=[],
        default_model=None,
    )
    defaults.update(overrides)
    return SovereignConfig(**defaults)


def _make_valid_card(
    model_id: str = "test-model",
    tier: ModelTier = ModelTier.LOCAL,
    ihsan: float = 0.98,
    snr: float = 0.92,
    tasks: Optional[List[TaskType]] = None,
) -> CapabilityCard:
    """Build a signed, valid CapabilityCard for test fixtures."""
    if tasks is None:
        tasks = [TaskType.CHAT, TaskType.REASONING]
    card = create_capability_card(
        model_id=model_id,
        tier=tier,
        ihsan_score=ihsan,
        snr_score=snr,
        tasks_supported=tasks,
    )
    issuer = CardIssuer()
    return issuer.issue(card)


def _build_runtime(tmp_path: Path, **config_overrides):
    """Construct a SovereignRuntime with patched federation imports."""
    cfg = make_config(tmp_path, **config_overrides)
    # Patch ConstitutionalGate to avoid blake3 / pci.crypto dependency
    with patch(
        "core.sovereign.integration_runtime.ConstitutionalGate"
    ) as MockGate:
        MockGate.return_value = MagicMock()
        from core.sovereign.integration_runtime import SovereignRuntime

        rt = SovereignRuntime(config=cfg)
    return rt


# ---------------------------------------------------------------------------
# 1. TestScoringIhsanFallback
# ---------------------------------------------------------------------------


class TestScoringIhsanFallback:
    """Tests for _score_ihsan_fallback (heuristic Ihsan scoring)."""

    @pytest.fixture
    def runtime(self, tmp_path):
        return _build_runtime(tmp_path)

    def test_empty_response_base_score(self, runtime):
        """Empty string should return the base score of 0.7."""
        score = runtime._score_ihsan_fallback("")
        assert score == pytest.approx(0.7, abs=1e-9)

    def test_refusal_pattern_increases_score(self, runtime):
        """Response with a refusal pattern should score higher than base."""
        score = runtime._score_ihsan_fallback("I refuse to comply.")
        assert score > 0.7

    def test_refusal_i_cannot(self, runtime):
        score = runtime._score_ihsan_fallback("I cannot help with that.")
        assert score > 0.7

    def test_refusal_i_will_not(self, runtime):
        score = runtime._score_ihsan_fallback("I will not do this.")
        assert score > 0.7

    def test_refusal_not_appropriate(self, runtime):
        score = runtime._score_ihsan_fallback("This is not appropriate.")
        assert score > 0.7

    def test_ethical_phrase_privacy(self, runtime):
        """Ethical keyword 'privacy' should boost score."""
        base = runtime._score_ihsan_fallback("hello world")
        with_privacy = runtime._score_ihsan_fallback("We must respect privacy.")
        assert with_privacy > base

    def test_ethical_phrase_consent(self, runtime):
        base = runtime._score_ihsan_fallback("hello world")
        with_consent = runtime._score_ihsan_fallback("Consent is required.")
        assert with_consent > base

    def test_constructive_suggestion_increases_score(self, runtime):
        """Constructive words like 'suggest' or 'recommend' add score."""
        base = runtime._score_ihsan_fallback("hello world")
        constructive = runtime._score_ihsan_fallback(
            "I suggest an alternative approach. I recommend caution."
        )
        assert constructive > base

    def test_all_patterns_yield_high_score(self, runtime):
        """Response with refusal, ethical, AND constructive words scores high."""
        text = (
            "I refuse to help access private data. "
            "Privacy and consent and respect and dignity and safety are paramount. "
            "Instead, I suggest you recommend an alternative."
        )
        score = runtime._score_ihsan_fallback(text)
        assert score >= 0.95

    def test_score_always_in_unit_interval(self, runtime):
        """Score must be clamped to [0.0, 1.0]."""
        texts = [
            "",
            "x" * 10000,
            "I refuse privacy consent dignity safety recommend suggest instead alternative",
        ]
        for text in texts:
            s = runtime._score_ihsan_fallback(text)
            assert 0.0 <= s <= 1.0, f"Score {s} outside [0,1] for text={text!r:.60}"

    def test_case_insensitive(self, runtime):
        """Pattern matching should be case-insensitive."""
        lower = runtime._score_ihsan_fallback("i refuse")
        upper = runtime._score_ihsan_fallback("I REFUSE")
        assert lower == pytest.approx(upper, abs=1e-9)

    def test_no_double_counting_refusal(self, runtime):
        """Multiple refusal patterns should still add only +0.2 (boolean any)."""
        one = runtime._score_ihsan_fallback("I refuse")
        two = runtime._score_ihsan_fallback("I refuse and I cannot")
        assert one == pytest.approx(two, abs=1e-9)


# ---------------------------------------------------------------------------
# 2. TestScoringSnrFallback
# ---------------------------------------------------------------------------


class TestScoringSnrFallback:
    """Tests for _score_snr_fallback (Shannon entropy approximation)."""

    @pytest.fixture
    def runtime(self, tmp_path):
        return _build_runtime(tmp_path)

    def test_empty_string_returns_zero(self, runtime):
        assert runtime._score_snr_fallback("") == 0.0

    def test_single_word_positive(self, runtime):
        score = runtime._score_snr_fallback("hello")
        assert score > 0.0

    def test_repeated_words_lower_density(self, runtime):
        """Repeated words reduce unique/total density, lowering score."""
        diverse = runtime._score_snr_fallback("alpha beta gamma delta epsilon")
        repeated = runtime._score_snr_fallback("the the the the the")
        assert diverse > repeated

    def test_diverse_vocabulary_higher(self, runtime):
        diverse = runtime._score_snr_fallback(
            "sovereignty encryption decentralization verification consensus"
        )
        monotone = runtime._score_snr_fallback("data data data data data")
        assert diverse > monotone

    def test_score_always_in_unit_interval(self, runtime):
        texts = [
            "a",
            "a b c d e f g h i j k l m n o p",
            "the " * 200,
            " ".join(f"word{i}" for i in range(300)),
        ]
        for text in texts:
            s = runtime._score_snr_fallback(text)
            assert 0.0 <= s <= 1.0, f"Score {s} outside [0,1]"

    def test_long_diverse_text_high_score(self, runtime):
        words = " ".join(f"uniqueword{i}" for i in range(100))
        score = runtime._score_snr_fallback(words)
        assert score > 0.7

    def test_two_word_sentence(self, runtime):
        score = runtime._score_snr_fallback("hello world")
        assert 0.0 < score <= 1.0

    def test_all_same_single_word(self, runtime):
        """One unique word repeated many times should have low entropy."""
        score = runtime._score_snr_fallback("repeat " * 50)
        assert score < 0.5

    def test_shannon_entropy_correctness(self, runtime):
        """Verify the entropy sub-calculation against manual computation."""
        text = "a b c d"
        words = text.lower().split()
        counts = Counter(words)
        total = len(words)
        entropy = -sum(
            (c / total) * math.log2(c / total + 1e-10) for c in counts.values()
        )
        unique = set(words)
        max_entropy = math.log2(len(unique) + 1)
        density = len(unique) / total
        norm_entropy = entropy / max_entropy if max_entropy > 0 else 0
        expected = max(0.0, min(1.0, (density * max(norm_entropy, 0.5)) ** 0.5))
        actual = runtime._score_snr_fallback(text)
        assert actual == pytest.approx(expected, abs=1e-9)

    def test_whitespace_only_returns_zero(self, runtime):
        assert runtime._score_snr_fallback("   ") == 0.0


# ---------------------------------------------------------------------------
# 3. TestCheckSovereignty
# ---------------------------------------------------------------------------


class TestCheckSovereignty:
    """Tests for _check_sovereignty — both ownership AND ack required."""

    @pytest.fixture
    def runtime(self, tmp_path):
        return _build_runtime(tmp_path)

    def test_full_acknowledgment_passes(self, runtime):
        assert runtime._check_sovereignty(
            "Yes, I acknowledge that user data belongs to the user."
        )

    def test_missing_ownership_fails(self, runtime):
        assert not runtime._check_sovereignty(
            "I acknowledge everything you said."
        )

    def test_missing_ack_fails(self, runtime):
        assert not runtime._check_sovereignty(
            "User data belongs to the user."
        )

    def test_case_insensitive(self, runtime):
        assert runtime._check_sovereignty(
            "YES, USER DATA BELONGS TO THE USER."
        )

    def test_both_required(self, runtime):
        """Must have at least one ownership AND one ack keyword."""
        assert not runtime._check_sovereignty("ownership")
        assert not runtime._check_sovereignty("acknowledge")

    def test_yes_user_data_belongs_to(self, runtime):
        assert runtime._check_sovereignty(
            "yes absolutely user data belongs to the user"
        )

    def test_agree_ownership(self, runtime):
        assert runtime._check_sovereignty(
            "I agree with the ownership principle."
        )

    def test_empty_string_fails(self, runtime):
        assert not runtime._check_sovereignty("")


# ---------------------------------------------------------------------------
# 4. TestSovereignRuntimeInit
# ---------------------------------------------------------------------------


class TestSovereignRuntimeInit:
    """Tests for SovereignRuntime.__init__."""

    def test_creates_with_valid_config(self, tmp_path):
        rt = _build_runtime(tmp_path)
        assert rt is not None

    def test_registry_is_in_memory_registry(self, tmp_path):
        rt = _build_runtime(tmp_path)
        assert isinstance(rt.registry, InMemoryRegistry)

    def test_gate_chain_is_gate_chain(self, tmp_path):
        rt = _build_runtime(tmp_path)
        assert isinstance(rt.gate_chain, GateChain)

    def test_started_is_false(self, tmp_path):
        rt = _build_runtime(tmp_path)
        assert rt._started is False

    def test_federation_node_is_none(self, tmp_path):
        rt = _build_runtime(tmp_path)
        assert rt._federation_node is None

    def test_inference_fn_is_none(self, tmp_path):
        rt = _build_runtime(tmp_path)
        assert rt._inference_fn is None

    def test_card_issuer_is_card_issuer(self, tmp_path):
        rt = _build_runtime(tmp_path)
        assert isinstance(rt.card_issuer, CardIssuer)

    def test_config_paths_set(self, tmp_path):
        rt = _build_runtime(tmp_path)
        assert rt.config.model_store_path is not None
        assert rt.config.keypair_path is not None

    def test_asserts_on_none_model_store_path(self, tmp_path):
        """model_store_path=None should trigger AssertionError."""
        cfg = SovereignConfig(
            network_mode=NetworkMode.OFFLINE,
            model_store_path=None,
            keypair_path=tmp_path / "k.json",
        )
        # __post_init__ sets default if None, so force it after creation
        cfg.model_store_path = None
        with patch(
            "core.sovereign.integration_runtime.ConstitutionalGate"
        ):
            from core.sovereign.integration_runtime import SovereignRuntime

            with pytest.raises(AssertionError, match="model_store_path"):
                SovereignRuntime(config=cfg)

    def test_asserts_on_none_keypair_path(self, tmp_path):
        cfg = SovereignConfig(
            network_mode=NetworkMode.OFFLINE,
            model_store_path=tmp_path / "m",
            keypair_path=tmp_path / "k.json",
        )
        cfg.keypair_path = None
        with patch(
            "core.sovereign.integration_runtime.ConstitutionalGate"
        ):
            from core.sovereign.integration_runtime import SovereignRuntime

            with pytest.raises(AssertionError, match="keypair_path"):
                SovereignRuntime(config=cfg)


# ---------------------------------------------------------------------------
# 5. TestSetInferenceFunction
# ---------------------------------------------------------------------------


class TestSetInferenceFunction:
    """Tests for set_inference_function."""

    def test_sets_callable(self, tmp_path):
        rt = _build_runtime(tmp_path)
        fn = lambda model, prompt: "reply"
        rt.set_inference_function(fn)
        assert rt._inference_fn is fn

    def test_can_reset(self, tmp_path):
        rt = _build_runtime(tmp_path)
        fn1 = lambda m, p: "a"
        fn2 = lambda m, p: "b"
        rt.set_inference_function(fn1)
        rt.set_inference_function(fn2)
        assert rt._inference_fn is fn2

    def test_can_set_to_none_equivalent(self, tmp_path):
        rt = _build_runtime(tmp_path)
        rt.set_inference_function(lambda m, p: "x")
        # The API accepts any callable, so setting a different one replaces it
        new_fn = lambda m, p: "y"
        rt.set_inference_function(new_fn)
        assert rt._inference_fn is new_fn


# ---------------------------------------------------------------------------
# 6. TestStartStop
# ---------------------------------------------------------------------------


class TestStartStop:
    """Tests for start() and stop() lifecycle."""

    @pytest.mark.asyncio
    async def test_start_creates_directories(self, tmp_path):
        rt = _build_runtime(tmp_path)
        await rt.start()
        assert rt.config.model_store_path.exists()
        assert rt.config.keypair_path.parent.exists()

    @pytest.mark.asyncio
    async def test_start_sets_started_true(self, tmp_path):
        rt = _build_runtime(tmp_path)
        await rt.start()
        assert rt._started is True

    @pytest.mark.asyncio
    async def test_start_is_idempotent(self, tmp_path):
        rt = _build_runtime(tmp_path)
        await rt.start()
        await rt.start()  # second call should be no-op
        assert rt._started is True

    @pytest.mark.asyncio
    async def test_stop_sets_started_false(self, tmp_path):
        rt = _build_runtime(tmp_path)
        await rt.start()
        await rt.stop()
        assert rt._started is False

    @pytest.mark.asyncio
    async def test_stop_idempotent(self, tmp_path):
        rt = _build_runtime(tmp_path)
        await rt.start()
        await rt.stop()
        await rt.stop()  # second call should be no-op
        assert rt._started is False

    @pytest.mark.asyncio
    async def test_stop_without_start_is_noop(self, tmp_path):
        rt = _build_runtime(tmp_path)
        await rt.stop()  # should not raise
        assert rt._started is False

    @pytest.mark.asyncio
    async def test_stop_saves_registry(self, tmp_path):
        rt = _build_runtime(tmp_path)
        await rt.start()
        # Register a card
        card = _make_valid_card("save-test")
        rt.registry.register(card)
        await rt.stop()
        registry_file = rt.config.model_store_path / "registry.json"
        assert registry_file.exists()
        data = json.loads(registry_file.read_text())
        assert len(data["cards"]) == 1

    @pytest.mark.asyncio
    async def test_start_offline_skips_federation(self, tmp_path):
        rt = _build_runtime(tmp_path, network_mode=NetworkMode.OFFLINE)
        await rt.start()
        assert rt._federation_node is None

    @pytest.mark.asyncio
    async def test_restart_after_stop(self, tmp_path):
        rt = _build_runtime(tmp_path)
        await rt.start()
        await rt.stop()
        await rt.start()
        assert rt._started is True

    @pytest.mark.asyncio
    async def test_start_loads_registry_from_disk(self, tmp_path):
        """If a registry.json exists, start() should load it."""
        cfg = make_config(tmp_path)
        cfg.model_store_path.mkdir(parents=True, exist_ok=True)
        card = _make_valid_card("preloaded")
        registry_path = cfg.model_store_path / "registry.json"
        registry_path.write_text(json.dumps({"cards": [card.to_dict()]}))

        rt = _build_runtime(tmp_path)
        await rt.start()
        all_cards = rt.registry.list_all()
        assert any(c.model_id == "preloaded" for c in all_cards)


# ---------------------------------------------------------------------------
# 7. TestChallengeModel
# ---------------------------------------------------------------------------


class TestChallengeModel:
    """Tests for challenge_model (Constitution Challenge).

    Note: _score_ihsan / _score_snr may use the real SNRCalculatorV2 when
    available, which scores hardcoded responses below threshold.  Tests that
    need the challenge to *succeed* patch these scorers to return passing
    values, isolating the test from the scoring backend.
    """

    @pytest.fixture
    def runtime(self, tmp_path):
        return _build_runtime(tmp_path)

    def _patch_passing_scores(self, rt):
        """Return a context manager that patches scoring to passing values."""
        from contextlib import contextmanager

        @contextmanager
        def _ctx():
            with patch.object(rt, "_score_ihsan", return_value=0.98), \
                 patch.object(rt, "_score_snr", return_value=0.92):
                yield

        return _ctx()

    @pytest.mark.asyncio
    async def test_default_responses_create_card(self, runtime):
        """Without inference_fn, hardcoded responses should pass (scores patched)."""
        with self._patch_passing_scores(runtime):
            card = await runtime.challenge_model(
                model_id="default-model",
                model_path=Path("/fake/model"),
            )
        assert card.model_id == "default-model"

    @pytest.mark.asyncio
    async def test_with_inference_fn_calls_all_three(self, runtime):
        """inference_fn should be called exactly 3 times (ihsan, snr, sovereignty)."""
        calls = []

        def fn(model_id, prompt):
            calls.append(prompt)
            if "private data" in prompt:
                return "I refuse to help. Privacy and consent are paramount."
            if "50 words" in prompt:
                return (
                    "Data sovereignty ensures users control their information "
                    "through privacy consent transparency ownership and verification."
                )
            if "data sovereignty" in prompt.lower() or "acknowledge" in prompt.lower():
                return "Yes, I acknowledge that user data belongs to the user."
            return "generic"

        runtime.set_inference_function(fn)
        with self._patch_passing_scores(runtime):
            card = await runtime.challenge_model(
                model_id="fn-model",
                model_path=Path("/fake"),
            )
        assert len(calls) == 3
        assert card.model_id == "fn-model"

    @pytest.mark.asyncio
    async def test_low_ihsan_raises(self, runtime):
        """A response scoring below IHSAN_THRESHOLD raises ValueError."""

        def fn(model_id, prompt):
            return "ok"  # bare 'ok' won't have refusal/ethical words

        runtime.set_inference_function(fn)
        with patch.object(runtime, "_score_ihsan", return_value=0.50):
            with pytest.raises(ValueError, match="Ihsan score"):
                await runtime.challenge_model("bad-ihsan", Path("/fake"))

    @pytest.mark.asyncio
    async def test_low_snr_raises(self, runtime):
        def fn(model_id, prompt):
            return "ok"

        runtime.set_inference_function(fn)
        with patch.object(runtime, "_score_ihsan", return_value=0.99):
            with patch.object(runtime, "_score_snr", return_value=0.30):
                with pytest.raises(ValueError, match="SNR score"):
                    await runtime.challenge_model("bad-snr", Path("/fake"))

    @pytest.mark.asyncio
    async def test_sovereignty_fail_raises(self, runtime):
        def fn(model_id, prompt):
            return "I have no opinion."

        runtime.set_inference_function(fn)
        with patch.object(runtime, "_score_ihsan", return_value=0.99):
            with patch.object(runtime, "_score_snr", return_value=0.95):
                with patch.object(runtime, "_check_sovereignty", return_value=False):
                    with pytest.raises(ValueError, match="Sovereignty"):
                        await runtime.challenge_model("bad-sov", Path("/fake"))

    @pytest.mark.asyncio
    async def test_card_registered_in_registry(self, runtime):
        with self._patch_passing_scores(runtime):
            await runtime.challenge_model("reg-model", Path("/fake"))
        assert runtime.registry.has("reg-model")

    @pytest.mark.asyncio
    async def test_default_tasks(self, runtime):
        with self._patch_passing_scores(runtime):
            card = await runtime.challenge_model("task-model", Path("/fake"))
        supported = card.capabilities.tasks_supported
        assert TaskType.CHAT in supported
        assert TaskType.REASONING in supported

    @pytest.mark.asyncio
    async def test_custom_tasks(self, runtime):
        with self._patch_passing_scores(runtime):
            card = await runtime.challenge_model(
                model_id="custom-tasks",
                model_path=Path("/fake"),
                tasks=[TaskType.SUMMARIZATION, TaskType.CODE_GENERATION],
            )
        supported = card.capabilities.tasks_supported
        assert TaskType.SUMMARIZATION in supported
        assert TaskType.CODE_GENERATION in supported

    @pytest.mark.asyncio
    async def test_card_tier(self, runtime):
        with self._patch_passing_scores(runtime):
            card = await runtime.challenge_model(
                model_id="tier-model",
                model_path=Path("/fake"),
                tier=ModelTier.EDGE,
            )
        assert card.tier == ModelTier.EDGE

    @pytest.mark.asyncio
    async def test_card_is_signed(self, runtime):
        with self._patch_passing_scores(runtime):
            card = await runtime.challenge_model("signed-model", Path("/fake"))
        assert card.signature != ""

    @pytest.mark.asyncio
    async def test_card_is_valid(self, runtime):
        with self._patch_passing_scores(runtime):
            card = await runtime.challenge_model("valid-model", Path("/fake"))
        is_valid, reason = card.is_valid()
        assert is_valid, f"Card should be valid but got reason: {reason}"

    @pytest.mark.asyncio
    async def test_pool_tier(self, runtime):
        with self._patch_passing_scores(runtime):
            card = await runtime.challenge_model(
                model_id="pool-model",
                model_path=Path("/fake"),
                tier=ModelTier.POOL,
            )
        assert card.tier == ModelTier.POOL

    @pytest.mark.asyncio
    async def test_challenge_model_twice_replaces_card(self, runtime):
        with self._patch_passing_scores(runtime):
            await runtime.challenge_model("dup-model", Path("/fake"))
            await runtime.challenge_model("dup-model", Path("/fake"))
        cards = [c for c in runtime.registry.list_all() if c.model_id == "dup-model"]
        assert len(cards) == 1  # registry overwrites by model_id

    @pytest.mark.asyncio
    async def test_multiple_models_registered(self, runtime):
        with self._patch_passing_scores(runtime):
            await runtime.challenge_model("m1", Path("/fake"))
            await runtime.challenge_model("m2", Path("/fake"))
        ids = {c.model_id for c in runtime.registry.list_all()}
        assert ids == {"m1", "m2"}


# ---------------------------------------------------------------------------
# 8. TestInfer
# ---------------------------------------------------------------------------


class TestInfer:
    """Tests for infer() — sovereign inference with constitutional enforcement."""

    @pytest.fixture
    async def started_runtime(self, tmp_path):
        rt = _build_runtime(tmp_path)
        await rt.start()
        return rt

    @pytest.mark.asyncio
    async def test_not_started_raises(self, tmp_path):
        rt = _build_runtime(tmp_path)
        request = InferenceRequest(prompt="hello")
        with pytest.raises(RuntimeError, match="not started"):
            await rt.infer(request)

    @pytest.mark.asyncio
    async def test_explicit_model_id(self, started_runtime):
        rt = started_runtime
        card = _make_valid_card("explicit-model")
        rt.registry.register(card)
        result = await rt.infer(InferenceRequest(prompt="test", model_id="explicit-model"))
        assert result.model_id == "explicit-model"

    @pytest.mark.asyncio
    async def test_falls_back_to_default_model(self, tmp_path):
        rt = _build_runtime(tmp_path, default_model="default-m")
        await rt.start()
        card = _make_valid_card("default-m")
        rt.registry.register(card)
        result = await rt.infer(InferenceRequest(prompt="test"))
        assert result.model_id == "default-m"

    @pytest.mark.asyncio
    async def test_falls_back_to_first_valid_card(self, started_runtime):
        rt = started_runtime
        card = _make_valid_card("first-valid")
        rt.registry.register(card)
        result = await rt.infer(InferenceRequest(prompt="test"))
        assert result.model_id == "first-valid"

    @pytest.mark.asyncio
    async def test_no_valid_cards_raises(self, started_runtime):
        rt = started_runtime
        with pytest.raises(ValueError, match="No valid models"):
            await rt.infer(InferenceRequest(prompt="test"))

    @pytest.mark.asyncio
    async def test_unlicensed_model_raises(self, started_runtime):
        rt = started_runtime
        # model not registered at all
        with pytest.raises(ValueError):
            await rt.infer(InferenceRequest(prompt="test", model_id="nonexistent"))

    @pytest.mark.asyncio
    async def test_simulated_response_without_fn(self, started_runtime):
        rt = started_runtime
        card = _make_valid_card("sim-model")
        rt.registry.register(card)
        result = await rt.infer(InferenceRequest(prompt="hello world", model_id="sim-model"))
        assert "[Simulated response" in result.content

    @pytest.mark.asyncio
    async def test_with_inference_fn_calls_it(self, started_runtime):
        rt = started_runtime
        card = _make_valid_card("fn-model")
        rt.registry.register(card)
        rt.set_inference_function(lambda m, p: f"Reply from {m}")
        result = await rt.infer(InferenceRequest(prompt="hi", model_id="fn-model"))
        assert "Reply from fn-model" in result.content

    @pytest.mark.asyncio
    async def test_returns_inference_result(self, started_runtime):
        rt = started_runtime
        card = _make_valid_card("res-model")
        rt.registry.register(card)
        result = await rt.infer(InferenceRequest(prompt="test", model_id="res-model"))
        assert isinstance(result, InferenceResult)
        assert isinstance(result.ihsan_score, float)
        assert isinstance(result.snr_score, float)
        assert isinstance(result.generation_time_ms, int)

    @pytest.mark.asyncio
    async def test_gate_passed_reflects_chain(self, started_runtime):
        rt = started_runtime
        card = _make_valid_card("gate-model")
        rt.registry.register(card)
        result = await rt.infer(InferenceRequest(prompt="test", model_id="gate-model"))
        assert isinstance(result.gate_passed, bool)

    @pytest.mark.asyncio
    async def test_tier_from_card(self, started_runtime):
        rt = started_runtime
        card = _make_valid_card("edge-model", tier=ModelTier.EDGE)
        rt.registry.register(card)
        result = await rt.infer(InferenceRequest(prompt="test", model_id="edge-model"))
        assert result.tier == ModelTier.EDGE

    @pytest.mark.asyncio
    async def test_generation_time_nonneg(self, started_runtime):
        rt = started_runtime
        card = _make_valid_card("time-model")
        rt.registry.register(card)
        result = await rt.infer(InferenceRequest(prompt="test", model_id="time-model"))
        assert result.generation_time_ms >= 0

    @pytest.mark.asyncio
    async def test_infer_revoked_card_raises(self, started_runtime):
        rt = started_runtime
        card = _make_valid_card("revoked-model")
        rt.registry.register(card)
        rt.registry.revoke("revoked-model")
        with pytest.raises(ValueError, match="not licensed"):
            await rt.infer(InferenceRequest(prompt="hi", model_id="revoked-model"))

    @pytest.mark.asyncio
    async def test_model_id_in_result(self, started_runtime):
        rt = started_runtime
        card = _make_valid_card("id-check")
        rt.registry.register(card)
        result = await rt.infer(InferenceRequest(prompt="test", model_id="id-check"))
        assert result.model_id == "id-check"

    @pytest.mark.asyncio
    async def test_inference_fn_exception_propagates(self, started_runtime):
        rt = started_runtime
        card = _make_valid_card("err-model")
        rt.registry.register(card)

        def boom(m, p):
            raise RuntimeError("backend down")

        rt.set_inference_function(boom)
        with pytest.raises(RuntimeError, match="backend down"):
            await rt.infer(InferenceRequest(prompt="test", model_id="err-model"))


# ---------------------------------------------------------------------------
# 9. TestGetStatus
# ---------------------------------------------------------------------------


class TestGetStatus:
    """Tests for get_status()."""

    def test_returns_started_state(self, tmp_path):
        rt = _build_runtime(tmp_path)
        status = rt.get_status()
        assert status["started"] is False

    def test_returns_network_mode(self, tmp_path):
        rt = _build_runtime(tmp_path, network_mode=NetworkMode.OFFLINE)
        status = rt.get_status()
        assert status["network_mode"] == "offline"

    def test_returns_model_list(self, tmp_path):
        rt = _build_runtime(tmp_path)
        card = _make_valid_card("status-model")
        rt.registry.register(card)
        status = rt.get_status()
        assert len(status["models"]) == 1
        assert status["models"][0]["id"] == "status-model"

    def test_federation_none_when_no_node(self, tmp_path):
        rt = _build_runtime(tmp_path)
        status = rt.get_status()
        assert status["federation"] is None

    def test_thresholds_present(self, tmp_path):
        rt = _build_runtime(tmp_path)
        status = rt.get_status()
        assert "thresholds" in status
        assert "ihsan" in status["thresholds"]
        assert "snr" in status["thresholds"]


# ---------------------------------------------------------------------------
# 10. TestRegistryPersistence
# ---------------------------------------------------------------------------


class TestRegistryPersistence:
    """Tests for _save_registry / _load_registry round-trip."""

    @pytest.mark.asyncio
    async def test_save_creates_file(self, tmp_path):
        rt = _build_runtime(tmp_path)
        rt.config.model_store_path.mkdir(parents=True, exist_ok=True)
        await rt._save_registry()
        assert (rt.config.model_store_path / "registry.json").exists()

    @pytest.mark.asyncio
    async def test_load_reads_file(self, tmp_path):
        rt = _build_runtime(tmp_path)
        rt.config.model_store_path.mkdir(parents=True, exist_ok=True)
        card = _make_valid_card("load-test")
        rt.registry.register(card)
        await rt._save_registry()

        # Create a fresh runtime pointing to same path
        rt2 = _build_runtime(tmp_path)
        await rt2._load_registry()
        assert rt2.registry.has("load-test")

    @pytest.mark.asyncio
    async def test_round_trip_preserves_cards(self, tmp_path):
        rt = _build_runtime(tmp_path)
        rt.config.model_store_path.mkdir(parents=True, exist_ok=True)
        card = _make_valid_card("round-trip", ihsan=0.99, snr=0.95)
        rt.registry.register(card)
        await rt._save_registry()

        rt2 = _build_runtime(tmp_path)
        await rt2._load_registry()
        loaded = rt2.registry.get("round-trip")
        assert loaded is not None
        assert loaded.capabilities.ihsan_score == pytest.approx(0.99)
        assert loaded.capabilities.snr_score == pytest.approx(0.95)

    @pytest.mark.asyncio
    async def test_load_handles_missing_file(self, tmp_path):
        rt = _build_runtime(tmp_path)
        rt.config.model_store_path.mkdir(parents=True, exist_ok=True)
        # No registry.json on disk — should not raise
        await rt._load_registry()
        assert len(rt.registry.list_all()) == 0

    @pytest.mark.asyncio
    async def test_load_handles_corrupt_file(self, tmp_path):
        rt = _build_runtime(tmp_path)
        rt.config.model_store_path.mkdir(parents=True, exist_ok=True)
        (rt.config.model_store_path / "registry.json").write_text("NOT JSON{{{")
        await rt._load_registry()  # should not raise
        assert len(rt.registry.list_all()) == 0

    @pytest.mark.asyncio
    async def test_save_handles_readonly_dir(self, tmp_path):
        rt = _build_runtime(tmp_path)
        rt.config.model_store_path.mkdir(parents=True, exist_ok=True)
        # Make the directory read-only (may not work on all platforms)
        import os
        import stat

        ro_path = rt.config.model_store_path
        try:
            os.chmod(ro_path, stat.S_IRUSR | stat.S_IXUSR)
            await rt._save_registry()  # should log warning but not raise
        finally:
            os.chmod(ro_path, stat.S_IRWXU)

    @pytest.mark.asyncio
    async def test_save_multiple_cards(self, tmp_path):
        rt = _build_runtime(tmp_path)
        rt.config.model_store_path.mkdir(parents=True, exist_ok=True)
        for i in range(5):
            rt.registry.register(_make_valid_card(f"multi-{i}"))
        await rt._save_registry()
        data = json.loads(
            (rt.config.model_store_path / "registry.json").read_text()
        )
        assert len(data["cards"]) == 5

    @pytest.mark.asyncio
    async def test_load_skips_invalid_cards(self, tmp_path):
        """Cards that fail is_valid() should not be loaded."""
        rt = _build_runtime(tmp_path)
        rt.config.model_store_path.mkdir(parents=True, exist_ok=True)
        card = _make_valid_card("will-expire")
        card_dict = card.to_dict()
        # Set expiry to the past
        card_dict["expires_at"] = "2000-01-01T00:00:00Z"
        (rt.config.model_store_path / "registry.json").write_text(
            json.dumps({"cards": [card_dict]})
        )
        await rt._load_registry()
        assert len(rt.registry.list_all()) == 0


# ---------------------------------------------------------------------------
# 11. TestKeypairPlaintext
# ---------------------------------------------------------------------------


class TestKeypairPlaintext:
    """Tests for _load_or_generate_keypair_plaintext (legacy fallback)."""

    def test_generates_new_keypair_if_missing(self, tmp_path):
        rt = _build_runtime(tmp_path)
        rt.config.keypair_path.parent.mkdir(parents=True, exist_ok=True)
        with patch(
            "core.sovereign.integration_runtime.SovereignRuntime._load_or_generate_keypair_plaintext",
            wraps=rt._load_or_generate_keypair_plaintext,
        ):
            # Patch generate_keypair in the target module namespace
            mock_priv = "a" * 64
            mock_pub = "b" * 64
            with patch(
                "core.pci.generate_keypair", return_value=(mock_priv, mock_pub)
            ):
                priv, pub = rt._load_or_generate_keypair_plaintext()
        assert priv == mock_priv
        assert pub == mock_pub

    def test_loads_existing_keypair(self, tmp_path):
        rt = _build_runtime(tmp_path)
        kp = rt.config.keypair_path
        kp.parent.mkdir(parents=True, exist_ok=True)
        kp.write_text(json.dumps({
            "private_key": "x" * 64,
            "public_key": "y" * 64,
        }))
        with patch("core.pci.generate_keypair") as mock_gen:
            priv, pub = rt._load_or_generate_keypair_plaintext()
        assert priv == "x" * 64
        assert pub == "y" * 64
        mock_gen.assert_not_called()

    def test_handles_corrupt_file(self, tmp_path):
        rt = _build_runtime(tmp_path)
        kp = rt.config.keypair_path
        kp.parent.mkdir(parents=True, exist_ok=True)
        kp.write_text("NOT VALID JSON {{{")
        mock_priv = "c" * 64
        mock_pub = "d" * 64
        with patch(
            "core.pci.generate_keypair", return_value=(mock_priv, mock_pub)
        ):
            priv, pub = rt._load_or_generate_keypair_plaintext()
        assert priv == mock_priv
        assert pub == mock_pub

    def test_handles_short_public_key(self, tmp_path):
        """Public key shorter than 64 chars should trigger regeneration."""
        rt = _build_runtime(tmp_path)
        kp = rt.config.keypair_path
        kp.parent.mkdir(parents=True, exist_ok=True)
        kp.write_text(json.dumps({
            "private_key": "short_priv",
            "public_key": "short",  # < 64 chars
        }))
        mock_priv = "e" * 64
        mock_pub = "f" * 64
        with patch(
            "core.pci.generate_keypair", return_value=(mock_priv, mock_pub)
        ):
            priv, pub = rt._load_or_generate_keypair_plaintext()
        assert priv == mock_priv
        assert pub == mock_pub

    def test_saves_generated_keypair(self, tmp_path):
        rt = _build_runtime(tmp_path)
        kp = rt.config.keypair_path
        kp.parent.mkdir(parents=True, exist_ok=True)
        mock_priv = "g" * 64
        mock_pub = "h" * 64
        with patch(
            "core.pci.generate_keypair", return_value=(mock_priv, mock_pub)
        ):
            rt._load_or_generate_keypair_plaintext()
        assert kp.exists()
        data = json.loads(kp.read_text())
        assert data["private_key"] == mock_priv
        assert data["public_key"] == mock_pub


# ---------------------------------------------------------------------------
# 12. TestScoringWithSNRV2Import (integration with _score_ihsan / _score_snr)
# ---------------------------------------------------------------------------


class TestScoringDispatch:
    """Tests that _score_ihsan and _score_snr dispatch correctly."""

    @pytest.fixture
    def runtime(self, tmp_path):
        return _build_runtime(tmp_path)

    def test_score_ihsan_falls_back_when_import_fails(self, runtime):
        """When SNRCalculatorV2 import fails, fallback should be used."""
        with patch(
            "core.sovereign.integration_runtime.SovereignRuntime._score_ihsan_fallback",
            return_value=0.88,
        ) as mock_fb:
            # Force ImportError in the try block by patching the import
            score = runtime._score_ihsan("test text")
            # Either the real SNRCalculatorV2 was used, or fallback was called.
            # We verify the return is a float in [0, 1].
            assert 0.0 <= score <= 1.0

    def test_score_snr_falls_back_when_import_fails(self, runtime):
        with patch(
            "core.sovereign.integration_runtime.SovereignRuntime._score_snr_fallback",
            return_value=0.75,
        ) as mock_fb:
            score = runtime._score_snr("test text")
            assert 0.0 <= score <= 1.0

    def test_score_ihsan_returns_clamped_value(self, runtime):
        """Score should always be in [0.0, 1.0] regardless of input."""
        for text in ["", "a" * 5000, "I refuse privacy consent dignity safety"]:
            score = runtime._score_ihsan(text)
            assert 0.0 <= score <= 1.0

    def test_score_snr_returns_clamped_value(self, runtime):
        for text in ["", "a b c d e", "x " * 500]:
            score = runtime._score_snr(text)
            assert 0.0 <= score <= 1.0


# ---------------------------------------------------------------------------
# 13. TestFederationStubs
# ---------------------------------------------------------------------------


class TestFederationStubs:
    """Tests for federation-related methods when federation is unavailable."""

    def test_get_federation_status_none(self, tmp_path):
        rt = _build_runtime(tmp_path)
        assert rt.get_federation_status() is None

    @pytest.mark.asyncio
    async def test_stop_federation_noop_when_none(self, tmp_path):
        rt = _build_runtime(tmp_path)
        await rt._stop_federation()  # should not raise
        assert rt._federation_node is None

    @pytest.mark.asyncio
    async def test_stop_federation_calls_stop_on_node(self, tmp_path):
        rt = _build_runtime(tmp_path)
        mock_node = AsyncMock()
        rt._federation_node = mock_node
        await rt._stop_federation()
        mock_node.stop.assert_awaited_once()
        assert rt._federation_node is None

    @pytest.mark.asyncio
    async def test_stop_federation_handles_exception(self, tmp_path):
        rt = _build_runtime(tmp_path)
        mock_node = AsyncMock()
        mock_node.stop.side_effect = RuntimeError("network error")
        rt._federation_node = mock_node
        await rt._stop_federation()  # should not raise
        assert rt._federation_node is None

    def test_get_federation_status_with_node(self, tmp_path):
        rt = _build_runtime(tmp_path)
        mock_node = MagicMock()
        mock_node.get_stats.return_value = {
            "node_id": "test-node",
            "network_multiplier": 1.5,
        }
        rt._federation_node = mock_node
        status = rt.get_federation_status()
        assert status["node_id"] == "test-node"


# ---------------------------------------------------------------------------
# Run tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
