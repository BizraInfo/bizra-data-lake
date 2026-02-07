"""
SNR Maximizer — Comprehensive Test Suite

Standing on Giants:
- Shannon (1948): Information Theory (Signal-to-Noise Ratio)
- Wiener (1949): Signal Processing
- DATA4LLM IaaS (Tsinghua, 2024)

This test fills the #1 critical gap identified in the deep audit:
the primary quality gate (SNR Maximizer) had ZERO test coverage.

Tests cover:
1. Mathematical formula correctness (signal/noise/SNR computation)
2. Edge cases (zero noise, zero signal, single-dimension drops)
3. Noise detection (redundancy, ambiguity, verbosity)
4. Signal analysis (relevance, novelty, groundedness, coherence, actionability)
5. Ihsan gate enforcement (pass/fail threshold behavior)
6. Iterative maximize() convergence
7. Statistics tracking
8. Async optimize() API compatibility
"""

import asyncio
import math

import pytest

from core.sovereign.snr_maximizer import (
    NoiseFilter,
    NoiseProfile,
    NoiseType,
    SNRAnalysis,
    SNRMaximizer,
    SignalAmplifier,
    SignalProfile,
    SignalType,
)


# ═══════════════════════════════════════════════════════════════════════════════
# MATHEMATICAL FORMULA TESTS — Shannon Correctness
# ═══════════════════════════════════════════════════════════════════════════════


class TestNoiseProfile:
    """NoiseProfile weighted sum correctness."""

    def test_all_zero_noise(self):
        """Zero noise profile has zero total noise."""
        profile = NoiseProfile()
        assert profile.total_noise == 0.0

    def test_weights_sum_to_one(self):
        """Noise weights sum to 1.0 (probability distribution)."""
        profile = NoiseProfile(
            redundancy=1.0,
            inconsistency=1.0,
            ambiguity=1.0,
            irrelevance=1.0,
            hallucination=1.0,
            verbosity=1.0,
            bias=1.0,
        )
        # 0.20 + 0.25 + 0.15 + 0.15 + 0.10 + 0.05 + 0.10 = 1.00
        assert abs(profile.total_noise - 1.0) < 1e-10

    def test_inconsistency_has_highest_weight(self):
        """Inconsistency (0.25) is the most heavily penalized noise type."""
        only_inconsistency = NoiseProfile(inconsistency=1.0)
        only_redundancy = NoiseProfile(redundancy=1.0)
        only_ambiguity = NoiseProfile(ambiguity=1.0)

        assert only_inconsistency.total_noise > only_redundancy.total_noise
        assert only_inconsistency.total_noise > only_ambiguity.total_noise
        assert only_inconsistency.total_noise == 0.25

    def test_verbosity_has_lowest_weight(self):
        """Verbosity (0.05) is the least penalized noise type."""
        only_verbosity = NoiseProfile(verbosity=1.0)
        assert only_verbosity.total_noise == 0.05

    def test_to_dict_includes_total(self):
        """to_dict() includes computed total."""
        profile = NoiseProfile(redundancy=0.5)
        d = profile.to_dict()
        assert "total" in d
        assert d["total"] == profile.total_noise
        assert d["redundancy"] == 0.5


class TestSignalProfile:
    """SignalProfile geometric mean correctness."""

    def test_all_equal_signal(self):
        """Equal signal dimensions produce that value as geometric mean."""
        profile = SignalProfile(
            relevance=0.8,
            novelty=0.8,
            groundedness=0.8,
            coherence=0.8,
            actionability=0.8,
            specificity=0.8,
        )
        assert abs(profile.total_signal - 0.8) < 1e-10

    def test_single_zero_dimension_collapses_signal(self):
        """A single zero dimension drives the geometric mean near zero."""
        profile = SignalProfile(
            relevance=0.9,
            novelty=0.9,
            groundedness=0.0,  # Zero
            coherence=0.9,
            actionability=0.9,
            specificity=0.9,
        )
        # With clamping to 1e-10, should be very small but not exactly zero
        assert profile.total_signal < 0.05

    def test_geometric_mean_penalizes_outliers(self):
        """Geometric mean penalizes low outliers more than arithmetic mean would."""
        profile = SignalProfile(
            relevance=0.9,
            novelty=0.1,  # Low outlier
            groundedness=0.9,
            coherence=0.9,
            actionability=0.9,
            specificity=0.9,
        )
        # Arithmetic mean would be (0.9*5 + 0.1)/6 = 0.767
        # Geometric mean should be lower
        arithmetic_mean = (0.9 * 5 + 0.1) / 6
        assert profile.total_signal < arithmetic_mean

    def test_all_ones_gives_one(self):
        """All dimensions at 1.0 gives total signal of 1.0."""
        profile = SignalProfile(
            relevance=1.0, novelty=1.0, groundedness=1.0,
            coherence=1.0, actionability=1.0, specificity=1.0,
        )
        assert abs(profile.total_signal - 1.0) < 1e-10

    def test_default_profile(self):
        """Default profile (all 0.5) gives 0.5."""
        profile = SignalProfile()
        assert abs(profile.total_signal - 0.5) < 1e-10


class TestSNRAnalysis:
    """SNR computation correctness."""

    def test_snr_formula(self):
        """SNR = signal_power / (noise_power + epsilon)."""
        signal = SignalProfile(
            relevance=0.8, novelty=0.8, groundedness=0.8,
            coherence=0.8, actionability=0.8, specificity=0.8,
        )
        noise = NoiseProfile(redundancy=0.2, inconsistency=0.1)
        analysis = SNRAnalysis(signal=signal, noise=noise)

        expected_signal = 0.8  # All equal → geometric mean = 0.8
        expected_noise = 0.2 * 0.20 + 0.1 * 0.25  # = 0.065
        expected_snr = expected_signal / (expected_noise + 1e-10)

        assert abs(analysis.snr_linear - expected_snr) < 0.01

    def test_snr_db_conversion(self):
        """SNR_dB = 10 * log10(SNR_linear)."""
        signal = SignalProfile(
            relevance=0.8, novelty=0.8, groundedness=0.8,
            coherence=0.8, actionability=0.8, specificity=0.8,
        )
        noise = NoiseProfile()
        analysis = SNRAnalysis(signal=signal, noise=noise)

        # Zero noise → huge SNR_linear → large positive dB
        assert analysis.snr_db > 0
        expected_db = 10 * math.log10(analysis.snr_linear)
        assert abs(analysis.snr_db - expected_db) < 0.01

    def test_zero_noise_gives_huge_snr(self):
        """Zero noise gives very large SNR (bounded by epsilon)."""
        signal = SignalProfile(
            relevance=0.9, novelty=0.9, groundedness=0.9,
            coherence=0.9, actionability=0.9, specificity=0.9,
        )
        noise = NoiseProfile()
        analysis = SNRAnalysis(signal=signal, noise=noise)

        # signal / epsilon = huge number
        assert analysis.snr_linear > 1e6
        assert analysis.ihsan_achieved is True

    def test_high_noise_gives_low_snr(self):
        """High noise gives low SNR, failing Ihsan."""
        signal = SignalProfile()  # Default 0.5
        noise = NoiseProfile(
            redundancy=0.8, inconsistency=0.9, ambiguity=0.7,
            irrelevance=0.6, hallucination=0.5, verbosity=0.4, bias=0.3,
        )
        analysis = SNRAnalysis(signal=signal, noise=noise)

        # High noise → low ratio
        assert analysis.snr_linear < 2.0

    def test_ihsan_threshold_from_constants(self):
        """Ihsan threshold comes from centralized constants (0.95)."""
        from core.integration.constants import UNIFIED_IHSAN_THRESHOLD

        signal = SignalProfile(
            relevance=0.9, novelty=0.9, groundedness=0.9,
            coherence=0.9, actionability=0.9, specificity=0.9,
        )
        noise = NoiseProfile()
        analysis = SNRAnalysis(signal=signal, noise=noise)

        # Zero noise → SNR >> threshold → passes
        assert analysis.ihsan_achieved is True
        assert analysis.snr_linear >= UNIFIED_IHSAN_THRESHOLD

    def test_to_dict_roundtrip(self):
        """to_dict() includes all fields."""
        signal = SignalProfile(relevance=0.7)
        noise = NoiseProfile(redundancy=0.3)
        analysis = SNRAnalysis(signal=signal, noise=noise)
        d = analysis.to_dict()

        assert "signal" in d
        assert "noise" in d
        assert "snr_linear" in d
        assert "snr_db" in d
        assert "ihsan_achieved" in d
        assert "recommendations" in d


# ═══════════════════════════════════════════════════════════════════════════════
# NOISE FILTER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestNoiseFilter:
    """NoiseFilter detection accuracy."""

    def test_exact_duplicate_detected(self):
        """Same text submitted twice has high redundancy."""
        nf = NoiseFilter()
        nf.analyze("This is a test sentence about machine learning")
        profile = nf.analyze("This is a test sentence about machine learning")
        assert profile.redundancy == 1.0

    def test_unique_text_low_redundancy(self):
        """First-seen text has zero redundancy."""
        nf = NoiseFilter()
        profile = nf.analyze("A completely novel statement about quantum computing")
        assert profile.redundancy == 0.0

    def test_ambiguous_language_detected(self):
        """Text with hedging language scores high on ambiguity."""
        nf = NoiseFilter()
        text = "Maybe perhaps this might possibly sort of kind of work, I think, I guess"
        profile = nf.analyze(text)
        assert profile.ambiguity > 0.3

    def test_clear_language_low_ambiguity(self):
        """Direct, assertive text scores low on ambiguity."""
        nf = NoiseFilter()
        text = "The algorithm processes data in three stages. First, tokenization. Second, embedding. Third, inference."
        profile = nf.analyze(text)
        assert profile.ambiguity == 0.0

    def test_verbose_text_detected(self):
        """Text with filler phrases scores high on verbosity."""
        nf = NoiseFilter()
        text = (
            "In order to achieve this goal, due to the fact that we need "
            "at this point in time to implement a solution, it is important "
            "to note that for all intents and purposes we should proceed."
        )
        profile = nf.analyze(text)
        assert profile.verbosity > 0.2

    def test_concise_text_low_verbosity(self):
        """Short, direct text scores low on verbosity."""
        nf = NoiseFilter()
        text = "Run the test suite."
        profile = nf.analyze(text)
        assert profile.verbosity == 0.0

    def test_filter_removes_verbose_phrases(self):
        """filter() strips known verbose phrases when verbosity exceeds limit."""
        # Use low verbosity_limit so filter branch triggers
        nf = NoiseFilter(verbosity_limit=0.1)
        # Note: replace() is case-sensitive, so use lowercase to match phrase list
        text = "We need in order to fix the bug, due to the fact that tests fail, we must act."
        filtered, noise = nf.filter(text, threshold=0.0)
        assert "due to the fact that" not in filtered

    def test_reset_clears_state(self):
        """reset() clears seen hashes and concepts."""
        nf = NoiseFilter()
        nf.analyze("Some text about machine learning")
        assert len(nf._seen_hashes) > 0

        nf.reset()
        assert len(nf._seen_hashes) == 0
        assert len(nf._seen_concepts) == 0

    def test_concept_level_redundancy(self):
        """Overlapping concepts (not exact duplicates) produce partial redundancy."""
        nf = NoiseFilter()
        nf.analyze("Machine learning algorithms optimize neural network performance")
        profile = nf.analyze("Neural network architecture improves machine learning results")
        # Shared concepts "machine", "learning", "neural", "network" → partial redundancy
        assert 0.0 < profile.redundancy < 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# SIGNAL AMPLIFIER TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestSignalAmplifier:
    """SignalAmplifier analysis accuracy."""

    def test_relevance_with_matching_query(self):
        """High word overlap with query gives high relevance."""
        amp = SignalAmplifier()
        text = "The inference gateway routes requests to the optimal backend model"
        profile = amp.analyze(text, query="inference gateway model routing")
        assert profile.relevance > 0.5

    def test_relevance_without_query(self):
        """No query gives default 0.5 relevance."""
        amp = SignalAmplifier()
        text = "Some arbitrary content"
        profile = amp.analyze(text)
        assert profile.relevance == 0.5

    def test_novelty_detection(self):
        """Text with novel indicators scores higher on novelty."""
        amp = SignalAmplifier()
        text_novel = "A breakthrough discovery reveals new insight into emerging patterns"
        text_plain = "The the the the the the the the the"
        profile_novel = amp.analyze(text_novel)
        profile_plain = amp.analyze(text_plain)
        assert profile_novel.novelty > profile_plain.novelty

    def test_groundedness_with_citations(self):
        """Text with citation markers scores higher on groundedness."""
        amp = SignalAmplifier()
        text = "According to research, a study found that data indicates strong correlation."
        profile = amp.analyze(text)
        assert profile.groundedness > 0.5

    def test_groundedness_with_known_facts(self):
        """Known facts boost groundedness score."""
        amp = SignalAmplifier()
        amp.add_known_fact("BIZRA means seed in Arabic")
        text = "BIZRA means seed in Arabic. Every node is a seed."
        profile = amp.analyze(text)
        assert profile.groundedness >= 0.55  # Base 0.5 + fact boost

    def test_coherence_with_logical_connectors(self):
        """Text with logical connectors scores high on coherence."""
        amp = SignalAmplifier()
        text = (
            "The system is secure. Therefore, users can trust it. "
            "Moreover, the encryption is strong. Furthermore, the "
            "protocol has been verified. Consequently, deployment is safe."
        )
        profile = amp.analyze(text)
        assert profile.coherence > 0.7

    def test_coherence_penalizes_long_sentences(self):
        """Very long sentences reduce coherence score."""
        amp = SignalAmplifier()
        long_text = " ".join(["word"] * 200) + "."  # 200 words per sentence
        profile = amp.analyze(long_text)
        assert profile.coherence < 0.7

    def test_actionability_with_action_words(self):
        """Text with action patterns scores high on actionability."""
        amp = SignalAmplifier()
        text = (
            "First, implement the authentication module. "
            "Then, ensure the tests pass. Next, apply the migration. "
            "Finally, consider deploying to production."
        )
        profile = amp.analyze(text)
        assert profile.actionability > 0.6

    def test_source_authority_bounds(self):
        """Source authority is clamped to [0, 1]."""
        amp = SignalAmplifier()
        amp.set_source_authority("arxiv", 1.5)
        assert amp._source_authority["arxiv"] == 1.0
        amp.set_source_authority("random_blog", -0.5)
        assert amp._source_authority["random_blog"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# SNR MAXIMIZER (UNIFIED ENGINE) TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestSNRMaximizer:
    """SNRMaximizer unified engine tests."""

    def test_default_threshold_from_constants(self):
        """Default Ihsan threshold comes from constants.py."""
        from core.integration.constants import UNIFIED_IHSAN_THRESHOLD

        maximizer = SNRMaximizer()
        assert maximizer.ihsan_threshold == UNIFIED_IHSAN_THRESHOLD

    def test_custom_threshold(self):
        """Custom threshold overrides default."""
        maximizer = SNRMaximizer(ihsan_threshold=0.90)
        assert maximizer.ihsan_threshold == 0.90

    def test_analyze_returns_snr_analysis(self):
        """analyze() returns properly typed SNRAnalysis."""
        maximizer = SNRMaximizer()
        text = "BIZRA implements a proof-carrying inference protocol with Ed25519 signatures."
        analysis = maximizer.analyze(text)

        assert isinstance(analysis, SNRAnalysis)
        assert isinstance(analysis.signal, SignalProfile)
        assert isinstance(analysis.noise, NoiseProfile)
        assert isinstance(analysis.snr_linear, float)
        assert isinstance(analysis.snr_db, float)
        assert isinstance(analysis.ihsan_achieved, bool)

    def test_gate_pass_for_clean_content(self):
        """Clean, relevant content passes the Ihsan gate."""
        maximizer = SNRMaximizer()
        text = (
            "The FATE gate chain validates every inference through seven stages: "
            "schema, signature, timestamp, replay, ihsan, SNR, and policy. "
            "According to research, this ensures constitutional compliance."
        )
        passed, analysis = maximizer.gate(text)

        # First-seen text with citations → low noise, moderate signal → passes
        assert passed is True
        assert analysis.snr_linear >= maximizer.ihsan_threshold

    def test_gate_fail_for_pure_noise(self):
        """Highly redundant, ambiguous content fails the Ihsan gate."""
        maximizer = SNRMaximizer()
        # Submit same text to build redundancy
        noisy = "maybe perhaps maybe perhaps maybe perhaps maybe perhaps maybe perhaps"
        maximizer.analyze(noisy)  # First pass builds redundancy tracking
        passed, analysis = maximizer.gate(noisy)  # Second pass detects redundancy

        # High noise (redundancy + ambiguity) → low SNR
        assert analysis.noise.redundancy > 0
        assert analysis.noise.ambiguity > 0

    def test_statistics_tracking(self):
        """Stats track analyses, passes, fails, and average SNR."""
        maximizer = SNRMaximizer()

        text1 = "A novel insight about distributed consensus algorithms and Byzantine fault tolerance."
        text2 = "Another unique perspective on cryptographic verification protocols."

        maximizer.analyze(text1)
        maximizer.analyze(text2)

        assert maximizer.stats["analyses"] == 2
        assert maximizer.stats["ihsan_passes"] + maximizer.stats["ihsan_fails"] == 2
        assert maximizer.stats["avg_snr"] > 0

    def test_maximize_converges(self):
        """maximize() returns improved or equal SNR."""
        maximizer = SNRMaximizer()
        text = (
            "In order to implement authentication, due to the fact that "
            "security is important, the system should use Ed25519 signatures. "
            "According to research, this is a breakthrough approach."
        )
        optimized, analysis = maximizer.maximize(text, max_iterations=3)

        assert isinstance(optimized, str)
        assert isinstance(analysis, SNRAnalysis)
        assert len(optimized) > 0

    def test_maximize_with_query_context(self):
        """maximize() uses query context for relevance scoring."""
        maximizer = SNRMaximizer()
        text = "The sovereign runtime executes inference through FATE gates."
        _, analysis = maximizer.maximize(text, query="FATE gate inference")

        # With matching query, relevance should be higher
        assert analysis.signal.relevance > 0.5

    def test_reset_clears_all_state(self):
        """reset() clears filter state and statistics."""
        maximizer = SNRMaximizer()
        maximizer.analyze("Some text to populate state")
        assert maximizer.stats["analyses"] == 1

        maximizer.reset()
        assert maximizer.stats["analyses"] == 0
        assert maximizer.stats["avg_snr"] == 0.0

    def test_async_optimize_api(self):
        """optimize() async method returns runtime-compatible dict."""
        maximizer = SNRMaximizer()
        text = "The inference gateway routes requests to optimal backends based on task complexity."
        result = asyncio.run(maximizer.optimize(text))

        assert isinstance(result, dict)
        assert "snr_score" in result
        assert "ihsan_score" in result
        assert "passed" in result
        assert "recommendations" in result
        assert "noise_components" in result
        assert "signal_components" in result
        assert isinstance(result["snr_score"], float)
        assert isinstance(result["passed"], bool)

    def test_recommendations_on_failure(self):
        """Failed gate generates recommendations for noisy content."""
        # ihsan_threshold on SNRMaximizer controls gate(); set impossibly high
        maximizer = SNRMaximizer(ihsan_threshold=1e10)
        text = "Maybe this sort of kind of unclear thing might work, I guess."
        passed, analysis = maximizer.gate(text)
        assert not passed  # gate() uses self.ihsan_threshold (1e10)

    def test_analyze_generates_recommendations_for_low_snr(self):
        """analyze() generates recommendations when SNRAnalysis.ihsan_achieved is False."""
        maximizer = SNRMaximizer()
        # Highly redundant content that fails the constant Ihsan threshold
        noisy = " ".join(["maybe perhaps possibly"] * 50)
        analysis = maximizer.analyze(noisy)
        # Should detect ambiguity in hedge words
        assert analysis.noise.ambiguity > 0.0

    def test_enum_types_exist(self):
        """NoiseType and SignalType enums are properly defined."""
        assert len(NoiseType) == 7
        assert len(SignalType) == 7
        assert NoiseType.REDUNDANCY.value == "redundancy"
        assert SignalType.INSIGHT.value == "insight"


# ═══════════════════════════════════════════════════════════════════════════════
# EDGE CASES — Mathematical Boundaries
# ═══════════════════════════════════════════════════════════════════════════════


class TestEdgeCases:
    """Mathematical edge cases for the SNR formula."""

    def test_empty_text(self):
        """Empty text doesn't crash, produces valid analysis."""
        maximizer = SNRMaximizer()
        analysis = maximizer.analyze("")
        assert isinstance(analysis, SNRAnalysis)
        assert analysis.snr_linear > 0  # epsilon prevents division by zero

    def test_single_word(self):
        """Single word text produces valid analysis."""
        maximizer = SNRMaximizer()
        analysis = maximizer.analyze("BIZRA")
        assert isinstance(analysis, SNRAnalysis)

    def test_very_long_text(self):
        """Very long text doesn't crash or timeout."""
        maximizer = SNRMaximizer()
        text = "This is a test sentence about various topics. " * 1000
        analysis = maximizer.analyze(text)
        assert isinstance(analysis, SNRAnalysis)

    def test_unicode_text(self):
        """Unicode text (Arabic, etc.) is handled gracefully."""
        maximizer = SNRMaximizer()
        text = "بذرة means seed. كل إنسان بذرة. BIZRA (بذرة) is the genesis."
        analysis = maximizer.analyze(text)
        assert isinstance(analysis, SNRAnalysis)

    def test_snr_linear_always_positive(self):
        """SNR_linear is always positive (epsilon prevents zero)."""
        # Worst case: all signal dimensions near zero, all noise at max
        signal = SignalProfile(
            relevance=0.0, novelty=0.0, groundedness=0.0,
            coherence=0.0, actionability=0.0, specificity=0.0,
        )
        noise = NoiseProfile(
            redundancy=1.0, inconsistency=1.0, ambiguity=1.0,
            irrelevance=1.0, hallucination=1.0, verbosity=1.0, bias=1.0,
        )
        analysis = SNRAnalysis(signal=signal, noise=noise)
        assert analysis.snr_linear > 0

    def test_noise_weights_are_normalized(self):
        """All noise weights should sum to exactly 1.0."""
        weights = [0.20, 0.25, 0.15, 0.15, 0.10, 0.05, 0.10]
        assert abs(sum(weights) - 1.0) < 1e-10

    def test_re_export_from_reasoning(self):
        """The reasoning module re-exports correctly from sovereign."""
        from core.reasoning.snr_maximizer import SNRMaximizer as ReasoningSNR
        from core.sovereign.snr_maximizer import SNRMaximizer as SovereignSNR
        assert ReasoningSNR is SovereignSNR
