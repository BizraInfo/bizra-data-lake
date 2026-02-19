"""
Tests for core.rdve.interdisciplinary — Domain Enums, Pattern Library,
and InterdisciplinaryTransfer Engine.

Covers:
    - Domain enum membership and values
    - TransferConfidence enum ordering
    - DomainPattern.matches_context scoring
    - CANONICAL_PATTERNS integrity (count, proven subset, domain coverage)
    - InterdisciplinaryTransfer initialization and pattern access
    - find_transfers with context_tags and target_domain filtering
    - find_transfers confidence filtering
    - get_patterns_by_domain and get_proven_patterns
    - add_pattern to library
    - get_statistics
"""

import pytest

from core.rdve.interdisciplinary import (
    CANONICAL_PATTERNS,
    Domain,
    DomainPattern,
    InterdisciplinaryTransfer,
    TransferConfidence,
    TransferResult,
)


# ============================================================================
# Domain Enum
# ============================================================================


class TestDomain:
    def test_has_twelve_members(self):
        assert len(Domain) == 12

    def test_key_domains_present(self):
        assert Domain.INFORMATION_THEORY.value == "information_theory"
        assert Domain.BIOLOGY.value == "biology"
        assert Domain.PHYSICS.value == "physics"
        assert Domain.ECONOMICS.value == "economics"
        assert Domain.PSYCHOLOGY.value == "psychology"
        assert Domain.MANUFACTURING.value == "manufacturing"
        assert Domain.MILITARY.value == "military"
        assert Domain.MATHEMATICS.value == "mathematics"
        assert Domain.DISTRIBUTED_SYSTEMS.value == "distributed_systems"
        assert Domain.NEUROSCIENCE.value == "neuroscience"
        assert Domain.GAME_THEORY.value == "game_theory"
        assert Domain.ETHICS.value == "ethics"

    def test_domain_is_str_enum(self):
        # Domain inherits from str, so values can be used as strings
        assert isinstance(Domain.BIOLOGY, str)
        assert Domain.BIOLOGY == "biology"


# ============================================================================
# TransferConfidence Enum
# ============================================================================


class TestTransferConfidence:
    def test_has_four_levels(self):
        assert len(TransferConfidence) == 4

    def test_values(self):
        assert TransferConfidence.PROVEN.value == "proven"
        assert TransferConfidence.HIGH.value == "high"
        assert TransferConfidence.MEDIUM.value == "medium"
        assert TransferConfidence.SPECULATIVE.value == "speculative"


# ============================================================================
# DomainPattern.matches_context
# ============================================================================


class TestDomainPatternMatchesContext:
    def setup_method(self):
        self.pattern = DomainPattern(
            id="test_pattern",
            name="Test Pattern",
            source_domain=Domain.INFORMATION_THEORY,
            core_principle="Test principle",
            transfer_conditions=["cond1"],
            instantiation_recipe=["step1"],
            historical_examples=[],
            target_domains=[Domain.BIOLOGY],
            confidence=TransferConfidence.HIGH,
            giant="Test Giant",
            tags=["quality", "filtering", "scoring", "multi-dimensional"],
        )

    def test_full_overlap_returns_one(self):
        context = {"quality", "filtering", "scoring", "multi-dimensional"}
        assert self.pattern.matches_context(context) == pytest.approx(1.0)

    def test_partial_overlap_returns_fraction(self):
        context = {"quality", "filtering"}
        assert self.pattern.matches_context(context) == pytest.approx(2 / 4)

    def test_no_overlap_returns_zero(self):
        context = {"unrelated", "tags"}
        assert self.pattern.matches_context(context) == pytest.approx(0.0)

    def test_empty_tags_returns_zero(self):
        empty_pattern = DomainPattern(
            id="empty",
            name="Empty",
            source_domain=Domain.BIOLOGY,
            core_principle="N/A",
            transfer_conditions=[],
            instantiation_recipe=[],
            historical_examples=[],
            target_domains=[],
            confidence=TransferConfidence.SPECULATIVE,
            giant="N/A",
            tags=[],
        )
        assert empty_pattern.matches_context({"quality"}) == pytest.approx(0.0)

    def test_superset_context_still_bounded_by_tag_count(self):
        context = {"quality", "filtering", "scoring", "multi-dimensional", "extra1", "extra2"}
        # overlap=4, len(tags)=4 => 4/4 = 1.0
        assert self.pattern.matches_context(context) == pytest.approx(1.0)


# ============================================================================
# CANONICAL_PATTERNS integrity
# ============================================================================


class TestCanonicalPatterns:
    def test_canonical_patterns_is_nonempty_list(self):
        assert isinstance(CANONICAL_PATTERNS, list)
        assert len(CANONICAL_PATTERNS) >= 10  # 13 patterns defined in source

    def test_all_entries_are_domain_patterns(self):
        for p in CANONICAL_PATTERNS:
            assert isinstance(p, DomainPattern)

    def test_ids_are_unique(self):
        ids = [p.id for p in CANONICAL_PATTERNS]
        assert len(ids) == len(set(ids))

    def test_proven_patterns_have_bizra_implementation(self):
        for p in CANONICAL_PATTERNS:
            if p.confidence == TransferConfidence.PROVEN:
                assert p.bizra_implementation is not None, (
                    f"Proven pattern '{p.id}' should have bizra_implementation set"
                )

    def test_multiple_source_domains_represented(self):
        domains = {p.source_domain for p in CANONICAL_PATTERNS}
        # At least 5 distinct source domains in the canonical library
        assert len(domains) >= 5


# ============================================================================
# InterdisciplinaryTransfer — Initialization
# ============================================================================


class TestInterdisciplinaryTransferInit:
    def test_default_init_loads_canonical_patterns(self):
        engine = InterdisciplinaryTransfer()
        stats = engine.get_statistics()
        assert stats["total_patterns"] == len(CANONICAL_PATTERNS)

    def test_custom_patterns_override_canonical(self):
        custom = [CANONICAL_PATTERNS[0]]
        engine = InterdisciplinaryTransfer(patterns=custom)
        assert engine.get_statistics()["total_patterns"] == 1


# ============================================================================
# InterdisciplinaryTransfer — find_transfers
# ============================================================================


class TestFindTransfers:
    def setup_method(self):
        self.engine = InterdisciplinaryTransfer()

    def test_find_transfers_with_quality_tags(self):
        results = self.engine.find_transfers(
            context_tags={"quality", "filtering", "scoring", "multi-dimensional"},
            target_domain=Domain.BIOLOGY,
        )
        assert isinstance(results, list)
        # The shannon_snr pattern targets BIOLOGY with exactly these tags
        assert len(results) >= 1
        # Results are sorted by applicability_score descending
        for i in range(len(results) - 1):
            assert results[i].applicability_score >= results[i + 1].applicability_score

    def test_find_transfers_returns_transfer_result_type(self):
        results = self.engine.find_transfers(
            context_tags={"quality"},
            target_domain=Domain.BIOLOGY,
        )
        for r in results:
            assert isinstance(r, TransferResult)
            assert isinstance(r.pattern, DomainPattern)
            assert isinstance(r.applicability_score, float)
            assert 0.0 <= r.applicability_score <= 1.0

    def test_empty_context_returns_patterns_boosted_by_confidence(self):
        # Even with no tag overlap, confidence boost may push above min_applicability
        results = self.engine.find_transfers(
            context_tags=set(),
            target_domain=Domain.ECONOMICS,
            min_confidence=TransferConfidence.PROVEN,
        )
        # Proven patterns get confidence_boost = 4 * 0.1 = 0.4
        # With zero tag overlap: score = 0.0 + 0.4 = 0.4 >= 0.3 (min_applicability)
        assert len(results) >= 1

    def test_speculative_min_confidence_allows_all(self):
        results = self.engine.find_transfers(
            context_tags={"prediction", "learning"},
            min_confidence=TransferConfidence.SPECULATIVE,
        )
        # Should include medium/speculative patterns that match
        assert len(results) >= 1

    def test_proven_min_confidence_filters_lower(self):
        results_proven = self.engine.find_transfers(
            context_tags={"quality", "filtering"},
            min_confidence=TransferConfidence.PROVEN,
        )
        results_all = self.engine.find_transfers(
            context_tags={"quality", "filtering"},
            min_confidence=TransferConfidence.SPECULATIVE,
        )
        assert len(results_proven) <= len(results_all)


# ============================================================================
# InterdisciplinaryTransfer — Domain Queries
# ============================================================================


class TestDomainQueries:
    def setup_method(self):
        self.engine = InterdisciplinaryTransfer()

    def test_get_patterns_by_domain(self):
        info_theory = self.engine.get_patterns_by_domain(Domain.INFORMATION_THEORY)
        assert all(
            p.source_domain == Domain.INFORMATION_THEORY for p in info_theory
        )
        # At least the shannon_snr pattern
        assert len(info_theory) >= 1

    def test_get_proven_patterns(self):
        proven = self.engine.get_proven_patterns()
        assert all(p.confidence == TransferConfidence.PROVEN for p in proven)
        assert len(proven) >= 5  # 7 proven patterns in canonical set

    def test_add_pattern_increases_count(self):
        # NOTE: InterdisciplinaryTransfer uses `patterns or list(CANONICAL_PATTERNS)`,
        # so an empty list falls through to canonical. We start from canonical count
        # and verify that add_pattern increments by one.
        engine = InterdisciplinaryTransfer()
        baseline = engine.get_statistics()["total_patterns"]

        new_pattern = DomainPattern(
            id="custom_test",
            name="Custom Test Pattern",
            source_domain=Domain.ETHICS,
            core_principle="Test",
            transfer_conditions=[],
            instantiation_recipe=[],
            historical_examples=[],
            target_domains=[Domain.BIOLOGY],
            confidence=TransferConfidence.SPECULATIVE,
            giant="Test",
            tags=["test"],
        )
        engine.add_pattern(new_pattern)
        assert engine.get_statistics()["total_patterns"] == baseline + 1

    def test_get_statistics_structure(self):
        stats = self.engine.get_statistics()
        assert "total_patterns" in stats
        assert "patterns_by_domain" in stats
        assert "proven_count" in stats
        assert "implemented_count" in stats
        assert "transfers_executed" in stats
        assert isinstance(stats["patterns_by_domain"], dict)
        assert stats["transfers_executed"] == 0  # No transfers executed yet
