"""
Tests for the Standing on Giants — Knowledge Attribution Protocol.

Covers: Giant data model, GiantsRegistry (register, lookup, category index,
applications, attribution formatting, algorithm explanation), and convenience functions.
"""

import pytest

from core.sovereign.runtime_engines.giants_registry import (
    Giant,
    GiantApplication,
    GiantCategory,
    GiantsRegistry,
    attribute,
    get_giants_registry,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Giant Data Model Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestGiant:
    """Test Giant data model."""

    def test_creation(self):
        g = Giant(
            name="Test Giant",
            year=2000,
            work="Test Work",
            contribution="Test contribution",
            category=GiantCategory.MATHEMATICS,
            citation="Test, T. (2000). Test Work.",
            key_insight="Testing is important.",
        )
        assert g.name == "Test Giant"
        assert g.year == 2000
        assert g.category == GiantCategory.MATHEMATICS

    def test_format_attribution(self):
        g = Giant(
            name="Claude Shannon",
            year=1948,
            work="A Mathematical Theory of Communication",
            contribution="Information theory",
            category=GiantCategory.INFORMATION_THEORY,
            citation="Shannon (1948)",
            key_insight="Information can be quantified.",
        )
        assert g.format_attribution() == "Claude Shannon (1948): A Mathematical Theory of Communication"

    def test_format_full_citation(self):
        g = Giant(
            name="Test",
            year=2000,
            work="Work",
            contribution="Contrib",
            category=GiantCategory.MATHEMATICS,
            citation="Test, T. (2000). Full citation here.",
            key_insight="Insight",
        )
        assert g.format_full_citation() == "Test, T. (2000). Full citation here."

    def test_hash_equality(self):
        """Giants with same name and year are equal."""
        g1 = Giant(
            name="Shannon", year=1948, work="A",
            contribution="C", category=GiantCategory.INFORMATION_THEORY,
            citation="X", key_insight="Y",
        )
        g2 = Giant(
            name="Shannon", year=1948, work="B",
            contribution="D", category=GiantCategory.INFORMATION_THEORY,
            citation="Z", key_insight="W",
        )
        assert g1 == g2
        assert hash(g1) == hash(g2)

    def test_inequality(self):
        g1 = Giant(
            name="Shannon", year=1948, work="A",
            contribution="C", category=GiantCategory.INFORMATION_THEORY,
            citation="X", key_insight="Y",
        )
        g2 = Giant(
            name="Lamport", year=1982, work="B",
            contribution="D", category=GiantCategory.DISTRIBUTED_SYSTEMS,
            citation="Z", key_insight="W",
        )
        assert g1 != g2

    def test_not_equal_to_non_giant(self):
        g = Giant(
            name="Test", year=2000, work="W",
            contribution="C", category=GiantCategory.MATHEMATICS,
            citation="X", key_insight="Y",
        )
        assert g != "not a giant"

    def test_default_lists(self):
        g = Giant(
            name="Test", year=2000, work="W",
            contribution="C", category=GiantCategory.MATHEMATICS,
            citation="X", key_insight="Y",
        )
        assert g.applications_in_bizra == []
        assert g.related_giants == []


# ═══════════════════════════════════════════════════════════════════════════════
# GiantsRegistry Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestGiantsRegistry:
    """Test the GiantsRegistry."""

    @pytest.fixture
    def registry(self):
        return GiantsRegistry()

    def test_initialization_has_giants(self, registry):
        """Registry should be pre-populated with foundational giants."""
        summary = registry.summary()
        assert summary["total_giants"] >= 15

    def test_get_by_name_and_year(self, registry):
        g = registry.get("Claude Shannon", 1948)
        assert g is not None
        assert g.name == "Claude Shannon"
        assert g.year == 1948

    def test_get_by_name_only(self, registry):
        g = registry.get("Leslie Lamport")
        assert g is not None
        assert g.name == "Leslie Lamport"

    def test_get_nonexistent(self, registry):
        assert registry.get("Nonexistent Person") is None

    def test_get_nonexistent_with_year(self, registry):
        assert registry.get("Claude Shannon", 9999) is None

    def test_get_by_category(self, registry):
        info_giants = registry.get_by_category(GiantCategory.INFORMATION_THEORY)
        assert len(info_giants) >= 2
        names = {g.name for g in info_giants}
        assert "Claude Shannon" in names

    def test_get_by_category_distributed(self, registry):
        dist_giants = registry.get_by_category(GiantCategory.DISTRIBUTED_SYSTEMS)
        assert len(dist_giants) >= 2
        names = {g.name for g in dist_giants}
        assert "Leslie Lamport" in names

    def test_get_by_category_philosophy(self, registry):
        phil_giants = registry.get_by_category(GiantCategory.PHILOSOPHY)
        assert len(phil_giants) >= 1

    def test_register_custom_giant(self, registry):
        g = Giant(
            name="Custom Giant",
            year=2025,
            work="Custom Work",
            contribution="Custom",
            category=GiantCategory.MATHEMATICS,
            citation="Custom (2025)",
            key_insight="Custom insight",
        )
        registry.register(g)
        retrieved = registry.get("Custom Giant", 2025)
        assert retrieved is not None
        assert retrieved.work == "Custom Work"

    def test_summary(self, registry):
        summary = registry.summary()
        assert "total_giants" in summary
        assert "categories" in summary
        assert "applications" in summary
        assert summary["total_giants"] > 0

    def test_all_categories_indexed(self, registry):
        """Every category should exist in the index (even if empty)."""
        for cat in GiantCategory:
            giants = registry.get_by_category(cat)
            assert isinstance(giants, list)


# ═══════════════════════════════════════════════════════════════════════════════
# Application Recording Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestGiantApplications:
    """Test application recording and lookup."""

    @pytest.fixture
    def registry(self):
        r = GiantsRegistry()
        r.record_application(
            module="snr_maximizer",
            method="compute_snr",
            giant_names=["Claude Shannon"],
            explanation="SNR computation based on Shannon entropy",
            performance_impact="O(n) per signal",
        )
        r.record_application(
            module="consensus",
            method="byzantine_vote",
            giant_names=["Leslie Lamport"],
            explanation="PBFT consensus derived from Byzantine generals",
        )
        return r

    def test_record_application(self, registry):
        summary = registry.summary()
        assert summary["applications"] >= 2

    def test_get_applications_for_giant(self, registry):
        apps = registry.get_applications_for("Claude Shannon")
        assert len(apps) >= 1
        assert apps[0].module == "snr_maximizer"

    def test_get_applications_for_unknown(self, registry):
        apps = registry.get_applications_for("Unknown Person")
        assert len(apps) == 0

    def test_format_attribution_header(self, registry):
        header = registry.format_attribution_header("snr_maximizer")
        assert "Standing on the Shoulders of Giants:" in header
        assert "Claude Shannon" in header

    def test_format_attribution_header_empty(self, registry):
        header = registry.format_attribution_header("nonexistent_module")
        assert header == ""

    def test_explain_algorithm(self, registry):
        explanation = registry.explain_algorithm("compute_snr")
        assert "compute_snr" in explanation
        assert "Shannon" in explanation
        assert "Key insight:" in explanation

    def test_explain_unknown_algorithm(self, registry):
        explanation = registry.explain_algorithm("unknown_method")
        assert "No recorded giants" in explanation

    def test_application_with_missing_giant(self, registry):
        """Recording with a nonexistent giant name should not crash."""
        registry.record_application(
            module="test",
            method="test_method",
            giant_names=["Nonexistent Giant"],
            explanation="Test",
        )
        apps = registry.get_applications_for("Nonexistent Giant")
        assert len(apps) == 0


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience Functions Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestConvenienceFunctions:
    """Test module-level convenience functions."""

    def test_get_giants_registry_singleton(self):
        r1 = get_giants_registry()
        r2 = get_giants_registry()
        assert r1 is r2

    def test_attribute_single(self):
        result = attribute(["Claude Shannon"])
        assert "Claude Shannon (1948)" in result

    def test_attribute_multiple(self):
        result = attribute(["Claude Shannon", "Leslie Lamport"])
        assert "Shannon" in result
        assert "Lamport" in result
        assert "|" in result

    def test_attribute_unknown(self):
        result = attribute(["Unknown Person"])
        assert result == ""

    def test_attribute_mixed(self):
        result = attribute(["Claude Shannon", "Unknown"])
        assert "Shannon" in result


# ═══════════════════════════════════════════════════════════════════════════════
# Specific Giants Validation Tests
# ═══════════════════════════════════════════════════════════════════════════════


class TestSpecificGiants:
    """Validate specific key giants are registered correctly."""

    @pytest.fixture
    def registry(self):
        return GiantsRegistry()

    def test_shannon_has_snr_application(self, registry):
        g = registry.get("Claude Shannon")
        assert g is not None
        assert any("SNR" in app for app in g.applications_in_bizra)

    def test_besta_has_got_application(self, registry):
        g = registry.get("Maciej Besta")
        assert g is not None
        assert any("ThoughtGraph" in app for app in g.applications_in_bizra)

    def test_anthropic_has_ihsan_application(self, registry):
        g = registry.get("Anthropic")
        assert g is not None
        assert any("Ihsan" in app for app in g.applications_in_bizra)

    def test_al_ghazali_has_muraqabah(self, registry):
        g = registry.get("Abu Hamid Al-Ghazali")
        assert g is not None
        assert any("Muraqabah" in app for app in g.applications_in_bizra)

    def test_lamport_has_byzantine(self, registry):
        g = registry.get("Leslie Lamport")
        assert g is not None
        assert any("Byzantine" in app or "PBFT" in app for app in g.applications_in_bizra)
