"""Tests for core.personaplex.guardians -- Guardian system and Ihsan vectors.

Covers:
- GuardianRole enum
- IhsanVector composite scores and thresholds
- Guardian data class: prompts, can_respond, serialization
- Module-level helper functions
- Pre-defined BIZRA_GUARDIANS registry
"""

import math

import pytest

from core.personaplex.guardians import (
    BIZRA_GUARDIANS,
    Guardian,
    GuardianRole,
    IhsanVector,
    compute_collective_ihsan,
    get_active_guardians,
    get_guardian,
    get_guardians_by_role,
    list_guardians,
)


# ---------------------------------------------------------------------------
# IhsanVector TESTS
# ---------------------------------------------------------------------------


class TestIhsanVector:

    def test_default_composite(self):
        iv = IhsanVector()
        expected = (0.85 + 0.90 + 0.85 + 0.85 + 0.80) / 5
        assert abs(iv.composite - expected) < 1e-9

    def test_custom_composite(self):
        iv = IhsanVector(
            correctness=1.0, safety=1.0, beneficence=1.0,
            transparency=1.0, sustainability=1.0,
        )
        assert iv.composite == 1.0

    def test_geometric_mean(self):
        iv = IhsanVector(
            correctness=0.9, safety=0.9, beneficence=0.9,
            transparency=0.9, sustainability=0.9,
        )
        assert abs(iv.geometric_mean - 0.9) < 1e-9

    def test_geometric_mean_penalizes_low_values(self):
        # One very low value should drag geometric mean down more than arithmetic
        iv = IhsanVector(
            correctness=0.95, safety=0.95, beneficence=0.95,
            transparency=0.95, sustainability=0.10,
        )
        assert iv.geometric_mean < iv.composite

    def test_passes_threshold_default(self):
        iv = IhsanVector()
        assert iv.passes_threshold(0.75) is True

    def test_passes_threshold_too_high(self):
        iv = IhsanVector(
            correctness=0.5, safety=0.5, beneficence=0.5,
            transparency=0.5, sustainability=0.5,
        )
        assert iv.passes_threshold(0.75) is False

    def test_passes_ihsan_true(self):
        iv = IhsanVector(
            correctness=0.95, safety=0.95, beneficence=0.95,
            transparency=0.95, sustainability=0.95,
        )
        assert iv.passes_ihsan() is True

    def test_passes_ihsan_false(self):
        iv = IhsanVector()  # Default composite is ~0.85
        assert iv.passes_ihsan() is False

    def test_weakest_dimension(self):
        iv = IhsanVector(
            correctness=0.90, safety=0.95, beneficence=0.85,
            transparency=0.80, sustainability=0.88,
        )
        name, value = iv.weakest_dimension
        assert name == "transparency"
        assert value == 0.80

    def test_to_dict(self):
        iv = IhsanVector()
        d = iv.to_dict()
        assert "correctness" in d
        assert "safety" in d
        assert "composite" in d
        assert "passes_ihsan" in d
        assert isinstance(d["passes_ihsan"], bool)

    @pytest.mark.parametrize("field,value", [
        ("correctness", 0.0),
        ("safety", 0.0),
        ("beneficence", 0.0),
        ("transparency", 0.0),
        ("sustainability", 0.0),
    ])
    def test_zero_dimension(self, field, value):
        iv = IhsanVector(**{field: value})
        # Should not raise, geometric mean uses max(val, 1e-10)
        assert iv.geometric_mean >= 0


# ---------------------------------------------------------------------------
# Guardian TESTS
# ---------------------------------------------------------------------------


class TestGuardian:

    @pytest.fixture
    def test_guardian(self):
        return Guardian(
            name="TestGuardian",
            role=GuardianRole.REASONING,
            domain="Testing",
            voice_prompt="NATM1",
            text_prompt="You are a test guardian.",
            expertise=["testing", "validation"],
        )

    def test_get_full_prompt(self, test_guardian):
        prompt = test_guardian.get_full_prompt()
        assert "TestGuardian" in prompt
        assert "Testing" in prompt
        assert "testing, validation" in prompt

    def test_get_full_prompt_no_expertise(self):
        g = Guardian(
            name="Basic",
            role=GuardianRole.KNOWLEDGE,
            domain="General Knowledge",
            voice_prompt="NATM0",
            text_prompt="Basic guardian.",
        )
        prompt = g.get_full_prompt()
        assert "General Knowledge" in prompt

    def test_can_respond_active_guardian(self, test_guardian):
        can, reason = test_guardian.can_respond("analysis")
        assert can is True
        assert "passed" in reason.lower()

    def test_can_respond_inactive_guardian(self, test_guardian):
        test_guardian.active = False
        can, reason = test_guardian.can_respond("analysis")
        assert can is False
        assert "inactive" in reason.lower()

    @pytest.mark.parametrize("purpose", [
        "harm someone",
        "deceive the user",
        "exploit a vulnerability",
        "commit fraud",
        "attack the system",
        "steal credentials",
    ])
    def test_can_respond_blocked_purposes(self, test_guardian, purpose):
        can, reason = test_guardian.can_respond(purpose)
        assert can is False
        assert "safety gate" in reason.lower()

    def test_can_respond_low_ihsan(self):
        g = Guardian(
            name="Weak",
            role=GuardianRole.CREATIVE,
            domain="Test",
            voice_prompt="NATM0",
            text_prompt="Weak guardian.",
            ihsan_constraints=IhsanVector(
                correctness=0.3, safety=0.3, beneficence=0.3,
                transparency=0.3, sustainability=0.3,
            ),
        )
        can, reason = g.can_respond("normal query")
        assert can is False
        assert "constraint failed" in reason.lower()

    def test_to_dict(self, test_guardian):
        d = test_guardian.to_dict()
        assert d["name"] == "TestGuardian"
        assert d["role"] == "reasoning"
        assert d["active"] is True
        assert "ihsan" in d


# ---------------------------------------------------------------------------
# BIZRA_GUARDIANS REGISTRY TESTS
# ---------------------------------------------------------------------------


class TestBIZRAGuardians:

    def test_all_eight_guardians_registered(self):
        expected = {
            "architect", "security", "ethics", "reasoning",
            "knowledge", "creative", "integration", "nucleus",
        }
        assert set(BIZRA_GUARDIANS.keys()) == expected

    def test_all_guardians_have_unique_voices(self):
        voices = [g.voice_prompt for g in BIZRA_GUARDIANS.values()]
        assert len(voices) == len(set(voices)), "Duplicate voice prompts found"

    def test_all_guardians_active_by_default(self):
        for name, guardian in BIZRA_GUARDIANS.items():
            assert guardian.active is True, f"{name} is not active"

    def test_ethics_guardian_highest_constraints(self):
        ethics = BIZRA_GUARDIANS["ethics"]
        assert ethics.ihsan_constraints.composite >= 0.95

    def test_security_guardian_high_safety(self):
        security = BIZRA_GUARDIANS["security"]
        assert security.ihsan_constraints.safety >= 0.95


# ---------------------------------------------------------------------------
# MODULE-LEVEL FUNCTION TESTS
# ---------------------------------------------------------------------------


class TestModuleFunctions:

    def test_get_guardian_exists(self):
        g = get_guardian("architect")
        assert g is not None
        assert g.name == "Architect"

    def test_get_guardian_case_insensitive(self):
        g = get_guardian("ARCHITECT")
        assert g is not None

    def test_get_guardian_not_found(self):
        g = get_guardian("nonexistent")
        assert g is None

    def test_list_guardians(self):
        names = list_guardians()
        assert len(names) == 8
        assert "architect" in names

    def test_get_guardians_by_role(self):
        guardians = get_guardians_by_role(GuardianRole.SECURITY)
        assert len(guardians) == 1
        assert guardians[0].name == "Security"

    def test_get_active_guardians(self):
        active = get_active_guardians()
        assert len(active) == 8  # All active by default

    def test_compute_collective_ihsan_empty(self):
        result = compute_collective_ihsan([])
        assert isinstance(result, IhsanVector)

    def test_compute_collective_ihsan_all_guardians(self):
        all_guardians = list(BIZRA_GUARDIANS.values())
        result = compute_collective_ihsan(all_guardians)
        assert 0.0 < result.composite < 1.0
        assert isinstance(result, IhsanVector)

    def test_compute_collective_ihsan_single(self):
        ethics = BIZRA_GUARDIANS["ethics"]
        result = compute_collective_ihsan([ethics])
        assert abs(result.correctness - ethics.ihsan_constraints.correctness) < 1e-9
