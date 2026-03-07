"""Tests for core.sovereign.human_lifecycle — 7-stage growth progression.

Pure function tests. No I/O, no mocks needed.
"""

import pytest

from core.integration.constants import HUMAN_STAGE_ORDER, HUMAN_STAGE_THRESHOLDS
from core.sovereign.human_lifecycle import (
    AGENT_TIER_MAP,
    STAGES,
    HumanStage,
    agent_tier_equivalent,
    human_stage,
    human_stage_detail,
    stage_progress,
)


class TestStagesConstruction:
    """STAGES list built correctly from constants.py."""

    def test_stages_count_matches_order(self):
        assert len(STAGES) == len(HUMAN_STAGE_ORDER)

    def test_stages_names_match_order(self):
        assert [s.name for s in STAGES] == HUMAN_STAGE_ORDER

    def test_stages_ranks_are_sequential(self):
        assert [s.rank for s in STAGES] == list(range(len(STAGES)))

    def test_stages_thresholds_from_constants(self):
        for stage in STAGES:
            assert stage.score_low == HUMAN_STAGE_THRESHOLDS[stage.name]

    def test_last_stage_high_is_one(self):
        assert STAGES[-1].score_high == 1.0

    def test_stages_are_frozen(self):
        with pytest.raises(AttributeError):
            STAGES[0].name = "Hacked"  # type: ignore[misc]

    def test_every_stage_has_description(self):
        for stage in STAGES:
            assert len(stage.description) > 0
            assert len(stage.unlock_condition) > 0


class TestHumanStage:
    """human_stage() maps sovereignty → stage name."""

    @pytest.mark.parametrize(
        "score, expected",
        [
            (0.0, "Seed"),
            (0.05, "Seed"),
            (0.10, "Node"),
            (0.19, "Node"),
            (0.20, "Apprentice"),
            (0.35, "Builder"),
            (0.54, "Builder"),
            (0.55, "Verifier"),
            (0.70, "Mentor"),
            (0.84, "Mentor"),
            (0.85, "Catalyst"),
            (1.0, "Catalyst"),
        ],
    )
    def test_stage_boundaries(self, score: float, expected: str):
        assert human_stage(score) == expected

    def test_negative_score_clamps_to_seed(self):
        assert human_stage(-0.5) == "Seed"

    def test_above_one_clamps_to_catalyst(self):
        assert human_stage(1.5) == "Catalyst"


class TestHumanStageDetail:
    """human_stage_detail() returns full HumanStage object."""

    def test_returns_humanstage_instance(self):
        result = human_stage_detail(0.5)
        assert isinstance(result, HumanStage)

    def test_builder_at_midpoint(self):
        stage = human_stage_detail(0.45)
        assert stage.name == "Builder"
        assert stage.rank == 3

    def test_seed_at_zero(self):
        stage = human_stage_detail(0.0)
        assert stage.name == "Seed"
        assert stage.rank == 0
        assert stage.score_low == 0.0


class TestStageProgress:
    """stage_progress() provides progress + next stage info."""

    def test_returns_expected_keys(self):
        result = stage_progress(0.5)
        expected_keys = {
            "current_stage",
            "rank",
            "progress",
            "sovereignty_score",
            "next_stage",
            "next_threshold",
            "points_to_next",
            "description",
            "unlock_condition",
        }
        assert set(result.keys()) == expected_keys

    def test_progress_at_stage_boundary(self):
        result = stage_progress(0.55)
        assert result["current_stage"] == "Verifier"
        assert result["progress"] == 0.0

    def test_progress_midway(self):
        result = stage_progress(0.275)
        assert result["current_stage"] == "Apprentice"
        # Apprentice: 0.20 → 0.35, midpoint = 0.275, progress = 0.5
        assert result["progress"] == 0.5

    def test_catalyst_has_no_next(self):
        result = stage_progress(0.95)
        assert result["current_stage"] == "Catalyst"
        assert result["next_stage"] is None
        assert result["next_threshold"] is None
        assert result["points_to_next"] == 0.0

    def test_points_to_next_computed(self):
        result = stage_progress(0.30)
        assert result["current_stage"] == "Apprentice"
        assert result["next_stage"] == "Builder"
        assert result["next_threshold"] == 0.35
        assert result["points_to_next"] == pytest.approx(0.05, abs=1e-4)

    def test_clamped_score_stored(self):
        result = stage_progress(2.0)
        assert result["sovereignty_score"] == 1.0


class TestAgentTierEquivalent:
    """agent_tier_equivalent() maps human stages to agent tiers."""

    def test_all_stages_have_mapping(self):
        for name in HUMAN_STAGE_ORDER:
            result = agent_tier_equivalent(name)
            assert result in AGENT_TIER_MAP.values()

    def test_unknown_stage_defaults_to_novice(self):
        assert agent_tier_equivalent("NonexistentStage") == "Novice"

    def test_catalyst_is_grandmaster(self):
        assert agent_tier_equivalent("Catalyst") == "Grandmaster"

    def test_seed_is_novice(self):
        assert agent_tier_equivalent("Seed") == "Novice"
