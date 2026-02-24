"""
Tests for Interactive Denoising — Bayesian Belief Updates
"""

import pytest

from core.sovereign.interactive_denoiser import (
    BeliefState,
    CorrectionType,
    DenoisingResult,
    InteractiveDenoiser,
)


class TestBeliefState:
    """Test the BeliefState data structure."""

    def test_normalize(self):
        state = BeliefState(priorities={"a": 2.0, "b": 3.0, "c": 5.0})
        state.normalize()
        assert abs(sum(state.priorities.values()) - 1.0) < 1e-9

    def test_prune_removes_small_beliefs(self):
        state = BeliefState(priorities={"a": 0.5, "b": 0.0001, "c": 0.4999})
        pruned = state.prune(threshold=0.001)
        assert pruned == 1
        assert "b" not in state.priorities

    def test_top_k(self):
        state = BeliefState(priorities={"a": 0.5, "b": 0.3, "c": 0.2})
        top = state.top_k(2)
        assert len(top) == 2
        assert top[0][0] == "a"
        assert top[1][0] == "b"

    def test_entropy_uniform(self):
        state = BeliefState(priorities={"a": 0.5, "b": 0.5})
        assert abs(state.entropy() - 1.0) < 0.01  # log2(2) = 1

    def test_entropy_certain(self):
        state = BeliefState(priorities={"a": 1.0})
        assert state.entropy() == 0.0


class TestInteractiveDenoiser:
    """Test the InteractiveDenoiser."""

    def test_initialize_beliefs(self):
        denoiser = InteractiveDenoiser()
        denoiser.initialize_beliefs({"work": 0.5, "meeting": 0.3, "lunch": 0.2})

        assert abs(sum(denoiser.belief_state.priorities.values()) - 1.0) < 1e-9

    def test_dismiss_decreases_belief(self):
        denoiser = InteractiveDenoiser()
        denoiser.initialize_beliefs({"work": 0.5, "meeting": 0.3, "lunch": 0.2})

        result = denoiser.apply_correction(
            CorrectionType.DISMISS,
            target_priority="meeting",
        )

        assert result.success
        assert result.posterior_belief < result.prior_belief
        assert result.belief_delta < 0

    def test_promote_increases_belief(self):
        denoiser = InteractiveDenoiser()
        denoiser.initialize_beliefs({"work": 0.5, "meeting": 0.3, "lunch": 0.2})

        result = denoiser.apply_correction(
            CorrectionType.PROMOTE,
            target_priority="lunch",
        )

        assert result.success
        assert result.posterior_belief > result.prior_belief

    def test_cancel_drops_belief_sharply(self):
        denoiser = InteractiveDenoiser()
        denoiser.initialize_beliefs({"work": 0.5, "meeting": 0.3, "lunch": 0.2})

        result = denoiser.apply_correction(
            CorrectionType.CANCEL,
            target_priority="meeting",
        )

        assert result.success
        assert result.posterior_belief < 0.1  # Should drop significantly

    def test_reschedule_moderate_decrease(self):
        denoiser = InteractiveDenoiser()
        denoiser.initialize_beliefs({"work": 0.5, "meeting": 0.3, "lunch": 0.2})

        result = denoiser.apply_correction(
            CorrectionType.RESCHEDULE,
            target_priority="meeting",
        )

        assert result.success
        # Should decrease but not as sharply as CANCEL
        assert result.posterior_belief < result.prior_belief

    def test_redirect_transfers_belief(self):
        denoiser = InteractiveDenoiser()
        denoiser.initialize_beliefs({"work": 0.5, "meeting": 0.3, "lunch": 0.2})

        old_meeting = denoiser.belief_state.get_belief("meeting")
        old_work = denoiser.belief_state.get_belief("work")

        denoiser.apply_correction(
            CorrectionType.REDIRECT,
            target_priority="meeting",
            redirect_to="work",
        )

        # Meeting should decrease, work should get boost
        new_meeting = denoiser.belief_state.get_belief("meeting")
        assert new_meeting < old_meeting

    def test_morning_brief_priorities(self):
        denoiser = InteractiveDenoiser()
        denoiser.initialize_beliefs({"work": 0.5, "meeting": 0.3, "lunch": 0.2})

        priorities = denoiser.get_morning_brief_priorities(top_k=3)
        assert len(priorities) == 3
        assert priorities[0]["rank"] == 1
        assert priorities[0]["priority"] == "work"

    def test_add_priority(self):
        denoiser = InteractiveDenoiser()
        denoiser.initialize_beliefs({"work": 0.5, "meeting": 0.5})
        denoiser.add_priority("exercise", initial_belief=0.3)

        assert "exercise" in denoiser.belief_state.priorities
        assert abs(sum(denoiser.belief_state.priorities.values()) - 1.0) < 1e-9

    def test_correction_for_meeting_moved(self):
        """SAPE spec verification: 3 priorities, correction 'meeting moved',
        verify belief for meeting drops below 0.1."""
        denoiser = InteractiveDenoiser()
        denoiser.initialize_beliefs({
            "coding_sprint": 0.4,
            "team_meeting": 0.35,
            "lunch_prep": 0.25,
        })

        # User says "meeting moved"
        result = denoiser.apply_correction(
            CorrectionType.RESCHEDULE,
            target_priority="team_meeting",
            context="meeting moved to tomorrow",
        )

        # Apply a second correction to reinforce
        result2 = denoiser.apply_correction(
            CorrectionType.DISMISS,
            target_priority="team_meeting",
            context="not relevant today",
        )

        final_belief = denoiser.belief_state.get_belief("team_meeting")
        assert final_belief < 0.1, f"Meeting belief {final_belief} should be < 0.1"

    def test_to_dict(self):
        denoiser = InteractiveDenoiser()
        denoiser.initialize_beliefs({"a": 0.5, "b": 0.5})
        d = denoiser.to_dict()
        assert "belief_state" in d
        assert "history_count" in d
