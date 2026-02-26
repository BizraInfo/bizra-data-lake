"""
Tests for AdaptivePriorLearning — Bayesian Prior Over Change Categories.

CI-safe: pure Python, no external services.
"""

from __future__ import annotations

from core.benchmark.adaptive_prior import AdaptivePriorLearning, CategoryBelief


class TestAdaptivePriorLearning:
    """Unit tests for AdaptivePriorLearning."""

    # ─── Initialisation ─────────────────────────────────────────────────────────

    def test_all_categories_initialised(self):
        """All CHANGE_CATEGORIES must have beliefs after construction."""
        prior = AdaptivePriorLearning()
        report = prior.get_report()
        for cat in AdaptivePriorLearning.CHANGE_CATEGORIES:
            assert cat in report, f"Category {cat} missing from report"

    def test_initial_posterior_mean(self):
        """Initial posterior_mean must match initial_prior."""
        initial = 0.5
        prior = AdaptivePriorLearning(initial_prior=initial, pseudo_count=10)
        for cat in AdaptivePriorLearning.CHANGE_CATEGORIES:
            belief = prior._beliefs[cat]
            assert abs(belief.posterior_mean - initial) < 1e-9

    # ─── update_beliefs ─────────────────────────────────────────────────────────

    def test_prior_updates_on_improvement(self):
        """Positive improvement_delta must increase alpha (success pseudo-count)."""
        prior = AdaptivePriorLearning()
        cat = "attention_changes"
        before_alpha = prior._beliefs[cat].alpha
        prior.update_beliefs("attention head", improvement_delta=0.05)
        after_alpha = prior._beliefs[cat].alpha
        assert after_alpha > before_alpha, "alpha should increase on success"

    def test_prior_updates_on_failure(self):
        """Zero/negative delta must increase beta (failure pseudo-count)."""
        prior = AdaptivePriorLearning()
        cat = "attention_changes"
        before_beta = prior._beliefs[cat].beta_param
        prior.update_beliefs("attention head", improvement_delta=-0.01)
        after_beta = prior._beliefs[cat].beta_param
        assert after_beta > before_beta, "beta should increase on failure"

    def test_attempts_incremented(self):
        """attempts counter must increment after each update."""
        prior = AdaptivePriorLearning()
        prior.update_beliefs("attention head", improvement_delta=0.02)
        prior.update_beliefs("attention head", improvement_delta=-0.01)
        cat = "attention_changes"
        assert prior._beliefs[cat].attempts == 2

    def test_successes_incremented_on_improvement(self):
        """successes counter must increment only on positive delta."""
        prior = AdaptivePriorLearning()
        prior.update_beliefs("depth layer", improvement_delta=0.03)
        prior.update_beliefs("depth layer", improvement_delta=-0.02)
        cat = "depth_changes"
        assert prior._beliefs[cat].successes == 1

    def test_update_returns_category_belief(self):
        """update_beliefs() must return a CategoryBelief."""
        prior = AdaptivePriorLearning()
        result = prior.update_beliefs("optimizer adam", improvement_delta=0.01)
        assert isinstance(result, CategoryBelief)

    # ─── suggest_priority_order ─────────────────────────────────────────────────

    def test_suggest_priority_order_is_sorted(self):
        """suggest_priority_order() must return categories sorted by posterior_mean descending."""
        prior = AdaptivePriorLearning()
        # Strongly reinforce one category using a unique keyword ("rms"
        # matches only normalization_changes, unlike "layer_norm" which
        # also contains "layer" → depth_changes).
        for _ in range(20):
            prior.update_beliefs("rms normalization", improvement_delta=0.05)
        order = prior.suggest_priority_order()
        assert (
            order[0] == "normalization_changes"
        ), "Most reinforced category should be first"
        # Verify sorted.
        means = [prior._beliefs[cat].posterior_mean for cat in order]
        assert means == sorted(means, reverse=True)

    def test_suggest_returns_all_categories(self):
        """suggest_priority_order() must return all CHANGE_CATEGORIES."""
        prior = AdaptivePriorLearning()
        order = prior.suggest_priority_order()
        assert sorted(order) == sorted(AdaptivePriorLearning.CHANGE_CATEGORIES)

    # ─── Bayesian convergence ───────────────────────────────────────────────────

    def test_beta_binomial_converges_to_empirical(self):
        """After many updates, posterior_mean should approach empirical success rate."""
        prior = AdaptivePriorLearning(pseudo_count=2)  # Weak prior
        cat = "width_changes"
        # 80% success rate.
        for _ in range(80):
            prior.update_beliefs("width hidden size", improvement_delta=0.01)
        for _ in range(20):
            prior.update_beliefs("width hidden size", improvement_delta=-0.01)
        belief = prior._beliefs[cat]
        # Posterior mean should be near 0.80.
        assert abs(belief.posterior_mean - 0.80) < 0.05

    # ─── Unknown category fallback ──────────────────────────────────────────────

    def test_unknown_category_maps_to_default(self):
        """Unknown change_type must map to DEFAULT_CATEGORY without raising."""
        prior = AdaptivePriorLearning()
        default = AdaptivePriorLearning.DEFAULT_CATEGORY
        before = prior._beliefs[default].attempts
        prior.update_beliefs("completely_unknown_xyz_component", improvement_delta=0.1)
        after = prior._beliefs[default].attempts
        assert (
            after == before + 1
        ), "Unknown category should fall back to DEFAULT_CATEGORY"

    # ─── get_report ─────────────────────────────────────────────────────────────

    def test_get_report_structure(self):
        """get_report() must include expected keys per category."""
        prior = AdaptivePriorLearning()
        report = prior.get_report()
        for cat, info in report.items():
            assert "posterior_mean" in info
            assert "alpha" in info
            assert "beta" in info
            assert "attempts" in info
            assert "successes" in info
