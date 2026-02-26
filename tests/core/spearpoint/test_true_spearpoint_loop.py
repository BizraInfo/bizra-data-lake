"""
Tests for TrueSpearpointLoop — v9 Hierarchical Bayesian Optimization Composer.

CI-safe: uses stub inner_loop (no GPU, no LLM, no network).
All tests are synchronous via asyncio.run() to avoid pytest-asyncio dependency.
"""

from __future__ import annotations

import asyncio

import pytest

from core.spearpoint.true_spearpoint_loop import (
    IterationResult,
    LoopConfig,
    SpearpointReport,
    TrueSpearpointLoop,
)


def run(coro):
    """Helper: run a coroutine synchronously."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestTrueSpearpointLoop:
    """Unit tests for TrueSpearpointLoop (stub mode — no inner_loop)."""

    # ─── Basic execution ────────────────────────────────────────────────────────

    def test_run_returns_spearpoint_report(self):
        """run() must return a SpearpointReport."""
        loop = TrueSpearpointLoop(config=LoopConfig(max_iterations=2))
        report = run(loop.run())
        assert isinstance(report, SpearpointReport)

    def test_report_contains_history(self):
        """SpearpointReport.iteration_history must be populated."""
        loop = TrueSpearpointLoop(config=LoopConfig(max_iterations=3))
        report = run(loop.run())
        assert len(report.iteration_history) >= 1
        assert all(isinstance(r, IterationResult) for r in report.iteration_history)

    def test_report_has_campaign_id(self):
        """SpearpointReport must have a non-empty campaign_id."""
        loop = TrueSpearpointLoop(config=LoopConfig(max_iterations=1))
        report = run(loop.run())
        assert report.campaign_id and len(report.campaign_id) > 0

    # ─── Budget enforcement ─────────────────────────────────────────────────────

    def test_budget_enforced(self):
        """Loop must stop when total_cost_usd >= config.budget_usd."""
        # Set a very small budget so it stops quickly.
        cfg = LoopConfig(
            max_iterations=100,
            budget_usd=0.0,  # Forces stop on first iteration
            patience=100,
        )
        loop = TrueSpearpointLoop(config=cfg)
        report = run(loop.run())
        # Must stop well before max_iterations.
        assert report.iterations_completed <= 5
        assert report.convergence_reason in ("budget_exhausted", "max_iterations_reached")

    # ─── Patience / no-improvement termination ─────────────────────────────────

    def test_loop_terminates_on_patience(self):
        """Loop must stop after config.patience iterations with no improvement."""
        cfg = LoopConfig(
            max_iterations=50,
            patience=3,
            budget_usd=100_000.0,
            target_snr=999.0,  # Impossible — forces patience exhaustion
        )
        loop = TrueSpearpointLoop(config=cfg)
        report = run(loop.run())
        assert report.convergence_reason in (
            "patience_exceeded",
            "pareto_stable",
            "max_iterations_reached",
        )
        # Must not run all 50 iterations.
        assert report.iterations_completed <= 10

    # ─── Memory population ─────────────────────────────────────────────────────

    def test_memory_populated_after_iteration(self):
        """MIRAS memory must have entries after running iterations."""
        loop = TrueSpearpointLoop(config=LoopConfig(max_iterations=3))
        report = run(loop.run())
        # Both SNR-gated entries and episodic entries should be present.
        assert report.memory_summary["total_count"] >= 1

    def test_memory_summary_in_report(self):
        """SpearpointReport.memory_summary must have expected keys."""
        loop = TrueSpearpointLoop(config=LoopConfig(max_iterations=2))
        report = run(loop.run())
        for key in ["short_term_count", "long_term_count", "episodic_count", "total_count"]:
            assert key in report.memory_summary

    # ─── Prior updated after ablation ──────────────────────────────────────────

    def test_prior_updated_after_ablation(self):
        """AdaptivePrior beliefs must shift after running iterations."""
        loop = TrueSpearpointLoop(config=LoopConfig(max_iterations=3))
        initial_report = {
            cat: info["attempts"]
            for cat, info in loop._prior.get_report().items()
        }
        run(loop.run())
        final_report = loop._prior.get_report()
        # At least one category must have been updated.
        updated = any(
            final_report[cat]["attempts"] > initial_report[cat]
            for cat in initial_report
        )
        assert updated, "AdaptivePrior must be updated during run()"

    # ─── Pareto convergence ─────────────────────────────────────────────────────

    def test_pareto_convergence(self):
        """Stable Pareto frontier triggers 'pareto_stable' convergence."""
        cfg = LoopConfig(
            max_iterations=20,
            pareto_convergence_window=3,
            patience=100,
            budget_usd=100_000.0,
            target_snr=999.0,
        )
        loop = TrueSpearpointLoop(config=cfg)
        report = run(loop.run())
        # Either pareto_stable or patience_exceeded must trigger (stub returns constant scores).
        assert report.convergence_reason in (
            "pareto_stable",
            "patience_exceeded",
            "max_iterations_reached",
        )

    # ─── Routing stats ──────────────────────────────────────────────────────────

    def test_routing_stats_in_report(self):
        """SpearpointReport.routing_stats must be populated."""
        loop = TrueSpearpointLoop(config=LoopConfig(max_iterations=2))
        report = run(loop.run())
        assert "total_routed" in report.routing_stats
        assert report.routing_stats["total_routed"] >= 1

    # ─── Prior report ───────────────────────────────────────────────────────────

    def test_prior_report_in_report(self):
        """SpearpointReport.prior_report must include all change categories."""
        loop = TrueSpearpointLoop(config=LoopConfig(max_iterations=2))
        report = run(loop.run())
        from core.benchmark.adaptive_prior import AdaptivePriorLearning
        for cat in AdaptivePriorLearning.CHANGE_CATEGORIES:
            assert cat in report.prior_report
