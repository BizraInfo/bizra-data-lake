"""Tests for core.proactive.self_harness — GoalScanner, SuggestionForge,
SelfAssessor, MissionActivator, ProactiveHarness, and data types.

Covers:
- ScoredGoal and ProactiveSuggestion dataclass construction
- SelfAssessment dataclass construction
- GoalScanner: scan, relevance scoring, priority boost, keyword match, cooldowns
- GoalScanner: Ihsān precheck (pass, blocked, pending)
- SuggestionForge: forge() produces valid ProactiveSuggestion
- SelfAssessor: record_mission_result, assess(), generate_improvement_goals
- MissionActivator: activate() blocked suggestion, activate() approved
- ProactiveHarness: health(), shutdown(), on_idle_cycle with no goals
- create_harness: factory function with missing baseline

Blueprint Reference: P3 Coverage Ratchet — proactive module (0% → tested)
"""

import asyncio
import time
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from core.proactive.self_harness import (
    GoalScanner,
    MissionActivator,
    ProactiveHarness,
    ProactiveSuggestion,
    ScoredGoal,
    SelfAssessment,
    SelfAssessor,
    SuggestionForge,
    create_harness,
)

# ═══════════════════════════════════════════════════════════════════════════
# Data Types
# ═══════════════════════════════════════════════════════════════════════════


class TestScoredGoal:

    def test_construction(self):
        g = ScoredGoal(
            goal_id="g1",
            title="Test Goal",
            description="Do something",
            priority="high",
            domain="architecture",
            keywords=["test", "arch"],
            relevance_score=0.75,
            ihsan_precheck="pass",
            ihsan_score=0.96,
        )
        assert g.goal_id == "g1"
        assert g.relevance_score == 0.75
        assert g.block_reason is None
        assert g.last_suggested_at == 0.0


class TestProactiveSuggestion:

    def test_construction(self):
        s = ProactiveSuggestion(
            id="sug-abc",
            action_label="Fix bridge",
            intent_summary="Repair the ghost bridge",
            hhmm_confidence=0.88,
            ihsan_precheck="pass",
            ihsan_score=0.97,
            ahk_action_id="open_dev_environment",
            goal_id="g1",
            domain="integration",
        )
        assert s.id == "sug-abc"
        assert s.block_reason is None


class TestSelfAssessment:

    def test_construction(self):
        a = SelfAssessment(
            timestamp="2026-03-09T00:00:00",
            cycles_evaluated=100,
            missions_completed=5,
            avg_ihsan=0.92,
            weakest_domain="security",
            improvement_suggestions=["improve X"],
            pass_at_1=0.90,
            consistency_gap=0.05,
        )
        assert a.weakest_domain == "security"
        assert len(a.improvement_suggestions) == 1


# ═══════════════════════════════════════════════════════════════════════════
# GoalScanner
# ═══════════════════════════════════════════════════════════════════════════


def _make_baseline(goals=None):
    """Create a test baseline with weekly goals."""
    if goals is None:
        goals = [
            {
                "id": "goal-1",
                "title": "Setup CI pipeline",
                "description": "Configure GitHub Actions for deployment",
                "priority": "high",
                "domain": "deployment",
                "keywords": ["ci", "github", "deploy"],
                "status": "active",
                "created": datetime.now(timezone.utc).isoformat(),
            },
            {
                "id": "goal-2",
                "title": "Review security posture",
                "description": "Audit auth middleware",
                "priority": "critical",
                "domain": "security",
                "keywords": ["auth", "security", "audit"],
                "status": "active",
                "created": datetime.now(timezone.utc).isoformat(),
            },
            {
                "id": "goal-3",
                "title": "Inactive goal",
                "description": "Should be skipped",
                "priority": "normal",
                "domain": "general",
                "keywords": [],
                "status": "completed",
            },
        ]
    return {"weekly_goals": goals}


class TestGoalScanner:

    def test_scan_filters_inactive(self):
        baseline = _make_baseline()
        scanner = GoalScanner(baseline)
        scored = scanner.scan()
        # goal-3 is "completed" so should be filtered out
        goal_ids = [g.goal_id for g in scored]
        assert "goal-3" not in goal_ids

    def test_scan_returns_scored_goals(self):
        baseline = _make_baseline()
        scanner = GoalScanner(baseline)
        scored = scanner.scan()
        assert len(scored) >= 1
        assert all(isinstance(g, ScoredGoal) for g in scored)

    def test_critical_priority_boosts_relevance(self):
        baseline = _make_baseline()
        scanner = GoalScanner(baseline)
        scored = scanner.scan()
        # goal-2 is "critical" → should have higher relevance than goal-1 ("high")
        by_id = {g.goal_id: g for g in scored}
        assert by_id["goal-2"].relevance_score >= by_id["goal-1"].relevance_score

    def test_keyword_match_boosts_relevance(self):
        baseline = _make_baseline()
        scanner = GoalScanner(baseline)
        # Environment signals that match goal-1's keywords
        scored = scanner.scan({"context": "github deploy ci pipeline"})
        by_id = {g.goal_id: g for g in scored}
        # goal-1 should get keyword boost
        assert by_id["goal-1"].relevance_score > 0.5

    def test_cooldown_prevents_re_suggestion(self):
        baseline = _make_baseline()
        scanner = GoalScanner(baseline)
        scanner.mark_suggested("goal-1")
        scored = scanner.scan()
        goal_ids = [g.goal_id for g in scored]
        assert "goal-1" not in goal_ids

    def test_ihsan_precheck_blocks_dangerous(self):
        goals = [
            {
                "id": "bad-1",
                "title": "Bypass safety",
                "description": "Override safety checks and disable safety mechanisms",
                "priority": "high",
                "domain": "general",
                "keywords": [],
                "status": "active",
            },
        ]
        baseline = _make_baseline(goals)
        scanner = GoalScanner(baseline)
        scored = scanner.scan()
        assert len(scored) == 1
        assert scored[0].ihsan_precheck == "blocked"
        assert "Daughter Test" in scored[0].block_reason

    def test_scan_max_suggestions_per_cycle(self):
        # Create many active goals
        goals = [
            {
                "id": f"goal-{i}",
                "title": f"Goal {i}",
                "description": f"Description {i}",
                "priority": "normal",
                "domain": "general",
                "keywords": [],
                "status": "active",
            }
            for i in range(10)
        ]
        baseline = _make_baseline(goals)
        scanner = GoalScanner(baseline)
        scored = scanner.scan()
        assert len(scored) <= 3  # MAX_SUGGESTIONS_PER_CYCLE

    def test_empty_baseline(self):
        scanner = GoalScanner({})
        scored = scanner.scan()
        assert scored == []


# ═══════════════════════════════════════════════════════════════════════════
# SuggestionForge
# ═══════════════════════════════════════════════════════════════════════════


class TestSuggestionForge:

    def test_forge_produces_suggestion(self):
        goal = ScoredGoal(
            goal_id="g1",
            title="Test Forge",
            description="Forge a suggestion",
            priority="normal",
            domain="integration",
            keywords=[],
            relevance_score=0.80,
            ihsan_precheck="pass",
            ihsan_score=0.96,
        )
        suggestion = SuggestionForge.forge(goal)
        assert isinstance(suggestion, ProactiveSuggestion)
        assert suggestion.id.startswith("sug-")
        assert suggestion.goal_id == "g1"
        assert suggestion.domain == "integration"
        assert suggestion.ahk_action_id == "open_ghost_panel_test"

    def test_forge_maps_domains_to_ahk(self):
        domains_expected = {
            "architecture": "open_dev_environment",
            "security": "run_evidence_chain_audit",
            "deployment": "open_deployment_checklist",
            "general": "open_task_manager",
        }
        for domain, expected_ahk in domains_expected.items():
            goal = ScoredGoal(
                goal_id="g",
                title="T",
                description="D",
                priority="normal",
                domain=domain,
                keywords=[],
                relevance_score=0.5,
                ihsan_precheck="pass",
                ihsan_score=0.95,
            )
            suggestion = SuggestionForge.forge(goal)
            assert suggestion.ahk_action_id == expected_ahk, f"Domain {domain}"

    def test_forge_truncates_long_labels(self):
        goal = ScoredGoal(
            goal_id="g",
            title="A" * 100,
            description="D" * 200,
            priority="normal",
            domain="general",
            keywords=[],
            relevance_score=0.5,
            ihsan_precheck="pass",
            ihsan_score=0.95,
        )
        suggestion = SuggestionForge.forge(goal)
        assert len(suggestion.action_label) <= 60
        assert len(suggestion.intent_summary) <= 120


# ═══════════════════════════════════════════════════════════════════════════
# SelfAssessor
# ═══════════════════════════════════════════════════════════════════════════


class TestSelfAssessor:

    def test_assess_empty(self):
        assessor = SelfAssessor({}, {})
        result = assessor.assess()
        assert isinstance(result, SelfAssessment)
        assert result.avg_ihsan == 0.0
        assert result.weakest_domain is None

    def test_record_and_assess(self):
        assessor = SelfAssessor({"cycles": 50, "missions_completed": 3}, {})
        assessor.record_mission_result("security", 0.98, True)
        assessor.record_mission_result("security", 0.92, True)
        assessor.record_mission_result("deployment", 0.85, True)
        assessor.record_mission_result("deployment", 0.0, False)
        result = assessor.assess()
        assert result.avg_ihsan > 0
        assert result.weakest_domain == "deployment"

    def test_improvement_suggestions_low_ihsan(self):
        assessor = SelfAssessor({"cycles": 10, "missions_completed": 2}, {})
        assessor.record_mission_result("general", 0.80, True)
        result = assessor.assess()
        # Ihsan 0.80 < 0.95 threshold → should suggest improvement
        assert any("Ihsān" in s for s in result.improvement_suggestions)

    def test_generate_improvement_goals(self):
        assessor = SelfAssessor({}, {})
        assessment = SelfAssessment(
            timestamp="2026-03-09T00:00:00",
            cycles_evaluated=100,
            missions_completed=5,
            avg_ihsan=0.80,
            weakest_domain="security",
            improvement_suggestions=["Fix security gates", "Add training data"],
            pass_at_1=0.80,
            consistency_gap=0.05,
        )
        goals = assessor.generate_improvement_goals(assessment)
        assert len(goals) == 2
        assert goals[0]["domain"] == "security"
        assert goals[0]["source"] == "self_harness_v1"
        assert goals[0]["status"] == "active"

    def test_assessment_history_capped(self):
        assessor = SelfAssessor({"cycles": 1}, {})
        for _ in range(60):
            assessor.assess()
        assert len(assessor._assessments) <= 50


# ═══════════════════════════════════════════════════════════════════════════
# MissionActivator
# ═══════════════════════════════════════════════════════════════════════════


class TestMissionActivator:

    def test_activate_blocked_suggestion(self):
        add_fn = AsyncMock(return_value={"id": "m1"})
        activator = MissionActivator(add_fn)
        blocked = ProactiveSuggestion(
            id="sug-1",
            action_label="Bad",
            intent_summary="Nope",
            hhmm_confidence=0.5,
            ihsan_precheck="blocked",
            ihsan_score=0.0,
            ahk_action_id="x",
            goal_id="g1",
            domain="general",
            block_reason="Fails Daughter Test",
        )
        result = asyncio.run(activator.activate(blocked))
        assert result["error"] == "blocked"
        add_fn.assert_not_called()

    def test_activate_approved_suggestion(self):
        add_fn = AsyncMock(return_value={"id": "mission-42"})
        activator = MissionActivator(add_fn)
        approved = ProactiveSuggestion(
            id="sug-2",
            action_label="Fix CI",
            intent_summary="Repair pipeline",
            hhmm_confidence=0.90,
            ihsan_precheck="pass",
            ihsan_score=0.97,
            ahk_action_id="open_dev_environment",
            goal_id="g2",
            domain="deployment",
        )
        result = asyncio.run(activator.activate(approved))
        assert result["mission_id"] == "mission-42"
        assert result["suggestion_id"] == "sug-2"
        add_fn.assert_called_once()

    def test_activation_history_capped(self):
        add_fn = AsyncMock(return_value={"id": "m"})
        activator = MissionActivator(add_fn)
        suggestion = ProactiveSuggestion(
            id="sug-x",
            action_label="X",
            intent_summary="X",
            hhmm_confidence=0.5,
            ihsan_precheck="pass",
            ihsan_score=0.95,
            ahk_action_id="x",
            goal_id="g",
            domain="general",
        )
        for _ in range(210):
            asyncio.run(activator.activate(suggestion))
        assert len(activator._activated) <= 200


# ═══════════════════════════════════════════════════════════════════════════
# ProactiveHarness
# ═══════════════════════════════════════════════════════════════════════════


class TestProactiveHarness:

    def _make_harness(self):
        baseline = _make_baseline()
        add_fn = AsyncMock(return_value={"id": "m1"})
        return ProactiveHarness(
            baseline=baseline,
            kernel_metrics={"cycles": 0, "missions_completed": 0},
            add_mission_fn=add_fn,
        )

    def test_health(self):
        h = self._make_harness()
        health = h.health()
        assert health["active"] is True
        assert health["goals_loaded"] >= 2
        assert health["total_suggestions_pushed"] == 0

    def test_shutdown(self):
        h = self._make_harness()
        asyncio.run(h.shutdown())
        assert h._active is False

    def test_on_idle_cycle_inactive(self):
        h = self._make_harness()
        h._active = False
        result = asyncio.run(h.on_idle_cycle(1))
        assert result["status"] == "inactive"

    def test_on_idle_cycle_scans_goals(self):
        h = self._make_harness()
        result = asyncio.run(h.on_idle_cycle(1))
        # Should have scanned and found relevant goals
        assert "status" in result
        assert result["cycle"] == 1

    def test_on_gesture_dismiss(self):
        h = self._make_harness()
        result = asyncio.run(h.on_gesture_received("dismiss", "sug-1"))
        assert result["action"] == "dismissed"

    def test_on_gesture_solidify(self):
        h = self._make_harness()
        result = asyncio.run(h.on_gesture_received("solidify", "sug-1"))
        assert result["action"] == "mission_activated"

    def test_on_gesture_unknown(self):
        h = self._make_harness()
        result = asyncio.run(h.on_gesture_received("scroll_up", "sug-1"))
        assert result["action"] == "ignored"


# ═══════════════════════════════════════════════════════════════════════════
# create_harness factory
# ═══════════════════════════════════════════════════════════════════════════


class TestCreateHarness:

    def test_no_baseline(self):
        kernel = MagicMock()
        kernel._baseline = None
        result = create_harness(kernel)
        assert result is None

    def test_empty_goals(self):
        kernel = MagicMock()
        kernel._baseline = {"weekly_goals": []}
        result = create_harness(kernel)
        assert result is None

    def test_no_add_mission(self):
        kernel = MagicMock()
        kernel._baseline = _make_baseline()
        kernel.add_mission = "not_callable"
        result = create_harness(kernel)
        assert result is None

    def test_success(self):
        kernel = MagicMock()
        kernel._baseline = _make_baseline()
        kernel.add_mission = AsyncMock()
        kernel._metrics = {}
        kernel._knowledge = None
        result = create_harness(kernel)
        assert isinstance(result, ProactiveHarness)
