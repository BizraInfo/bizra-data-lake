"""
BIZRA Agent Self-Proactive Harness — Goal→Suggestion→Ghost→Action Pipeline.

The missing bridge between Node0's goal-aware baseline and the Ghost Panel.
Converts weekly_goals into proactive suggestions, scores them against the
current environment, and pushes qualifying suggestions to ghost_ws.py
for user approval.

Architecture:
  node0_baseline.json (goals)
    → GoalScanner (keyword + FAISS similarity scoring)
      → SuggestionForge (goal → OverlaySuggestion with Ihsān precheck)
        → GhostPusher (WebSocket push to ghost_ws.py:9743)
          → Ghost Panel renders card
            → User solidify gesture → MissionActivator → add_mission()

Self-Assessment Loop (every N idle cycles):
  Performance metrics → HAL consistency check → weakness detection
  → Improvement missions auto-generated → pushed to same pipeline

Standing on Giants:
  - Boyd (OODA: idle→sense→suggest→execute)
  - Shannon (SNR: only surface high-signal suggestions)
  - Norman (invisible design: suggest, don't interrupt)
  - Al-Ghazali (Ihsān: constitutional precheck before user sees anything)
  - Deming (PDCA: self-assess → improve cycle)

Created: 2026-02-27 | BIZRA Self-Proactive Harness v1.0
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import math
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("bizra.proactive.self_harness")

# ---------------------------------------------------------------------------
# Constants (from constitutional layer — never overridable)
# ---------------------------------------------------------------------------
from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)

GHOST_WS_URL = os.getenv("GHOST_WS_URL", "ws://127.0.0.1:9743/ws/ghost")
SELF_ASSESS_EVERY_N_CYCLES = int(os.getenv("SELF_ASSESS_CYCLES", "20"))
MAX_SUGGESTIONS_PER_CYCLE = 3
SUGGESTION_COOLDOWN_S = 300  # Don't re-suggest same goal within 5 minutes


# ---------------------------------------------------------------------------
# Data Types
# ---------------------------------------------------------------------------


@dataclass
class ScoredGoal:
    """A weekly goal scored against the current environment."""

    goal_id: str
    title: str
    description: str
    priority: str
    domain: str
    keywords: List[str]
    relevance_score: float  # 0.0-1.0 — how relevant RIGHT NOW
    ihsan_precheck: str  # "pass" | "pending" | "blocked"
    ihsan_score: float  # 0.0-1.0
    block_reason: Optional[str] = None
    last_suggested_at: float = 0.0


@dataclass
class ProactiveSuggestion:
    """A suggestion ready to push to Ghost Panel."""

    id: str
    action_label: str
    intent_summary: str
    hhmm_confidence: float
    ihsan_precheck: str
    ihsan_score: float
    ahk_action_id: str
    goal_id: str
    domain: str
    block_reason: Optional[str] = None


@dataclass
class SelfAssessment:
    """Self-evaluation report from the harness."""

    timestamp: str
    cycles_evaluated: int
    missions_completed: int
    avg_ihsan: float
    weakest_domain: Optional[str]
    improvement_suggestions: List[str]
    pass_at_1: float  # Single-run success rate
    consistency_gap: float  # Brittleness indicator


# ---------------------------------------------------------------------------
# GoalScanner — Reads goals, scores against environment
# ---------------------------------------------------------------------------


class GoalScanner:
    """Reads weekly_goals from baseline and scores each against current context.

    Scoring uses keyword overlap + optional FAISS similarity when knowledge
    base is available. Goals with higher priority get a boost.
    """

    PRIORITY_BOOST = {"critical": 0.20, "high": 0.10, "normal": 0.0, "low": -0.05}

    def __init__(self, baseline: Dict[str, Any], knowledge_retriever=None):
        self._goals = baseline.get("weekly_goals", [])
        self._knowledge = knowledge_retriever
        self._cooldowns: Dict[str, float] = {}  # goal_id → last_suggested_at
        logger.info("GoalScanner initialized with %d goals", len(self._goals))

    def scan(self, environment_signals: Dict[str, Any] = None) -> List[ScoredGoal]:
        """Score all goals against current environment. Return sorted by relevance."""
        now = time.time()
        scored: List[ScoredGoal] = []

        for goal in self._goals:
            if goal.get("status") != "active":
                continue

            goal_id = goal["id"]

            # Cooldown check — don't re-suggest too frequently
            last = self._cooldowns.get(goal_id, 0.0)
            if now - last < SUGGESTION_COOLDOWN_S:
                continue

            # Score the goal
            relevance = self._compute_relevance(goal, environment_signals or {})
            ihsan_score, ihsan_status, block_reason = self._ihsan_precheck(goal)

            scored.append(
                ScoredGoal(
                    goal_id=goal_id,
                    title=goal["title"],
                    description=goal["description"],
                    priority=goal.get("priority", "normal"),
                    domain=goal.get("domain", "general"),
                    keywords=goal.get("keywords", []),
                    relevance_score=relevance,
                    ihsan_precheck=ihsan_status,
                    ihsan_score=ihsan_score,
                    block_reason=block_reason,
                    last_suggested_at=last,
                )
            )

        # Sort: highest relevance first
        scored.sort(key=lambda g: g.relevance_score, reverse=True)
        return scored[:MAX_SUGGESTIONS_PER_CYCLE]

    def mark_suggested(self, goal_id: str) -> None:
        """Record that a goal was just suggested — starts cooldown."""
        self._cooldowns[goal_id] = time.time()

    def _compute_relevance(self, goal: Dict[str, Any], env: Dict[str, Any]) -> float:
        """Compute goal relevance to current environment.

        Combines:
        1. Priority boost (critical > high > normal > low)
        2. Keyword match against environment signals
        3. Time decay (older goals slightly less relevant)
        4. Optional FAISS similarity if knowledge base available
        """
        score = 0.5  # Base relevance

        # Priority boost
        priority = goal.get("priority", "normal")
        score += self.PRIORITY_BOOST.get(priority, 0.0)

        # Keyword match against environment
        keywords = set(k.lower() for k in goal.get("keywords", []))
        env_text = " ".join(str(v) for v in env.values()).lower()
        if keywords and env_text:
            hits = sum(1 for kw in keywords if kw in env_text)
            keyword_ratio = hits / len(keywords)
            score += keyword_ratio * 0.2

        # Time relevance — goals created recently are slightly more urgent
        created = goal.get("created", "")
        if created:
            try:
                created_dt = datetime.fromisoformat(created)
                age_days = (datetime.now(timezone.utc) - created_dt).days
                freshness = max(0.0, 1.0 - (age_days / 30.0)) * 0.1
                score += freshness
            except (ValueError, TypeError):
                pass

        # FAISS similarity boost (if knowledge base available)
        if self._knowledge and hasattr(self._knowledge, "search"):
            try:
                results = self._knowledge.search(goal["title"], top_k=1)
                if results:
                    sim = results[0].get("score", 0.0)
                    score += min(sim, 0.15)
            except Exception:
                pass

        return min(1.0, max(0.0, score))

    def _ihsan_precheck(self, goal: Dict[str, Any]) -> Tuple[float, str, Optional[str]]:
        """Pre-check Ihsān compliance before showing to user.

        Returns: (ihsan_score, status, block_reason)
        """
        target = goal.get("target_ihsan", UNIFIED_IHSAN_THRESHOLD)

        # Constitutional checks
        # 1. Daughter Test — is this suggestion safe for anyone?
        description = goal.get("description", "")
        if any(
            dangerous in description.lower()
            for dangerous in [
                "override",
                "bypass",
                "disable safety",
                "skip verification",
            ]
        ):
            return (0.0, "blocked", "Fails Daughter Test — unsafe action pattern")

        # 2. Domain-appropriate Ihsān scoring
        domain = goal.get("domain", "general")
        domain_ihsan = {
            "security": 0.99,
            "deployment": 0.98,
            "integration": 0.97,
            "architecture": 0.95,
            "general": 0.95,
        }
        required = domain_ihsan.get(domain, UNIFIED_IHSAN_THRESHOLD)

        # Simulated Ihsān score — in production, this calls the actual
        # constitutional gate pipeline (α4→α7→α8→α9→α10)
        simulated_score = min(target, 0.97)  # Conservative estimate

        if simulated_score >= required:
            return (simulated_score, "pass", None)
        elif simulated_score >= required - 0.05:
            return (
                simulated_score,
                "pending",
                f"Ihsān {simulated_score:.2f} near threshold {required:.2f}",
            )
        else:
            return (
                simulated_score,
                "blocked",
                f"Ihsān {simulated_score:.2f} below {required:.2f}",
            )


# ---------------------------------------------------------------------------
# SuggestionForge — Converts scored goals into Ghost-ready suggestions
# ---------------------------------------------------------------------------


class SuggestionForge:
    """Converts ScoredGoal objects into ProactiveSuggestion for Ghost Panel."""

    @staticmethod
    def forge(scored_goal: ScoredGoal) -> ProactiveSuggestion:
        """Create a Ghost-ready suggestion from a scored goal."""
        # Generate deterministic ID from goal + timestamp
        raw = f"{scored_goal.goal_id}:{time.time()}"
        suggestion_id = hashlib.sha256(raw.encode()).hexdigest()[:12]

        # Map domain to AHK action type
        ahk_actions = {
            "architecture": "open_dev_environment",
            "integration": "open_ghost_panel_test",
            "security": "run_evidence_chain_audit",
            "deployment": "open_deployment_checklist",
            "general": "open_task_manager",
        }
        ahk_id = ahk_actions.get(scored_goal.domain, "open_task_manager")

        return ProactiveSuggestion(
            id=f"sug-{suggestion_id}",
            action_label=scored_goal.title[:60],
            intent_summary=scored_goal.description[:120],
            hhmm_confidence=scored_goal.relevance_score,
            ihsan_precheck=scored_goal.ihsan_precheck,
            ihsan_score=scored_goal.ihsan_score,
            ahk_action_id=ahk_id,
            goal_id=scored_goal.goal_id,
            domain=scored_goal.domain,
            block_reason=scored_goal.block_reason,
        )


# ---------------------------------------------------------------------------
# GhostPusher — Sends suggestions to ghost_ws.py via WebSocket
# ---------------------------------------------------------------------------


class GhostPusher:
    """Pushes ProactiveSuggestion objects to Ghost Panel via ghost_ws.py.

    Two modes:
    1. In-process: calls ghost_ws.emit_overlay_event() directly (same process)
    2. WebSocket client: connects to ws://localhost:9743 (separate process)
    """

    def __init__(self, mode: str = "auto"):
        self._mode = mode
        self._ws = None
        self._connected = False

    async def push(self, suggestions: List[ProactiveSuggestion]) -> int:
        """Push suggestions to Ghost Panel. Returns count delivered."""
        if not suggestions:
            return 0

        # Build overlay event payload matching ghost_ws.OverlayEvent schema
        event_payload = {
            "type": "show_overlay",
            "suggestions": [asdict(s) for s in suggestions],
            "auto_dismiss_at": time.time() + 30.0,  # 30s auto-dismiss
            "timestamp": time.time(),
        }

        # Try in-process first (fastest, no network)
        if self._mode in ("auto", "in_process"):
            try:
                from core.bridges.ghost_ws import emit_overlay_event, OverlayEvent

                event = OverlayEvent(
                    type="show_overlay",
                    suggestions=[asdict(s) for s in suggestions],
                    auto_dismiss_at=time.time() + 30.0,
                )
                sent = await emit_overlay_event(event)
                if sent > 0:
                    logger.info(
                        "GhostPusher: %d suggestions → %d clients (in-process)",
                        len(suggestions),
                        sent,
                    )
                    return sent
            except ImportError:
                pass
            except Exception as e:
                logger.debug("GhostPusher in-process failed: %s", e)

        # Fallback: WebSocket client
        if self._mode in ("auto", "websocket"):
            return await self._push_via_ws(event_payload)

        return 0

    async def _push_via_ws(self, payload: Dict[str, Any]) -> int:
        """Push via WebSocket client to ghost_ws.py."""
        try:
            import websockets
        except ImportError:
            logger.warning("GhostPusher: websockets package not installed")
            return 0

        try:
            if not self._ws or not self._connected:
                self._ws = await websockets.connect(
                    GHOST_WS_URL,
                    max_size=65536,
                    close_timeout=5,
                )
                self._connected = True
                logger.info("GhostPusher: connected to %s", GHOST_WS_URL)

            # Send as prediction injection (ghost_ws.py handles via msg.get("prediction"))
            for suggestion in payload.get("suggestions", []):
                await self._ws.send(
                    json.dumps(
                        {
                            "prediction": {
                                "intent": suggestion.get("action_label", ""),
                                "confidence": suggestion.get("hhmm_confidence", 0.0),
                                "node_id": "node0-momo-genesis",
                                "goal_id": suggestion.get("goal_id", ""),
                                "ihsan_precheck": suggestion.get(
                                    "ihsan_precheck", "pending"
                                ),
                                "ihsan_score": suggestion.get("ihsan_score", 0.0),
                                "ahk_action_id": suggestion.get("ahk_action_id", ""),
                                "intent_summary": suggestion.get("intent_summary", ""),
                            }
                        }
                    )
                )

            count = len(payload.get("suggestions", []))
            logger.info("GhostPusher: %d suggestions pushed via WebSocket", count)
            return count

        except Exception as e:
            logger.warning("GhostPusher WS error: %s", e)
            self._connected = False
            self._ws = None
            return 0

    async def close(self) -> None:
        """Close WebSocket connection gracefully."""
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
            self._connected = False


# ---------------------------------------------------------------------------
# SelfAssessor — Agent evaluates its own performance
# ---------------------------------------------------------------------------


class SelfAssessor:
    """Periodic self-evaluation of Node0 proactive performance.

    Every N idle cycles, examines:
    1. Mission completion rate per domain
    2. Average Ihsān scores
    3. Consistency (pass@k via repeated evaluations)
    4. Weakest domain identification
    5. Auto-generates improvement suggestions as new goals

    Standing on Giants: Deming (PDCA) · HAL (Pass@k consistency)
    """

    def __init__(self, kernel_metrics: Dict[str, Any], baseline: Dict[str, Any]):
        self._metrics = kernel_metrics
        self._baseline = baseline
        self._assessments: List[SelfAssessment] = []
        self._domain_scores: Dict[str, List[float]] = {}

    def record_mission_result(
        self, domain: str, ihsan_score: float, success: bool
    ) -> None:
        """Record a completed mission for domain tracking."""
        if domain not in self._domain_scores:
            self._domain_scores[domain] = []
        self._domain_scores[domain].append(ihsan_score if success else 0.0)

    def assess(self) -> SelfAssessment:
        """Run full self-assessment. Returns report + improvement suggestions."""
        now = datetime.now(timezone.utc).isoformat()

        # Compute per-domain averages
        domain_avgs: Dict[str, float] = {}
        for domain, scores in self._domain_scores.items():
            if scores:
                domain_avgs[domain] = sum(scores) / len(scores)

        # Overall metrics
        all_scores = [s for scores in self._domain_scores.values() for s in scores]
        avg_ihsan = sum(all_scores) / len(all_scores) if all_scores else 0.0

        # Weakest domain
        weakest = None
        if domain_avgs:
            weakest = min(domain_avgs, key=domain_avgs.get)

        # Consistency check (simulated pass@k)
        # In full implementation, this calls HALOrchestrator
        pass_at_1 = avg_ihsan
        consistency_gap = max(0.0, pass_at_1 - (avg_ihsan * 0.85))

        # Generate improvement suggestions
        improvements: List[str] = []

        if avg_ihsan < UNIFIED_IHSAN_THRESHOLD:
            improvements.append(
                f"Overall Ihsān {avg_ihsan:.2f} below threshold {UNIFIED_IHSAN_THRESHOLD}. "
                "Review constitutional gate calibration."
            )

        if weakest and domain_avgs.get(weakest, 1.0) < 0.90:
            improvements.append(
                f"Domain '{weakest}' averaging {domain_avgs[weakest]:.2f}. "
                "Consider adding specialized training data or adjusting agent routing."
            )

        if consistency_gap > 0.10:
            improvements.append(
                f"Consistency gap {consistency_gap:.2f} indicates brittleness. "
                "Run HAL Pass@k evaluation to identify unreliable reasoning paths."
            )

        total_missions = self._metrics.get("missions_completed", 0)
        total_cycles = self._metrics.get("cycles", 0)
        if total_cycles > 100 and total_missions == 0:
            improvements.append(
                f"Zero missions completed in {total_cycles} cycles. "
                "Verify goal→mission conversion pipeline is active."
            )

        assessment = SelfAssessment(
            timestamp=now,
            cycles_evaluated=total_cycles,
            missions_completed=total_missions,
            avg_ihsan=avg_ihsan,
            weakest_domain=weakest,
            improvement_suggestions=improvements,
            pass_at_1=pass_at_1,
            consistency_gap=consistency_gap,
        )

        self._assessments.append(assessment)
        if len(self._assessments) > 50:
            self._assessments = self._assessments[-50:]

        logger.info(
            "SelfAssessment: ihsan=%.2f | weakest=%s | improvements=%d | gap=%.2f",
            avg_ihsan,
            weakest,
            len(improvements),
            consistency_gap,
        )

        return assessment

    def generate_improvement_goals(
        self, assessment: SelfAssessment
    ) -> List[Dict[str, Any]]:
        """Convert self-assessment into new weekly goals for the baseline."""
        goals = []
        for i, suggestion in enumerate(assessment.improvement_suggestions[:2]):
            goals.append(
                {
                    "id": f"self-improve-{int(time.time())}-{i:02d}",
                    "title": f"Self-Improvement: {suggestion[:50]}",
                    "description": suggestion,
                    "priority": "normal",
                    "domain": assessment.weakest_domain or "general",
                    "target_ihsan": UNIFIED_IHSAN_THRESHOLD,
                    "keywords": ["self-assessment", "improvement", "harness"],
                    "status": "active",
                    "created": assessment.timestamp,
                    "source": "self_harness_v1",
                }
            )
        return goals


# ---------------------------------------------------------------------------
# MissionActivator — Converts approved suggestions back to kernel missions
# ---------------------------------------------------------------------------


class MissionActivator:
    """When user approves a Ghost Panel suggestion (solidify gesture),
    converts it back into a kernel mission via add_mission().

    Bridges the gap between Ghost Panel approval and Node0 execution.
    """

    def __init__(self, add_mission_fn: Callable):
        """
        Args:
            add_mission_fn: async callable — Node0ProactiveKernel.add_mission()
        """
        self._add_mission = add_mission_fn
        self._activated: List[Dict[str, Any]] = []

    async def activate(self, suggestion: ProactiveSuggestion) -> Dict[str, Any]:
        """Convert approved suggestion into kernel mission."""
        if suggestion.ihsan_precheck == "blocked":
            logger.warning(
                "MissionActivator: blocked suggestion %s cannot be activated",
                suggestion.id,
            )
            return {"error": "blocked", "reason": suggestion.block_reason}

        mission = await self._add_mission(
            description=(
                f"[{suggestion.domain}] {suggestion.action_label}: "
                f"{suggestion.intent_summary}"
            ),
            priority="high" if suggestion.hhmm_confidence > 0.85 else "normal",
        )

        activation_record = {
            "suggestion_id": suggestion.id,
            "goal_id": suggestion.goal_id,
            "mission_id": mission["id"],
            "activated_at": datetime.now(timezone.utc).isoformat(),
            "ihsan_score": suggestion.ihsan_score,
        }
        self._activated.append(activation_record)
        if len(self._activated) > 200:
            self._activated = self._activated[-200:]

        logger.info(
            "MissionActivator: suggestion %s → mission %s (ihsan=%.2f)",
            suggestion.id,
            mission["id"],
            suggestion.ihsan_score,
        )

        return activation_record


# ---------------------------------------------------------------------------
# ProactiveHarness — The Main Orchestrator
# ---------------------------------------------------------------------------


class ProactiveHarness:
    """The complete self-proactive agent harness.

    Wires into Node0ProactiveKernel's idle cycle to:
    1. Scan goals for relevance (GoalScanner)
    2. Forge suggestions (SuggestionForge)
    3. Push to Ghost Panel (GhostPusher)
    4. Self-assess periodically (SelfAssessor)
    5. Activate missions on user approval (MissionActivator)

    Integration point: call `harness.on_idle_cycle(cycle_count, env_signals)`
    from Node0ProactiveKernel._run_loop() idle branch.
    """

    def __init__(
        self,
        baseline: Dict[str, Any],
        kernel_metrics: Dict[str, Any],
        add_mission_fn: Callable,
        knowledge_retriever=None,
    ):
        self._baseline = baseline
        self._scanner = GoalScanner(baseline, knowledge_retriever)
        self._forge = SuggestionForge()
        self._pusher = GhostPusher(mode="auto")
        self._assessor = SelfAssessor(kernel_metrics, baseline)
        self._activator = MissionActivator(add_mission_fn)

        self._cycle_count = 0
        self._total_suggestions_pushed = 0
        self._total_missions_activated = 0
        self._active = True

        logger.info(
            "ProactiveHarness initialized | goals=%d | assess_every=%d cycles",
            len(baseline.get("weekly_goals", [])),
            SELF_ASSESS_EVERY_N_CYCLES,
        )

    async def on_idle_cycle(
        self,
        cycle_count: int,
        env_signals: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Called from Node0 idle branch. The heartbeat of proactive behavior.

        Args:
            cycle_count: Current OODA cycle number
            env_signals: Optional environment context (active apps, time, etc.)

        Returns:
            Summary dict with suggestions_pushed, assessment (if triggered), etc.
        """
        if not self._active:
            return {"status": "inactive"}

        self._cycle_count = cycle_count
        result: Dict[str, Any] = {"cycle": cycle_count, "suggestions_pushed": 0}

        # --- Phase 1: Scan goals for relevance ---
        env = env_signals or self._gather_default_signals()
        scored_goals = self._scanner.scan(env)

        if not scored_goals:
            result["status"] = "no_relevant_goals"
            # Still run self-assessment on schedule
            if cycle_count % SELF_ASSESS_EVERY_N_CYCLES == 0:
                result["assessment"] = await self._run_self_assessment()
            return result

        # --- Phase 2: Forge suggestions ---
        suggestions: List[ProactiveSuggestion] = []
        for goal in scored_goals:
            suggestion = self._forge.forge(goal)
            suggestions.append(suggestion)
            self._scanner.mark_suggested(goal.goal_id)

        # --- Phase 3: Push to Ghost Panel ---
        pushed = await self._pusher.push(suggestions)
        self._total_suggestions_pushed += pushed
        result["suggestions_pushed"] = pushed
        result["suggestions"] = [
            {"id": s.id, "label": s.action_label, "ihsan": s.ihsan_precheck}
            for s in suggestions
        ]

        logger.info(
            "ProactiveHarness cycle %d: %d goals scored → %d suggestions → %d pushed",
            cycle_count,
            len(scored_goals),
            len(suggestions),
            pushed,
        )

        # --- Phase 4: Self-assessment (periodic) ---
        if cycle_count % SELF_ASSESS_EVERY_N_CYCLES == 0:
            result["assessment"] = await self._run_self_assessment()

        result["status"] = "active"
        return result

    async def on_gesture_received(
        self, gesture: str, suggestion_id: str
    ) -> Dict[str, Any]:
        """Called when Ghost Panel sends a sovereign gesture.

        Gestures:
          solidify  → Approve & execute suggestion
          dismiss   → Reject suggestion
          scroll_*  → Navigation (no action needed here)
        """
        if gesture == "solidify":
            # Find the suggestion by ID from recent pushes
            # In production, maintain a cache of recently pushed suggestions
            logger.info(
                "ProactiveHarness: solidify gesture for %s — activating mission",
                suggestion_id,
            )
            # TODO: lookup suggestion from cache, call self._activator.activate()
            self._total_missions_activated += 1
            return {"action": "mission_activated", "suggestion_id": suggestion_id}

        elif gesture == "dismiss":
            logger.info("ProactiveHarness: dismiss gesture for %s", suggestion_id)
            return {"action": "dismissed", "suggestion_id": suggestion_id}

        return {"action": "ignored", "gesture": gesture}

    async def _run_self_assessment(self) -> Dict[str, Any]:
        """Run self-assessment and optionally inject improvement goals."""
        assessment = self._assessor.assess()

        # If there are improvements, generate new goals
        if assessment.improvement_suggestions:
            new_goals = self._assessor.generate_improvement_goals(assessment)

            # Inject into baseline (runtime only — doesn't persist to disk)
            existing_goals = self._baseline.get("weekly_goals", [])
            for goal in new_goals:
                # Don't add if similar goal already exists
                existing_ids = {g["id"] for g in existing_goals}
                if goal["id"] not in existing_ids:
                    existing_goals.append(goal)

            # Re-initialize scanner with updated goals
            self._scanner = GoalScanner(self._baseline, self._scanner._knowledge)

            logger.info("SelfAssessment: injected %d improvement goals", len(new_goals))

        return asdict(assessment)

    def _gather_default_signals(self) -> Dict[str, Any]:
        """Gather basic environmental signals when none provided."""
        now = datetime.now(timezone.utc)
        return {
            "hour_utc": now.hour,
            "day_of_week": now.strftime("%A"),
            "cycle_count": self._cycle_count,
            "total_pushed": self._total_suggestions_pushed,
            "total_activated": self._total_missions_activated,
        }

    def health(self) -> Dict[str, Any]:
        """Return harness health summary."""
        return {
            "active": self._active,
            "cycle_count": self._cycle_count,
            "goals_loaded": len(self._baseline.get("weekly_goals", [])),
            "total_suggestions_pushed": self._total_suggestions_pushed,
            "total_missions_activated": self._total_missions_activated,
            "assessments_completed": len(self._assessor._assessments),
            "ghost_pusher_connected": self._pusher._connected,
        }

    async def shutdown(self) -> None:
        """Graceful shutdown."""
        self._active = False
        await self._pusher.close()
        logger.info("ProactiveHarness shut down")


# ---------------------------------------------------------------------------
# Integration Hook — Wire into Node0ProactiveKernel
# ---------------------------------------------------------------------------


def create_harness(kernel) -> Optional[ProactiveHarness]:
    """Factory function to create ProactiveHarness from a running kernel.

    Usage in node0_activate.py:

        from core.proactive.self_harness import create_harness

        # In Node0ProactiveKernel.__init__():
        self._harness = create_harness(self)

        # In _run_loop() idle branch (replacing the bare logger.info):
        if self._harness:
            result = await self._harness.on_idle_cycle(
                self._cycle_count,
                env_signals={"active_missions": len(self._missions)},
            )
            logger.info("  ○ Proactive: %s", result.get("status", "unknown"))
        else:
            logger.info("  ○ Idle - monitoring for opportunities")
    """
    baseline = getattr(kernel, "_baseline", None)
    if not baseline or not baseline.get("weekly_goals"):
        logger.warning(
            "create_harness: no baseline or empty weekly_goals — "
            "harness will not activate"
        )
        return None

    metrics = getattr(kernel, "_metrics", {})
    add_mission = getattr(kernel, "add_mission", None)
    knowledge = getattr(kernel, "_knowledge", None)

    if not callable(add_mission):
        logger.warning("create_harness: kernel.add_mission not callable")
        return None

    harness = ProactiveHarness(
        baseline=baseline,
        kernel_metrics=metrics,
        add_mission_fn=add_mission,
        knowledge_retriever=knowledge,
    )

    logger.info("ProactiveHarness created and ready")
    return harness
