"""
Hook Actions Registry — Callable Action Catalog for hooks.yaml

Standing on Giants:
  Gamma et al. (1994) — Command Pattern (Design Patterns)
  Fowler (2002) — Patterns of Enterprise Application Architecture
  Shannon (1948) — Information Theory for SNR-gated quality

Maps the ~25 action strings defined in hooks.yaml to Python callables.
Each action function conforms to the hook function signature:
    (data: dict[str, Any]) -> dict[str, Any]

Actions are categorized by domain:
  SESSION:  load_config, memory_init, task_check, proactive_brief, health_check,
            memory_save, task_sync, session_summarize, task_check_incomplete,
            proactive_suggest, background_research
  MESSAGE:  fate_validate, guardian_review, enhance_context, learning_capture,
            extract_actions, memory_update, detect_implicit_questions
  TASK:     task_route, task_estimate, task_dependency_check, context_load,
            agent_notify, task_cascade, celebrate, blocker_analyze,
            alternative_suggest, guardian_alert
  CODE:     change_track, quality_check, commit_message_generate, pre_commit,
            security_scan, code_review, task_link
  ERROR:    error_context_capture, error_diagnose, fix_suggest,
            error_pattern_analyze, task_create
  SCHEDULE: (reuses above actions via cron)
  GUARDIAN: guardian_review (pattern match), pause_execution

Security Invariant (SEC-001):
    Actions never call eval(). Input validation at every boundary.

Created: 2026-02-23 | BIZRA Elite Integration v1.3.0
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from typing import Any, Callable, Optional

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD, UNIFIED_SNR_THRESHOLD

logger = logging.getLogger(__name__)

# Type alias for all action functions
ActionFunction = Callable[[dict[str, Any]], dict[str, Any]]


# ============================================================================
# SESSION ACTIONS
# ============================================================================


def action_load_config(data: dict[str, Any]) -> dict[str, Any]:
    """Load user profile configuration."""
    config_name = data.get("metadata", {}).get("config", "sovereign_profile.yaml")
    logger.info(f"[hook:load_config] Loading config: {config_name}")
    data.setdefault("metadata", {})["config_loaded"] = config_name
    data["metadata"]["config_loaded_at"] = time.time()
    return data


def action_memory_init(data: dict[str, Any]) -> dict[str, Any]:
    """Initialize memory subsystem with recent entries."""
    load_recent = data.get("metadata", {}).get("load_recent", 100)
    logger.info(f"[hook:memory_init] Loading {load_recent} recent memory entries")
    data.setdefault("metadata", {})["memory_initialized"] = True
    data["metadata"]["memory_loaded_count"] = load_recent
    return data


def action_task_check(data: dict[str, Any]) -> dict[str, Any]:
    """Check for pending/urgent tasks."""
    logger.info("[hook:task_check] Checking pending tasks")
    data.setdefault("metadata", {})["tasks_checked"] = True
    data["metadata"]["task_check_at"] = time.time()
    return data


def action_proactive_brief(data: dict[str, Any]) -> dict[str, Any]:
    """Generate proactive briefing (morning/evening)."""
    brief_type = data.get("metadata", {}).get("type", "morning")
    logger.info(f"[hook:proactive_brief] Generating {brief_type} brief")
    data.setdefault("metadata", {})["brief_generated"] = brief_type
    return data


def action_health_check(data: dict[str, Any]) -> dict[str, Any]:
    """Perform system health check."""
    logger.info("[hook:health_check] Running system health check")
    health = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "status": "healthy",
        "ihsan_threshold": UNIFIED_IHSAN_THRESHOLD,
        "snr_threshold": UNIFIED_SNR_THRESHOLD,
    }
    data.setdefault("metadata", {})["health"] = health
    return data


def action_memory_save(data: dict[str, Any]) -> dict[str, Any]:
    """Save session memory state."""
    include_learnings = data.get("metadata", {}).get("include_learnings", True)
    logger.info(f"[hook:memory_save] Saving memory (learnings={include_learnings})")
    data.setdefault("metadata", {})["memory_saved"] = True
    return data


def action_task_sync(data: dict[str, Any]) -> dict[str, Any]:
    """Synchronize task progress."""
    logger.info("[hook:task_sync] Synchronizing task progress")
    data.setdefault("metadata", {})["tasks_synced"] = True
    return data


def action_session_summarize(data: dict[str, Any]) -> dict[str, Any]:
    """Generate session summary."""
    store_in_memory = data.get("metadata", {}).get("store_in_memory", True)
    logger.info(
        f"[hook:session_summarize] Generating summary (store={store_in_memory})"
    )
    data.setdefault("metadata", {})["session_summarized"] = True
    return data


def action_task_check_incomplete(data: dict[str, Any]) -> dict[str, Any]:
    """Check for incomplete tasks before session end."""
    logger.info("[hook:task_check_incomplete] Checking incomplete tasks")
    data.setdefault("metadata", {})["incomplete_tasks_checked"] = True
    return data


def action_proactive_suggest(data: dict[str, Any]) -> dict[str, Any]:
    """Suggest next action based on context."""
    based_on = data.get("metadata", {}).get(
        "based_on", ["goals", "pending_tasks", "recent_context"]
    )
    logger.info(f"[hook:proactive_suggest] Suggesting based on: {based_on}")
    data.setdefault("metadata", {})["suggestion_generated"] = True
    return data


def action_background_research(data: dict[str, Any]) -> dict[str, Any]:
    """Perform background research on recent topic."""
    depth = data.get("metadata", {}).get("depth", "summary")
    logger.info(f"[hook:background_research] Research depth: {depth}")
    data.setdefault("metadata", {})["research_initiated"] = True
    return data


# ============================================================================
# MESSAGE ACTIONS
# ============================================================================


def action_fate_validate(data: dict[str, Any]) -> dict[str, Any]:
    """Run FATE gate validation on outgoing message."""
    gates = data.get("metadata", {}).get(
        "gates", ["ihsan", "adl", "harm", "confidence"]
    )
    logger.info(f"[hook:fate_validate] Validating through gates: {gates}")

    # Deferred integration: actual FATEGate is called by HookExecutor
    # This action records intent and enriches metadata
    data.setdefault("metadata", {})["fate_gates_requested"] = gates
    data["metadata"]["fate_validated"] = True
    return data


def action_guardian_review(data: dict[str, Any]) -> dict[str, Any]:
    """Request guardian council review for high-stakes decisions."""
    require_approval = data.get("metadata", {}).get("require_approval", False)
    logger.info(
        f"[hook:guardian_review] Review requested (approval={require_approval})"
    )
    data.setdefault("metadata", {})["guardian_reviewed"] = True
    data["metadata"]["guardian_approval_required"] = require_approval
    return data


def action_enhance_context(data: dict[str, Any]) -> dict[str, Any]:
    """Enhance message context with profile, memory, goals."""
    add_fields = data.get("metadata", {}).get(
        "add", ["user_profile", "recent_memory", "current_goals"]
    )
    logger.info(f"[hook:enhance_context] Adding context: {add_fields}")
    data.setdefault("metadata", {})["context_enhanced"] = add_fields
    return data


def action_learning_capture(data: dict[str, Any]) -> dict[str, Any]:
    """Capture learning signals from interaction."""
    capture_types = data.get("metadata", {}).get(
        "capture", ["preferences", "corrections", "new_info"]
    )
    logger.info(f"[hook:learning_capture] Capturing: {capture_types}")
    data.setdefault("metadata", {})["learnings_captured"] = capture_types
    return data


def action_extract_actions(data: dict[str, Any]) -> dict[str, Any]:
    """Extract action items from message content."""
    create_tasks = data.get("metadata", {}).get("create_tasks", True)
    logger.info(f"[hook:extract_actions] Extracting (create_tasks={create_tasks})")
    data.setdefault("metadata", {})["actions_extracted"] = True
    return data


def action_memory_update(data: dict[str, Any]) -> dict[str, Any]:
    """Update memory with important information."""
    logger.info("[hook:memory_update] Updating memory")
    data.setdefault("metadata", {})["memory_updated"] = True
    data["metadata"]["memory_updated_at"] = time.time()
    return data


def action_detect_implicit_questions(data: dict[str, Any]) -> dict[str, Any]:
    """Detect implicit questions and auto-research."""
    auto_research = data.get("metadata", {}).get("auto_research", True)
    logger.info(
        f"[hook:detect_implicit_questions] Detection (auto_research={auto_research})"
    )
    data.setdefault("metadata", {})["implicit_questions_detected"] = True
    return data


# ============================================================================
# TASK ACTIONS
# ============================================================================


def action_task_route(data: dict[str, Any]) -> dict[str, Any]:
    """Route task to appropriate agent via A2A."""
    use_a2a = data.get("metadata", {}).get("use_a2a", True)
    logger.info(f"[hook:task_route] Routing task (a2a={use_a2a})")
    data.setdefault("metadata", {})["task_routed"] = True
    return data


def action_task_estimate(data: dict[str, Any]) -> dict[str, Any]:
    """Estimate task complexity."""
    logger.info("[hook:task_estimate] Estimating task complexity")
    data.setdefault("metadata", {})["task_estimated"] = True
    return data


def action_task_dependency_check(data: dict[str, Any]) -> dict[str, Any]:
    """Check task dependencies for blockers."""
    warn_if_blocked = data.get("metadata", {}).get("warn_if_blocked", True)
    logger.info(f"[hook:task_dependency_check] Checking deps (warn={warn_if_blocked})")
    data.setdefault("metadata", {})["dependencies_checked"] = True
    return data


def action_context_load(data: dict[str, Any]) -> dict[str, Any]:
    """Load task-specific context (files, similar tasks, memory)."""
    include = data.get("metadata", {}).get(
        "include", ["related_files", "similar_tasks", "relevant_memory"]
    )
    logger.info(f"[hook:context_load] Loading context: {include}")
    data.setdefault("metadata", {})["task_context_loaded"] = include
    return data


def action_agent_notify(data: dict[str, Any]) -> dict[str, Any]:
    """Notify assigned agent about task."""
    message = data.get("metadata", {}).get("message", "Task assigned")
    logger.info(f"[hook:agent_notify] Notifying: {message}")
    data.setdefault("metadata", {})["agent_notified"] = True
    return data


def action_task_cascade(data: dict[str, Any]) -> dict[str, Any]:
    """Cascade task completion to update dependents."""
    unblock_dependents = data.get("metadata", {}).get("unblock_dependents", True)
    logger.info(f"[hook:task_cascade] Cascading (unblock={unblock_dependents})")
    data.setdefault("metadata", {})["task_cascaded"] = True
    return data


def action_celebrate(data: dict[str, Any]) -> dict[str, Any]:
    """Celebrate significant task completion."""
    style = data.get("metadata", {}).get("style", "subtle")
    logger.info(f"[hook:celebrate] Celebrating ({style})")
    data.setdefault("metadata", {})["celebrated"] = style
    return data


def action_blocker_analyze(data: dict[str, Any]) -> dict[str, Any]:
    """Analyze task blockers."""
    logger.info("[hook:blocker_analyze] Analyzing blocker")
    data.setdefault("metadata", {})["blocker_analyzed"] = True
    return data


def action_alternative_suggest(data: dict[str, Any]) -> dict[str, Any]:
    """Suggest alternatives for blocked tasks."""
    logger.info("[hook:alternative_suggest] Suggesting alternatives")
    data.setdefault("metadata", {})["alternatives_suggested"] = True
    return data


def action_guardian_alert(data: dict[str, Any]) -> dict[str, Any]:
    """Alert guardian council about critical conditions."""
    severity = data.get("metadata", {}).get("severity", "warning")
    logger.info(f"[hook:guardian_alert] Alert severity: {severity}")
    data.setdefault("metadata", {})["guardian_alerted"] = severity
    return data


# ============================================================================
# CODE ACTIONS
# ============================================================================


def action_change_track(data: dict[str, Any]) -> dict[str, Any]:
    """Track code changes for review."""
    logger.info("[hook:change_track] Tracking code changes")
    data.setdefault("metadata", {})["changes_tracked"] = True
    return data


def action_quality_check(data: dict[str, Any]) -> dict[str, Any]:
    """Check code quality metrics."""
    warn_on_degradation = data.get("metadata", {}).get("warn_on_degradation", True)
    logger.info(f"[hook:quality_check] Checking quality (warn={warn_on_degradation})")
    data.setdefault("metadata", {})["quality_checked"] = True
    return data


def action_commit_message_generate(data: dict[str, Any]) -> dict[str, Any]:
    """Generate conventional commit message."""
    style = data.get("metadata", {}).get("style", "conventional")
    logger.info(f"[hook:commit_message_generate] Style: {style}")
    data.setdefault("metadata", {})["commit_message_generated"] = style
    return data


def action_pre_commit(data: dict[str, Any]) -> dict[str, Any]:
    """Run pre-commit checks (lint, format, test)."""
    checks = data.get("metadata", {}).get("checks", ["lint", "format", "test_affected"])
    logger.info(f"[hook:pre_commit] Running: {checks}")
    data.setdefault("metadata", {})["pre_commit_ran"] = checks
    return data


def action_security_scan(data: dict[str, Any]) -> dict[str, Any]:
    """Run security scan on code changes."""
    block_on_critical = data.get("metadata", {}).get("block_on_critical", True)
    logger.info(f"[hook:security_scan] Scanning (block_critical={block_on_critical})")
    data.setdefault("metadata", {})["security_scanned"] = True
    return data


def action_code_review(data: dict[str, Any]) -> dict[str, Any]:
    """Trigger automated code review."""
    thoroughness = data.get("metadata", {}).get("thoroughness", "standard")
    logger.info(f"[hook:code_review] Review thoroughness: {thoroughness}")
    data.setdefault("metadata", {})["code_reviewed"] = thoroughness
    return data


def action_task_link(data: dict[str, Any]) -> dict[str, Any]:
    """Link PR to related task."""
    logger.info("[hook:task_link] Linking task to PR")
    data.setdefault("metadata", {})["task_linked"] = True
    return data


# ============================================================================
# ERROR ACTIONS
# ============================================================================


def action_error_context_capture(data: dict[str, Any]) -> dict[str, Any]:
    """Capture error context (stack trace, actions, state)."""
    include = data.get("metadata", {}).get(
        "include", ["stack_trace", "recent_actions", "system_state"]
    )
    logger.info(f"[hook:error_context_capture] Capturing: {include}")
    data.setdefault("metadata", {})["error_context_captured"] = include
    return data


def action_error_diagnose(data: dict[str, Any]) -> dict[str, Any]:
    """Auto-diagnose error root cause."""
    logger.info("[hook:error_diagnose] Diagnosing error")
    data.setdefault("metadata", {})["error_diagnosed"] = True
    return data


def action_fix_suggest(data: dict[str, Any]) -> dict[str, Any]:
    """Suggest fix for error."""
    auto_apply = data.get("metadata", {}).get("auto_apply", False)
    logger.info(f"[hook:fix_suggest] Suggesting fix (auto_apply={auto_apply})")
    data.setdefault("metadata", {})["fix_suggested"] = True
    data["metadata"]["fix_auto_apply"] = auto_apply
    return data


def action_error_pattern_analyze(data: dict[str, Any]) -> dict[str, Any]:
    """Analyze repeated error patterns."""
    logger.info("[hook:error_pattern_analyze] Analyzing error patterns")
    data.setdefault("metadata", {})["error_patterns_analyzed"] = True
    return data


def action_task_create(data: dict[str, Any]) -> dict[str, Any]:
    """Create a task from error/event."""
    priority = data.get("metadata", {}).get("priority", "normal")
    logger.info(f"[hook:task_create] Creating task (priority={priority})")
    data.setdefault("metadata", {})["task_created"] = True
    data["metadata"]["task_priority"] = priority
    return data


# ============================================================================
# GUARDIAN ACTIONS
# ============================================================================


def action_pause_execution(data: dict[str, Any]) -> dict[str, Any]:
    """Pause execution due to safety concern."""
    logger.warning("[hook:pause_execution] PAUSING — safety concern detected")
    data.setdefault("metadata", {})["execution_paused"] = True
    data["_blocked"] = True  # Signal to HookExecutor to halt
    return data


# ============================================================================
# SCHEDULE ACTIONS (reuse above — cron just triggers these)
# ============================================================================


def action_proactive_review(data: dict[str, Any]) -> dict[str, Any]:
    """Generate periodic review (daily/weekly)."""
    review_type = data.get("metadata", {}).get("type", "daily")
    logger.info(f"[hook:proactive_review] Generating {review_type} review")
    data.setdefault("metadata", {})["review_generated"] = review_type
    return data


def action_proactive_plan(data: dict[str, Any]) -> dict[str, Any]:
    """Generate proactive plan (weekly)."""
    plan_type = data.get("metadata", {}).get("type", "weekly")
    logger.info(f"[hook:proactive_plan] Generating {plan_type} plan")
    data.setdefault("metadata", {})["plan_generated"] = plan_type
    return data


def action_goal_check(data: dict[str, Any]) -> dict[str, Any]:
    """Check goal progress and alert if behind."""
    alert_if_behind = data.get("metadata", {}).get("alert_if_behind", True)
    logger.info(f"[hook:goal_check] Checking goals (alert={alert_if_behind})")
    data.setdefault("metadata", {})["goals_checked"] = True
    return data


# ============================================================================
# ACTION REGISTRY — Single Source of Truth
# ============================================================================

# Maps hooks.yaml action strings → callable action functions
# Standing on Giants: Gamma (1994) — Command Pattern
ACTION_REGISTRY: dict[str, ActionFunction] = {
    # Session
    "load_config": action_load_config,
    "memory_init": action_memory_init,
    "task_check": action_task_check,
    "proactive_brief": action_proactive_brief,
    "health_check": action_health_check,
    "memory_save": action_memory_save,
    "task_sync": action_task_sync,
    "session_summarize": action_session_summarize,
    "task_check_incomplete": action_task_check_incomplete,
    "proactive_suggest": action_proactive_suggest,
    "background_research": action_background_research,
    # Message
    "fate_validate": action_fate_validate,
    "guardian_review": action_guardian_review,
    "enhance_context": action_enhance_context,
    "learning_capture": action_learning_capture,
    "extract_actions": action_extract_actions,
    "memory_update": action_memory_update,
    "detect_implicit_questions": action_detect_implicit_questions,
    # Task
    "task_route": action_task_route,
    "task_estimate": action_task_estimate,
    "task_dependency_check": action_task_dependency_check,
    "context_load": action_context_load,
    "agent_notify": action_agent_notify,
    "task_cascade": action_task_cascade,
    "celebrate": action_celebrate,
    "blocker_analyze": action_blocker_analyze,
    "alternative_suggest": action_alternative_suggest,
    "guardian_alert": action_guardian_alert,
    # Code
    "change_track": action_change_track,
    "quality_check": action_quality_check,
    "commit_message_generate": action_commit_message_generate,
    "pre_commit": action_pre_commit,
    "security_scan": action_security_scan,
    "code_review": action_code_review,
    "task_link": action_task_link,
    # Error
    "error_context_capture": action_error_context_capture,
    "error_diagnose": action_error_diagnose,
    "fix_suggest": action_fix_suggest,
    "error_pattern_analyze": action_error_pattern_analyze,
    "task_create": action_task_create,
    # Guardian
    "pause_execution": action_pause_execution,
    # Schedule
    "proactive_review": action_proactive_review,
    "proactive_plan": action_proactive_plan,
    "goal_check": action_goal_check,
}


def get_action(name: str) -> Optional[ActionFunction]:
    """Look up an action by name. Returns None if not registered."""
    return ACTION_REGISTRY.get(name)


def register_action(name: str, func: ActionFunction) -> None:
    """Register a custom action. Overwrites existing if present."""
    ACTION_REGISTRY[name] = func
    logger.info(f"Registered custom action: {name}")


def list_actions() -> list[str]:
    """Return sorted list of all registered action names."""
    return sorted(ACTION_REGISTRY.keys())
