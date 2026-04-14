"""
DEMA Router — Directive Execution & Mission Assignment
=======================================================
Routes human input to the best PAT agent based on intent,
persona context, and agent capabilities.

Standing on Giants: CQRS (Meyer) + Intent Classification (NLU)
Constitutional Constraint: Ihsan >= 0.95
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class DEMARouter:
    """
    Routes user directives to PAT agents.

    Combines:
      1. Keyword-based role matching (via UserContextManager.select_pat_agent)
      2. Priority weighting (urgent → high priority)
      3. DEMA persona assignment (strategic vs tactical)

    Usage:
        router = DEMARouter(
            pat_runtime=pat_runtime,
            select_agent_fn=user_context.select_pat_agent,
        )
        result = router.route("Analyze the quarterly revenue trends")
    """

    # Priority keywords
    URGENT_KEYWORDS = frozenset({
        "urgent", "asap", "immediately", "critical", "emergency",
        "now", "right away", "blocker",
    })
    HIGH_KEYWORDS = frozenset({
        "important", "priority", "soon", "deadline", "review",
    })

    def __init__(
        self,
        pat_runtime: Optional[Any] = None,
        select_agent_fn: Optional[Any] = None,
        pat_team: Optional[List[Any]] = None,
    ):
        self._pat_runtime = pat_runtime
        self._select_agent_fn = select_agent_fn
        self._pat_team = pat_team or []

        # Metrics
        self._routes_count = 0
        self._routes_by_role: Dict[str, int] = {}

    def _classify_priority(self, query: str) -> int:
        """Classify mission priority from query text (1=highest, 10=lowest)."""
        lower = query.lower()
        words = set(lower.split())
        if words & self.URGENT_KEYWORDS:
            return 1
        if words & self.HIGH_KEYWORDS:
            return 3
        return 5

    def _select_role(self, query: str) -> Optional[str]:
        """Select the best PAT role for this query."""
        if self._select_agent_fn:
            try:
                return self._select_agent_fn(query, self._pat_team)
            except (TypeError, ValueError, RuntimeError):
                pass

        # Inline fallback — same keyword map as user_context.select_pat_agent
        lower = query.lower()
        role_keywords = {
            "strategist": ["plan", "strategy", "roadmap", "vision", "goal", "direction"],
            "researcher": ["research", "investigate", "study", "analyze", "find", "explore"],
            "developer": ["code", "build", "implement", "fix", "debug", "develop", "create"],
            "analyst": ["data", "metrics", "report", "trend", "revenue", "performance"],
            "reviewer": ["review", "audit", "check", "verify", "assess", "evaluate"],
            "executor": ["run", "execute", "deploy", "launch", "start", "activate"],
            "guardian": ["security", "protect", "guard", "monitor", "alert", "safety"],
        }
        for role, keywords in role_keywords.items():
            if any(kw in lower for kw in keywords):
                return role
        return None

    def route(
        self,
        query: str,
        requester_id: str = "",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Route a user directive to the appropriate PAT agent.

        Returns a routing decision dict that can be used by the caller
        to submit to PAT runtime.
        """
        mission_id = f"dema-{uuid.uuid4().hex[:12]}"
        priority = self._classify_priority(query)
        role = self._select_role(query)

        self._routes_count += 1
        if role:
            self._routes_by_role[role] = self._routes_by_role.get(role, 0) + 1

        routing = {
            "mission_id": mission_id,
            "content": query,
            "target_role": role,
            "priority": priority,
            "requester_id": requester_id,
            "metadata": metadata or {},
            "routed_at": datetime.now(timezone.utc).isoformat(),
        }

        logger.debug(
            f"DEMA route: {mission_id} → {role or 'round-robin'} (P{priority})"
        )

        # Auto-submit to PAT runtime if available
        if self._pat_runtime and hasattr(self._pat_runtime, "submit_fire_and_forget"):
            try:
                from core.pat.runtime import MissionRequest

                request = MissionRequest(
                    mission_id=mission_id,
                    content=query,
                    requester_id=requester_id,
                    target_role=role,
                    priority=priority,
                    metadata=metadata or {},
                )
                self._pat_runtime.submit_fire_and_forget(request)
                routing["submitted"] = True
            except (ImportError, RuntimeError, TypeError) as e:
                routing["submitted"] = False
                routing["submit_error"] = str(e)
        else:
            routing["submitted"] = False

        return routing

    def get_status(self) -> Dict[str, Any]:
        return {
            "routes_total": self._routes_count,
            "routes_by_role": dict(self._routes_by_role),
            "pat_runtime_connected": self._pat_runtime is not None,
        }
