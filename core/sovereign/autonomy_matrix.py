"""
Autonomy Matrix — 5-Level Autonomy Framework
============================================
Defines levels of autonomous operation from Observer (human-in-loop)
to Sovereign (full agency). Each level has constitutional guardrails.

Standing on Giants: SAE Levels of Automation + Constitutional AI + Risk Management
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class AutonomyLevel(IntEnum):
    """
    5 Levels of Autonomous Operation.

    Inspired by SAE driving automation levels,
    adapted for AI sovereign operation.
    """
    OBSERVER = 0      # Watch only, no action
    SUGGESTER = 1     # Suggest actions, require approval
    AUTOLOW = 2       # Execute low-risk, notify after
    AUTOMEDIUM = 3    # Execute medium-risk, notify before
    AUTOHIGH = 4      # Execute high-risk, require pre-approval
    SOVEREIGN = 5     # Full agency (emergencies only)


@dataclass
class AutonomyConstraints:
    """Constraints for an autonomy level."""
    max_cost_percent: float = 0.0    # Max % of resources to use
    max_risk_score: float = 0.0      # Max risk (0-1)
    min_ihsan_score: float = 1.0     # Min quality threshold
    reversible_only: bool = True     # Must be reversible
    require_approval: bool = True    # Needs human approval
    notify_before: bool = True       # Notify before action
    notify_after: bool = True        # Notify after action


# Default constraints per autonomy level
DEFAULT_CONSTRAINTS: Dict[AutonomyLevel, AutonomyConstraints] = {
    AutonomyLevel.OBSERVER: AutonomyConstraints(
        max_cost_percent=0.0,
        max_risk_score=0.0,
        min_ihsan_score=1.0,
        reversible_only=True,
        require_approval=True,
        notify_before=True,
        notify_after=True,
    ),
    AutonomyLevel.SUGGESTER: AutonomyConstraints(
        max_cost_percent=0.0,
        max_risk_score=0.2,
        min_ihsan_score=0.95,
        reversible_only=True,
        require_approval=True,
        notify_before=True,
        notify_after=True,
    ),
    AutonomyLevel.AUTOLOW: AutonomyConstraints(
        max_cost_percent=1.0,     # 1% of resources
        max_risk_score=0.3,
        min_ihsan_score=0.97,
        reversible_only=True,
        require_approval=False,
        notify_before=False,
        notify_after=True,
    ),
    AutonomyLevel.AUTOMEDIUM: AutonomyConstraints(
        max_cost_percent=5.0,     # 5% of resources
        max_risk_score=0.5,
        min_ihsan_score=0.98,
        reversible_only=False,
        require_approval=False,
        notify_before=True,
        notify_after=True,
    ),
    AutonomyLevel.AUTOHIGH: AutonomyConstraints(
        max_cost_percent=10.0,    # 10% of resources
        max_risk_score=0.7,
        min_ihsan_score=0.99,
        reversible_only=False,
        require_approval=True,   # Pre-approval required
        notify_before=True,
        notify_after=True,
    ),
    AutonomyLevel.SOVEREIGN: AutonomyConstraints(
        max_cost_percent=100.0,   # Full authority
        max_risk_score=1.0,
        min_ihsan_score=1.0,      # Must be perfect
        reversible_only=False,
        require_approval=False,   # Emergency authority
        notify_before=False,      # No time
        notify_after=True,
    ),
}


@dataclass
class ActionContext:
    """Context for evaluating an action's autonomy requirements."""
    action_type: str = ""
    description: str = ""
    cost_percent: float = 0.0       # Estimated cost as % of resources
    risk_score: float = 0.0         # Risk score 0-1
    ihsan_score: float = 0.95       # Quality/ethics score
    is_reversible: bool = True
    is_emergency: bool = False
    domain: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AutonomyDecision:
    """Decision about autonomy for an action."""
    action_type: str = ""
    determined_level: AutonomyLevel = AutonomyLevel.OBSERVER
    can_execute: bool = False
    requires_approval: bool = True
    notify_before: bool = True
    notify_after: bool = True
    constraints_met: bool = False
    violated_constraints: List[str] = field(default_factory=list)
    reasoning: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class AutonomyMatrix:
    """
    5-Level Autonomy Framework with Constitutional Guardrails.

    Determines what level of autonomy is appropriate for each action
    and enforces constraints at each level.
    """

    def __init__(
        self,
        default_level: AutonomyLevel = AutonomyLevel.SUGGESTER,
        ihsan_threshold: float = 0.95,
    ):
        self.default_level = default_level
        self.ihsan_threshold = ihsan_threshold

        # Configurable level constraints
        self._constraints = DEFAULT_CONSTRAINTS.copy()

        # Per-action-type level overrides
        self._action_levels: Dict[str, AutonomyLevel] = {}

        # User overrides (temporary elevations/restrictions)
        self._user_overrides: Dict[str, AutonomyLevel] = {}

        # Stats
        self._decisions: List[AutonomyDecision] = []
        self._decision_count = 0

    def set_constraint(
        self,
        level: AutonomyLevel,
        constraint_name: str,
        value: Any,
    ) -> None:
        """Modify a constraint for an autonomy level."""
        if level in self._constraints:
            if hasattr(self._constraints[level], constraint_name):
                setattr(self._constraints[level], constraint_name, value)

    def set_action_level(
        self,
        action_type: str,
        level: AutonomyLevel,
    ) -> None:
        """Set default autonomy level for an action type."""
        self._action_levels[action_type] = level

    def user_override(
        self,
        action_type: str,
        level: AutonomyLevel,
    ) -> None:
        """User override for an action type's autonomy level."""
        self._user_overrides[action_type] = level
        logger.info(f"User override: {action_type} -> {level.name}")

    def clear_override(self, action_type: str) -> None:
        """Clear user override."""
        if action_type in self._user_overrides:
            del self._user_overrides[action_type]

    def determine_autonomy(self, context: ActionContext) -> AutonomyDecision:
        """
        Determine appropriate autonomy level for an action.

        Algorithm:
        1. Check for emergency -> SOVEREIGN
        2. Check user overrides
        3. Check action-type defaults
        4. Assess risk/cost to determine level
        5. Verify constraints are satisfied
        """
        decision = AutonomyDecision(action_type=context.action_type)

        # Emergency override
        if context.is_emergency:
            decision.determined_level = AutonomyLevel.SOVEREIGN
            decision.reasoning = "Emergency conditions detected"
            return self._finalize_decision(decision, context)

        # Check user override
        if context.action_type in self._user_overrides:
            decision.determined_level = self._user_overrides[context.action_type]
            decision.reasoning = "User override applied"
            return self._finalize_decision(decision, context)

        # Check action-type default
        if context.action_type in self._action_levels:
            decision.determined_level = self._action_levels[context.action_type]
            decision.reasoning = "Action-type default"
            return self._finalize_decision(decision, context)

        # Determine by risk/cost assessment
        level = self._assess_level(context)
        decision.determined_level = level
        decision.reasoning = f"Assessed from risk={context.risk_score:.2f}, cost={context.cost_percent:.1f}%"

        return self._finalize_decision(decision, context)

    def _assess_level(self, context: ActionContext) -> AutonomyLevel:
        """Assess appropriate autonomy level based on context."""
        # Start from default and adjust
        level = self.default_level

        # Low risk, low cost -> can be AUTOLOW
        if context.risk_score <= 0.3 and context.cost_percent <= 1.0:
            if context.is_reversible and context.ihsan_score >= 0.97:
                level = AutonomyLevel.AUTOLOW

        # Medium risk -> AUTOMEDIUM or SUGGESTER
        elif context.risk_score <= 0.5 and context.cost_percent <= 5.0:
            if context.ihsan_score >= 0.98:
                level = AutonomyLevel.AUTOMEDIUM
            else:
                level = AutonomyLevel.SUGGESTER

        # Higher risk -> AUTOHIGH or SUGGESTER
        elif context.risk_score <= 0.7:
            if context.ihsan_score >= 0.99:
                level = AutonomyLevel.AUTOHIGH
            else:
                level = AutonomyLevel.SUGGESTER

        # Very high risk -> OBSERVER only
        else:
            level = AutonomyLevel.OBSERVER

        return level

    def _finalize_decision(
        self,
        decision: AutonomyDecision,
        context: ActionContext,
    ) -> AutonomyDecision:
        """Finalize decision by checking constraints."""
        constraints = self._constraints[decision.determined_level]
        violations = []

        # Check each constraint
        if context.cost_percent > constraints.max_cost_percent:
            violations.append(f"cost {context.cost_percent:.1f}% > {constraints.max_cost_percent:.1f}%")

        if context.risk_score > constraints.max_risk_score:
            violations.append(f"risk {context.risk_score:.2f} > {constraints.max_risk_score:.2f}")

        if context.ihsan_score < constraints.min_ihsan_score:
            violations.append(f"ihsan {context.ihsan_score:.2f} < {constraints.min_ihsan_score:.2f}")

        if constraints.reversible_only and not context.is_reversible:
            violations.append("action is not reversible")

        # Apply constraint results
        decision.violated_constraints = violations
        decision.constraints_met = len(violations) == 0

        # Set execution flags
        if decision.constraints_met:
            decision.can_execute = not constraints.require_approval
            decision.requires_approval = constraints.require_approval
            decision.notify_before = constraints.notify_before
            decision.notify_after = constraints.notify_after
        else:
            # Constraints violated - downgrade to SUGGESTER
            decision.can_execute = False
            decision.requires_approval = True
            decision.reasoning += f" (downgraded due to: {', '.join(violations)})"

        # Record decision
        self._decisions.append(decision)
        self._decision_count += 1

        # Trim history
        if len(self._decisions) > 1000:
            self._decisions = self._decisions[-500:]

        return decision

    def check_constraints(
        self,
        context: ActionContext,
        level: AutonomyLevel,
    ) -> bool:
        """Check if context satisfies constraints for a level."""
        constraints = self._constraints[level]

        if context.cost_percent > constraints.max_cost_percent:
            return False
        if context.risk_score > constraints.max_risk_score:
            return False
        if context.ihsan_score < constraints.min_ihsan_score:
            return False
        if constraints.reversible_only and not context.is_reversible:
            return False

        return True

    def get_level_description(self, level: AutonomyLevel) -> str:
        """Get human-readable description of autonomy level."""
        descriptions = {
            AutonomyLevel.OBSERVER: "Watch only - no autonomous action",
            AutonomyLevel.SUGGESTER: "Suggest actions - all require approval",
            AutonomyLevel.AUTOLOW: "Auto-execute low-risk - notify after",
            AutonomyLevel.AUTOMEDIUM: "Auto-execute medium-risk - notify before",
            AutonomyLevel.AUTOHIGH: "Auto-execute high-risk - pre-approval required",
            AutonomyLevel.SOVEREIGN: "Full autonomy - emergency authority only",
        }
        return descriptions.get(level, "Unknown")

    def stats(self) -> Dict[str, Any]:
        """Get autonomy matrix statistics."""
        level_counts = {level.name: 0 for level in AutonomyLevel}
        for d in self._decisions:
            level_counts[d.determined_level.name] += 1

        return {
            "total_decisions": self._decision_count,
            "decisions_by_level": level_counts,
            "action_type_overrides": len(self._action_levels),
            "user_overrides": len(self._user_overrides),
            "default_level": self.default_level.name,
        }


__all__ = [
    "ActionContext",
    "AutonomyConstraints",
    "AutonomyDecision",
    "AutonomyLevel",
    "AutonomyMatrix",
]
