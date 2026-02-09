"""
Background Agents — Domain-Specific Proactive Plugins
======================================================
Extensible background agent system for domain-specific
proactive optimization (calendar, email, files, etc.)

Integrates with:
- ProactiveSovereignEntity (orchestration)
- MuraqabahEngine (monitoring)
- AutonomyMatrix (permission control)
- ProactiveScheduler (execution timing)

Standing on Giants: Agent-Based Systems + Background Processing
"""

import asyncio
import hashlib
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from .autonomy_matrix import ActionContext, AutonomyLevel, AutonomyMatrix
from .event_bus import EventPriority, get_event_bus
from .proactive_scheduler import (
    ProactiveScheduler,
)

logger = logging.getLogger(__name__)


class AgentState(str, Enum):
    """Background agent execution states."""

    IDLE = "idle"
    RUNNING = "running"
    PAUSED = "paused"
    DISABLED = "disabled"
    ERROR = "error"


class ActionType(str, Enum):
    """Types of proactive actions."""

    CALENDAR_OPTIMIZE = "calendar_optimize"
    EMAIL_TRIAGE = "email_triage"
    FILE_ORGANIZE = "file_organize"
    SYSTEM_HEALTH = "system_health"
    FINANCIAL_MONITOR = "financial_monitor"
    HEALTH_REMIND = "health_remind"
    RELATIONSHIP_MANAGE = "relationship_manage"
    CUSTOM = "custom"


class ApprovalStatus(str, Enum):
    """Action approval status."""

    PENDING = "pending"
    APPROVED = "approved"
    REJECTED = "rejected"
    EXPIRED = "expired"
    AUTO_APPROVED = "auto_approved"


class ExecutionStatus(str, Enum):
    """Action execution status."""

    PLANNED = "planned"
    EXECUTING = "executing"
    SUCCESS = "success"
    FAILED = "failed"
    CANCELLED = "cancelled"


class Reversibility(str, Enum):
    """How easily an action can be undone."""

    INSTANT = "instant"
    WITHIN_HOUR = "within_hour"
    WITHIN_DAY = "within_day"
    WITHIN_WEEK = "within_week"
    NOT_APPLICABLE = "not_applicable"
    MAYBE_NOT = "maybe_not"


@dataclass
class ProactiveOpportunity:
    """An identified opportunity for proactive action."""

    opportunity_type: str
    description: str
    priority: int = 0  # Higher = more important
    potential_value: float = 0.0  # Estimated $ value
    urgency: float = 0.5  # 0-1 scale
    context: Dict[str, Any] = field(default_factory=dict)
    detected_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


@dataclass
class ProactiveAction:
    """A planned proactive action with constitutional validation."""

    agent_id: str
    action_type: ActionType
    autonomy_level: AutonomyLevel
    description: str
    rationale: str
    ihsan_score: float
    financial_impact: float = 0.0
    reversibility: Reversibility = Reversibility.INSTANT
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    requires_approval: bool = True
    approval_status: Optional[ApprovalStatus] = None
    execution_status: ExecutionStatus = ExecutionStatus.PLANNED
    outcome_value: Optional[float] = None
    context_snapshot: Optional[Dict] = None
    priority: int = 0

    def to_dict(self) -> Dict:
        """Convert action to dictionary."""
        return {
            "agent_id": self.agent_id,
            "action_type": self.action_type.value,
            "autonomy_level": self.autonomy_level.value,
            "description": self.description,
            "rationale": self.rationale,
            "ihsan_score": self.ihsan_score,
            "financial_impact": self.financial_impact,
            "reversibility": self.reversibility.value,
            "timestamp": self.timestamp.isoformat(),
            "requires_approval": self.requires_approval,
            "approval_status": (
                self.approval_status.value if self.approval_status else None
            ),
            "execution_status": self.execution_status.value,
            "outcome_value": self.outcome_value,
            "priority": self.priority,
        }

    def constitutional_check(self, autonomy_matrix: AutonomyMatrix) -> bool:
        """Check if action meets constitutional constraints."""
        context = ActionContext(
            action_type=self.action_type.value,
            description=self.description,
            risk_score=1.0 - self.ihsan_score,
            cost_percent=self.financial_impact,
            ihsan_score=self.ihsan_score,
            is_reversible=self.reversibility
            in [Reversibility.INSTANT, Reversibility.WITHIN_HOUR],
        )

        decision = autonomy_matrix.determine_autonomy(context)

        if not decision.can_execute:
            logger.warning(f"Constitutional check failed: {decision.reasoning}")
            return False

        self.requires_approval = not decision.can_execute
        return True


class BackgroundAgent(ABC):
    """
    Abstract base class for all background agents.

    Background agents:
    - Run periodically in the background
    - Identify opportunities in their domain
    - Plan actions within autonomy constraints
    - Execute approved actions
    - Learn from user feedback
    """

    def __init__(
        self,
        agent_id: str,
        name: str,
        action_type: ActionType,
        autonomy_level: AutonomyLevel = AutonomyLevel.AUTOLOW,
        run_interval: int = 300,  # seconds
        ihsan_threshold: float = 0.95,
    ):
        self.agent_id = agent_id
        self.name = name
        self.action_type = action_type
        self.autonomy_level = autonomy_level
        self.run_interval = run_interval
        self.ihsan_threshold = ihsan_threshold

        self.state = AgentState.IDLE
        self.last_run: Optional[datetime] = None
        self.user_preferences: Dict[str, Any] = {}

        # Statistics
        self.actions_taken = 0
        self.actions_approved = 0
        self.actions_rejected = 0
        self.total_value_created = 0.0
        self.avg_ihsan_score = 0.0

        # Event bus for integration
        self.event_bus = get_event_bus()

    def should_run(self) -> bool:
        """Check if agent should run based on interval."""
        if self.state == AgentState.DISABLED:
            return False

        if not self.last_run:
            return True

        elapsed = (datetime.now(timezone.utc) - self.last_run).total_seconds()
        return elapsed >= self.run_interval

    @abstractmethod
    async def identify_opportunities(
        self, context: Dict[str, Any]
    ) -> List[ProactiveOpportunity]:
        """
        Identify proactive opportunities in the agent's domain.

        Args:
            context: Current context from ContextMonitor/Muraqabah

        Returns:
            List of identified opportunities
        """
        pass

    @abstractmethod
    async def plan_action(
        self,
        opportunity: ProactiveOpportunity,
        user_preferences: Dict[str, Any],
    ) -> ProactiveAction:
        """
        Plan an action for an identified opportunity.

        Args:
            opportunity: The opportunity to address
            user_preferences: Learned user preferences

        Returns:
            Planned action with constitutional metadata
        """
        pass

    @abstractmethod
    async def execute_action(self, action: ProactiveAction) -> bool:
        """
        Execute the planned action.

        Args:
            action: The action to execute

        Returns:
            True if successful, False otherwise
        """
        pass

    def update_preferences(self, preference: str, value: Any) -> None:
        """Update user preference from feedback."""
        self.user_preferences[preference] = value
        logger.info(f"[{self.name}] Updated preference: {preference}={value}")

    def update_stats(self, action: ProactiveAction, success: bool) -> None:
        """Update agent statistics after action."""
        self.actions_taken += 1

        if success:
            self.actions_approved += 1
            if action.outcome_value:
                self.total_value_created += action.outcome_value

        # Update average Ihsan
        self.avg_ihsan_score = (
            self.avg_ihsan_score * (self.actions_taken - 1) + action.ihsan_score
        ) / self.actions_taken

    def stats(self) -> Dict[str, Any]:
        """Get agent statistics."""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "state": self.state.value,
            "last_run": self.last_run.isoformat() if self.last_run else None,
            "actions_taken": self.actions_taken,
            "actions_approved": self.actions_approved,
            "actions_rejected": self.actions_rejected,
            "total_value_created": self.total_value_created,
            "avg_ihsan_score": self.avg_ihsan_score,
            "run_interval": self.run_interval,
            "autonomy_level": self.autonomy_level.name,
        }


# =============================================================================
# DOMAIN-SPECIFIC AGENT IMPLEMENTATIONS
# =============================================================================


class CalendarOptimizer(BackgroundAgent):
    """
    Optimizes user's calendar proactively.

    Capabilities:
    - Block focus time
    - Consolidate meetings
    - Suggest breaks
    - Detect scheduling conflicts
    """

    def __init__(self):
        super().__init__(
            agent_id="calendar_optimizer_v1",
            name="Calendar Optimizer",
            action_type=ActionType.CALENDAR_OPTIMIZE,
            autonomy_level=AutonomyLevel.AUTOLOW,
            run_interval=900,  # 15 minutes
            ihsan_threshold=0.95,
        )

        # Default preferences
        self.user_preferences = {
            "focus_hours": {"start": 9, "end": 11},
            "meeting_days": ["Tuesday", "Thursday"],
            "break_interval_minutes": 90,
            "auto_block_focus": True,
        }

    async def identify_opportunities(
        self, context: Dict[str, Any]
    ) -> List[ProactiveOpportunity]:
        """Identify calendar optimization opportunities."""
        opportunities = []

        calendar_data = context.get("calendar", {})
        events = calendar_data.get("events", [])
        current_time = context.get("current_time", datetime.now(timezone.utc))

        # 1. Check for missing focus time
        has_focus = any(
            e.get("summary", "").lower() == "focus time"
            for e in events
            if e.get("start", datetime.max) > current_time
        )

        if not has_focus and self.user_preferences.get("auto_block_focus"):
            opportunities.append(
                ProactiveOpportunity(
                    opportunity_type="missing_focus_time",
                    description="No focus time scheduled",
                    priority=3,
                    potential_value=2.0,
                    urgency=0.6,
                )
            )

        # 2. Check for back-to-back meetings
        sorted_events = sorted(events, key=lambda x: x.get("start", datetime.min))
        for i in range(len(sorted_events) - 1):
            current_end = sorted_events[i].get("end")
            next_start = sorted_events[i + 1].get("start")

            if current_end and next_start:
                gap_minutes = (next_start - current_end).total_seconds() / 60
                if 0 < gap_minutes < 15:  # Less than 15 min break
                    opportunities.append(
                        ProactiveOpportunity(
                            opportunity_type="needs_break",
                            description=f"Back-to-back events, {gap_minutes:.0f}min gap",
                            priority=1,
                            potential_value=0.1,
                            urgency=0.3,
                        )
                    )

        return opportunities

    async def plan_action(
        self,
        opportunity: ProactiveOpportunity,
        user_preferences: Dict[str, Any],
    ) -> ProactiveAction:
        """Plan a calendar optimization action."""
        opp_type = opportunity.opportunity_type

        if opp_type == "missing_focus_time":
            return ProactiveAction(
                agent_id=self.agent_id,
                action_type=self.action_type,
                autonomy_level=self.autonomy_level,
                description="Block 2-hour focus time in calendar",
                rationale="Protect deep work time for maximum productivity",
                ihsan_score=0.96,
                financial_impact=opportunity.potential_value,
                reversibility=Reversibility.INSTANT,
                priority=opportunity.priority,
            )

        elif opp_type == "needs_break":
            return ProactiveAction(
                agent_id=self.agent_id,
                action_type=self.action_type,
                autonomy_level=self.autonomy_level,
                description="Schedule 15-minute break",
                rationale="Prevent burnout and maintain cognitive performance",
                ihsan_score=0.92,
                financial_impact=opportunity.potential_value,
                reversibility=Reversibility.INSTANT,
                priority=opportunity.priority,
            )

        raise ValueError(f"Unknown opportunity type: {opp_type}")

    async def execute_action(self, action: ProactiveAction) -> bool:
        """Execute calendar optimization."""
        try:
            logger.info(f"📅 [{self.name}] Executing: {action.description}")

            # In production: integrate with Google Calendar/Outlook API
            await asyncio.sleep(0.5)  # Simulate execution

            action.execution_status = ExecutionStatus.SUCCESS
            action.outcome_value = action.financial_impact

            # Emit event
            await self.event_bus.emit(
                topic="background_agent.action.executed",
                payload=action.to_dict(),
                priority=EventPriority.NORMAL,
            )

            self.update_stats(action, True)
            return True

        except Exception as e:
            logger.error(f"Calendar optimization failed: {e}")
            action.execution_status = ExecutionStatus.FAILED
            return False


class EmailTriage(BackgroundAgent):
    """
    Proactively triages and organizes emails.

    Capabilities:
    - Flag urgent emails
    - Archive low-priority
    - Batch categorization
    - Detect email overload
    """

    def __init__(self):
        super().__init__(
            agent_id="email_triage_v1",
            name="Email Triage",
            action_type=ActionType.EMAIL_TRIAGE,
            autonomy_level=AutonomyLevel.SUGGESTER,  # Start conservative
            run_interval=600,  # 10 minutes
            ihsan_threshold=0.95,
        )

        self.priority_keywords = [
            "urgent",
            "asap",
            "important",
            "action required",
            "deadline",
        ]
        self.low_priority_keywords = [
            "newsletter",
            "promotion",
            "notification",
            "unsubscribe",
        ]

    async def identify_opportunities(
        self, context: Dict[str, Any]
    ) -> List[ProactiveOpportunity]:
        """Identify email triage opportunities."""
        opportunities = []

        email_data = context.get("email", {})
        unread_count = email_data.get("unread_count", 0)
        recent_emails = email_data.get("recent_emails", [])

        # Check for urgent emails
        for email in recent_emails:
            subject = email.get("subject", "").lower()
            sender = email.get("from", "")

            if any(kw in subject for kw in self.priority_keywords):
                opportunities.append(
                    ProactiveOpportunity(
                        opportunity_type="urgent_email",
                        description=f"Urgent email from {sender[:20]}",
                        priority=4,
                        potential_value=5.0,
                        urgency=0.9,
                        context={"email_id": email.get("id")},
                    )
                )

            elif any(kw in subject for kw in self.low_priority_keywords):
                opportunities.append(
                    ProactiveOpportunity(
                        opportunity_type="low_priority_email",
                        description=f"Low priority: {subject[:30]}...",
                        priority=1,
                        potential_value=0.1,
                        urgency=0.2,
                        context={"email_id": email.get("id")},
                    )
                )

        # Check for email overload
        if unread_count > 20:
            opportunities.append(
                ProactiveOpportunity(
                    opportunity_type="email_overload",
                    description=f"High volume: {unread_count} unread emails",
                    priority=3,
                    potential_value=10.0,
                    urgency=0.7,
                )
            )

        return opportunities

    async def plan_action(
        self,
        opportunity: ProactiveOpportunity,
        user_preferences: Dict[str, Any],
    ) -> ProactiveAction:
        """Plan an email triage action."""
        opp_type = opportunity.opportunity_type

        if opp_type == "urgent_email":
            return ProactiveAction(
                agent_id=self.agent_id,
                action_type=self.action_type,
                autonomy_level=self.autonomy_level,
                description=f"Flag as urgent: {opportunity.description}",
                rationale="Important communication requiring attention",
                ihsan_score=0.94,
                financial_impact=opportunity.potential_value,
                reversibility=Reversibility.INSTANT,
                priority=opportunity.priority,
                context_snapshot=opportunity.context,
            )

        elif opp_type == "low_priority_email":
            return ProactiveAction(
                agent_id=self.agent_id,
                action_type=self.action_type,
                autonomy_level=self.autonomy_level,
                description="Archive low priority email",
                rationale="Reduce inbox clutter for better focus",
                ihsan_score=0.86,
                financial_impact=opportunity.potential_value,
                reversibility=Reversibility.INSTANT,
                priority=opportunity.priority,
                context_snapshot=opportunity.context,
            )

        elif opp_type == "email_overload":
            return ProactiveAction(
                agent_id=self.agent_id,
                action_type=self.action_type,
                autonomy_level=self.autonomy_level,
                description="Batch process and categorize emails",
                rationale="Reduce cognitive load from email overload",
                ihsan_score=0.91,
                financial_impact=opportunity.potential_value,
                reversibility=Reversibility.WITHIN_HOUR,
                priority=opportunity.priority,
            )

        raise ValueError(f"Unknown opportunity type: {opp_type}")

    async def execute_action(self, action: ProactiveAction) -> bool:
        """Execute email triage action."""
        try:
            logger.info(f"📧 [{self.name}] Executing: {action.description}")

            # In production: integrate with email APIs
            await asyncio.sleep(0.3)

            action.execution_status = ExecutionStatus.SUCCESS
            action.outcome_value = action.financial_impact

            await self.event_bus.emit(
                topic="background_agent.action.executed",
                payload=action.to_dict(),
                priority=EventPriority.NORMAL,
            )

            self.update_stats(action, True)
            return True

        except Exception as e:
            logger.error(f"Email triage failed: {e}")
            action.execution_status = ExecutionStatus.FAILED
            return False


class FileOrganizer(BackgroundAgent):
    """
    Organizes files and documents proactively.

    Capabilities:
    - Organize downloads folder
    - Clean desktop
    - Find duplicates
    - Archive old files
    """

    def __init__(self):
        super().__init__(
            agent_id="file_organizer_v1",
            name="File Organizer",
            action_type=ActionType.FILE_ORGANIZE,
            autonomy_level=AutonomyLevel.AUTOLOW,
            run_interval=1800,  # 30 minutes
            ihsan_threshold=0.95,
        )

        self.file_categories = {
            "documents": [".pdf", ".doc", ".docx", ".txt", ".rtf", ".odt"],
            "spreadsheets": [".xls", ".xlsx", ".csv", ".ods"],
            "presentations": [".ppt", ".pptx", ".key", ".odp"],
            "images": [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp"],
            "code": [".py", ".js", ".ts", ".java", ".cpp", ".rs", ".go"],
            "archives": [".zip", ".tar", ".gz", ".rar", ".7z"],
        }

    async def identify_opportunities(
        self, context: Dict[str, Any]
    ) -> List[ProactiveOpportunity]:
        """Identify file organization opportunities."""
        opportunities = []

        file_data = context.get("files", {})
        downloads = file_data.get("downloads_folder", [])
        desktop = file_data.get("desktop_files", [])

        # Check downloads folder
        if len(downloads) > 10:
            opportunities.append(
                ProactiveOpportunity(
                    opportunity_type="organize_downloads",
                    description=f"Organize {len(downloads)} files in Downloads",
                    priority=2,
                    potential_value=5.0,
                    urgency=0.4,
                )
            )

        # Check desktop
        if len(desktop) > 15:
            opportunities.append(
                ProactiveOpportunity(
                    opportunity_type="clean_desktop",
                    description=f"Organize {len(desktop)} files on Desktop",
                    priority=1,
                    potential_value=3.0,
                    urgency=0.3,
                )
            )

        # Check for duplicates (simplified)
        all_files = downloads + desktop
        seen_hashes = {}
        duplicates = []

        for f in all_files:
            name = f.get("name", "")
            h = hashlib.md5(name.encode(), usedforsecurity=False).hexdigest()[:8]
            if h in seen_hashes:
                duplicates.append(f)
            seen_hashes[h] = f

        if duplicates:
            opportunities.append(
                ProactiveOpportunity(
                    opportunity_type="remove_duplicates",
                    description=f"Found {len(duplicates)} potential duplicates",
                    priority=3,
                    potential_value=2.0,
                    urgency=0.5,
                )
            )

        return opportunities

    async def plan_action(
        self,
        opportunity: ProactiveOpportunity,
        user_preferences: Dict[str, Any],
    ) -> ProactiveAction:
        """Plan a file organization action."""
        opp_type = opportunity.opportunity_type

        if opp_type == "organize_downloads":
            return ProactiveAction(
                agent_id=self.agent_id,
                action_type=self.action_type,
                autonomy_level=self.autonomy_level,
                description=opportunity.description,
                rationale="Keep downloads folder organized for easy access",
                ihsan_score=0.87,
                financial_impact=opportunity.potential_value,
                reversibility=Reversibility.INSTANT,
                priority=opportunity.priority,
            )

        elif opp_type == "clean_desktop":
            return ProactiveAction(
                agent_id=self.agent_id,
                action_type=self.action_type,
                autonomy_level=self.autonomy_level,
                description=opportunity.description,
                rationale="Clean desktop improves focus",
                ihsan_score=0.89,
                financial_impact=opportunity.potential_value,
                reversibility=Reversibility.INSTANT,
                priority=opportunity.priority,
            )

        elif opp_type == "remove_duplicates":
            return ProactiveAction(
                agent_id=self.agent_id,
                action_type=self.action_type,
                autonomy_level=self.autonomy_level,
                description=opportunity.description,
                rationale="Remove duplicate files to save storage",
                ihsan_score=0.85,
                financial_impact=opportunity.potential_value,
                reversibility=Reversibility.WITHIN_HOUR,
                priority=opportunity.priority,
            )

        raise ValueError(f"Unknown opportunity type: {opp_type}")

    async def execute_action(self, action: ProactiveAction) -> bool:
        """Execute file organization action."""
        try:
            logger.info(f"📁 [{self.name}] Executing: {action.description}")

            # In production: use OS file system APIs
            await asyncio.sleep(0.5)

            action.execution_status = ExecutionStatus.SUCCESS
            action.outcome_value = action.financial_impact

            await self.event_bus.emit(
                topic="background_agent.action.executed",
                payload=action.to_dict(),
                priority=EventPriority.NORMAL,
            )

            self.update_stats(action, True)
            return True

        except Exception as e:
            logger.error(f"File organization failed: {e}")
            action.execution_status = ExecutionStatus.FAILED
            return False


# =============================================================================
# AGENT REGISTRY AND MANAGER
# =============================================================================


class BackgroundAgentRegistry:
    """
    Registry for background agents.

    Manages agent lifecycle and integration with
    ProactiveSovereignEntity components.
    """

    def __init__(
        self,
        autonomy_matrix: Optional[AutonomyMatrix] = None,
        scheduler: Optional[ProactiveScheduler] = None,
    ):
        self.autonomy_matrix = autonomy_matrix
        self.scheduler = scheduler
        self._agents: Dict[str, BackgroundAgent] = {}
        self._running = False

    def register(self, agent: BackgroundAgent) -> None:
        """Register a background agent."""
        self._agents[agent.agent_id] = agent
        logger.info(f"Registered background agent: {agent.name} ({agent.agent_id})")

    def unregister(self, agent_id: str) -> None:
        """Unregister a background agent."""
        if agent_id in self._agents:
            del self._agents[agent_id]
            logger.info(f"Unregistered background agent: {agent_id}")

    def get_agent(self, agent_id: str) -> Optional[BackgroundAgent]:
        """Get an agent by ID."""
        return self._agents.get(agent_id)

    def list_agents(self) -> List[Dict[str, Any]]:
        """List all registered agents."""
        return [agent.stats() for agent in self._agents.values()]

    async def run_agent(
        self,
        agent_id: str,
        context: Dict[str, Any],
    ) -> List[ProactiveAction]:
        """Run a specific agent and get its actions."""
        agent = self._agents.get(agent_id)
        if not agent:
            return []

        if not agent.should_run():
            return []

        agent.state = AgentState.RUNNING
        agent.last_run = datetime.now(timezone.utc)

        try:
            # Identify opportunities
            opportunities = await agent.identify_opportunities(context)

            actions = []
            for opp in opportunities:
                # Plan action
                action = await agent.plan_action(opp, agent.user_preferences)

                # Constitutional check
                if self.autonomy_matrix:
                    if not action.constitutional_check(self.autonomy_matrix):
                        continue

                actions.append(action)

            return actions

        finally:
            agent.state = AgentState.IDLE

    async def run_all_agents(
        self,
        context: Dict[str, Any],
    ) -> List[ProactiveAction]:
        """Run all agents and collect actions."""
        all_actions = []

        for agent_id in self._agents:
            actions = await self.run_agent(agent_id, context)
            all_actions.extend(actions)

        return all_actions

    def stats(self) -> Dict[str, Any]:
        """Get registry statistics."""
        return {
            "registered_agents": len(self._agents),
            "agents": self.list_agents(),
            "total_actions": sum(a.actions_taken for a in self._agents.values()),
            "total_value_created": sum(
                a.total_value_created for a in self._agents.values()
            ),
        }


def create_default_registry(
    autonomy_matrix: Optional[AutonomyMatrix] = None,
    scheduler: Optional[ProactiveScheduler] = None,
) -> BackgroundAgentRegistry:
    """Create registry with default agents."""
    registry = BackgroundAgentRegistry(
        autonomy_matrix=autonomy_matrix,
        scheduler=scheduler,
    )

    # Register default agents
    registry.register(CalendarOptimizer())
    registry.register(EmailTriage())
    registry.register(FileOrganizer())

    return registry


__all__ = [
    # Enums
    "AgentState",
    "ActionType",
    "ApprovalStatus",
    "ExecutionStatus",
    "Reversibility",
    # Data classes
    "ProactiveOpportunity",
    "ProactiveAction",
    # Base class
    "BackgroundAgent",
    # Implementations
    "CalendarOptimizer",
    "EmailTriage",
    "FileOrganizer",
    # Registry
    "BackgroundAgentRegistry",
    "create_default_registry",
]
