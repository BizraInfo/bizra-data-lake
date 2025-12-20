# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA Constellation - Trigger System v1.0
# ═══════════════════════════════════════════════════════════════════════════════
"""
Condition-based trigger system for automatic agent activation:
- Pattern matching triggers
- Time-based triggers
- Event-driven triggers
- Threshold-based triggers
- Compound triggers with AND/OR logic
"""

from __future__ import annotations

import re
import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from typing import Optional, Any, Callable, Awaitable
from enum import Enum
from abc import ABC, abstractmethod


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# TRIGGER TYPES
# ─────────────────────────────────────────────────────────────────────────────

class TriggerType(str, Enum):
    """Types of triggers."""
    PATTERN = "pattern"          # Text pattern matching
    EVENT = "event"              # Hook event triggers
    SCHEDULE = "schedule"        # Time-based triggers
    THRESHOLD = "threshold"      # Metric threshold triggers
    COMPOUND = "compound"        # Multiple conditions
    AGENT_OUTPUT = "agent_output"  # Triggered by another agent's output
    KNOWLEDGE = "knowledge"      # Knowledge graph changes


class TriggerState(str, Enum):
    """State of a trigger."""
    ACTIVE = "active"
    PAUSED = "paused"
    FIRED = "fired"
    COOLDOWN = "cooldown"
    DISABLED = "disabled"


# ─────────────────────────────────────────────────────────────────────────────
# TRIGGER CONDITIONS
# ─────────────────────────────────────────────────────────────────────────────

class TriggerCondition(ABC):
    """Base class for trigger conditions."""
    
    @abstractmethod
    def evaluate(self, context: dict) -> bool:
        """Evaluate if condition is met."""
        pass
        
    @abstractmethod
    def describe(self) -> str:
        """Human-readable description of condition."""
        pass


class PatternCondition(TriggerCondition):
    """Matches text against a pattern."""
    
    def __init__(
        self,
        pattern: str,
        field: str = "content",
        is_regex: bool = False,
        case_sensitive: bool = False,
    ):
        self.pattern = pattern
        self.field = field
        self.is_regex = is_regex
        self.case_sensitive = case_sensitive
        
        if is_regex:
            flags = 0 if case_sensitive else re.IGNORECASE
            self._compiled = re.compile(pattern, flags)
        else:
            self._compiled = None
            
    def evaluate(self, context: dict) -> bool:
        text = context.get(self.field, "")
        
        if not isinstance(text, str):
            return False
            
        if self.is_regex:
            return bool(self._compiled.search(text))
        else:
            if self.case_sensitive:
                return self.pattern in text
            else:
                return self.pattern.lower() in text.lower()
                
    def describe(self) -> str:
        kind = "regex" if self.is_regex else "text"
        return f"Pattern match ({kind}): '{self.pattern}' in {self.field}"


class EventCondition(TriggerCondition):
    """Matches specific events."""
    
    def __init__(
        self,
        event_type: str,
        filters: Optional[dict] = None,
    ):
        self.event_type = event_type
        self.filters = filters or {}
        
    def evaluate(self, context: dict) -> bool:
        if context.get("event_type") != self.event_type:
            return False
            
        for key, expected in self.filters.items():
            if context.get(key) != expected:
                return False
                
        return True
        
    def describe(self) -> str:
        filters_str = ", ".join(f"{k}={v}" for k, v in self.filters.items())
        return f"Event: {self.event_type}" + (f" with {filters_str}" if filters_str else "")


class ThresholdCondition(TriggerCondition):
    """Evaluates numeric thresholds."""
    
    def __init__(
        self,
        field: str,
        operator: str,  # >, <, >=, <=, ==, !=
        value: float,
    ):
        self.field = field
        self.operator = operator
        self.value = value
        
        self._ops = {
            ">": lambda a, b: a > b,
            "<": lambda a, b: a < b,
            ">=": lambda a, b: a >= b,
            "<=": lambda a, b: a <= b,
            "==": lambda a, b: a == b,
            "!=": lambda a, b: a != b,
        }
        
    def evaluate(self, context: dict) -> bool:
        actual = context.get(self.field)
        
        if actual is None:
            return False
            
        try:
            return self._ops[self.operator](float(actual), self.value)
        except (ValueError, KeyError):
            return False
            
    def describe(self) -> str:
        return f"Threshold: {self.field} {self.operator} {self.value}"


class CompoundCondition(TriggerCondition):
    """Combines multiple conditions with AND/OR logic."""
    
    def __init__(
        self,
        conditions: list[TriggerCondition],
        operator: str = "AND",  # AND or OR
    ):
        self.conditions = conditions
        self.operator = operator.upper()
        
    def evaluate(self, context: dict) -> bool:
        results = [c.evaluate(context) for c in self.conditions]
        
        if self.operator == "AND":
            return all(results)
        else:  # OR
            return any(results)
            
    def describe(self) -> str:
        descs = [c.describe() for c in self.conditions]
        return f" {self.operator} ".join(f"({d})" for d in descs)


# ─────────────────────────────────────────────────────────────────────────────
# TRIGGER ACTION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TriggerAction:
    """Action to take when trigger fires."""
    action_type: str  # "invoke_agent", "run_command", "send_message", "emit_event"
    target: str  # Agent slug, command name, etc.
    parameters: dict = field(default_factory=dict)
    priority: str = "normal"  # high, normal, low
    
    def to_dict(self) -> dict:
        return {
            "action_type": self.action_type,
            "target": self.target,
            "parameters": self.parameters,
            "priority": self.priority,
        }


# ─────────────────────────────────────────────────────────────────────────────
# TRIGGER DEFINITION
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Trigger:
    """A complete trigger definition."""
    id: str
    name: str
    trigger_type: TriggerType
    condition: TriggerCondition
    actions: list[TriggerAction]
    
    # Configuration
    enabled: bool = True
    cooldown_seconds: int = 0  # Minimum time between firings
    max_fires: Optional[int] = None  # Maximum times to fire
    
    # State
    state: TriggerState = TriggerState.ACTIVE
    fire_count: int = 0
    last_fired: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Metadata
    description: Optional[str] = None
    owner_agent: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    
    def can_fire(self) -> bool:
        """Check if trigger can currently fire."""
        if not self.enabled:
            return False
            
        if self.state not in [TriggerState.ACTIVE, TriggerState.FIRED]:
            return False
            
        # Check max fires
        if self.max_fires and self.fire_count >= self.max_fires:
            return False
            
        # Check cooldown
        if self.last_fired and self.cooldown_seconds > 0:
            last = datetime.fromisoformat(self.last_fired.replace('Z', '+00:00'))
            elapsed = (datetime.now(timezone.utc) - last).total_seconds()
            if elapsed < self.cooldown_seconds:
                return False
                
        return True
        
    def evaluate(self, context: dict) -> bool:
        """Evaluate if trigger conditions are met."""
        if not self.can_fire():
            return False
        return self.condition.evaluate(context)
        
    def record_fire(self) -> None:
        """Record that trigger has fired."""
        self.fire_count += 1
        self.last_fired = datetime.now(timezone.utc).isoformat()
        self.state = TriggerState.FIRED
        
        if self.cooldown_seconds > 0:
            self.state = TriggerState.COOLDOWN
            
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "trigger_type": self.trigger_type.value,
            "condition": self.condition.describe(),
            "actions": [a.to_dict() for a in self.actions],
            "enabled": self.enabled,
            "cooldown_seconds": self.cooldown_seconds,
            "max_fires": self.max_fires,
            "state": self.state.value,
            "fire_count": self.fire_count,
            "last_fired": self.last_fired,
            "created_at": self.created_at,
            "description": self.description,
            "owner_agent": self.owner_agent,
            "tags": self.tags,
        }


# ─────────────────────────────────────────────────────────────────────────────
# TRIGGER ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class TriggerEngine:
    """
    Engine for managing and evaluating triggers.
    
    Provides:
    - Trigger registration and management
    - Real-time evaluation against context
    - Action execution
    - Scheduled trigger evaluation
    """
    
    def __init__(self):
        self._triggers: dict[str, Trigger] = {}
        self._action_handlers: dict[str, Callable] = {}
        self._trigger_counter = 0
        self._running = False
        self._scheduler_task: Optional[asyncio.Task] = None
        
        # Register default action handlers
        self._register_default_handlers()
        
    def _register_default_handlers(self) -> None:
        """Register default action handlers."""
        self._action_handlers["invoke_agent"] = self._handle_invoke_agent
        self._action_handlers["emit_event"] = self._handle_emit_event
        self._action_handlers["log"] = self._handle_log
        
    async def _handle_invoke_agent(
        self,
        action: TriggerAction,
        context: dict,
    ) -> Any:
        """Handle agent invocation action."""
        # This will be connected to the orchestrator
        logger.info(f"Trigger action: invoke agent {action.target}")
        return {"agent": action.target, "parameters": action.parameters}
        
    async def _handle_emit_event(
        self,
        action: TriggerAction,
        context: dict,
    ) -> Any:
        """Handle event emission action."""
        logger.info(f"Trigger action: emit event {action.target}")
        return {"event": action.target, "data": action.parameters}
        
    async def _handle_log(
        self,
        action: TriggerAction,
        context: dict,
    ) -> Any:
        """Handle logging action."""
        message = action.parameters.get("message", "Trigger fired")
        level = action.parameters.get("level", "info")
        getattr(logger, level)(f"Trigger {context.get('trigger_id')}: {message}")
        return {"logged": True}
        
    def register_trigger(self, trigger: Trigger) -> str:
        """Register a new trigger."""
        self._triggers[trigger.id] = trigger
        logger.info(f"Registered trigger: {trigger.name} ({trigger.id})")
        return trigger.id
        
    def create_trigger(
        self,
        name: str,
        trigger_type: TriggerType,
        condition: TriggerCondition,
        actions: list[TriggerAction],
        **kwargs,
    ) -> Trigger:
        """Create and register a trigger."""
        self._trigger_counter += 1
        trigger_id = f"trigger_{self._trigger_counter:05d}"
        
        trigger = Trigger(
            id=trigger_id,
            name=name,
            trigger_type=trigger_type,
            condition=condition,
            actions=actions,
            **kwargs,
        )
        
        self.register_trigger(trigger)
        return trigger
        
    def remove_trigger(self, trigger_id: str) -> bool:
        """Remove a trigger."""
        if trigger_id in self._triggers:
            del self._triggers[trigger_id]
            return True
        return False
        
    def enable_trigger(self, trigger_id: str) -> bool:
        """Enable a trigger."""
        if trigger_id in self._triggers:
            self._triggers[trigger_id].enabled = True
            self._triggers[trigger_id].state = TriggerState.ACTIVE
            return True
        return False
        
    def disable_trigger(self, trigger_id: str) -> bool:
        """Disable a trigger."""
        if trigger_id in self._triggers:
            self._triggers[trigger_id].enabled = False
            self._triggers[trigger_id].state = TriggerState.DISABLED
            return True
        return False
        
    def register_action_handler(
        self,
        action_type: str,
        handler: Callable,
    ) -> None:
        """Register a custom action handler."""
        self._action_handlers[action_type] = handler
        
    async def evaluate(self, context: dict) -> list[dict]:
        """
        Evaluate all triggers against context.
        
        Returns list of fired triggers and their action results.
        """
        fired = []
        
        for trigger in self._triggers.values():
            if trigger.evaluate(context):
                # Execute actions
                action_results = []
                
                for action in trigger.actions:
                    handler = self._action_handlers.get(action.action_type)
                    if handler:
                        try:
                            result = await handler(
                                action,
                                {**context, "trigger_id": trigger.id},
                            )
                            action_results.append({
                                "action": action.to_dict(),
                                "result": result,
                                "success": True,
                            })
                        except Exception as e:
                            action_results.append({
                                "action": action.to_dict(),
                                "error": str(e),
                                "success": False,
                            })
                    else:
                        action_results.append({
                            "action": action.to_dict(),
                            "error": f"No handler for action type: {action.action_type}",
                            "success": False,
                        })
                        
                trigger.record_fire()
                
                fired.append({
                    "trigger": trigger.to_dict(),
                    "action_results": action_results,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })
                
        return fired
        
    def get_trigger(self, trigger_id: str) -> Optional[Trigger]:
        """Get trigger by ID."""
        return self._triggers.get(trigger_id)
        
    def get_triggers(
        self,
        trigger_type: Optional[TriggerType] = None,
        state: Optional[TriggerState] = None,
        owner_agent: Optional[str] = None,
    ) -> list[Trigger]:
        """Get triggers with optional filtering."""
        triggers = list(self._triggers.values())
        
        if trigger_type:
            triggers = [t for t in triggers if t.trigger_type == trigger_type]
        if state:
            triggers = [t for t in triggers if t.state == state]
        if owner_agent:
            triggers = [t for t in triggers if t.owner_agent == owner_agent]
            
        return triggers
        
    def get_stats(self) -> dict:
        """Get trigger statistics."""
        return {
            "total": len(self._triggers),
            "by_type": {
                tt.value: len([t for t in self._triggers.values() if t.trigger_type == tt])
                for tt in TriggerType
            },
            "by_state": {
                ts.value: len([t for t in self._triggers.values() if t.state == ts])
                for ts in TriggerState
            },
            "total_fires": sum(t.fire_count for t in self._triggers.values()),
        }


# ─────────────────────────────────────────────────────────────────────────────
# TRIGGER BUILDER
# ─────────────────────────────────────────────────────────────────────────────

class TriggerBuilder:
    """Fluent builder for creating triggers."""
    
    def __init__(self, engine: TriggerEngine):
        self.engine = engine
        self._name: Optional[str] = None
        self._type: TriggerType = TriggerType.PATTERN
        self._conditions: list[TriggerCondition] = []
        self._actions: list[TriggerAction] = []
        self._cooldown: int = 0
        self._max_fires: Optional[int] = None
        self._description: Optional[str] = None
        self._owner: Optional[str] = None
        self._tags: list[str] = []
        
    def named(self, name: str) -> "TriggerBuilder":
        """Set trigger name."""
        self._name = name
        return self
        
    def of_type(self, trigger_type: TriggerType) -> "TriggerBuilder":
        """Set trigger type."""
        self._type = trigger_type
        return self
        
    def when_pattern(
        self,
        pattern: str,
        field: str = "content",
        is_regex: bool = False,
    ) -> "TriggerBuilder":
        """Add pattern condition."""
        self._conditions.append(PatternCondition(pattern, field, is_regex))
        return self
        
    def when_event(
        self,
        event_type: str,
        **filters,
    ) -> "TriggerBuilder":
        """Add event condition."""
        self._conditions.append(EventCondition(event_type, filters))
        return self
        
    def when_threshold(
        self,
        field: str,
        operator: str,
        value: float,
    ) -> "TriggerBuilder":
        """Add threshold condition."""
        self._conditions.append(ThresholdCondition(field, operator, value))
        return self
        
    def with_and_logic(self) -> "TriggerBuilder":
        """Combine conditions with AND (default)."""
        # Already default
        return self
        
    def with_or_logic(self) -> "TriggerBuilder":
        """Combine conditions with OR."""
        if len(self._conditions) > 1:
            self._conditions = [CompoundCondition(self._conditions, "OR")]
        return self
        
    def invoke_agent(
        self,
        agent_slug: str,
        **parameters,
    ) -> "TriggerBuilder":
        """Add agent invocation action."""
        self._actions.append(TriggerAction(
            action_type="invoke_agent",
            target=agent_slug,
            parameters=parameters,
        ))
        return self
        
    def emit_event(
        self,
        event_name: str,
        **data,
    ) -> "TriggerBuilder":
        """Add event emission action."""
        self._actions.append(TriggerAction(
            action_type="emit_event",
            target=event_name,
            parameters=data,
        ))
        return self
        
    def with_cooldown(self, seconds: int) -> "TriggerBuilder":
        """Set cooldown between firings."""
        self._cooldown = seconds
        return self
        
    def max_fires(self, count: int) -> "TriggerBuilder":
        """Set maximum fire count."""
        self._max_fires = count
        return self
        
    def described_as(self, description: str) -> "TriggerBuilder":
        """Set description."""
        self._description = description
        return self
        
    def owned_by(self, agent_slug: str) -> "TriggerBuilder":
        """Set owner agent."""
        self._owner = agent_slug
        return self
        
    def tagged(self, *tags: str) -> "TriggerBuilder":
        """Add tags."""
        self._tags.extend(tags)
        return self
        
    def build(self) -> Trigger:
        """Build and register the trigger."""
        if not self._name:
            raise ValueError("Trigger must have a name")
        if not self._conditions:
            raise ValueError("Trigger must have at least one condition")
        if not self._actions:
            raise ValueError("Trigger must have at least one action")
            
        # Combine conditions if multiple
        if len(self._conditions) == 1:
            condition = self._conditions[0]
        else:
            condition = CompoundCondition(self._conditions, "AND")
            
        return self.engine.create_trigger(
            name=self._name,
            trigger_type=self._type,
            condition=condition,
            actions=self._actions,
            cooldown_seconds=self._cooldown,
            max_fires=self._max_fires,
            description=self._description,
            owner_agent=self._owner,
            tags=self._tags,
        )


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL ENGINE ACCESS
# ─────────────────────────────────────────────────────────────────────────────

_engine: Optional[TriggerEngine] = None


def get_trigger_engine() -> TriggerEngine:
    """Get the global trigger engine."""
    global _engine
    if _engine is None:
        _engine = TriggerEngine()
    return _engine


def trigger(name: str) -> TriggerBuilder:
    """Create a trigger builder."""
    return TriggerBuilder(get_trigger_engine()).named(name)
