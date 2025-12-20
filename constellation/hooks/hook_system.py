# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA Constellation - Hooking System v1.0
# ═══════════════════════════════════════════════════════════════════════════════
"""
Event-driven hook system for:
- Agent lifecycle events (start, complete, error)
- Tool invocation hooks (before, after)
- Output processing hooks
- Memory operations hooks
- Cross-agent communication hooks
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import (
    Optional,
    Any,
    Callable,
    Awaitable,
    TypeVar,
    Generic,
    Union,
)
from enum import Enum
from functools import wraps


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# HOOK TYPES
# ─────────────────────────────────────────────────────────────────────────────

class HookPoint(str, Enum):
    """Points where hooks can be attached."""
    # Agent Lifecycle
    AGENT_START = "agent.start"
    AGENT_COMPLETE = "agent.complete"
    AGENT_ERROR = "agent.error"
    AGENT_HANDOFF = "agent.handoff"
    
    # Task Processing
    TASK_RECEIVED = "task.received"
    TASK_ANALYZED = "task.analyzed"
    TASK_ROUTED = "task.routed"
    TASK_COMPLETED = "task.completed"
    
    # Team Operations
    TEAM_ASSEMBLED = "team.assembled"
    TEAM_WORK_START = "team.work_start"
    TEAM_WORK_COMPLETE = "team.work_complete"
    
    # Tool Invocation
    TOOL_BEFORE = "tool.before"
    TOOL_AFTER = "tool.after"
    TOOL_ERROR = "tool.error"
    
    # Output Processing
    OUTPUT_GENERATED = "output.generated"
    OUTPUT_VERIFIED = "output.verified"
    OUTPUT_FAILED_SNR = "output.failed_snr"
    
    # Memory Operations
    MEMORY_STORE = "memory.store"
    MEMORY_RECALL = "memory.recall"
    MEMORY_CONSOLIDATE = "memory.consolidate"
    
    # Knowledge Graph
    KNOWLEDGE_ADD = "knowledge.add"
    KNOWLEDGE_QUERY = "knowledge.query"
    KNOWLEDGE_VERIFY = "knowledge.verify"
    
    # Cross-Agent
    MESSAGE_SENT = "message.sent"
    MESSAGE_RECEIVED = "message.received"
    DELEGATION = "delegation"
    
    # System Events
    SYSTEM_START = "system.start"
    SYSTEM_SHUTDOWN = "system.shutdown"
    HEALTH_CHECK = "system.health_check"


class HookPriority(int, Enum):
    """Priority levels for hook execution order."""
    HIGHEST = 0
    HIGH = 25
    NORMAL = 50
    LOW = 75
    LOWEST = 100


# ─────────────────────────────────────────────────────────────────────────────
# HOOK EVENT DATA
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class HookEvent:
    """Data passed to hook handlers."""
    hook_point: HookPoint
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    agent_slug: Optional[str] = None
    session_id: Optional[str] = None
    task_id: Optional[str] = None
    data: dict = field(default_factory=dict)
    metadata: dict = field(default_factory=dict)
    
    # Mutable fields for hook chain
    cancelled: bool = False
    modified_data: Optional[dict] = None
    
    def cancel(self) -> None:
        """Cancel the operation this hook is attached to."""
        self.cancelled = True
        
    def modify(self, key: str, value: Any) -> None:
        """Modify data for subsequent handlers."""
        if self.modified_data is None:
            self.modified_data = dict(self.data)
        self.modified_data[key] = value
        
    def get_data(self) -> dict:
        """Get the current data (original or modified)."""
        return self.modified_data if self.modified_data else self.data


@dataclass
class HookResult:
    """Result from hook execution."""
    hook_point: HookPoint
    handlers_executed: int
    cancelled: bool
    errors: list[str]
    execution_time_ms: float
    final_data: dict


# ─────────────────────────────────────────────────────────────────────────────
# HOOK HANDLER TYPES
# ─────────────────────────────────────────────────────────────────────────────

# Sync handler
SyncHandler = Callable[[HookEvent], None]

# Async handler
AsyncHandler = Callable[[HookEvent], Awaitable[None]]

# Either type
Handler = Union[SyncHandler, AsyncHandler]


@dataclass
class RegisteredHook:
    """A registered hook handler."""
    id: str
    hook_point: HookPoint
    handler: Handler
    priority: HookPriority = HookPriority.NORMAL
    enabled: bool = True
    description: Optional[str] = None
    agent_filter: Optional[str] = None  # Only trigger for specific agent
    once: bool = False  # Run only once then unregister
    
    # Execution stats
    invocation_count: int = 0
    last_invoked: Optional[str] = None
    total_time_ms: float = 0.0


# ─────────────────────────────────────────────────────────────────────────────
# HOOK REGISTRY
# ─────────────────────────────────────────────────────────────────────────────

class HookRegistry:
    """
    Central registry for all hooks in the constellation.
    
    Provides:
    - Hook registration and unregistration
    - Priority-ordered execution
    - Async and sync handler support
    - Agent-specific filtering
    - Execution statistics
    """
    
    def __init__(self):
        self._hooks: dict[HookPoint, list[RegisteredHook]] = {
            hp: [] for hp in HookPoint
        }
        self._hook_counter = 0
        self._global_enabled = True
        
    def register(
        self,
        hook_point: HookPoint,
        handler: Handler,
        priority: HookPriority = HookPriority.NORMAL,
        description: Optional[str] = None,
        agent_filter: Optional[str] = None,
        once: bool = False,
    ) -> str:
        """
        Register a hook handler.
        
        Args:
            hook_point: When to trigger this hook
            handler: Function to call
            priority: Execution order (lower = earlier)
            description: Human-readable description
            agent_filter: Only trigger for specific agent
            once: Unregister after first execution
            
        Returns:
            Hook ID for later management
        """
        self._hook_counter += 1
        hook_id = f"hook_{self._hook_counter:05d}"
        
        registered = RegisteredHook(
            id=hook_id,
            hook_point=hook_point,
            handler=handler,
            priority=priority,
            description=description,
            agent_filter=agent_filter,
            once=once,
        )
        
        self._hooks[hook_point].append(registered)
        
        # Sort by priority
        self._hooks[hook_point].sort(key=lambda h: h.priority.value)
        
        logger.debug(f"Registered hook {hook_id} for {hook_point.value}")
        return hook_id
        
    def unregister(self, hook_id: str) -> bool:
        """Unregister a hook by ID."""
        for hooks in self._hooks.values():
            for i, hook in enumerate(hooks):
                if hook.id == hook_id:
                    hooks.pop(i)
                    logger.debug(f"Unregistered hook {hook_id}")
                    return True
        return False
        
    def enable(self, hook_id: str) -> bool:
        """Enable a hook."""
        hook = self._find_hook(hook_id)
        if hook:
            hook.enabled = True
            return True
        return False
        
    def disable(self, hook_id: str) -> bool:
        """Disable a hook temporarily."""
        hook = self._find_hook(hook_id)
        if hook:
            hook.enabled = False
            return True
        return False
        
    async def trigger(
        self,
        hook_point: HookPoint,
        agent_slug: Optional[str] = None,
        session_id: Optional[str] = None,
        task_id: Optional[str] = None,
        data: Optional[dict] = None,
        metadata: Optional[dict] = None,
    ) -> HookResult:
        """
        Trigger all hooks for a hook point.
        
        Executes handlers in priority order, allowing cancellation
        and data modification.
        """
        import time
        
        if not self._global_enabled:
            return HookResult(
                hook_point=hook_point,
                handlers_executed=0,
                cancelled=False,
                errors=[],
                execution_time_ms=0.0,
                final_data=data or {},
            )
            
        start = time.perf_counter()
        
        event = HookEvent(
            hook_point=hook_point,
            agent_slug=agent_slug,
            session_id=session_id,
            task_id=task_id,
            data=data or {},
            metadata=metadata or {},
        )
        
        handlers_executed = 0
        errors = []
        to_remove = []
        
        for hook in self._hooks[hook_point]:
            # Skip disabled hooks
            if not hook.enabled:
                continue
                
            # Check agent filter
            if hook.agent_filter and hook.agent_filter != agent_slug:
                continue
                
            try:
                # Execute handler
                hook_start = time.perf_counter()
                
                if asyncio.iscoroutinefunction(hook.handler):
                    await hook.handler(event)
                else:
                    hook.handler(event)
                    
                hook_time = (time.perf_counter() - hook_start) * 1000
                
                # Update stats
                hook.invocation_count += 1
                hook.last_invoked = datetime.now(timezone.utc).isoformat()
                hook.total_time_ms += hook_time
                
                handlers_executed += 1
                
                # Mark for removal if one-time
                if hook.once:
                    to_remove.append(hook.id)
                    
            except Exception as e:
                error_msg = f"Hook {hook.id} error: {str(e)}"
                errors.append(error_msg)
                logger.error(error_msg, exc_info=True)
                
            # Stop if cancelled
            if event.cancelled:
                break
                
        # Remove one-time hooks
        for hook_id in to_remove:
            self.unregister(hook_id)
            
        total_time = (time.perf_counter() - start) * 1000
        
        return HookResult(
            hook_point=hook_point,
            handlers_executed=handlers_executed,
            cancelled=event.cancelled,
            errors=errors,
            execution_time_ms=total_time,
            final_data=event.get_data(),
        )
        
    def trigger_sync(
        self,
        hook_point: HookPoint,
        **kwargs,
    ) -> HookResult:
        """Synchronous trigger for non-async contexts."""
        return asyncio.get_event_loop().run_until_complete(
            self.trigger(hook_point, **kwargs)
        )
        
    def get_hooks(
        self,
        hook_point: Optional[HookPoint] = None,
    ) -> list[RegisteredHook]:
        """Get registered hooks, optionally filtered by point."""
        if hook_point:
            return list(self._hooks[hook_point])
        return [h for hooks in self._hooks.values() for h in hooks]
        
    def get_stats(self) -> dict:
        """Get hook execution statistics."""
        stats = {}
        for hook_point, hooks in self._hooks.items():
            if hooks:
                stats[hook_point.value] = {
                    "count": len(hooks),
                    "total_invocations": sum(h.invocation_count for h in hooks),
                    "total_time_ms": sum(h.total_time_ms for h in hooks),
                }
        return stats
        
    def pause_all(self) -> None:
        """Pause all hook execution globally."""
        self._global_enabled = False
        
    def resume_all(self) -> None:
        """Resume hook execution globally."""
        self._global_enabled = True
        
    def _find_hook(self, hook_id: str) -> Optional[RegisteredHook]:
        """Find a hook by ID."""
        for hooks in self._hooks.values():
            for hook in hooks:
                if hook.id == hook_id:
                    return hook
        return None


# ─────────────────────────────────────────────────────────────────────────────
# HOOK DECORATORS
# ─────────────────────────────────────────────────────────────────────────────

# Global registry instance
_registry = HookRegistry()


def get_hook_registry() -> HookRegistry:
    """Get the global hook registry."""
    return _registry


def on(
    hook_point: HookPoint,
    priority: HookPriority = HookPriority.NORMAL,
    agent_filter: Optional[str] = None,
    once: bool = False,
):
    """
    Decorator to register a function as a hook handler.
    
    Usage:
        @on(HookPoint.AGENT_START)
        def my_handler(event: HookEvent):
            print(f"Agent started: {event.agent_slug}")
    """
    def decorator(func: Handler) -> Handler:
        _registry.register(
            hook_point=hook_point,
            handler=func,
            priority=priority,
            description=func.__doc__,
            agent_filter=agent_filter,
            once=once,
        )
        return func
    return decorator


def before_tool(tool_name: Optional[str] = None):
    """Decorator for before-tool hooks."""
    def decorator(func: Handler) -> Handler:
        async def wrapper(event: HookEvent):
            if tool_name is None or event.data.get("tool_name") == tool_name:
                if asyncio.iscoroutinefunction(func):
                    await func(event)
                else:
                    func(event)
                    
        _registry.register(
            hook_point=HookPoint.TOOL_BEFORE,
            handler=wrapper,
            description=f"Before tool: {tool_name or 'any'}",
        )
        return func
    return decorator


def after_tool(tool_name: Optional[str] = None):
    """Decorator for after-tool hooks."""
    def decorator(func: Handler) -> Handler:
        async def wrapper(event: HookEvent):
            if tool_name is None or event.data.get("tool_name") == tool_name:
                if asyncio.iscoroutinefunction(func):
                    await func(event)
                else:
                    func(event)
                    
        _registry.register(
            hook_point=HookPoint.TOOL_AFTER,
            handler=wrapper,
            description=f"After tool: {tool_name or 'any'}",
        )
        return func
    return decorator


# ─────────────────────────────────────────────────────────────────────────────
# HOOK MIDDLEWARE
# ─────────────────────────────────────────────────────────────────────────────

class HookMiddleware:
    """
    Middleware for integrating hooks into agent operations.
    
    Wraps agent methods to automatically trigger hooks
    at appropriate points.
    """
    
    def __init__(self, registry: Optional[HookRegistry] = None):
        self.registry = registry or get_hook_registry()
        
    def wrap_agent(self, agent_func: Callable) -> Callable:
        """Wrap an agent function with lifecycle hooks."""
        @wraps(agent_func)
        async def wrapper(
            agent_slug: str,
            *args,
            session_id: Optional[str] = None,
            task_id: Optional[str] = None,
            **kwargs,
        ):
            # Trigger start hook
            start_result = await self.registry.trigger(
                HookPoint.AGENT_START,
                agent_slug=agent_slug,
                session_id=session_id,
                task_id=task_id,
                data={"args": args, "kwargs": kwargs},
            )
            
            if start_result.cancelled:
                return None
                
            try:
                # Execute agent
                if asyncio.iscoroutinefunction(agent_func):
                    result = await agent_func(agent_slug, *args, **kwargs)
                else:
                    result = agent_func(agent_slug, *args, **kwargs)
                    
                # Trigger complete hook
                await self.registry.trigger(
                    HookPoint.AGENT_COMPLETE,
                    agent_slug=agent_slug,
                    session_id=session_id,
                    task_id=task_id,
                    data={"result": result},
                )
                
                return result
                
            except Exception as e:
                # Trigger error hook
                await self.registry.trigger(
                    HookPoint.AGENT_ERROR,
                    agent_slug=agent_slug,
                    session_id=session_id,
                    task_id=task_id,
                    data={"error": str(e), "error_type": type(e).__name__},
                )
                raise
                
        return wrapper
        
    def wrap_tool(self, tool_func: Callable, tool_name: str) -> Callable:
        """Wrap a tool function with before/after hooks."""
        @wraps(tool_func)
        async def wrapper(*args, **kwargs):
            # Get context from kwargs if available
            agent_slug = kwargs.pop("_agent_slug", None)
            session_id = kwargs.pop("_session_id", None)
            
            # Trigger before hook
            before_result = await self.registry.trigger(
                HookPoint.TOOL_BEFORE,
                agent_slug=agent_slug,
                session_id=session_id,
                data={
                    "tool_name": tool_name,
                    "args": args,
                    "kwargs": kwargs,
                },
            )
            
            if before_result.cancelled:
                return None
                
            # Use potentially modified args
            final_kwargs = before_result.final_data.get("kwargs", kwargs)
            
            try:
                # Execute tool
                if asyncio.iscoroutinefunction(tool_func):
                    result = await tool_func(*args, **final_kwargs)
                else:
                    result = tool_func(*args, **final_kwargs)
                    
                # Trigger after hook
                await self.registry.trigger(
                    HookPoint.TOOL_AFTER,
                    agent_slug=agent_slug,
                    session_id=session_id,
                    data={
                        "tool_name": tool_name,
                        "result": result,
                    },
                )
                
                return result
                
            except Exception as e:
                await self.registry.trigger(
                    HookPoint.TOOL_ERROR,
                    agent_slug=agent_slug,
                    session_id=session_id,
                    data={
                        "tool_name": tool_name,
                        "error": str(e),
                    },
                )
                raise
                
        return wrapper
