"""
╔══════════════════════════════════════════════════════════════════════════════╗
║   BIZRA SKILLS — Skill Router & Invocation Engine                           ║
╠══════════════════════════════════════════════════════════════════════════════╣
║   Routes skill invocations to appropriate agents with FATE gate validation. ║
╚══════════════════════════════════════════════════════════════════════════════╝

Standing on Giants:
- Martin Fowler (2003): Enterprise Integration Patterns (routing)
- Alistair Cockburn (2005): Hexagonal Architecture (adapters)
- Anthropic (2023): Constitutional AI (FATE gate)
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD

from .mcp_bridge import MCPBridge, MCPPermission
from .registry import SkillRegistry, SkillStatus, get_skill_registry

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# INVOCATION RESULT
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class SkillInvocationResult:
    """
    Result of a skill invocation.

    Contains success status, output, metrics, and audit trail.
    """

    # Status
    success: bool
    skill_name: str

    # Output
    output: Any = None
    error: Optional[str] = None

    # Metrics
    duration_ms: float = 0.0
    token_count: int = 0

    # FATE validation
    fate_score: float = 0.0
    ihsan_passed: bool = False

    # Audit
    execution_id: str = ""
    started_at: str = ""
    completed_at: str = ""
    agent_used: str = ""
    tools_used: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict."""
        return {
            "success": self.success,
            "skill_name": self.skill_name,
            "output": (
                self.output
                if not isinstance(self.output, Exception)
                else str(self.output)
            ),
            "error": self.error,
            "duration_ms": self.duration_ms,
            "token_count": self.token_count,
            "fate_score": self.fate_score,
            "ihsan_passed": self.ihsan_passed,
            "execution_id": self.execution_id,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "agent_used": self.agent_used,
            "tools_used": self.tools_used,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SKILL ROUTER
# ═══════════════════════════════════════════════════════════════════════════════


class SkillRouter:
    """
    Routes skill invocations to appropriate execution paths.

    Responsibilities:
    - Resolve skill → agent mapping
    - Validate FATE gate before execution
    - Track invocation metrics
    - Provide audit trail
    """

    def __init__(
        self,
        registry: Optional[SkillRegistry] = None,
        mcp_bridge: Optional[MCPBridge] = None,
        ihsan_threshold: float = UNIFIED_IHSAN_THRESHOLD,
    ):
        """
        Initialize router.

        Args:
            registry: Skill registry (uses global if None)
            mcp_bridge: MCP bridge for tool resolution
            ihsan_threshold: Minimum Ihsān score for invocation
        """
        self.registry = registry or get_skill_registry()
        self.mcp_bridge = mcp_bridge or MCPBridge()
        self.ihsan_threshold = ihsan_threshold

        # Agent handlers: agent_name -> handler function
        self._handlers: Dict[str, Callable] = {}

        # Invocation history (for audit)
        self._history: List[SkillInvocationResult] = []
        self._max_history = 100

        # Statistics
        self._total_invocations = 0
        self._success_count = 0
        self._blocked_count = 0

    def register_handler(self, agent_name: str, handler: Callable):
        """
        Register a handler for an agent.

        Args:
            agent_name: Name of the agent (e.g., "sovereign-coder")
            handler: Async callable that executes skills for this agent
        """
        self._handlers[agent_name] = handler
        logger.info(f"Registered handler for agent: {agent_name}")

    def resolve_agent(self, skill_name: str) -> Optional[str]:
        """
        Resolve which agent should handle a skill.

        Args:
            skill_name: Name of the skill

        Returns:
            Agent name or None if skill not found
        """
        skill = self.registry.get(skill_name)
        if not skill:
            return None
        return skill.manifest.agent

    def validate_ihsan(self, ihsan_score: float) -> bool:
        """Check if Ihsān score meets threshold."""
        return ihsan_score >= self.ihsan_threshold

    def validate_permissions(
        self,
        skill_name: str,
        allowed_permissions: List[MCPPermission],
    ) -> bool:
        """
        Check if skill's required permissions are allowed.

        Args:
            skill_name: Skill to check
            allowed_permissions: Permissions the caller has

        Returns:
            True if all required permissions are allowed
        """
        required = self.mcp_bridge.get_permissions(skill_name)
        return all(p in allowed_permissions for p in required)

    def validate_tools(self, skill_name: str) -> bool:
        """Check if all required tools are available."""
        return self.mcp_bridge.can_execute(skill_name)

    async def invoke(
        self,
        skill_name: str,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
        ihsan_score: float = 1.0,
        allowed_permissions: Optional[List[MCPPermission]] = None,
    ) -> SkillInvocationResult:
        """
        Invoke a skill.

        Args:
            skill_name: Name of the skill to invoke
            inputs: Input parameters for the skill
            context: Optional execution context
            ihsan_score: Current Ihsān score
            allowed_permissions: Permissions the caller has

        Returns:
            SkillInvocationResult with output or error
        """
        import uuid

        execution_id = str(uuid.uuid4())[:12]
        started_at = datetime.now(timezone.utc).isoformat()
        start_time = time.perf_counter()

        self._total_invocations += 1

        # Create base result
        result = SkillInvocationResult(
            success=False,
            skill_name=skill_name,
            execution_id=execution_id,
            started_at=started_at,
        )

        try:
            # 1. Get skill from registry
            skill = self.registry.get(skill_name)
            if not skill:
                result.error = f"Skill not found: {skill_name}"
                self._blocked_count += 1
                return result

            # 2. Check skill status
            if skill.status not in (SkillStatus.AVAILABLE, SkillStatus.ACTIVE):
                result.error = f"Skill not available: {skill.status.value}"
                self._blocked_count += 1
                return result

            # 3. FATE Gate: Ihsān check
            if not self.validate_ihsan(ihsan_score):
                result.error = f"Ihsān too low: {ihsan_score} < {self.ihsan_threshold}"
                result.fate_score = ihsan_score
                self._blocked_count += 1
                return result

            result.ihsan_passed = True
            result.fate_score = ihsan_score

            # 4. Permission check
            if allowed_permissions is not None:
                if not self.validate_permissions(skill_name, allowed_permissions):
                    result.error = f"Insufficient permissions for {skill_name}"
                    self._blocked_count += 1
                    return result

            # 5. Tool availability check
            if not self.validate_tools(skill_name):
                availability = self.mcp_bridge.check_availability(skill_name)
                missing = [t for t, avail in availability.items() if not avail]
                result.error = f"Missing tools: {missing}"
                self._blocked_count += 1
                return result

            # 6. Resolve agent
            agent_name = skill.manifest.agent
            result.agent_used = agent_name

            # 7. Execute (with 60s timeout to prevent runaway handlers)
            handler = self._handlers.get(agent_name)
            if handler:
                # Use registered handler
                output = await asyncio.wait_for(
                    handler(skill, inputs, context), timeout=60.0
                )
            else:
                # No handler - return skill info for external execution
                output = {
                    "skill": skill.manifest.to_dict(),
                    "agent": agent_name,
                    "tools": self.mcp_bridge.get_all_tools(skill_name),
                    "inputs": inputs,
                    "context": context,
                    "message": f"No handler for agent '{agent_name}'. Invoke externally.",
                }

            result.success = True
            result.output = output
            result.tools_used = self.mcp_bridge.get_required_tools(skill_name)

            self._success_count += 1

            # Update skill metrics
            duration_ms = (time.perf_counter() - start_time) * 1000
            skill.record_invocation(success=True, duration_ms=duration_ms)

        except Exception as e:
            result.error = str(e)
            logger.exception(f"Skill invocation failed: {skill_name}")

            # Update skill metrics on failure
            skill = self.registry.get(skill_name)
            if skill:
                duration_ms = (time.perf_counter() - start_time) * 1000
                skill.record_invocation(success=False, duration_ms=duration_ms)

        finally:
            # Record completion
            result.completed_at = datetime.now(timezone.utc).isoformat()
            result.duration_ms = (time.perf_counter() - start_time) * 1000

            # Add to history
            self._history.append(result)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history :]

        return result

    def get_available_skills(self, ihsan_score: float = 1.0) -> List[str]:
        """
        Get list of skills available at current Ihsān level.

        Args:
            ihsan_score: Current Ihsān score

        Returns:
            List of skill names that can be invoked
        """
        available = []
        for skill in self.registry.get_all():
            if self.registry.can_invoke(skill.manifest.name, ihsan_score):
                available.append(skill.manifest.name)
        return available

    def get_skills_by_agent(self, agent_name: str) -> List[str]:
        """Get skills handled by a specific agent."""
        skills = self.registry.find_by_agent(agent_name)
        return [s.manifest.name for s in skills]

    def get_stats(self) -> Dict[str, Any]:
        """Get router statistics."""
        return {
            "total_invocations": self._total_invocations,
            "success_count": self._success_count,
            "blocked_count": self._blocked_count,
            "success_rate": self._success_count / max(self._total_invocations, 1),
            "registered_handlers": list(self._handlers.keys()),
            "history_size": len(self._history),
            "registry_stats": self.registry.get_stats(),
            "mcp_stats": self.mcp_bridge.get_usage_stats(),
        }

    def get_recent_history(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get recent invocation history."""
        return [r.to_dict() for r in self._history[-n:]]


# ═══════════════════════════════════════════════════════════════════════════════
# CONVENIENCE FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


_router: Optional[SkillRouter] = None


def get_skill_router() -> SkillRouter:
    """Get the global skill router."""
    global _router
    if _router is None:
        _router = SkillRouter()
    return _router


async def invoke_skill(
    skill_name: str,
    inputs: Dict[str, Any],
    context: Optional[Dict[str, Any]] = None,
) -> SkillInvocationResult:
    """
    Convenience function to invoke a skill.

    Uses global router with default settings.
    """
    router = get_skill_router()
    return await router.invoke(skill_name, inputs, context)
