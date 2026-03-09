"""
Agent Exception Hierarchy — Constitutional Fail-Closed Design

Standing on Giants: Meyer (DbC, 1986) · Liskov (LSP, 1987) · Al-Ghazali (Ihsān, 1095)

Exception taxonomy for autonomous agents with constitutional constraints:
- Constitutional violations → fail-closed
- Tool failures → recoverable  
- Task failures → retryable

Each exception type carries structured context for SNR-optimal error handling.
"""

from typing import Any, Dict, Optional


# ══════════════════════════════════════════════════════════════════════════
# Base Agent Exception
# ══════════════════════════════════════════════════════════════════════════


class AgentException(Exception):
    """Base exception for all agent failures."""

    def __init__(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
        recoverable: bool = False,
    ):
        super().__init__(message)
        self.message = message
        self.context = context or {}
        self.recoverable = recoverable

    def __str__(self) -> str:
        ctx_str = f" [{self.context}]" if self.context else ""
        return f"{self.message}{ctx_str}"


# ══════════════════════════════════════════════════════════════════════════
# Tier 1: Constitutional Violations (CRITICAL — fail-closed)
# ══════════════════════════════════════════════════════════════════════════


class ConstitutionalException(AgentException):
    """Exception during constitutional validation (Ihsān gate)."""

    def __init__(self, message: str, action: str, **context: Any):
        super().__init__(
            message=message,
            context={"action": action, **context},
            recoverable=False,  # Constitutional failures are NOT recoverable
        )


class LLMValidationException(ConstitutionalException):
    """LLM call failed during constitutional check."""

    def __init__(self, message: str, action: str, original_error: Exception):
        super().__init__(
            message=f"LLM validation failed: {message}",
            action=action,
            original_error=str(original_error),
            error_type=type(original_error).__name__,
        )


# ══════════════════════════════════════════════════════════════════════════
# Tier 2: Tool Execution Failures (HIGH — often recoverable)
# ══════════════════════════════════════════════════════════════════════════


class ToolException(AgentException):
    """Exception during tool execution."""

    def __init__(
        self,
        message: str,
        tool_name: str,
        tool_input: Dict[str, Any],
        original_error: Optional[Exception] = None,
        recoverable: bool = True,
    ):
        super().__init__(
            message=message,
            context={
                "tool_name": tool_name,
                "tool_input": tool_input,
                "original_error": str(original_error) if original_error else None,
                "error_type": (
                    type(original_error).__name__ if original_error else "Unknown"
                ),
            },
            recoverable=recoverable,
        )


class ToolNotFoundException(ToolException):
    """Requested tool not registered."""

    def __init__(self, tool_name: str):
        super().__init__(
            message=f"Tool not found: {tool_name}",
            tool_name=tool_name,
            tool_input={},
            recoverable=False,  # Missing tool is not recoverable
        )


class ToolTimeoutException(ToolException):
    """Tool execution exceeded timeout."""

    def __init__(self, tool_name: str, timeout_seconds: float):
        super().__init__(
            message=f"Tool timeout after {timeout_seconds}s",
            tool_name=tool_name,
            tool_input={},
            recoverable=True,  # Timeout can be retried
        )


# ══════════════════════════════════════════════════════════════════════════
# Tier 3: Task Execution Failures (MEDIUM — retryable)
# ══════════════════════════════════════════════════════════════════════════


class TaskException(AgentException):
    """Exception during task execution."""

    def __init__(
        self,
        message: str,
        task_id: str,
        task_name: str,
        original_error: Optional[Exception] = None,
        recoverable: bool = True,
    ):
        super().__init__(
            message=message,
            context={
                "task_id": task_id,
                "task_name": task_name,
                "original_error": str(original_error) if original_error else None,
                "error_type": (
                    type(original_error).__name__ if original_error else "Unknown"
                ),
            },
            recoverable=recoverable,
        )


class PlanningException(TaskException):
    """Exception during task planning phase."""

    def __init__(self, task_id: str, task_name: str, original_error: Exception):
        super().__init__(
            message=f"Planning failed for task '{task_name}'",
            task_id=task_id,
            task_name=task_name,
            original_error=original_error,
            recoverable=True,
        )


class ExecutionException(TaskException):
    """Exception during task execution phase."""

    def __init__(
        self,
        task_id: str,
        task_name: str,
        step: str,
        original_error: Exception,
    ):
        super().__init__(
            message=f"Execution failed at step '{step}'",
            task_id=task_id,
            task_name=task_name,
            original_error=original_error,
            recoverable=True,
        )
        self.context["step"] = step


# ══════════════════════════════════════════════════════════════════════════
# Network & I/O Exceptions (cross-cutting)
# ══════════════════════════════════════════════════════════════════════════


class NetworkException(AgentException):
    """Network-related failures (httpx, socket, DNS)."""

    def __init__(self, message: str, operation: str, original_error: Exception):
        super().__init__(
            message=f"Network error during {operation}: {message}",
            context={
                "operation": operation,
                "original_error": str(original_error),
                "error_type": type(original_error).__name__,
            },
            recoverable=True,  # Network errors are transient
        )
