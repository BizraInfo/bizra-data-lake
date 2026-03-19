"""Tests for core.agentic.exceptions — Constitutional Exception Hierarchy.

Covers:
- AgentException base: message, context, recoverable, str repr
- Tier 1 (Constitutional): ConstitutionalException, LLMValidationException
- Tier 2 (Tool): ToolException, ToolNotFoundException, ToolTimeoutException
- Tier 3 (Task): TaskException, PlanningException, ExecutionException
- Cross-cutting: NetworkException
- Inheritance tree integrity
- Context propagation through exception chain

Standing on Giants: Meyer (DbC, 1986) · Liskov (LSP, 1987)
"""

import pytest

from core.agentic.exceptions import (
    AgentException,
    ConstitutionalException,
    ExecutionException,
    LLMValidationException,
    NetworkException,
    PlanningException,
    TaskException,
    ToolException,
    ToolNotFoundException,
    ToolTimeoutException,
)

# ═══════════════════════════════════════════════════════════════════════════
# AgentException BASE
# ═══════════════════════════════════════════════════════════════════════════


class TestAgentException:

    def test_base_defaults(self):
        exc = AgentException("test error")
        assert exc.message == "test error"
        assert exc.context == {}
        assert exc.recoverable is False

    def test_base_with_context(self):
        ctx = {"key": "value", "count": 42}
        exc = AgentException("with context", context=ctx, recoverable=True)
        assert exc.context == ctx
        assert exc.recoverable is True

    def test_str_without_context(self):
        exc = AgentException("bare message")
        assert str(exc) == "bare message"

    def test_str_with_context(self):
        exc = AgentException("msg", context={"a": 1})
        result = str(exc)
        assert "msg" in result
        assert "a" in result

    def test_is_exception(self):
        exc = AgentException("test")
        assert isinstance(exc, Exception)

    def test_args_passthrough(self):
        exc = AgentException("test")
        assert exc.args == ("test",)


# ═══════════════════════════════════════════════════════════════════════════
# TIER 1: Constitutional Violations (fail-closed)
# ═══════════════════════════════════════════════════════════════════════════


class TestConstitutionalException:

    def test_inherits_agent_exception(self):
        exc = ConstitutionalException("gate failed", action="delete files")
        assert isinstance(exc, AgentException)

    def test_never_recoverable(self):
        exc = ConstitutionalException("ihsan below threshold", action="risky op")
        assert exc.recoverable is False

    def test_action_in_context(self):
        exc = ConstitutionalException("blocked", action="rm -rf /")
        assert exc.context["action"] == "rm -rf /"

    def test_extra_context_kwargs(self):
        exc = ConstitutionalException(
            "gate failed",
            action="test",
            ihsan_score=0.72,
            gate_name="daughter_test",
        )
        assert exc.context["ihsan_score"] == 0.72
        assert exc.context["gate_name"] == "daughter_test"


class TestLLMValidationException:

    def test_inherits_constitutional(self):
        orig = ValueError("timeout")
        exc = LLMValidationException(
            "call failed", action="validate", original_error=orig
        )
        assert isinstance(exc, ConstitutionalException)
        assert isinstance(exc, AgentException)

    def test_never_recoverable(self):
        orig = RuntimeError("model unavailable")
        exc = LLMValidationException("fail", action="check", original_error=orig)
        assert exc.recoverable is False

    def test_wraps_original_error(self):
        orig = ConnectionError("refused")
        exc = LLMValidationException("wrapped", action="validate", original_error=orig)
        assert "ConnectionError" in exc.context["error_type"]
        assert "refused" in exc.context["original_error"]

    def test_message_prefix(self):
        orig = TimeoutError("30s")
        exc = LLMValidationException("timeout", action="check", original_error=orig)
        assert "LLM validation failed" in exc.message


# ═══════════════════════════════════════════════════════════════════════════
# TIER 2: Tool Failures (recoverable)
# ═══════════════════════════════════════════════════════════════════════════


class TestToolException:

    def test_inherits_agent_exception(self):
        exc = ToolException("tool broke", tool_name="search", tool_input={"q": "x"})
        assert isinstance(exc, AgentException)

    def test_default_recoverable(self):
        exc = ToolException("err", tool_name="echo", tool_input={})
        assert exc.recoverable is True

    def test_explicit_not_recoverable(self):
        exc = ToolException("fatal", tool_name="echo", tool_input={}, recoverable=False)
        assert exc.recoverable is False

    def test_context_has_tool_info(self):
        exc = ToolException(
            "fail",
            tool_name="calculator",
            tool_input={"expr": "2+2"},
            original_error=ZeroDivisionError("div by 0"),
        )
        assert exc.context["tool_name"] == "calculator"
        assert exc.context["tool_input"] == {"expr": "2+2"}
        assert "ZeroDivisionError" in exc.context["error_type"]


class TestToolNotFoundException:

    def test_inherits_tool_exception(self):
        exc = ToolNotFoundException("nonexistent_tool")
        assert isinstance(exc, ToolException)
        assert isinstance(exc, AgentException)

    def test_not_recoverable(self):
        exc = ToolNotFoundException("missing")
        assert exc.recoverable is False

    def test_message_contains_tool_name(self):
        exc = ToolNotFoundException("magic_tool")
        assert "magic_tool" in exc.message


class TestToolTimeoutException:

    def test_inherits_tool_exception(self):
        exc = ToolTimeoutException("slow_tool", timeout_seconds=30.0)
        assert isinstance(exc, ToolException)

    def test_is_recoverable(self):
        exc = ToolTimeoutException("api_call", timeout_seconds=10.0)
        assert exc.recoverable is True

    def test_message_contains_timeout(self):
        exc = ToolTimeoutException("fetch", timeout_seconds=5.0)
        assert "5.0s" in exc.message


# ═══════════════════════════════════════════════════════════════════════════
# TIER 3: Task Failures (retryable)
# ═══════════════════════════════════════════════════════════════════════════


class TestTaskException:

    def test_inherits_agent_exception(self):
        exc = TaskException("failed", task_id="t1", task_name="search_docs")
        assert isinstance(exc, AgentException)

    def test_default_recoverable(self):
        exc = TaskException("err", task_id="t2", task_name="analyze")
        assert exc.recoverable is True

    def test_context_has_task_info(self):
        exc = TaskException(
            "blew up",
            task_id="t3",
            task_name="process_data",
            original_error=FileNotFoundError("data.csv"),
        )
        assert exc.context["task_id"] == "t3"
        assert exc.context["task_name"] == "process_data"
        assert "FileNotFoundError" in exc.context["error_type"]


class TestPlanningException:

    def test_inherits_task_exception(self):
        orig = ValueError("no valid plan")
        exc = PlanningException(task_id="p1", task_name="research", original_error=orig)
        assert isinstance(exc, TaskException)
        assert isinstance(exc, AgentException)

    def test_is_recoverable(self):
        orig = RuntimeError("planning timeout")
        exc = PlanningException(task_id="p2", task_name="plan", original_error=orig)
        assert exc.recoverable is True

    def test_message_contains_task_name(self):
        orig = KeyError("missing key")
        exc = PlanningException(
            task_id="p3", task_name="design_review", original_error=orig
        )
        assert "design_review" in exc.message


class TestExecutionException:

    def test_inherits_task_exception(self):
        orig = OSError("disk full")
        exc = ExecutionException(
            task_id="e1",
            task_name="write_file",
            step="save",
            original_error=orig,
        )
        assert isinstance(exc, TaskException)

    def test_step_in_context(self):
        orig = PermissionError("denied")
        exc = ExecutionException(
            task_id="e2",
            task_name="deploy",
            step="chmod",
            original_error=orig,
        )
        assert exc.context["step"] == "chmod"

    def test_message_contains_step(self):
        orig = RuntimeError("oops")
        exc = ExecutionException(
            task_id="e3",
            task_name="build",
            step="compile",
            original_error=orig,
        )
        assert "compile" in exc.message


# ═══════════════════════════════════════════════════════════════════════════
# CROSS-CUTTING: NetworkException
# ═══════════════════════════════════════════════════════════════════════════


class TestNetworkException:

    def test_inherits_agent_exception(self):
        orig = ConnectionRefusedError("port closed")
        exc = NetworkException("refused", operation="connect", original_error=orig)
        assert isinstance(exc, AgentException)

    def test_is_recoverable(self):
        orig = TimeoutError("timed out")
        exc = NetworkException("timeout", operation="fetch", original_error=orig)
        assert exc.recoverable is True

    def test_context_has_operation(self):
        orig = OSError("unreachable")
        exc = NetworkException("down", operation="healthcheck", original_error=orig)
        assert exc.context["operation"] == "healthcheck"

    def test_message_contains_operation(self):
        orig = ConnectionError("reset")
        exc = NetworkException("reset", operation="send_message", original_error=orig)
        assert "send_message" in exc.message


# ═══════════════════════════════════════════════════════════════════════════
# INHERITANCE TREE INTEGRITY
# ═══════════════════════════════════════════════════════════════════════════


class TestInheritanceTree:
    """Verify the 3-tier hierarchy is correct."""

    def test_tier1_is_not_recoverable(self):
        """Constitutional exceptions must NEVER be recoverable."""
        exc = ConstitutionalException("test", action="test")
        assert exc.recoverable is False

    def test_tier2_defaults_recoverable(self):
        """Tool exceptions should default to recoverable."""
        exc = ToolException("test", tool_name="t", tool_input={})
        assert exc.recoverable is True

    def test_tier3_defaults_recoverable(self):
        """Task exceptions should default to recoverable."""
        exc = TaskException("test", task_id="t1", task_name="test")
        assert exc.recoverable is True

    def test_all_are_agent_exceptions(self):
        """All exception types must descend from AgentException."""
        types = [
            ConstitutionalException("t", action="t"),
            ToolException("t", tool_name="t", tool_input={}),
            ToolNotFoundException("t"),
            ToolTimeoutException("t", timeout_seconds=1.0),
            TaskException("t", task_id="t", task_name="t"),
            NetworkException("t", operation="t", original_error=RuntimeError("t")),
        ]
        for exc in types:
            assert isinstance(
                exc, AgentException
            ), f"{type(exc).__name__} is not AgentException"

    def test_can_catch_by_tier(self):
        """Catching by tier base class captures all subtypes."""
        tool_not_found = ToolNotFoundException("x")
        tool_timeout = ToolTimeoutException("x", timeout_seconds=1.0)

        caught = []
        for exc in [tool_not_found, tool_timeout]:
            try:
                raise exc
            except ToolException as e:
                caught.append(e)

        assert len(caught) == 2

    def test_planning_caught_by_task_exception(self):
        orig = ValueError("bad plan")
        planning = PlanningException(task_id="p", task_name="p", original_error=orig)
        try:
            raise planning
        except TaskException:
            pass  # Expected

    def test_llm_validation_caught_by_constitutional(self):
        orig = RuntimeError("model error")
        llm = LLMValidationException("fail", action="check", original_error=orig)
        try:
            raise llm
        except ConstitutionalException:
            pass  # Expected
