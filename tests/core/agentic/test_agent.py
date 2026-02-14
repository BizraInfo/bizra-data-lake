"""Tests for core.agentic.agent -- Autonomous Agent system.

Covers:
- AgentState, TaskPriority, TaskStatus enums
- AgentTask: instantiation, is_ready, serialization
- AgentThought, AgentAction data classes
- AutonomousAgent / SimpleAgent: tools, tasks, Ihsan validation
"""

from datetime import datetime, timezone

import pytest

from core.agentic.agent import (
    AgentAction,
    AgentState,
    AgentTask,
    AgentThought,
    SimpleAgent,
    TaskPriority,
    TaskStatus,
)


# ---------------------------------------------------------------------------
# ENUM TESTS
# ---------------------------------------------------------------------------


class TestEnums:

    def test_agent_states(self):
        expected = {"idle", "planning", "executing", "waiting", "reflecting", "error", "halted"}
        actual = {s.value for s in AgentState}
        assert actual == expected

    def test_task_priorities(self):
        expected = {"critical", "high", "normal", "low", "background"}
        actual = {p.value for p in TaskPriority}
        assert actual == expected

    def test_task_statuses(self):
        expected = {"pending", "in_progress", "completed", "failed", "cancelled"}
        actual = {s.value for s in TaskStatus}
        assert actual == expected


# ---------------------------------------------------------------------------
# AgentTask TESTS
# ---------------------------------------------------------------------------


class TestAgentTask:

    def test_default_values(self):
        task = AgentTask(name="Test Task")
        assert task.priority == TaskPriority.NORMAL
        assert task.status == TaskStatus.PENDING
        assert task.max_retries == 3
        assert task.retries == 0

    def test_is_ready_no_dependencies(self):
        task = AgentTask(name="Independent")
        assert task.is_ready(set()) is True

    def test_is_ready_dependencies_met(self):
        task = AgentTask(name="Dependent", depends_on={"task_a", "task_b"})
        assert task.is_ready({"task_a", "task_b", "task_c"}) is True

    def test_is_ready_dependencies_not_met(self):
        task = AgentTask(name="Blocked", depends_on={"task_a", "task_b"})
        assert task.is_ready({"task_a"}) is False

    def test_to_dict(self):
        task = AgentTask(
            name="Serialize Test",
            description="Test serialization",
            priority=TaskPriority.HIGH,
        )
        d = task.to_dict()
        assert d["name"] == "Serialize Test"
        assert d["priority"] == "high"
        assert d["status"] == "pending"
        assert "created_at" in d

    def test_to_dict_with_dates(self):
        task = AgentTask(name="Dated")
        task.started_at = datetime.now(timezone.utc)
        task.completed_at = datetime.now(timezone.utc)
        d = task.to_dict()
        assert d["started_at"] is not None
        assert d["completed_at"] is not None


# ---------------------------------------------------------------------------
# AgentThought & AgentAction TESTS
# ---------------------------------------------------------------------------


class TestAgentThought:

    def test_instantiation(self):
        thought = AgentThought(
            content="Analyzing the problem",
            thought_type="observation",
            confidence=0.9,
        )
        assert thought.content == "Analyzing the problem"
        assert thought.confidence == 0.9


class TestAgentAction:

    def test_success_action(self):
        action = AgentAction(
            tool="search",
            input={"query": "test"},
            output=["result1"],
            success=True,
            duration_ms=50.0,
        )
        assert action.success is True
        assert action.error is None

    def test_failure_action(self):
        action = AgentAction(
            tool="search",
            input={"query": "test"},
            success=False,
            error="Connection timeout",
        )
        assert action.success is False


# ---------------------------------------------------------------------------
# SimpleAgent TESTS
# ---------------------------------------------------------------------------


class TestSimpleAgent:

    @pytest.fixture
    def agent(self):
        return SimpleAgent(name="TestAgent")

    def test_initial_state(self, agent):
        assert agent.state == AgentState.IDLE
        assert agent.name == "TestAgent"

    def test_register_tool(self, agent):
        agent.register_tool("echo", lambda text="": text, "Echo tool")
        assert "echo" in agent._tools

    def test_add_task_and_sort(self, agent):
        low = AgentTask(name="Low", priority=TaskPriority.LOW)
        high = AgentTask(name="High", priority=TaskPriority.HIGH)
        agent.add_task(low)
        agent.add_task(high)
        assert len(agent._task_queue) == 2

    def test_get_status(self, agent):
        status = agent.get_status()
        assert status["name"] == "TestAgent"
        assert status["state"] == "idle"
        assert status["queued_tasks"] == 0
        assert status["success_rate"] == 0.0


@pytest.mark.timeout(60)
class TestAgentIhsanValidation:

    @pytest.mark.asyncio
    async def test_validate_safe_action(self):
        agent = SimpleAgent(name="SafeAgent")
        result = await agent._validate_ihsan("analyze the data")
        assert result is True

    @pytest.mark.asyncio
    @pytest.mark.parametrize("dangerous_action", [
        "delete all files",
        "drop table users",
        "rm -rf /",
        "format the disk",
        "shutdown the server",
        "reboot now",
    ])
    async def test_validate_blocks_dangerous_actions(self, dangerous_action):
        agent = SimpleAgent(name="SafeAgent")
        result = await agent._validate_ihsan(dangerous_action)
        assert result is False

    @pytest.mark.asyncio
    async def test_validate_with_llm_fn(self):
        # LLM that always says YES
        agent = SimpleAgent(name="LLMAgent", llm_fn=lambda prompt: "YES")
        result = await agent._validate_ihsan("safe action")
        assert result is True

    @pytest.mark.asyncio
    async def test_validate_with_llm_fn_rejection(self):
        # LLM that always says NO
        agent = SimpleAgent(name="LLMAgent", llm_fn=lambda prompt: "NO")
        result = await agent._validate_ihsan("any action")
        assert result is False


@pytest.mark.timeout(60)
class TestAgentExecution:

    @pytest.mark.asyncio
    async def test_plan(self):
        agent = SimpleAgent(name="Planner")
        task = AgentTask(name="Test", description="Do something")
        plan = await agent.plan(task)
        assert len(plan) == 1
        assert plan[0]["action"] == "execute"

    @pytest.mark.asyncio
    async def test_execute_step_no_llm(self):
        agent = SimpleAgent(name="Executor")
        success, result = await agent.execute_step({"action": "execute", "task": "test"})
        assert success is True

    @pytest.mark.asyncio
    async def test_execute_step_with_tool(self):
        agent = SimpleAgent(name="ToolUser")
        agent.register_tool("echo", lambda text="": f"echoed: {text}")
        success, result = await agent.execute_step(
            {"action": "echo", "input": {"text": "hello"}}
        )
        assert success is True
        assert result == "echoed: hello"

    @pytest.mark.asyncio
    async def test_run_task_success(self):
        agent = SimpleAgent(name="Runner")
        task = AgentTask(name="SimpleTask", description="Complete this")
        success = await agent.run_task(task)
        assert success is True
        assert task.status == TaskStatus.COMPLETED
        assert agent.state == AgentState.IDLE

    @pytest.mark.asyncio
    async def test_use_tool_unknown(self):
        agent = SimpleAgent(name="ToolTester")
        success, result = await agent._use_tool("nonexistent", {})
        assert success is False
        assert "not found" in result.lower()
