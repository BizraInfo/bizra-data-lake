"""
Tests for BIZRA Living Ecosystem

Validates:
- Living Memory (encode, retrieve, consolidate, heal)
- Agentic System (task execution, orchestration)
- PAT Bridge (message handling, constitutional validation)
- Ecosystem integration
"""

import pytest
import asyncio
from datetime import datetime, timezone
from pathlib import Path
import tempfile

# Living Memory
from core.living_memory.core import (
    LivingMemoryCore,
    MemoryEntry,
    MemoryType,
    MemoryState,
)
from core.living_memory.proactive import (
    ProactiveRetriever,
    PredictionContext,
)
from core.living_memory.healing import (
    MemoryHealer,
    CorruptionType,
)

# Agentic System
from core.agentic.agent import (
    AutonomousAgent,
    SimpleAgent,
    AgentTask,
    AgentState,
    TaskPriority,
    TaskStatus,
)
from core.agentic.orchestrator import (
    AgentOrchestrator,
    OrchestratorState,
)

# PAT Bridge
from core.pat.bridge import (
    PATBridge,
    PATMessage,
    MessageType,
    ChannelType,
)


# ============================================================================
# LIVING MEMORY TESTS
# ============================================================================

class TestLivingMemoryCore:
    """Tests for living memory core."""

    @pytest.fixture
    def memory(self, tmp_path):
        return LivingMemoryCore(
            storage_path=tmp_path / "memory",
            max_entries=1000,
            ihsan_threshold=0.95,
        )

    @pytest.mark.asyncio
    async def test_initialize(self, memory):
        """Test memory initialization."""
        await memory.initialize()
        assert memory._initialized
        assert memory.storage_path.exists()

    @pytest.mark.asyncio
    async def test_encode_semantic(self, memory):
        """Test encoding semantic memory."""
        await memory.initialize()

        entry = await memory.encode(
            content="Machine learning is a subset of artificial intelligence.",
            memory_type=MemoryType.SEMANTIC,
            source="test",
        )

        assert entry is not None
        assert entry.memory_type == MemoryType.SEMANTIC
        assert entry.ihsan_score > 0

    @pytest.mark.asyncio
    async def test_encode_episodic(self, memory):
        """Test encoding episodic memory."""
        await memory.initialize()

        entry = await memory.encode(
            content="User asked about neural networks at 10:00 AM.",
            memory_type=MemoryType.EPISODIC,
            source="user",
        )

        assert entry is not None
        assert entry.memory_type == MemoryType.EPISODIC
        assert entry.id in memory._working_memory

    @pytest.mark.asyncio
    async def test_retrieve(self, memory):
        """Test memory retrieval."""
        await memory.initialize()

        # Encode some memories (longer content to pass quality filter)
        entry1 = await memory.encode(
            "Deep learning uses neural networks for pattern recognition and classification tasks.",
            MemoryType.SEMANTIC
        )
        entry2 = await memory.encode(
            "Machine learning algorithms learn patterns from training data to make predictions.",
            MemoryType.SEMANTIC
        )
        entry3 = await memory.encode(
            "Python is a programming language widely used for data science applications.",
            MemoryType.SEMANTIC
        )

        # Verify entries were created
        assert entry1 is not None
        assert entry2 is not None
        assert entry3 is not None

        # Retrieve all (without embedding, retrieval is based on recency/access)
        results = await memory.retrieve(
            query=None,  # Get all recent
            top_k=3,
            min_score=0.0,  # Lower threshold
        )

        assert len(results) >= 1

    @pytest.mark.asyncio
    async def test_forget(self, memory):
        """Test forgetting memory."""
        await memory.initialize()

        # Use longer content to pass quality filter
        entry = await memory.encode(
            "This is temporary information that will be forgotten soon after being stored.",
            MemoryType.WORKING
        )
        assert entry is not None

        # Soft delete
        success = await memory.forget(entry.id)
        assert success
        assert memory._memories[entry.id].state == MemoryState.DELETED

        # Hard delete
        success = await memory.forget(entry.id, hard_delete=True)
        assert success
        assert entry.id not in memory._memories

    @pytest.mark.asyncio
    async def test_consolidate(self, memory):
        """Test memory consolidation."""
        await memory.initialize()

        # Encode memories
        await memory.encode("Test memory 1", MemoryType.EPISODIC)
        await memory.encode("Test memory 2", MemoryType.SEMANTIC)

        stats = await memory.consolidate()

        assert isinstance(stats, dict)
        assert "archived" in stats

    @pytest.mark.asyncio
    async def test_quality_filter(self, memory):
        """Test that low-quality content is rejected."""
        await memory.initialize()

        # Very short content should be filtered
        entry = await memory.encode("x", MemoryType.SEMANTIC)
        assert entry is None

    @pytest.mark.asyncio
    async def test_stats(self, memory):
        """Test memory statistics."""
        await memory.initialize()

        await memory.encode("Test content for stats.", MemoryType.SEMANTIC)

        stats = memory.get_stats()

        assert stats.total_entries >= 1
        assert stats.active_entries >= 1


class TestProactiveRetriever:
    """Tests for proactive information retrieval."""

    @pytest.fixture
    def memory(self, tmp_path):
        return LivingMemoryCore(
            storage_path=tmp_path / "memory",
            max_entries=1000,
        )

    @pytest.fixture
    def retriever(self, memory):
        return ProactiveRetriever(
            memory=memory,
            max_suggestions=5,
        )

    def test_update_context(self, retriever):
        """Test context updating."""
        retriever.update_context(query="machine learning")

        assert "machine learning" == retriever._context.current_query
        assert "machine" in retriever._context.active_topics
        assert "learning" in retriever._context.active_topics

    def test_predict_next_topics(self, retriever):
        """Test topic prediction."""
        # Simulate topic sequence
        retriever.update_context(query="neural networks")
        retriever.update_context(query="deep learning")
        retriever.update_context(query="neural networks")
        retriever.update_context(query="backpropagation")

        predictions = retriever.predict_next_topics()

        assert isinstance(predictions, list)

    @pytest.mark.asyncio
    async def test_get_suggestions(self, memory, retriever):
        """Test proactive suggestions."""
        await memory.initialize()

        # Add some memories
        await memory.encode("Neural networks are powerful.", MemoryType.SEMANTIC)
        await memory.encode("Deep learning revolutionized AI.", MemoryType.SEMANTIC)

        retriever.update_context(query="machine learning")

        suggestions = await retriever.get_proactive_suggestions()

        assert isinstance(suggestions, list)


class TestMemoryHealer:
    """Tests for memory healing."""

    @pytest.fixture
    def memory(self, tmp_path):
        return LivingMemoryCore(
            storage_path=tmp_path / "memory",
            max_entries=1000,
        )

    @pytest.fixture
    def healer(self, memory):
        return MemoryHealer(
            memory=memory,
            ihsan_threshold=0.95,
            snr_threshold=0.85,
        )

    @pytest.mark.asyncio
    async def test_scan_for_corruption(self, memory, healer):
        """Test corruption scanning."""
        await memory.initialize()

        # Add healthy memory
        await memory.encode("Healthy content here.", MemoryType.SEMANTIC)

        reports = await healer.scan_for_corruption()

        assert isinstance(reports, list)

    @pytest.mark.asyncio
    async def test_optimize(self, memory, healer):
        """Test optimization pass."""
        await memory.initialize()

        # Use longer content to pass quality filter
        await memory.encode(
            "This is the first test content for memory optimization testing purposes.",
            MemoryType.SEMANTIC
        )
        await memory.encode(
            "This is the second test content for memory optimization verification.",
            MemoryType.SEMANTIC
        )

        stats = await healer.optimize()

        assert stats.entries_scanned > 0

    def test_health_report(self, memory, healer):
        """Test health report generation."""
        report = healer.get_health_report()

        assert "health_score" in report
        assert 0 <= report["health_score"] <= 1


# ============================================================================
# AGENTIC SYSTEM TESTS
# ============================================================================

class TestAutonomousAgent:
    """Tests for autonomous agents."""

    @pytest.fixture
    def agent(self):
        return SimpleAgent(
            name="TestAgent",
            ihsan_threshold=0.95,
        )

    def test_agent_creation(self, agent):
        """Test agent creation."""
        assert agent.name == "TestAgent"
        assert agent.state == AgentState.IDLE

    def test_register_tool(self, agent):
        """Test tool registration."""
        def dummy_tool(x: int) -> int:
            return x * 2

        agent.register_tool("double", dummy_tool)

        assert "double" in agent._tools

    def test_add_task(self, agent):
        """Test adding task."""
        task = AgentTask(
            name="Test Task",
            description="Do something",
            priority=TaskPriority.HIGH,
        )

        agent.add_task(task)

        assert len(agent._task_queue) == 1
        assert agent._task_queue[0].name == "Test Task"

    @pytest.mark.asyncio
    async def test_ihsan_validation(self, agent):
        """Test Ihsān constitutional validation."""
        # Safe action
        safe = await agent._validate_ihsan("Read a file")
        assert safe

        # Dangerous action
        dangerous = await agent._validate_ihsan("delete all files")
        assert not dangerous

    @pytest.mark.asyncio
    async def test_run_task(self, agent):
        """Test task execution."""
        task = AgentTask(
            name="Simple Task",
            description="Return success",
        )

        success = await agent.run_task(task)

        assert task.status in (TaskStatus.COMPLETED, TaskStatus.FAILED)

    def test_get_status(self, agent):
        """Test status retrieval."""
        status = agent.get_status()

        assert "id" in status
        assert "name" in status
        assert "state" in status


class TestAgentOrchestrator:
    """Tests for agent orchestrator."""

    @pytest.fixture
    def orchestrator(self, tmp_path):
        memory = LivingMemoryCore(storage_path=tmp_path / "memory")
        return AgentOrchestrator(
            memory=memory,
            max_agents=5,
        )

    @pytest.mark.asyncio
    async def test_initialize(self, orchestrator):
        """Test orchestrator initialization."""
        await orchestrator.initialize()

        assert orchestrator.state == OrchestratorState.RUNNING
        assert len(orchestrator._agents) >= 1

    @pytest.mark.asyncio
    async def test_register_agent(self, orchestrator):
        """Test agent registration."""
        await orchestrator.initialize()

        agent = SimpleAgent(name="CustomAgent")
        success = await orchestrator.register_agent(agent)

        assert success
        assert agent.id in orchestrator._agents

    @pytest.mark.asyncio
    async def test_submit_task(self, orchestrator):
        """Test task submission."""
        await orchestrator.initialize()

        task = await orchestrator.submit_task(
            name="Test Task",
            description="Do something",
            priority=TaskPriority.NORMAL,
        )

        assert task.id is not None
        assert task.status == TaskStatus.PENDING

    @pytest.mark.asyncio
    async def test_maintenance(self, orchestrator):
        """Test maintenance cycle."""
        await orchestrator.initialize()

        stats = await orchestrator.run_maintenance()

        assert "checked" in stats

    @pytest.mark.asyncio
    async def test_shutdown(self, orchestrator):
        """Test graceful shutdown."""
        await orchestrator.initialize()
        await orchestrator.shutdown()

        assert orchestrator.state == OrchestratorState.STOPPED


# ============================================================================
# PAT BRIDGE TESTS
# ============================================================================

class TestPATBridge:
    """Tests for PAT bridge."""

    @pytest.fixture
    def bridge(self, tmp_path):
        memory = LivingMemoryCore(storage_path=tmp_path / "memory")
        return PATBridge(
            memory=memory,
            ihsan_threshold=0.95,
        )

    def test_register_tool(self, bridge):
        """Test tool registration."""
        def echo(text: str) -> str:
            return text

        bridge.register_tool("echo", echo)

        assert "echo" in bridge._tools

    @pytest.mark.asyncio
    async def test_validate_ihsan(self, bridge):
        """Test message Ihsān validation."""
        # Safe message
        safe_msg = PATMessage(content="Hello, how are you?")
        valid = await bridge._validate_ihsan(safe_msg)
        assert valid

        # Dangerous message
        dangerous_msg = PATMessage(content="rm -rf /")
        valid = await bridge._validate_ihsan(dangerous_msg)
        assert not valid

    @pytest.mark.asyncio
    async def test_process_local(self, bridge, tmp_path):
        """Test local message processing."""
        # Initialize memory
        await bridge.memory.initialize()

        # Process without LLM
        response = await bridge.process_local("Hello")

        assert response is not None
        assert isinstance(response, str)

    def test_get_status(self, bridge):
        """Test status retrieval."""
        status = bridge.get_status()

        assert "connected" in status
        assert "total_messages" in status


# ============================================================================
# INTEGRATION TESTS
# ============================================================================

class TestLivingEcosystemIntegration:
    """Integration tests for the living ecosystem."""

    @pytest.fixture
    def mock_llm(self):
        """Simple mock LLM for testing."""
        def llm_fn(prompt: str) -> str:
            if "safe" in prompt.lower() or "evaluate" in prompt.lower():
                return "YES"
            return f"Response to: {prompt[:50]}..."
        return llm_fn

    @pytest.mark.asyncio
    async def test_memory_agent_integration(self, tmp_path, mock_llm):
        """Test memory and agent integration."""
        # Create memory
        memory = LivingMemoryCore(
            storage_path=tmp_path / "memory",
            llm_fn=mock_llm,
        )
        await memory.initialize()

        # Create orchestrator with memory
        orchestrator = AgentOrchestrator(
            memory=memory,
            llm_fn=mock_llm,
        )
        await orchestrator.initialize()

        # Submit task
        task = await orchestrator.submit_task(
            name="Learn Test",
            description="Learn something new",
        )

        assert task is not None

        # Check memory is accessible (use longer content)
        await memory.encode(
            "This is integration test content for verifying memory and agent integration works correctly.",
            MemoryType.SEMANTIC
        )
        stats = memory.get_stats()
        assert stats.total_entries >= 1

        await orchestrator.shutdown()

    @pytest.mark.asyncio
    async def test_full_ecosystem_flow(self, tmp_path, mock_llm):
        """Test full ecosystem flow."""
        # Create components
        memory = LivingMemoryCore(
            storage_path=tmp_path / "memory",
            llm_fn=mock_llm,
        )
        await memory.initialize()

        retriever = ProactiveRetriever(memory=memory)
        healer = MemoryHealer(memory=memory)

        # Encode knowledge
        await memory.encode(
            "The sky is blue during clear days.",
            MemoryType.SEMANTIC,
            importance=0.9,
        )
        await memory.encode(
            "Water freezes at 0 degrees Celsius.",
            MemoryType.SEMANTIC,
            importance=0.8,
        )

        # Update proactive context
        retriever.update_context(query="weather and temperature")

        # Retrieve
        results = await memory.retrieve(query="sky", top_k=3)
        assert len(results) > 0

        # Get suggestions
        suggestions = await retriever.get_proactive_suggestions()
        assert isinstance(suggestions, list)

        # Run healing
        stats = await healer.optimize()
        assert stats.entries_scanned > 0

        # Check health
        report = healer.get_health_report()
        assert report["health_score"] > 0.5
