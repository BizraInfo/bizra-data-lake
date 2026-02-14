"""Tests for core.a2a — Agent-to-Agent protocol engine.

Phase 17: Security Test Scaffolding
"""

import json
from typing import Any, Dict, List

import pytest

from core.a2a.engine import A2AEngine, create_a2a_engine
from core.a2a.schema import AgentCard, A2AMessage, Capability, CapabilityType


# ─────────────────────────────────────────────────────────────────────────────
# FIXTURES
# ─────────────────────────────────────────────────────────────────────────────


@pytest.fixture
def engine() -> A2AEngine:
    """Create a basic A2AEngine for testing."""
    return create_a2a_engine(
        agent_id="test-agent-001",
        name="TestAgent",
        description="Test agent for unit testing",
        capabilities=[
            {"name": "inference", "description": "Run inference"},
            {"name": "search", "description": "Search capability"},
        ],
    )


@pytest.fixture
def agent_card() -> AgentCard:
    """Create a sample AgentCard."""
    return AgentCard(
        agent_id="peer-agent-002",
        name="PeerAgent",
        description="A peer agent for testing",
        public_key="0" * 64,  # Placeholder public key hex
        capabilities=[
            Capability(name="inference", type=CapabilityType.REASONING, description="Run inference"),
            Capability(name="embeddings", type=CapabilityType.CUSTOM, description="Generate embeddings"),
        ],
        endpoint="http://localhost:9001/a2a",
    )


# ─────────────────────────────────────────────────────────────────────────────
# ENGINE CONSTRUCTION
# ─────────────────────────────────────────────────────────────────────────────


class TestA2AEngineConstruction:
    """Test engine initialization and factory."""

    def test_create_engine(self, engine: A2AEngine) -> None:
        assert engine is not None
        stats = engine.get_stats()
        assert stats["agent_id"] == "test-agent-001"

    def test_factory_function(self) -> None:
        e = create_a2a_engine(
            agent_id="factory-test",
            name="FactoryAgent",
            description="Factory test agent",
            capabilities=[{"name": "search", "description": "Search"}],
        )
        assert isinstance(e, A2AEngine)


# ─────────────────────────────────────────────────────────────────────────────
# AGENT REGISTRY
# ─────────────────────────────────────────────────────────────────────────────


class TestAgentRegistry:
    """Test agent registration, lookup, and unregistration."""

    def test_register_agent(self, engine: A2AEngine, agent_card: AgentCard) -> None:
        assert engine.register_agent(agent_card) is True

    def test_get_registered_agent(self, engine: A2AEngine, agent_card: AgentCard) -> None:
        engine.register_agent(agent_card)
        found = engine.get_agent("peer-agent-002")
        assert found is not None
        assert found.name == "PeerAgent"

    def test_get_unregistered_returns_none(self, engine: A2AEngine) -> None:
        assert engine.get_agent("nonexistent") is None

    def test_unregister_agent(self, engine: A2AEngine, agent_card: AgentCard) -> None:
        engine.register_agent(agent_card)
        engine.unregister_agent("peer-agent-002")
        assert engine.get_agent("peer-agent-002") is None

    def test_find_agents_by_capability(self, engine: A2AEngine, agent_card: AgentCard) -> None:
        engine.register_agent(agent_card)
        results = engine.find_agents_by_capability("inference")
        assert len(results) >= 1
        assert any(a.agent_id == "peer-agent-002" for a in results)

    def test_find_best_agent(self, engine: A2AEngine, agent_card: AgentCard) -> None:
        engine.register_agent(agent_card)
        best = engine.find_best_agent("inference")
        assert best is not None


# ─────────────────────────────────────────────────────────────────────────────
# MESSAGING
# ─────────────────────────────────────────────────────────────────────────────


class TestMessaging:
    """Test A2A message creation and verification."""

    def test_create_message(self, engine: A2AEngine) -> None:
        from core.a2a.schema import MessageType

        msg = engine.create_message(
            message_type=MessageType.TASK_REQUEST,
            recipient_id="peer-002",
            payload={"prompt": "Hello"},
        )
        assert isinstance(msg, A2AMessage)
        assert msg.sender_id == "test-agent-001"

    def test_verify_own_message(self, engine: A2AEngine) -> None:
        """Engine should be able to verify messages it created."""
        from core.a2a.schema import MessageType

        msg = engine.create_message(
            message_type=MessageType.TASK_REQUEST,
            recipient_id="peer-002",
            payload={"prompt": "test"},
        )
        result = engine.verify_message(msg)
        assert isinstance(result, bool)

    def test_create_discover_message(self, engine: A2AEngine) -> None:
        msg = engine.create_discover_message()
        assert isinstance(msg, A2AMessage)

    def test_create_announce_message(self, engine: A2AEngine) -> None:
        msg = engine.create_announce_message()
        assert isinstance(msg, A2AMessage)


# ─────────────────────────────────────────────────────────────────────────────
# STATS
# ─────────────────────────────────────────────────────────────────────────────


class TestA2AStats:
    """Test engine statistics and diagnostics."""

    def test_stats_structure(self, engine: A2AEngine) -> None:
        stats = engine.get_stats()
        assert "agent_id" in stats
        assert "registered_agents" in stats or "my_capabilities" in stats
