"""
Tests for User Context — The System Knows Its Human
====================================================
Verifies the critical integration layer between PAT infrastructure
and the human it serves:

1. UserProfile stores and persists user identity
2. ConversationMemory records and recalls turns
3. UserContextManager builds contextual system prompts
4. PAT agent routing selects the right agent
5. Persistence survives restart

Standing on Giants: Tulving (memory types) + Shannon (information) + Anthropic (alignment)
"""

import json
import tempfile
from pathlib import Path

import pytest

from core.sovereign.user_context import (
    ConversationMemory,
    ConversationTurn,
    UserContextManager,
    UserProfile,
    select_pat_agent,
)


# =============================================================================
# UserProfile
# =============================================================================


class TestUserProfile:
    """The system must know who it serves."""

    def test_empty_profile_not_populated(self):
        profile = UserProfile()
        assert not profile.is_populated()

    def test_populated_profile(self):
        profile = UserProfile(
            name="Mohammed",
            mission="Empower 8 billion humans with sovereign AI",
        )
        assert profile.is_populated()

    def test_profile_serialization(self):
        profile = UserProfile(
            name="Mohammed",
            bio="Founder of BIZRA",
            mission="Every human is a node, every node is a seed",
            expertise=["AI", "distributed systems", "Islamic finance"],
            values=["sovereignty", "ihsan", "justice"],
            pain_points=["centralized AI serves corporations not humans"],
            goals_short=["Get PAT working for one user"],
            goals_long=["Scale to 8 billion nodes"],
            dreams=["A world where technology serves human flourishing"],
        )
        data = profile.to_dict()
        restored = UserProfile.from_dict(data)

        assert restored.name == "Mohammed"
        assert restored.mission == profile.mission
        assert "AI" in restored.expertise
        assert "sovereignty" in restored.values
        assert len(restored.dreams) == 1

    def test_summary_for_prompt(self):
        profile = UserProfile(
            name="Mohammed",
            mission="Empower humanity",
            expertise=["AI", "systems"],
        )
        summary = profile.summary_for_prompt()
        assert "Mohammed" in summary
        assert "Empower humanity" in summary
        assert "AI" in summary

    def test_empty_summary(self):
        profile = UserProfile()
        summary = profile.summary_for_prompt()
        assert summary == ""  # No content = no summary


# =============================================================================
# ConversationMemory
# =============================================================================


class TestConversationMemory:
    """The system must remember what was said."""

    def test_add_turns(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(Path(tmpdir))
            mem.add_human_turn("What is sovereignty?")
            mem.add_pat_turn("Sovereignty is self-governance...", agent_role="strategist")

            assert mem.get_turn_count() == 2

    def test_recent_context(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(Path(tmpdir))
            mem.add_human_turn("Hello")
            mem.add_pat_turn("Welcome!", agent_role="strategist")

            context = mem.get_recent_context()
            assert "Human: Hello" in context
            assert "PAT (strategist): Welcome!" in context

    def test_context_window_limit(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(Path(tmpdir))
            for i in range(20):
                mem.add_human_turn(f"Question {i}")
                mem.add_pat_turn(f"Answer {i}")

            # Default window is 10 turns
            context = mem.get_recent_context(max_turns=4)
            # Should only contain last 4 turns
            assert "Question 18" in context or "Question 19" in context

    def test_persistence(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            mem1 = ConversationMemory(Path(tmpdir))
            mem1.add_human_turn("First question")
            mem1.add_pat_turn("First answer", agent_role="researcher")
            mem1.save()

            # Load
            mem2 = ConversationMemory(Path(tmpdir))
            mem2.load()
            assert mem2.get_turn_count() == 2
            context = mem2.get_recent_context()
            assert "First question" in context

    def test_persistable_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem = ConversationMemory(Path(tmpdir))
            mem.add_human_turn("test")
            state = mem.get_persistable_state()

            assert state["total_turns"] == 1
            assert len(state["turns"]) == 1

    def test_restore_persistable_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mem1 = ConversationMemory(Path(tmpdir))
            mem1.add_human_turn("restored turn")
            state = mem1.get_persistable_state()

            mem2 = ConversationMemory(Path(tmpdir))
            mem2.restore_persistable_state(state)
            assert mem2.get_turn_count() == 1


# =============================================================================
# ConversationTurn
# =============================================================================


class TestConversationTurn:
    def test_turn_serialization(self):
        turn = ConversationTurn(
            role="human",
            content="What are my goals?",
        )
        data = turn.to_dict()
        restored = ConversationTurn.from_dict(data)
        assert restored.role == "human"
        assert restored.content == "What are my goals?"

    def test_pat_turn_with_scores(self):
        turn = ConversationTurn(
            role="pat",
            content="Your goals are...",
            agent_role="strategist",
            snr_score=0.95,
            ihsan_score=0.97,
        )
        data = turn.to_dict()
        restored = ConversationTurn.from_dict(data)
        assert restored.agent_role == "strategist"
        assert restored.snr_score == 0.95


# =============================================================================
# UserContextManager
# =============================================================================


class TestUserContextManager:
    """The integration point that wires everything together."""

    def test_build_system_prompt_empty_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = UserContextManager(Path(tmpdir))
            prompt = mgr.build_system_prompt()

            assert "Personal Agentic Team" in prompt
            assert "not yet populated" in prompt

    def test_build_system_prompt_populated(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = UserContextManager(Path(tmpdir))
            mgr.profile = UserProfile(
                name="Mohammed",
                mission="Every human is a node",
            )
            prompt = mgr.build_system_prompt()

            assert "Mohammed" in prompt
            assert "Every human is a node" in prompt
            assert "not yet populated" not in prompt

    def test_system_prompt_includes_conversation(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = UserContextManager(Path(tmpdir))
            mgr.conversation.add_human_turn("What is BIZRA?")
            mgr.conversation.add_pat_turn("BIZRA means seed...")

            prompt = mgr.build_system_prompt()
            assert "What is BIZRA?" in prompt
            assert "BIZRA means seed" in prompt

    def test_system_prompt_includes_pat_info(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = UserContextManager(Path(tmpdir))
            prompt = mgr.build_system_prompt(
                pat_team_info="Available: strategist, researcher, developer"
            )
            assert "strategist" in prompt

    def test_system_prompt_includes_memory(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = UserContextManager(Path(tmpdir))
            prompt = mgr.build_system_prompt(
                memory_context="User previously worked on federation protocol"
            )
            assert "federation protocol" in prompt

    def test_persistence_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create and save
            mgr1 = UserContextManager(Path(tmpdir))
            mgr1.profile = UserProfile(name="Test User", mission="Test mission")
            mgr1.conversation.add_human_turn("Hello PAT")
            mgr1.save()

            # Load
            mgr2 = UserContextManager(Path(tmpdir))
            mgr2.load()
            assert mgr2.profile.name == "Test User"
            assert mgr2.conversation.get_turn_count() == 1

    def test_update_profile(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = UserContextManager(Path(tmpdir))
            mgr.update_profile(name="Mohammed", mission="Empower 8B humans")

            assert mgr.profile.name == "Mohammed"
            assert mgr.profile.mission == "Empower 8B humans"

            # Verify persisted
            mgr2 = UserContextManager(Path(tmpdir))
            mgr2.load()
            assert mgr2.profile.name == "Mohammed"

    def test_persistable_state_for_coordinator(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = UserContextManager(Path(tmpdir))
            mgr.profile = UserProfile(name="Test")
            mgr.conversation.add_human_turn("query")

            state = mgr.get_persistable_state()
            assert "profile" in state
            assert "conversation" in state
            assert state["profile"]["name"] == "Test"

    def test_restore_from_coordinator_state(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr1 = UserContextManager(Path(tmpdir))
            mgr1.profile = UserProfile(name="Restored")
            mgr1.conversation.add_human_turn("survived restart")
            state = mgr1.get_persistable_state()

            mgr2 = UserContextManager(Path(tmpdir))
            mgr2.restore_persistable_state(state)
            assert mgr2.profile.name == "Restored"
            assert mgr2.conversation.get_turn_count() == 1


# =============================================================================
# PAT Agent Routing
# =============================================================================


class TestPATAgentRouting:
    """The right agent should handle the right query."""

    def test_strategy_query(self):
        agent = select_pat_agent("What should our strategy be for Q2?", [])
        assert agent == "strategist"

    def test_research_query(self):
        agent = select_pat_agent("Research the latest trends in federated learning", [])
        assert agent == "researcher"

    def test_development_query(self):
        agent = select_pat_agent("Implement the new API endpoint for user auth", [])
        assert agent == "developer"

    def test_analysis_query(self):
        agent = select_pat_agent("Show me the performance metrics dashboard", [])
        assert agent == "analyst"

    def test_review_query(self):
        agent = select_pat_agent("Review the security audit findings", [])
        assert agent == "reviewer"

    def test_execution_query(self):
        agent = select_pat_agent("Execute the scheduled automate task now", [])
        assert agent == "executor"

    def test_guardian_query(self):
        agent = select_pat_agent("Monitor for threat and protect against harm", [])
        assert agent == "guardian"

    def test_generic_query_returns_none(self):
        agent = select_pat_agent("Hello, how are you today?", [])
        assert agent is None

    def test_multi_keyword_picks_strongest(self):
        # "research and analyze" has both researcher and analyst keywords
        agent = select_pat_agent("Research and analyze the data trends", [])
        # researcher has "research" + "analyze", analyst has "data"
        assert agent in ("researcher", "analyst")
