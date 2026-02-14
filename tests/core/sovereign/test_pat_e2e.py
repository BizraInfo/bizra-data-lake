"""
PAT End-to-End Smoke Test
=========================
Validates the full PAT pipeline without requiring Ollama:
1. UserProfile loads and enriches prompts
2. ConversationMemory records and recalls turns
3. Agent routing selects the right agent
4. System prompt carries user context
5. Runtime query() integrates everything

Standing on Giants: Shannon + Tulving + Al-Ghazali + Anthropic
"""

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.sovereign.runtime_types import RuntimeConfig, SovereignQuery
from core.sovereign.user_context import (
    UserContextManager,
    UserProfile,
    select_pat_agent,
)


class TestPATEndToEnd:
    """Full pipeline integration test."""

    @pytest.mark.asyncio
    async def test_runtime_loads_user_context(self):
        """Runtime initializes user context from disk."""
        from core.sovereign.runtime_core import SovereignRuntime

        with tempfile.TemporaryDirectory() as tmpdir:
            config = RuntimeConfig.minimal()
            config.state_dir = Path(tmpdir)

            # Pre-seed profile
            import json

            (Path(tmpdir) / "user_profile.json").write_text(
                json.dumps({"name": "Test User", "mission": "Test mission"})
            )

            runtime = SovereignRuntime(config)
            await runtime._init_components()
            runtime._init_user_context()

            assert runtime._user_context is not None
            assert runtime._user_context.profile.name == "Test User"

    @pytest.mark.asyncio
    async def test_contextual_prompt_includes_user(self):
        """_build_contextual_prompt() enriches the prompt with user context."""
        from core.sovereign.runtime_core import SovereignRuntime

        with tempfile.TemporaryDirectory() as tmpdir:
            config = RuntimeConfig.minimal()
            config.state_dir = Path(tmpdir)

            runtime = SovereignRuntime(config)
            runtime._initialized = True
            runtime._user_context = UserContextManager(Path(tmpdir))
            runtime._user_context.profile = UserProfile(
                name="Mohammed",
                mission="Empower 8 billion humans",
            )

            query = SovereignQuery(text="What should I focus on?")
            prompt = await runtime._build_contextual_prompt("What should I focus on?", query)

            assert "Mohammed" in prompt
            assert "Empower 8 billion humans" in prompt
            assert "Personal Agentic Team" in prompt

    @pytest.mark.asyncio
    async def test_contextual_prompt_includes_conversation(self):
        """Previous conversation appears in prompt."""
        from core.sovereign.runtime_core import SovereignRuntime

        with tempfile.TemporaryDirectory() as tmpdir:
            config = RuntimeConfig.minimal()
            config.state_dir = Path(tmpdir)

            runtime = SovereignRuntime(config)
            runtime._initialized = True
            runtime._user_context = UserContextManager(Path(tmpdir))
            runtime._user_context.conversation.add_human_turn("Tell me about PAT")
            runtime._user_context.conversation.add_pat_turn("PAT has 7 agents...")

            query = SovereignQuery(text="Which agent handles strategy?")
            prompt = await runtime._build_contextual_prompt("Which agent handles strategy?", query)

            assert "Tell me about PAT" in prompt
            assert "PAT has 7 agents" in prompt

    @pytest.mark.asyncio
    async def test_query_records_conversation(self):
        """Each query adds turns to conversation memory."""
        from core.sovereign.runtime_core import SovereignRuntime

        with tempfile.TemporaryDirectory() as tmpdir:
            config = RuntimeConfig.minimal()
            config.state_dir = Path(tmpdir)

            runtime = SovereignRuntime(config)
            runtime._initialized = True
            runtime._user_context = UserContextManager(Path(tmpdir))
            # CRITICAL-1: Initialize gate chain so queries pass (fail-closed)
            runtime._init_gate_chain()

            result = await runtime.query("What is sovereignty?")

            # Should have recorded human turn + PAT response
            assert runtime._user_context.conversation.get_turn_count() == 2

    @pytest.mark.asyncio
    async def test_agent_routing_in_prompt(self):
        """Agent routing label appears in contextual prompt."""
        from core.sovereign.genesis_identity import AgentIdentity, GenesisState, NodeIdentity
        from core.sovereign.runtime_core import SovereignRuntime

        with tempfile.TemporaryDirectory() as tmpdir:
            config = RuntimeConfig.minimal()
            config.state_dir = Path(tmpdir)

            runtime = SovereignRuntime(config)
            runtime._initialized = True
            runtime._user_context = UserContextManager(Path(tmpdir))

            # Create mock genesis with PAT team
            identity = NodeIdentity(
                node_id="BIZRA-TEST0001",
                public_key="test",
                name="Test Node",
                location="test",
                created_at=0,
                identity_hash=b"",
            )
            runtime._genesis = GenesisState(
                identity=identity,
                pat_team=[
                    AgentIdentity(
                        agent_id="PAT-STRAT",
                        role="strategist",
                        public_key="test",
                        capabilities=["plan"],
                        giants=["Besta"],
                        created_at=0,
                        agent_hash=b"",
                    ),
                ],
            )

            query = SovereignQuery(text="What should our strategy be?")
            prompt = await runtime._build_contextual_prompt("What should our strategy be?", query)

            assert "STRATEGIST" in prompt
            assert query.context.get("_responding_agent") == "strategist"

    @pytest.mark.asyncio
    async def test_shutdown_saves_context(self):
        """Shutdown persists user context to disk."""
        from core.sovereign.runtime_core import SovereignRuntime

        with tempfile.TemporaryDirectory() as tmpdir:
            config = RuntimeConfig.minimal()
            config.state_dir = Path(tmpdir)

            runtime = SovereignRuntime(config)
            runtime._running = True
            runtime._memory_coordinator = AsyncMock()
            runtime._shutdown_event = asyncio.Event()
            runtime._user_context = UserContextManager(Path(tmpdir))
            runtime._user_context.profile.name = "Saved User"
            runtime._user_context.conversation.add_human_turn("Will this survive?")

            await runtime.shutdown()

            # Verify file was written
            profile_file = Path(tmpdir) / "user_profile.json"
            assert profile_file.exists()

            import json

            data = json.loads(profile_file.read_text())
            assert data["name"] == "Saved User"


class TestAgentRoutingComprehensive:
    """Ensure all 7 PAT agents can be reached."""

    @pytest.mark.parametrize(
        "query,expected",
        [
            ("Plan our roadmap and prioritize features", "strategist"),
            ("Research federated learning algorithms", "researcher"),
            ("Implement the API endpoint for users", "developer"),
            ("Analyze the performance metrics data", "analyst"),
            ("Review the audit findings and validate", "reviewer"),
            ("Execute the automate schedule task", "executor"),
            ("Monitor for threat and guard against risk", "guardian"),
        ],
    )
    def test_agent_routing(self, query: str, expected: str):
        agent = select_pat_agent(query, [])
        assert agent == expected, f"Expected {expected} for: {query}"

    def test_generic_returns_none(self):
        assert select_pat_agent("Hello there", []) is None


class TestProfilePersistence:
    """Profile survives process restart."""

    @pytest.mark.asyncio
    async def test_profile_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Session 1: Set profile
            mgr1 = UserContextManager(Path(tmpdir))
            mgr1.update_profile(
                name="Mohammed",
                mission="If system fails for one, how for 8 billion",
                pain_points=["centralized AI"],
                dreams=["sovereign AI for all"],
            )

            # Session 2: Load profile
            mgr2 = UserContextManager(Path(tmpdir))
            mgr2.load()

            assert mgr2.profile.name == "Mohammed"
            assert "centralized AI" in mgr2.profile.pain_points
            assert mgr2.profile.dreams[0] == "sovereign AI for all"

    @pytest.mark.asyncio
    async def test_conversation_survives_restart(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            # Session 1
            mgr1 = UserContextManager(Path(tmpdir))
            mgr1.conversation.add_human_turn("What is my mission?")
            mgr1.conversation.add_pat_turn("Your mission is...", agent_role="strategist")
            mgr1.save()

            # Session 2
            mgr2 = UserContextManager(Path(tmpdir))
            mgr2.load()

            assert mgr2.conversation.get_turn_count() == 2
            context = mgr2.conversation.get_recent_context()
            assert "What is my mission?" in context
            assert "strategist" in context
