"""Tests for core/adk/agent.py — Agent base class, charter, identity."""

import hashlib
import pytest

from core.adk.agent import Agent, charter, AgentIdentity, _DraftOutput, _RefuseOutput
from core.adk.mission import GovernanceClass, Mission
from core.adk.tools import tool


# ── Charter tests ──

def test_charter_decorator_sets_hash():
    @charter("I am a test agent.")
    class TestAgent(Agent):
        name = "TestAgent"
        async def act(self, mission):
            return self.draft(content="hello", evidence=[])

    expected = hashlib.blake2b(b"I am a test agent.", digest_size=32).hexdigest()
    assert TestAgent._charter_hash == expected
    assert TestAgent._charter_text == "I am a test agent."


def test_charter_strips_whitespace():
    @charter("  \n  I am trimmed.  \n  ")
    class TrimAgent(Agent):
        name = "TrimAgent"
        async def act(self, mission):
            return self.draft(content="x", evidence=[])

    assert TrimAgent._charter_text == "I am trimmed."


@pytest.mark.asyncio
async def test_agent_without_charter_blocked_at_run():
    """An agent without @charter is blocked at run time (NIYYAH step)."""
    class NoCharterAgent(Agent):
        name = "NoCharter"
        async def act(self, mission):
            return self.draft(content="x", evidence=[])

    agent = NoCharterAgent()
    assert agent._charter_hash == ""
    result = await agent.run(Mission(question="test"))
    assert not result.success
    assert result.verdict == "BLOCKED_BY_CHARTER"


# ── Identity tests ──

def test_identity_fields():
    @charter("Test identity agent.")
    class IdAgent(Agent):
        name = "IdAgent"
        governance_class = "SAT"
        async def act(self, mission):
            return self.draft(content="x", evidence=[])

    agent = IdAgent()
    ident = agent.identity
    assert isinstance(ident, AgentIdentity)
    assert ident.name == "IdAgent"
    assert ident.governance_class == GovernanceClass.SAT
    assert len(ident.charter_hash) == 64


def test_charter_hash_is_deterministic():
    @charter("Deterministic charter text.")
    class A(Agent):
        name = "A"
        async def act(self, mission):
            return self.draft(content="x", evidence=[])

    @charter("Deterministic charter text.")
    class B(Agent):
        name = "B"
        async def act(self, mission):
            return self.draft(content="x", evidence=[])

    assert A._charter_hash == B._charter_hash


def test_different_charters_different_hashes():
    @charter("Charter alpha.")
    class A(Agent):
        name = "A"
        async def act(self, mission):
            return self.draft(content="x", evidence=[])

    @charter("Charter beta.")
    class B(Agent):
        name = "B"
        async def act(self, mission):
            return self.draft(content="x", evidence=[])

    assert A._charter_hash != B._charter_hash


# ── Draft / Refuse tests ──

def test_draft_returns_draft_output():
    @charter("Drafter.")
    class D(Agent):
        name = "D"
        async def act(self, mission):
            return self.draft(content="answer", evidence=["ref:1"])

    agent = D()
    out = agent.draft(content="test", evidence=["a", "b"])
    assert isinstance(out, _DraftOutput)
    assert out.content == "test"
    assert out.evidence_refs == ["a", "b"]


def test_refuse_returns_refuse_output():
    @charter("Refuser.")
    class R(Agent):
        name = "R"
        async def act(self, mission):
            return self.refuse(reason="not enough data")

    agent = R()
    out = agent.refuse(reason="nope")
    assert isinstance(out, _RefuseOutput)
    assert out.reason == "nope"


def test_draft_handles_various_evidence_types():
    @charter("Evidence handler.")
    class E(Agent):
        name = "E"
        async def act(self, mission):
            return self.draft(content="x", evidence=[])

    agent = E()

    class FakeEvidence:
        uri = "file:///data/test.txt"

    out = agent.draft(content="x", evidence=["str-ref", FakeEvidence(), 42])
    assert out.evidence_refs == ["str-ref", "file:///data/test.txt", "42"]


def test_draft_with_no_evidence():
    @charter("No evidence agent.")
    class N(Agent):
        name = "N"
        async def act(self, mission):
            return self.draft(content="x", evidence=[])

    agent = N()
    out = agent.draft(content="bare claim", evidence=None)
    assert out.evidence_refs == []


# ── Tool discovery tests ──

def test_tool_discovery():
    @charter("Tool agent.")
    class T(Agent):
        name = "T"

        @tool
        def search(self, query: str) -> list:
            return []

        @tool(max_results=5)
        def fetch(self, url: str) -> str:
            return ""

        async def act(self, mission):
            return self.draft(content="x", evidence=[])

    agent = T()
    tools = agent.tools
    assert "search" in tools
    assert "fetch" in tools
    assert len(tools) == 2
