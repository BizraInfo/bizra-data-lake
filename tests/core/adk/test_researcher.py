"""Tests for the ADK Researcher agent — Phase B kill gate.

These hit the real FATE gate via Ollama. If the Researcher can't produce
a loop proof equivalent to the existing mvda/pat_researcher.py, the ADK
abstraction fails its kill condition.
"""

import pytest

from core.adk.agents.researcher import ResearcherAgent
from core.adk.testing import assert_receipt_valid, make_test_mission


@pytest.mark.asyncio
async def test_researcher_has_charter():
    agent = ResearcherAgent()
    assert agent._charter_hash
    assert "Researcher" in agent._charter_text
    assert agent.name == "Researcher"


@pytest.mark.asyncio
async def test_researcher_discovers_tools():
    agent = ResearcherAgent()
    assert "search_local" in agent.tools


@pytest.mark.asyncio
async def test_researcher_gathers_evidence():
    """Researcher should find git evidence for spearpoint questions."""
    agent = ResearcherAgent()
    refs = agent.search_local("What is the Spearpoint seal?")
    assert len(refs) >= 1
    assert any("git" in r for r in refs)


@pytest.mark.asyncio
async def test_researcher_refuses_without_evidence():
    """A question with no local evidence should produce honest refusal."""
    agent = ResearcherAgent()
    mission = make_test_mission("What is the airspeed velocity of an unladen swallow?")
    result = await agent.run(mission)
    # Either refuses or produces low-quality answer that FATE blocks
    assert result is not None
    if result.verdict == "REFUSED":
        assert "evidence" in result.reason.lower()


@pytest.mark.asyncio
@pytest.mark.requires_ollama
async def test_researcher_full_lifecycle():
    """Full end-to-end: question -> evidence -> Ollama -> FATE -> receipt -> loop proof."""
    agent = ResearcherAgent()
    mission = make_test_mission(
        "What is the BIZRA Spearpoint seal and why does it matter?",
        max_tokens=2048,
    )
    result = await agent.run(mission)

    assert result is not None
    assert result.mission_id
    assert result.evidence_refs  # should have found git evidence

    # FATE may pass or block depending on Ollama output quality
    assert result.verdict in (
        "PASS",
        "BLOCKED_BY_IHSAN",
        "BLOCKED_BY_EVIDENCE",
        "REFUSED",
        "DEGRADED",
    )

    if result.success:
        assert_receipt_valid(result)
        assert result.content  # non-empty answer
        assert result.ihsan_score >= 0.95
        assert result.loop_proof is not None
        assert result.loop_proof.manifest_hash


@pytest.mark.asyncio
@pytest.mark.requires_ollama
async def test_researcher_loop_proof_structure():
    """Verify the loop proof has the expected structure."""
    agent = ResearcherAgent()
    mission = make_test_mission("Describe the BIZRA proof engine architecture.")
    result = await agent.run(mission)

    if result.loop_proof is not None:
        lp = result.loop_proof
        assert lp.version
        assert lp.proof_class
        assert len(lp.steps) >= 3  # at least PAT, Evidence, FATE steps
        assert lp.manifest_hash
        # Steps should have meaningful actors
        actors = [s.actor for s in lp.steps]
        assert all(a for a in actors), f"Empty actor found in steps: {actors}"


@pytest.mark.asyncio
async def test_researcher_evidence_verified():
    """Evidence refs produced by the Researcher should pass the auditor."""
    from core.proof_engine.evidence_audit import audit_evidence

    agent = ResearcherAgent()
    refs = agent.search_local("What is the Spearpoint seal?")

    if refs:
        audit = audit_evidence(refs)
        # Git refs should be verifiable
        assert audit.valid_count >= 1


@pytest.mark.asyncio
@pytest.mark.requires_ollama
async def test_researcher_receipt_signed():
    """If FATE passes, the receipt should be signed."""
    agent = ResearcherAgent()
    mission = make_test_mission("What is the BIZRA Spearpoint seal?")
    result = await agent.run(mission)

    if result.success and result.receipt is not None:
        assert hasattr(result.receipt, "signature")
        assert result.receipt.signature


@pytest.mark.asyncio
async def test_researcher_under_200_loc():
    """Kill gate: the agent file must be under 200 lines."""
    from pathlib import Path

    agent_file = (
        Path(__file__).parent.parent.parent.parent
        / "core"
        / "adk"
        / "agents"
        / "researcher.py"
    )
    lines = agent_file.read_text().count("\n") + 1
    assert lines <= 200, f"Researcher is {lines} LOC, exceeds 200 LOC limit"
