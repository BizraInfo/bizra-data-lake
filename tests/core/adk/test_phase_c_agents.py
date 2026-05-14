"""Phase C Tests — 5 new PAT agents through the ADK lifecycle.

Each agent must:
- Import cleanly
- Have a charter with hash
- Declare governance_class PAT
- Have at least 1 @tool
- Be under 200 LOC
- Pass FATE gate on well-formed questions (when Ollama available)
"""

import pytest
from pathlib import Path

from core.adk.testing import make_test_mission

# ── Import tests (must all pass regardless of Ollama) ──


def test_strategist_imports():
    from core.adk.agents.strategist import StrategistAgent

    agent = StrategistAgent()
    assert agent.name == "Strategist"
    assert agent.governance_class == "PAT"
    assert agent._charter_hash
    assert "gather_strategic_context" in agent.tools


def test_analyst_imports():
    from core.adk.agents.analyst import AnalystAgent

    agent = AnalystAgent()
    assert agent.name == "Analyst"
    assert agent.governance_class == "PAT"
    assert agent._charter_hash
    assert "gather_metrics" in agent.tools


def test_creator_imports():
    from core.adk.agents.creator import CreatorAgent

    agent = CreatorAgent()
    assert agent.name == "Creator"
    assert agent.governance_class == "PAT"
    assert agent._charter_hash
    assert "gather_source_material" in agent.tools


def test_executor_imports():
    from core.adk.agents.executor import ExecutorAgent

    agent = ExecutorAgent()
    assert agent.name == "Executor"
    assert agent.governance_class == "PAT"
    assert agent._charter_hash
    assert "run_safe_command" in agent.tools


def test_coordinator_imports():
    from core.adk.agents.coordinator import CoordinatorAgent

    agent = CoordinatorAgent()
    assert agent.name == "Coordinator"
    assert agent.governance_class == "PAT"
    assert agent._charter_hash
    assert "plan_delegation" in agent.tools


# ── LOC limits ──


def test_all_agents_under_200_loc():
    agents_dir = Path(__file__).parent.parent.parent.parent / "core" / "adk" / "agents"
    for agent_file in agents_dir.glob("*.py"):
        if agent_file.name == "__init__.py":
            continue
        loc = agent_file.read_text().count("\n") + 1
        assert loc <= 200, f"{agent_file.name} is {loc} LOC, exceeds 200 limit"


# ── Charter uniqueness ──


def test_all_charters_unique():
    from core.adk.agents.researcher import ResearcherAgent
    from core.adk.agents.strategist import StrategistAgent
    from core.adk.agents.analyst import AnalystAgent
    from core.adk.agents.creator import CreatorAgent
    from core.adk.agents.executor import ExecutorAgent
    from core.adk.agents.coordinator import CoordinatorAgent

    hashes = [
        ResearcherAgent._charter_hash,
        StrategistAgent._charter_hash,
        AnalystAgent._charter_hash,
        CreatorAgent._charter_hash,
        ExecutorAgent._charter_hash,
        CoordinatorAgent._charter_hash,
    ]
    assert len(set(hashes)) == 6, "Charter hashes must be unique"


# ── Tool evidence gathering (no Ollama needed) ──


def test_strategist_gathers_evidence():
    from core.adk.agents.strategist import StrategistAgent

    agent = StrategistAgent()
    refs = agent.gather_strategic_context("What is the BIZRA architecture?")
    assert len(refs) >= 1


def test_analyst_gathers_metrics():
    from core.adk.agents.analyst import AnalystAgent

    agent = AnalystAgent()
    refs = agent.gather_metrics("How many tests does BIZRA have?")
    assert len(refs) >= 1


def test_creator_gathers_sources():
    from core.adk.agents.creator import CreatorAgent

    agent = CreatorAgent()
    refs = agent.gather_source_material("Summarize the BIZRA architecture")
    assert len(refs) >= 1


def test_executor_safe_commands():
    from core.adk.agents.executor import ExecutorAgent

    agent = ExecutorAgent()
    # Test that git_status works (safe, fast)
    refs = agent.run_safe_command("git_status")
    assert len(refs) == 1
    assert "executor:git_status" in refs[0]


def test_executor_refuses_unsafe():
    from core.adk.agents.executor import ExecutorAgent

    agent = ExecutorAgent()
    refs = agent.run_safe_command("rm_rf_everything")
    assert "refused" in refs[0].lower()


# ── Full lifecycle tests (need Ollama) ──


@pytest.mark.asyncio
async def test_executor_full_lifecycle():
    """Executor should run git_status through full FATE pipeline."""
    from core.adk.agents.executor import ExecutorAgent

    agent = ExecutorAgent()
    mission = make_test_mission("Run git status check")
    result = await agent.run(mission)
    assert result is not None
    assert result.mission_id
    # Executor doesn't need Ollama for the command itself
    assert result.evidence_refs


@pytest.mark.asyncio
async def test_analyst_full_lifecycle():
    """Analyst should gather real metrics through FATE pipeline."""
    from core.adk.agents.analyst import AnalystAgent

    agent = AnalystAgent()
    mission = make_test_mission("How many tests does the proof engine have?")
    result = await agent.run(mission)
    assert result is not None
    assert result.mission_id
