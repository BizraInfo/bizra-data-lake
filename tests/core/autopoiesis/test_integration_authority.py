"""Authority-monotonicity tests for autopoietic production integration.

A candidate may be high-fitness and Ihsan-compliant without acquiring authority
for production promotion. Eligibility is evidence; approval is authority.
"""

from types import SimpleNamespace

import pytest

from core.autopoiesis.genome import AgentGenome
from core.autopoiesis.loop import AutopoieticLoop


def _eligible_genome() -> AgentGenome:
    genome = AgentGenome()
    genome.fitness = 0.99
    assert genome.is_ihsan_compliant()
    return genome


async def _integrate_once(loop: AutopoieticLoop, genome: AgentGenome) -> None:
    evolution_result = SimpleNamespace(final_population=[genome])
    emergence_report = SimpleNamespace(novel_genomes=set())
    await loop._phase_integrate(evolution_result, emergence_report)


@pytest.mark.asyncio
async def test_eligible_candidate_without_authority_is_not_promoted():
    """Fitness + Ihsan compliance must not manufacture integration authority."""
    loop = AutopoieticLoop(on_integration=None)
    genome = _eligible_genome()

    await _integrate_once(loop, genome)

    assert loop.get_production_agents() == []
    assert loop.state.integrations_performed == 0


@pytest.mark.asyncio
async def test_explicit_refusal_blocks_promotion():
    """An explicit negative authority decision must keep the candidate out."""
    loop = AutopoieticLoop(on_integration=lambda _candidate: False)
    genome = _eligible_genome()

    await _integrate_once(loop, genome)

    assert loop.get_production_agents() == []
    assert loop.state.integrations_performed == 0


@pytest.mark.asyncio
async def test_explicit_positive_authority_allows_promotion():
    """Only an explicit positive authority decision may promote the candidate."""
    loop = AutopoieticLoop(on_integration=lambda _candidate: True)
    genome = _eligible_genome()

    await _integrate_once(loop, genome)

    assert [agent.id for agent in loop.get_production_agents()] == [genome.id]
    assert loop.state.integrations_performed == 1
