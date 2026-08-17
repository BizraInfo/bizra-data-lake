"""Authority-monotonicity tests for autopoietic production integration.

A candidate may be high-fitness and Ihsan-compliant without acquiring authority
for production promotion. Eligibility is evidence; learning acceptance is not
authority; approval must come from a distinct promotion authority surface.
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
async def test_learning_acceptance_is_not_promotion_authority():
    """A learning hook may accept a candidate without promoting it to production."""
    observed = []

    def accept_for_learning(candidate):
        observed.append(candidate.genome.id)
        return True

    loop = AutopoieticLoop(on_integration=accept_for_learning)
    genome = _eligible_genome()

    await _integrate_once(loop, genome)

    assert observed == [genome.id]
    assert loop.get_production_agents() == []
    assert loop.state.integrations_performed == 0


@pytest.mark.asyncio
async def test_explicit_refusal_blocks_promotion():
    """An explicit negative promotion-authority decision must keep the candidate out."""
    loop = AutopoieticLoop(authorize_integration=lambda _candidate: False)
    genome = _eligible_genome()

    await _integrate_once(loop, genome)

    assert loop.get_production_agents() == []
    assert loop.state.integrations_performed == 0


@pytest.mark.asyncio
async def test_truthy_non_boolean_authority_does_not_promote():
    """Authority must return exact True; generic truthy values are not decisions."""
    loop = AutopoieticLoop(authorize_integration=lambda _candidate: 1)
    genome = _eligible_genome()

    await _integrate_once(loop, genome)

    assert loop.get_production_agents() == []
    assert loop.state.integrations_performed == 0


@pytest.mark.asyncio
async def test_explicit_positive_authority_allows_promotion():
    """Only an explicit positive promotion-authority decision may promote."""
    loop = AutopoieticLoop(authorize_integration=lambda _candidate: True)
    genome = _eligible_genome()

    await _integrate_once(loop, genome)

    assert [agent.id for agent in loop.get_production_agents()] == [genome.id]
    assert loop.state.integrations_performed == 1


@pytest.mark.asyncio
async def test_learning_hook_and_authority_are_independent_surfaces():
    """Learning notification can coexist with, but cannot replace, authority."""
    observed = []
    authorized = []

    def observe(candidate):
        observed.append(candidate.genome.id)
        return False

    def authorize(candidate):
        authorized.append(candidate.genome.id)
        return True

    loop = AutopoieticLoop(
        on_integration=observe,
        authorize_integration=authorize,
    )
    genome = _eligible_genome()

    await _integrate_once(loop, genome)

    assert observed == [genome.id]
    assert authorized == [genome.id]
    assert [agent.id for agent in loop.get_production_agents()] == [genome.id]
    assert loop.state.integrations_performed == 1


@pytest.mark.asyncio
async def test_authority_exception_fails_closed_without_mutation():
    """A broken authority surface must never partially promote a candidate."""

    def broken_authority(_candidate):
        raise RuntimeError("authority unavailable")

    loop = AutopoieticLoop(authorize_integration=broken_authority)
    genome = _eligible_genome()

    with pytest.raises(RuntimeError, match="authority unavailable"):
        await _integrate_once(loop, genome)

    assert loop.get_production_agents() == []
    assert loop.state.integrations_performed == 0
