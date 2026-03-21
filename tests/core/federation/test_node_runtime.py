from __future__ import annotations

import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from core.federation.consensus import ConsensusEngine
from core.federation.node import FederationNode
from core.pci.crypto import generate_keypair


def _engine(node_id: str = "node0") -> ConsensusEngine:
    private_key, public_key = generate_keypair()
    return ConsensusEngine(node_id, private_key, public_key)


def test_check_and_finalize_surfaces_committed_proposals_once() -> None:
    engine = _engine()
    proposal = engine.propose_pattern({"pattern": "test", "impact_score": 0.92})
    assert proposal is not None

    assert engine._commit_proposal(proposal.proposal_id) is True

    first = engine.check_and_finalize()
    second = engine.check_and_finalize()

    assert first == [(proposal.proposal_id, True, 0.92)]
    assert second == []


def test_check_and_finalize_falls_back_to_vote_average() -> None:
    engine = _engine()
    proposal = engine.propose_pattern({"pattern": "test"})
    assert proposal is not None

    vote = engine.cast_vote(proposal, ihsan_score=0.96)
    assert vote is not None
    engine.votes[proposal.proposal_id].append(vote)
    assert engine._commit_proposal(proposal.proposal_id) is True

    results = engine.check_and_finalize()

    assert results == [(proposal.proposal_id, True, pytest.approx(0.96))]


@pytest.mark.asyncio
async def test_consensus_check_loop_broadcasts_new_acceptance(monkeypatch) -> None:
    node = FederationNode(node_id="loop-node", bind_address="127.0.0.1:0")
    proposal = node.consensus.propose_pattern({"pattern": "test", "impact_score": 0.91})
    assert proposal is not None
    assert node.consensus._commit_proposal(proposal.proposal_id) is True

    broadcasts: list[dict[str, object]] = []
    node._broadcast_pattern = lambda data: broadcasts.append(json.loads(data.decode("utf-8")))  # type: ignore[assignment]

    async def immediate_sleep(_seconds: float) -> None:
        return None

    original = node.consensus.check_and_finalize

    def one_shot_finalize() -> list[tuple[str, bool, float]]:
        node._running = False
        return original()

    monkeypatch.setattr("core.federation.node.asyncio.sleep", immediate_sleep)
    node.consensus.check_and_finalize = one_shot_finalize  # type: ignore[method-assign]
    node._running = True

    await node._consensus_check_loop()

    assert broadcasts == [
        {
            "type": "PATTERN_ACCEPTED",
            "pattern_id": proposal.proposal_id,
            "final_impact": 0.91,
        }
    ]
    assert node.contribution_count == 1


@pytest.mark.asyncio
async def test_stop_cancels_background_tasks() -> None:
    node = FederationNode(node_id="stop-node", bind_address="127.0.0.1:0")
    node.gossip.start = AsyncMock()
    node.gossip.stop = AsyncMock()

    pattern_cancelled = asyncio.Event()
    consensus_cancelled = asyncio.Event()

    async def fake_pattern_loop() -> None:
        try:
            while True:
                await asyncio.sleep(10)
        except asyncio.CancelledError:
            pattern_cancelled.set()
            raise

    async def fake_consensus_loop() -> None:
        try:
            while True:
                await asyncio.sleep(10)
        except asyncio.CancelledError:
            consensus_cancelled.set()
            raise

    node._pattern_sync_loop = fake_pattern_loop  # type: ignore[assignment]
    node._consensus_check_loop = fake_consensus_loop  # type: ignore[assignment]

    await node.start()
    await asyncio.sleep(0)
    assert len(node._background_tasks) == 2

    await node.stop()

    assert pattern_cancelled.is_set()
    assert consensus_cancelled.is_set()
    assert node._background_tasks == set()
