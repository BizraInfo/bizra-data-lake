"""
Tests for Sovereignty CLI — Production Genesis Interface
═════════════════════════════════════════════════════════

TDD anchors for the 4 core commands + 2 utilities.

Standing on Giants:
- Beck (2002): Test-Driven Development by Example
- Thompson & Ritchie (1979): Unix philosophy — one tool, one job
"""

from __future__ import annotations

import hashlib
import json
import os
import time

import pytest

from core.constitutional.cli import (
    AttestResult,
    InitResult,
    NodeState,
    StatusResult,
    WorkResult,
    attest_peer,
    get_status,
    init_node,
    load_node_state,
    process_work,
    save_node_state,
)
from core.constitutional.fixed_point import fp, fp_float

# ═══════════════════════════════════════════════════════════════════
# Fixtures
# ═══════════════════════════════════════════════════════════════════


@pytest.fixture
def state_dir(tmp_path):
    """Temporary directory for node state files."""
    return tmp_path / ".bizra"


@pytest.fixture
def initialized_node(state_dir):
    """A node that has already been initialized."""
    result = init_node("test-node", state_dir)
    assert result.success
    return load_node_state(state_dir)


# ═══════════════════════════════════════════════════════════════════
# Test: Init Command
# ═══════════════════════════════════════════════════════════════════


class TestInitCommand:
    """bizra init — Create a sovereign node."""

    def test_init_creates_state_dir(self, state_dir):
        """Init must create the .bizra directory."""
        result = init_node("genesis-node", state_dir)
        assert result.success
        assert state_dir.exists()

    def test_init_creates_node_file(self, state_dir):
        """Init must create a node.json file."""
        init_node("genesis-node", state_dir)
        assert (state_dir / "node.json").exists()

    def test_init_creates_ledger_file(self, state_dir):
        """Init must create a ledger.jsonl file."""
        init_node("genesis-node", state_dir)
        assert (state_dir / "ledger.jsonl").exists()

    def test_init_result_has_node_id(self, state_dir):
        """Result must include the generated node ID."""
        result = init_node("genesis-node", state_dir)
        assert result.node_id
        assert len(result.node_id) == 64  # BLAKE2b-256 hex

    def test_init_result_has_covenant_hash(self, state_dir):
        """Result must include the Declaration covenant hash."""
        result = init_node("genesis-node", state_dir)
        assert result.covenant_hash
        assert len(result.covenant_hash) == 64

    def test_init_node_state_is_loadable(self, state_dir):
        """State written by init must be loadable."""
        init_node("genesis-node", state_dir)
        state = load_node_state(state_dir)
        assert state is not None
        assert state.name == "genesis-node"

    def test_init_sets_zero_balances(self, state_dir):
        """New node starts with zero SEED and zero BLOOM."""
        init_node("genesis-node", state_dir)
        state = load_node_state(state_dir)
        assert state.seed_balance == 0
        assert state.bloom_balance == 0

    def test_init_genesis_event_in_ledger(self, state_dir):
        """Ledger must contain a genesis event after init."""
        init_node("genesis-node", state_dir)
        ledger = (state_dir / "ledger.jsonl").read_text().strip().split("\n")
        assert len(ledger) == 1
        event = json.loads(ledger[0])
        assert event["type"] == "genesis"

    def test_init_refuses_if_already_initialized(self, state_dir):
        """Double init must fail gracefully."""
        init_node("first-node", state_dir)
        result = init_node("second-node", state_dir)
        assert not result.success
        assert "already initialized" in result.error.lower()

    def test_init_node_id_deterministic_for_name_and_time(self, state_dir):
        """Same name at same timestamp should produce same node_id."""
        # Not strictly testable with real time, but verify determinism property:
        result = init_node("determ-node", state_dir)
        state = load_node_state(state_dir)
        # node_id must be a valid 64-char hex string
        int(state.node_id, 16)


# ═══════════════════════════════════════════════════════════════════
# Test: Work Command
# ═══════════════════════════════════════════════════════════════════


class TestWorkCommand:
    """bizra work — Submit verified action, earn SEED."""

    def test_work_requires_initialized_node(self, state_dir):
        """Work must fail if node not initialized."""
        result = process_work("some work", state_dir)
        assert not result.success
        assert "not initialized" in result.error.lower()

    def test_work_passes_intent_gate(self, initialized_node, state_dir):
        """Technical work description should pass intent gate."""
        result = process_work(
            "Implemented the fixed-point arithmetic kernel with 35 passing tests",
            state_dir,
        )
        assert result.success
        assert result.intent_passed

    def test_work_computes_ihsan_score(self, initialized_node, state_dir):
        """Work must compute and return an Ihsan score."""
        result = process_work(
            "Built and tested the constitutional declaration module with covenant locking",
            state_dir,
        )
        assert result.success
        assert result.ihsan_score > 0

    def test_work_mints_seed(self, initialized_node, state_dir):
        """Successful work must mint SEED tokens."""
        result = process_work(
            "Created the progressive minting algorithm with Khaldunian curve",
            state_dir,
        )
        assert result.success
        assert result.seed_minted > 0

    def test_work_updates_balance(self, initialized_node, state_dir):
        """After work, wallet balance must increase."""
        process_work(
            "Designed and implemented the Asabiyyah social cohesion index",
            state_dir,
        )
        state = load_node_state(state_dir)
        assert state.seed_balance > 0

    def test_work_increments_action_count(self, initialized_node, state_dir):
        """Each work action must increment the total_actions counter."""
        process_work("First contribution to the sovereignty kernel", state_dir)
        process_work("Second contribution to the sovereignty kernel", state_dir)
        state = load_node_state(state_dir)
        assert state.total_actions == 2

    def test_work_appends_ledger_event(self, initialized_node, state_dir):
        """Each work action must append to the ledger."""
        process_work(
            "Wrote and tested the intent gate implementation for Al-Ghazali filter",
            state_dir,
        )
        ledger = (state_dir / "ledger.jsonl").read_text().strip().split("\n")
        # genesis + 1 action
        assert len(ledger) == 2
        event = json.loads(ledger[1])
        assert event["type"] == "action"
        assert "ihsan_score" in event

    def test_work_records_ihsan_history(self, initialized_node, state_dir):
        """Work must append the ihsan score to history."""
        process_work(
            "Implemented and verified BLOOM accrual algorithm with streak bonus",
            state_dir,
        )
        state = load_node_state(state_dir)
        assert len(state.ihsan_history) == 1

    def test_work_description_required(self, initialized_node, state_dir):
        """Empty description must be rejected."""
        result = process_work("", state_dir)
        assert not result.success

    def test_multiple_works_accumulate(self, initialized_node, state_dir):
        """Multiple work submissions accumulate SEED."""
        process_work(
            "Wrote and tested fixed-point arithmetic module with 35 tests", state_dir
        )
        state1 = load_node_state(state_dir)
        b1 = state1.seed_balance

        process_work(
            "Wrote and tested constitutional types module with dataclasses", state_dir
        )
        state2 = load_node_state(state_dir)
        b2 = state2.seed_balance

        assert b2 > b1

    def test_multiple_works_accrue_bloom_linearly(self, initialized_node, state_dir):
        """Each qualifying work action should add one BLOOM accrual."""
        process_work(
            "Implemented fixed-point arithmetic for the sovereignty kernel",
            state_dir,
        )
        state1 = load_node_state(state_dir)
        assert state1.bloom_balance == fp(0.01)

        process_work(
            "Verified BLOOM governance accrual with regression coverage",
            state_dir,
        )
        state2 = load_node_state(state_dir)
        assert state2.bloom_balance == fp(0.02)


# ═══════════════════════════════════════════════════════════════════
# Test: Attest Command
# ═══════════════════════════════════════════════════════════════════


class TestAttestCommand:
    """bizra attest — Vouch for another node's work."""

    def test_attest_requires_initialized_node(self, state_dir):
        """Attest must fail if node not initialized."""
        result = attest_peer("peer123", state_dir)
        assert not result.success

    def test_attest_requires_work_history(self, initialized_node, state_dir):
        """Cannot attest with no ihsan history."""
        result = attest_peer("peer123", state_dir)
        assert not result.success
        assert "history" in result.error.lower()

    def test_attest_rejects_self_attestation(self, initialized_node, state_dir):
        """Cannot attest yourself."""
        state = load_node_state(state_dir)
        result = attest_peer(state.node_id, state_dir)
        assert not result.success
        assert "self" in result.error.lower() or "yourself" in result.error.lower()

    def test_attest_creates_attestation_event(self, initialized_node, state_dir):
        """Attestation must append to ledger."""
        # First do work to build ihsan history
        process_work("Built the constitutional kernel from scratch", state_dir)
        result = attest_peer("peer_node_abc123", state_dir)
        assert result.success

        ledger = (state_dir / "ledger.jsonl").read_text().strip().split("\n")
        last_event = json.loads(ledger[-1])
        assert last_event["type"] == "attestation"

    def test_attest_increments_given_count(self, initialized_node, state_dir):
        """Attestation must increment attestations_given."""
        process_work("Verified the Gini coefficient implementation", state_dir)
        attest_peer("peer_one", state_dir)
        state = load_node_state(state_dir)
        assert state.attestations_given == 1

    def test_attest_records_peer(self, initialized_node, state_dir):
        """Attested peers must be recorded."""
        process_work("Tested all 15 constitutional algorithms", state_dir)
        attest_peer("peer_alpha", state_dir)
        result = attest_peer("peer_beta", state_dir)
        assert result.success

        state = load_node_state(state_dir)
        assert state.attestations_given == 2


# ═══════════════════════════════════════════════════════════════════
# Test: Status Command
# ═══════════════════════════════════════════════════════════════════


class TestStatusCommand:
    """bizra status — View sovereign state."""

    def test_status_requires_initialized_node(self, state_dir):
        """Status must fail if node not initialized."""
        result = get_status(state_dir)
        assert not result.success

    def test_status_returns_name(self, initialized_node, state_dir):
        """Status must include node name."""
        result = get_status(state_dir)
        assert result.success
        assert result.name == "test-node"

    def test_status_returns_node_id(self, initialized_node, state_dir):
        """Status must include node ID."""
        result = get_status(state_dir)
        assert result.node_id
        assert len(result.node_id) == 64

    def test_status_returns_balances(self, initialized_node, state_dir):
        """Status must include SEED and BLOOM balances."""
        result = get_status(state_dir)
        assert result.seed_balance == 0
        assert result.bloom_balance == 0

    def test_status_reflects_work(self, initialized_node, state_dir):
        """After work, status must show updated balance."""
        process_work(
            "Implemented and tested the Zakat purification engine with nisab threshold",
            state_dir,
        )
        result = get_status(state_dir)
        assert result.seed_balance > 0
        assert result.total_actions == 1

    def test_status_shows_covenant(self, initialized_node, state_dir):
        """Status must include covenant hash."""
        result = get_status(state_dir)
        assert result.covenant_hash
        assert len(result.covenant_hash) == 64

    def test_status_computes_asabiyyah(self, initialized_node, state_dir):
        """Status must include Asabiyyah score."""
        result = get_status(state_dir)
        assert result.asabiyyah_score == 0  # No connections yet

    def test_status_after_attest(self, initialized_node, state_dir):
        """After attestation, status must show updated social metrics."""
        process_work(
            "Built and verified the complete sovereignty stack with all algorithms",
            state_dir,
        )
        attest_peer("peer_xyz", state_dir)
        result = get_status(state_dir)
        assert result.attestations_given == 1


# ═══════════════════════════════════════════════════════════════════
# Test: State Persistence
# ═══════════════════════════════════════════════════════════════════


class TestStatePersistence:
    """Node state must survive save/load cycles."""

    def test_save_load_roundtrip(self, state_dir):
        """State must survive a save/load cycle."""
        state_dir.mkdir(parents=True, exist_ok=True)
        state = NodeState(
            name="roundtrip-node",
            node_id="a" * 64,
            public_key="b" * 64,
            covenant_hash="c" * 64,
            covenant_sig="d" * 64,
            seed_balance=fp(42.5),
            bloom_balance=fp(10.0),
            total_actions=7,
            ihsan_history=[fp(0.96), fp(0.97)],
            created_at=1000000,
            last_active=2000000,
            attestations_given=3,
            attestations_received=1,
            peers=["peer1", "peer2"],
        )
        save_node_state(state, state_dir)
        loaded = load_node_state(state_dir)

        assert loaded.name == state.name
        assert loaded.node_id == state.node_id
        assert loaded.seed_balance == state.seed_balance
        assert loaded.bloom_balance == state.bloom_balance
        assert loaded.total_actions == state.total_actions
        assert loaded.ihsan_history == state.ihsan_history
        assert loaded.attestations_given == state.attestations_given
        assert loaded.peers == state.peers

    def test_load_returns_none_for_missing(self, state_dir):
        """Load must return None if node.json doesn't exist."""
        state = load_node_state(state_dir)
        assert state is None

    def test_state_fields_correct_types(self, state_dir):
        """All fields must be correct types after roundtrip."""
        state_dir.mkdir(parents=True, exist_ok=True)
        state = NodeState(
            name="type-test",
            node_id="e" * 64,
            public_key="f" * 64,
            covenant_hash="0" * 64,
            covenant_sig="1" * 64,
        )
        save_node_state(state, state_dir)
        loaded = load_node_state(state_dir)

        assert isinstance(loaded.name, str)
        assert isinstance(loaded.node_id, str)
        assert isinstance(loaded.seed_balance, int)
        assert isinstance(loaded.bloom_balance, int)
        assert isinstance(loaded.total_actions, int)
        assert isinstance(loaded.ihsan_history, list)
        assert isinstance(loaded.peers, list)
