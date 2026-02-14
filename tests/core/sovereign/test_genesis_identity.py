"""
Tests for Genesis Identity Loader
==================================
Verifies that Node0 loads its persistent identity from the genesis ceremony.

Standing on Giants: Al-Ghazali (1095), Lamport (1982), Nakamoto (2008)
"""

import json
import pytest
from pathlib import Path
from unittest.mock import patch

from core.sovereign.genesis_identity import (
    AgentIdentity,
    GenesisState,
    NodeIdentity,
    load_and_validate_genesis,
    load_genesis,
    validate_genesis_hash,
)


# =============================================================================
# FIXTURES
# =============================================================================

SAMPLE_GENESIS = {
    "identity": {
        "node_id": "node0_test123",
        "public_key": "ed25519_pub_abc123",
        "name": "TestNode",
        "location": "Test City",
        "created_at": 1738000000,
        "identity_hash": list(b"\x01\x02\x03"),
    },
    "pat_team": {
        "agents": [
            {
                "agent_id": "pat_strategist_001",
                "role": "Strategist",
                "public_key": "ed25519_pat_strat",
                "capabilities": ["planning", "analysis"],
                "giants": ["Sun Tzu", "Clausewitz"],
                "created_at": 1738000001,
                "agent_hash": list(b"\x10\x20"),
            },
            {
                "agent_id": "pat_developer_002",
                "role": "Developer",
                "public_key": "ed25519_pat_dev",
                "capabilities": ["coding", "review"],
                "giants": ["Knuth", "Dijkstra"],
                "created_at": 1738000002,
                "agent_hash": list(b"\x30\x40"),
            },
        ],
        "team_hash": list(b"\xaa\xbb"),
    },
    "sat_team": {
        "agents": [
            {
                "agent_id": "sat_validator_001",
                "role": "Validator",
                "public_key": "ed25519_sat_val",
                "capabilities": ["validation"],
                "giants": ["Lamport"],
                "created_at": 1738000003,
                "agent_hash": list(b"\x50\x60"),
            },
        ],
        "team_hash": list(b"\xcc\xdd"),
        "governance": {
            "quorum": 0.67,
            "voting_period_hours": 72,
            "upgrade_threshold": 0.8,
        },
    },
    "partnership_hash": list(b"\xee\xff"),
    "genesis_hash": list(b"\x01\x23\x45\x67\x89\xab\xcd\xef"),
    "hardware": {"gpu": "RTX 4090", "ram_gb": 128},
    "knowledge": {"conversations": 100},
    "timestamp": 1738000000,
}


@pytest.fixture
def genesis_dir(tmp_path):
    """Create a temporary genesis directory with test data."""
    genesis_file = tmp_path / "node0_genesis.json"
    genesis_file.write_text(json.dumps(SAMPLE_GENESIS, indent=2))

    # Write hash file matching the genesis_hash
    hash_file = tmp_path / "genesis_hash.txt"
    genesis_hash = bytes(SAMPLE_GENESIS["genesis_hash"]).hex()
    hash_file.write_text(genesis_hash)

    return tmp_path


@pytest.fixture
def empty_dir(tmp_path):
    """Create an empty state directory (no genesis)."""
    return tmp_path


# =============================================================================
# TESTS: NodeIdentity
# =============================================================================


class TestNodeIdentity:
    def test_from_dict(self):
        data = SAMPLE_GENESIS["identity"]
        identity = NodeIdentity.from_dict(data)
        assert identity.node_id == "node0_test123"
        assert identity.public_key == "ed25519_pub_abc123"
        assert identity.name == "TestNode"
        assert identity.location == "Test City"
        assert identity.created_at == 1738000000

    def test_from_dict_minimal(self):
        data = {"node_id": "minimal", "public_key": "key123"}
        identity = NodeIdentity.from_dict(data)
        assert identity.node_id == "minimal"
        assert identity.name == ""
        assert identity.location == ""


# =============================================================================
# TESTS: AgentIdentity
# =============================================================================


class TestAgentIdentity:
    def test_from_dict(self):
        data = SAMPLE_GENESIS["pat_team"]["agents"][0]
        agent = AgentIdentity.from_dict(data)
        assert agent.agent_id == "pat_strategist_001"
        assert agent.role == "Strategist"
        assert "planning" in agent.capabilities
        assert "Sun Tzu" in agent.giants

    def test_from_dict_minimal(self):
        data = {
            "agent_id": "min_agent",
            "role": "Tester",
            "public_key": "key",
        }
        agent = AgentIdentity.from_dict(data)
        assert agent.capabilities == []
        assert agent.giants == []


# =============================================================================
# TESTS: GenesisState
# =============================================================================


class TestGenesisState:
    def test_properties(self, genesis_dir):
        state = load_genesis(genesis_dir)
        assert state is not None
        assert state.node_id == "node0_test123"
        assert state.node_name == "TestNode"

    def test_pat_agent_ids(self, genesis_dir):
        state = load_genesis(genesis_dir)
        ids = state.pat_agent_ids
        assert len(ids) == 2
        assert "pat_strategist_001" in ids
        assert "pat_developer_002" in ids

    def test_sat_agent_ids(self, genesis_dir):
        state = load_genesis(genesis_dir)
        ids = state.sat_agent_ids
        assert len(ids) == 1
        assert "sat_validator_001" in ids

    def test_get_agent_found(self, genesis_dir):
        state = load_genesis(genesis_dir)
        agent = state.get_agent("pat_strategist_001")
        assert agent is not None
        assert agent.role == "Strategist"

    def test_get_agent_not_found(self, genesis_dir):
        state = load_genesis(genesis_dir)
        assert state.get_agent("nonexistent") is None

    def test_get_agents_by_role(self, genesis_dir):
        state = load_genesis(genesis_dir)
        devs = state.get_agents_by_role("Developer")
        assert len(devs) == 1
        assert devs[0].agent_id == "pat_developer_002"

    def test_summary(self, genesis_dir):
        state = load_genesis(genesis_dir)
        summary = state.summary()
        assert summary["node_id"] == "node0_test123"
        assert summary["name"] == "TestNode"
        assert summary["pat_agents"] == 2
        assert summary["sat_agents"] == 1

    def test_governance(self, genesis_dir):
        state = load_genesis(genesis_dir)
        assert state.quorum == 0.67
        assert state.voting_period_hours == 72
        assert state.upgrade_threshold == 0.8


# =============================================================================
# TESTS: load_genesis
# =============================================================================


class TestLoadGenesis:
    def test_loads_valid_genesis(self, genesis_dir):
        state = load_genesis(genesis_dir)
        assert state is not None
        assert state.node_id == "node0_test123"
        assert len(state.pat_team) == 2
        assert len(state.sat_team) == 1
        assert state.hardware["gpu"] == "RTX 4090"
        assert state.knowledge["conversations"] == 100

    def test_returns_none_when_no_file(self, empty_dir):
        state = load_genesis(empty_dir)
        assert state is None

    def test_raises_on_corrupted_json(self, tmp_path):
        genesis_file = tmp_path / "node0_genesis.json"
        genesis_file.write_text("{invalid json")
        with pytest.raises(ValueError, match="corrupted"):
            load_genesis(tmp_path)

    def test_raises_on_missing_node_id(self, tmp_path):
        genesis_file = tmp_path / "node0_genesis.json"
        genesis_file.write_text(json.dumps({"identity": {"public_key": "key"}}))
        with pytest.raises(ValueError, match="missing node_id"):
            load_genesis(tmp_path)

    def test_hashes_parsed(self, genesis_dir):
        state = load_genesis(genesis_dir)
        assert state.genesis_hash == bytes([0x01, 0x23, 0x45, 0x67, 0x89, 0xAB, 0xCD, 0xEF])
        assert state.pat_team_hash == bytes([0xAA, 0xBB])
        assert state.sat_team_hash == bytes([0xCC, 0xDD])
        assert state.partnership_hash == bytes([0xEE, 0xFF])


# =============================================================================
# TESTS: validate_genesis_hash
# =============================================================================


class TestValidateGenesisHash:
    def test_valid_hash(self, genesis_dir):
        state = load_genesis(genesis_dir)
        assert validate_genesis_hash(state, genesis_dir) is True

    def test_invalid_hash(self, genesis_dir):
        # Overwrite hash file with wrong value
        (genesis_dir / "genesis_hash.txt").write_text("deadbeef")
        state = load_genesis(genesis_dir)
        assert validate_genesis_hash(state, genesis_dir) is False

    def test_no_hash_file(self, genesis_dir):
        (genesis_dir / "genesis_hash.txt").unlink()
        state = load_genesis(genesis_dir)
        assert validate_genesis_hash(state, genesis_dir) is False


# =============================================================================
# TESTS: load_and_validate_genesis
# =============================================================================


class TestLoadAndValidateGenesis:
    def test_full_load_and_validate(self, genesis_dir):
        state = load_and_validate_genesis(genesis_dir)
        assert state is not None
        assert state.node_id == "node0_test123"

    def test_returns_none_for_empty_dir(self, empty_dir):
        assert load_and_validate_genesis(empty_dir) is None

    def test_loads_even_with_hash_mismatch(self, genesis_dir):
        (genesis_dir / "genesis_hash.txt").write_text("deadbeef")
        state = load_and_validate_genesis(genesis_dir)
        assert state is not None  # Still loads, just warns


# =============================================================================
# TESTS: Real genesis files (integration)
# =============================================================================


class TestRealGenesis:
    """Test against actual sovereign_state/ files if available."""

    @pytest.fixture
    def real_state_dir(self):
        state_dir = Path("sovereign_state")
        if not (state_dir / "node0_genesis.json").exists():
            pytest.skip("No real genesis files available")
        return state_dir

    def test_real_genesis_loads(self, real_state_dir):
        state = load_and_validate_genesis(real_state_dir)
        assert state is not None
        assert state.node_id == "node0_ce5af35c848ce889"
        assert state.node_name == "MoMo (محمد)"

    def test_real_genesis_pat_team(self, real_state_dir):
        state = load_genesis(real_state_dir)
        assert len(state.pat_team) == 7
        roles = {a.role for a in state.pat_team}
        assert roles == {"Strategist", "Researcher", "Developer", "Analyst", "Reviewer", "Executor", "Guardian"}

    def test_real_genesis_sat_team(self, real_state_dir):
        state = load_genesis(real_state_dir)
        assert len(state.sat_team) == 5
        roles = {a.role for a in state.sat_team}
        assert roles == {"Validator", "Oracle", "Mediator", "Archivist", "Sentinel"}

    def test_real_genesis_hash_valid(self, real_state_dir):
        state = load_genesis(real_state_dir)
        assert validate_genesis_hash(state, real_state_dir) is True
