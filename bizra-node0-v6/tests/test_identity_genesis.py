"""Tests for BIZRA Identity Genesis — Sovereign Node Identity."""

import json
import os
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from identity_genesis import (
    NACL_AVAILABLE,
    AgentKey,
    NodeIdentity,
    create_identity,
    load_public_record,
    save_identity,
)


@pytest.fixture
def identity():
    return create_identity()


class TestGenesisEvent:
    """Identity creation — the Genesis Event."""

    def test_creates_identity(self):
        ident = create_identity()
        assert isinstance(ident, NodeIdentity)

    def test_node_id_is_64_hex_chars(self, identity):
        assert len(identity.node_id) == 64
        int(identity.node_id, 16)  # Must be valid hex

    def test_public_key_is_64_hex_chars(self, identity):
        assert len(identity.public_key_hex) == 64
        int(identity.public_key_hex, 16)

    def test_genesis_timestamp_is_recent(self, identity):
        import time

        assert abs(time.time() - identity.genesis_timestamp) < 5

    def test_genesis_domain_is_set(self, identity):
        assert "identity-genesis" in identity.genesis_domain

    def test_node_id_is_deterministic_from_pubkey(self, identity):
        import hashlib

        expected = hashlib.sha256(bytes.fromhex(identity.public_key_hex)).hexdigest()
        assert identity.node_id == expected

    def test_two_identities_are_different(self):
        a = create_identity()
        b = create_identity()
        assert a.node_id != b.node_id
        assert a.public_key_hex != b.public_key_hex


class TestAgentKeys:
    """HD key derivation for PAT and SAT agents."""

    def test_seven_pat_agents(self, identity):
        assert len(identity.pat_agents) == 7

    def test_five_sat_agents(self, identity):
        assert len(identity.sat_agents) == 5

    def test_total_agents_is_twelve(self, identity):
        assert identity.total_agents == 12

    def test_pat_agent_names(self, identity):
        names = [a.agent_name for a in identity.pat_agents]
        assert names == [
            "Planner",
            "Researcher",
            "Coder",
            "Evaluator",
            "Ethicist",
            "Publisher",
            "Integrator",
        ]

    def test_sat_agent_names(self, identity):
        names = [a.agent_name for a in identity.sat_agents]
        assert names == [
            "ComputeScheduler",
            "SecurityMonitor",
            "PerformanceAnalyzer",
            "ConsensusValidator",
            "NetworkOrchestrator",
        ]

    def test_agent_types_correct(self, identity):
        for a in identity.pat_agents:
            assert a.agent_type == "pat"
        for a in identity.sat_agents:
            assert a.agent_type == "sat"

    def test_derivation_indices_sequential(self, identity):
        all_agents = identity.pat_agents + identity.sat_agents
        indices = [a.derivation_index for a in all_agents]
        assert indices == list(range(12))

    def test_all_public_keys_unique(self, identity):
        keys = [a.public_key_hex for a in identity.pat_agents + identity.sat_agents]
        assert len(set(keys)) == 12

    def test_agent_keys_differ_from_master(self, identity):
        for a in identity.pat_agents + identity.sat_agents:
            assert a.public_key_hex != identity.public_key_hex

    def test_get_agent_by_name(self, identity):
        coder = identity.get_agent("Coder")
        assert coder is not None
        assert coder.agent_name == "Coder"
        assert coder.agent_type == "pat"

    def test_get_nonexistent_agent(self, identity):
        assert identity.get_agent("NonExistent") is None


class TestDomainSeparatedSigning:
    """Cryptographic signing with domain separation."""

    def test_master_sign_returns_bytes(self, identity):
        sig = identity.sign_master(b"test message", "test-domain")
        assert isinstance(sig, bytes)
        assert len(sig) > 0

    def test_agent_sign_returns_bytes(self, identity):
        coder = identity.get_agent("Coder")
        sig = coder.sign(b"test", "test-domain")
        assert isinstance(sig, bytes)
        assert len(sig) > 0

    def test_different_domains_produce_different_signatures(self, identity):
        msg = b"same message"
        sig_a = identity.sign_master(msg, "domain-a")
        sig_b = identity.sign_master(msg, "domain-b")
        assert sig_a != sig_b

    def test_different_messages_produce_different_signatures(self, identity):
        sig_a = identity.sign_master(b"message_a", "domain")
        sig_b = identity.sign_master(b"message_b", "domain")
        assert sig_a != sig_b

    def test_different_agents_produce_different_signatures(self, identity):
        msg = b"same message"
        domain = "test-domain"
        sig_planner = identity.get_agent("Planner").sign(msg, domain)
        sig_coder = identity.get_agent("Coder").sign(msg, domain)
        assert sig_planner != sig_coder

    @pytest.mark.skipif(not NACL_AVAILABLE, reason="PyNaCl not installed")
    def test_master_verify_valid_signature(self, identity):
        msg = b"verified message"
        domain = "verify-domain"
        sig = identity.sign_master(msg, domain)
        assert identity.verify_master(msg, sig, domain) is True

    @pytest.mark.skipif(not NACL_AVAILABLE, reason="PyNaCl not installed")
    def test_master_verify_rejects_wrong_message(self, identity):
        msg = b"original"
        domain = "verify-domain"
        sig = identity.sign_master(msg, domain)
        assert identity.verify_master(b"tampered", sig, domain) is False

    @pytest.mark.skipif(not NACL_AVAILABLE, reason="PyNaCl not installed")
    def test_master_verify_rejects_wrong_domain(self, identity):
        msg = b"cross-domain"
        sig = identity.sign_master(msg, "domain-a")
        assert identity.verify_master(msg, sig, "domain-b") is False

    @pytest.mark.skipif(not NACL_AVAILABLE, reason="PyNaCl not installed")
    def test_agent_verify_valid_signature(self, identity):
        coder = identity.get_agent("Coder")
        msg = b"agent signed"
        domain = "bizra-evidence-v1"
        sig = coder.sign(msg, domain)
        assert coder.verify(msg, sig, domain) is True


class TestPublicRecord:
    """Public-safe identity serialization."""

    def test_as_public_record_has_node_id(self, identity):
        r = identity.as_public_record()
        assert r["node_id"] == identity.node_id

    def test_as_public_record_has_agents(self, identity):
        r = identity.as_public_record()
        assert len(r["pat_agents"]) == 7
        assert len(r["sat_agents"]) == 5

    def test_public_record_has_no_private_keys(self, identity):
        r = json.dumps(identity.as_public_record())
        # Private keys should never appear in public record
        assert "_signing_key" not in r
        assert "master_signing" not in r

    def test_public_record_agent_format(self, identity):
        r = identity.as_public_record()
        agent = r["pat_agents"][0]
        assert "name" in agent
        assert "public_key" in agent
        assert len(agent) == 2  # Only name and public_key


class TestPersistence:
    """Identity save/load."""

    def test_save_creates_file(self, identity, tmp_path):
        path = tmp_path / "identity.json"
        save_identity(identity, path)
        assert path.exists()

    def test_saved_file_is_valid_json(self, identity, tmp_path):
        path = tmp_path / "identity.json"
        save_identity(identity, path)
        data = json.loads(path.read_text())
        assert data["node_id"] == identity.node_id

    def test_load_public_record(self, identity, tmp_path):
        path = tmp_path / "identity.json"
        save_identity(identity, path)
        loaded = load_public_record(path)
        assert loaded["node_id"] == identity.node_id
        assert loaded["total_agents"] == 12

    def test_save_creates_parent_dirs(self, identity, tmp_path):
        path = tmp_path / "deep" / "nested" / "identity.json"
        save_identity(identity, path)
        assert path.exists()
