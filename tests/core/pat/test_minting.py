"""
Tests for BIZRA Identity Minting Protocol

Tests cover:
1. Identity Card creation and verification
2. PAT (Personal Autonomous Task) Agent creation and capabilities
3. SAT (System Autonomous Task) Agent creation
4. Full onboarding workflow
5. Cryptographic signature verification
6. Economic model validation (7 PAT + 5 SAT = 12 agents)

Standing on Giants: pytest + Ed25519 + BLAKE3
"""

import pytest
from datetime import datetime, timezone

from core.pci.crypto import generate_keypair
from core.pat.identity_card import (
    IdentityCard,
    IdentityStatus,
    SovereigntyTier,
    generate_identity_keypair,
    _generate_node_id,
)
from core.pat.agent import (
    PATAgent,
    SATAgent,
    AgentType,
    AgentStatus,
    OwnershipType,
    DEFAULT_CAPABILITIES,
    USER_AGENT_ALLOCATION,
    SYSTEM_AGENT_ALLOCATION,
)
from core.pat.minting import (
    IdentityMinter,
    MinterState,
    OnboardingResult,
    mint_identity_card,
    mint_pat_agents,
    onboard_user,
    generate_and_onboard,
    PAT_AGENT_COUNT,
    SAT_AGENT_COUNT,
    USER_AGENT_COUNT,
    SYSTEM_AGENT_COUNT,
    TOTAL_AGENTS_PER_USER,
    SYSTEM_TREASURY_ID,
)


class TestIdentityCard:
    """Tests for IdentityCard class."""

    def test_create_identity_card(self):
        """Test creating a basic identity card."""
        _, public_key = generate_keypair()
        card = IdentityCard.create(public_key)

        assert card.node_id.startswith("BIZRA-")
        assert len(card.node_id) == 14
        assert card.public_key == public_key
        assert card.sovereignty_score == 0.0
        assert card.status == IdentityStatus.PENDING

    def test_node_id_deterministic(self):
        """Test that node_id is deterministic from public key."""
        _, public_key = generate_keypair()

        node_id_1 = _generate_node_id(public_key)
        node_id_2 = _generate_node_id(public_key)

        assert node_id_1 == node_id_2

    def test_node_id_uniqueness(self):
        """Test that different public keys produce different node_ids."""
        _, pub1 = generate_keypair()
        _, pub2 = generate_keypair()

        node_id_1 = _generate_node_id(pub1)
        node_id_2 = _generate_node_id(pub2)

        assert node_id_1 != node_id_2

    def test_sovereignty_tiers(self):
        """Test sovereignty tier calculation."""
        _, public_key = generate_keypair()

        card = IdentityCard.create(public_key)
        card.sovereignty_score = 0.0
        assert card.sovereignty_tier == SovereigntyTier.SEED

        card.sovereignty_score = 0.3
        assert card.sovereignty_tier == SovereigntyTier.SPROUT

        card.sovereignty_score = 0.6
        assert card.sovereignty_tier == SovereigntyTier.TREE

        card.sovereignty_score = 0.9
        assert card.sovereignty_tier == SovereigntyTier.FOREST

    def test_minter_signature(self):
        """Test minter signature creation and verification."""
        minter_priv, minter_pub = generate_keypair()
        _, user_pub = generate_keypair()

        card = IdentityCard.create(user_pub)
        card.sign_as_minter(minter_priv, minter_pub)

        assert card.minter_signature is not None
        assert card.minter_public_key == minter_pub
        assert card.verify_minter_signature()

    def test_self_signature(self):
        """Test owner self-signature creation and verification."""
        minter_priv, minter_pub = generate_keypair()
        user_priv, user_pub = generate_keypair()

        card = IdentityCard.create(user_pub)
        card.sign_as_minter(minter_priv, minter_pub)
        card.sign_as_owner(user_priv)

        assert card.self_signature is not None
        assert card.status == IdentityStatus.ACTIVE
        assert card.verify_self_signature()
        assert card.is_fully_verified()

    def test_serialization(self):
        """Test to_dict and from_dict roundtrip."""
        minter_priv, minter_pub = generate_keypair()
        _, user_pub = generate_keypair()

        card = IdentityCard.create(user_pub, metadata={"test": "value"})
        card.sign_as_minter(minter_priv, minter_pub)

        d = card.to_dict()
        restored = IdentityCard.from_dict(d)

        assert restored.node_id == card.node_id
        assert restored.public_key == card.public_key
        assert restored.minter_signature == card.minter_signature
        assert restored.metadata == card.metadata

    def test_invalid_public_key_length(self):
        """Test validation of public key length."""
        with pytest.raises(ValueError, match="Invalid public_key length"):
            IdentityCard.create("abc123")  # Too short

    def test_invalid_node_id_format(self):
        """Test validation of node_id format."""
        _, public_key = generate_keypair()
        with pytest.raises(ValueError, match="Invalid node_id format"):
            IdentityCard(
                node_id="INVALID-123",
                public_key=public_key,
                creation_timestamp="2026-02-03T00:00:00Z",
            )


class TestPATAgent:
    """Tests for PATAgent class (Personal Autonomous Task)."""

    def test_create_agent(self):
        """Test creating a basic PAT agent."""
        agent = PATAgent.create(
            owner_id="BIZRA-12345678",
            agent_type=AgentType.WORKER,
            index=1,
        )

        assert agent.agent_id.startswith("PAT-")
        assert agent.owner_id == "BIZRA-12345678"
        assert agent.agent_type == AgentType.WORKER
        assert agent.status == AgentStatus.DORMANT
        assert len(agent.public_key) == 64

    def test_agent_id_format(self):
        """Test agent ID format."""
        agent = PATAgent.create(
            owner_id="BIZRA-ABCD1234",
            agent_type=AgentType.RESEARCHER,
            index=5,
        )

        # Format: PAT-{owner_suffix}-{type_prefix}-{index:03d}
        assert agent.agent_id == "PAT-ABCD1234-RSC-005"

    def test_default_capabilities(self):
        """Test that agents get default capabilities for their type."""
        worker = PATAgent.create("BIZRA-12345678", AgentType.WORKER, 1)
        guardian = PATAgent.create("BIZRA-12345678", AgentType.GUARDIAN, 2)

        assert "task.execute" in worker.capabilities
        assert "validate.ihsan" in guardian.capabilities

    def test_capability_management(self):
        """Test granting and revoking capabilities."""
        agent = PATAgent.create("BIZRA-12345678", AgentType.WORKER, 1)

        # Grant new capability
        assert agent.grant_capability("custom.action")
        assert agent.has_capability("custom.action")

        # Duplicate grant returns False
        assert not agent.grant_capability("custom.action")

        # Revoke capability
        assert agent.revoke_capability("custom.action")
        assert not agent.has_capability("custom.action")

    def test_agent_lifecycle(self):
        """Test agent status transitions."""
        agent = PATAgent.create("BIZRA-12345678", AgentType.WORKER, 1)

        assert agent.status == AgentStatus.DORMANT
        assert not agent.is_active

        assert agent.activate()
        assert agent.status == AgentStatus.ACTIVE
        assert agent.is_active

        assert agent.suspend()
        assert agent.status == AgentStatus.SUSPENDED

        assert agent.retire()
        assert agent.status == AgentStatus.RETIRED

    def test_task_completion_tracking(self):
        """Test success rate tracking."""
        agent = PATAgent.create("BIZRA-12345678", AgentType.WORKER, 1)

        assert agent.task_count == 0
        assert agent.success_rate == 1.0

        agent.record_task_completion(success=True)
        assert agent.task_count == 1

        agent.record_task_completion(success=False)
        assert agent.task_count == 2
        assert agent.success_rate < 1.0

    def test_user_ownership(self):
        """Test PAT agent user ownership."""
        agent = PATAgent.create(
            owner_id="BIZRA-12345678",
            agent_type=AgentType.WORKER,
            index=1,
            ownership_type=OwnershipType.USER,
        )

        assert agent.is_user_owned
        assert not agent.is_system_owned

    def test_minter_signature(self):
        """Test agent minter signature."""
        minter_priv, minter_pub = generate_keypair()

        agent = PATAgent.create("BIZRA-12345678", AgentType.WORKER, 1)
        agent.sign_as_minter(minter_priv, minter_pub)

        assert agent.minter_signature is not None
        assert agent.verify_minter_signature()


class TestSATAgent:
    """Tests for SATAgent class (System Autonomous Task)."""

    def test_create_sat_agent(self):
        """Test creating a SAT agent."""
        agent = SATAgent.create(
            agent_type=AgentType.VALIDATOR,
            index=1,
            contribution_source="BIZRA-12345678",
        )

        assert agent.agent_id.startswith("SAT-")
        assert agent.owner_id == SYSTEM_TREASURY_ID
        assert agent.ownership_type == OwnershipType.SYSTEM
        assert agent.contribution_source == "BIZRA-12345678"

    def test_sat_agent_always_system_owned(self):
        """Test that SAT agents are always system-owned."""
        agent = SATAgent.create(
            agent_type=AgentType.GUARDIAN,
            index=1,
            contribution_source="BIZRA-12345678",
        )

        # owner_id and ownership_type should always be SYSTEM
        assert agent.owner_id == SYSTEM_TREASURY_ID
        assert agent.ownership_type == OwnershipType.SYSTEM

    def test_sat_agent_task_pool(self):
        """Test SAT agent task pool assignment."""
        agent = SATAgent.create(
            agent_type=AgentType.COORDINATOR,
            index=1,
            contribution_source="BIZRA-12345678",
            task_pool="consensus",
            federation_assignment="node0",
        )

        assert agent.task_pool == "consensus"
        assert agent.federation_assignment == "node0"

    def test_sat_agent_signature(self):
        """Test SAT agent minter signature."""
        minter_priv, minter_pub = generate_keypair()

        agent = SATAgent.create(
            agent_type=AgentType.VALIDATOR,
            index=1,
            contribution_source="BIZRA-12345678",
        )
        agent.sign_as_minter(minter_priv, minter_pub)

        assert agent.minter_signature is not None
        assert agent.verify_minter_signature()


class TestIdentityMinter:
    """Tests for IdentityMinter class."""

    def test_create_minter(self):
        """Test creating a new minter."""
        minter = IdentityMinter.create()

        assert len(minter.public_key) == 64
        assert minter.current_block == 0

    def test_mint_identity_card(self):
        """Test minting a single identity card."""
        minter = IdentityMinter.create()
        _, user_pub = generate_keypair()

        card = minter.mint_identity_card(user_pub)

        assert card.node_id.startswith("BIZRA-")
        assert card.verify_minter_signature()
        assert minter.stats["total_identities_minted"] == 1

    def test_mint_pat_agents(self):
        """Test minting PAT agents."""
        minter = IdentityMinter.create()

        agents = minter.mint_pat_agents(
            owner_id="BIZRA-12345678",
            agent_types=[AgentType.WORKER, AgentType.RESEARCHER],
        )

        assert len(agents) == 2
        assert all(a.verify_minter_signature() for a in agents)
        assert all(a.agent_id.startswith("PAT-") for a in agents)
        assert minter.stats["total_agents_minted"] == 2

    def test_mint_sat_agents(self):
        """Test minting SAT agents."""
        minter = IdentityMinter.create()

        agents = minter.mint_sat_agents(
            contribution_source="BIZRA-12345678",
            agent_types=[AgentType.VALIDATOR, AgentType.GUARDIAN],
            task_pool="consensus",
        )

        assert len(agents) == 2
        assert all(a.verify_minter_signature() for a in agents)
        assert all(a.agent_id.startswith("SAT-") for a in agents)
        assert all(a.owner_id == SYSTEM_TREASURY_ID for a in agents)

    def test_invalid_public_key(self):
        """Test rejection of invalid public key."""
        minter = IdentityMinter.create()

        with pytest.raises(ValueError, match="Invalid public key length"):
            minter.mint_identity_card("short")

        with pytest.raises(ValueError, match="valid hexadecimal"):
            minter.mint_identity_card("g" * 64)  # Invalid hex


class TestOnboarding:
    """Tests for full onboarding workflow."""

    def test_full_onboarding(self):
        """Test complete user onboarding."""
        _, user_pub = generate_keypair()
        result = onboard_user(user_pub)

        assert result.success
        assert result.identity_card is not None
        assert result.pat_agent_count == PAT_AGENT_COUNT
        assert result.sat_agent_count == SAT_AGENT_COUNT
        assert result.total_agents_minted == TOTAL_AGENTS_PER_USER

    def test_economic_model_7_5_split(self):
        """Test the 7 PAT + 5 SAT = 12 economic model."""
        _, user_pub = generate_keypair()
        result = onboard_user(user_pub)

        # User gets 7 PAT agents (58.3%)
        assert len(result.pat_agents) == 7
        assert all(a.agent_id.startswith("PAT-") for a in result.pat_agents)
        assert all(a.is_user_owned for a in result.pat_agents)

        # System gets 5 SAT agents (41.7%)
        assert len(result.sat_agents) == 5
        assert all(a.agent_id.startswith("SAT-") for a in result.sat_agents)
        assert all(a.owner_id == SYSTEM_TREASURY_ID for a in result.sat_agents)

        # Total is 12
        assert result.total_agents_minted == 12

    def test_backward_compatibility_aliases(self):
        """Test backward compatibility aliases for agent counts."""
        _, user_pub = generate_keypair()
        result = onboard_user(user_pub)

        # Test aliases
        assert result.user_agents == result.pat_agents
        assert result.system_agents == result.sat_agents
        assert result.user_agent_count == result.pat_agent_count
        assert result.system_agent_count == result.sat_agent_count

        # Test constants
        assert USER_AGENT_COUNT == PAT_AGENT_COUNT
        assert SYSTEM_AGENT_COUNT == SAT_AGENT_COUNT

    def test_pat_agent_allocation(self):
        """Test that user gets the correct PAT agent types."""
        _, user_pub = generate_keypair()
        result = onboard_user(user_pub)

        user_types = [a.agent_type for a in result.pat_agents]

        # Should have 2 workers, 1 researcher, 1 guardian, etc.
        assert user_types.count(AgentType.WORKER) == 2
        assert user_types.count(AgentType.RESEARCHER) == 1
        assert user_types.count(AgentType.GUARDIAN) == 1
        assert user_types.count(AgentType.SYNTHESIZER) == 1
        assert user_types.count(AgentType.VALIDATOR) == 1
        assert user_types.count(AgentType.COORDINATOR) == 1

    def test_sat_agent_allocation(self):
        """Test that system gets the correct SAT agent types."""
        _, user_pub = generate_keypair()
        result = onboard_user(user_pub)

        system_types = [a.agent_type for a in result.sat_agents]

        assert AgentType.VALIDATOR in system_types
        assert AgentType.GUARDIAN in system_types
        assert AgentType.COORDINATOR in system_types
        assert AgentType.EXECUTOR in system_types
        assert AgentType.SYNTHESIZER in system_types

    def test_sat_agent_contribution_source(self):
        """Test that SAT agents track their contribution source."""
        _, user_pub = generate_keypair()
        result = onboard_user(user_pub)

        # All SAT agents should reference the onboarded user
        for agent in result.sat_agents:
            assert agent.contribution_source == result.identity_card.node_id

    def test_signature_verification(self):
        """Test that all signatures are valid after onboarding."""
        minter = IdentityMinter.create()
        _, user_pub = generate_keypair()

        result = minter.onboard_user(user_pub)
        report = minter.verify_onboarding(result)

        assert report["all_valid"]
        assert report["identity_card_valid"]
        assert all(report["pat_agents_valid"])
        assert all(report["sat_agents_valid"])

    def test_auto_activate(self):
        """Test auto-activation of agents."""
        minter = IdentityMinter.create()
        _, user_pub = generate_keypair()

        result = minter.onboard_user(user_pub, auto_activate=True)

        assert all(a.is_active for a in result.pat_agents)
        assert all(a.is_active for a in result.sat_agents)

    def test_generate_and_onboard(self):
        """Test the complete generate + onboard helper."""
        private_key, public_key, result = generate_and_onboard()

        assert len(private_key) == 64
        assert len(public_key) == 64
        assert result.success
        assert result.identity_card.is_fully_verified()

    def test_block_number_increment(self):
        """Test that block numbers increment correctly."""
        minter = IdentityMinter.create()

        _, pub1 = generate_keypair()
        _, pub2 = generate_keypair()

        result1 = minter.onboard_user(pub1)
        result2 = minter.onboard_user(pub2)

        assert result1.block_number == 1
        assert result2.block_number == 2
        assert minter.current_block == 2

    def test_onboarding_result_serialization(self):
        """Test OnboardingResult serialization."""
        _, user_pub = generate_keypair()
        result = onboard_user(user_pub)

        d = result.to_dict()

        assert d["success"]
        assert d["identity_card"]["node_id"].startswith("BIZRA-")
        assert len(d["pat_agents"]) == PAT_AGENT_COUNT
        assert len(d["sat_agents"]) == SAT_AGENT_COUNT
        assert d["summary"]["total_agents"] == TOTAL_AGENTS_PER_USER
        assert d["summary"]["pat_percentage"] == pytest.approx(58.3, abs=0.1)
        assert d["summary"]["sat_percentage"] == pytest.approx(41.7, abs=0.1)

    def test_failed_onboarding(self):
        """Test onboarding failure handling."""
        minter = IdentityMinter.create()

        # Invalid public key should cause failure
        result = minter.onboard_user("invalid")

        assert not result.success
        assert result.error is not None
        assert result.identity_card is None


class TestConvenienceFunctions:
    """Tests for module-level convenience functions."""

    def test_mint_identity_card_function(self):
        """Test standalone mint_identity_card function."""
        _, user_pub = generate_keypair()
        card, state = mint_identity_card(user_pub)

        assert card.node_id.startswith("BIZRA-")
        assert card.verify_minter_signature()
        assert state.total_identities_minted == 1

    def test_mint_pat_agents_function(self):
        """Test standalone mint_pat_agents function."""
        agents, state = mint_pat_agents("BIZRA-12345678", count=5)

        assert len(agents) == 5
        assert state.total_agents_minted == 5


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_sovereignty_score_boundaries(self):
        """Test sovereignty score at boundaries."""
        _, public_key = generate_keypair()

        # Valid boundaries
        card = IdentityCard.create(public_key)
        card.sovereignty_score = 0.0
        assert card.sovereignty_tier == SovereigntyTier.SEED

        card.sovereignty_score = 1.0
        assert card.sovereignty_tier == SovereigntyTier.FOREST

        # Invalid boundaries
        with pytest.raises(ValueError):
            IdentityCard(
                node_id="BIZRA-12345678",
                public_key=public_key,
                creation_timestamp="2026-02-03T00:00:00Z",
                sovereignty_score=1.1,
            )

    def test_empty_metadata(self):
        """Test handling of empty metadata."""
        _, public_key = generate_keypair()
        card = IdentityCard.create(public_key)

        assert card.metadata == {}

        d = card.to_dict()
        restored = IdentityCard.from_dict(d)
        assert restored.metadata == {}

    def test_large_agent_index(self):
        """Test agent creation with large index."""
        agent = PATAgent.create(
            owner_id="BIZRA-12345678",
            agent_type=AgentType.WORKER,
            index=999,
        )

        assert "999" in agent.agent_id

    def test_digest_stability(self):
        """Test that digest is stable for same input."""
        _, public_key = generate_keypair()
        card = IdentityCard.create(public_key)

        digest1 = card.compute_digest()
        digest2 = card.compute_digest()

        assert digest1 == digest2

    def test_sat_agent_id_validation(self):
        """Test SAT agent ID format validation."""
        with pytest.raises(ValueError, match="Invalid SAT agent_id format"):
            SATAgent(
                agent_id="PAT-12345678-VAL-001",  # Wrong prefix
                agent_type=AgentType.VALIDATOR,
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
