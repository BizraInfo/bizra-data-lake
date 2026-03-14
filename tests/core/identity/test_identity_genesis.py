"""Tests for Identity Genesis -- Phase 61 Step 1.

11 TDD anchors validating Definition 1.6 (Identity) and Definition 1.7 (Node Body).

Standing on Giants: Bernstein (Ed25519) | BIP-32 (HD derivation)
"""

from __future__ import annotations

import hashlib
import os

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
import pytest

from core.identity.genesis import (
    GENESIS_SIGNATURE_DOMAIN,
    GenesisWalletState,
    HumanAttestation,
    IdentityGenesis,
    NodeBody,
    PersonaSeed,
    SovereigntyClass,
    SovereigntyScope,
    derive_agent_keypairs,
    derive_identity_id,
)


class TestIdentityDerivation:
    """P1: Identity ID is deterministic SHA-256 of public key."""

    def test_identity_id_is_sha256_of_public_key(self):
        pk = os.urandom(32)
        identity_id = derive_identity_id(pk)
        expected = hashlib.sha256(pk).hexdigest()
        assert identity_id == expected

    def test_identity_id_deterministic(self):
        pk = os.urandom(32)
        id1 = derive_identity_id(pk)
        id2 = derive_identity_id(pk)
        assert id1 == id2

    def test_same_pk_same_id(self):
        pk = os.urandom(32)
        genesis1 = IdentityGenesis.create(pk)
        genesis2 = IdentityGenesis.create(pk)
        assert genesis1.identity_id == genesis2.identity_id
        assert genesis1.public_key == genesis2.public_key

    def test_different_pk_different_id(self):
        pk1 = os.urandom(32)
        pk2 = os.urandom(32)
        genesis1 = IdentityGenesis.create(pk1)
        genesis2 = IdentityGenesis.create(pk2)
        assert genesis1.identity_id != genesis2.identity_id


class TestAgentKeypairs:
    """P3: HD-derived agent keypairs."""

    def test_agent_keypairs_count_12(self):
        seed = os.urandom(32)
        keypairs = derive_agent_keypairs(seed)
        assert len(keypairs) == 12

    def test_agent_keypairs_deterministic(self):
        seed = os.urandom(32)
        kp1 = derive_agent_keypairs(seed)
        kp2 = derive_agent_keypairs(seed)
        assert kp1 == kp2

    def test_agent_keypairs_all_distinct(self):
        seed = os.urandom(32)
        keypairs = derive_agent_keypairs(seed)
        public_keys = [pk for pk, _sk in keypairs]
        assert len(set(public_keys)) == 12

    def test_different_seeds_different_keypairs(self):
        seed1 = os.urandom(32)
        seed2 = os.urandom(32)
        kp1 = derive_agent_keypairs(seed1)
        kp2 = derive_agent_keypairs(seed2)
        assert kp1 != kp2


class TestSovereigntyClass:
    """P4: Sovereignty class ordering and bounds."""

    def test_sovereignty_class_monotonic(self):
        assert SovereigntyClass.SEED < SovereigntyClass.SPROUT
        assert SovereigntyClass.SPROUT < SovereigntyClass.TREE
        assert SovereigntyClass.TREE < SovereigntyClass.FOREST

    def test_sovereignty_class_bounded_0_3(self):
        assert SovereigntyClass.SEED == 0
        assert SovereigntyClass.FOREST == 3
        assert len(SovereigntyClass) == 4

    def test_genesis_starts_at_seed(self):
        pk = os.urandom(32)
        genesis = IdentityGenesis.create(pk)
        assert genesis.sovereignty_class == SovereigntyClass.SEED


class TestNodeBody:
    """Definition 1.7: Node Body resource inventory."""

    def test_body_surplus_non_negative(self):
        body = NodeBody(
            cpu_cores=4,
            gpu_vram_mb=0,
            ram_bytes=8_000_000_000,
            disk_bytes=100_000_000_000,
        )
        util = {
            "cpu_used": 8,
            "gpu_used_mb": 0,
            "ram_used": 16_000_000_000,
            "disk_used": 200_000_000_000,
        }
        surplus = body.surplus(util)
        assert surplus["cpu_free"] >= 0
        assert surplus["gpu_vram_free_mb"] >= 0
        assert surplus["ram_free_bytes"] >= 0
        assert surplus["disk_free_bytes"] >= 0

    def test_body_surplus_no_util(self):
        body = NodeBody(
            cpu_cores=16,
            gpu_vram_mb=24000,
            ram_bytes=128_000_000_000,
            disk_bytes=1_000_000_000_000,
        )
        surplus = body.surplus()
        assert surplus["cpu_free"] == 16
        assert surplus["gpu_vram_free_mb"] == 24000
        assert surplus["ram_free_bytes"] == 128_000_000_000
        assert surplus["disk_free_bytes"] == 1_000_000_000_000

    def test_body_can_execute_with_model(self):
        body = NodeBody(
            cpu_cores=16,
            gpu_vram_mb=24000,
            gpu_compute_cap=8.9,
            ram_bytes=128_000_000_000,
            disk_bytes=1_000_000_000_000,
            loaded_models={"phi3:mini", "mxbai-embed-large"},
        )
        assert body.can_execute_local(
            {
                "required_models": {"phi3:mini"},
                "min_gpu_vram_mb": 8000,
            }
        )

    def test_body_cannot_execute_without_model(self):
        body = NodeBody(
            cpu_cores=16,
            gpu_vram_mb=24000,
            gpu_compute_cap=8.9,
            ram_bytes=128_000_000_000,
            disk_bytes=1_000_000_000_000,
            loaded_models={"phi3:mini"},
        )
        assert not body.can_execute_local(
            {
                "required_models": {"llama3.1-70b"},
                "min_gpu_vram_mb": 40000,
            }
        )

    def test_body_cpu_only_delegates_gpu_missions(self):
        body = NodeBody(
            cpu_cores=8,
            gpu_vram_mb=0,
            ram_bytes=16_000_000_000,
            disk_bytes=500_000_000_000,
        )
        assert not body.can_execute_local({"min_gpu_vram_mb": 1})

    def test_body_empty_requirements_always_passes(self):
        body = NodeBody(
            cpu_cores=2,
            gpu_vram_mb=0,
            ram_bytes=4_000_000_000,
            disk_bytes=50_000_000_000,
        )
        assert body.can_execute_local({})

    def test_body_ram_gate(self):
        body = NodeBody(
            cpu_cores=4,
            gpu_vram_mb=0,
            ram_bytes=4_000_000_000,
            disk_bytes=100_000_000_000,
        )
        assert not body.can_execute_local({"min_ram_bytes": 64_000_000_000})


class TestIdentityGenesisDataclass:
    """Frozen dataclass behavior and wallet defaults."""

    def test_frozen_immutable(self):
        pk = os.urandom(32)
        genesis = IdentityGenesis.create(pk)
        with pytest.raises(AttributeError):
            genesis.sovereignty_class = SovereigntyClass.FOREST  # type: ignore[misc]

    def test_wallet_defaults_zero(self):
        pk = os.urandom(32)
        genesis = IdentityGenesis.create(pk)
        assert genesis.wallet_seed_balance == 0.0
        assert genesis.wallet_bloom_balance == 0.0
        assert genesis.genesis_wallet_state == GenesisWalletState()

    def test_assert_uniqueness_same_pk(self):
        pk = os.urandom(32)
        g1 = IdentityGenesis.create(pk)
        g2 = IdentityGenesis.create(pk)
        g1.assert_uniqueness(g2)

    def test_assert_uniqueness_different_pk(self):
        pk1 = os.urandom(32)
        pk2 = os.urandom(32)
        g1 = IdentityGenesis.create(pk1)
        g2 = IdentityGenesis.create(pk2)
        g1.assert_uniqueness(g2)


class TestIdentityGenesisExtensions:
    """IDG §1.2 extensions for persona, attestation, and signed genesis."""

    def test_persona_attestation_scope_and_wallet_state(self):
        pk = os.urandom(32)
        persona = PersonaSeed(
            display_name="Mumo",
            mission_statement="Stand on giants without forking authority",
            locale="en-AE",
        )
        wallet_state = GenesisWalletState(seed_balance=3.5, bloom_balance=1.25)

        genesis = IdentityGenesis.create(
            pk,
            persona_seed=persona,
            human_attestation=HumanAttestation.DEVICE_WITNESSED,
            sovereignty_scope=SovereigntyScope.NODE_LOCAL,
            wallet_state=wallet_state,
            created_at=1_700_000_000.0,
        )

        assert genesis.persona_seed == persona
        assert genesis.human_attestation is HumanAttestation.DEVICE_WITNESSED
        assert genesis.sovereignty_scope is SovereigntyScope.NODE_LOCAL
        assert genesis.genesis_wallet_state == wallet_state
        assert genesis.wallet_seed_balance == 3.5
        assert genesis.wallet_bloom_balance == 1.25

    def test_genesis_hash_is_stable_for_fixed_payload_and_changes_with_persona(self):
        pk = os.urandom(32)
        persona = PersonaSeed(display_name="Mumo", mission_statement="BIZRA", locale="en")
        same_1 = IdentityGenesis.create(pk, persona_seed=persona, created_at=1234.5)
        same_2 = IdentityGenesis.create(pk, persona_seed=persona, created_at=1234.5)
        changed = IdentityGenesis.create(
            pk,
            persona_seed=PersonaSeed(display_name="DEMA", mission_statement="BIZRA", locale="en"),
            created_at=1234.5,
        )

        assert same_1.genesis_hash == same_2.genesis_hash
        assert len(same_1.genesis_hash) == 64
        assert same_1.genesis_hash != changed.genesis_hash

    def test_genesis_signature_verifies_with_domain_separation(self):
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        private_bytes = private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )

        genesis = IdentityGenesis.create(
            public_key,
            persona_seed=PersonaSeed(display_name="Mumo", mission_statement="BIZRA", locale="en"),
            created_at=1234.5,
            genesis_signing_key=private_bytes,
        )

        assert genesis.genesis_signature_domain == GENESIS_SIGNATURE_DOMAIN
        assert len(genesis.genesis_signature) == 128
        assert genesis.verify_genesis_signature()

        verifier = private_key.public_key()
        verifier.verify(bytes.fromhex(genesis.genesis_signature), genesis.signable_payload())
        with pytest.raises(InvalidSignature):
            verifier.verify(bytes.fromhex(genesis.genesis_signature), genesis.genesis_hash.encode("ascii"))

    def test_genesis_signature_rejects_mismatched_signing_key(self):
        expected_private_key = Ed25519PrivateKey.generate()
        wrong_private_key = Ed25519PrivateKey.generate()
        public_key = expected_private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        wrong_private_bytes = wrong_private_key.private_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PrivateFormat.Raw,
            encryption_algorithm=serialization.NoEncryption(),
        )

        with pytest.raises(ValueError, match="does not match"):
            IdentityGenesis.create(public_key, genesis_signing_key=wrong_private_bytes)
