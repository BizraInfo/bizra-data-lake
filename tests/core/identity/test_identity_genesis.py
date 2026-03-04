"""Tests for Identity Genesis -- Phase 61 Step 1.

11 TDD anchors validating Definition 1.6 (Identity) and Definition 1.7 (Node Body).

Standing on Giants: Bernstein (Ed25519) | BIP-32 (HD derivation)
"""

from __future__ import annotations

import hashlib
import os

import pytest

from core.identity.genesis import (
    IdentityGenesis,
    NodeBody,
    SovereigntyClass,
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
        assert body.can_execute_local({
            "required_models": {"phi3:mini"},
            "min_gpu_vram_mb": 8000,
        })

    def test_body_cannot_execute_without_model(self):
        body = NodeBody(
            cpu_cores=16,
            gpu_vram_mb=24000,
            gpu_compute_cap=8.9,
            ram_bytes=128_000_000_000,
            disk_bytes=1_000_000_000_000,
            loaded_models={"phi3:mini"},
        )
        assert not body.can_execute_local({
            "required_models": {"llama3.1-70b"},
            "min_gpu_vram_mb": 40000,
        })

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
