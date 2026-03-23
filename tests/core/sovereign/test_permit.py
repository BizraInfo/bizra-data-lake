"""
Tests for the Telescript Permit System (Task 3.3).

Covers:
- Authority creation, delegation, chain verification
- Permit creation, signing, verification
- Capability mapping for HDA methods
- Budget enforcement (action + token limits)
- TTL expiry
- Signature integrity (tamper detection)
- Full HDA action check (signature + expiry + budget + capability)

Standing on Giants: General Magic (1994) · Lamport (1978) · Shannon (1948)
"""

from __future__ import annotations

import time

import pytest

from core.sovereign.permit import (
    HDA_CAPABILITY_MAP,
    MAX_DELEGATION_DEPTH,
    Authority,
    Capability,
    Permit,
    ResourceBudget,
    create_hda_permit,
)

# ---------------------------------------------------------------------------
# Authority
# ---------------------------------------------------------------------------


class TestAuthority:
    def test_genesis_creation(self) -> None:
        genesis = Authority.genesis()
        assert genesis.delegation_depth == 0
        assert genesis.delegated_from is None
        assert genesis.name == "Node0-Genesis"
        assert len(genesis.chain_hash) == 64  # SHA-256 hex

    def test_genesis_verify_chain(self) -> None:
        genesis = Authority.genesis()
        assert genesis.verify_chain()

    def test_delegation(self) -> None:
        genesis = Authority.genesis()
        child = genesis.delegate("ops-agent")
        assert child.delegation_depth == 1
        assert child.delegated_from == genesis.id
        assert child.name == "ops-agent"
        assert child.chain_hash != genesis.chain_hash

    def test_multi_level_delegation(self) -> None:
        auth = Authority.genesis()
        for i in range(MAX_DELEGATION_DEPTH):
            auth = auth.delegate(f"level-{i + 1}")
        assert auth.delegation_depth == MAX_DELEGATION_DEPTH

    def test_delegation_depth_exceeded(self) -> None:
        auth = Authority.genesis()
        for i in range(MAX_DELEGATION_DEPTH):
            auth = auth.delegate(f"level-{i + 1}")
        with pytest.raises(ValueError, match="exceeds max"):
            auth.delegate("too-deep")

    def test_to_dict(self) -> None:
        genesis = Authority.genesis()
        d = genesis.to_dict()
        assert d["name"] == "Node0-Genesis"
        assert d["delegation_depth"] == 0
        assert "..." in d["chain_hash"]


# ---------------------------------------------------------------------------
# ResourceBudget
# ---------------------------------------------------------------------------


class TestResourceBudget:
    def test_fresh_budget(self) -> None:
        budget = ResourceBudget(max_actions=10, max_tokens=1000)
        assert budget.actions_remaining == 10
        assert budget.tokens_remaining == 1000
        assert not budget.exhausted

    def test_consume_action(self) -> None:
        budget = ResourceBudget(max_actions=3, max_tokens=100)
        assert budget.consume_action(10)
        assert budget.actions_remaining == 2
        assert budget.tokens_used == 10

    def test_budget_exhaustion(self) -> None:
        budget = ResourceBudget(max_actions=1, max_tokens=100)
        assert budget.consume_action()
        assert budget.exhausted
        assert not budget.consume_action()  # Exhausted


# ---------------------------------------------------------------------------
# Permit
# ---------------------------------------------------------------------------


class TestPermit:
    def test_create_signed_permit(self) -> None:
        genesis = Authority.genesis()
        permit = Permit.create(
            issuer=genesis,
            capabilities=[Capability.GO, Capability.COMPUTE],
            ttl_seconds=600,
            signing_key="test-key",
        )
        assert len(permit.signature) == 64  # HMAC-SHA256 hex
        assert permit.ttl_seconds == 600
        assert Capability.GO in permit.capabilities
        assert Capability.COMPUTE in permit.capabilities

    def test_verify_valid_permit(self) -> None:
        genesis = Authority.genesis()
        permit = Permit.create(
            issuer=genesis,
            capabilities=[Capability.GO],
            signing_key="test-key",
        )
        result = permit.verify(signing_key="test-key")
        assert result.valid
        assert result.reason == "OK"
        assert result.remaining_actions > 0

    def test_verify_expired_permit(self) -> None:
        genesis = Authority.genesis()
        permit = Permit.create(
            issuer=genesis,
            capabilities=[Capability.GO],
            ttl_seconds=1,
            signing_key="test-key",
        )
        # Force expiry
        permit.expires_at = time.time() - 10
        result = permit.verify(signing_key="test-key")
        assert not result.valid
        assert "expired" in result.reason.lower()

    def test_verify_tampered_permit(self) -> None:
        genesis = Authority.genesis()
        permit = Permit.create(
            issuer=genesis,
            capabilities=[Capability.GO],
            signing_key="test-key",
        )
        # Tamper with capabilities
        permit.capabilities.append(Capability.NETWORK)
        result = permit.verify(signing_key="test-key")
        assert not result.valid
        assert "signature" in result.reason.lower()

    def test_verify_wrong_key(self) -> None:
        genesis = Authority.genesis()
        permit = Permit.create(
            issuer=genesis,
            capabilities=[Capability.GO],
            signing_key="correct-key",
        )
        result = permit.verify(signing_key="wrong-key")
        assert not result.valid

    def test_verify_exhausted_budget(self) -> None:
        genesis = Authority.genesis()
        permit = Permit.create(
            issuer=genesis,
            capabilities=[Capability.GO],
            max_actions=1,
            signing_key="test-key",
        )
        permit.consume()
        result = permit.verify(signing_key="test-key")
        assert not result.valid
        assert "exhausted" in result.reason.lower()

    def test_has_capability(self) -> None:
        genesis = Authority.genesis()
        permit = Permit.create(
            issuer=genesis,
            capabilities=[Capability.GO, Capability.STORE],
            signing_key="test-key",
        )
        assert permit.has_capability(Capability.GO)
        assert permit.has_capability(Capability.STORE)
        assert not permit.has_capability(Capability.NETWORK)

    def test_to_dict(self) -> None:
        genesis = Authority.genesis()
        permit = Permit.create(
            issuer=genesis,
            capabilities=[Capability.GO],
            signing_key="test-key",
        )
        d = permit.to_dict()
        assert "permit_id" in d
        assert d["capabilities"] == ["GO"]
        assert "budget" in d
        assert "..." in d["signature"]


# ---------------------------------------------------------------------------
# HDA Action Checks
# ---------------------------------------------------------------------------


class TestHDAActionCheck:
    def test_all_hda_methods_mapped(self) -> None:
        expected = {
            "open_app",
            "switch_window",
            "type_text",
            "click_element",
            "screenshot",
            "read_clipboard",
            "file_open",
            "browser_navigate",
        }
        assert set(HDA_CAPABILITY_MAP.keys()) == expected

    def test_check_permitted_action(self) -> None:
        genesis = Authority.genesis()
        permit = Permit.create(
            issuer=genesis,
            capabilities=[Capability.GO],
            signing_key="test-key",
        )
        result = permit.check_action("switch_window", signing_key="test-key")
        assert result.valid

    def test_check_unpermitted_action(self) -> None:
        genesis = Authority.genesis()
        permit = Permit.create(
            issuer=genesis,
            capabilities=[Capability.GO],  # No STORE capability
            signing_key="test-key",
        )
        result = permit.check_action("file_open", signing_key="test-key")
        assert not result.valid
        assert "STORE" in result.reason

    def test_check_unknown_method(self) -> None:
        genesis = Authority.genesis()
        permit = Permit.create(
            issuer=genesis,
            capabilities=[Capability.GO],
            signing_key="test-key",
        )
        result = permit.check_action("unknown_method", signing_key="test-key")
        assert not result.valid
        assert "Unknown" in result.reason

    def test_budget_decrements_on_consume(self) -> None:
        genesis = Authority.genesis()
        permit = Permit.create(
            issuer=genesis,
            capabilities=[Capability.COMPUTE],
            max_actions=3,
            signing_key="test-key",
        )
        assert permit.consume(tokens=100)
        assert permit.consume(tokens=200)
        assert permit.budget.actions_remaining == 1
        assert permit.budget.tokens_used == 300


# ---------------------------------------------------------------------------
# create_hda_permit convenience
# ---------------------------------------------------------------------------


class TestCreateHDAPermit:
    def test_default_permit_has_all_desktop_caps(self) -> None:
        permit = create_hda_permit(signing_key="test-key")
        for method in HDA_CAPABILITY_MAP:
            result = permit.check_action(method, signing_key="test-key")
            assert result.valid, f"Method {method} should be permitted"

    def test_delegation_chain(self) -> None:
        permit = create_hda_permit(signing_key="test-key")
        assert permit.issuer.delegation_depth == 1
        assert permit.issuer.name == "founder-ops-agent"

    def test_custom_ttl_and_budget(self) -> None:
        permit = create_hda_permit(
            ttl_seconds=60,
            max_actions=5,
            signing_key="test-key",
        )
        assert permit.ttl_seconds == 60
        assert permit.budget.max_actions == 5
