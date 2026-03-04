"""Tests for Interaction Boundary Enforcement — Axiom 1.6.

Phase 61 Step 2 — Proof Chain v2.
10 TDD anchors covering: attack elimination, surface reduction,
boundary enforcement, pool mediation, and exception hierarchy.
"""

from __future__ import annotations

import pytest

from core.federation.interaction_boundary import (
    ELIMINATED_BY_BOUNDARY,
    REQUIRES_IDENTITY_MITIGATION,
    AttackClass,
    BoundaryAuditResult,
    BoundaryViolation,
    PoolMediatedMessage,
    assert_no_direct_channel,
)

# ---------------------------------------------------------------------------
# 1. seven_attacks_eliminated
# ---------------------------------------------------------------------------


class TestSevenAttacksEliminated:
    """Axiom 1.6 eliminates exactly 7 attack classes."""

    def test_seven_attacks_eliminated(self) -> None:
        assert len(ELIMINATED_BY_BOUNDARY) == 7

    def test_eclipse_eliminated(self) -> None:
        assert AttackClass.ECLIPSE in ELIMINATED_BY_BOUNDARY

    def test_mitm_eliminated(self) -> None:
        assert AttackClass.MITM in ELIMINATED_BY_BOUNDARY

    def test_bgp_hijacking_eliminated(self) -> None:
        assert AttackClass.BGP_HIJACKING in ELIMINATED_BY_BOUNDARY

    def test_ddos_peer_discovery_eliminated(self) -> None:
        assert AttackClass.DDOS_PEER_DISCOVERY in ELIMINATED_BY_BOUNDARY

    def test_poisoned_peer_data_eliminated(self) -> None:
        assert AttackClass.POISONED_PEER_DATA in ELIMINATED_BY_BOUNDARY

    def test_network_mapping_eliminated(self) -> None:
        assert AttackClass.NETWORK_MAPPING in ELIMINATED_BY_BOUNDARY

    def test_routing_table_poisoning_eliminated(self) -> None:
        assert AttackClass.ROUTING_TABLE_POISONING in ELIMINATED_BY_BOUNDARY


# ---------------------------------------------------------------------------
# 2. only_sybil_remains
# ---------------------------------------------------------------------------


class TestOnlySybilRemains:
    """Only Sybil remains viable under Axiom 1.6."""

    def test_sybil_is_the_only_remaining_attack(self) -> None:
        assert REQUIRES_IDENTITY_MITIGATION == frozenset({AttackClass.SYBIL})

    def test_sybil_not_eliminated(self) -> None:
        assert AttackClass.SYBIL not in ELIMINATED_BY_BOUNDARY

    def test_remaining_has_exactly_one_member(self) -> None:
        assert len(REQUIRES_IDENTITY_MITIGATION) == 1


# ---------------------------------------------------------------------------
# 3. attack_surface_reduction_87_5_percent
# ---------------------------------------------------------------------------


class TestAttackSurfaceReduction:
    """Boundary reduces attack surface by 87.5% (7/8)."""

    def test_attack_surface_reduction_is_seven_eighths(self) -> None:
        result = BoundaryAuditResult()
        assert abs(result.attack_surface_reduction - 0.875) < 1e-9

    def test_reduction_matches_class_constant(self) -> None:
        result = BoundaryAuditResult()
        assert (
            abs(
                result.attack_surface_reduction - BoundaryAuditResult.EXPECTED_REDUCTION
            )
            < 1e-9
        )


# ---------------------------------------------------------------------------
# 4. boundary_audit_is_enforced
# ---------------------------------------------------------------------------


class TestBoundaryAuditIsEnforced:
    """Audit result reflects enforcement state and is immutable."""

    def test_default_boundary_enforced(self) -> None:
        result = BoundaryAuditResult()
        assert result.boundary_enforced is True

    def test_audit_boundary_classmethod(self) -> None:
        result = BoundaryAuditResult.audit_boundary()
        assert result.boundary_enforced is True
        assert result.eliminated_attacks == ELIMINATED_BY_BOUNDARY
        assert result.remaining_attacks == REQUIRES_IDENTITY_MITIGATION

    def test_audit_result_is_frozen(self) -> None:
        result = BoundaryAuditResult()
        with pytest.raises(AttributeError):
            result.boundary_enforced = False  # type: ignore[misc]

    def test_audit_timestamp_is_positive(self) -> None:
        result = BoundaryAuditResult.audit_boundary()
        assert result.timestamp > 0


# ---------------------------------------------------------------------------
# 5. no_direct_channel_between_distinct_nodes
# ---------------------------------------------------------------------------


class TestNoDirectChannelBetweenDistinctNodes:
    """All direct channels between distinct nodes are forbidden."""

    def test_raises_for_distinct_nodes(self) -> None:
        with pytest.raises(BoundaryViolation, match="Axiom 1.6 violation"):
            assert_no_direct_channel("node_alpha", "node_beta")

    def test_raises_for_any_distinct_pair(self) -> None:
        with pytest.raises(BoundaryViolation):
            assert_no_direct_channel("a", "b")

    def test_error_message_contains_node_ids(self) -> None:
        with pytest.raises(BoundaryViolation, match="node_x.*node_y"):
            assert_no_direct_channel("node_x", "node_y")


# ---------------------------------------------------------------------------
# 6. self_channel_allowed
# ---------------------------------------------------------------------------


class TestSelfChannelAllowed:
    """Same node can 'talk to itself' — no boundary crossing."""

    def test_self_channel_does_not_raise(self) -> None:
        # Should complete without exception
        assert_no_direct_channel("node_42", "node_42")

    def test_self_channel_with_empty_id(self) -> None:
        # Edge case: empty string identity, same node
        assert_no_direct_channel("", "")


# ---------------------------------------------------------------------------
# 7. pool_mediated_message_valid
# ---------------------------------------------------------------------------


class TestPoolMediatedMessageValid:
    """Messages with proper pool signature and timestamp pass validation."""

    def test_valid_message_passes(self) -> None:
        msg = PoolMediatedMessage(
            sender_id="node_a",
            payload=b"hello world",
            pool_timestamp=1709510400.0,
            pool_signature=b"\x01\x02\x03\x04",
        )
        assert msg.validate_pool_mediation() is True

    def test_valid_message_attributes(self) -> None:
        msg = PoolMediatedMessage(
            sender_id="node_a",
            payload=b"data",
            pool_timestamp=1.0,
            pool_signature=b"sig",
        )
        assert msg.sender_id == "node_a"
        assert msg.payload == b"data"
        assert msg.pool_timestamp == 1.0
        assert msg.pool_signature == b"sig"


# ---------------------------------------------------------------------------
# 8. pool_mediated_message_invalid_no_signature
# ---------------------------------------------------------------------------


class TestPoolMediatedMessageInvalid:
    """Messages without pool signature or timestamp fail validation."""

    def test_empty_signature_fails(self) -> None:
        msg = PoolMediatedMessage(
            sender_id="node_a",
            payload=b"hello",
            pool_timestamp=1709510400.0,
            pool_signature=b"",
        )
        assert msg.validate_pool_mediation() is False

    def test_zero_timestamp_fails(self) -> None:
        msg = PoolMediatedMessage(
            sender_id="node_a",
            payload=b"hello",
            pool_timestamp=0.0,
            pool_signature=b"valid_sig",
        )
        assert msg.validate_pool_mediation() is False

    def test_negative_timestamp_fails(self) -> None:
        msg = PoolMediatedMessage(
            sender_id="node_a",
            payload=b"hello",
            pool_timestamp=-1.0,
            pool_signature=b"valid_sig",
        )
        assert msg.validate_pool_mediation() is False

    def test_both_invalid_fails(self) -> None:
        msg = PoolMediatedMessage(
            sender_id="node_a",
            payload=b"hello",
            pool_timestamp=0.0,
            pool_signature=b"",
        )
        assert msg.validate_pool_mediation() is False


# ---------------------------------------------------------------------------
# 9. all_attack_classes_accounted_for
# ---------------------------------------------------------------------------


class TestAllAttackClassesAccountedFor:
    """Eliminated + remaining = full set of attack classes."""

    def test_union_is_complete(self) -> None:
        all_attacks = ELIMINATED_BY_BOUNDARY | REQUIRES_IDENTITY_MITIGATION
        assert all_attacks == frozenset(AttackClass)

    def test_no_overlap(self) -> None:
        overlap = ELIMINATED_BY_BOUNDARY & REQUIRES_IDENTITY_MITIGATION
        assert overlap == frozenset()

    def test_total_count_is_eight(self) -> None:
        total = len(ELIMINATED_BY_BOUNDARY) + len(REQUIRES_IDENTITY_MITIGATION)
        assert total == 8
        assert total == len(AttackClass)


# ---------------------------------------------------------------------------
# 10. boundary_violation_is_exception
# ---------------------------------------------------------------------------


class TestBoundaryViolationIsException:
    """BoundaryViolation is a proper Exception subclass."""

    def test_is_exception_subclass(self) -> None:
        assert issubclass(BoundaryViolation, Exception)

    def test_can_be_raised_and_caught(self) -> None:
        with pytest.raises(BoundaryViolation):
            raise BoundaryViolation("test violation")

    def test_carries_message(self) -> None:
        with pytest.raises(BoundaryViolation, match="constitutional"):
            raise BoundaryViolation("constitutional violation detected")

    def test_caught_by_generic_except(self) -> None:
        try:
            raise BoundaryViolation("test")
        except Exception as exc:
            assert isinstance(exc, BoundaryViolation)
            assert str(exc) == "test"
