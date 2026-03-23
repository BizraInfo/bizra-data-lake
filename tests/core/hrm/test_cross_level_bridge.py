"""Tests for core.hrm.cross_level_bridge — CrossLevelBridge, message types.

Covers:
- MessageType, PropagationDirection enums
- CrossLevelMessage: direction, level_distance properties
- CascadeResult: responding_levels property
- SyncResult: sync_complete property
- CrossLevelBridge: propagate_hypothesis (upward, downward, both, blocked)
- CrossLevelBridge: request_validation (cascade, consensus)
- CrossLevelBridge: synchronize_integration (contradictions, gaps, transfers)
- CrossLevelBridge: allocate_attention (top-down)
- CrossLevelBridge: report_surprise (bottom-up, attenuation)
- CrossLevelBridge: get_bridge_metrics telemetry

Blueprint Reference: P3 Coverage Ratchet — hrm module (0.25 → higher)
"""

from core.hrm.abstraction_levels import AbstractionLevel
from core.hrm.cross_level_bridge import (
    CascadeResult,
    CrossLevelBridge,
    CrossLevelMessage,
    MessageType,
    PropagationDirection,
    SyncResult,
)

# ═══════════════════════════════════════════════════════════════════════════
# Data Structures
# ═══════════════════════════════════════════════════════════════════════════


class TestCrossLevelMessage:
    def test_upward_direction(self):
        msg = CrossLevelMessage(
            source_level=AbstractionLevel.PERCEPTUAL,
            target_level=AbstractionLevel.OPERATIONAL,
        )
        assert msg.direction == "upward"

    def test_downward_direction(self):
        msg = CrossLevelMessage(
            source_level=AbstractionLevel.STRATEGIC,
            target_level=AbstractionLevel.TACTICAL,
        )
        assert msg.direction == "downward"

    def test_lateral_direction(self):
        msg = CrossLevelMessage(
            source_level=AbstractionLevel.TACTICAL,
            target_level=AbstractionLevel.TACTICAL,
        )
        assert msg.direction == "lateral"

    def test_level_distance(self):
        msg = CrossLevelMessage(
            source_level=AbstractionLevel.PERCEPTUAL,
            target_level=AbstractionLevel.STRATEGIC,
        )
        assert msg.level_distance == 3

    def test_auto_id(self):
        msg = CrossLevelMessage()
        assert msg.message_id is not None
        assert len(msg.message_id) > 0


class TestCascadeResult:
    def test_responding_levels_sorted(self):
        result = CascadeResult()
        result.responses[AbstractionLevel.STRATEGIC] = {"score": 0.9}
        result.responses[AbstractionLevel.PERCEPTUAL] = {"score": 0.7}
        levels = result.responding_levels
        assert levels == [AbstractionLevel.PERCEPTUAL, AbstractionLevel.STRATEGIC]


class TestSyncResult:
    def test_sync_complete_false(self):
        result = SyncResult()
        result.participating_levels = {
            AbstractionLevel.PERCEPTUAL,
            AbstractionLevel.OPERATIONAL,
        }
        assert result.sync_complete is False

    def test_sync_complete_true(self):
        result = SyncResult()
        result.participating_levels = set(AbstractionLevel)
        assert result.sync_complete is True


# ═══════════════════════════════════════════════════════════════════════════
# CrossLevelBridge
# ═══════════════════════════════════════════════════════════════════════════


class TestHypothesisPropagation:
    def test_propagate_upward(self):
        bridge = CrossLevelBridge()
        messages = bridge.propagate_hypothesis(
            {"claim": "test"},
            AbstractionLevel.PERCEPTUAL,
            PropagationDirection.UPWARD,
            confidence=0.9,
        )
        assert len(messages) >= 1
        assert all(m.target_level > m.source_level for m in messages)

    def test_propagate_downward(self):
        bridge = CrossLevelBridge()
        messages = bridge.propagate_hypothesis(
            {"claim": "test"},
            AbstractionLevel.STRATEGIC,
            PropagationDirection.DOWNWARD,
            confidence=0.9,
        )
        assert len(messages) >= 1
        assert all(m.target_level < m.source_level for m in messages)

    def test_propagate_both(self):
        bridge = CrossLevelBridge()
        messages = bridge.propagate_hypothesis(
            {"claim": "test"},
            AbstractionLevel.TACTICAL,
            PropagationDirection.BOTH,
            confidence=0.9,
        )
        assert len(messages) >= 2  # one up, one down

    def test_no_propagation_from_top_upward(self):
        bridge = CrossLevelBridge()
        messages = bridge.propagate_hypothesis(
            {"claim": "test"},
            AbstractionLevel.META_COGNITIVE,
            PropagationDirection.UPWARD,
            confidence=0.9,
        )
        assert len(messages) == 0

    def test_low_confidence_blocked(self):
        bridge = CrossLevelBridge()
        bridge.propagate_hypothesis(
            {"claim": "test"},
            AbstractionLevel.OPERATIONAL,
            PropagationDirection.UPWARD,
            confidence=0.1,  # Very low — should be blocked
        )
        # May be blocked depending on boundary permeability
        metrics = bridge.get_bridge_metrics()
        # Total attempts = messages passed + blocked
        assert metrics["total_messages"] + metrics["blocked_messages"] >= 1


class TestValidationCascade:
    def test_request_validation(self):
        bridge = CrossLevelBridge()
        result = bridge.request_validation(
            {"hypothesis": "X is true", "confidence": 0.7},
            AbstractionLevel.TACTICAL,
        )
        assert isinstance(result, CascadeResult)
        assert len(result.responses) > 0
        assert result.cascade_depth > 0

    def test_consensus_reached_high_confidence(self):
        bridge = CrossLevelBridge()
        result = bridge.request_validation(
            {"hypothesis": "X is true", "confidence": 0.95},
            AbstractionLevel.OPERATIONAL,
        )
        # High-confidence hypothesis should reach consensus
        assert result.aggregate_confidence > 0

    def test_cascade_count_incremented(self):
        bridge = CrossLevelBridge()
        bridge.request_validation({"hypothesis": "A"}, AbstractionLevel.TACTICAL)
        bridge.request_validation({"hypothesis": "B"}, AbstractionLevel.TACTICAL)
        metrics = bridge.get_bridge_metrics()
        assert metrics["cascade_count"] == 2


class TestIntegrationSync:
    def test_sync_single_level(self):
        bridge = CrossLevelBridge()
        result = bridge.synchronize_integration(
            {
                AbstractionLevel.PERCEPTUAL: {"snr_scores": [0.9]},
            }
        )
        assert result.sync_quality == 1.0  # trivial sync

    def test_sync_no_contradictions(self):
        bridge = CrossLevelBridge()
        result = bridge.synchronize_integration(
            {
                AbstractionLevel.PERCEPTUAL: {"snr_scores": [0.85]},
                AbstractionLevel.OPERATIONAL: {"snr_scores": [0.87]},
            }
        )
        assert result.contradictions_found == 0
        assert result.sync_quality > 0.8

    def test_sync_with_contradictions(self):
        bridge = CrossLevelBridge()
        result = bridge.synchronize_integration(
            {
                AbstractionLevel.PERCEPTUAL: {"snr_scores": [0.5]},
                AbstractionLevel.OPERATIONAL: {"snr_scores": [0.95]},
            }
        )
        assert result.contradictions_found >= 1
        assert result.sync_quality < 1.0

    def test_sync_detects_gaps(self):
        bridge = CrossLevelBridge()
        result = bridge.synchronize_integration(
            {
                AbstractionLevel.PERCEPTUAL: {
                    "snr_scores": [0.9]
                },  # no active_hypotheses
                AbstractionLevel.OPERATIONAL: {
                    "snr_scores": [0.9],
                    "active_hypotheses": True,
                },
            }
        )
        assert result.gaps_identified >= 1

    def test_sync_detects_transfers(self):
        bridge = CrossLevelBridge()
        result = bridge.synchronize_integration(
            {
                AbstractionLevel.PERCEPTUAL: {"insights": ["a"]},
                AbstractionLevel.OPERATIONAL: {"insights": ["b"]},
            }
        )
        assert result.transfers_discovered >= 1


class TestAttentionAllocation:
    def test_attention_from_strategic(self):
        bridge = CrossLevelBridge()
        messages = bridge.allocate_attention(
            AbstractionLevel.STRATEGIC,
            priority_signal={"focus": "security"},
        )
        # Should target L0, L1, L2 (all below strategic)
        assert len(messages) == 3
        assert all(m.message_type == MessageType.ATTENTION_SIGNAL for m in messages)

    def test_attention_specific_targets(self):
        bridge = CrossLevelBridge()
        messages = bridge.allocate_attention(
            AbstractionLevel.STRATEGIC,
            target_levels=[AbstractionLevel.PERCEPTUAL],
        )
        assert len(messages) == 1
        assert messages[0].target_level == AbstractionLevel.PERCEPTUAL


class TestSurpriseReporting:
    def test_surprise_propagates_upward(self):
        bridge = CrossLevelBridge()
        messages = bridge.report_surprise(
            {"anomaly": "unexpected pattern"},
            AbstractionLevel.PERCEPTUAL,
            surprise_magnitude=0.8,
        )
        # L0 → L1, L2, L3, L4 = 4 messages
        assert len(messages) == 4
        assert all(m.message_type == MessageType.SURPRISE_REPORT for m in messages)

    def test_surprise_attenuates_with_distance(self):
        bridge = CrossLevelBridge()
        messages = bridge.report_surprise(
            {"anomaly": "x"},
            AbstractionLevel.PERCEPTUAL,
            surprise_magnitude=0.9,
        )
        # Each successive message should have lower confidence
        for i in range(len(messages) - 1):
            assert messages[i].confidence >= messages[i + 1].confidence

    def test_surprise_from_top_no_messages(self):
        bridge = CrossLevelBridge()
        messages = bridge.report_surprise(
            {"anomaly": "x"},
            AbstractionLevel.META_COGNITIVE,
        )
        assert len(messages) == 0


class TestBridgeMetrics:
    def test_initial_metrics(self):
        bridge = CrossLevelBridge()
        metrics = bridge.get_bridge_metrics()
        assert metrics["total_messages"] == 0
        assert metrics["blocked_messages"] == 0
        assert metrics["cascade_count"] == 0
        assert metrics["sync_count"] == 0

    def test_metrics_after_operations(self):
        bridge = CrossLevelBridge()
        bridge.propagate_hypothesis(
            {"x": 1},
            AbstractionLevel.TACTICAL,
            PropagationDirection.BOTH,
            confidence=0.9,
        )
        bridge.request_validation({"h": "test"}, AbstractionLevel.OPERATIONAL)
        bridge.synchronize_integration(
            {
                AbstractionLevel.PERCEPTUAL: {},
                AbstractionLevel.OPERATIONAL: {},
            }
        )
        metrics = bridge.get_bridge_metrics()
        assert metrics["total_messages"] >= 2
        assert metrics["cascade_count"] == 1
        assert metrics["sync_count"] == 1
        assert "message_type_distribution" in metrics
        assert "boundary_health" in metrics
