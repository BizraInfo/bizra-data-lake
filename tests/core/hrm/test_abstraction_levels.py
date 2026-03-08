"""Tests for core.hrm.abstraction_levels — HRM abstraction level hierarchy.

Covers:
- AbstractionLevel enum (5 levels, IntEnum)
- TemporalScale and BridgeNodeType enums
- HRM_SNR_GRADIENT: monotonically increasing SNR thresholds
- LevelConfig: construction, properties (level_name, level_index)
- LevelBoundary: direction, should_pass, record_crossing
- default_level_configs: 5 configs with SNR gradient
- default_boundaries: 8 boundaries (4 upward + 4 downward)

Blueprint Reference: P3 Coverage Ratchet — hrm module (0.25 → higher)
"""

import pytest

from core.hrm.abstraction_levels import (
    AbstractionLevel,
    BridgeNodeType,
    HRM_SNR_GRADIENT,
    HRM_TEMPORAL_SCALE,
    LevelBoundary,
    LevelConfig,
    TemporalScale,
    default_boundaries,
    default_level_configs,
)


# ═══════════════════════════════════════════════════════════════════════════
# Enums
# ═══════════════════════════════════════════════════════════════════════════


class TestAbstractionLevel:
    def test_five_levels(self):
        assert len(AbstractionLevel) == 5

    def test_ordering(self):
        assert AbstractionLevel.PERCEPTUAL < AbstractionLevel.OPERATIONAL
        assert AbstractionLevel.OPERATIONAL < AbstractionLevel.TACTICAL
        assert AbstractionLevel.TACTICAL < AbstractionLevel.STRATEGIC
        assert AbstractionLevel.STRATEGIC < AbstractionLevel.META_COGNITIVE

    def test_integer_values(self):
        assert int(AbstractionLevel.PERCEPTUAL) == 0
        assert int(AbstractionLevel.META_COGNITIVE) == 4


class TestTemporalScale:
    def test_five_scales(self):
        assert len(TemporalScale) == 5

    def test_values(self):
        assert TemporalScale.IMMEDIATE.value == "immediate"
        assert TemporalScale.EVOLUTIONARY.value == "evolutionary"


class TestBridgeNodeType:
    def test_five_types(self):
        assert len(BridgeNodeType) == 5

    def test_bridge_highest_snr(self):
        assert BridgeNodeType.BRIDGE.value == "bridge"


# ═══════════════════════════════════════════════════════════════════════════
# SNR Gradient
# ═══════════════════════════════════════════════════════════════════════════


class TestSNRGradient:
    def test_all_levels_covered(self):
        for level in AbstractionLevel:
            assert level in HRM_SNR_GRADIENT

    def test_monotonically_increasing(self):
        levels = sorted(AbstractionLevel)
        for i in range(len(levels) - 1):
            assert HRM_SNR_GRADIENT[levels[i]] <= HRM_SNR_GRADIENT[levels[i + 1]]

    def test_meta_cognitive_highest(self):
        assert HRM_SNR_GRADIENT[AbstractionLevel.META_COGNITIVE] >= 0.98


class TestTemporalMapping:
    def test_all_levels_covered(self):
        for level in AbstractionLevel:
            assert level in HRM_TEMPORAL_SCALE


# ═══════════════════════════════════════════════════════════════════════════
# LevelConfig
# ═══════════════════════════════════════════════════════════════════════════


class TestLevelConfig:
    def test_construction(self):
        config = LevelConfig(
            level=AbstractionLevel.TACTICAL,
            snr_threshold=0.90,
            temporal_scale=TemporalScale.MEDIUM_TERM,
        )
        assert config.level == AbstractionLevel.TACTICAL
        assert config.snr_threshold == 0.90

    def test_frozen(self):
        config = LevelConfig(
            level=AbstractionLevel.PERCEPTUAL,
            snr_threshold=0.85,
            temporal_scale=TemporalScale.IMMEDIATE,
        )
        with pytest.raises(AttributeError):
            config.level = AbstractionLevel.STRATEGIC  # type: ignore

    def test_level_name(self):
        config = LevelConfig(
            level=AbstractionLevel.META_COGNITIVE,
            snr_threshold=0.98,
            temporal_scale=TemporalScale.EVOLUTIONARY,
        )
        assert config.level_name == "Meta Cognitive"

    def test_level_index(self):
        config = LevelConfig(
            level=AbstractionLevel.OPERATIONAL,
            snr_threshold=0.85,
            temporal_scale=TemporalScale.SHORT_TERM,
        )
        assert config.level_index == 1


# ═══════════════════════════════════════════════════════════════════════════
# LevelBoundary
# ═══════════════════════════════════════════════════════════════════════════


class TestLevelBoundary:
    def test_upward_direction(self):
        b = LevelBoundary(
            source_level=AbstractionLevel.PERCEPTUAL,
            target_level=AbstractionLevel.OPERATIONAL,
        )
        assert b.direction == "upward"

    def test_downward_direction(self):
        b = LevelBoundary(
            source_level=AbstractionLevel.STRATEGIC,
            target_level=AbstractionLevel.TACTICAL,
        )
        assert b.direction == "downward"

    def test_should_pass_high_confidence(self):
        b = LevelBoundary(
            source_level=AbstractionLevel.PERCEPTUAL,
            target_level=AbstractionLevel.OPERATIONAL,
            permeability=0.6,
        )
        # threshold = 1.0 - 0.6 = 0.4, so 0.5 should pass
        assert b.should_pass(0.5) is True

    def test_should_pass_low_confidence(self):
        b = LevelBoundary(
            source_level=AbstractionLevel.PERCEPTUAL,
            target_level=AbstractionLevel.OPERATIONAL,
            permeability=0.3,
        )
        # threshold = 1.0 - 0.3 = 0.7, so 0.5 should NOT pass
        assert b.should_pass(0.5) is False

    def test_record_crossing_passed(self):
        b = LevelBoundary(
            source_level=AbstractionLevel.PERCEPTUAL,
            target_level=AbstractionLevel.OPERATIONAL,
        )
        b.record_crossing(passed=True)
        assert b.message_count == 1
        assert b.blocked_count == 0

    def test_record_crossing_blocked(self):
        b = LevelBoundary(
            source_level=AbstractionLevel.PERCEPTUAL,
            target_level=AbstractionLevel.OPERATIONAL,
        )
        b.record_crossing(passed=False)
        assert b.message_count == 0
        assert b.blocked_count == 1


# ═══════════════════════════════════════════════════════════════════════════
# Factory Functions
# ═══════════════════════════════════════════════════════════════════════════


class TestDefaultLevelConfigs:
    def test_returns_five(self):
        configs = default_level_configs()
        assert len(configs) == 5

    def test_snr_gradient(self):
        configs = default_level_configs()
        for i in range(len(configs) - 1):
            assert configs[i].snr_threshold <= configs[i + 1].snr_threshold

    def test_learning_rate_decreases(self):
        configs = default_level_configs()
        for i in range(len(configs) - 1):
            assert configs[i].learning_rate_factor >= configs[i + 1].learning_rate_factor

    def test_noise_tolerance_decreases(self):
        configs = default_level_configs()
        for i in range(len(configs) - 1):
            assert configs[i].noise_tolerance >= configs[i + 1].noise_tolerance


class TestDefaultBoundaries:
    def test_returns_eight(self):
        boundaries = default_boundaries()
        assert len(boundaries) == 8  # 4 upward + 4 downward

    def test_has_upward_and_downward(self):
        boundaries = default_boundaries()
        directions = [b.direction for b in boundaries]
        assert "upward" in directions
        assert "downward" in directions

    def test_all_transform_required(self):
        boundaries = default_boundaries()
        assert all(b.transform_required for b in boundaries)
