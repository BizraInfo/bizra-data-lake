"""
Tests for BIZRA Isnad Risk Propagation (IRP) algorithm.

Tests cover: source construction, chain strength, Isnad grading,
variance adjustment, position sizing, MAWDU exclusion, and the
PAT->SAT boundary assessment pipeline.
"""

import math
import pytest

from core.irp import (
    IsnadGrade,
    Source,
    IsnadChain,
    DataPoint,
    IrpAssessment,
    chain_strength,
    aggregate_strength,
    irp_variance_adjustment,
    irp_position_size,
    pat_assess,
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def reuters():
    return Source(id="reuters", name="Reuters", reliability=0.95, verified=True)

@pytest.fixture
def bloomberg():
    return Source(id="bloomberg", name="Bloomberg", reliability=0.93, verified=True)

@pytest.fixture
def coinbase():
    return Source(id="coinbase", name="Coinbase", reliability=0.90, verified=True)

@pytest.fixture
def telegram():
    return Source(id="telegram", name="Random Telegram", reliability=0.3)

@pytest.fixture
def anon():
    return Source(id="anon", name="Anonymous", reliability=0.05)


# ============================================================================
# SOURCE VALIDATION
# ============================================================================

class TestSource:
    def test_valid_source(self, reuters):
        assert reuters.reliability == 0.95
        assert reuters.verified is True

    def test_reliability_bounds(self):
        with pytest.raises(ValueError, match="must be in"):
            Source(id="bad", name="Bad", reliability=1.5)
        with pytest.raises(ValueError, match="must be in"):
            Source(id="bad", name="Bad", reliability=-0.1)

    def test_frozen(self, reuters):
        with pytest.raises(AttributeError):
            reuters.reliability = 0.5  # type: ignore[misc]


# ============================================================================
# CHAIN STRENGTH
# ============================================================================

class TestChainStrength:
    def test_strong_beats_weak(self, reuters, telegram):
        strong = IsnadChain(sources=[reuters])
        weak = IsnadChain(sources=[telegram])
        assert chain_strength(strong) > chain_strength(weak)

    def test_long_chain_decays(self, reuters, bloomberg, telegram):
        short = IsnadChain(sources=[reuters])
        long = IsnadChain(sources=[reuters, bloomberg, telegram])
        assert chain_strength(long) < chain_strength(short)

    def test_empty_chain_rejected(self):
        with pytest.raises(ValueError, match="at least one"):
            IsnadChain(sources=[])

    def test_chain_hash_deterministic(self, reuters, bloomberg):
        c1 = IsnadChain(sources=[reuters, bloomberg])
        c2 = IsnadChain(sources=[reuters, bloomberg])
        assert c1.chain_hash == c2.chain_hash


# ============================================================================
# ISNAD GRADING
# ============================================================================

class TestIsnadGrading:
    def test_single_strong_is_hasan(self, reuters):
        """Single chain can never be SAHIH — needs mutawatir."""
        dp = DataPoint(asset_id="BTC", value=50000.0,
                       chains=[IsnadChain(sources=[reuters])])
        assert dp.grade == IsnadGrade.HASAN

    def test_three_independent_is_sahih(self, reuters, bloomberg, coinbase):
        """Three independent verified chains = SAHIH."""
        chains = [
            IsnadChain(sources=[reuters]),
            IsnadChain(sources=[bloomberg]),
            IsnadChain(sources=[coinbase]),
        ]
        dp = DataPoint(asset_id="BTC", value=50000.0, chains=chains)
        assert dp.grade == IsnadGrade.SAHIH

    def test_weak_single_is_daif(self, telegram):
        dp = DataPoint(asset_id="SHITCOIN", value=0.001,
                       chains=[IsnadChain(sources=[telegram])])
        assert dp.grade == IsnadGrade.DAIF


# ============================================================================
# VARIANCE ADJUSTMENT & POSITION SIZING
# ============================================================================

class TestRiskModification:
    def test_sahih_no_inflation(self, reuters, bloomberg, coinbase):
        chains = [IsnadChain(sources=[s]) for s in [reuters, bloomberg, coinbase]]
        dp = DataPoint(asset_id="BTC", value=50000.0, chains=chains)
        assert irp_variance_adjustment(0.04, dp) == pytest.approx(0.04)

    def test_hasan_moderate_inflation(self, reuters):
        dp = DataPoint(asset_id="BTC", value=50000.0,
                       chains=[IsnadChain(sources=[reuters])])
        assert irp_variance_adjustment(0.04, dp) == pytest.approx(0.06)

    def test_daif_significant_inflation(self, telegram):
        dp = DataPoint(asset_id="X", value=1.0,
                       chains=[IsnadChain(sources=[telegram])])
        assert irp_variance_adjustment(0.04, dp) == pytest.approx(0.12)

    def test_mawdu_infinite_variance(self, anon):
        dp = DataPoint(asset_id="SCAM", value=999.0,
                       chains=[IsnadChain(sources=[anon])])
        assert irp_variance_adjustment(0.04, dp) == float('inf')

    def test_mawdu_zero_position(self, anon):
        dp = DataPoint(asset_id="SCAM", value=999.0,
                       chains=[IsnadChain(sources=[anon])])
        pos = irp_position_size(0.003, 0.04, dp, 100_000.0)
        assert pos == 0.0

    def test_sahih_larger_than_daif(self, reuters, bloomberg, coinbase, telegram):
        chains_sahih = [IsnadChain(sources=[s]) for s in [reuters, bloomberg, coinbase]]
        dp_sahih = DataPoint(asset_id="BTC", value=50000.0, chains=chains_sahih)
        dp_daif = DataPoint(asset_id="X", value=1.0,
                            chains=[IsnadChain(sources=[telegram])])
        pos_sahih = irp_position_size(0.003, 0.04, dp_sahih, 100_000.0)
        pos_daif = irp_position_size(0.003, 0.04, dp_daif, 100_000.0)
        assert pos_sahih > pos_daif > 0


# ============================================================================
# PAT->SAT BOUNDARY
# ============================================================================

class TestPatSatBoundary:
    def test_assessment_grade(self, reuters, bloomberg, coinbase):
        chains = [IsnadChain(sources=[s]) for s in [reuters, bloomberg, coinbase]]
        dp = DataPoint(asset_id="BTC", value=50000.0, chains=chains)
        assessment = pat_assess(dp)
        assert assessment.grade == IsnadGrade.SAHIH
        assert assessment.asset_id == "BTC"

    def test_assessment_has_hash(self, reuters):
        dp = DataPoint(asset_id="ETH", value=3000.0,
                       chains=[IsnadChain(sources=[reuters])])
        assessment = pat_assess(dp)
        assert len(assessment.assessment_hash) == 64  # blake2b 32-byte hex

    def test_assessment_preserves_value(self, reuters):
        dp = DataPoint(asset_id="ETH", value=3000.0,
                       chains=[IsnadChain(sources=[reuters])])
        assessment = pat_assess(dp)
        assert assessment.assessed_value == 3000.0
        assert assessment.recommended_variance_multiplier == 1.5  # HASAN
