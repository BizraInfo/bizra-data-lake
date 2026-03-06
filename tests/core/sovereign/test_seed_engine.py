"""
Seed Engine Tests — Phase 71
==============================

TDD anchors for the DDAGI Seed Potential Engine: growth episodes,
sovereignty tier progression, hash-chained receipts, and self-RLVR
integration into the SovereignRuntime lifecycle.

Standing on Giants:
- Beck (2002): TDD by Example
- Deming (1986): PDCA through measurement
"""

from __future__ import annotations

from core.sovereign.seed_engine import (
    TIER_FOREST,
    TIER_ORDER,
    TIER_SEED,
    TIER_SPROUT,
    TIER_TREE,
    GrowthEpisode,
    SeedEngine,
    SeedEngineConfig,
    SeedPotential,
    create_seed_engine,
    sovereignty_tier,
)

# ======================================================================
# sovereignty_tier()
# ======================================================================


class TestSovereigntyTier:
    """Tier mapping from score to name."""

    def test_zero_is_seed(self) -> None:
        assert sovereignty_tier(0.0) == TIER_SEED

    def test_low_is_seed(self) -> None:
        assert sovereignty_tier(0.20) == TIER_SEED

    def test_quarter_is_sprout(self) -> None:
        assert sovereignty_tier(0.25) == TIER_SPROUT

    def test_half_is_tree(self) -> None:
        assert sovereignty_tier(0.50) == TIER_TREE

    def test_high_is_forest(self) -> None:
        assert sovereignty_tier(0.80) == TIER_FOREST

    def test_one_is_forest(self) -> None:
        assert sovereignty_tier(1.0) == TIER_FOREST

    def test_negative_clamps_to_seed(self) -> None:
        assert sovereignty_tier(-0.5) == TIER_SEED

    def test_above_one_clamps_to_forest(self) -> None:
        assert sovereignty_tier(1.5) == TIER_FOREST

    def test_tier_order_is_correct(self) -> None:
        assert TIER_ORDER == [TIER_SEED, TIER_SPROUT, TIER_TREE, TIER_FOREST]


# ======================================================================
# SeedEngine — Basic Operations
# ======================================================================


class TestSeedEngineBasic:
    """Core engine operations."""

    def test_creates_with_defaults(self) -> None:
        engine = SeedEngine()
        assert engine._node_id == "node0"
        assert engine._total_count == 0

    def test_creates_with_custom_node_id(self) -> None:
        engine = SeedEngine(node_id="my-node")
        assert engine._node_id == "my-node"

    def test_creates_with_config(self) -> None:
        cfg = SeedEngineConfig(compile_streak=5)
        engine = SeedEngine(config=cfg)
        assert engine._config.compile_streak == 5

    def test_initial_potential_is_zero(self) -> None:
        engine = SeedEngine()
        p = engine.potential()
        assert p.sovereignty_score == 0.0
        assert p.tier == TIER_SEED
        assert p.episodes_total == 0
        assert p.potential_remaining == 1.0

    def test_initial_health(self) -> None:
        engine = SeedEngine()
        h = engine.health()
        assert h["active"] is False
        assert h["episodes"] == 0
        assert h["tier"] == TIER_SEED
        assert h["compiled"] is False


# ======================================================================
# SeedEngine — Recording Episodes
# ======================================================================


class TestSeedEngineEpisodes:
    """Episode recording and scoring."""

    def test_record_returns_episode(self) -> None:
        engine = SeedEngine()
        ep = engine.record_episode({"snr": 0.95, "ihsan": 0.96})
        assert isinstance(ep, GrowthEpisode)
        assert ep.index == 1
        assert ep.snr == 0.95
        assert ep.ihsan == 0.96

    def test_episode_has_receipt_hash(self) -> None:
        engine = SeedEngine()
        ep = engine.record_episode({"snr": 0.95, "ihsan": 0.96})
        assert len(ep.receipt_hash) == 64  # SHA-256 hex

    def test_consecutive_episodes_chain_hashes(self) -> None:
        engine = SeedEngine()
        ep1 = engine.record_episode({"snr": 0.90, "ihsan": 0.92})
        ep2 = engine.record_episode({"snr": 0.91, "ihsan": 0.93})
        assert ep1.receipt_hash != ep2.receipt_hash
        assert engine._previous_hash == ep2.receipt_hash

    def test_episode_increments_count(self) -> None:
        engine = SeedEngine()
        engine.record_episode({"snr": 0.90, "ihsan": 0.92})
        engine.record_episode({"snr": 0.91, "ihsan": 0.93})
        assert engine._total_count == 2

    def test_high_quality_episode_qualifies(self) -> None:
        engine = SeedEngine()
        ep = engine.record_episode(
            {
                "snr": 0.96,
                "ihsan": 0.97,
                "tokens_used": 500,
                "quality": 0.95,
                "user_feedback": 0.90,
                "verified": True,
                "penalties": 0.0,
            }
        )
        assert ep.qualified is True

    def test_low_snr_episode_does_not_qualify(self) -> None:
        engine = SeedEngine()
        ep = engine.record_episode(
            {
                "snr": 0.50,  # Below threshold
                "ihsan": 0.97,
                "verified": True,
            }
        )
        assert ep.qualified is False

    def test_unverified_episode_does_not_qualify(self) -> None:
        engine = SeedEngine()
        ep = engine.record_episode(
            {
                "snr": 0.96,
                "ihsan": 0.97,
                "verified": False,
            }
        )
        assert ep.qualified is False

    def test_recent_episodes_returns_last_n(self) -> None:
        engine = SeedEngine()
        for i in range(5):
            engine.record_episode({"snr": 0.90 + i * 0.01, "ihsan": 0.92})
        recent = engine.recent_episodes(limit=3)
        assert len(recent) == 3
        assert recent[-1]["index"] == 5


# ======================================================================
# SeedEngine — Streak and Compilation
# ======================================================================


class TestSeedEngineStreak:
    """Compile streak and reflex promotion."""

    def _high_episode(self) -> dict:
        return {
            "snr": 0.96,
            "ihsan": 0.97,
            "tokens_used": 500,
            "quality": 0.95,
            "user_feedback": 0.92,
            "verified": True,
            "penalties": 0.0,
        }

    def test_streak_increments_on_qualified(self) -> None:
        engine = SeedEngine()
        engine.record_episode(self._high_episode())
        assert engine._streak == 1

    def test_streak_resets_on_failure(self) -> None:
        engine = SeedEngine()
        engine.record_episode(self._high_episode())
        engine.record_episode({"snr": 0.50, "ihsan": 0.50, "verified": True})
        assert engine._streak == 0

    def test_compile_after_streak(self) -> None:
        engine = SeedEngine(config=SeedEngineConfig(compile_streak=3))
        assert engine._compiled is False
        for _ in range(3):
            engine.record_episode(self._high_episode())
        assert engine._compiled is True

    def test_compile_stays_true_after_failure(self) -> None:
        engine = SeedEngine(config=SeedEngineConfig(compile_streak=2))
        engine.record_episode(self._high_episode())
        engine.record_episode(self._high_episode())
        assert engine._compiled is True
        engine.record_episode({"snr": 0.50, "ihsan": 0.50, "verified": True})
        assert engine._compiled is True  # Once compiled, stays compiled


# ======================================================================
# SeedEngine — Sovereignty Score and Tier Progression
# ======================================================================


class TestSeedEngineSovereignty:
    """Sovereignty score computation and tier progression."""

    def _high_episode(self) -> dict:
        return {
            "snr": 0.96,
            "ihsan": 0.97,
            "tokens_used": 500,
            "quality": 0.95,
            "user_feedback": 0.92,
            "verified": True,
            "penalties": 0.0,
        }

    def test_score_increases_with_qualified_episodes(self) -> None:
        engine = SeedEngine()
        engine.record_episode(self._high_episode())
        score_1 = engine._compute_sovereignty_score()
        for _ in range(4):
            engine.record_episode(self._high_episode())
        score_5 = engine._compute_sovereignty_score()
        assert score_5 > score_1

    def test_tier_progresses_with_growth(self) -> None:
        engine = SeedEngine(config=SeedEngineConfig(compile_streak=2))
        # Record enough high-quality episodes to advance tiers
        for _ in range(10):
            engine.record_episode(self._high_episode())

        p = engine.potential()
        # Should have advanced beyond SEED
        tier_idx = TIER_ORDER.index(p.tier)
        assert tier_idx > 0, f"Expected beyond SEED, got {p.tier}"

    def test_potential_unlocked_matches_sovereignty(self) -> None:
        engine = SeedEngine()
        for _ in range(5):
            engine.record_episode(self._high_episode())
        p = engine.potential()
        assert p.potential_unlocked == p.sovereignty_score
        assert abs(p.potential_unlocked + p.potential_remaining - 1.0) < 0.001

    def test_tier_progress_within_range(self) -> None:
        engine = SeedEngine()
        for _ in range(3):
            engine.record_episode(self._high_episode())
        p = engine.potential()
        assert 0.0 <= p.tier_progress <= 1.0


# ======================================================================
# SeedEngine — Growth Velocity
# ======================================================================


class TestSeedEngineVelocity:
    """Growth velocity tracking."""

    def test_velocity_zero_with_no_episodes(self) -> None:
        engine = SeedEngine()
        p = engine.potential()
        assert p.growth_velocity == 0.0

    def test_positive_velocity_with_improvement(self) -> None:
        engine = SeedEngine()
        # Start with low-quality, then improve
        engine.record_episode({"snr": 0.50, "ihsan": 0.50, "verified": True})
        for _ in range(5):
            engine.record_episode(
                {
                    "snr": 0.96,
                    "ihsan": 0.97,
                    "tokens_used": 500,
                    "quality": 0.95,
                    "user_feedback": 0.92,
                    "verified": True,
                    "penalties": 0.0,
                }
            )
        p = engine.potential()
        assert p.growth_velocity >= 0.0  # Should be improving


# ======================================================================
# SeedEngine — Dimension Balance
# ======================================================================


class TestSeedEngineDimensions:
    """Weakest dimension detection and balance."""

    def test_identifies_weakest_dimension(self) -> None:
        engine = SeedEngine()
        # Record episodes with low SNR but high everything else
        for _ in range(5):
            engine.record_episode(
                {
                    "snr": 0.50,
                    "ihsan": 0.97,
                    "tokens_used": 500,
                    "quality": 0.95,
                    "user_feedback": 0.92,
                    "verified": True,
                }
            )
        p = engine.potential()
        assert p.weakest_dimension == "snr"

    def test_no_weakest_with_no_episodes(self) -> None:
        engine = SeedEngine()
        p = engine.potential()
        assert p.weakest_dimension is None


# ======================================================================
# SeedEngine — Health and API
# ======================================================================


class TestSeedEngineHealth:
    """Health reporting for /v1/health integration."""

    def test_health_after_episodes(self) -> None:
        engine = SeedEngine()
        engine.record_episode({"snr": 0.95, "ihsan": 0.96, "verified": True})
        h = engine.health()
        assert h["active"] is True
        assert h["episodes"] == 1

    def test_health_shows_compiled(self) -> None:
        engine = SeedEngine(config=SeedEngineConfig(compile_streak=1))
        engine.record_episode(
            {
                "snr": 0.96,
                "ihsan": 0.97,
                "tokens_used": 500,
                "quality": 0.95,
                "user_feedback": 0.92,
                "verified": True,
                "penalties": 0.0,
            }
        )
        h = engine.health()
        assert h["compiled"] is True


# ======================================================================
# SeedEngine — Hash Chain Integrity
# ======================================================================


class TestSeedEngineChain:
    """Receipt chain integrity."""

    def test_genesis_hash(self) -> None:
        engine = SeedEngine()
        assert engine._previous_hash == "GENESIS"

    def test_hash_chain_not_genesis_after_episode(self) -> None:
        engine = SeedEngine()
        engine.record_episode({"snr": 0.90, "ihsan": 0.92})
        assert engine._previous_hash != "GENESIS"
        assert len(engine._previous_hash) == 64

    def test_chain_valid_in_potential(self) -> None:
        engine = SeedEngine()
        engine.record_episode({"snr": 0.90, "ihsan": 0.92})
        p = engine.potential()
        assert p.chain_valid is True

    def test_last_receipt_hash_in_potential(self) -> None:
        engine = SeedEngine()
        ep = engine.record_episode({"snr": 0.90, "ihsan": 0.92})
        p = engine.potential()
        assert p.last_receipt_hash == ep.receipt_hash


# ======================================================================
# create_seed_engine()
# ======================================================================


class TestCreateSeedEngine:
    """Factory function."""

    def test_creates_with_defaults(self) -> None:
        engine = create_seed_engine()
        assert isinstance(engine, SeedEngine)
        assert engine._node_id == "node0"

    def test_creates_with_custom_node_id(self) -> None:
        engine = create_seed_engine(node_id="my-seed")
        assert engine._node_id == "my-seed"

    def test_creates_with_config(self) -> None:
        cfg = SeedEngineConfig(compile_streak=7)
        engine = create_seed_engine(config=cfg)
        assert engine._config.compile_streak == 7

    def test_creates_from_runtime_with_identity(self) -> None:
        class MockIdentity:
            node_id = "genesis-node"

        class MockRuntime:
            _identity = MockIdentity()

        engine = create_seed_engine(runtime=MockRuntime())
        assert engine._node_id == "genesis-node"


# ======================================================================
# SeedPotential dataclass
# ======================================================================


class TestSeedPotential:
    """SeedPotential dataclass correctness."""

    def test_potential_fields_present(self) -> None:
        engine = SeedEngine()
        p = engine.potential()
        assert isinstance(p, SeedPotential)
        assert hasattr(p, "sovereignty_score")
        assert hasattr(p, "tier")
        assert hasattr(p, "potential_remaining")
        assert hasattr(p, "growth_velocity")
        assert hasattr(p, "last_receipt_hash")

    def test_potential_remaining_always_non_negative(self) -> None:
        engine = SeedEngine(config=SeedEngineConfig(compile_streak=1))
        for _ in range(20):
            engine.record_episode(
                {
                    "snr": 0.99,
                    "ihsan": 0.99,
                    "tokens_used": 100,
                    "quality": 0.99,
                    "user_feedback": 0.99,
                    "verified": True,
                    "penalties": 0.0,
                }
            )
        p = engine.potential()
        assert p.potential_remaining >= 0.0
