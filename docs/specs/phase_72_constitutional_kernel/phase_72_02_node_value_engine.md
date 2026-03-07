# Phase 72.02: Node Value Engine

**Target file:** `core/sovereign/node_value.py`

## Purpose

Unify the five measurable factors of node value into a single composite KPI.
Every factor is already tracked by existing modules. This engine composes them.

## Data Sources

```pseudocode
# Factor 1: Potential — from SeedEngine
potential = seed_engine.potential().sovereignty_score  # float 0-1
# Source: core/sovereign/seed_engine.py:316-375

# Factor 2: Activation Rate — from mission completion count
activation_rate = completed_missions / active_days  # DAM (Daily Active Missions)
# Source: inferred from mission history (EvidenceLedger or SeedEngine episode count)

# Factor 3: Verification Quality — from Ihsan gate scores
verification_quality = mean(recent_ihsan_scores)  # 6-dim composite, 0-1
# Source: SeedEngine._dimension_scores["ihsan"] or IhsanGate history

# Factor 4: Compounding Time — age × streak
compounding_time = age_in_days * (1 + streak_bonus(streak))
# streak_bonus: log(1 + streak) / log(10)  — diminishing returns, never zero

# Factor 5: Network Synergy — federation × attestations
network_synergy = asabiyyah_score * (1 + log(1 + attestation_count))
# Source: core/federation/ + attestation records
# NOTE: For Node0 (pre-federation), defaults to 1.0 × 1.0 = 1.0
```

## Pseudocode

```pseudocode
IMPORT SeedEngine FROM core.sovereign.seed_engine
IMPORT UNIFIED_IHSAN_THRESHOLD FROM core.integration.constants
IMPORT UNIFIED_SNR_THRESHOLD FROM core.integration.constants

@dataclass
CLASS NodeValueSnapshot:
    potential: float           # 0-1
    activation_rate: float     # missions/day
    verification_quality: float # 0-1, mean ihsan
    compounding_time: float    # age_days * streak_multiplier
    network_synergy: float     # asabiyyah * attestations
    composite: float           # product of all five
    tier: str                  # sovereignty tier
    human_stage: str           # Seed..Catalyst
    timestamp: str             # ISO 8601

CLASS NodeValueEngine:
    """Computes the unified KPI for a sovereign node.

    Lightweight, deterministic, no I/O. Depends only on SeedEngine
    state and optional federation metrics.
    """

    CONSTRUCTOR(seed_engine: SeedEngine,
                genesis_timestamp: str = None):
        self._engine = seed_engine
        self._genesis = genesis_timestamp OR now_utc()
        self._mission_count = 0
        self._mission_start_day = today()

    FUNCTION record_mission():
        """Call after each mission completion."""
        self._mission_count += 1

    FUNCTION compute() -> NodeValueSnapshot:
        pot = self._engine.potential()

        # Factor 1: Potential
        potential = pot.sovereignty_score  # 0-1

        # Factor 2: Activation Rate (DAM)
        active_days = max(1, days_since(self._genesis))
        activation_rate = self._mission_count / active_days

        # Factor 3: Verification Quality
        # Use SeedEngine's tracked ihsan dimension
        ihsan_scores = self._engine._dimension_scores.get("ihsan", [])
        IF len(ihsan_scores) > 0:
            verification_quality = mean(ihsan_scores[-50:])  # recent window
        ELSE:
            verification_quality = 0.0

        # Factor 4: Compounding Time
        age_days = max(1.0, days_since(self._genesis))
        streak = pot.streak
        streak_multiplier = 1.0 + (log(1 + streak) / log(10))
        compounding_time = age_days * streak_multiplier

        # Factor 5: Network Synergy
        # Pre-federation default: 1.0 (no network bonus, no penalty)
        network_synergy = self._compute_network_synergy()

        # Composite: product
        composite = (
            potential
            * activation_rate
            * verification_quality
            * compounding_time
            * network_synergy
        )

        RETURN NodeValueSnapshot(
            potential=potential,
            activation_rate=round(activation_rate, 4),
            verification_quality=round(verification_quality, 4),
            compounding_time=round(compounding_time, 4),
            network_synergy=round(network_synergy, 4),
            composite=round(composite, 4),
            tier=pot.tier,
            human_stage=human_stage(pot.sovereignty_score),
            timestamp=now_utc_iso(),
        )

    FUNCTION _compute_network_synergy() -> float:
        """Stub for pre-federation. Returns 1.0 until A2A is live."""
        # When federation module is wired:
        #   asabiyyah = federation.asabiyyah_score()
        #   attestations = evidence_ledger.attestation_count()
        #   return asabiyyah * (1 + log(1 + attestations))
        RETURN 1.0

    FUNCTION health() -> dict:
        RETURN {
            "engine": "node_value",
            "mission_count": self._mission_count,
            "genesis": self._genesis,
            "has_federation": False,  # until A2A wired
        }
```

## Edge Cases

```pseudocode
CASE "brand new node, zero missions":
    potential = 0.0, activation = 0.0
    composite = 0.0 (correct — no value yet earned)

CASE "active node, zero ihsan":
    verification_quality = 0.0
    composite = 0.0 (correct — unverified work has no value)

CASE "node with streak reset":
    streak_multiplier = 1.0 (log(1)/log(10) = 0)
    compounding_time = age_days * 1.0 (no bonus)

CASE "pre-federation single node":
    network_synergy = 1.0 (neutral multiplier)
    composite = potential * activation * quality * time * 1.0
```

## TDD Anchors

```pseudocode
TEST "zero-mission node has zero value":
    engine = SeedEngine("test")
    nv = NodeValueEngine(engine)
    result = nv.compute()
    ASSERT result.composite == 0.0
    ASSERT result.human_stage == "Seed"

TEST "qualified episodes increase node value":
    engine = SeedEngine("test")
    nv = NodeValueEngine(engine)
    FOR i IN 1..10:
        engine.record_episode({"snr": 0.95, "ihsan": 0.96, "quality": 0.9})
        nv.record_mission()
    result = nv.compute()
    ASSERT result.composite > 0.0
    ASSERT result.potential > 0.0
    ASSERT result.activation_rate > 0.0

TEST "streak multiplier increases compounding":
    engine = SeedEngine("test")
    nv = NodeValueEngine(engine, genesis_timestamp=30_days_ago())
    FOR i IN 1..5:
        engine.record_episode({"snr": 0.95, "ihsan": 0.96, "quality": 0.9})
        nv.record_mission()
    result = nv.compute()
    ASSERT result.compounding_time > 30.0  # streak bonus applied

TEST "unqualified episodes yield zero verification quality":
    engine = SeedEngine("test")
    nv = NodeValueEngine(engine)
    engine.record_episode({"snr": 0.10, "ihsan": 0.20})
    nv.record_mission()
    result = nv.compute()
    ASSERT result.verification_quality < UNIFIED_IHSAN_THRESHOLD

TEST "network synergy defaults to 1.0 pre-federation":
    engine = SeedEngine("test")
    nv = NodeValueEngine(engine)
    ASSERT nv._compute_network_synergy() == 1.0

TEST "composite is product of all five factors":
    engine = SeedEngine("test")
    nv = NodeValueEngine(engine, genesis_timestamp=1_day_ago())
    FOR i IN 1..5:
        engine.record_episode({"snr": 0.95, "ihsan": 0.96, "quality": 0.9})
        nv.record_mission()
    result = nv.compute()
    expected = (result.potential * result.activation_rate *
                result.verification_quality * result.compounding_time *
                result.network_synergy)
    ASSERT abs(result.composite - expected) < 0.01

TEST "human_stage maps correctly":
    ASSERT human_stage(0.00) == "Seed"
    ASSERT human_stage(0.15) == "Node"
    ASSERT human_stage(0.30) == "Apprentice"
    ASSERT human_stage(0.50) == "Builder"
    ASSERT human_stage(0.60) == "Verifier"
    ASSERT human_stage(0.75) == "Mentor"
    ASSERT human_stage(0.90) == "Catalyst"

TEST "health returns expected shape":
    engine = SeedEngine("test")
    nv = NodeValueEngine(engine)
    h = nv.health()
    ASSERT "mission_count" IN h
    ASSERT "genesis" IN h
```
