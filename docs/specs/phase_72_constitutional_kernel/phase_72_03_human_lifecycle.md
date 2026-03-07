# Phase 72.03: Human Lifecycle — Seed to Catalyst

**Target file:** `core/sovereign/human_lifecycle.py`

## Purpose

Map sovereignty scores to the 7-stage human growth progression. This is the
human-readable mirror of the agent skill tree (Novice → Grandmaster). Both
progressions are earned through verified work, gated by quality, and compound
over time.

## Design Constraints

- Pure functions, no I/O
- Thresholds from `constants.py` only
- Parallel structure to `sovereignty_tier()` in `seed_engine.py`
- Each stage has: name, score range, description, unlock conditions

## Pseudocode

```pseudocode
IMPORT UNIFIED_IHSAN_THRESHOLD FROM core.integration.constants

# ─────────────────────────────────────────────────────────────
# Human Growth Stages — the journey of every seed
# ─────────────────────────────────────────────────────────────

@dataclass
CLASS HumanStage:
    name: str
    rank: int               # 0-6
    score_low: float        # inclusive
    score_high: float       # exclusive (except Catalyst)
    description: str
    unlock_condition: str   # what the human did to reach this stage

STAGES = [
    HumanStage(
        name="Seed",
        rank=0,
        score_low=0.00, score_high=0.10,
        description="First install. Identity created. Potential is infinite.",
        unlock_condition="Install Node0, generate Ed25519 keypair",
    ),
    HumanStage(
        name="Node",
        rank=1,
        score_low=0.10, score_high=0.20,
        description="First mission completed. The seed has sprouted.",
        unlock_condition="Complete first mission with Ihsan >= 0.85",
    ),
    HumanStage(
        name="Apprentice",
        rank=2,
        score_low=0.20, score_high=0.35,
        description="Consistent work. Building habits. Learning the system.",
        unlock_condition="10+ qualified episodes, qualification rate >= 50%",
    ),
    HumanStage(
        name="Builder",
        rank=3,
        score_low=0.35, score_high=0.55,
        description="Compiled first reflex. Work is becoming automatic.",
        unlock_condition="First reflex compiled (3+ consecutive qualified)",
    ),
    HumanStage(
        name="Verifier",
        rank=4,
        score_low=0.55, score_high=0.70,
        description="Trusted to attest others' work. Quality is habitual.",
        unlock_condition="Sovereignty >= 0.55, qualification rate >= 75%",
    ),
    HumanStage(
        name="Mentor",
        rank=5,
        score_low=0.70, score_high=0.85,
        description="Skills published to marketplace. Helping others grow.",
        unlock_condition="Published 3+ compiled reflexes as tradeable skills",
    ),
    HumanStage(
        name="Catalyst",
        rank=6,
        score_low=0.85, score_high=1.00,
        description="Network effect multiplier. The seed has become a forest.",
        unlock_condition="Sovereignty >= 0.85, 5+ mentored nodes, FOREST tier",
    ),
]

FUNCTION human_stage(sovereignty_score: float) -> str:
    """Map sovereignty score to human lifecycle stage name."""
    clamped = clamp(sovereignty_score, 0.0, 1.0)
    FOR stage IN reversed(STAGES):
        IF clamped >= stage.score_low:
            RETURN stage.name
    RETURN "Seed"

FUNCTION human_stage_detail(sovereignty_score: float) -> HumanStage:
    """Full stage metadata for UI/API."""
    clamped = clamp(sovereignty_score, 0.0, 1.0)
    FOR stage IN reversed(STAGES):
        IF clamped >= stage.score_low:
            RETURN stage
    RETURN STAGES[0]

FUNCTION stage_progress(sovereignty_score: float) -> dict:
    """Progress within current stage + next stage info."""
    stage = human_stage_detail(sovereignty_score)
    range_size = stage.score_high - stage.score_low
    progress = (sovereignty_score - stage.score_low) / range_size IF range_size > 0 ELSE 1.0

    # Next stage
    next_stage = None
    IF stage.rank < 6:
        next_stage = STAGES[stage.rank + 1]

    RETURN {
        "current_stage": stage.name,
        "rank": stage.rank,
        "progress": round(clamp(progress, 0.0, 1.0), 4),
        "sovereignty_score": round(sovereignty_score, 4),
        "next_stage": next_stage.name IF next_stage ELSE None,
        "next_threshold": next_stage.score_low IF next_stage ELSE None,
        "points_to_next": round(next_stage.score_low - sovereignty_score, 4)
                          IF next_stage AND sovereignty_score < next_stage.score_low
                          ELSE 0.0,
        "description": stage.description,
        "unlock_condition": stage.unlock_condition,
    }

# ─────────────────────────────────────────────────────────────
# Agent ↔ Human tier alignment
# ─────────────────────────────────────────────────────────────

AGENT_TIER_MAP = {
    "Seed":      "Novice",       # Agent equivalent
    "Node":      "Apprentice",
    "Apprentice": "Journeyman",
    "Builder":   "Craftsman",
    "Verifier":  "Expert",
    "Mentor":    "Master",
    "Catalyst":  "Grandmaster",
}

FUNCTION agent_tier_equivalent(human_stage_name: str) -> str:
    """What agent skill tier matches this human stage?"""
    RETURN AGENT_TIER_MAP.get(human_stage_name, "Novice")
```

## TDD Anchors

```pseudocode
TEST "boundary values map correctly":
    ASSERT human_stage(0.00) == "Seed"
    ASSERT human_stage(0.09) == "Seed"
    ASSERT human_stage(0.10) == "Node"
    ASSERT human_stage(0.20) == "Apprentice"
    ASSERT human_stage(0.35) == "Builder"
    ASSERT human_stage(0.55) == "Verifier"
    ASSERT human_stage(0.70) == "Mentor"
    ASSERT human_stage(0.85) == "Catalyst"
    ASSERT human_stage(1.00) == "Catalyst"

TEST "negative and overflow scores clamp":
    ASSERT human_stage(-1.0) == "Seed"
    ASSERT human_stage(5.0) == "Catalyst"

TEST "stage_progress returns correct structure":
    result = stage_progress(0.50)
    ASSERT result["current_stage"] == "Builder"
    ASSERT result["next_stage"] == "Verifier"
    ASSERT result["next_threshold"] == 0.55
    ASSERT result["points_to_next"] == 0.05
    ASSERT 0.0 <= result["progress"] <= 1.0

TEST "catalyst has no next stage":
    result = stage_progress(0.90)
    ASSERT result["current_stage"] == "Catalyst"
    ASSERT result["next_stage"] IS None
    ASSERT result["points_to_next"] == 0.0

TEST "all 7 stages are reachable":
    reached = set()
    FOR score IN [0.0, 0.10, 0.20, 0.35, 0.55, 0.70, 0.85]:
        reached.add(human_stage(score))
    ASSERT len(reached) == 7

TEST "agent tier equivalents all valid":
    FOR stage IN STAGES:
        equiv = agent_tier_equivalent(stage.name)
        ASSERT equiv IS NOT None
        ASSERT len(equiv) > 0

TEST "stages are monotonically ordered":
    FOR i IN 0..len(STAGES)-2:
        ASSERT STAGES[i].score_low < STAGES[i+1].score_low
        ASSERT STAGES[i].rank < STAGES[i+1].rank

TEST "stage descriptions are non-empty":
    FOR stage IN STAGES:
        ASSERT len(stage.description) > 10
        ASSERT len(stage.unlock_condition) > 5
```
