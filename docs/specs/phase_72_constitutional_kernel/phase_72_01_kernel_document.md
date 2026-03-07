# Phase 72.01: Constitutional Kernel Document

**Target file:** `docs/KERNEL.md`

## Purpose

The one-page document that every user, investor, and regulator reads first. Not marketing. Not poetry. Operating law with source citations.

## Structure Specification

### Section 1: Mission (5 lines max)

```markdown
# BIZRA Constitutional Kernel

**Mission:** BIZRA is a decentralized developmental AGI operating system that
turns every human into a sovereign node, every node into a living seed, and
every verified act of growth into shared intelligence, capability, and value.

**Version:** v1.0.0 | **Anchored to:** constitution.toml v5.0.0-GENESIS
```

### Section 2: Five Invariants (table)

```pseudocode
FOR EACH output IN system:
    ASSERT ihsan_score(output) >= 0.95        # I-1: Excellence gate
    ASSERT snr(output) >= 0.85                # I-2: Signal quality
    ASSERT gini(ledger) <= 0.35               # I-3: Justice constraint
    ASSERT private_key.location == LOCAL_ONLY  # I-4: Sovereignty
    ASSERT receipt_chain.valid()              # I-5: Accountability

    IF ANY invariant FAILS:
        REJECT output  # fail-closed, no exceptions
```

Each invariant must cite:
- The constant name in `core/integration/constants.py`
- The enforcement function
- The test file that proves it

### Section 3: 7-Layer Stack (table with hyperlinks)

```pseudocode
LAYERS = [
    Layer(0, "Human Seed",          "Constitutional anchor",           NA),
    Layer(1, "Sovereign Node",      "core/sovereign/identity_genesis", 332),
    Layer(2, "Agentic Development", "core/sovereign/mission",          38),
    Layer(3, "Verification",        "core/proof_engine/",              50+),
    Layer(4, "Learning",            "core/sovereign/seed_engine",      46),
    Layer(5, "Economic",            "core/token/",                     100+),
    Layer(6, "Civilizational",      "core/federation/ + core/a2a/",    60+),
]

FOR EACH layer IN LAYERS:
    ASSERT layer.source_exists()
    ASSERT layer.test_count > 0 OR layer.id == 0
```

### Section 4: Node Lifecycle (7 stages)

```pseudocode
HUMAN_STAGES = {
    "Seed":      (0.00, 0.10),  # First install, identity created
    "Node":      (0.10, 0.20),  # First mission completed
    "Apprentice": (0.20, 0.35), # Consistent qualified episodes
    "Builder":   (0.35, 0.55),  # Compiled first reflex
    "Verifier":  (0.55, 0.70),  # Attesting other nodes' work
    "Mentor":    (0.70, 0.85),  # Skills published to marketplace
    "Catalyst":  (0.85, 1.00),  # Network effect multiplier
}

FUNCTION human_stage(sovereignty_score: float) -> str:
    FOR stage_name, (low, high) IN reversed(HUMAN_STAGES):
        IF sovereignty_score >= low:
            RETURN stage_name
    RETURN "Seed"
```

Parallel to agent tiers (Novice → Grandmaster). Both earned through verified work.

### Section 5: Reward Loop (4 steps)

```pseudocode
LOOP forever:
    1. EARN   — Complete a mission (work verified by SAT-5 Oracle)
    2. VERIFY — Pass Ihsan gate (6-dim tensor, fail-closed)
    3. COMPILE — 3+ consecutive qualified → reflex precipitation
    4. TRADE  — Compiled reflex → skill on marketplace → SEED tokens

    # Compounding: each cycle raises sovereignty_score
    # which unlocks higher-trust missions (RAC-gated)
    # which produce harder-to-compile reflexes
    # which are worth more on the marketplace
```

### Section 6: KPI Formula

```pseudocode
FUNCTION node_value(node) -> float:
    potential   = node.seed_engine.potential().sovereignty_score  # 0-1
    activation  = node.missions_per_day                          # DAM
    quality     = node.average_ihsan_6dim                        # 0-1
    compounding = node.age_days * node.streak_multiplier         # time factor
    synergy     = node.asabiyyah * node.attestation_count        # network factor

    RETURN potential * activation * quality * compounding * synergy
```

### Section 7: Moat (3 sentences)

```pseudocode
# Not included in the formal kernel, but referenced as the "why"
# 1. Every new node brings hardware (compute), data (knowledge), and
#    intelligence (compiled reflexes) to the network.
# 2. Performance IMPROVES with growth (reverse scaling).
# 3. Value accrues to nodes, not to the platform.
```

## TDD Anchors

```pseudocode
TEST "kernel document exists and has all sections":
    doc = read("docs/KERNEL.md")
    ASSERT "Mission" IN doc.sections
    ASSERT "Invariants" IN doc.sections
    ASSERT "7-Layer Stack" IN doc.sections
    ASSERT "Node Lifecycle" IN doc.sections
    ASSERT "Reward Loop" IN doc.sections
    ASSERT "KPI Formula" IN doc.sections

TEST "every invariant citation resolves":
    FOR EACH invariant IN kernel.invariants:
        ASSERT file_exists(invariant.source_file)
        ASSERT line_contains(invariant.source_file, invariant.line, invariant.constant)

TEST "every layer source exists":
    FOR EACH layer IN kernel.layers:
        IF layer.id > 0:
            ASSERT path_exists(layer.source_path)

TEST "kernel is under 200 lines":
    doc = read("docs/KERNEL.md")
    ASSERT len(doc.lines) <= 200
```
