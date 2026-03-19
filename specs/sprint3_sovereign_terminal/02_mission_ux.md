# 02 — Mission UX: Task → Agents → Receipt → Display

## Problem

User types a task. What happens? Right now: raw protocol output.
Need: beautiful, informative, trust-building display.

## The Flow

```
User: "Organize my Downloads folder"
  │
  ├─ P1 Navigator classifies: type=file_management, complexity=medium
  ├─ P2 Scholar retrieves: user prefers folders by date (from memory)
  ├─ P3 Artisan plans: scan → classify → move → rename
  ├─ P4 Guardian checks: no sensitive files exposed? Daughter Test?
  │
  ├─ Execution: Ollama inference → action plan → execute steps
  │
  ├─ P5 Mentor extracts: "user organizes downloads weekly" (new pattern)
  ├─ Receipt: BLAKE3-chained, Ed25519-signed, Ihsan scored
  │
  └─ Display:
      ╔═══════════════════════════════════════════════════╗
      ║  ✓ Mission Complete — Organize Downloads          ║
      ╠═══════════════════════════════════════════════════╣
      ║  Agents: P1→P2→P3→P4 (3 consulted, Guardian ✓)  ║
      ║  Actions: 23 files moved, 4 folders created       ║
      ║  Ihsan: 0.97  |  Model: qwen2.5:3b               ║
      ║  Receipt: f10aa36b...  (chain #47)                ║
      ║  SEED: +12 earned  |  Balance: 1,124,707          ║
      ║  Memory: "weekly organizer" pattern → 4/5 compile ║
      ╚═══════════════════════════════════════════════════╝
```

## Pseudocode: Mission Display

```python
def display_mission_result(result):
    """Rich terminal output for a completed mission."""
    r = result

    # Header
    status = "✓" if r.guardian_approved else "✗"
    color = GREEN if r.guardian_approved else RED
    print(f"  {color}{status} Mission Complete — {r.task[:50]}{RESET}")

    # Agent chain
    agents = " → ".join(r.agents_consulted)
    print(f"  Agents: {agents} (Guardian {'✓' if r.guardian_approved else '✗ VETO'})")

    # Inference
    if r.inference_executed:
        print(f"  Model: {r.model}  |  Ihsan: {r.ihsan:.2f}")

    # Receipt (the trust differentiator)
    print(f"  Receipt: {r.receipt_id[:12]}...  (chain #{r.chain_position})")

    # Economy
    seed_delta = calculate_seed_reward(r)
    print(f"  SEED: +{seed_delta} earned  |  Balance: {r.balance:,}")

    # Memory update
    if r.near_compile:
        remaining = 5 - r.pattern_count
        print(f"  Memory: \"{r.pattern_name}\" → {r.pattern_count}/5 compile")

    # Reflex compiled!
    if r.reflex_compiled:
        print(f"  ⚡ NEW REFLEX: \"{r.pattern_name}\" compiled — instant next time")
```

## MMRPG Elements in Mission Display

| Element | MMRPG Equivalent | Display |
|---------|-----------------|---------|
| Ihsan score | Damage dealt | Quality bar |
| SEED earned | Gold/XP gained | +N SEED |
| Pattern progress | Skill XP bar | 4/5 → compile |
| Reflex compiled | Skill learned! | ⚡ animation |
| Chain position | Achievement count | chain #47 |
| Guardian approval | Loot roll success | ✓ or ✗ |

## TDD Anchors

```python
def test_mission_display_shows_receipt():
    result = mock_mission_result(guardian_approved=True)
    output = capture_display(result)
    assert "Receipt:" in output
    assert len(result.receipt_id) == 64

def test_mission_display_shows_seed():
    result = mock_mission_result(seed_earned=12)
    output = capture_display(result)
    assert "+12" in output

def test_vetoed_mission_shows_rejection():
    result = mock_mission_result(guardian_approved=False)
    output = capture_display(result)
    assert "VETO" in output
```
