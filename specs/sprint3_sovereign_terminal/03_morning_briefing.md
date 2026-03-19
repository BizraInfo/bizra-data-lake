# 03 — Morning Briefing: Your AI Knows You

## Standing on Giants
- **Tulving (1972)**: Episodic vs semantic memory
- **Park et al. (2023)**: Generative agent memory architecture
- **OpenClaw**: Persistent context across sessions
- **MMRPG**: Login splash with daily rewards

## The Experience

```
  ╔═══════════════════════════════════════════════════════╗
  ║         بِذْرَة — BIZRA SOVEREIGN AI                  ║
  ╚═══════════════════════════════════════════════════════╝

  Good morning, Mumo.
  ▸ It's been 14 hours. Here's what I remember.
  ▸ Your main focus has been Architecture (23 missions).
  ▸ Last mission: P3 Artisan completed "wire identity registry" — Ihsan 0.97.
  ▸ Pattern "organize files" is 4/5 — 1 more to compile a reflex.
  ▸ Your quality scores are trending up. Excellent trajectory.
  ▸ You're on a 12-mission streak. Personal best: 15.
  ▸ Your test suite could use attention — shall I run diagnostics?
  ▸ Ready for your next mission.

  SEED: 1,124,707  |  Tier: Expert  |  Streak: 12  |  Ihsan: 0.973 ↑
```

## Pseudocode: Enhanced Briefing

```python
def enhanced_briefing(memory, rust_bridge=None):
    """Generate morning briefing with MMRPG progression elements."""
    lines = memory.generate_morning_briefing()

    # Add MMRPG status bar
    u = memory.user
    seed = u.total_missions * 12  # Approximate SEED from missions
    trend = "↑" if u.ihsan_trend == "improving" else "↓" if u.ihsan_trend == "declining" else "→"

    status_bar = (
        f"SEED: {seed:,}  |  "
        f"Tier: {u.tier}  |  "
        f"Streak: {u.current_streak}  |  "
        f"Ihsan: {u.avg_ihsan:.3f} {trend}"
    )
    lines.append("")
    lines.append(status_bar)

    # Check Rust feedback for system-level insights
    if rust_bridge:
        feedback = rust_bridge.poll_feedback()
        if feedback.get("compile_pending", 0) > 0:
            lines.append("⚡ New reflexes compiled overnight!")
        if feedback.get("reinforce_pending", 0) > 5:
            lines.append("Your agents learned from successful patterns while idle.")

    return lines
```

## Daily Rewards (MMRPG Pattern)

| Day | Reward | Condition |
|-----|--------|-----------|
| Login streak 3 | +50 SEED bonus | 3 consecutive days |
| Login streak 7 | +200 SEED + skill point | 7 consecutive days |
| Login streak 30 | +1000 SEED + tier unlock | 30 consecutive days |
| First mission of day | +10 SEED | Complete any mission |
| Perfect Ihsan day | +100 SEED | All missions ≥ 0.95 |

## Integration

- `LivingMemory.generate_morning_briefing()` — already exists in brain.py
- Called at `bizra` TUI startup (interactive mode)
- Called by proactive engine at user's detected wake time
- Rust heartbeat triggers memory sync before briefing generates
