# Sprint 3 — Sovereign Terminal: From Backend to Product

## Context

Sprint 1 wired the proof pyramid. Sprint 2 made it trustworthy.
Sprint 3 makes it **usable**. The market proved terminal-first wins
(Claude Code, OpenClaw, Codex). BIZRA has more than all of them —
but users can't see it yet.

**Standing on Giants**: OpenClaw (proactive), Agent Zero (OS-as-tool),
Claude Code (hooks/skills/MCP), Prexiet (persistent memory),
MMRPG economies (WoW/Minecraft/Islamic Finance)

## Sprint 3 Objectives

```
FROM: 30 commits, 26 crates, 12,644 tests, Block 0, but users can't touch it
TO:   `bizra` command → 12 agents work for you, remember you, prove everything
```

## Spec Files

| File | Content | Priority |
|------|---------|----------|
| `01_proactive_mode.md` | 24/7 agent operation (OpenClaw pattern) | P0 |
| `02_mission_ux.md` | Mission input → agent routing → receipt → display | P0 |
| `03_morning_briefing.md` | Living Memory → personalized greeting at boot | P0 |
| `04_skill_marketplace.md` | Install/create agent skills (Claude Code pattern) | P1 |
| `05_mmrpg_progression.md` | XP, levels, skill tree, achievements | P1 |

## Success Criteria

1. `bizra` starts, shows morning briefing, accepts missions
2. Missions route through Ollama, produce signed receipts
3. Living Memory remembers user across sessions
4. Proactive mode suggests actions without being asked
5. User sees their SEED balance, agent roster, receipt chain
