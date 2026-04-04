# BIZRA Node CLI v0 — Master Specification
## The Sovereign Mission Terminal
### Version: 0.1.0 | Date: 2026-04-04

---

## 1. Identity

**BIZRA Node CLI = Your Sovereign Mission Terminal**

One-liner: **Objective in. Receipts out. Sovereignty retained.**

**Not:** a terminal chatbot. Not a local model launcher. Not a coding CLI clone.
**Is:** a mission-centric operating system surface for one user on one machine.

## 2. Existing Foundation

| Asset | Location | State |
|-------|----------|-------|
| Bash TUI | `scripts/bizra` (44KB) | 15 commands, interactive mode, banner |
| Rust CLI | `bizra-omega/bizra-cli/` | clap parser, 5 widgets, Python bridge |
| Rust binary | `bizra-omega/target/release/bizra-node` | 2.7MB sovereign binary |
| Python kernel | `core/sovereign/` | Mission execution, GoT, SNR, receipts |

**Decision:** Rust CLI is the production target. Bash script becomes dev/fallback.
Python kernel remains the cognitive engine — Rust CLI is the shell/surface.

## 3. Architecture

```
┌─────────────────────────────────────────────────────┐
│                 BIZRA Node CLI (Rust)                 │
│           bizra-omega/bizra-cli/src/                  │
│                                                       │
│  ┌──────────┐ ┌──────────┐ ┌──────────┐             │
│  │ Commands │ │ TUI App  │ │ Widgets  │             │
│  │ (clap)   │ │ (ratatui)│ │ (panels) │             │
│  └────┬─────┘ └────┬─────┘ └────┬─────┘             │
│       └─────────┬───┘            │                    │
│                 │                │                    │
│       ┌─────────▼────────────────▼──────┐            │
│       │        Node Runtime Bridge       │            │
│       │   (REST to kernel / direct Rust) │            │
│       └──────────────┬──────────────────┘            │
└──────────────────────┼───────────────────────────────┘
                       │
         ┌─────────────▼─────────────┐
         │    Python Kernel (8010)    │   OR   Rust Engine (8080)
         │    Mission execution       │
         │    GoT + SNR + receipts   │
         └───────────────────────────┘
```

**L4 contract enforced:** CLI reads kernel truth via REST. Never originates receipts.

## 4. Command Grammar (8 commands)

```
bizra                          # Launch TUI (primary screen)
bizra init                     # Boot local node context
bizra genesis                  # Mint node identity + agents
bizra agents [list|show|chat]  # 12-agent topology
bizra mission "<objective>"    # THE PRIMARY COMMAND
bizra trust [chain|verify]     # Constitutional status
bizra receipt [list|show|replay] # Inspect/verify receipts
bizra memory [list|search]     # Mission memory + reflexes
bizra node [status|health]     # Device/resource topology
```

### Mapping to existing commands

| v0 Command | Existing Rust | Existing Bash | Action |
|------------|---------------|---------------|--------|
| `bizra` (TUI) | `Commands::Tui` | interactive mode | Enhance with new widgets |
| `bizra init` | — | partial in `status` | NEW |
| `bizra genesis` | — | — | NEW (maps to `bizra-node` genesis) |
| `bizra agents` | `Commands::Agent` | `agents` | Extend with SAT-5 |
| `bizra mission` | `Commands::Query` | `mission` | Rewrite to governed loop |
| `bizra trust` | — | `receipts` | NEW |
| `bizra receipt` | — | `receipts` | NEW (split from trust) |
| `bizra memory` | — | `memory` | NEW |
| `bizra node` | `Commands::Status` | `status` | Extend |

## 5. Primary Screen (TUI Layout)

```
╔══════════════════════════════════════════════════════════════════════╗
║  بِذْرَة  BIZRA Node CLI v0.1.0          node0-momo    Dubai GMT+4  ║
╠══════════════════╦═══════════════════╦═══════════════════════════════╣
║  PAT-7           ║  Active Mission   ║  SAT-5                       ║
║  ♟ Strategist  ● ║                   ║  ⚖ FairVote       ●          ║
║  🔍 Researcher  ● ║  "Analyze the     ║  🛡 HarmFilter     ●          ║
║  ⚙ Developer   ● ║   Ihsan scoring   ║  📜 Constitutional  ●          ║
║  📊 Analyst     ● ║   pipeline..."    ║  🔒 SecurityGate   ●          ║
║  ✓ Reviewer    ● ║                   ║  📊 QualityAudit   ●          ║
║  ▶ Executor    ● ║  ┌─────────────┐  ║                               ║
║  🛡 Guardian    ● ║  │ SNR: 0.61   │  ║                               ║
║                   ║  │ Ihsan: 61%  │  ║                               ║
║                   ║  │ Gate: REVIEW│  ║                               ║
║                   ║  └─────────────┘  ║                               ║
╠══════════════════╩═══════════════════╩═══════════════════════════════╣
║  Ghost Feed                                                          ║
║  🔮 Guardian: "SNR below threshold — consider adding citations"       ║
║  🔮 Strategist: "Similar mission PERMIT'd 2 days ago — warm path?"   ║
║  🔮 Analyst: "FAISS index has 102K vectors, 3 relevant clusters"     ║
╠══════════════════════════════════════════════════════════════════════╣
║  Receipt Rail                              Trust Rail                ║
║  ✉ fa5f4e98 REVIEW 00:48     Chain: 34 entries ✓ INTACT             ║
║  ✉ d01419b6 REVIEW 21:39     Parity: 5/5 ✓     SEED: 1,124,695    ║
╠══════════════════════════════════════════════════════════════════════╣
║  بذرة › _                                                           ║
╚══════════════════════════════════════════════════════════════════════╝
```

## 6. Proof Card (Mission Output)

Every completed mission ends with:

```
╔══════════════════════════════════════════════════════════════╗
║  MISSION COMPLETE                                            ║
╠══════════════════════════════════════════════════════════════╣
║  Verdict:   ⚠ REVIEW                                        ║
║  SNR:       0.6150  ░░░░░░░█████░░░░ (threshold: 0.85)     ║
║  Ihsan:     61.50%  ░░░░░░░█████░░░░ (threshold: 95%)      ║
║  Receipt:   fa5f4e98f5b6cdaa7cf5a26fa769b9ca                ║
║  Evidence:  seq=34, chain INTACT                             ║
║  Agents:    4/7 PAT contributed                              ║
║  Tokens:    +0.12 SEED, +30.28 IMPT, -0.003 zakat          ║
║  Memory:    30 entries persisted                              ║
║                                                              ║
║  Replay:    bizra receipt replay fa5f4e98                     ║
╚══════════════════════════════════════════════════════════════╝
```

## 7. First-Run Experience (< 15 minutes)

```
Step 1: bizra init        → Scan environment (30s)
Step 2: bizra genesis     → Mint identity + agents (60s)
Step 3: bizra agents      → See your 12-agent parliament
Step 4: bizra mission "…" → First real mission (2-5 min)
```

Total: ~8 minutes to first receipt. Well under 15-minute target.

## 8. MVP Proves 5 Things

| # | Capability | How CLI Proves It |
|---|-----------|-------------------|
| 1 | Local sovereignty | `bizra init` discovers local resources, no cloud |
| 2 | 12-agent parliament | `bizra agents` shows PAT-7 + SAT-5 live |
| 3 | Mission execution | `bizra mission` runs governed loop |
| 4 | Receipt + trust | Proof card + `bizra trust` chain verification |
| 5 | Memory / reflex | `bizra memory` shows carryover, next run faster |

## 9. What NOT to Build in v0

- Marketplace / skill trading
- Federation / multi-node discovery
- Blockchain / on-chain anything
- 50+ commands
- Settings UI
- Model management UI (BYOB = just works)

## 10. Phase Files

| File | Scope |
|------|-------|
| `01_first_run_flow.md` | init → genesis → agents → first mission |
| `02_mission_command.md` | The primary command — full governed loop |
| `03_tui_layout.md` | Parliament view, trust rail, ghost feed |
| `04_proof_card.md` | Receipt display, verification, replay |
| `05_memory_reflex.md` | Carryover between missions, warm path |

---

*"I gave BIZRA one real mission, it executed it locally, showed me which*
*agents worked on it, proved what it did, remembered the result, and*
*made the next run faster." — That is the moment.*
