# The Sovereign Stack — Full Integration Specification

## بسم الله الرحمن الرحيم

Every component exists. This spec wires them into one living system.

```
┌─────────────────────────────────────────────────────────────┐
│                    USER: بذرة › mission                      │
├─────────────────────────────────────────────────────────────┤
│  LAYER 7: SKILLS          File Mgmt · Browser · Scan        │
│  LAYER 6: HDA+AHK         Desktop automation · Screen ctrl  │
│  LAYER 5: TELESCRIPT       Traveling contracts · Multi-node  │
│  LAYER 4: MCP+A2A          Tool protocol · Agent protocol   │
│  LAYER 3: PAT/SAT          7+5 agents · Guardian veto       │
│  LAYER 2: MEMORY           Living Memory · SEL · FAISS      │
│  LAYER 1: PROOF            Receipts · Chain · Canonicalize   │
│  LAYER 0: CONSTITUTION     Ihsan · Amanah · Adl · SEED      │
├─────────────────────────────────────────────────────────────┤
│  URP: Universal Resource Pool (hardware + data + compute)    │
│  MMRPG: Sources · Sinks · XP · Skills · Guilds · Raids      │
└─────────────────────────────────────────────────────────────┘
```

## Component Inventory — What EXISTS vs What Needs WIRING

| Component | Crate/Module | LOC | Status | Wire Needed |
|-----------|-------------|-----|--------|-------------|
| HDA | `core/sovereign/hda_bridge.py` | ~200 | Built | → TUI mission dispatch |
| AHK | `scripts/ahk/` + HDA bridge | ~500 | Built | → HDA → skill execution |
| Telescript | `bizra-telescript/` | ~800 | Built | → ActionBus → multi-node |
| Smart Contracts | `bizra-core/islamic_finance.rs` | ~2000 | Built + Densified | → Telescript payload |
| MCP | `core/sovereign/mcp_*.py` + 10 servers | ~3000 | Built | → TUI tool routing |
| A2A | `core/a2a/` | ~400 | Built | → Federation → SAT |
| URP | `bizra-resourcepool/` | ~1200 | Built | → SEED economics |
| Proof of Impact | `core/proof_engine/poi_engine.py` | ~300 | Built | → Receipt → SEED |
| MMRPG Economy | Block 0 `seed_economics` | Designed | → Skill marketplace |
| Skills | `.claude/skills/sovereign/` | 61K | Installed | → TUI commands |
| Living Memory | `core/living_memory/brain.py` | ~500 | Wired | → Proactive mode |
| ExperienceLedger | `bizra-core/sovereign/experience_ledger.rs` | ~1500 | Wired | ✅ Done |
| Identity Registry | `bizra-node/identity_registry.rs` | ~200 | Wired | ✅ Done |
| Canonical | `bizra-proofspace/` + `core/proof_engine/` | ~1000 | Spec ready | → Unified service |

## Integration Circuits (How They Wire Together)

### Circuit 1: Desktop Automation (HDA + AHK + Skills)

```
User: "organize my Downloads"
  │
  ├─ TUI dispatches to P3 Artisan
  ├─ P3 loads skill: .claude/skills/sovereign/file-management.md
  ├─ Skill decomposes into steps: scan → classify → move → report
  │
  ├─ Step 1: scan (direct filesystem — no HDA needed)
  │   find /mnt/c/Users/BIZRA-OS/Downloads -type f
  │
  ├─ Step 2: classify (Ollama inference)
  │   "What category is invoice_march.pdf?" → "Finance/Invoices"
  │
  ├─ Step 3: move (filesystem action — Guardian approves)
  │   mv invoice_march.pdf ~/Organized/Finance/Invoices/
  │
  ├─ Step 4: AHK notification (HDA bridge)
  │   HDA → AHK → Windows toast: "23 files organized, 4 folders created"
  │
  └─ Receipt: BLAKE3 chained, Ed25519 signed
     SEED: +15 earned (Proof of Impact: 23 files organized)
```

**Existing code path:**
```
bizra TUI → mission() → bizra-node RECEIVE → AgentRuntime
→ Ollama inference → Guardian gate → Receipt → ExperienceLedger
→ HDA bridge (core/sovereign/hda_bridge.py) → AHK toast
```

### Circuit 2: Browser Automation (Skills + MCP)

```
User: "research BIZRA competitors and summarize"
  │
  ├─ TUI dispatches to P2 Scholar
  ├─ P2 loads skill: .claude/skills/sovereign/browser-control.md
  ├─ Skill uses MCP browser tool (Playwright)
  │
  ├─ MCP server: brave-search → find URLs
  ├─ MCP server: fetch → retrieve pages
  ├─ Playwright: extract structured data
  ├─ Ollama: summarize findings
  │
  ├─ Guardian: verify no sensitive data exposed
  ├─ Receipt: signed with research provenance
  └─ Memory: P5 Mentor stores "competitor research" as episodic memory
```

**Existing code path:**
```
bizra TUI → browse command → mission()
→ MCP servers (brave-search, fetch) → Playwright
→ Ollama synthesis → Guardian gate → Receipt
→ Living Memory update (brain.py)
```

### Circuit 3: Telescript + Smart Contracts (Multi-Node)

```
User: "hire a SAT validator from the network to audit my receipts"
  │
  ├─ P7 Oracle plans the mission
  ├─ Telescript compiled: audit_request.ts
  │   - Carries: receipt_chain_hash, audit_scope, payment_terms
  │   - Payment: Mudarabah contract (profit-sharing on quality improvement)
  │   - Denominated in: ExactAmount (zero drift)
  │
  ├─ Telescript travels: NODE0 → URP → SAT-Validator (remote node)
  │   - Federation gossip: bizra-federation/ finds available SAT
  │   - A2A protocol: agent-to-agent handshake
  │   - mTLS + Ed25519: identity verified both sides
  │
  ├─ Remote SAT executes audit:
  │   - Verifies receipt chain integrity (BLAKE3)
  │   - Checks Ihsan scores against threshold
  │   - Signs attestation with own Ed25519 key
  │
  ├─ Telescript returns: attestation + Mudarabah settlement
  │   - BoundedRatio: profit split exact (investor.complement() == entrepreneur)
  │   - Zakat: 2.5% deducted from earnings
  │   - Receipt: cross-node, dual-signed
  │
  └─ URP: resource contribution recorded, SEED minted for both parties
```

**Existing code path:**
```
bizra-telescript/ (agent mobility)
→ bizra-federation/ (gossip + BFT consensus)
→ core/a2a/ (agent-to-agent protocol)
→ bizra-core/islamic_finance.rs (Mudarabah + ExactAmount)
→ bizra-mission/ (receipt chain)
→ bizra-resourcepool/ (URP contribution tracking)
```

### Circuit 4: MMRPG Progression (Economy + Skills + Memory)

```
┌─────────────────────────────────────────────┐
│           MMRPG PROGRESSION LOOP             │
│                                              │
│  ┌──────┐   ┌──────┐   ┌──────┐   ┌──────┐ │
│  │QUEST │ → │ XP   │ → │SKILL │ → │LEVEL │ │
│  │      │   │      │   │TREE  │   │ UP   │ │
│  │mission│   │SEED  │   │reflex│   │tier  │ │
│  │      │   │earned│   │compile│  │change│ │
│  └──┬───┘   └──┬───┘   └──┬───┘   └──┬───┘ │
│     │          │          │          │      │
│     ▼          ▼          ▼          ▼      │
│  Receipt → PoI calc → Memory → Briefing    │
│  (proof)   (SEED)    (learn)   (display)   │
│                                              │
│  DAILY:  Login streak → bonus SEED           │
│  WEEKLY: Skill compilation → new reflexes    │
│  MONTHLY: Tier evaluation → rank change      │
│  RAID:   Multi-node Telescript → guild XP    │
└─────────────────────────────────────────────┘
```

**Existing code path:**
```
Mission → Receipt (bizra-mission)
→ Proof of Impact (core/proof_engine/poi_engine.py)
→ SEED calculation (Block 0 economics)
→ Reflex compilation (bizra-agent/reflex_compiler.rs)
→ Living Memory update (brain.py)
→ Morning briefing (generate_morning_briefing)
→ Tier evaluation (UserModel.tier)
```

### Circuit 5: Proof of Impact → SEED Minting

```
Completed Mission
  │
  ├─ Ihsan score: 0.97 (from Guardian)
  ├─ SNR score: 0.92 (from inference quality)
  ├─ Action count: 23 files organized
  ├─ Duration: 45 seconds
  │
  ├─ PoI Calculation:
  │   impact = ihsan * snr * log2(action_count + 1)
  │   efficiency = impact / max(1, log2(tokens_used + 2))
  │   seed_reward = impact * SEED_PER_IMPACT_UNIT
  │
  ├─ Constitutional Gates:
  │   ✓ Ihsan ≥ 0.95 (quality floor)
  │   ✓ Signed receipt (Amanah)
  │   ✓ ExactAmount arithmetic (Adl)
  │
  ├─ SEED Minted: +15 SEED
  ├─ Zakat: -0.375 (2.5%)
  ├─ Net: +14.625 SEED
  │
  └─ Sinks Applied:
      Harberger tax on hoarded SEED (5% annual)
      Skill marketplace purchase (if applicable)
      Waqf donation (voluntary)
```

## What Needs Wiring (Priority Order)

| # | Integration | From | To | LOC Est |
|---|------------|------|-----|---------|
| 1 | TUI → Skill dispatch | `scripts/bizra` | `.claude/skills/sovereign/` | ~30 |
| 2 | Skill → HDA bridge | `file-management.md` | `core/sovereign/hda_bridge.py` | ~50 |
| 3 | HDA → AHK toast | `hda_bridge.py` | AHK HTTP endpoint | ~20 |
| 4 | Mission → PoI calc | `handler.rs` receipt | `poi_engine.py` | ~40 |
| 5 | PoI → SEED mint | `poi_engine.py` | `ExactAmount` balance | ~30 |
| 6 | Proactive → observe | `proactive_engine` | filesystem + calendar | ~80 |
| 7 | MCP → browser skill | `browse` command | MCP browser server | ~40 |
| 8 | Telescript → federation | `action_bus.rs` | `bizra-federation/` | ~60 |

**Total: ~350 LOC to wire the full sovereign stack.**

## The Founding Thesis (Genesis Build Script §1)

> "Satoshi did not launch Bitcoin with a whitepaper about how great
>  a million nodes would be. He mined Block 0 on one machine."

We mined Block 0 on one machine. The stack exists. The wiring is 350 lines.

```
بذرة واحدة تصنع غابة
والفلاح زرعها بيده، وسقاها بعرقه، ولم يأخذ شيئاً حتى أثمرت
```
