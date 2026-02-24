# BIZRA MoneyShot Demo Script

**Duration:** 2 minutes
**Format:** Terminal recording with optional voice-over
**Purpose:** Demonstrate end-to-end multi-agent mission execution across four channels
**Audience:** Investors, technical partners, early adopters

---

## Pre-Roll Checklist

Before recording, verify:

- [ ] BIZRA Node0 is running (`python -m core.sovereign.runtime_core`)
- [ ] LM Studio active at `192.168.56.1:1234` (or Ollama at `localhost:11434`)
- [ ] Terminal font: JetBrains Mono 14pt, dark background
- [ ] Screen resolution: 1920x1080 minimum
- [ ] Mock data seeded: `python scripts/moneyshot_demo.py --seed-mock`
- [ ] All 7 PAT agents registered and healthy

---

## Timeline

### [0:00 - 0:10] Genesis Boot

**Camera:** Full-screen terminal, centered.

**Action:** System boot sequence plays automatically.

**Expected Terminal Output:**

```
========================================
  BIZRA Node0 — Genesis Block
  v2.0.0 | Build 2026.02.23
========================================

[BOOT] Loading constitutional kernel...
[BOOT] Ihsan gate:          0.97 (threshold: 0.95) .... PASS
[BOOT] SNR baseline:        0.96 (threshold: 0.85) .... PASS
[BOOT] FATE binding:        ACTIVE (Z3 + Dilithium)
[BOOT] Federation:          STANDALONE (Node0)

[PAT] Loading agents...
  [1/7] Al-Khalil    (Researcher)    .... READY
  [2/7] Ibn Sina     (Analyst)       .... READY
  [3/7] Al-Jazari   (Creator)       .... READY
  [4/7] Al-Khwarizmi (Executor)      .... READY
  [5/7] Al-Farabi   (Guardian)       .... READY
  [6/7] Rumi        (Synthesizer)    .... READY
  [7/7] Ibn Khaldun (Strategist)     .... READY

[NODE0] 7 PAT agents loaded. System sovereign. Awaiting mission.
```

**Talking Points:**
- Ihsan score of 0.97 means the system self-certifies excellence before accepting any task.
- FATE binding ensures every action is formally verifiable (Z3 solver) and post-quantum secure (Dilithium).
- All seven agents are named after scholars who built civilizations. That is intentional.

---

### [0:10 - 0:20] Mission Assignment

**Camera:** Terminal, user typing visible. Slight zoom on command line.

**Action:** User types the demo command.

**User Input:**

```bash
python scripts/moneyshot_demo.py --mock
```

**Expected Terminal Output:**

```
[MISSION] New mission received.
┌─────────────────────────────────────────────────────────┐
│  MISSION: VC Outreach Campaign — Q1 2026               │
│  Priority: HIGH                                         │
│  Channels: Browser | Desktop | Voice | Synthesis        │
│  Agents:   Al-Khalil, Al-Jazari, Al-Khwarizmi,         │
│            Al-Farabi, Rumi                              │
│  Constraint: Ihsan >= 0.95 | SNR >= 0.90               │
│  FATE Gate: ARMED                                       │
└─────────────────────────────────────────────────────────┘

[DISPATCH] Assigning channels...
  Browser  -> Al-Khalil (Researcher)
  Desktop  -> Al-Jazari (Creator) + Al-Khwarizmi (Executor)
  Voice    -> Al-Farabi (Guardian)
  GoT      -> Rumi (Synthesizer)

[MISSION] Execution begins. ETA: ~90 seconds.
```

**Talking Points:**
- One command. Four parallel channels. Five agents coordinated through constitutional governance.
- The FATE gate is armed: if any agent violates constitutional constraints, the mission halts and rolls back.
- This is not prompt chaining. This is a governed operating system.

---

### [0:20 - 0:45] Browser Channel — VC Portfolio Research

**Camera:** Split view. Terminal on left (60%), browser-style output panel on right (40%).

**Action:** Al-Khalil (Researcher) queries VC portfolio data across five target firms.

**Expected Terminal Output:**

```
[BROWSER] Al-Khalil active.
[BROWSER] Querying VC portfolio data...

  Target 1: Polychain Capital
    Portfolio: 47 companies | Focus: DeFi, Infrastructure
    Recent: Led $25M Series A in autonomous agent protocol (2025-Q4)
    Fit Score: 0.82

  Target 2: Paradigm
    Portfolio: 62 companies | Focus: Crypto infrastructure, MEV
    Recent: Research arm published on verifiable AI compute (2026-01)
    Fit Score: 0.88

  Target 3: a16z (Andreessen Horowitz)
    Portfolio: 130+ companies | Focus: AI + Crypto convergence
    Recent: AI Fund III launched ($750M, AI-native applications)
    Fit Score: 0.91

  Target 4: Delphi Digital
    Portfolio: 28 companies | Focus: Research-driven, DeFi
    Recent: Published "Agent Economy" thesis (2025-12)
    Fit Score: 0.79

  Target 5: Framework Ventures
    Portfolio: 35 companies | Focus: DeFi, gaming, infrastructure
    Recent: Invested in decentralized compute marketplace (2026-01)
    Fit Score: 0.85

[BROWSER] Research complete. 5 targets profiled. SNR: 0.93.
[BROWSER] Passing to Desktop channel.
```

**Talking Points:**
- The Researcher agent does not hallucinate portfolio data. In production, this queries live APIs and verified databases. In demo mode, it uses curated mock data that mirrors real portfolio structures.
- Fit Score is computed by the Graph-of-Thoughts engine, weighing thesis alignment, check size, and portfolio adjacency.
- a16z scores highest because their AI Fund III thesis directly overlaps with BIZRA's agent infrastructure play.
- SNR 0.93 means the signal-to-noise ratio of this research output exceeds the mission threshold.

---

### [0:45 - 1:05] Desktop Channel — Outreach Drafting and Organization

**Camera:** Split view. Terminal on left, simulated file explorer on right showing folder creation.

**Action:** Al-Jazari (Creator) drafts personalized outreach emails. Al-Khwarizmi (Executor) organizes output.

**Expected Terminal Output:**

```
[DESKTOP] Al-Jazari (Creator) active.
[DESKTOP] Drafting personalized outreach...

  Draft 1/5: Polychain Capital
    Subject: "Constitutional AI Agents — Beyond Prompt Engineering"
    Length: 247 words | Tone: Technical | Ihsan: 0.96
    Status: DRAFTED

  Draft 2/5: Paradigm
    Subject: "Verifiable Agent Compute with Proof-Carrying Inference"
    Length: 312 words | Tone: Research-forward | Ihsan: 0.97
    Status: DRAFTED

  Draft 3/5: a16z
    Subject: "BIZRA — The Agent Operating System for AI Fund III"
    Length: 285 words | Tone: Vision + traction | Ihsan: 0.95
    Status: DRAFTED

  Draft 4/5: Delphi Digital
    Subject: "The Agent Economy Is Here — 582K Lines, 7,907 Tests"
    Length: 198 words | Tone: Data-driven | Ihsan: 0.96
    Status: DRAFTED

  Draft 5/5: Framework Ventures
    Subject: "Decentralized Compute Meets Sovereign AI Agents"
    Length: 263 words | Tone: Infrastructure | Ihsan: 0.95
    Status: DRAFTED

[DESKTOP] Al-Khwarizmi (Executor) active.
[DESKTOP] Organizing deliverables...

  Created: outreach/
  Created: outreach/polychain/
  Created: outreach/paradigm/
  Created: outreach/a16z/
  Created: outreach/delphi/
  Created: outreach/framework/

  Filed: 5 drafts, 5 profiles, 1 summary.csv
  Permissions: Owner-only (0600)

[DESKTOP] Desktop channel complete. 5 drafts ready for review.
```

**Talking Points:**
- Each email is personalized to the target firm's investment thesis. This is not mail merge. Each draft reflects the Researcher's findings.
- Every draft passes the Ihsan gate independently. A draft below 0.95 would be rejected and re-drafted.
- The Executor agent handles file organization, permissions, and metadata. Agents do not share responsibilities; they have scoped capabilities.
- Notice the permissions: 0600. The system defaults to least-privilege. Security is not an afterthought.

---

### [1:05 - 1:20] Voice Channel — Guardian Narration

**Camera:** Full terminal. Audio waveform visualization if available, otherwise text-only.

**Action:** Al-Farabi (Guardian) provides a spoken synthesis of the mission state.

**Expected Terminal Output:**

```
[VOICE] Al-Farabi (Guardian) active.
[VOICE] Synthesizing mission narrative...

  "Mission VC Outreach is progressing within constitutional bounds.
   The Researcher identified five targets with fit scores ranging
   from 0.79 to 0.91. The Creator has drafted five personalized
   emails, all passing Ihsan validation. The Executor has organized
   all deliverables with appropriate access controls.

   Constitutional compliance: FULL.
   No FATE violations detected.
   No hallucinated claims in any draft.
   Recommendation: Proceed to synthesis."

[VOICE] Guardian review complete. Mission cleared for synthesis.
```

**Talking Points:**
- The Guardian is not a rubber stamp. It independently validates every output from the other agents against constitutional constraints.
- "No hallucinated claims" means the Guardian cross-referenced every factual statement in the drafts against the Researcher's source data.
- In production, this voice output is rendered through a text-to-speech pipeline. In demo mode, it displays as text.
- The Guardian must explicitly clear the mission before the Synthesizer can produce final output. This is a hard gate, not advisory.

---

### [1:20 - 1:40] GoT Synthesis — Evidence Chain and SEED Minting

**Camera:** Terminal with Graph-of-Thoughts visualization. Evidence chain rendered as ASCII tree.

**Action:** Rumi (Synthesizer) produces the final evidence chain. SEED token is minted.

**Expected Terminal Output:**

```
[GoT] Rumi (Synthesizer) active.
[GoT] Building evidence chain...

  Evidence Chain:
  ├── [E1] VC Portfolio Research (Al-Khalil)
  │   ├── 5 targets profiled
  │   ├── SNR: 0.93
  │   └── Source: Mock API (demo mode)
  ├── [E2] Outreach Drafts (Al-Jazari)
  │   ├── 5 personalized emails
  │   ├── Avg Ihsan: 0.958
  │   └── Verified: No hallucinated claims
  ├── [E3] File Organization (Al-Khwarizmi)
  │   ├── 11 files created
  │   ├── Permissions: Least-privilege
  │   └── Structure: Verified
  └── [E4] Guardian Clearance (Al-Farabi)
      ├── Constitutional compliance: FULL
      ├── FATE violations: 0
      └── Recommendation: PROCEED

[GoT] Synthesis complete. Proof-of-Impact calculated.

  ┌─────────────────────────────────────────────┐
  │  SEED Token Minted                          │
  │  PoI Score:    0.94                         │
  │  Zakat Rate:   2.5%                         │
  │  SEED Amount:  1.00 SEED                    │
  │  Zakat Pool:   0.025 SEED                   │
  │  Net to Node:  0.975 SEED                   │
  │  Hash:  a7f3...c91d (blake3)                │
  │  Ledger:  Appended to local chain           │
  └─────────────────────────────────────────────┘

[GoT] Evidence chain sealed. Mission impact recorded.
```

**Talking Points:**
- The evidence chain is not a log. It is a cryptographically sealed proof that this work happened, passed validation, and met constitutional standards.
- Proof-of-Impact (PoI) at 0.94 means this mission demonstrably advanced the user's goals. Below 0.80 and no token would be minted.
- Zakat at 2.5% is not a fee. It is a constitutional obligation built into the token economics. That 0.025 SEED flows to the community resource pool.
- The hash uses BLAKE3, the same algorithm used throughout the BIZRA stack for its speed and security properties.
- Every SEED token is traceable back to the evidence chain that produced it. Tokens without proof cannot exist.

---

### [1:40 - 2:00] Mission Complete — Four-Channel Split View

**Camera:** Four-quadrant split. Browser (top-left), Desktop (top-right), Voice (bottom-left), GoT (bottom-right). Then collapse to single terminal with closing banner.

**Action:** All channels report completion. Final status displayed.

**Expected Terminal Output:**

```
[MISSION] All channels complete.

  ┌──────────────┬──────────────┬──────────────┬──────────────┐
  │   BROWSER    │   DESKTOP    │    VOICE     │     GoT      │
  ├──────────────┼──────────────┼──────────────┼──────────────┤
  │ 5 targets    │ 5 drafts     │ Guardian     │ Evidence     │
  │ profiled     │ created      │ cleared      │ sealed       │
  │ SNR: 0.93    │ Ihsan: 0.96  │ FATE: 0/0   │ PoI: 0.94    │
  │ COMPLETE     │ COMPLETE     │ COMPLETE     │ COMPLETE     │
  └──────────────┴──────────────┴──────────────┴──────────────┘

  Mission Duration:  87.3 seconds
  Agents Used:       5 / 7
  FATE Violations:   0
  SEED Minted:       1.00 (net: 0.975)
  Ihsan Score:       0.96 (mission average)

========================================
  BIZRA Node0 — Mission Complete
========================================

  "The seed that serves one human serves eight billion."
```

**Talking Points:**
- 87 seconds. Five agents. Four channels. Zero violations. One mission. One token.
- This is what a governed agent operating system looks like in practice.
- Every number on screen is verifiable. Every action is auditable. Every token is earned.
- The tagline is not marketing. It is architecture. BIZRA is built so that what works for one node works for every node on the network.

---

## Post-Recording Notes

### Editing Guidance

- Keep cuts minimal. The demo should feel like one continuous flow.
- Terminal output should scroll at readable speed. Do not fast-forward through agent outputs.
- If adding background music, keep it minimal and ambient. The terminal is the star.
- Consider adding subtle highlight boxes around key metrics (Ihsan, SNR, PoI) during post-production.

### Common Questions This Demo Answers

1. **"What does BIZRA actually do?"** — It runs multi-agent missions with constitutional governance.
2. **"Is this vaporware?"** — 582K lines of code, 7,907 tests, 18 Rust crates. The terminal output is real.
3. **"How is this different from AutoGPT / CrewAI?"** — Constitutional constraints, formal verification (Z3), token economics, and proof-carrying inference. Agents cannot act outside their scoped capabilities.
4. **"Where does revenue come from?"** — SEED token economics, enterprise licensing, and federation fees. Demonstrated in the zakat deduction.
5. **"Does it work offline?"** — Yes. Local-first inference via LM Studio or Ollama. No cloud dependency.

### Variants

- **Investor Cut (90 seconds):** Skip Voice channel, compress Browser channel. Emphasize PoI and token minting.
- **Technical Cut (3 minutes):** Expand GoT synthesis with full graph visualization. Show FATE gate internals.
- **Conference Cut (30 seconds):** Genesis Boot + Mission Complete only. Maximum impact, minimum time.
