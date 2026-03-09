# BIZRA Terminal Information Architecture v1.0

> **Locked:** 2026-03-07
> **Scope:** 7 views for the sovereign terminal product
> **Rule:** The terminal is the primary control surface. Everything else is secondary.

---

## Design Principle

The terminal is not a shell wrapper. It is the sovereign cockpit.
It must feel like a cross between:
- a best-in-class terminal (speed, keyboard-first, scriptable),
- an MMORPG character sheet (progression, skills, stats),
- a mission control center (real-time status, proof, verification),
- and a personal operating system (memory, continuity, identity).

---

## View 1: BRIEFING (Home)

**Purpose:** Morning briefing from DEMA. The first thing the user sees.

**Content:**
- Time-aware greeting ("Good morning, Mumo")
- Time since last session
- Last mission summary + outcome
- Main active project
- Near-compiling reflex patterns (with count toward threshold)
- Quality trend (Ihsān over last 10 missions)
- Suggested next mission
- SEED/BLOOM balance (compact)

**Trigger:** Shown on terminal startup. Refreshable with `briefing` command.

**Data sources:**
- `sovereign_state/mission_signer.json` (identity)
- Living Memory — semantic store (user model)
- Living Memory — episodic store (last 10 missions)
- Living Memory — procedural store (near-compilation candidates)
- `/v1/health` (health metrics)
- `/v1/wallet/balance` (token balances)

**States:**
- `LIVE_BACKEND_DATA` — all data from live API
- `OFFLINE_SIMULATION` — derived from local state files

---

## View 2: MISSION (Console)

**Purpose:** One-line mission submission → autonomous execution → result return.

**Content:**
- Mission input (single line or multi-line with `\`)
- Live execution stream (agent routing, gate results, step progress)
- Route visibility: which agents were invoked, why
- Gate pass/reject visibility: Ihsān score per gate
- Step-level progress with timing
- Final result + receipt hash
- SEED earned + pool share
- Reflex candidate notification if applicable

**Interaction model (Two-Touch Principle):**
- Touch 1: User states the mission
- Touch 2: System returns with result

**Interrupt only when:**
- Constitutional risk is non-trivial → `REQUIRES_CONFIRMATION`
- Permission boundary exceeded → `PERMISSION_BLOCKED`
- Irreversible destructive operation → `REQUIRES_CONFIRMATION`
- Required data missing → `MISSION_ESCALATED`

**Data flow:**
1. User input → `mission.submitted` event
2. Bounded cognition estimates novelty/route
3. Constitutional gates evaluate
4. Agent runtime plans + executes
5. ActionBus executes through safe handlers
6. Verification occurs
7. Receipt produced + signed
8. Memory updated
9. Reflex candidate scoring
10. Terminal returns final result

---

## View 3: RECEIPTS (Proof Chain)

**Purpose:** Cryptographic evidence trail. Every action has a receipt.

**Content:**
- Receipt list (scrollable, filterable by mission/action/time)
- Per receipt: PoI score, Ihsān components (8 dims), CPVA, SEED yield, pool share
- Hash chain visualization (prev_hash → event_hash linkage)
- Verification status (signed, audited, or pending)
- Reason codes for rejections
- Export: JSON or CSV for external audit

**Data sources:**
- EventBus chain (local)
- `/v1/evidence/chain` (if backend available)

---

## View 4: WALLET (Economy)

**Purpose:** Make the economic engine visible, intelligible, and felt.

**Content:**
- Token balances: SEED (liquid), BLOOM (soulbound), BRANCH (reputation)
- Locked SEED (staking)
- Zakat contributed (cumulative 2.5%)
- Community pool share (cumulative 50%)
- Supply cap utilization (gauge: 0-100%)
- Earning factors: sovereignty, activation, quality, compounding, synergy
- Reward breakdown for last mission
- Gini coefficient (network-wide, with ≤ 0.35 threshold)

**Data sources:**
- `useWallet` hook (live API + offline fallback)
- `/v1/wallet/balance`, `/v1/wallet/supply`, `/v1/wallet/potential`

**States:**
- `LIVE_BACKEND_DATA` — green indicator
- `OFFLINE_SIMULATION` — yellow indicator with "estimates only" note

---

## View 5: MEMORY (Living Context)

**Purpose:** Show the user what the system knows about them.

**Content:**
- Semantic profile: preferred domains, active hours, vocabulary signature
- Active projects (with last activity date)
- Last 10 missions (with outcomes, Ihsān scores)
- Work streak (consecutive days with missions)
- Preferred agent (most-used PAT agent)
- Strongest skill domain
- Weakest skill domain (growth opportunity)
- Recommended next move

**Data sources:**
- Living Memory — semantic, episodic, procedural stores
- `/v1/memory/profile` (if backend available)

**Privacy note:** All data is local. Nothing leaves the device.

---

## View 6: SKILLS (Skill Tree + Reflexes)

**Purpose:** Progression visualization. The MMORPG character sheet.

**Content:**
- Current tier (Novice → Adept → Expert → Master)
- Current stage (Seed → Node → Apprentice → Builder → Verifier → Mentor → Catalyst)
- Tier progress bar (actions completed / required for next tier)
- Unlocked skills (by category: local, external, meta)
- Locked skills (with requirements shown)
- Compiled reflexes (with avg Ihsān, execution count, avg latency)
- Near-compilation candidates (with count toward threshold: N/3)
- Diffusion eligibility (can this reflex be published?)

**Data sources:**
- Skill Tier calculator
- Reflex cache
- Living Memory — procedural store

---

## View 7: SYSTEM (Health + Security)

**Purpose:** Node health, security posture, service status.

**Content:**
- Local services status (healthy/unhealthy for each container)
- Uptime
- Constitutional metrics: Ihsān, SNR, Gini (with threshold indicators)
- Myelination ratio (S1 hit rate)
- Evidence chain height + integrity check
- Security posture: headers, TLS, secrets, DAST
- WebSocket connection status
- Sync status (last backend sync time)
- Current model/provider routing
- Runtime version

**Data sources:**
- `/v1/health`
- Docker/K3d status
- Local state files

**States displayed:**
- `SAFE` — all checks pass
- `DEGRADED` — some services unhealthy
- `OFFLINE` — no backend connection
- `CONSTITUTIONAL_VIOLATION` — invariant breached (red alert)

---

## Navigation

| Key | View | Command |
|-----|------|---------|
| `1` or `briefing` | Briefing | Morning briefing |
| `2` or `mission <task>` | Mission | Execute a mission |
| `3` or `receipts` | Receipts | Proof chain browser |
| `4` or `wallet` | Wallet | Token balances |
| `5` or `memory` | Memory | Living context |
| `6` or `skills` | Skills | Skill tree + reflexes |
| `7` or `system` | System | Health + security |
| `help` | — | Show all commands |
| `exit` | — | Exit terminal |

---

## Output Modes

| Mode | Flag | Use Case |
|------|------|----------|
| Rich | (default) | Human-friendly with colors, tables, panels |
| Plain | `--plain` | No colors, no Unicode (for piping) |
| JSON | `--json` | Machine-readable (for scripting, CI) |
| Compact | `--compact` | One-line summaries (for dashboards) |

---

## Startup Sequence

1. Display BIZRA banner (0.1s)
2. Read node identity from `sovereign_state/` (0.01s)
3. Attempt backend health check (0.5s timeout)
4. Load Living Memory semantic profile (0.1s)
5. Generate morning briefing (0.2s)
6. Display briefing
7. Show prompt: `bizra>`
8. Ready for commands

Total cold start: < 1 second.

---

## Constitutional Visibility

Every view must surface constitutional compliance:

- Ihsān score: green ≥ 0.95, yellow ≥ 0.85, red < 0.85
- SNR: green ≥ 0.85, red < 0.85
- Gini: green ≤ 0.35, red > 0.35
- Myelination: progress bar (higher = more System-1)
- SEED mint status: "minting" if Ihsān ≥ 0.95, "paused" if below
- Pool contribution: always shown alongside SEED (50% split visible)

The user must never be confused about whether the system is healthy,
earning, or constitutionally compliant.