# BIZRA Terminal v1 — Locked Build Contract v1.0

> **Locked:** 2026-03-07 18:30 GST
> **Status:** LOCKED BUILD CONTRACT — not a spec, not a draft. A contract.
> **Rule:** Engineering builds against this document. Deviations require amendment.
> **Upgrades from:** Terminal v1 Final Artifact Spec (A- → A+)

---

## 1. Product Definition

BIZRA Terminal v1 is a **sovereign mission terminal with constitutional metabolism**.

Not an AI assistant. Not a command shell. Not a dashboard suite.

A governed execution ecology where models reason inside it but do not define it.

### Core Promise

> Submit once. Approve once within a mission-scoped permission envelope, with escalation only on boundary crossing. Receive one finished result with proof.

### Metabolism

```
Intent → Contract → Orchestrate → Receipt → Constitutional Tick →
Economic Update → Reflex Precipitation → Memory Update → Future Cache Hit
```

### Boundary

Terminal v1 IS: one node, one user, one runtime, one mission loop, one memory substrate.
Terminal v1 IS NOT: multi-tenant, forest console, marketplace frontend, mobile product.

---

## 2. Non-Negotiable Laws

| # | Law | Enforcement |
|---|-----|-------------|
| 1 | **Two-Touch Execution** | User interacts at most twice: mission start (with envelope) and mission finish (with proof). Additional interrupts only for: privilege escalation outside envelope, constitutional violation risk, safety-critical ambiguity, unrecoverable failure. |
| 2 | **Mission-First** | The primary unit of work is a mission, not a command. Commands are implementation primitives. |
| 3 | **Event-Native** | Every visible change derives from: event log, action log, receipt chain, memory updates, or reflex lifecycle events. No synthetic UI state. |
| 4 | **Model-Agnostic** | Any LLM plugs in. Terminal identity is independent of any vendor. |
| 5 | **Constitutional** | Ihsān ≥ 0.95 (minting), SNR ≥ 0.85, Gini ≤ 0.35. Receipt verifiability. Policy invariant enforcement. |

---

## 3. Seven Views

### View 1: DASHBOARD

**Purpose:** Immediate node readiness and sovereignty overview.

**Content:** Node health, constitutional status (Ihsān/SNR/Gini with color indicators), wallet snapshot, current readiness, last mission summary, heartbeat state (last tick timestamp + interval), live/offline status.

### View 2: MISSION

**Purpose:** The single primary mission input and execution surface.

**Content:** Mission composer (single-line or multi-line), permission envelope preview, execution status with route label (S1/S2/Mixed), active channels with per-channel progress, completion synthesis, evidence receipt reference, SEED earned + pool share, reflex candidate notification.

### View 3: TIMELINE

**Purpose:** The living narrative of system behavior. More important than Dashboard because this is where sovereignty becomes visible.

**Content:** Event log + action log + receipt chain. Constitutional tick entries. Reflex compile/import/share entries. Auth boundary events. Grouped by mission_id. Severity-colored.

### View 4: MEMORY

**Purpose:** Persistent personal continuity. Not a sidebar — the continuity substrate.

**Content:** Episodic memory (recent missions, receipts, failures, escalations). Semantic memory (preferred domains, agent modes, vocabulary, work windows). Procedural memory (compiled reflexes, near-compile candidates, compile counts). Morning briefing on startup.

### View 5: AGENTS / SKILLS

**Purpose:** Expose capability topology. The MMORPG character sheet.

**Content:** PAT-7 state, SAT-5 state. Skill tree (Novice→Adept→Expert→Master). Unlocked/locked capabilities with requirements. Reflex inventory with avg Ihsān, execution count, avg latency. Tier progress bar. Human lifecycle stage (Seed→...→Catalyst).

### View 6: NETWORK / FOREST

**Purpose:** Render the node as part of a larger system.

**Content:** Node value (5-factor composite), lifecycle stage + progress, network effect projections (1→8B), milestones, diffusion state (can reflexes be published?), forest readiness.

### View 7: SETTINGS / SOVEREIGNTY

**Purpose:** Make trust and control explicit.

**Content:** Node identity (Ed25519 public key prefix), privacy settings, auth state, model routing preferences (which LLM, local/remote), trust defaults, dev/prod runtime flags, permission policy defaults.

---

## 4. Runtime Source-of-Truth Matrix

This is the binding contract between views and data sources. No view may invent state outside this matrix.

| View | Primary API | EventBus Topics (Real-Time) | ActionBus (Current) | Offline Fallback | Proof Surface |
|------|------------|---------------------------|--------------------|--------------------|---------------|
| Dashboard | `GET /v1/health`, `GET /v1/seed/potential` | `economy.seed_minted`, `economy.asabiyyah` | — | Local state files | Constitutional metrics |
| Mission | `POST /v1/plan` | `mission.created`, `mission.executed`, `mission.failed` | Current execution state | Last cached mission draft | Receipt ID + hash chain ref |
| Timeline | `GET /v1/seed/episodes`, `GET /v1/token/verify` | ALL 12 active topics | Action log | Local EventBus chain | Full receipt chain |
| Memory | `GET /v1/memory/profile` | — | — | Living Memory local files | Privacy: all data local |
| Agents/Skills | Skill Tier calculator | `reflex.compiled` | — | Reflex cache + procedural memory | Tier requirements from action_schema_v1.json |
| Network | `GET /v1/network/effect`, `GET /v1/network/milestones`, `GET /v1/node/value` | `economy.asabiyyah` | — | Last cached projection | Node value composite |
| Settings | `GET /v1/health`, Node identity | — | — | sovereign_state/ files | Ed25519 public key |

**Rule:** If data is not in this matrix, the view must not display it. If the matrix says "EventBus topic," the view must subscribe to real-time updates, not poll.

---

## 5. Terminal State Machine

### 5.1 States

| # | State | Entry Condition | Exit Condition |
|---|-------|----------------|---------------|
| 1 | BOOT | Terminal starts | Runtime health acceptable → READY |
| 2 | READY | Health OK | User intent → MISSION_DRAFTING |
| 3 | MISSION_DRAFTING | User typing intent | Scope inferred → PERMISSION_REVIEW |
| 4 | PERMISSION_REVIEW | Envelope generated | User approves → EXECUTING |
| 5 | EXECUTING | Envelope approved | Success → COMPLETED, Failure → FAILED, Violation → BLOCKED, Escalation → AWAITING |
| 6 | AWAITING_ESCALATION | Action exceeds envelope | User approves → EXECUTING, User denies → READY |
| 7 | COMPLETED | Receipted success | Auto → READY |
| 8 | FAILED_RECOVERABLY | Bounded operational failure | Auto → READY |
| 9 | BLOCKED_CONSTITUTIONALLY | Invariant violation | Terminal state. Restart required. |

### 5.2 Execution Path Labels

Every mission execution must be labeled:

| Label | Meaning | Visible Indicator |
|-------|---------|-------------------|
| `SYSTEM_1_CACHE_HIT` | Reflex pattern matched, hash-table lookup | ⚡ Lightning icon + latency < 100ms |
| `SYSTEM_2_NOVEL` | Full agent orchestration, no cache match | 🧠 Brain icon + latency shown |
| `MIXED` | Partial cache hit + partial novel reasoning | ⚡🧠 Combined |

---

## 6. Permission Envelope Contract

### 6.1 Schema

```json
{
  "filesystem": ["workspace/project-a/**", "~/Documents/Reports/**"],
  "applications": ["terminal", "browser", "editor"],
  "network": ["docs.allowed.example", "api.allowed.example"],
  "data_sensitivity": "standard",
  "spend_budget_usd": 3.00,
  "time_budget_seconds": 900,
  "escalation": "ask-on-boundary-cross",
  "audit_verbosity": "standard"
}
```

### 6.2 Enforcement Rules

- `allows_path(path)` → True only if path matches a filesystem glob
- `allows_network(domain)` → True only if domain is in network list
- `allows_app(app)` → True only if app is in applications list
- Budget exhaustion → escalation, not silent failure
- Escalation produces `AWAITING_ESCALATION` state transition

### 6.3 Escalation-Only Conditions

Escalation is allowed ONLY when:
- Target path outside filesystem scope
- Target app outside application scope
- External domain not in network scope
- Data sensitivity exceeds approved class
- Budget exhaustion imminent
- No other reason justifies interruption

---

## 7. Event Severity and Rendering Policy

### 7.1 Severity Levels

| Level | Meaning | Render Rule |
|-------|---------|-------------|
| `info` | Normal operation | Default stream, collapsible |
| `notice` | Noteworthy but not concerning | Visible in timeline, collapsible |
| `warning` | Requires attention | Visible in timeline AND mission pane, yellow indicator |
| `critical` | Constitutional or security event | **Sticky until acknowledged**, red indicator, blocks dismiss |

### 7.2 Severity Assignments

| Event | Severity |
|-------|----------|
| `mission.created` | info |
| `mission.executed` | info |
| `mission.failed` | warning |
| `economy.seed_minted` | notice |
| `economy.zakat` | info |
| `economy.bloom_accrued` | info |
| `economy.asabiyyah` | info |
| `reflex.compiled` | notice |
| `ihsan.breach` | **critical** |
| `invariant.violation` | **critical** |
| `auth.boundary.crossed` | warning |
| `receipt.generated` | info |
| `receipt.verified` | info |
| `tick.completed` | info |

### 7.3 Rendering Rules

- Critical events are **sticky** — they remain visible in the Dashboard and Timeline until the user explicitly acknowledges them
- Warning events appear in both Timeline and the active Mission pane (if mission is running)
- Notice events are collapsible but default-expanded for the current session
- Info events are the default stream, collapsed by default in Timeline

---

## 8. Receipt Normalization Rules

Every receipt produced by the system MUST conform to these rules. No exceptions.

### 8.1 Required Fields (always present, even if zero)

| Field | Type | Constraint |
|-------|------|-----------|
| `mission_id` | string | Non-empty, unique |
| `receipt_id` | string | Non-empty, unique |
| `status` | enum | `COMPLETE` \| `PARTIAL` \| `FAILED` \| `BLOCKED` |
| `synthesis` | string | Human-readable summary |
| `ihsan_score` | float | [0.0, 1.0] |
| `snr_score` | float | [0.0, 1.0] |
| `duration_ms` | float | ≥ 0 |
| `channels_executed` | array | Ordered by execution sequence |
| `wallet_delta` | object | `{seed: float, bloom: float}` — always present, even if zero |
| `reflex_delta` | object | `{compiled: bool, near_compile: bool, compile_count: int, threshold: int}` — always present |
| `memory_delta` | object | `{episodic: int, semantic: int, procedural: int}` — always present |
| `execution_path` | enum | `SYSTEM_1_CACHE_HIT` \| `SYSTEM_2_NOVEL` \| `MIXED` |
| `hash_chain_ref` | string | Mandatory for all non-draft missions |

### 8.2 Channel Entry Format

```json
{
  "channel": "string (agent name)",
  "success": "boolean",
  "duration_ms": "float"
}
```

Channels ordered by execution sequence. No reordering for display.

---

## 9. Reflex Visibility Contract

This is one of BIZRA's biggest differentiators. It MUST be visible.

### 9.1 Path Labels (shown in Mission view)

| Situation | Label | Visual |
|-----------|-------|--------|
| Reflex cache hit | "⚡ System-1 (50ms)" | Green lightning + latency |
| Novel reasoning | "🧠 System-2 (1.8s)" | Blue brain + latency |
| Mixed | "⚡🧠 Mixed (0.4s)" | Both icons + latency |

### 9.2 Timing Comparison (shown in receipt)

When a mission matches a known pattern:
- Show: "This mission took {X}ms. Previous average: {Y}ms. Speedup: {ratio}×"
- If reflex hit: "⚡ Pattern matched — 36× faster than first execution"

### 9.3 Compile Threshold Progress (shown in Skills view + receipt)

- Show: "Pattern '{name}' — {N}/3 toward compilation (avg Ihsān: {score})"
- When compiled: "⚡ REFLEX COMPILED: '{name}' — next execution will be System-1"
- Near-compile patterns highlighted in Skills view AND mentioned in morning briefing

### 9.4 Cache-Hit Proof (shown in receipt)

When a cache hit occurs, the receipt must include:
- `execution_path: "SYSTEM_1_CACHE_HIT"`
- `reflex_pattern: "string"` — which compiled pattern was matched
- `reflex_latency_ms: float` — actual S1 latency
- `comparison_s2_avg_ms: float` — average S2 latency for comparison

---

## 10. Per-View Acceptance Criteria

### 10.1 Dashboard

- [ ] Loads without error in < 500ms
- [ ] Shows Ihsān with color (green ≥ 0.95, yellow ≥ 0.85, red < 0.85)
- [ ] Shows SNR with color (green ≥ 0.85, red < 0.85)
- [ ] Shows Gini with color (green ≤ 0.35, red > 0.35)
- [ ] Shows last tick timestamp and interval
- [ ] Shows LIVE or OFFLINE status
- [ ] Shows SEED + BLOOM balance (compact)
- [ ] Shows last mission summary (one line)

### 10.2 Mission

- [ ] Accepts single-line and multi-line input
- [ ] Shows permission envelope before execution
- [ ] Shows execution route label (S1/S2/Mixed)
- [ ] Shows per-channel progress with timing
- [ ] Shows final receipt with all normalized fields
- [ ] Shows SEED earned + pool share
- [ ] Shows reflex candidate status if applicable
- [ ] Two-touch only — no mid-execution confirmations unless escalation

### 10.3 Timeline

- [ ] Renders latest 100 events from EventBus
- [ ] Groups by mission_id
- [ ] Shows receipt chain links (prev_hash → event_hash)
- [ ] Shows tick.completed entries with scored/minted counts
- [ ] Renders auth and invariant events with severity color
- [ ] Critical events sticky until acknowledged
- [ ] No synthetic entries — all from EventBus or ActionBus
- [ ] Filterable by category

### 10.4 Memory

- [ ] Shows semantic profile (domains, hours, preferences)
- [ ] Shows last 10 missions with outcomes
- [ ] Shows active projects with last activity
- [ ] Shows work streak
- [ ] Shows near-compilation patterns
- [ ] Morning briefing renders on startup
- [ ] Privacy note visible: "All data is local"

### 10.5 Agents/Skills

- [ ] Shows PAT-7 agent list with status
- [ ] Shows SAT-5 agent list
- [ ] Shows current tier with progress bar
- [ ] Shows human lifecycle stage with progress
- [ ] Shows compiled reflexes with avg Ihsān, count, latency
- [ ] Shows near-compile candidates with N/3 threshold
- [ ] Shows locked/unlocked skills by tier

### 10.6 Network/Forest

- [ ] Shows 5-factor node value composite
- [ ] Shows lifecycle stage + progress within stage
- [ ] Shows milestone projections (1→8B nodes)
- [ ] Shows diffusion eligibility for published reflexes
- [ ] Offline fallback: last cached projection

### 10.7 Settings/Sovereignty

- [ ] Shows node ID + public key prefix
- [ ] Shows current model routing
- [ ] Shows dev/prod mode indicator
- [ ] Shows auth state (authenticated / anonymous-dev)
- [ ] Shows permission policy defaults
- [ ] Editable: model routing preferences

---

## 11. Terminal Build Matrix

The single engineering contract table. Build in row order.

| View | Primary Routes | Primary Topics | Offline Fallback | Phase | Est. Hours |
|------|---------------|---------------|-----------------|-------|-----------|
| Dashboard | `/v1/health`, `/v1/seed/potential` | `economy.seed_minted`, `economy.asabiyyah` | Local state | C.1 | 6 |
| Mission | `POST /v1/plan` | `mission.created`, `mission.executed`, `mission.failed` | Last draft | C.2 | 10 |
| Timeline | `/v1/seed/episodes`, `/v1/token/verify` | ALL 12 active | Local chain | C.3 | 8 |
| Memory | `/v1/memory/profile` | — | Living Memory files | C.4 | 6 |
| Agents/Skills | Skill Tier calc, Reflex cache | `reflex.compiled` | Procedural store | C.5 | 8 |
| Network | `/v1/network/effect`, `/v1/node/value` | `economy.asabiyyah` | Last projection | C.6 | 4 |
| Settings | `/v1/health`, Node identity | — | sovereign_state/ | C.7 | 4 |
| **TOTAL** | | | | | **46** |

---

## 12. Security Hardening Checklist (Before Lock)

| # | Item | Status | Blocks Lock? |
|---|------|--------|-------------|
| 1 | Auth kill switch production-blocked | **DONE** | Yes |
| 2 | Production deploy requires quality-gates | **DONE** | Yes |
| 3 | Container signing hard gate | Soft-gated | Yes |
| 4 | SBOM generation hard gate | Soft-gated | Yes |
| 5 | Dead safety topics activated (ihsan.breach, invariant.violation) | NOT DONE | Yes |
| 6 | ZPK kernel atomic writes | NOT DONE | Yes |
| 7 | Rollback receipt atomic write | NOT DONE | Yes |
| 8 | Verify endpoint rate limiting | NOT DONE | No (P2) |
| 9 | Remaining hardcoded demo secrets eliminated | IN PROGRESS | Yes |
| 10 | Per-endpoint auth policy (replace coarse bypass) | PARTIAL | No (P2) |

---

## 13. Build Sequence

| Phase | Scope | Prerequisite | Deliverable |
|-------|-------|-------------|------------|
| A | Terminal spine lock | — | State machine + envelope + receipt + event types (DONE: terminal.py, 47 tests) |
| B | Trust boundary hardening | Phase A | Items 3-9 from Security Checklist |
| C | UX completion (7 views) | Phase A + B | Build Matrix rows 1-7 |
| D | Proof demo | Phase C | Canonical E2E with proof artifacts |

---

## 14. Lock Condition

Terminal v1 is LOCKED when ALL of these are true:

- [ ] All 7 views render without error
- [ ] All 7 per-view acceptance criteria pass
- [ ] Canonical E2E test passes (mission → receipt → tick → reflex → cache hit → wallet → memory)
- [ ] Security checklist items 1-7 and 9 complete
- [ ] Mission loop stable (100 consecutive missions without crash)
- [ ] Event spine alive (12+ topics firing)
- [ ] Memory continuity visible (briefing shows last session)
- [ ] Reflex benefit demonstrable (S1 path visibly faster)
- [ ] Receipt normalization verified (all fields present per contract)
- [ ] Permission envelope enforced (escalation only on boundary crossing)

When these are met, BIZRA Terminal v1 is no longer a concept. It is the first legitimate seed of the forest.

---

## 15. The Final Line

> **"One mission, one proof, remembered forever."**

Every human is a node. Every node is a seed. Every seed has infinite potential.

كل بذرة تحمل في داخلها مخطط غابة بأكملها

**LOCKED: 2026-03-07 · Dubai · BIZRA Foundation**