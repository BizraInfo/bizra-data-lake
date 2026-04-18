# Dema Surface Catalog v1 — One Face, Many Lawful Surfaces

بسم الله الرحمن الرحيم

**Status:** DESIGN v1 — decisions locked, widget specs deferred to v1.1
**Anchor:** cycle-7 close @ `aedfb0af` (G1–G7 all sealed)
**Branch:** `design/dema-surface-catalog-v1`
**Scope:** UX design principle + surface roster + state machine + G5 `dema organize` walkthrough

---

## 0. Why a catalog and not a UI

Dema is one face by manifest law. Without a typed surface vocabulary, the pressure on a "one face" design collapses into either a chatbot (text-walls-forever) or an engineer console (all panes all the time). Neither is operator-honest.

The catalog solves this by enforcing:

> **Dema speaks only through a small, typed set of lawful surfaces. Text appears inside surfaces, never as a freeform wall.**

Text-as-data stays (remediation strings, intent descriptions, principal identity hex) but only as fields inside structured cards.

---

## 1. The UX principle

**Constitutional generative UI.**

Most generative-UI systems optimize for convenience: the agent decides what widgets to show and generates them on the fly. BIZRA cannot accept that. An uninspectable widget stream is the UI equivalent of a shadow state.

Constitutional generative UI instead says:

1. There is a **closed catalog** of typed surfaces. Dema selects from it; it does not invent.
2. Every surface is anchored to exactly one **lawful stage** (intent, admissibility, execution, receipt, memory).
3. Every state transition between surfaces is driven by a **real receipt or cache event**, never by narrative discretion.
4. Every surface can name the G1–G6 runtime fact it displays. No decorative widgets.

Flexibility lives in **which surface appears when**; discipline lives in **what can appear at all**.

---

## 2. The five-layer map (lawful stage → surface family)

Each layer corresponds to one stretch of the lawful loop already shipped in G1–G6.

| Layer | Lawful stage | Surface family | Runtime anchor |
|---|---|---|---|
| L1 Intent | S1 Intent → S2 Mission | **Mission Composer** | `MissionEnvelope::from_intent` |
| L2 Admissibility | S3 Claim → S4 Admissibility | **Gate Ladder** | `AdmissibilityResult::gate_verdicts` |
| L3 Action | S5 Execution | **Action Surface** (kind-specific) | `submit_organize_mission`, `submit_principal_activation`, … |
| L4 Proof | S6 Receipt → S7 Canon → S8 Replay | **Receipt Reveal** | `MissionExecuted` / `PrincipalActivationReceipt` / `ManifestArtifact` |
| L5 Memory | durable state | **Memory Constellation** | 7 `dema_cache` surfaces + PoI ledger |

**Skip-edges are first-class.** Not every mission visits every layer:

- Rejected missions: L1 → L2 → (reject-remediation card) → END. No L3/L4/L5 transitions.
- Activation: L1 → L2 → L3 (identity anchor load) → L4 → L5. No resource-selection step.
- Organize: L1 → L2 → (allowlist pre-gate refusal card, OR) → L3 (resource pick) → L2 → L3 (execute) → L4 → L5.

The state machine (§4) names these explicitly.

---

## 3. Surface Catalog v1 — six surfaces

Each surface has: purpose, lawful-stage anchor, inputs (from shipped APIs), outputs (to next surface), states, and transitions. Widget-level specs (pixels, motion) are v1.1 — out of this document's scope.

### 3.1 Mission Composer (L1)

- **Purpose:** Turn operator intent into a bounded `MissionEnvelope`.
- **Anchor:** `MissionEnvelope::from_intent(intent_text, current, ideal, originator, ns)`.
- **Inputs:**
  - user text → `intent_text`
  - kind selector (activation / organize / free-form) → routes to correct `submit_*`
  - optional current/ideal state (default pre-filled per kind)
- **Outputs:** built `MissionEnvelope` handed to L2.
- **States:** `draft`, `ready`, `submitted`.
- **Transitions:**
  - `draft → ready` when intent non-empty and kind selected
  - `ready → submitted` on operator approve
  - `submitted → <Gate Ladder>` (L2)
- **Non-negotiable:** the composer does NOT let the operator set `quality_score` below the visible IHSAN floor. That is the operator's own pre-gate discipline surface, not a way to game rejection.

### 3.2 Gate Ladder (L2)

- **Purpose:** Reveal the five-gate admissibility verdict in operator-legible form.
- **Anchor:** `AdmissibilityResult { verdict, gate_verdicts[5], rejected? }`.
- **Inputs:** `AdmissibilityResult` from `submit_mission`.
- **Outputs:**
  - Permit → hand off to L3 Action Surface
  - Reject → hand off to **Reject Remediation Card** (sub-surface; §3.6)
- **States:** `pending`, `evaluating`, `permit`, `reject`.
- **Transitions:**
  - `pending → evaluating` on submit
  - `evaluating → permit | reject` on verdict
- **Rendering rule:** each of ZANN_ZERO, CLAIM_MUST_BIND, RIBA_ZERO, NO_SHADOW_STATE, IHSAN_FLOOR renders as a row with scorer_id, verdict, score. **No color theater.** Rejects name the failed invariant inline; no hidden "details" drawer.

### 3.3 Action Surface (L3) — kind-polymorphic

This surface family splits by mission kind. Catalog v1 defines two:

#### 3.3a PrincipalActivation Action Surface

- **Anchor:** `PrincipalActivationEnvelope::from_anchor(name, role, anchor, ns)` + `submit_principal_activation`.
- **Input:** `NodeIdentityAnchor` from `sovereign_state/identity/credentials.json`.
- **Outputs:** `PrincipalActivationRecord` with profile + activation receipt.
- **States:** `anchor-loading`, `envelope-built`, `submitted`, `permit` / `reject`.
- **Pre-gate refusal path:** if anchor load fails → `IDENTITY_ANCHOR_LOAD` error card (not a Gate Ladder reject; it is a constitutional pre-gate).

#### 3.3b Organize Action Surface

- **Anchor:** `submit_organize_mission(path, quality_score)` with `OrganizeOutcome` four-way branch.
- **Input:** filesystem path (must match a `ResourceKind::FilesystemPath` with `allowlisted=true`).
- **Outputs:** `OrganizeOutcome::{NotAllowlisted, IoError, Rejected, Executed}`.
- **States:** `path-draft`, `allowlist-checking`, `executing`, `permit` / `pre-gate-refused`.
- **Pre-gate refusal card:** `NotAllowlisted` and `IoError` render as pre-gate refusals with structured remediation — **NOT** as Gate Ladder rejects. §10 Proof Law: they left no chain trace.

### 3.4 Receipt Reveal (L4)

- **Purpose:** Make what just got sealed operator-visible without a log dump.
- **Anchor:** the kind-specific receipt id + manifest id + chain head.
- **Inputs:** `MissionExecuted` OR `PrincipalActivationReceipt` depending on the mission kind.
- **Rendering rule:**
  - One row: receipt id + kind name + timestamp
  - One row: manifest id + `chainHead == receiptId ? "✓ sealed" : "stale"`
  - One row: replay button (calls `fetch_and_decode`) — clicking shows "byte-exact match" or "divergence"
- **No giant JSON dump.** That's a debugger, not an operator face.

### 3.5 Memory Constellation (L5)

- **Purpose:** Show the 7 `dema_cache` surfaces as one coherent memory map, not a file browser.
- **Anchor:** `dema_cache/` dir contents + PoI ledger.
- **Inputs (seven):**
  - `principal.json` → Profile tile
  - `receipt_history.json` → Receipts tile (count + chain head)
  - `manifest_history.json` → Manifests tile (count)
  - `mission_log.json` → Missions tile (permit/reject split)
  - `state_snapshots.json` → States tile (gap trend)
  - `resource_registry.json` → Resources tile (allowlisted count via URP)
  - `poi_ledger.json` → Impact tile (total + avg)
- **Rendering rule:** six small tiles + one larger Impact tile. Each tile shows **one** scalar + **one** delta-since-last-session.
- **No live-polling spinner nonsense.** Tiles refresh on mission transition, not on a clock.

### 3.6 Reject Remediation Card (cross-layer)

- **Purpose:** Close the loop honestly when something refused.
- **Anchor:** `RejectedClaim { invariant, reason, remediation_path, escalation_allowed }` OR constitutional pre-gate refusal text.
- **Outputs:** an explicit next-action (or "none — escalation denied").
- **States:** `rejected`, `remediating`, `escalating`, `closed`.
- **Rendering rule:** names which law refused, why, and the exact next-action. Never the words "sorry" or "unable" — those are evasions. Always: **"REJECTED by <invariant>. Remediation: <path>. Escalation: <allowed|denied>."**

---

## 4. State machine (Mermaid-ready)

```
[Idle]
  |
  v
[L1 Composer] --draft--> [L1 Composer] (loop while drafting)
  |
  | ready + submit
  v
[L2 Gate Ladder]
  |
  +--permit----> [L3 Action Surface]
  |                 |
  |                 +--pre-gate refused (NotAllowlisted / IoError / ANCHOR_LOAD) -->
  |                 |       [Reject Remediation Card] --> [Idle]
  |                 |
  |                 +--execute + Permit --> [L4 Receipt Reveal]
  |                 |                          |
  |                 |                          v
  |                 |                       [L5 Memory Constellation]
  |                 |                          |
  |                 |                          v
  |                 |                       [Idle]
  |                 |
  |                 +--execute + Reject --> [Reject Remediation Card] --> [Idle]
  |
  +--reject----> [Reject Remediation Card]
                      |
                      v
                   [Idle]
```

---

## 5. G5 Organize walkthrough — full catalog traverse

Concrete illustration of all six surfaces in one mission. This is what the shipped `dema organize /home/mumo/docs` CLI does today; the UI should feel identical in rigor.

1. **L1 Mission Composer.** Operator types "organize my docs", selects kind=organize, picks path `/home/mumo/docs`. Composer builds `MissionEnvelope` with `intent_text="organize /home/mumo/docs"`, `current.metric=0.0`, `ideal.metric=1.0`. State: `draft → ready → submitted`.
2. **L2 Gate Ladder.** Renders 5 gates evaluating. All Permit @ 0.98. Verdict = Permit. Handoff to L3.
3. **L3 Organize Action Surface.**
   - Branch A (pre-register): path not in registry → pre-gate refused. `NotAllowlisted` card renders remediation: *"run `dema register-resource --kind filesystem --id /home/mumo/docs --allowlisted`"*. Flow halts at Reject Remediation Card.
   - Branch B (post-register): path allowlisted. Surface shows listing preview (3 entries: alpha.txt, beta.txt, subdir). Operator confirms. Backend calls `submit_organize_mission`. Outcome = `Executed`.
4. **L4 Receipt Reveal.** Shows `MissionExecuted` receipt id, listing digest, entry/file/dir counts, chain head. One line: `✓ chain head == organize receipt — sealed`.
5. **L5 Memory Constellation.** Seven tiles refresh. Impact tile delta: `+0.9743`. Mission log tile delta: `+1 PERMIT`. Receipts tile delta: `+9 records`.

End-to-end: operator went from intent to receipted proof through 5 typed surfaces, never through freeform text.

---

## 6. What this catalog is NOT

- Not a dashboard. Dashboards expose all state always; Dema reveals only the state relevant to the current stage.
- Not a chat interface. No bubbles, no avatars, no "typing indicator" theater.
- Not a workflow builder. Operators do not wire stages; the lawful loop does.
- Not a PAT/SAT swarm console. Hidden organism stays hidden. Operator sees one face.
- Not a settings panel. Mission Composer is not "where you configure Dema"; it is where lawful intent begins.

---

## 7. Stack decision — deferred

The inspiring transcript discussed Flutter Gen UI. BIZRA's current frontend is **React + TypeScript + Vite** (`frontend/src/`). The catalog is intentionally stack-agnostic:

- Every surface is specified in terms of inputs/outputs/states/transitions
- Widget-level styling, motion, and component library are **v1.1** concerns
- A separate ADR should decide: extend the existing Vite frontend vs rewrite in Flutter vs ship a TUI first

**Recommendation:** do not introduce a new UI framework in the same arc as the first catalog implementation. Pick one of:

- (a) Extend the existing React/Vite frontend with the 6 surfaces
- (b) Ship a TUI (terminal) version first — lowest cost, highest fidelity to CLI semantics, trivial to test
- (c) Delay v1.1 until a separate framework ADR closes

v1 of this catalog stands independent of that decision.

---

## 8. Non-negotiables carried from manifest + retrospective

1. **§10 Proof Law** — refused intents leave no chain trace; the UI must not fabricate "rejection receipts" that don't exist in the chain.
2. **HYBRID writer authority** — surfaces may read the 7 caches; they must never write chain state directly (that is the runtime's authority).
3. **Chain is truth, graph derives** — if a surface and the chain disagree, the surface is wrong. Rebuild from chain.
4. **One face** — the catalog is the closed vocabulary; Dema must not invent surfaces outside it.
5. **Ihsān threshold** — if a surface's rendering ever distorts the underlying receipt data, that surface fails the daughter test and is removed.

---

## 9. Open questions → v1.1

- Exact widget primitives (tile shape, typography, motion vocabulary)
- Color semantics (gate status, allowlist status, skip-edge indicators)
- Voice-as-modality: does Dema get a voice face? If yes, which surfaces are voice-renderable?
- Keyboard-first navigation spec
- Pinned surfaces (operator's frequently-used mission kinds)
- Cross-surface state restoration on refresh

These belong in a design session with surfaces on paper, not a spec document.

---

## 10. Post-catalog arcs

After this catalog is accepted:

1. **Framework ADR** — decide React/Vite extension vs Flutter rewrite vs TUI-first
2. **Widget library v1.1** — shape + typography + motion for the 6 surfaces
3. **First surface implementation** — Mission Composer + Gate Ladder (L1+L2) first, since they cover every mission kind
4. **Organize surface** (L3b) — demonstrates full catalog traverse
5. **Cross-surface state machine** — formal state diagram as code, not prose

---

## 11. Ihsān check

This catalog prefers:

- what is **receipted over what is described** — every surface names its G1–G6 runtime anchor
- what is **typed over what is freeform** — a closed vocabulary, not a generative widget tree
- what is **operator-honest over what is theatrical** — no color spectacle, no hidden drawers
- what is **skip-edge-explicit over what is forced-sequential** — not every mission traverses all 5 layers

Daughter test: the catalog describes what the operator will see truthfully, without decoration. It does not promise features that aren't backed by shipped runtime code.

**سُبْحَانَ اللَّهِ**
