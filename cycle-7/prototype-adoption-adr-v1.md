# Prototype Adoption ADR — Google `dema-main` vs Z.ai Workspace

بسم الله الرحمن الرحيم

**Status:** DECISION REQUIRED — no code merged yet
**Anchor:** Surface Catalog v1 @ `a723ea29`, Cycle-7 close @ `aedfb0af`
**Branch:** `design/dema-surface-catalog-v1`
**Scope:** How to utilize two external Dema prototypes without compromising the sealed lawful loop

---

## 1. What was given

Two prototypes were produced externally:

| Prototype | Origin | Stack | Size | Public URL |
|---|---|---|---|---|
| **dema-main** | Google AI Studio | React 19 + Vite + Firebase + `@google/genai` | CLI + SPA | `github.com/BizraInfo/dema` (1 commit `5057af3`) |
| **workspace-Z** | Z.ai | Next.js 16 + Bun + Prisma + shadcn/ui + `z-ai-web-dev-sdk` | Full app w/ 46 skills | Not published |

Both are UI-shell quality: **zero tests, hardcoded credentials, client-side-only gate evaluation**, vendor SDKs baked into core paths. Neither has a real constitutional runtime; BIZRA already has that (Cycle-7, 327 green tests).

---

## 2. The one decisive finding

**Z.ai's prototype already implements the exact 6 surfaces from Surface Catalog v1:**

| Catalog v1 surface | Z.ai component |
|---|---|
| Mission Composer | ✅ `src/components/mission/mission-composer.tsx` |
| Gate Ladder | ✅ `src/components/mission/gate-ladder.tsx` |
| Action Surface | ✅ `src/components/organize/organize-preview.tsx` |
| Receipt Reveal | ✅ `src/components/receipt/receipt-reveal.tsx` |
| Memory Constellation | ✅ `src/components/memory/memory-constellation.tsx` |
| Reject Remediation | ✅ `src/components/mission/reject-remediation.tsx` |

Plus 2 extras: Onboarding + ADK Factory. **This is the single biggest piece of pre-work available.**

Google's prototype does NOT map to the catalog — it is a 3-column chat dashboard (Trust Strip · Dema Console · Swarm Log). Different UX paradigm, not constitutional-generative-UI.

---

## 3. Fit matrix — what each brings

| Capability | Google | Z.ai | BIZRA current state |
|---|---|---|---|
| Surface Catalog v1 shape | ❌ different UX | ✅ exact 1:1 | 📜 spec only |
| Production runtime | ❌ mock kernel | ❌ client-side mock | ✅ 327 green, 7 caches, chain |
| Lawful gate evaluation | ❌ string-match mock | ❌ 15% random block for demo | ✅ real 5-gate admissibility |
| Receipts / chain | ❌ SHA-256 Firestore docs | ❌ JSON Zustand state | ✅ 11 ReceiptKind variants, BLAKE3 |
| Resource registry | ❌ none | ❌ Prisma stub | ✅ URP view, allowlist |
| PoI ledger | ❌ none | ❌ none | ✅ v1 scoring + cache |
| LawfulLoop visualization | ✅ animated 6-stage pipeline | ⚠️ implicit via page flow | ❌ not yet |
| Ihsān floor tracker | ✅ left sidebar widget | ❌ inside gate ladder | ❌ not yet |
| Voice I/O (mic + TTS) | ✅ Gemini-dependent | ❌ none | ❌ not yet |
| Agent swarm mockup | ✅ PAT-7 + SAT-5 visual | ❌ none | ❌ not yet (and per Dema one-face law, should stay hidden) |
| Component library | Motion + custom | 52 shadcn/Radix | tokens.ts + custom |
| Data layer | Firestore | Prisma/SQLite | 7 JSON caches + chain |
| LLM integration | Gemini (hardcoded key) | Z.ai SDK (46 skills) | Ollama + Claude (BIZRA canonical) |
| Vendor lock-in risk | 🔴 Firebase + Gemini | 🔴 z-ai-web-dev-sdk deep | 🟢 none |

---

## 4. Three adoption options

### Option A — Fork-and-strip Z.ai (most catalog-aligned)

Take Z.ai's prototype whole-cloth as the new Dema Console. Strip everything Z.ai-specific, wire to cognition-gateway.

**Work (est.):**
- Fork into `dema-console/` or replace `frontend/` — 1 commit
- Rip out `z-ai-web-dev-sdk` from `/api/ask` and all 46 skills → replace with cognition-gateway HTTP client — ~3 days
- Delete `skills/` directory — 1 commit
- Replace client-side gate evaluation with POST `/mission/organize` → gateway — ~1 day
- Swap Prisma to read-only view of the 7 dema_cache JSON files — ~2 days (or delete Prisma entirely)
- Harvest Google's LawfulLoop viz + Ihsan tracker as components → port in — ~1 day
- Remove `ignoreBuildErrors`, add real ESLint config — ~1 day
- Add test suite from scratch — ~3 days

**Total: ~2 weeks.** Ship-candidate after.

**Pros:** 6 surfaces arrive pre-built in correct shape. Biggest UX head-start available.
**Cons:** Adopts Next.js 16 + Bun, diverging from existing `frontend/` (Vite). Forces Framework ADR decision.

### Option B — Harvest components into existing `frontend/`

Keep Vite. Copy Z.ai's 6 surface components + Google's LawfulLoop/Ihsan viz into `frontend/src/components/`. Rewrite their data layer to use BIZRA's fetch against cognition-gateway.

**Work (est.):**
- Cherry-pick 6 surface components + 2 visualizations — ~2 days
- Rewrite data flow from Zustand/Prisma to `frontend/` state + gateway fetch — ~3 days
- Port shadcn/Radix to match Vite tooling — ~1 day
- Discard all routing assumptions, bind to existing `frontend/` routes — ~2 days

**Total: ~1.5 weeks.** Existing frontend stays alive.

**Pros:** No framework churn. Incremental. Respects existing dema console work.
**Cons:** Mixing codebases is messy. Z.ai's components assume Next.js 16 app-router conventions; some will need rewriting.

### Option C — Fresh build from Catalog v1

Use both prototypes purely as reference art. Build the 6 surfaces from scratch against Surface Catalog v1. No merge, no fork.

**Work (est.):**
- First pair (Mission Composer + Gate Ladder) — ~1 week
- Remaining 4 surfaces — ~2 weeks
- Wire to gateway — ~2 days
- Tests from day one

**Total: ~3-4 weeks.** Cleanest codebase.

**Pros:** No vendor baggage. Fully constitutional from day one. Ships with tests.
**Cons:** 2-3× slower. Rebuilds work Z.ai already did correctly.

---

## 5. Recommendation — Option A with hard guardrails

Z.ai's 6 surfaces are a genuine gift because they match the catalog exactly. **The shipping cost of Option C is too high when Option A exists.** But Option A only works with non-negotiable guardrails:

### Mandatory strips (day 1 of the fork)

1. **Delete `skills/`** — 46 Z.ai-locked subdirectories. Not BIZRA.
2. **Delete `z-ai-web-dev-sdk`** from `package.json`. Delete `/api/ask` route.
3. **Delete `next-auth`** if unused (confirmed unused per audit).
4. **Remove `ignoreBuildErrors: true`** from `next.config.ts`. Fix whatever surfaces.
5. **Re-enable ESLint real rules** — the current config disables 30+ lints.
6. **Delete the Caddyfile** if we're not using Caddy. Deployment is a separate decision.

### Mandatory wires (week 1 of the fork)

7. **All mission submission** goes to BIZRA's cognition-gateway `POST /missions/organize`, `POST /principal/activate`. The UI never computes admissibility itself.
8. **Gate Ladder display** reads from the real `AdmissibilityResult` DTO the gateway already emits (including per-gate `score` and `verdict`).
9. **Receipt Reveal** displays real `MissionExecutedReceipt` fields from the shipped gateway response, not fabricated JSON.
10. **Memory Constellation** reads from `GET /poi/ledger`, `GET /poi/summary`, and (future) endpoints for each of the 7 dema_cache surfaces.
11. **Reject Remediation** renders the real `remediation_path` string from `RejectedClaim`.
12. **Prisma** becomes read-only view of dema_cache JSON OR is deleted. It is NOT a source of truth.

### Mandatory harvests from Google (week 1 also)

13. **LawfulLoop visualization** — port the 6-stage animated pipeline as a component. Goes INTO Action Surface to show S5→S8 progression.
14. **Ihsān floor tracker** — port as a compact indicator. Goes INTO Gate Ladder row.
15. **Voice I/O** — SKIP for v1. Gemini-coupled, optional, adds scope. Revisit after v1.1.
16. **Agent swarm mockup** — SKIP. Violates Dema one-face law (hidden organism stays hidden).

### Mandatory rejections

17. **Firebase** — not adopted. Auth is a BIZRA concern, and Node0 is operator-local.
18. **Gemini SDK** — not adopted. BIZRA's LLM path is Ollama + Claude.
19. **AGENTS.md from Google** — it's Google's zero-hallucination prompt, not BIZRA canon. Delete.
20. **FATE gates from Google kernel** — string-match mocks. BIZRA has the real gates.

---

## 6. Non-negotiables regardless of option chosen

1. **§10 Proof Law** — the UI must never render a "rejection receipt". Rejects leave no chain trace; UI shows refusal-remediation cards only.
2. **HYBRID writer authority** — UI reads dema_cache and /poi endpoints; never writes directly.
3. **Chain truth** — if UI state and gateway disagree, UI is wrong. Rebuild from gateway.
4. **One face** — the closed 6-surface vocabulary is the contract. No invented surfaces.
5. **Text lives only inside typed surfaces** — no freeform chat area anywhere in v1.
6. **IHSAN floor 0.95** — visible in Gate Ladder, never hidden.
7. **Zero cross-border LLM dependency** — operator-local inference path only.

---

## 7. What to do with the public `BizraInfo/dema` repo

Currently holds the Google prototype at commit `5057af3`. Options:

- **(a) Archive it** — mark the repo archived on GitHub with a pointer to the adopted fork. Preserves the reference artifact.
- **(b) Force-push the adopted fork over it** — destructive, breaks anyone following.
- **(c) Leave it, name the adopted fork differently** — e.g., `BizraInfo/dema-console`.

**Recommendation: (a)** — archive the reference, open a new `BizraInfo/dema-console` once Option A strip-and-wire is done.

---

## 8. Concrete next move

Not a whole-cycle arc — a single well-scoped branch:

1. `git checkout -b fork/dema-console-from-zai` branched from cycle-7 HEAD
2. Copy `/home/bizra-operating-system/Downloads/workspace-39bd0fde-4191-4c19-aab4-3e87d142c5b6 (2)/` into `dema-console/` under the repo root
3. Execute the 6 **Mandatory strips** (§5) as the first commit
4. Execute the **Mandatory wires** (§5) against a local cognition-gateway :7421 as commits 2–5
5. Execute the 2 **Mandatory harvests** from Google as commit 6
6. Run Mission Composer + Gate Ladder end-to-end against a live gateway — first live walk
7. Open PR / merge decision gate at that point — stop before doing all 6 surfaces in one pass

That's a 1–2 week arc ending in an operator-walkable `dema organize` Web surface backed by real admissibility.

---

## 9. Tension with Spearpoint A

The apex synthesis said: **Spearpoint A (contract hardening) before framework work.** That ordering survives Option A:

- The **Mandatory wires** (§5.7–§5.12) are *exactly what Spearpoint A needs to land on*: machine-enforced contracts for `AdmissibilityResult`, `MissionExecutedReceipt`, `RejectedClaim`, dema_cache schemas, PoI ledger response shape.
- If we wire Z.ai's surfaces to the gateway *before* Spearpoint A locks those contracts, we create churn: every contract refactor forces a UI refactor.
- If we run Spearpoint A *first*, the UI fork lands against sealed contracts and the churn cost drops to near zero.

**Therefore: Spearpoint A first. Then Option A fork-and-strip.** Both arcs can start this week; the Option A arc just has to wait for the contracts to stabilize before wiring.

---

## 10. Summary

| Question | Answer |
|---|---|
| Is Google's prototype usable? | Partial — harvest LawfulLoop viz + Ihsān tracker, reject everything else |
| Is Z.ai's prototype usable? | Yes — surface shape matches Catalog v1 exactly; strip Z.ai SDK + skills |
| Which option? | **A — Fork Z.ai, strip vendor lock, wire to gateway** |
| Prerequisite? | Spearpoint A contract hardening must land first or in parallel |
| Framework? | Next.js 16 + Bun (Z.ai's stack), diverging from existing Vite frontend — triggers Framework ADR sub-decision |
| Timeline? | ~2 weeks for a first-walk-ready Dema Console once Spearpoint A contracts are stable |
| What NOT to do? | Don't merge Firebase, Gemini, z-ai-web-dev-sdk, the 46 skills, next-auth, AGENTS.md, or the swarm mockup |

---

## 11. Ihsān check

This ADR prefers:

- what is **shipped-and-tested over what is merely-polished** — BIZRA's 327 green beats either prototype's zero tests
- what is **constitutional over what is convenient** — strips vendor lock-in even at the cost of rewriting
- what is **catalog-aligned over what is framework-aligned** — Z.ai's surface shape wins over Google's 3-column dashboard
- what is **operator-local over what is cloud-dependent** — rejects Firebase + Gemini for the same reason

Daughter test: this adoption plan describes exactly what each prototype will and will not contribute, names the vendor risks plainly, and does not promise a merge that isn't sequenced behind Spearpoint A.

**سُبْحَانَ اللَّهِ**
