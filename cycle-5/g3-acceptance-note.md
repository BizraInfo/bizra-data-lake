# Cycle-5 — Gate G3 (Principal Activation End-to-End) Acceptance Note

بسم الله الرحمن الرحيم

**Cycle:** 5 (Principal Activation)
**Gate:** G3 — First principal-activation receipt through the lawful loop
**Sealed:** 2026-04-17 (Friday) per system `date`
**Split:** G3a (gateway POST /mission) + G3b (Next.js proxy)
**Commits:** `b031fec8` (G3a, bizra-data-lake) + `40a6832` (G3b, award-winner-design)

---

## Context

Cycle-4 established the narrow-real chain bridge. G1 confirmed the Dema empty-state behaves honestly. G2 landed the mission-runtime on NODE0 with 3 constitutional patches (A reject gate, B stage walk to S8, C manifest module). G3 closes the §16 minimum undeniable loop: **a principal's intent enters through one authoritative entry, is bound into a MissionEnvelope, evaluated by one 5-gate admissibility chain, and — on PERMIT — sealed as one canonical receipt visible in Dema**.

## What landed

### G3a — `bizra-cognition-gateway` v0.2 (commit `b031fec8`)

Gateway state evolved from `Arc<RwLock<ReceiptChain>>` to `Arc<RwLock<CognitionRuntime>>`. New endpoint:

```
POST /mission
  Body:     { intent, currentState, idealState, originator?, qualityScore? }
  200 OK:   { missionId, admissibility: { verdict, gateVerdicts[], rejected? },
              receiptId, finalStage, chainHead }
  422 Rej:  { error: { code: "ADMISSIBILITY_REJECTED", admissibility: {...},
              rejected: { invariant, reason, remediationPath, escalationAllowed }}}
  400 Bad:  empty intent, invalid JSON
  503 Down: gateway unreachable
```

Bootstrap: empty ThoughtGraph + empty ReceiptChain + zero genesis. Sufficient for mission submission (no graph traversal required). Originator defaults to `System` — operator-session propagation deferred to a future arc.

### G3b — Next.js proxy (commit `40a6832`)

`/api/missions POST` now proxies to gateway `/mission`. Translates gateway response to the UI's stable `Mission` + `AdmissibilityResult` shape. Reject path (HTTP 422) passes through verbatim so the UI can render `RejectedClaim` remediation guidance.

TS type additions: `GatewayVerdict`, `GatewayGateVerdict`, `GatewayRejectedClaim`, `GatewayAdmissibility`. UI-stable shapes (`AdmissibilityResult`, `GateVerdict`, `Verdict`) kept unchanged to avoid component regression. `MissionStage` reconciled to Rust-aligned values: `Intent`/`Mission`/`Claim`/`Admissibility`/`Execution`/`Receipt`/`Canonicalization`/`Replayability`/`Reflex`.

## First real activation receipt (live curl, 2026-04-17)

Intent: `"activate my dual agentic system"`

| Gate | Verdict | Score |
|---|---|---|
| ZANN_ZERO | Permit | 1.0 |
| CLAIM_MUST_BIND | Permit | 1.0 |
| RIBA_ZERO | Permit | 1.0 |
| NO_SHADOW_STATE | Permit | 1.0 |
| IHSAN_FLOOR | Permit | 0.98 |

Result:
- `missionId`: `4eac2f9366d34cadcc2dc1371b0634ce73a0b1aa6a00f67f267823c5e9564189`
- `receiptId`: **`62a35dcd4b141a24ebe789ca13e36ec5d7027a5c47c7752c0408e97da76d93e8`**
- `finalStage`: `Replayability` (S8 — patch B working)
- `chainHead` == `receiptId` (chain advanced cleanly)
- Chain length: **7** (1 mission envelope + 5 gate verdicts + 1 final NodeLifecycle receipt)

Reject path verified with `qualityScore: 0.5`:
- HTTP 422
- All 4 non-Ihsan gates Permit; IHSAN_FLOOR Reject ("score 0.5000 below floor 0.9500")
- `rejected.remediationPath`: *"Improve claim quality score to ≥ 0.95 before resubmitting. Add tests, documentation, or constitutional alignment evidence to raise the Ihsan score."*
- No canonicalization receipt emitted (patch A working)

## Evidence

### Test results

| Layer | Count | Delta |
|---|---|---|
| `bizra-cognition` (kernel) | 64/64 green | (unchanged since G2) |
| `bizra-cognition-gateway` | **7/7 green** | +3 (mission permit, reject 422, empty-intent 400) |
| `tsc --noEmit` (frontend) | clean | — |
| `vitest` (frontend) | 135/135 green | — |
| Workspace cargo test | all crates green, 1,200+ tests | — |

### Constitutional fidelity (end-to-end)

| Anchor | How G3 upholds it |
|---|---|
| ZANN_ZERO | mission bound to evidence before canonicalization |
| CLAIM_MUST_BIND | MissionEnvelope appended to chain before evaluation |
| RIBA_ZERO | economic_pattern optional; default None does not canonicalize extraction |
| NO_SHADOW_STATE | gateway returns null timestamps where payload is not decodable; reject path emits NO fabricated success receipt (patch A) |
| IHSAN_FLOOR | 0.95 floor enforced; qualityScore default 0.98 passes comfortably |

## Scope discipline

- Touched files (G3a): `bizra-cognition-gateway/src/main.rs` (1 file)
- Touched files (G3b): `app/api/missions/route.ts`, `app/api/missions/[id]/route.ts`, `lib/dema/types.ts` (3 files)
- **Not touched**: the 5 other Dema stub endpoints (`/api/gates/:missionId`, `/api/missions/[id]/replay`, `/api/manifest/today`) remain stubbed — separate follow-up arcs. This keeps G3 narrow to the activation surface alone.
- **Not touched**: the ~200 pre-existing dirty files in bizra-data-lake and the 16 parallel-session files in award-winner-design — Path 1 discipline preserved.

## What G3 claims and does not claim

**G3 claims:**
- The end-to-end loop works: principal intent → MissionEnvelope → 5-gate admissibility → ReceiptArtifact → chain → gateway → UI
- The first real principal-activation receipt exists on the chain
- Both permit and reject paths are structurally honest (no shadow state on either)
- IntentEntry in Dema can submit real missions once the gateway is live locally

**G3 does NOT claim:**
- The **browser** end-to-end has been tested with a real Mumo session (requires authenticated /dema walkthrough with gateway running — Mumo's visual acceptance)
- Gates/manifest/replay endpoints are bridged (still WIRED_PARTIAL)
- The slice is CANONICAL yet. Per Cycle-4 canonicality gate, CANONICAL requires PROVEN + visible operator-path confirmation + Daughter Test. Mumo typing his intent in browser and seeing his own activation receipt is that confirmation — pending.

## Canonicality re-label

- `bizra-cognition` — still **PROVEN, trending CANONICAL**. Visible operator-path confirmation gated on Mumo's browser walkthrough with G3 live.
- `bizra-cognition-gateway` — still **PROVEN**. Same.
- The `/api/chain` slice — was VALIDATED as of G1; stays VALIDATED.
- The `/api/missions` slice — moves from **WIRED_PARTIAL** to **WIRED_REAL** (Dema UX still unverified in browser end-to-end).

## Next step

**G4 (optional final gate): Browser confirmation.** With both services up (gateway release binary + `pnpm dev`), Mumo logs into `/dema`, types *"activate my dual agentic system"* in the Mission Intent box, and observes:
1. A Mission appears with real 64-char mission_id
2. GateViewer shows 5 PERMIT gates
3. ReceiptExplorer gains one new receipt at the current chain head
4. No red console errors

When G4 passes, the `bizra-cognition` + gateway + frontend bridge slice can be relabeled **CANONICAL**. That is when the minimum undeniable loop becomes a lived experience for the founder.

---

Close it. Prove it. Reveal it.
