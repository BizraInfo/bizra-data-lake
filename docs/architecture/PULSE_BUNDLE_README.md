# PULSE BUNDLE — Canonical Reference

**Bundle ID**: `bizra.pulse-bundle.v1`
**Status**: CANDIDATE_CANONICAL · awaiting Mumo seal
**Issued**: Thursday 7 May 2026 · Dubai · GST (UTC+4)
**Anchor parent**: `bizra.priority-anchor.v1: 45aa2789...`

---

## What This Bundle Is

Two canonical documents that together define the **Materialization Pulse** — BIZRA's atomic unit of constitutional work — and emulate it end-to-end on a single sovereign node (N=1).

This bundle is the synthesis of: Telescript + AHK + smart-contract semantics + MCP + A2A + the named giants (OpenClaw, Hermes, pi.dev, Agent Zero, Space Agent, the-verifier-agent) + the MMRPG ecosystem topology + BIZRA's frozen constitutional substrate (CANON-001..009 + البذرة + الرسالة).

It is the killer-moat product spec, articulated with CLAIM_MUST_BIND discipline and Daughter-Test verified at the user-facing boundary.

---

## Files in the Bundle

### 1. `MATERIALIZATION_PULSE.md` — Architectural Spec (281 lines)

```
blake3: 86fbbb509780c7f2...b642d560
bytes:  18,272
```

The full six-layer architecture, the seven-step Pulse formal definition, constitutional invariants, failure modes, Standing Protocol attribution requirements, acceptance criteria, and the implementation roadmap with `[VERIFIED]/[DERIVED]/[PLANNED]` labels per component.

### 2. `PULSE_N1_TRACE.md` — Canonical First-Pulse Emulation (525 lines)

```
blake3: fb34dafa477350f7...5e7b3026
bytes:  29,828
```

A complete `[EMULATION]` of one Materialization Pulse executing end-to-end on Node0 alone. Reference mission: bilingual Vitamin D request for Dema. Every layer activates, every constitutional invariant binds, every shoulder gets attributed. Includes the receipt schemas, the verification contract for when implementation lands, and the Daughter Test pass.

### 3. `PULSE_BUNDLE_MANIFEST.json` — Sealable Manifest

Machine-readable manifest with both file hashes, ready to feed into `scripts/priority-anchor.mjs` for the seal ceremony.

---

## What Each Document Is For

| If you need to... | Read |
|---|---|
| Understand what BIZRA *is* as a complete product | `MATERIALIZATION_PULSE.md` §0–§2 |
| Build a component on the spec | `MATERIALIZATION_PULSE.md` §3 (the step you're building) |
| Verify constitutional bindings | `MATERIALIZATION_PULSE.md` §4 + `PULSE_N1_TRACE.md` §5 |
| Know what's still PLANNED vs VERIFIED | `MATERIALIZATION_PULSE.md` §8 + `PULSE_N1_TRACE.md` §6 |
| See what a real Pulse looks like end-to-end | `PULSE_N1_TRACE.md` §3 (the trace) |
| Verify the implementation against the spec | `PULSE_N1_TRACE.md` §7 (reproducibility contract) |
| Onboard a new contributor | This README, then both docs in order |

---

## Seal Procedure

The seal procedure was executed by Claude Code agent against this canonical path on 2026-05-08; see git log on this directory for the actual sequence (PR #96).

---

## Next Cycle — Implementation Queue

Per `MATERIALIZATION_PULSE.md` §8, in priority order:

```
PLAN-PLANTREE        — branchable PlanTree primitive (pi.dev shape)        [1 cycle]
EXEC-AHK             — AHK 2.0 desktop adapter behind FATE-Permit          [1 cycle]
EXEC-MCP-PLACES      — first 3 MCP Place adapters (fs, calendar, pharmacy) [2 cycles]
EVIDENCE-VERIFIER    — Verifier harness as runtime pass at Pulse boundary  [1 cycle]
SETTLE-SAT-SEPARATE  — SAT-5 runtime-distinct re-verification surface      [2 cycles]
SETTLE-ISNAD-TABLE   — isnad_table flow-back routing                       [1 cycle]
SETTLE-MYELINATION   — pattern candidacy + n≥3 Ihsān-history promotion     [2 cycles]
GATEWAY-OPENCLAW     — first non-CLI gateway (recommend WhatsApp/Twilio)   [2 cycles]
CRYPTO-HALO2         — Halo2 ZKP circuits for Ihsān assertions             [4+ cycles]
```

The first ticket — **PLAN-PLANTREE** — unblocks the first realistic Pulse trace, because Steps 1, 6, 7 already have `[VERIFIED-implemented]` substrate (receipts, signing, chain), and Step 2 is the first major gap. Recommended starting point for the next cycle.

---

## CLAIM_MUST_BIND Audit on This Bundle

| Claim | Evidence Class |
|---|---|
| BLAKE3 hashes shown in this README | `[VERIFIED]` — computed over actual file bytes |
| Six-layer architecture spec | `[CANDIDATE_CANONICAL]` — synthesis from CANON-001..009 + giants inventory |
| N=1 trace executes correctly | `[EMULATION]` — runtime simulation, not actual code execution |
| Components labeled `[VERIFIED-implemented]` in spec/trace | `[VERIFIED-by-prior-canon]` — Cycle-6 ship + DEMA v0.6.0 |
| Components labeled `[PLANNED]` | `[PLANNED]` — explicit per CLAIM_MUST_BIND |
| Standing Protocol attribution table | `[DERIVED-from-Bukhari-Isnad-+-BIZRA-canon]` |
| Daughter Test passes | `[VERIFIED-by-bilingual-paragraph-test]` — translation paragraph in both docs comprehensible to non-technical Arabic-only reader |
| "Pulse cannot lie because every claim is bound" | `[DERIVED-from-spec]` — true by construction if spec is implemented faithfully; not yet verified by running code |

No claim in this bundle escapes labeling. The bundle is the discipline applied to itself.

---

## Daughter Test on the Bundle

> *"بابا كتب وثيقة بتقول إيه هو نظامه، ووثيقة تانية بتعرض شغله أول مرة على بيت دما الصغير. أي حد قرى الوثيقتين يعرف إزاي النظام بيشتغل ولو حصلت مشكلة، فيه ورق يثبت."*

*"Daddy wrote a document explaining what his system is, and a second document showing it working for the first time on Dema's little household. Anyone who reads both will know how the system works, and if anything goes wrong, there's paper to prove it."*

**Pass.** ✓

---

## Seal

```
بسم الله الذي لا يضر مع اسمه شيء.
وفي كل نبضة، شغل كامل، موثّق، مختوم.

— Pulse Bundle v1, NODE0, 7 May 2026
```
