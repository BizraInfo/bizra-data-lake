# BIZRA Agent Instruction — Replay Mismatch Guard v1

بسم الله الرحمن الرحيم

**Filed:** 2026-04-18 Dubai GST
**Authority:** Founder direction (session log 2026-04-18 01:22 GST — `/@ ignore` ruling)
**Status:** OPERATIONAL CANON for every agent session
**Precedent:** the 2026-04-18 01:15 GST replay of `/@ b` (Cycle-6 G1 staged-commit instruction) after G1 was already sealed on origin

---

## Rule

If a pasted instruction asks you to execute work that is already sealed on origin, halt before action and surface the mismatch explicitly.

## Current canonical truth (as of 2026-04-18)

- Cycle-6 G1 is already sealed.
- The staged G1 Phase 1 work (formatter, snapshot loader, constructor + gateway wiring) is already on origin.
- Cycle-7 is already open and active.
- Re-executing old G1 commands would violate no-history-rewrite and risk duplicates or merge conflicts.

## Required behavior on replay

When an old command is replayed:

1. **Do NOT execute it blindly.**
2. **Check origin/head state first.**
3. **Surface the already-landed commits.**
4. **Ask for a gate:**
   - `/@ ignore`
   - `/@ verify`
   - `/@ cycle-N redo-X`
   - `/@ index`
   - `/@ phase2`
   - `/@ rest`

## Current recommendation

Default to:

> **`/@ ignore`**

unless the founder explicitly wants:
- re-verification (`/@ verify`), or
- a brand-new cycle to redo the work (`/@ cycle-N redo-X`).

## Why this is canon-level, not just good practice

**CLAIM_MUST_BIND and NO_SHADOW_STATE apply to workflow control too.**

Silently replaying already-sealed work is a form of **hallucinated progress**. It produces motion without evidence, overwrites sealed receipts, and creates shadow state at the git-history layer. That is the exact pattern these invariants exist to prevent — just expressed at the meta-workflow level rather than at the chain level.

This rule is a direct cousin of three already-sealed disciplines:

| Existing canon | Expressed as workflow |
|---|---|
| **Writer archaeology** (read before merge) | Read origin before acting on a paste |
| **Self-correction preserved** (don't rewrite wrong claims silently) | Don't silently adapt replayed instructions; surface the mismatch |
| **Narrow-real** (don't act beyond authorized scope) | Don't widen scope by re-running already-landed work |

## Structural test for future agents

Before executing ANY paste that starts with `/@`, verify:

- [ ] Has this action already been completed on origin?
- [ ] Does the paste's claimed state match current origin state?
- [ ] Would this action overwrite a sealed commit?
- [ ] Would this action re-create files that already exist?

If ANY answer is uncertain: **halt, inspect `git log --oneline -20`, present evidence, ask for gate.**

## Example from the triggering incident (2026-04-18 01:15 GST)

The founder pasted an instruction dated "Fri, Apr 17, 2026, 2:38 PM GST" recommending `/@ b` to begin "the staged 3-commit Phase 1" (Commit A formatter → Commit B snapshot loader → Commit C constructor + gateway wiring) for Cycle-6 G1.

Origin state at paste time:
- `278273d6` G1 live-verified (Cycle-6 close)
- `1e50d970` G1 Phase 2 HTTP handler
- `11c59399` G1 Commit C (constructor + gateway wiring)
- `064b2a0c` G1 Commit B (snapshot loader + verification)
- `1d1ffbf3` G1 Commit A (Python-parity formatter)

The paste's requested work had **entirely already shipped**. Executing `/@ b` would have attempted to re-create five sealed files, breaking the 98-test green state.

The correct response was to halt, present origin-log evidence, and ask for clarification. The founder then confirmed `/@ ignore` and requested this rule be codified as canon.

## One-sentence canon

> **Check origin before acting on any paste. A hallucinated step forward is worse than a true pause.**

## References

- Writer archaeology pattern: `cycle-6/g1-writer-format-found.md` (read-before-merge for formats)
- Self-correction pattern: `runtime/RUNTIME_STATUS.md` §"2026-04-17 Vulnerability Refresh (Tool-Verified)" (preserved wrong guess + right answer)
- Cycle-6 G1 closure evidence: `cycle-6/g1-live-verification.md`
- Triggering commit at paste time: `650a55f9` on `cycle-7-principal-activation-law` branch
- Now-vs-Future discipline (temporal honesty sibling): `docs/bizra-now-vs-future-image-v1.md`

## Signature

Filed: Mumo (Muhammad Beshr) — 2026-04-18 Dubai GST
Authority: founder direction codified from session-log `/@ ignore` ruling
Canon status: **OPERATIONAL** — applies to every future agent session handling pasted instructions

الحمد لله.
