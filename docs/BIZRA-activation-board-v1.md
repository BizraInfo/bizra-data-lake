# BIZRA Activation Board — v1

بسم الله الرحمن الرحيم

**Filed:** 2026-04-17 Dubai GST
**Post-cycle:** Cycle-6 sealed (origin `86899c5c`); 4 gates closed by machine evidence
**Purpose:** One-page operational canvas. Every future commit is judged against one of these five workstreams + explicit acceptance signals. Prevents cycle drift; enables multi-arc concurrency without scope-widening.

---

## Operational invariants (all workstreams)

1. **Fail-closed on ambiguity.** If evidence is unclear, halt and gather. Never act on inferred state.
2. **Tool evidence outranks grep / speculation.** Especially for security + integrity.
3. **Writer archaeology for any integrity-critical format.** Read the authoritative writer; verify against live data; then port.
4. **Narrow-real scope.** Each arc closes before the next opens. No scope-widening mid-pass.
5. **Path-1 preserved.** Pre-existing dirty tree is not janitorial material; it is queued work.
6. **Self-correction preserved, not rewritten.** Superseded claims stay visible with annotation.
7. **Constitutional filter on every change.** ZANN_ZERO, CLAIM_MUST_BIND, RIBA_ZERO, NO_SHADOW_STATE, IHSAN_FLOOR(0.95).

---

## Workstream 1 — Core Runtime Truth

**Scope:** The lawful loop. Kernel crates (`bizra-cognition`, `bizra-cognition-gateway`), runtime constructors, chain verification.

**Current state (post-Cycle-6):** G1 durable-read COMPLETE (live-verified); submit_mission PROVEN; rehydrate_mission PROVEN; 5-gate admissibility structurally enforced; chain fail-closes on integrity.

**Acceptance signals for future commits in this workstream:**
- [ ] Every new runtime method has a fail-closed path (no silent success on bad input)
- [ ] Every new chain operation has a live-fixture byte-parity test if it interacts with `sovereign_state/`
- [ ] `verify_continuity` remains structurally impossible to bypass
- [ ] Constitutional invariants unchanged at the kernel layer

**Queued arcs:**
- Cycle-6.5b: Signer audit (G1 amendment — `mission_signer.json` binding)
- Cycle-6.5c: Constitutional-threshold drift check (Rust const vs `block_zero.json`)
- Cycle-7+ candidate: Trust Compiler substrate (narrow `Executor`/`SubReceipt` extraction from Downloads archive)
- Cycle-7+ candidate: Rust-writes-back capability (current posture: Python authoritative writer)

---

## Workstream 2 — DevSecOps Truth Automation

**Scope:** CI, CD, security tooling, audit coverage, rollback discipline.

**Current state:** Justfile 31 recipes; cargo-audit + pip-audit installed and wired; 22 CI workflows; `docs-quality` green; rollback runbook on origin; intentional-red pattern validated then retired on schedule; `e2e-polyglot.yml` hardened with Rust toolchain + cache + failure artifact upload.

**Acceptance signals for future commits in this workstream:**
- [ ] No CI workflow stays red > 7 days without a visible explanation commit
- [ ] No security claim ships without tool output (cargo audit, pip-audit, `gh api`)
- [ ] Every new Justfile recipe has a paired CI step OR is explicitly marked dev-only
- [ ] Every release binary build is reproducible from a tagged SHA

**Queued arcs:**
- Cycle-6.5d: Jarvis P1 patch (langchain-core ×5, langsmith ×1, starlette ×2 — surfaced when jaeger pin repaired)
- Cycle-7+ candidate: Wire `just audit-rust` + `just audit-python` into a fail-closed CI workflow (currently available but not gated)
- Cycle-7+ candidate: SBOM generation + cosign signing for container images
- Cycle-7+ candidate: `gh api` reconciliation between Dependabot and local tool counts

---

## Workstream 3 — Performance and Resilience

**Scope:** Latency targets, resource budgets, canary + rollback + anomaly + restart-survival drills.

**Current state (post-Cycle-6):** Restart-survival exercised (G1 live curl). Rollback runbook exists. Release binary builds clean with `cargo build --release` (opt-level 3, LTO fat).

**Acceptance signals for future commits in this workstream:**
- [ ] Any new kernel path has an upper-bound latency assertion (P95 < 250 ms end-to-end)
- [ ] Any new durable path has a restart-survival test
- [ ] Any new external integration has a canary + rollback plan documented BEFORE landing
- [ ] Anomaly injection drill covers at least one rarely-fired circuit per cycle

**Queued arcs:**
- Canary drill — deploy a non-trivial Dema change behind a flag; measure regression; roll back
- Rollback drill — execute a real `git revert` through CI on a staging equivalent
- Anomaly drill — inject a tampered receipt; verify fail-closed path triggers
- Restart-survival drill — prove rehydration across 10 consecutive restarts without drift

---

## Workstream 4 — Product Face (DEMA)

**Scope:** The operator's truth surface. `award-winner-design` Next.js (primary per G3 ADR) + `dema` CLI (fallback).

**Current state:** G3 formalized — external Next.js authoritative; in-repo `frontend/` retained as historical. Cycle-5 G3b shipped the first live principal-activation flow; receipt `62a35dcd…` sealed through external proxy. `dema` CLI ships 7 subcommands; live-walk verified.

**Acceptance signals for future commits in this workstream:**
- [ ] DEMA surfaces: trust state · current→ideal delta · latest sealed receipt · next admissible action
- [ ] No UI element simulates capability the gateway does not actually expose
- [ ] No shadow state between UI and gateway (single source of truth = gateway's `/chain`, `/chain/{hash}`, `/v1/mission`)
- [ ] Every product claim traceable to a specific gateway endpoint + contract

**Queued arcs:**
- Cycle-7+ candidate: `dema-overlay.jsx` (from Downloads archive) placement into `award-winner-design`
- Cycle-7+ candidate: Operator-session propagation (originator currently defaults to `System`)
- Cycle-7+ candidate: Browser-verified walkthrough (Mumo types intent → sees own activation receipt)
- Cycle-7+ candidate: `dema organize` subcommand (from Downloads archive `dema_cli_v02_organize.rs`) — first real impact surface

---

## Workstream 5 — Governance and Activation Discipline

**Scope:** Cycle chain integrity, doctrine amendment path, activation order, institutional memory.

**Current state:** 6 cycles sealed (3, 4, 5, 6 retrospectives on origin); Manifesto v1 + Trust Compiler Thesis + FTAP seed filed; G2/G3/G1 ADRs filed; Downloads archive classified + 1 promotion filed; al-mithaq-al-tasisi founding covenant in `docs/`.

**Acceptance signals for future commits in this workstream:**
- [ ] Every cycle closes with a retrospective before the next niyyah opens
- [ ] Every authority decision ends in an ADR — never re-opened without founder direction
- [ ] Every contradiction is logged + resolved within the same cycle OR explicitly deferred with scope gate
- [ ] Every Downloads / external bundle is archived with INVENTORY before any file moves into canonical paths
- [ ] Self-Amendment Circuit (SAC) path documented before any Block-0-style public claim

**Queued arcs:**
- Cycle-7+ candidate: Self-Amendment Circuit — canonical doctrine amendment procedure (currently: `docs/manifesto-amendments/v0-to-v1.md` is one-off; SAC generalizes)
- Cycle-7+ candidate: Witness review + constitutional ratification phases per pre-activation sequence
- Cycle-7+ candidate: Controlled activation — first external witness of the lawful loop

---

## Per-cycle sprint discipline

Every cycle must produce (in order):

1. **Niyyah** (intent declaration, ≤ 1 page, 4 gates max, out-of-scope explicit)
2. **Bayyinah** (evidence ingested, with source paths)
3. **Hadd** (scope boundary — in/out with Daughter Test)
4. **Amanah** (execution — commits with hashes + test deltas)
5. **Thamara** (reward computation against cycle-weighted rubric)
6. **Iisal** (proof manifest — single block summary)
7. **Retrospective** (contradictions log + topology changes + next-cycle candidates)

Every ADR in a cycle is founder-gated. Every contradiction is founder-visible. Every retrospective names the specific self-corrections made and the institutional memory they became.

---

## Reward weighting rubric (for Thamara phase)

Per cycle-specific component weights (Cycle-5 and Cycle-6 set precedent):

| Component | Typical weight | Notes |
|---|---|---|
| Tests delta (new + regression-free) | 0.15 – 0.20 | Weight rises with kernel-touching cycles |
| Runtime closure (new lawful path proven) | 0.15 – 0.20 | Highest for boundary-crossing cycles |
| Authority formalization (ADR quality) | 0.10 | Only applies when a gate ends in ADR |
| Security hardening | 0.10 | Weight rises when CVEs closed in-cycle |
| Constitutional fidelity | 0.10 – 0.15 | Non-negotiable floor; always > 0 |
| Self-correction quality | 0.10 | How well contradictions were resolved |
| DevOps discipline | 0.05 – 0.10 | Commit hygiene, Path-1, rollback paths |
| Founder-feedback adherence | 0.05 | How well session synthesis was executed |
| Cycle closure completeness | 0.05 | All gates sealed by machine evidence |

**Cycle-success floor:** 0.70
**Cycle-5 reward:** 0.971
**Cycle-6 reward:** 0.998

A reward below 0.70 triggers a cycle-opening re-audit, not a continuation.

---

## Activation order (pre-Block-0 discipline)

Per Manifesto v1 §activation, in strict order:

1. **Integrity Proof Sprint** — lawful loop proven end-to-end on single node. **✅ Cycle-6 closed this phase.**
2. **Witness Review** — first external witness verifies the sealed loop. QUEUED
3. **Constitutional Ratification** — doctrine sealed with witnesses' signatures + SAC path filed. QUEUED
4. **Controlled Activation** — first principal activation observed by witness(es). QUEUED
5. **Block-0 Public Claim** — only after steps 1-4 complete. NOT BEFORE.

---

## Reference points

- Cycle chain: `cycle-3/`, `cycle-4/`, `cycle-5/`, `cycle-6/` retrospectives
- Doctrine: `docs/bizra-trust-compiler-thesis.md`, `docs/dema-cli-manifesto-v1.md`, `docs/ftap-function-registry-rfc-seed.md`
- Founding covenant: `docs/al-mithaq-al-tasisi.md`
- Handover + inventory: `docs/BIZRA-Handover-v1.md`, `docs/BIZRA-Repo-Inventory-v1.md`
- DevOps audit: `docs/CI-POLICY-AUDIT-v1.md`
- Rollback runbook: `docs/ROLLBACK-RUNBOOK-Cycle-5.md`
- Downloads archive: `archive/downloads-files-7-2026-04-17/INVENTORY.md`

## Signature

Filed: Mumo (Muhammad Beshr) — 2026-04-17 Dubai GST
Canon status: **v1** — amendable by SAC (when SAC is filed); until then amendable by founder direction + new version entry

الحمد لله.
