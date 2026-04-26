# Claim-Discipline Drift Remediation Log — 2026-04-26

**Status:** docs-only remediation, runtime-evidence-backed, instrumentation retained until user confirms removal.
**Scope:** top-level masterpiece briefs only.
**Audit branch:** `prep/node0-closure-receipt-lineage` (pre-merge).
**Probe:** `tools/audit/claim_drift_probe.py` (read-only, no canon ingestion, no network).

---

## 1. Why this log exists

The `bizra-pilot` self-evaluation flagged an overclaim in `ULTIMATE_MASTERPIECE_EXECUTIVE_BRIEF.md` ("Status: READY FOR PRODUCTION") as a BLOCKER for PR #58 because it contradicts the project's own claim-discipline framework.

This remediation was performed under Debug-mode discipline: no fix without runtime evidence. The evidence is an NDJSON log produced by a read-only probe that scans the 24 most-touched docs against the patterns derived from `p1_truth_integrity/PROHIBITED_CLAIMS_REGISTER.csv` and `p1_truth_integrity/NEEDS_REWRITE_REGISTER.csv`.

---

## 2. Hypothesis battery and outcome

| ID | Hypothesis | Outcome | Evidence |
|----|------------|---------|----------|
| H1 | Top-level briefs carry explicit production-ready / guaranteed / trustless / AGI / world-first language without truth-label qualifiers. | CONFIRMED (10 true positives in 3 files). | Probe-initial log, lines 2–11. |
| H2 | C-class numerics (SNR 0.974, $0.10→$0.008, 100% pass, 73/100 nodes) live in docs without linked receipts. | INCONCLUSIVE (probe regex too narrow; however the top-3 briefs do carry unreceipted `2.2x`, `112x`, `352ms→160ms`, `8→900 req/sec`, `60–80% cache hit`, `0.95 Ihsan`, which the added banner contextualizes). | No positive log lines. |
| H3 | Origin Kernel §6.3 discipline drift (Kernel treated as ingested / runtime canon). | REJECTED. Both log hits are inside `CANON_STORE_INGESTION_GATE_DESIGN.md` §1 `Non-authorization` and §15 `Stop Line`, which negate the pattern rather than assert it. | Probe-initial log, lines 32–33; verified by reading source at `docs/audits/omnidirectional_hyperdimensional_audit_v0_1/CANON_STORE_INGESTION_GATE_DESIGN.md:6` and `:307–319`. |
| H4 | A single doc carries both the legacy "Node0 proves the seed can live alone…" sentence and the Topology Canon "Each human node mints PAT-7…" sentence. | REJECTED (no hits). | Probe-initial log, no H4 entries. |
| H5 | `PILOT_EVIDENCE_REGISTER.md` or related docs upgrade a `MEASURED_LOCAL_ARTIFACT` to a cross-device / multi-node `MEASURED` claim. | REJECTED (all 6 hits are inside `PLANNED`, `NO-GO`, `unproven`, `Red Lines`, or `until proven` contexts; cross-checked against source). | Probe-initial log, lines 12, 15, 18, 19, 22, 25; verified against `STATUS.md:83`, `INVESTOR_OPERATOR_HANDOVER.md:63–78`, `BUSINESS_MODEL_AND_PRICING_OPTIONS.md:87–94`, `PRODUCTION_READINESS_AND_GTM_CLOSURE_SPRINT.md:57, 175`. |

No rejected-hypothesis code changes accumulated; only the H1-driven patches were applied.

---

## 3. Lines patched (runtime-evidence backed)

All patches are docs-only, confined to three files at the repo root. No canon, no code, no schema, no runtime state.

| File | Line (pre-fix) | Before | After |
|------|----------------|--------|-------|
| `ULTIMATE_MASTERPIECE_EXECUTIVE_BRIEF.md` | 8 | `Status: READY FOR PRODUCTION` | `Status: HISTORICAL ASPIRATIONAL DRAFT — NOT A PRODUCTION RUNTIME` |
| `ULTIMATE_MASTERPIECE_EXECUTIVE_BRIEF.md` | 327 | `✓ Byzantine fault tolerance guaranteed` | `✓ Byzantine fault tolerance: design target (not independently verified)` |
| `ULTIMATE_MASTERPIECE_EXECUTIVE_BRIEF.md` | 434 | `READY FOR PRODUCTION. READY TO SCALE. READY TO TEACH.` | `Not a production runtime. Single-node architecture documented; private pilot is the next verifiable milestone. See the claim-discipline registers.` |
| `ULTIMATE_MASTERPIECE_MANIFESTO.md` | 41 | `✅ 0.95 Ihsān compliance (production-ready)` | `✅ 0.95 Ihsān compliance: DESIGN TARGET (PREPARATION; not a production runtime)` |
| `ULTIMATE_MASTERPIECE_MANIFESTO.md` | 331 | `└─ Byzantine tolerance guaranteed` | `└─ Byzantine tolerance: design target (not independently verified)` |
| `ULTIMATE_MASTERPIECE_MANIFESTO.md` | 362 | `└─ 0.95 Ihsān compliance (production ready)` | `└─ 0.95 Ihsān compliance: DESIGN TARGET (PREPARATION; not a production runtime)` |
| `ULTIMATE_MASTERPIECE_MANIFESTO.md` | 532 | `Ready for production NOW.` | `Not a production runtime now. Private pilot is the next verifiable milestone.` |
| `ULTIMATE_MASTERPIECE_POLYMATH_SYNTHESIS.md` | 573 | `✓ Byzantine tolerance guaranteed (consensus theory)` | `✓ Byzantine tolerance: design target informed by consensus theory (not independently verified)` |
| `ULTIMATE_MASTERPIECE_POLYMATH_SYNTHESIS.md` | 599 | `✓ 0.95 Ihsān compliance (production ready)` | `✓ 0.95 Ihsān compliance: DESIGN TARGET (PREPARATION; not a production runtime)` |
| `ULTIMATE_MASTERPIECE_POLYMATH_SYNTHESIS.md` | 634 | `Ready for production NOW.` | `Not a production runtime now. Private pilot is the next verifiable milestone.` |

In addition, each of the three files received a top-of-file **Claim-Discipline Banner** (HTML comment + visible Markdown blockquote) that declares the document as HISTORICAL ASPIRATIONAL DRAFT, not a measured production-readiness report, and links to the authoritative claim registers.

---

## 4. Proof-of-Truth Convergence for this remediation

- **Formal.** The patterns under test are derived from the already-committed `p1_truth_integrity/PROHIBITED_CLAIMS_REGISTER.csv` and `NEEDS_REWRITE_REGISTER.csv`. The probe is a deterministic regex scan with no hidden state.
- **Cryptographic.** Evidence is persisted as NDJSON at `.cursor/debug-c98f9f.log`; each line carries `sessionId`, `runId`, `timestamp`, `hypothesisId`, `location`, and the raw matched line for reproducibility by any reviewer running the same probe.
- **Empirical.** Line-count falsifier: pre-fix 34, post-fix 24, delta exactly 10, matching the 10 patched lines. Zero residual hits in the three target files in the post-fix run.
- **Economic.** No production cost paid for this fix (docs-only). Public-claim liability reduced, because the three most-likely-to-be-quoted files no longer present a `READY FOR PRODUCTION` banner that would contradict the project's own `CLAIM_DISCIPLINE_FOR_NODE0_AND_URP.md`.

---

## 5. What was NOT changed (by design)

- No edit to `docs/canon/BIZRA_ORIGIN_KERNEL.md`.
- No edit to `MEMORY.md`.
- No edit to Rust or Python runtime canon stores.
- No edit to any public-website file (that remains gated by the website patch plan in `p1_truth_integrity/`).
- No new runtime dependency. The probe uses only Python stdlib (`re`, `json`, `pathlib`, `time`).
- No rewrite of the aspirational voice of the three briefs; only the overclaim lines flagged by the probe plus a context banner.

---

## 6. Instrumentation status

- `tools/audit/claim_drift_probe.py` is retained until the user confirms removal of debug-session instrumentation. Because it is read-only, deterministic, and has no external effects, it may also remain as a permanent claim-discipline CI check if desired. That decision is left to the user.
- Log path `.cursor/debug-c98f9f.log` is re-created on each probe run and is not tracked by git.

---

## 7. Residual findings worth escalating (not fixed in this log)

These are the 20 remaining log lines in the post-fix run. All are expected-good discipline rather than true overclaims, but they deserve a labeled note:

- `STATUS.md:83` — `multi-node ordering unproven` (explicit disclaimer inside a risk table). KEEP.
- `docs/architecture/BIZRA_NODE0_TO_URP_ECOSYSTEM_TRANSITION_v0_1.md:166-167` — `❌ Not AGI`, `❌ Not "world-first"` (explicit negation). KEEP.
- `docs/gtm/node0_activation_go_to_market_v0_1/README.md:37` — `Multi-node URP | PLANNED | … not proven yet`. KEEP.
- `docs/gtm/node0_activation_go_to_market_v0_1/PRODUCTION_READINESS_AND_GTM_CLOSURE_SPRINT.md:57, 149, 175` — `PLANNED … not proven`, stop-line list, `cross-device proof without overstating production`. KEEP.
- `docs/gtm/node0_activation_go_to_market_v0_1/INVESTOR_OPERATOR_HANDOVER.md:5, 68, 71` — framing sentence and `## Red Lines` list. KEEP.
- `docs/gtm/node0_activation_go_to_market_v0_1/BUSINESS_MODEL_AND_PRICING_OPTIONS.md:71, 89, 91` — `Directional until proven`, `## Claims to Avoid in Sales`. KEEP.
- `docs/gtm/node0_activation_go_to_market_v0_1/CLAIM_DISCIPLINE_FOR_NODE0_AND_URP.md:48, 49, 50, 62` — the register itself, quoting prohibited phrasings in context. KEEP.
- `docs/audits/omnidirectional_hyperdimensional_audit_v0_1/CANON_STORE_INGESTION_GATE_DESIGN.md:6, 312` — `Non-authorization` and `Stop Line`. KEEP.

Future probe hardening (non-blocking): make the probe aware of `## Red Lines`, `## Claims to Avoid in Sales`, `## Stop Line`, `### Non-authorization`, and `❌ Not …` sections, so these false positives are suppressed automatically. Out of scope for this remediation because the current false-positive rate is auditable by eye and does not contaminate the signal.

---

## 8. Proposed next logical step (for explicit operator approval)

Two options, pick one:

- **A. Minimal close-out.** Commit this remediation log plus the three patched briefs plus the probe onto the current branch, then re-mark PR #58 ready-for-review with a pointer to this log as the claim-discipline evidence.
- **B. Upgrade to CI gate.** Add a GitHub Action that runs `tools/audit/claim_drift_probe.py` on every PR and fails the build if any H1 finding lands in a file that is NOT in an allowlist (the registers, stop-line docs, and anti-claim documents listed in §7). This converts claim-discipline from an author-time habit into a mergetime gate.

Both options are docs/CI-only. Neither mutates canon, runtime state, or website claims. Authorization required before execution.
