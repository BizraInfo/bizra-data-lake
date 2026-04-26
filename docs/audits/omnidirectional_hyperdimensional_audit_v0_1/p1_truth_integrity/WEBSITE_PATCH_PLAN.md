# Website Patch Plan — BIZRA v0.1

**Date:** 2026-04-24 GST
**Purpose:** Surface-by-surface patch roadmap for closing `G-FW-003 Public
claims clean`. Owner columns are placeholders; the operator assigns actual
owners.
**Constraint:** This plan is a documentation artefact only. No website source
is edited here.

---

## Legend

- **Priority:** P0 = platform-policy risk; P1 = truth-debt; P2 = improvement.
- **Evidence required:** receipt or citation the rewrite must carry OR
  `n/a — safe directional`.

---

## 1. `bizra.ai` hero (the surface that appears in the shell HTML)

Source: `WEBSITE_PUBLIC_CLAIMS_AUDIT.md` § 2 + `SAFE_REWRITE_PACK.md` § Hero.

| # | Claim | Current (risky) | Safe replacement | Priority | Evidence | Owner |
|---|-------|------------------|------------------|----------|----------|-------|
| W-H-01 | C1: sovereignty | "local agents / no cloud dependency / no telemetry." | "Your agents run on your machine. Your keys, your data, your node." | P1 | Soft directional; conditional on `/privacy` live for "no telemetry" variant | site copy owner |
| W-H-02 | C2: no telemetry | (same) | Merge into W-H-01; defer "no telemetry" until privacy policy published | P0 | Must have `/privacy` receipt OR drop the line | site copy + legal |
| W-H-03 | C3: crypto surface | "Ed25519 receipt signatures." | Move to Under-the-Hood sub-page with commit-hash citation to receipt code. | P1 | Link to `bizra-omega/bizra-core/src/canonical_receipt.rs` + sample signed receipt | site + core |
| W-H-04 | C4: cost figures | "cost per action dropping from about $0.10 toward $0.008." | "Designed to make verified action radically cheaper than cloud AI." | **P0** | Platform-policy: unsupported economic claim | site copy owner |
| W-H-05 | C5: SNR exact | "SNR 0.974." | "A signal-vs-noise discipline that keeps outputs tied to evidence." | **P0** | Platform-policy: unsupported quantitative claim | site copy owner |
| W-H-06 | C6: test count | "8,072 verified tests." | "Thousands of verified tests." OR "View the last CI run →" with link | P1 | Must link to a specific GH Actions run if kept exact | site + CI |
| W-H-07 | C7: pass rate | "100% pass rate." | "CI must pass before merge — the same discipline we apply to our claims." | **P0** | Platform-policy: brittle, falsifiable mid-campaign | site copy owner |
| W-H-08 | C8: Ihsan Gate | "Ihsan Gate >= 0.95." | Keep; label as internal gate with `constants.py` citation. | P2 | Link to `core/integration/constants.py` | site + core |
| W-H-09 | C9: 73/100 nodes | "73 of 100 nodes remaining." | Live counter wired to source of truth, OR remove entirely. Never use in paid ads. | **P0** | Platform-policy: deceptive practices risk if static | site + data pipeline |
| W-H-10 | K1: "BIZRA is live" | (in media kit copy) | "The Seed is public." | P1 | n/a | brand owner |

---

## 2. `bizra.ai` sub-pages (client-side-rendered content not visible in shell)

All NEEDS_REWRITE surface categories land here once the hero is safe.

| Theme | Scope (source docs → sub-page) | Action | Priority |
|-------|---------------------------------|--------|----------|
| Production readiness | `docs/BATCHING_QUICK_START.md`, `docs/WIRING_GUIDE.md`, `docs/PROJECT_MANAGEMENT_ROADMAP_v1.0.md`, `docs/ALPHA_100_ROLLOUT.md`, `docs/ENTERPRISE_IMPLEMENTATION_BLUEPRINT.md` | Rewrite `PRODUCTION READY` → `tested locally, pending external validation` or stage-accurate status line. Apply across 75 instances. | P1 |
| Benchmark / performance superiority | `docs/SAPE_SNR_MASTER_AUDIT_v1.md`, `docs/BIZRA_STRATEGY_DECK_2026.md`, `docs/BIZRA_TECHNICAL_BRIEF_INVESTORS.md` | Move exact SNR numbers off consumer pages; keep on Under-the-Hood with receipt link. 12 instances. | P1 |
| Financial / economic implication | `docs/BIZRA_STRATEGY_DECK_2026.md`, `docs/ALPHA_100_ROLLOUT.md` | Replace hard $ cost figures with directional. 7 instances. | **P0** |
| Sovereignty absolutism | Across docs | Scope every absolute to Ed25519+BLAKE3 receipt chain language. | P1 |
| AI capability overclaim | `docs/BIZRA_STRATEGY_DECK_2026.md`, `docs/DDAGI_CONSTITUTION_v1.1.0-FINAL.md`, `docs/BIZRA_IDENTITY_CANON.md` | Rename "DDAGI" everywhere to system-accurate language. 14 PROHIBITED instances. | **P0** |
| Vague universal / certification | `docs/PROJECT_HANDOVER.md`, `docs/ENTERPRISE_IMPLEMENTATION_BLUEPRINT.md`, `docs/KERNEL.md`, `docs/QUALITY_ASSURANCE_STRATEGY.md`, `docs/THREAT-MODEL-V3.md` | Remove false certifications. 5 PROHIBITED instances. | **P0** |

---

## 3. Organic + paid social surfaces

| Surface | Allowed today | Blocked until |
|---------|---------------|---------------|
| Organic social (A-class claims only) | ✅ hero copy, sovereignty copy, receipts copy | — |
| Paid ads | ❌ **do not run** | W-H-04, W-H-05, W-H-07, W-H-09 resolved |
| Investor deck | ✅ with caveats and receipt links | AGI-claim rewrite lands (14 instances) |
| Press release | ✅ A + B-class only | W-H-02 privacy policy live |

---

## 4. Measurement & drift prevention

| Measure | Owner |
|---------|-------|
| Re-run omni-audit monthly; diff `claims_register.json` | audit owner |
| Re-run Flywheel Kernel against refreshed artefacts; watch G-FW-003 status | audit owner |
| Watch for new PROHIBITED patterns in newly authored docs | docs owner |
| Headless-Chromium capture of live DOM (WC4 from WEBSITE_PUBLIC_CLAIMS_AUDIT) | audit engine owner |

---

## 5. Sequencing

1. **Batch A — platform-policy unblock (P0 only):** W-H-04, W-H-05, W-H-07, W-H-09; plus the 7 `Explicit cost figure` and 14 `AGI claim` instances in docs cited by site copy.
2. **Batch B — truth-debt (P1):** hero/sovereignty/receipts/Node0 copy roll-in; Under-the-Hood page created.
3. **Batch C — improvement (P2):** measurement plumbing (headless capture, drift diff, OG meta tags).
4. After Batch A, re-run the Flywheel Kernel. Expected decision: still
   `P1_TRUTH_INTEGRITY` (docs are not yet rewritten at source) — this is
   acceptable. G-FW-003 closes only when the `claims_register.json` regenerated
   from rewritten docs shows `PROHIBITED = 0 and NEEDS_REWRITE = 0`.
