# SNR Signal / Noise Register — BIZRA v0.1

**Source:** `artifacts/snr_findings.json`. Counts: **9 signal · 7 watchlist · 1 noise.**

**Signal = evidence-backed, actionable, changes architecture/risk/readiness.**
**Noise = low-signal or currently non-blocking after evidence refresh.**
**Watchlist = promising or real, but needs hot-path validation before escalation.**

---

## SIGNAL (9)

| # | Finding (summary) | Evidence | Signal | Noise | Actionable | Conf | Next action |
|---|---|---|---:|---:|---|---:|---|
| 1 | Workspace without Cargo.lock: `filedfs/Cargo.toml` | `artifacts/dependencies.json` | 0.75 | 0.30 | ✅ | 0.75 | Add/pin lockfile; establish SBOM generation in CI |
| 2 | Workspace without Cargo.lock: `desktop/rust/Cargo.toml` | `artifacts/dependencies.json` | 0.75 | 0.30 | ✅ | 0.75 | Add/pin lockfile; establish SBOM generation in CI |
| 3 | SBOM artifact not located in repo | `artifacts/dependencies.json` | 0.75 | 0.30 | ✅ | 0.75 | Emit SBOM on release |
| 4 | Python `subprocess(shell=True)` x1 — command-injection surface | `artifacts/code_risks.json` | 0.70 | 0.30 | ✅ | 0.75 | Remove or tightly justify; add rule-level budget |
| 5 | 20 PROHIBITED-class claim patterns (AGI, first-in-world, etc.) in scanned docs | `artifacts/claims_register.json` | 0.90 | 0.30 | ✅ | 0.75 | Rewrite or remove each before any public reuse |
| 6 | 94 NEEDS_REWRITE claim patterns (production-ready, exact cost, SNR, 100% pass, scarcity) | `artifacts/claims_register.json` | 0.88 | 0.30 | ✅ | 0.75 | Remove from hero; move to receipt-backed page |
| 7 | 367 PROOF_REQUIRED claim patterns (Ed25519, BLAKE3, Ihsan, no-telemetry, post-quantum) | `artifacts/claims_register.json` | 0.75 | 0.30 | ✅ | 0.75 | Publish receipt per claim OR soften to directional wording |
| 8 | bizra.ai is SPA; non-JS fetchers see shell only | `artifacts/website_claims.json` | 0.80 | 0.30 | ✅ | 0.75 | Add OG meta tags in shell; consider SSR / prerender |
| 9 | bizra.info redirects to bizra.ai but non-JS capture still sees shell-only destination | `artifacts/website_claims.json` | 0.80 | 0.30 | ✅ | 0.75 | Keep redirect; improve destination shell metadata |

## WATCHLIST (7)

| # | Finding | Evidence | Signal | Noise | Next action |
|---|---|---|---:|---:|---|
| W1 | 806 Rust `.unwrap()` occurrences — panic surface on hot paths | `artifacts/code_risks.json` | 0.45 | 0.30 | Triage receipt / mission hot paths; raise severity only where panic bypasses receipts |
| W2 | Python broad `except Exception` x126 — may mask errors | `artifacts/code_risks.json` | 0.45 | 0.30 | Sweep; tighten exception classes |
| W3 | Rust TODO/FIXME x1 — tech-debt signal | `artifacts/code_risks.json` | 0.45 | 0.30 | Convert to tracked issue or remove |
| W4 | Python TODO/FIXME x4 — tech-debt signal | `artifacts/code_risks.json` | 0.45 | 0.30 | Convert to tracked issue or remove |
| W5 | bizra.info 302 → bizra.ai confirmed — no split claim surface | `artifacts/website_claims.json` | 0.60 | 0.30 | None — keep brand-defense redirect |
| W6 | 132 doctrine-class documents present (good, but dedup needed) | `artifacts/evidence_index.json` | 0.55 | 0.30 | Index and deduplicate; ingestion gate is the single forward path |
| W7 | `CLAUDE.md` exists and stable — agent contract surface | `CLAUDE.md` | 0.60 | 0.30 | Review quarterly; keep in sync with module decomposition |

## NOISE (1)

| # | Finding | Evidence | Signal | Noise | Next action |
|---|---|---|---:|---:|---|
| N1 | No matches from secret-pattern scanner in configured scan roots | `artifacts/secret_findings.json` | 0.20 | 0.40 | Expand scan roots / add CI gate for ongoing coverage |

## Cross-cut: findings by domain

| Domain | Count |
|---|---:|
| PUBLIC_CLAIMS | 6 |
| CODE_QUALITY | 5 |
| DEPENDENCY | 3 |
| SECURITY | 1 |
| DOCUMENTATION | 2 |

## How to read this register

- Items with **Signal ≥ 0.65** and **Actionable = ✅** are the first tranche to close. These are the 9 signal items above.
- Items with **Signal 0.35–0.65** are watchlist — re-check in next run.
- Items with **Signal ≤ 0.35** are noise — ignored by priority but logged for completeness.

## Re-run guidance

Re-run quarterly and diff the register against previous quarter. Specifically watch:

- Any new PROHIBITED or NEEDS_REWRITE count increase → claim-discipline regression.
- Any new secret-finding → immediate triage.
- Any lockfile gap reappearing after it was closed → supply-chain regression.
- Any decrease in signal count with no corresponding action → risk that findings are being suppressed.
