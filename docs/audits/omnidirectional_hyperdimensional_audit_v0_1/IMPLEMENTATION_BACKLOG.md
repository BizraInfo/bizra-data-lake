# Implementation Backlog — BIZRA v0.1

Grouped by P-level. P0 = security + truth; P1 = Node0 activation; P2 = website / public claims; P3 = media / organic launch; P4 = paid ads; P5 = Genesis 100.

---

## P0 — Security and truth blockers

| ID | Title | Effort | Owner | Notes |
|---|---|---|---|---|
| P0.1 | Maintain zero secret-pattern findings; keep scanner gate current | XS–S | operator | Superseded by P0+1 hardening; current `secret_findings.json` count is 0. Keep scanner in pre-commit/CI path rather than re-opening old triage. |
| P0.2 | Pre-commit secret-pattern scanner gate | S | repo-ops | Wire `tools/audit/omni_audit/secret_pattern_scanner.py` into `pre-commit` |
| P0.3 | Verify PR #50 (full-body receipt signature) is merged | XS | operator | If not merged, land it |
| P0.4 | Remove Python `subprocess(shell=True)` + lint rule | S | backend lead | 1 occurrence |
| P0.5 | Log-of-event vs. real-key classification discipline for `.claude/logs/` | S | operator | Document |

## P1 — Node0 activation blockers

| ID | Title | Effort | Owner | Notes |
|---|---|---|---|---|
| P1.1 | Hot-path `.unwrap()` audit in receipt / mission crates | M | runtime lead | See `ERROR_HANDLING_AUDIT.md §9` |
| P1.2 | PAT/SAT gateway wiring completion | M | runtime lead | Scoreboard row |
| P1.3 | Canon Store Ingestion Gate ADR (spec only, no code) | M | architecture lead | Separate typed-auth lane |
| P1.4 | Node-onboarding runbook (install → seal → join URP) | M | operator | Tier D blocker |
| P1.5 | Minimum-hardware profile | S | runtime lead | Tier D blocker |
| P1.6 | Operator kill-switch doc | S | operator | Tier D + paid-ads blocker |
| P1.7 | 2 `panic!()` sites audit | S | runtime lead | Triage legitimate fast-fail vs. uncontrolled panic; runtime work remains blocked under WAIT until Phase 2 unlocks |

## P2 — Website / public-claims blockers

| ID | Title | Effort | Owner | Notes |
|---|---|---|---|---|
| P2.1 | Remove / receipt-ify bizra.ai C4 (cost $0.10→$0.008) | S | operator + web lead | `WEBSITE_PUBLIC_CLAIMS_AUDIT.md §2` |
| P2.2 | Remove / receipt-ify bizra.ai C5 (SNR 0.974) | S | operator + web lead | same |
| P2.3 | Replace C7 (100% pass rate) with policy claim | S | operator + web lead | same |
| P2.4 | Remove or wire live counter for C9 (73/100 nodes) | S–M | web lead | same |
| P2.5 | Consumer hero: move Ed25519 to dev docs; use claim-safe hero | S | web lead | `CLAIM_SAFE_LAUNCH_COPY.md §4` |
| P2.6 | Publish privacy policy OR soften C1/C2 | M | operator | Legal-adjacent review |
| P2.7 | Add OG meta tags to SPA shell HTML | S | web lead | Link-preview quality |
| P2.8 | Headless-Chromium DOM capture script (audit evidence) | S | audit-tooling | Replace pre-check skeleton |
| P2.9 | Arabic parity pass for any public copy change | S | operator + Arabic reviewer | Every change |

## P3 — Media / organic launch blockers

| ID | Title | Effort | Owner | Notes |
|---|---|---|---|---|
| P3.1 | Visual QA of 12 `rendered_concepts/` PNGs | S | operator | Kit's README warning |
| P3.2 | Visual QA of 11 `ready_to_post/` rasters | S | operator | — |
| P3.3 | Arabic reviewer pass on `CLAIM_SAFE_LAUNCH_COPY.md §1–§4` | S | operator + Arabic reviewer | — |
| P3.4 | Render SVG templates to PNGs per platform size | S | creative lead | — |
| P3.5 | Claim handles on X / LinkedIn / IG / YouTube / Threads | S | operator | — |
| P3.6 | Upload avatar + cover + bio (silent Phase 1) | S | operator | — |
| P3.7 | Phase 2 launch moment (coordinated post) | S | operator | — |
| P3.8 | Phase 3 first-week support posts | M | operator | 1 h / day, 5 days |

## P4 — Paid-ads blockers

| ID | Title | Effort | Owner | Notes |
|---|---|---|---|---|
| P4.1 | ADS_READINESS_CHECKLIST all-green | varies | operator | Depends on P2 + P3 |
| P4.2 | Platform ad-account + 2FA + billing + kill-switch | M | operator | — |
| P4.3 | UTM conventions defined | XS | operator | — |
| P4.4 | Ad concept selection + budget envelope | S | operator | After organic telemetry |
| P4.5 | First campaign launch + 24h monitoring | S–M | operator | — |
| P4.6 | Platform-policy pre-review for Arabic targeting | S | operator | MENA specifics |

## P5 — Genesis 100 blockers

| ID | Title | Effort | Owner | Notes |
|---|---|---|---|---|
| P5.1 | `docs/gtm/node0_activation_go_to_market_v0_1/` authored | L | operator | Separate lane |
| P5.2 | Multi-peer federation benchmark (N=10/100/1000) | L | runtime lead | Separate lane |
| P5.3 | Cost-model receipt publication | L | architecture lead | Separate lane |
| P5.4 | SBOM generation on every release | M | repo-ops | Adds supply-chain attestation |
| P5.5 | `cargo-deny` license + advisory gate | M | repo-ops | — |
| P5.6 | Python `uv pip compile` lockfile adoption | M | repo-ops | Reproducible Python builds |
| P5.7 | Cargo.lock for `filedfs/` + `desktop/rust/` | S | repo-ops | — |
| P5.8 | Node-onboarding runbook field-tested with first external human | M | operator | Validates P1.4 |

---

## Scheduling heuristic

- **P0 always first.** Security + truth drift does not wait.
- **P1 + P2 in parallel.** Independent owners, independent blockers.
- **P3 after P2.** Organic launch traffic lands on the site; site must be clean first.
- **P4 after P3 telemetry.** Ad copy should be informed by organic audience reaction.
- **P5 is a post-activation lane.** Don't block activation on Genesis-100 planning.

## Stop line

After P0 + P1.4 + P2 closes and organic launch ships: **land the plane**. Don't stack P4/P5 work onto the same push. Re-run this audit quarterly.
