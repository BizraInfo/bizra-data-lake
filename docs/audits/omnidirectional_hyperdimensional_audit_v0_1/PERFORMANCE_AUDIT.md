# Performance Audit — BIZRA v0.1

**Scope:** separate *measured* from *simulated* from *target* from *unverified / public-site-claim* performance metrics. Identify bottlenecks and instrumentation gaps.

**Discipline:** "If it has not been measured, phrase as a direction. If it has been measured, cite the receipt. If it is uncertain, mark it uncertain." (Brand canon §5 / `CLAIM_DISCIPLINE.md`.)

---

## 1. Measured metrics (receipt-backed)

| Metric | Value | Receipt / source |
|---|---|---|
| BIZRA Node0 genesis seal creation | Deterministic / time-bounded | `bizra-omega/bizra-core/src/genesis_seal.rs` + 246-test parity |
| Receipt chain append throughput | Not measured in this audit | — (benchmark receipt not located) |
| Cognitive Foundry audit engine duration | 9.86 s on ~2 000-file evidence aperture | `artifacts/audit_summary.json` |
| Foundry review pipeline 4-stage pass | ~seconds end-to-end | Artifact set in `tools/cognitive_foundry/.../output/20260424T000948Z_.../` |

## 2. Simulated metrics

| Metric | Scope | Note |
|---|---|---|
| Canonical spearpoint replay | PR #49 (row 4) — 38/38 tests green | Simulated at test-harness level. Not a field measurement. |
| Mission state machine transitions | ReceiptStateMachine unit tests | Deterministic, in-memory — not a field throughput claim. |

## 3. Target metrics (declared, not yet measured / published)

From `core/integration/constants.py` (authoritative) + Node0 doctrine:

| Target | Value |
|---|---|
| `IHSAN_THRESHOLD` | 0.95 |
| `IHSAN_THRESHOLD_CI` | 0.90 |
| `STRICT_IHSAN_THRESHOLD` | 0.99 |
| `RUNTIME_IHSAN_THRESHOLD` | 1.0 |
| `SNR_THRESHOLD` | 0.85 (min) |
| `SNR_THRESHOLD_T1_HIGH` | 0.95 |
| `SNR_THRESHOLD_T0_ELITE` | 0.98 |
| `ADL_GINI_THRESHOLD` | 0.35 (operational gate) |
| `CONSTITUTIONAL_GINI_THRESHOLD` | 0.45 |
| `CI coverage floor` | 65% |

These are **targets / gates**, not field-measured user-visible metrics. Suitable for technical audience + internal docs. Not suitable verbatim in consumer hero.

## 4. Unverified / public-site claims (must downgrade or receipt)

From `artifacts/website_claims.json` + operator pre-check:

| Public claim | Class | Required action |
|---|---|---|
| `SNR 0.974` | NEEDS_REWRITE | Publish benchmark receipt or remove. |
| `cost per action $0.10 → $0.008` | NEEDS_REWRITE | Remove; directional reframe. |
| `8 072 verified tests` | PROOF_REQUIRED | Auto-link to CI with timestamp + commit hash, or soften. |
| `100% pass rate` | NEEDS_REWRITE | Replace with policy claim ("CI must pass before merge"). |
| `73 of 100 nodes remaining` | NEEDS_REWRITE | Live counter or remove. |
| `Ed25519 receipt signatures` | PROOF_REQUIRED | Keep in dev docs; remove from consumer hero. |
| `Ihsan Gate ≥ 0.95` | PROOF_REQUIRED | Contextualize as internal-gate value. |

## 5. Bottlenecks (architectural)

| Bottleneck | Observable signal |
|---|---|
| Receipt-chain serialization under concurrent append | Rust workspace relies on `blake3+rayon`; throughput under high concurrency not published. |
| Z3 solver inclusion in FATE gates | Heavy crate; `sudo apt install libz3-dev` prerequisite; can dominate build + gate-decision time. |
| `cargo test --workspace` contaminates `action_receipts.jsonl` (memory `project_cargo_test_audit_contamination.md`) | Operational bottleneck — a known workaround exists (snapshot + restore). |
| Panic surface (806 Rust `.unwrap()`) | Performance under error conditions is indeterminate on hot paths. |

## 6. Instrumentation gaps

- **No published receipt-throughput benchmark.** Architecture says receipts chain on every visible effect; there's no measured p50/p99 appendLatency or QPS published.
- **No public bench receipt for any of the hero numeric claims** (SNR, pass rate, cost, test count).
- **No observability stack published.** Privacy tension: "no telemetry" in public copy vs. operators needing SLO monitoring. Defensible framing exists — not yet written into docs.
- **No SBOM** → supply-chain trust inferred, not measured (see `DEPENDENCY_AUDIT.md`).

## 7. Performance-debt summary

| # | Debt | Impact | Action |
|---|---|---|---|
| PD1 | Publish a benchmark receipt for at least one hero numeric claim | HIGH (unblocks public quantitative copy) | Pick one (test count is easiest) + publish JSON receipt |
| PD2 | Hot-path `.unwrap()` audit in receipt/mission crates | MEDIUM | Replace with `Result`-returning paths + graceful degradation |
| PD3 | Self-hosted, privacy-respecting observability statement | MEDIUM | Documents what's measured where, enabling honest "no telemetry" claim |
| PD4 | Receipt-append throughput benchmark | MEDIUM | Microbench + commit hash + JSON receipt |
| PD5 | SBOM in CI (supply-chain "measured") | MEDIUM | Generate spdx.json or cdx.json on release |

---

## One-line bottom line

**Public marketing copy makes performance claims the internal instrumentation cannot yet back.** Either the instrumentation catches up (receipts + benchmarks + SBOM), or the copy downgrades to directional language. This is the claim-discipline line crossing the engineering line — which is exactly where Ihsan applies.
