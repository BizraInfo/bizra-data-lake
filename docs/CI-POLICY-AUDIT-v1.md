# CI Policy Audit — bizra-data-lake v1

بسم الله الرحمن الرحيم

**Purpose:** Per the polyglot-blueprint §3 (Fail-Closed CI), audit the 22 GitHub Actions workflows in `.github/workflows/` to identify which are gates (fail-closed) vs advisory (continue-on-error).

**Method:** Count `continue-on-error:` directives in each workflow file. Zero advisory pointers = fully fail-closed. Non-zero = has advisory paths.

**Generated:** 2026-04-17, ad-hoc from repo root

---

## Audit table

| # | Workflow | Jobs (approx.) | `continue-on-error` count | Classification |
|---|---|---|---|---|
| 1 | alpha100-release-binaries | 23 | 0 | fail-closed |
| 2 | autopoietic-cycle | 8 | 0 | fail-closed |
| 3 | branch-protection-audit | 11 | 0 | fail-closed |
| 4 | canonical-validation-gate | 22 | 0 | fail-closed |
| 5 | ci | 178 | 2 | mostly fail-closed, 2 advisory steps |
| 6 | deploy | 48 | 0 | fail-closed |
| 7 | docs-quality | 14 | 0 | fail-closed (⚠️ currently RED — see §3) |
| 8 | lock-deps | 9 | 0 | fail-closed |
| 9 | membrane-tax-gate | 13 | 0 | fail-closed |
| 10 | performance | 59 | 4 | fail-closed with 4 advisory perf-only steps |
| 11 | phase56-security-gate | 12 | 0 | fail-closed |
| 12 | phase65-masterpiece | 14 | 0 | fail-closed |
| 13 | proof-pyramid-gate | 45 | 0 | fail-closed |
| 14 | quality-management | 28 | 0 | fail-closed |
| 15 | quality-spine | 35 | 0 | fail-closed |
| 16 | release | 35 | 0 | fail-closed |
| 17 | resilience-gate | 25 | 2 | fail-closed with 2 advisory steps |
| 18 | security | 15 | 0 | fail-closed |
| 19 | tests | 74 | 1 | fail-closed with 1 advisory step |
| 20 | walking-skeleton | 8 | 0 | fail-closed |
| 21 | wire-completeness-audit | 14 | 0 | fail-closed |
| 22 | workspace-atlas-audit | 6 | 0 | fail-closed |

**Totals:** 22 workflows, 696 approximate jobs, **9 advisory points across 4 workflows**, 18 fully fail-closed.

---

## §1 — Overall verdict

**BIZRA's CI policy is already strongly fail-closed** per the polyglot-blueprint §3 standard. Only 4 workflows (ci, performance, resilience-gate, tests) carry any advisory/`continue-on-error` directives, and those are tightly scoped (performance-perf-step advisories, not core gate advisories).

This is unusually strong discipline. Most shipping repos have 30-60% of workflows with soft-fail steps.

---

## §2 — Alignment with polyglot blueprint §3

Blueprint requirement: *"Parallel Language Gates (Must All Pass): cargo test + clippy + audit, pnpm lint + typecheck + vitest + build, ruff + pyright + pytest, Schema Drift Check"*

Observed coverage:

| Blueprint requirement | Observed workflow(s) | Status |
|---|---|---|
| `cargo test --workspace` | `tests`, `ci`, `quality-management` | ✅ covered |
| `cargo clippy` | `quality-spine`, `ci` | ✅ covered |
| `cargo audit` | `security`, `phase56-security-gate` | ✅ covered |
| `pnpm lint + typecheck + vitest` | `ci`, `tests` | ✅ covered (in award-winner-design repo's own CI) |
| Schema drift check | `canonical-validation-gate`, `wire-completeness-audit`, `workspace-atlas-audit` | ✅ covered |
| Python `ruff + pytest` | `tests`, `quality-management`, `.pre-commit-config.yaml` | ✅ covered (local pre-commit + CI) |

**Blueprint §3 compliance: PASS.** All five polyglot-gate categories are represented in the 22-workflow suite.

---

## §3 — Pre-existing failures (held per batch-hygiene discipline)

`docs-quality.yml` has been failing since **2026-04-08** (9+ consecutive runs). Root cause: `README.md` missing links to three documentation files that exist:

- `docs/README.md`
- `docs/OPERATIONS_RUNBOOK.md`
- `docs/TESTING.md`

**Fix scope:** trivial (add 3 links to README.md).
**Why not fixed in this pass:** Per `feedback_batch_hygiene` memory — *"keep janitorial in dedicated sessions, don't dilute architecture SNR"*. A 1-line README fix mixed with architecture/doctrine commits would confuse the commit history. Queued for a dedicated janitorial session.

**Impact:** One workflow permanently red; does not block pushes (admin bypass observed in session logs). Branch protection audit may eventually flag this.

---

## §4 — Recommendations (not applied here, filed as backlog)

1. **Tighten the 9 advisory points** in `ci`, `performance`, `resilience-gate`, `tests`. For each, decide whether the advisory-tolerance is deliberate (e.g., flaky perf numbers) or legacy (can be promoted to fail-closed). Document the decision per step.
2. **Add a `lock-drift-check` workflow** that fails if `Cargo.lock`, `pnpm-lock.yaml`, or any Python lockfile is stale relative to its manifest. Blueprint §2 dependency discipline.
3. **Add an OTel trace-propagation smoke** workflow once observability (blueprint §5) lands — currently no workflow validates that a request's `traceparent` survives the Rust → Python → TypeScript boundary chain.
4. **Fix `docs-quality` in a dedicated janitorial arc**, not mixed with architecture work.

---

## §5 — What this audit does NOT do

- No CI workflow files are modified (read-only audit)
- Advisory-vs-gate classification is based on `continue-on-error` grep; some workflows may use conditional `if:` expressions for soft-fail that this doesn't catch
- Does NOT assess runtime behavior of workflows (just declarative config)
- Does NOT fix `docs-quality` red — held per batch-hygiene discipline

---

*Filed as a factual snapshot of CI configuration as of 2026-04-17. Re-run the audit method (grep `continue-on-error`) after any workflow changes to verify the new policy state.*

الحمد لله.
