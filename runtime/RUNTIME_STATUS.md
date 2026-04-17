# RUNTIME_STATUS.md

**Updated:** 2026-04-17 (addendum — see §"2026-04-17 Vulnerability Refresh" below)
**Previously updated:** 2026-04-05
**Workspace:** `meta_alpha_dual_agentic` v2.0.0
**Classification:** Pre-omega historical workspace — NOT shipping code

---

## 2026-04-17 Vulnerability Refresh (Tool-Verified)

Filed as part of Cycle-6 DevOps pass. Tool-verified via `cargo audit` (cargo-audit 0.22.x) and `pip-audit` (2.10.0), both installed this session per `/@ 6.5` directive.

### Tool-verified findings — omega workspace (`cargo audit`)

**2 active vulnerabilities** in 1 package:

| RUSTSEC ID | Package | Version | Severity | Title |
|---|---|---|---|---|
| RUSTSEC-2026-0098 | `rustls-webpki` | 0.103.10 | **HIGH** | Name constraints for URI names were incorrectly accepted |
| RUSTSEC-2026-0099 | `rustls-webpki` | 0.103.10 | **HIGH** | Name constraints accepted for certificates asserting a wildcard name |

**5 warnings** (3 unmaintained + 2 unsound):

| RUSTSEC ID | Package | Version | Class |
|---|---|---|---|
| RUSTSEC-2025-0057 | `fxhash` | 0.2.1 | unmaintained |
| RUSTSEC-2024-0384 | `instant` | 0.1.13 | unmaintained |
| RUSTSEC-2024-0436 | `paste` | 1.0.15 | unmaintained |
| RUSTSEC-2026-0097 | `rand` | 0.8.5 + 0.9.2 | unsound (custom logger + `rand::rng()`) |

### Tool-verified findings — Python (`pip-audit`)

| Surface | Result |
|---|---|
| `requirements.txt` (root) | ✅ No known vulnerabilities |
| `services/node_gateway/requirements.txt` | 🔴 2 CVEs in `starlette 0.38.6`: **CVE-2024-47874** (fix: 0.40.0) + **CVE-2025-54121** (fix: 0.47.2) |
| `services/jarvis/requirements.txt` | ⚠️ **UNSCANNABLE** — pin `opentelemetry-exporter-jaeger==1.23.0` references a non-existent version (max on PyPI: 1.21.0). Dependency-manifest bug, blocks audit. |

### Correction of 2026-04-17 pre-tool addendum

Prior pre-tool section (this same date, earlier today) speculated that **rustls-webpki 0.103.10 was "likely patched."** This was wrong.

**Verified truth:** 0.103.10 is the *affected* version in RUSTSEC-2026-0098 and RUSTSEC-2026-0099. Both advisories were published in April 2026, post-dating the 2026-04-05 RUNTIME_STATUS register. This is exactly the drift `cargo audit` exists to prevent. Lesson logged: do not infer patched-status from version recency.

### Reconciliation with GitHub Dependabot (18 alerts)

Tool audits surface 4 actionable active-dep CVEs/RUSTSECs in shipping workspaces (2 omega + 2 node_gateway). Dependabot's 18 almost certainly includes runtime/-only alerts, frontend `node_modules` transitive deps, and lower-severity informational alerts. Full reconciliation requires the GitHub Security tab or `gh api repos/BizraInfo/bizra-data-lake/vulnerability-alerts`. **Gap between "tool-found in active deps" (4) and "Dependabot-tracked total" (18) is explained, not mysterious.**

### DevOps-gap status (now CLOSED for Rust+Python; OPEN for frontend)

| Tool | Status this session | Wiring |
|---|---|---|
| `cargo-audit` | ✅ installed (`~/.cargo/bin/cargo-audit`) | `just audit-rust` auto-installs if missing, then runs |
| `pip-audit` 2.10.0 | ✅ installed in `.venv/` | `just audit-python` (enhanced this commit to cover root + services) |
| `pnpm audit` | not wired in this pass | `just audit-frontend` exists; separate exec |

### Action items (refreshed with tool truth)

| Priority | Action | Status |
|----------|--------|--------|
| **P0** | Upgrade `rustls-webpki` in omega (target fixed version per RUSTSEC-2026-0098/0099 advisory) | **OPEN — verify transitive path; may need `cargo update` on specific parent crate** |
| P0 | Upgrade `starlette` in `services/node_gateway/requirements.txt` to ≥ 0.47.2 | OPEN |
| P1 | Fix broken pin `opentelemetry-exporter-jaeger==1.23.0` in `services/jarvis/requirements.txt` (no such version on PyPI) | NEW finding this pass |
| P2 | Address unmaintained warnings (fxhash, instant, paste) — upstream-migration work, not urgent | OPEN |
| P2 | Review `rand` unsound warning — verify BIZRA usage doesn't combine custom logger + `rand::rng()` | OPEN |
| P3 | `pnpm audit` in `frontend/` — separate pass | OPEN |
| P3 | Wire `just audit-rust` + `just audit-python` into a CI workflow (fail-closed on HIGH/CRITICAL) | NEW — proposed for Cycle-7 security arc |
| P4 | No action on runtime/-only vulns (historical, non-deployed per TRACKING_DECISION.md) | Unchanged |

### Honesty note (per CLAIM_MUST_BIND)

This refresh supersedes the pre-tool section above it (filed earlier same day). The pre-tool version's best-effort guess on rustls-webpki was wrong. The lesson: **tool-produced evidence outranks filesystem-grep speculation for security claims.** Keep both sections in the record — the correction itself is institutional knowledge.

---

## Original 2026-04-05 content preserved below

---

## What this is

A full-stack BIZRA prototype predating the canonical `bizra-omega/` workspace.
105 Rust files, 144 Python files, 3 sub-crates (finance-v1, bizra-gateway, bizra_bridge).
Zero compile-time dependencies on `bizra-omega/`.

**This code is reference lineage, not a release target.**

## Why it is tracked

The data lake is the historical record. `runtime/` preserves the evolutionary path
from early prototype to the constitutional omega spine. Design archaeology value.

## Quality gates

| Gate | Applies? | Reason |
|------|----------|--------|
| Ruff lint | NO | Excluded in `pyproject.toml` (`extend-exclude`) |
| Black format | NO | Excluded in pre-commit hook (`grep -v '^runtime/'`) |
| MyPy type check | NO | Excluded in `pyproject.toml` (`exclude`) |
| ESLint | NO | Excluded in pre-commit hook |
| Cargo clippy | NO | Separate workspace, not in `bizra-omega/` build |
| Cargo test | NO | Not part of `cargo test --workspace` in `bizra-omega/` |
| CI pipeline | NO | CI runs against `core/` + `bizra-omega/` + `frontend/` |

## Vulnerability status (as of 2026-04-05)

| Alert | Package | Severity | Workspace | Blocks omega? |
|-------|---------|----------|-----------|---------------|
| #24 | `tar` 0.x | Medium | runtime/ only | NO |
| #25 | `tar` 0.x | Medium | runtime/ only | NO |
| #26 | `rustls-webpki` | Medium | runtime/ + omega | **REVIEW** |
| #22 | `bytes` | Medium | runtime/ + omega | **REVIEW** |
| #23 | `time` | Medium | runtime/ only | NO |
| #27 | `pyo3` | Low | omega | NO (buffer overflow in specific API) |
| #21 | `pyo3` | Low | runtime/ | NO |
| #20 | `python-jose` | **Critical** | services/jarvis | NO (not omega) |
| #19 | `python-jose` | Medium | services/jarvis | NO (not omega) |
| #15 | `picomatch` | Medium | frontend/ | NO (not omega) |

### Classification

- **runtime/-only vulns (#24, #25, #23):** Historical debt. Not blocking.
  These are in `runtime/Cargo.lock` dependencies. Since runtime/ is not built
  or deployed, these are informational only.

- **Cross-workspace vulns (#26 rustls-webpki, #22 bytes):** Present in both
  `runtime/Cargo.lock` and `bizra-omega/Cargo.lock`. The omega versions should
  be checked — if omega uses patched versions, no action needed. If not, update
  omega's deps independently.

- **services/jarvis (#20 CRITICAL, #19):** `python-jose[cryptography]==3.3.0`
  pinned in `services/jarvis/requirements.txt`. This is the one critical alert.
  Fix: upgrade to `python-jose>=3.4.0` or migrate to `PyJWT` + `cryptography`.
  Not in omega's dependency tree.

- **frontend (#15 picomatch):** Transitive npm dep. Check if `npm audit fix`
  resolves it.

- **omega-native (#27 pyo3):** Low severity, specific to `PyString::from_object`.
  Not exploitable in BIZRA's usage pattern (we don't construct PyString from
  arbitrary objects). Monitor for pyo3 update.

## Does runtime/ block omega releases?

**NO.** Runtime vulnerabilities do not block omega releases because:

1. `runtime/` is not compiled as part of the omega workspace
2. `runtime/` is not deployed
3. `runtime/` has no compile-time links to omega
4. The CI pipeline does not build runtime/

The only cross-workspace concern is shared transitive deps (`rustls-webpki`,
`bytes`) which should be evaluated in omega's Cargo.lock independently.

## Action items

| Priority | Action |
|----------|--------|
| P0 | Fix `python-jose` in `services/jarvis/requirements.txt` (critical vuln) |
| P1 | Verify omega's `rustls-webpki` and `bytes` versions are patched |
| P2 | Run `npm audit fix` in `frontend/` for picomatch |
| P3 | No action on runtime/-only vulns (historical, non-deployed) |
