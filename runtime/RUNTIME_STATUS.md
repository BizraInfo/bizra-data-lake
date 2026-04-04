# RUNTIME_STATUS.md

**Updated:** 2026-04-05
**Workspace:** `meta_alpha_dual_agentic` v2.0.0
**Classification:** Pre-omega historical workspace — NOT shipping code

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
