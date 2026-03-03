# Phase 56: Security Hardening — Full Workspace Audit Remediation

> Standing on Giants: Rescorla (TLS/DTLS) · Bernstein (NaCl/Ed25519) · Saltzer & Schroeder (fail-closed, 1975) · OWASP (Top 10) · Tanenbaum (token bucket, 1981)

## Scope

Remediate all 15 findings from the 2026-03-01 full-workspace security audit.
Four spec modules, grouped by blast radius:

| Module | File | Findings | Domain |
|--------|------|----------|--------|
| 01 | `phase_56_01_critical_transport_bridges.md` | F1, F2, F12 | Transport auth + WS bridges |
| 02 | `phase_56_02_high_execution_auth.md` | F3–F7, F9, F10 | ZPK, RLM, auth middleware, rate limiting, command safety |
| 03 | `phase_56_03_high_infra_secrets.md` | F8, F11 | K8s secrets, infra defaults, service binding |
| 04 | `phase_56_04_medium_frontend_ops.md` | F13, F14, F15 | Service worker, CSP, RBAC, image tags |

## Finding Index

| ID | Severity | Title | File(s) |
|----|----------|-------|---------|
| F1 | Critical | Transport MITM — no peer identity binding | `core/federation/secure_transport.py` |
| F2 | Critical | Unauthenticated WS bridge (bizra-bridge.mjs) | `filedfs/bizra-bridge.mjs` |
| F3 | High | ZPK manifest bypass — unsigned policy fields | `core/zpk/kernel.py` |
| F4 | High | ZPK sync worker timeout gap | `core/zpk/kernel.py` |
| F5 | High | RLM sandbox no execution time limit | `core/inference/rlm_bridge.py` |
| F6 | High | Auth middleware fail-open | `core/auth/middleware.py` |
| F7 | High | API identity endpoints unauthenticated | `bizra-omega/bizra-api/src/lib.rs` |
| F8 | High | K8s NODE_SECRET not consumed + default perms | `bizra-omega/k8s/deployment.yaml`, `bizra-api/src/main.rs` |
| F9 | High | Rate limiter ineffective against bursts | `bizra-omega/bizra-api/src/middleware/rate_limit.rs` |
| F10 | High | Command safety bypass via whitespace | `core/benchmark/guardrails.py`, `core/sovereign/tiered_verification.py` |
| F11 | High | Infra defaults expose services without auth | `deploy/node0/node0-manifest.yaml`, `deploy/elite-compose.yaml`, systemd units |
| F12 | High | Localhost bridge drive-by WS + protocol injection | `filedfs/bridge.mjs`, `filedfs/useBizraNode.js` |
| F13 | Medium | Service worker caches dynamic responses | `filedfs/service-worker.js` |
| F14 | Medium | Missing CSP in frontend HTML | `filedfs/index.html` |
| F15 | Medium | Mutable :latest tags + broad RBAC | `bizra-omega/docker-compose.yml`, `deploy/k8s/base/rbac.yaml` |

## Constraints

- All thresholds from `core/integration/constants.py` (single source of truth)
- No hardcoded secrets — env vars or secret managers only
- Each fix must have ≥1 TDD anchor (test name + assertion)
- Patches must not break existing CI (9 green gates)
- Backward compatibility: existing API clients must not break without migration path

## Acceptance Criteria (Phase-Level)

1. All 15 findings have patches merged to `main`
2. `pytest tests/` passes — no regressions
3. `cargo test --workspace` passes — no regressions
4. No new `bandit` or `cargo-audit` findings introduced
5. CI remains 9/9 green gates
