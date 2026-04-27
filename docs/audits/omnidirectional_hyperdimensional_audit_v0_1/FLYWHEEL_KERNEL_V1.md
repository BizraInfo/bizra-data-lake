# BIZRA Autonomous Flywheel Kernel v1

**Date:** 2026-04-24 GST
**Status:** Implemented as isolated stdlib tooling under `tools/audit/flywheel_kernel/`; verified locally.

## Purpose

The Flywheel Kernel turns an audit result into an executable improvement loop:

```text
Signal -> Root Cause -> Fix -> Test -> Validate -> Document -> Encode -> Repeat
```

It is deliberately non-destructive. It reads audit artifacts, evaluates guards,
chooses the next priority, and emits a JSON report. It does not mutate source,
publish claims, rotate secrets, or run network actions.

## Components

| Component | File | Role |
|---|---|---|
| Pattern registry | `tools/audit/flywheel_kernel/patterns.json` | Encodes reusable lessons from prior loops. |
| State loader | `tools/audit/flywheel_kernel/kernel.py` | Reads `audit_summary`, secrets, claims, code risks, and dependencies. |
| Pre-execution guards | `kernel.py` | Blocks or warns on missing artifacts, secrets, claim debt, dependency gaps, and runtime hardening risks. |
| Adaptive priority engine | `kernel.py` | Shifts focus from P0 secrets to P1 truth integrity when secrets are clear. |
| Trigger detector | `kernel.py` | Maps changed paths to patterns that require re-audit. |
| Regression tests | `tests/tools/test_flywheel_kernel.py` | Locks priority shift and trigger behavior. |

## Priority Ladder

1. `P-BOOTSTRAP-AUDIT`: Generate audit artifacts if required state is missing.
2. `P0_SECRET_TRIAGE`: Secrets remain highest immediate blast-radius risk.
3. `P1_TRUTH_INTEGRITY`: Public claims become the next constraint once secrets are zero.
4. `P2_SUPPLY_CHAIN_TRUST`: Lockfiles, SBOMs, and dependency policy after claim debt.
5. `P3_RUNTIME_HARDENING`: Panic, shell execution, and dynamic execution surfaces.
6. `P4_MONITOR_AND_RELOOP`: No dominant blocker; schedule the next audit.

## CLI

```bash
python -m tools.audit.flywheel_kernel.kernel \
  --audit-dir docs/audits/omnidirectional_hyperdimensional_audit_v0_1/artifacts \
  --changed-path runtime/core/autoconfig.py \
  --changed-path docs/brand/public_launch_readiness/PUBLIC_CLAIMS_REGISTER.md \
  --out /tmp/bizra-flywheel-report.json
```

Use `--strict` to return exit code `2` when any guard is `BLOCK`.

## Expected Current Decision

Given the latest hardened audit state:

```text
secrets = 0
PROHIBITED = 20
NEEDS_REWRITE = 94
PROOF_REQUIRED = 367
```

the kernel should choose:

```text
P1_TRUTH_INTEGRITY
```

That means the flywheel no longer spends its next cycle on secret triage. It
automatically shifts to public claim truth alignment.

## Safety Boundary

The kernel is autonomous in diagnosis and prioritization only. Implementation
still requires an operator or a separate authorized worker path. This boundary
prevents the system from silently rewriting public claims, rotating credentials,
or changing runtime behavior without review.

## Relationship to `tools/execution_flywheel/`

This kernel (`tools/audit/flywheel_kernel/`) is the **audit-state-driven** lane:
it pulls from the omni-audit artefact tree and ranks system-level
constraints. The sibling package `tools/execution_flywheel/` is the
**action-context-driven** lane: it guards individual edit decisions. They
share the "observable evidence only" discipline but do not share data
structures. Both are advisory-only, stdlib-only, tools-layer only.
