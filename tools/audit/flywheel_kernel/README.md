# BIZRA Autonomous Flywheel Kernel v1

Non-destructive, stdlib-only engineering kernel that converts omni-audit
artefacts into a deterministic next-action report:

```
Signal → Root Cause → Fix → Test → Validate → Document → Encode → Repeat
```

The kernel reads `audit_summary.json`, `secret_findings.json`,
`claims_register.json`, `code_risks.json`, `dependencies.json`, evaluates five
guards, ranks the next constraint, and emits a machine-readable JSON report.
It **never** mutates source, publishes, rotates secrets, calls the network, or
touches git state.

## Components

- `patterns.json` — pattern registry (FW-P001 … FW-P006)
- `kernel.py` — state loader + guard engine + priority engine + trigger
  detector + CLI
- see `docs/audits/omnidirectional_hyperdimensional_audit_v0_1/FLYWHEEL_KERNEL_V1.md`
  for the design doc

## CLI

```bash
python3 -m tools.audit.flywheel_kernel.kernel \
  --audit-dir docs/audits/omnidirectional_hyperdimensional_audit_v0_1/artifacts \
  --changed-path runtime/core/autoconfig.py \
  --changed-path docs/brand/public_launch_readiness/PUBLIC_CLAIMS_REGISTER.md \
  --out /tmp/bizra-flywheel-report.json
```

Strict mode (exit 2 when any guard is `BLOCK`):

```bash
python3 -m tools.audit.flywheel_kernel.kernel \
  --audit-dir docs/audits/omnidirectional_hyperdimensional_audit_v0_1/artifacts \
  --strict
```

## Tests

```bash
.venv/bin/python -m pytest tests/tools/test_flywheel_kernel.py -q
```

## Priority ladder

| ID | Fires when |
|----|------------|
| `P-BOOTSTRAP-AUDIT` | Required audit artefacts are missing |
| `P0_SECRET_TRIAGE` | `secret_count > 0` |
| `P1_TRUTH_INTEGRITY` | Secrets zero + public claim debt (PROHIBITED/NEEDS_REWRITE/PROOF_REQUIRED) |
| `P2_SUPPLY_CHAIN_TRUST` | Claims clean + dependency gaps remain |
| `P3_RUNTIME_HARDENING` | Supply clean + runtime panic/shell/unwrap findings |
| `P4_MONITOR_AND_RELOOP` | No dominant blocker |

## Safety boundary

Autonomous up to diagnosis and prioritization. Every mutation (code change,
claim rewrite, credential rotation, publication, runtime/canon change) still
requires operator approval.
