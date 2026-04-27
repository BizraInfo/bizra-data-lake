# GO / NO-GO After P1 Truth-Integrity Closure

**Date:** 2026-04-24 GST
**Scope:** The decision criteria the operator applies after the
implementation lane has worked through `WEBSITE_PATCH_PLAN.md` and
`SAFE_REWRITE_PACK.md`. This pack itself does not claim to have closed
P1; it only stages the close.

---

## Entry state

- Flywheel Kernel v1 against refreshed default artefacts returns:
  `priority_id = P1_TRUTH_INTEGRITY`, `blocked_by = [G-FW-003]`.
- PROHIBITED claims in docs: **20**.
- NEEDS_REWRITE claims in docs: **94**.
- PROOF_REQUIRED claims in docs: **367** (backlogged, not required for P1
  exit).
- Live site hero carries C4/C5/C7/C9 (the four platform-policy blockers).

## Exit states

### GO — promote to `P2_SUPPLY_CHAIN_TRUST`

All of the following must be true on a re-run of the omni audit:

| # | Criterion | Measured by |
|---|-----------|-------------|
| 1 | PROHIBITED count is 0 on internal docs | refreshed `claims_register.json` |
| 2 | NEEDS_REWRITE count is ≤ 10 on internal docs | refreshed `claims_register.json` (residuals only) |
| 3 | Live hero no longer contains C4, C5, C7, C9 | operator pre-check OR headless capture |
| 4 | `/privacy` page published (blocks W-H-02) | live URL + content review |
| 5 | Under-the-Hood sub-page exists, every number links to a receipt | spot check 3 numbers |
| 6 | Flywheel Kernel returns `G-FW-003: PASS` or at worst `WARN` on refreshed artefacts | `flywheel_kernel` CLI |
| 7 | Genesis 100 counter is either removed from hero or wired to a live source | visual check |

Then the kernel should return one of:

- `P2_SUPPLY_CHAIN_TRUST` (if dependency reproducibility is the next gap), or
- `P3_RUNTIME_HARDENING` (if supply chain is already clean), or
- `P4_MONITOR_AND_RELOOP` (if nothing dominates).

### HOLD — remain on `P1_TRUTH_INTEGRITY`

If any of:

- PROHIBITED > 0, or
- NEEDS_REWRITE > 10 on internal docs, or
- The live hero still carries any of C4/C5/C7/C9 after a paid-ads dry-run
  window.

The implementation lane continues on the `WEBSITE_PATCH_PLAN.md` batches in
order. No paid distribution. No press release. No enterprise outreach that
cites the prohibited claims.

### NO-GO — ESCALATE

If the audit rerun shows the PROHIBITED count has **grown** between runs,
escalate to a hard halt:

1. Identify the docs-authoring process that re-introduced the pattern.
2. Add a CI advisory job that runs the omni audit on every PR and comments
   a delta (no block, only surface).
3. Add a new pattern to `tools/audit/flywheel_kernel/patterns.json` if the
   mechanism is a new one (e.g. "strategy-deck drift").

Do not proceed with public launch work while the count is rising.

---

## Things this pack explicitly does not decide

- Which owner applies each row of `WEBSITE_PATCH_PLAN.md`.
- Whether the AGI-claim rewrite requires a canon-pack update. (It probably
  does; canon packs are **out of scope** for this pass.)
- Whether the receipt-linking backlog in `RECEIPT_LINKING_BACKLOG.md` is
  sequenced by a human or by a separate scheduling pass.
- Paid-ad launch date. (Not until Batch A of `WEBSITE_PATCH_PLAN.md` is
  live AND audit rerun confirms the four platform-policy claims are gone.)

---

## Measurement commands

```bash
# Refresh audit artefacts (already done for this pack):
.venv/bin/python -m tools.audit.omni_audit.run_audit \
    --repo-root . \
    --out-dir /tmp/bizra-omni-audit-refresh \
    --no-network

# Re-run Flywheel Kernel after implementation edits:
.venv/bin/python -m tools.audit.flywheel_kernel.kernel \
    --audit-dir docs/audits/omnidirectional_hyperdimensional_audit_v0_1/artifacts \
    --out /tmp/bizra-flywheel-p1-report.json

# Inspect G-FW-003 status and blocked_by list:
python3 -c "
import json
d = json.load(open('/tmp/bizra-flywheel-p1-report.json'))
print('priority:', d['priority']['priority_id'])
for g in d['guards']:
    if g['guard_id'] == 'G-FW-003':
        print('G-FW-003:', g['status'], '-', g['signal'])
"
```
