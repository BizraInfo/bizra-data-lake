# User Zero Shadow Pilot Runbook

## Purpose
Run internal-only sovereign marketing sessions before any public exposure.

## Command
```bash
python3 scripts/pilot/run_user_zero_shadow.py --default-consent
```

## Expected Outputs
1. `artifacts/pilot/user_zero_shadow_sessions.jsonl`
2. `artifacts/pilot/user_zero_shadow_summary.json`

## Gate Checks
1. `chain_ok` must be `true`.
2. All denied sessions must include explicit redline events.
3. No claim-bearing successful response may have empty `evidence_refs`.

## Incident Handling
1. If chain verification fails: stop pilot, quarantine artifacts, rerun with fixed script.
2. If redline events are missing where expected: treat as policy regression.
3. Keep this runbook internal-only; do not market as audited benchmark.
