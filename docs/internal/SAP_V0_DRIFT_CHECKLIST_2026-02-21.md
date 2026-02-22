# SAP v0 Drift Checklist (2026-02-21)

## Findings Before Lock
1. Duplicate fixture naming schemes existed (`01_*` and `valid_*`).
2. Docs used stale field names not aligned to locked canonical contract.
3. `session_limits` semantics were inconsistent between docs and fixtures.
4. Some schema/docs references used stale aliases and inconsistent paths.

## Corrective Actions Applied
1. Replaced fixture set with single canonical naming scheme only.
2. Rewrote SAP docs to locked field contracts and strict limit ceilings.
3. Updated schemas to match canonical contract exactly.
4. Rewrote validator to enforce strict limits and reject non-canonical shapes.

## Verification
1. `python3 scripts/spec/validate_sap_v0.py` passes (all positives pass, all negatives fail).
