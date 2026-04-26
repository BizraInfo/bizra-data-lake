# Pilot Evidence Register

## Status

**Truth label:** MEASURED_LOCAL_ARTIFACT  
**Scope:** Local Node0 private-pilot handshake proof only.  
**Non-claim:** This does not prove production federation, open discovery, SAT-5 network coordination, Genesis 100, or external user-node readiness.

## What Was Verified

The new private-pilot operator path can produce a signed handshake artifact and verify it fail-closed.

Verified commands:

```bash
python3 -m py_compile scripts/node0_standalone.py tests/scripts/test_node0_standalone.py
```

Direct smoke verification generated and verified a local handshake artifact:

- `artifacts/proofs/node0-private-pilot-handshake-smoke.json`
- `artifacts/proofs/node0-private-pilot-handshake-tampered-smoke.json`
- `artifacts/proofs/node0-private-pilot-verification-report.json`
- `artifacts/proofs/node0-private-pilot-evidence-receipt.json`

## Evidence Receipt

| Field | Value |
|---|---|
| Receipt file | `artifacts/proofs/node0-private-pilot-evidence-receipt.json` |
| Receipt hash | `453068aa93754869332ca62331aaaeeeb28506dfbfeb514908a1ba72f5705407` |
| Verification status | `pass_with_pytest_environment_gap` |
| Confidence | `strong_local_cli_verification` |

## Environment Gap

`pytest` could not run because the system Python environment does not include `pytest`, and no `.venv-linux` or `.venv` interpreter was present in the workspace. Syntax compilation and direct handshake smoke verification passed.

`PILOT_EVIDENCE_REGISTER.md` cites the evidence receipt but is intentionally excluded from the receipt's artifact hash set to avoid self-referential hash churn.

## CLI Verification Results

| Check | Result |
|---|---|
| `pilot-verify` on signed smoke artifact | PASS, `reason_code=OK` |
| `pilot-doctor` on current Node0 state | PASS, `status=ready`, `blocking=[]` |
| `pilot-verify` on tampered artifact | PASS as rejection, `reason_code=PAYLOAD_DIGEST_MISMATCH` |

## Current Upgrade in Truth Label

Before this implementation:

- Private pilot handshake was PLANNED.
- Source-code blueprint did not exist.
- No local pilot handshake artifact existed.

After this implementation:

- Private pilot handshake artifact generation is MEASURED locally.
- Artifact verification is MEASURED locally.
- Production multi-node transport remains PLANNED.

## Next Evidence Needed

To upgrade from `MEASURED_LOCAL_ARTIFACT` to a real private-pilot cross-device claim:

1. Run `pilot-doctor` on Node0 after lifecycle gates are green.
2. Generate a handshake artifact for a real user device.
3. Verify the Node0 artifact on the user device.
4. Generate a reciprocal user-device artifact.
5. Verify the user-device artifact on Node0.
6. Archive both artifacts with device profile and restart recovery notes.
