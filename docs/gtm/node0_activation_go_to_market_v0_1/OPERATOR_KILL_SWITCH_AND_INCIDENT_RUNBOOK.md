# Operator Kill-Switch and Incident Runbook

## Purpose

Give the operator a clear procedure to pause pilot activity, isolate devices, preserve evidence, and communicate truthfully during incidents.

## Kill-Switch Authority

Only the operator or a named technical lead can activate the pilot kill-switch. The action should be recorded with timestamp, reason, affected devices, and evidence path.

## Kill-Switch Triggers

Activate immediately if any occur:

- Private key exposure is suspected.
- A node accepts an invalid signature.
- A node reports a different chain state than expected.
- A user device behaves unexpectedly or cannot be identified.
- The pilot exposes private data outside the intended boundary.
- Public claims drift beyond measured evidence.
- A security scanner reports a new high-confidence secret.

## Immediate Actions

1. Pause new onboarding.
2. Stop active handshake or mission tests.
3. Record current node states.
4. Preserve logs and receipt artifacts.
5. Isolate affected device from pilot peers.
6. Rotate or revoke affected pilot credentials if needed.
7. Classify severity.
8. Write an incident note before resuming.

## Severity Levels

| Severity | Meaning | Resume condition |
|---|---|---|
| S1 | Key compromise, invalid verification, data leak | No resume until root cause and credential rotation are complete. |
| S2 | Handshake inconsistency or restart failure | Resume only after fix and repeat test. |
| S3 | Install friction or device incompatibility | Resume unaffected devices; document limitation. |
| S4 | Documentation or claim wording issue | Fix copy before external use. |

## Evidence to Preserve

- Device alias.
- Commit hash.
- Command or action that triggered issue.
- Receipt hash if available.
- Signature verification output if available.
- Logs.
- Operator notes.
- Decision to resume or stop.

## Communications Rule

Never say "the network failed" if only a pilot device failed. Never say "the node is connected" if receipt verification did not pass. Use precise labels: ready, degraded, blocked, revoked, or paused.

## Resume Checklist

- Root cause documented.
- Affected credentials handled.
- Reproduction attempted.
- Fix or mitigation applied.
- Verification rerun.
- Operator signs off.
