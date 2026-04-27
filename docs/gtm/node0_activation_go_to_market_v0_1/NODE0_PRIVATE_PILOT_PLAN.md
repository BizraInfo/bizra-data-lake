# Node0 Private Pilot Plan

## Purpose

Safely connect a small number of ready user devices to the BIZRA system without claiming full production federation. The pilot exists to prove a narrow, measurable truth: two or more sovereign nodes can verify each other's signed receipts while preserving local authority.

## Pilot Scope

### In Scope

- 2-5 trusted user devices.
- Local install and readiness check.
- Node identity generation or import.
- Public-key exchange with operator approval.
- One signed receipt exchange between Node0 and each user node.
- Independent receipt verification on both sides.
- Restart recovery check after pilot handshake.
- Evidence pack per device.

### Out of Scope

- Paid public onboarding.
- Open network discovery.
- Token economy activation.
- SAT-5 network coordination.
- Federated cognition or pooled learning.
- Claims that URP is production-live at scale.

## Pilot Entry Criteria

- Operator approves device owner.
- Device meets minimum hardware profile.
- User accepts private pilot terms and understands data boundaries.
- Node0 local health is green.
- Kill-switch procedure is available.
- Public/private key handling procedure is understood.

## Pilot Flow

1. Register candidate user and device.
2. Run hardware and OS readiness checks.
3. Install dependencies.
4. Initialize local node identity.
5. Verify local genesis or pilot identity artifact.
6. Exchange public keys with Node0.
7. Run handshake mission.
8. Verify signed receipt on both devices.
9. Restart both nodes and verify persisted state.
10. Save evidence pack and operator notes.

## Success Criteria

A pilot device is successful only if all are true:

- Device completes readiness check.
- Device has a stable node identity.
- Node0 receives a signed receipt from the user node.
- User node receives a signed receipt from Node0.
- Both receipts verify with the published public keys.
- Chain head or handshake state survives restart.
- Operator can explain what succeeded and what did not.

## Pilot Evidence Pack

Each device should produce:

- Device profile summary.
- Install log.
- Node identity public key fingerprint.
- Handshake receipt hash.
- Verification result.
- Restart recovery result.
- Operator notes.
- Known issues.

## Go / No-Go for Expansion

| Expansion level | Required evidence |
|---|---|
| Add devices 3-5 | Two devices complete signed receipt exchange and restart recovery. |
| Invite-only alpha | At least five devices complete onboarding with repeatable operator support. |
| Public beta | Onboarding automation, privacy policy, support path, SBOM, and incident runbook are green. |
| Paid growth | Website claim cleanup, creative QA, ad policy checks, UTM, kill-switch, and proof links are green. |
