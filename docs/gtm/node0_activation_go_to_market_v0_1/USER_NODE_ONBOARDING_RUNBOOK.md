# User Node Onboarding Runbook

## Purpose

Provide an operator-safe sequence for onboarding a trusted user device into the private Node0 pilot. This is not a public self-serve installer yet.

## Roles

| Role | Responsibility |
|---|---|
| Operator | Approves user, controls pilot keys, decides go/no-go. |
| Device owner | Provides device, runs local commands, confirms consent. |
| Technical lead | Troubleshoots install, receipt verification, and runtime errors. |
| Evidence keeper | Archives receipt hashes, logs, and pilot notes. |

## Pre-Onboarding Checklist

- User is known and approved for private pilot.
- Device owner understands this is experimental.
- Device meets `MINIMUM_HARDWARE_PROFILE.md`.
- No confidential third-party data is used in pilot missions.
- Operator kill-switch is ready.
- Evidence folder for the device is prepared.

## Onboarding Sequence

### 1. Device Intake

Record:

- Owner name or pilot alias.
- Device type.
- OS version.
- CPU, RAM, GPU if any.
- Available disk.
- Network environment.
- Local LLM runtime availability if any.

### 2. Local Environment

Install only the minimum needed for the pilot:

- Git.
- Python 3.11 or 3.12 with venv.
- Rust stable if building Rust crates locally.
- Node 20 if the UI or supporting frontend is needed.
- Local model runtime only if the pilot mission requires inference.

### 3. Repository and Dependencies

- Clone or copy the approved repo state.
- Record commit hash.
- Install dependencies from documented project commands.
- Do not use untracked secrets or copied private credentials.

### 4. Identity Setup

- Generate or import pilot node identity.
- Record public key fingerprint.
- Do not share private keys.
- Operator records the public key and device alias.

### 5. Local Health

Run local health/readiness checks. A device cannot proceed if:

- Required binaries are missing.
- Local state cannot be written.
- Key files are unreadable.
- The runtime cannot emit or verify a local receipt.

### 6. Handshake

- Operator approves peer address or local connection method.
- Node0 sends signed handshake receipt.
- User node verifies Node0 receipt.
- User node sends signed handshake receipt.
- Node0 verifies user-node receipt.

### 7. Restart Recovery

- Stop the node.
- Restart the node.
- Verify identity and last known handshake state.
- Record result.

## Exit Criteria

The device is either:

- `pilot-ready`: handshake and restart recovery passed.
- `degraded`: local runtime works but network/handshake failed.
- `blocked`: identity, install, receipt, or persistence failed.

## Support Rules

- Do not bypass verification to make a demo pass.
- Do not accept a receipt without signature verification.
- Do not promote a user node into broader testing if it cannot recover after restart.
- Do not describe a degraded or blocked device as connected.
