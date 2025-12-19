# Incident Management (Template)

## Goals
- Reduce MTTR, preserve evidence, and protect users (Amanah).
- Make decisions consistent and explainable (Adl).

## Severity levels (example)
- SEV0: data loss / security breach / total outage
- SEV1: major feature unavailable / widespread errors
- SEV2: partial degradation
- SEV3: minor defect

## Required artifacts
- Incident timeline (UTC)
- Root cause analysis
- Customer impact statement
- Remediations (short-term + long-term)
- Evidence bundle location: `docs/evidence/<timestamp>/...`

## 30/60/90 follow-up
- 30d: fix root cause + add regression test
- 60d: improve detection + alerting
- 90d: run game day / chaos test covering failure mode

