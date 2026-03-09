# Sovereign Agent Protocol (SAP) v0

Version: `0.1.0-internal`
Date: `2026-02-21`
Status: `Internal spec release`

## Purpose
SAP v0 defines a sovereign, consent-first, fail-closed protocol overlay for
agent negotiation and action execution using existing BIZRA wire verbs only.
No runtime transport changes are required in this milestone.

## Hard Scope
1. Internal-first docs/schemas/conformance/evidence package.
2. Local-first wire mapping over existing verbs.
3. Retail agentic ads profile as the first domain profile.

## Out of Scope
1. New wire verbs.
2. Cross-node GO/MEET transport implementation.
3. Token settlement implementation.
4. External audit claims.

## Canonical Types
1. `SovereignAgentCard`
2. `PermitEnvelope`
3. `MeetOpen`
4. `MeetMessage`
5. `Offer`
6. `Disclosure`
7. `ConsentReceipt`
8. `OutcomeReceipt`
9. `RedlineViolation`

## Mandatory v0 Additions
1. `compilation` trust block in `SovereignAgentCard`.
2. `session_limits` in `MeetOpen` with strict ceilings:
   `max_messages<=50`, `max_duration_seconds<=300`, `max_payload_bytes<=65536`.
3. Mandatory `ConsentReceipt` for any accepted flow that shares user data.

## Normative Rules (MUST)
1. Child permits are strict subsets of parent capability/budget envelope.
2. Out-of-scope consent requests fail-closed and emit `RedlineViolation`.
3. Sessions that exceed limits terminate with denial status.
4. Offers must include non-empty `provenance_hashes`.
5. Data sharing requires `ConsentReceipt` before offer acceptance.
6. Receipts are append-only hash-chained and predecessor-validated.
7. Ads profile negotiation path must include user-side sovereign agent.
8. Disclosures must include source references and uncertainty.

## Wire Mapping
SAP metadata rides in `PLAN_ACTION`/`RUN_ACTION`/`ACTION_STATUS`/
`ACTION_HISTORY`/`EXPLAIN` payload JSON under top-level:
- `profile: "sap-ads-retail-v0"`
- `sap: { type, body }`

## Package Index
- `01-core-primitives.md`
- `02-sovereignty-constraints.md`
- `03-wire-mapping-local-first.md`
- `04-conformance.md`
- `profiles/agentic-ads-retail-v0.md`
- `schemas/sap/v0/*.schema.json`
- `tests/conformance/sap_v0/**/*.json`
- `scripts/spec/validate_sap_v0.py`

## Governance
This is an internal truth artifact. It must not use external-audit language.
Evidence posture must be tracked in `docs/internal/SAP_V0_EVIDENCE_MATRIX.md`
with `Implemented | Planned | Hypothesis` tags.
