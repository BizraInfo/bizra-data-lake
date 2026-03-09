# SAP v0 Sovereignty Constraints

Stable rule IDs are used in conformance fixtures and violation logs.

## SC-01 Permit Subset + Degradation
Child `PermitEnvelope.capabilities` MUST be strict subset of parent.
Child limits MUST not exceed parent effective limits after degradation.

## SC-02 Consent Scope Fail-Closed
Any out-of-scope request relative to `consent_scope` MUST be denied and
produce `RedlineViolation`.

## SC-03 Session Limits Hard Stop
If `max_messages`, `max_duration_seconds`, or `max_payload_bytes` are
exceeded, session MUST terminate with denial status.

## SC-04 Offer Provenance Required
`Offer.provenance_hashes` MUST be non-empty.

## SC-05 Consent Before Shared Data Acceptance
If `data_shared` is involved in acceptance flow, valid `ConsentReceipt`
MUST exist before acceptance.

## SC-06 Receipt Chain Integrity
`OutcomeReceipt` MUST be append-only with predecessor hash validation.
Tamper detection MUST fail verification.

## SC-07 Ads Role Pairing
Ads profile sessions MUST include user-side sovereign agent role path.
Brand-only negotiation is invalid.

## SC-08 Disclosure Transparency
Claims MUST include source references and uncertainty.
