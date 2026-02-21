# SAP Profile: Agentic Ads Retail v0

Profile ID: `sap-ads-retail-v0`
Mode: internal-first

## Lifecycle
1. Impression
2. Engage
3. Discover
4. Compare
5. Offer
6. Confirm
7. Execute
8. Receipt

## Role Rules
1. Session MUST include a user-side sovereign agent path.
2. Brand-to-brand only negotiation is invalid.
3. Role pairing violations emit `RedlineViolation`.

## Consent Rules
1. Data sharing requires `ConsentReceipt` before acceptance.
2. `revocation_endpoint` is mandatory.
3. Out-of-scope access requests fail-closed.

## Session Limits
Strict ceilings:
1. `max_messages <= 50`
2. `max_duration_seconds <= 300`
3. `max_payload_bytes <= 65536`

## Wire Surface
Uses only existing verbs with profile marker:
- `PLAN_ACTION`
- `RUN_ACTION`
- `ACTION_STATUS`
- `ACTION_HISTORY`
- `EXPLAIN`

## Internal KPI Hooks
See `docs/internal/SAP_AGENTIC_ADS_PILOT_KPIS.md`.
