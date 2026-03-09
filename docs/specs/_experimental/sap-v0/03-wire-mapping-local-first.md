# SAP v0 Wire Mapping (Local-First)

SAP v0 introduces no new wire verbs.

## Existing Verbs Used
1. `PLAN_ACTION`
2. `RUN_ACTION`
3. `ACTION_STATUS`
4. `ACTION_HISTORY`
5. `EXPLAIN`

## Payload Overlay
All SAP payloads are wrapped inside existing JSON body:

```json
{
  "profile": "sap-ads-retail-v0",
  "sap": {
    "type": "MeetOpen|MeetMessage|Offer|Disclosure|ConsentReceipt|OutcomeReceipt|RedlineViolation",
    "body": { "...": "..." }
  }
}
```

## Mapping
1. `PLAN_ACTION` plans SAP primitives.
2. `RUN_ACTION` executes approved actions after guardian/permit checks.
3. `ACTION_STATUS` returns lifecycle state for SAP-tagged action IDs.
4. `ACTION_HISTORY` returns receipt-chain artifacts for auditability.
5. `EXPLAIN` remains retrieval-only explanation path.

## Profile Marker
`profile: "sap-ads-retail-v0"` is mandatory in payload for profile-bound flows.
