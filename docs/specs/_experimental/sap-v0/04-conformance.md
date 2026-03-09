# SAP v0 Conformance

Validator entrypoint:

```bash
python3 scripts/spec/validate_sap_v0.py
```

## Required Scenario Coverage (12)
1. Valid permit inheritance passes.
2. Capability escalation attempt fails.
3. Out-of-consent-scope request fails with `RedlineViolation`.
4. Offer without provenance fails.
5. Tampered receipt chain fails verification.
6. Expired `MeetOpen` fails.
7. Invalid role pairing fails.
8. End-to-end mapping over existing wire verbs passes.
9. Missing `compilation` block in `SovereignAgentCard` fails.
10. `MeetOpen` exceeding strict limits fails.
11. Data sharing without `ConsentReceipt` fails.
12. Missing `revocation_endpoint` fails profile compliance.

## Determinism Requirements
1. Positive fixtures must all pass.
2. Negative fixtures must all fail.
3. Legacy/non-canonical fixture shapes must be rejected.
4. Validator output must be stable run-to-run.
