# Quarantine — fixture-polluted audit receipts (2026-06-11)

**What:** 6 action-receipt lines were appended to the tracked audit log `bizra-omega/bizra-node/data/audit/action_receipts.jsonl` and removed here.
**Why:** They are test/demo data, not real receipts — synthetic IDs (`act_00000001`/`pln_00000001`), placeholder `receipt_hash` (`abab…`/`cccc…`), `ts:1000000000`, and `err:MISSING_BRIDGE_TOKEN`.
**Provenance:** values originate in `bizra-omega/bizra-node/tests/audit_integration_tests.rs`; the integration test wrote into the real tracked log. The audit log is restored to its committed content; the pollution is preserved here, not deleted.
