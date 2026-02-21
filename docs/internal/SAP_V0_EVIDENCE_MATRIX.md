# SAP v0 Evidence Matrix (Internal)

Purpose: map normative SAP claims to deterministic evidence.
Allowed states: `Implemented`, `Planned`, `Hypothesis`.

State scoring model:
- `Implemented = 1.0`
- `Planned = 0.5`
- `Hypothesis = 0.0`

| Claim ID | Normative Claim | State | Score | Evidence Link | Notes/Risk |
|---|---|---|---:|---|---|
| SAP-C01 | Child permits are strict subset + degraded budgets | Implemented | 1.0 | `schemas/sap/v0/permit_envelope.schema.json`, `tests/conformance/sap_v0/negative/02_permit_capability_escalation.json` | Enforced in v0 conformance artifacts; runtime depth wiring remains post-v0. |
| SAP-C02 | Out-of-scope requests fail-closed + redline | Implemented | 1.0 | `tests/conformance/sap_v0/negative/03_out_of_scope_request.json`, `schemas/sap/v0/redline_violation.schema.json` | Conformance-level fail-closed behavior locked. |
| SAP-C03 | Session strict limits enforced (`50/300/65536`) | Implemented | 1.0 | `schemas/sap/v0/meet_open.schema.json`, `tests/conformance/sap_v0/negative/08_meet_open_exceeds_limits.json` | Strict hard ceilings active in schema + validator. |
| SAP-C04 | Offer provenance is mandatory | Implemented | 1.0 | `schemas/sap/v0/offer.schema.json`, `tests/conformance/sap_v0/negative/04_offer_without_provenance.json` | Empty provenance rejected deterministically. |
| SAP-C05 | Data sharing requires ConsentReceipt before acceptance | Implemented | 1.0 | `schemas/sap/v0/consent_receipt.schema.json`, `tests/conformance/sap_v0/negative/09_data_sharing_without_consent.json` | Conformance gate blocks consentless sharing flow. |
| SAP-C06 | Outcome receipts are hash-chained predecessor-validated | Implemented | 1.0 | `bizra-omega/bizra-agent/src/action_types.rs`, `tests/conformance/sap_v0/negative/05_tampered_receipt_chain.json` | Runtime receipts exist; tampered chain rejected in conformance. |
| SAP-C07 | Ads profile requires user-side sovereign agent path | Implemented | 1.0 | `specs/sap-v0/profiles/agentic-ads-retail-v0.md`, `tests/conformance/sap_v0/negative/07_invalid_role_pairing.json` | Invalid pairing blocked in profile conformance. |
| SAP-C08 | Disclosure includes uncertainty + source refs | Implemented | 1.0 | `schemas/sap/v0/disclosure.schema.json`, `tests/conformance/sap_v0/positive/06_disclosure_valid.json` | Contract requires both fields. |
| SAP-C09 | Agent card has mandatory compilation trust block | Implemented | 1.0 | `schemas/sap/v0/agent_card.schema.json`, `tests/conformance/sap_v0/negative/01_agent_card_missing_compilation.json` | Missing compilation block rejected. |
| SAP-C10 | No new wire verbs (additive payload overlay only) | Implemented | 1.0 | `specs/sap-v0/03-wire-mapping-local-first.md`, `bizra-omega/bizra-node/src/protocol.rs` | Existing verb surface retained. |
| SAP-C11 | Consent revocation endpoint mandatory | Implemented | 1.0 | `schemas/sap/v0/consent_receipt.schema.json`, `tests/conformance/sap_v0/negative/10_missing_revocation_endpoint.json` | Empty/missing endpoint rejected. |
| SAP-C12 | Cross-node GO/MEET transport production-ready | Hypothesis | 0.0 | `specs/sap-v0/README.md` | Explicitly out of scope for SAP v0. |

## Score Summary
1. Full-matrix score: `(11*1.0 + 0*0.5 + 1*0.0)/12 = 0.9167`.
2. In-scope v0 score (excluding SAP-C12): `11/11 = 1.0000`.

## Validation Run Snapshot
- SAP conformance result: `22/22` passing via `python3 scripts/spec/validate_sap_v0.py`.

Internal note: this matrix is not an external audit report.
