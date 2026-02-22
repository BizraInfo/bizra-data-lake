# SAP v0 Core Primitives

This document defines canonical field contracts for SAP v0.
All payloads are JSON and validated by `schemas/sap/v0/*.schema.json`.

## 1) SovereignAgentCard
Required fields:
1. `agent_id`
2. `owner_node_id`
3. `role`
4. `policy_hash`
5. `capabilities`
6. `endpoints`
7. `version`
8. `compilation`

`compilation` required fields:
1. `genesis_version`
2. `ihsan_threshold`
3. `compiled_reflex_count`
4. `compilation_coverage`

## 2) PermitEnvelope
Required fields:
1. `permit_id`
2. `issuer`
3. `holder_agent_id`
4. `capabilities`
5. `limits`
6. `parent_permit_hash`
7. `degradation_factor`
8. `signature`

## 3) MeetOpen
Required fields:
1. `session_id`
2. `initiator_agent_id`
3. `responder_agent_id`
4. `place_id`
5. `objective`
6. `consent_scope`
7. `session_limits`
8. `expires_at`

`session_limits` strict v0 ceilings:
1. `max_messages <= 50`
2. `max_duration_seconds <= 300`
3. `max_payload_bytes <= 65536`

## 4) MeetMessage
Required fields:
1. `session_id`
2. `message_id`
3. `sender_role`
4. `intent`
5. `payload_hash`
6. `content_ref`
7. `timestamp`
8. `signature`

## 5) Offer
Required fields:
1. `offer_id`
2. `product_sku`
3. `price`
4. `currency`
5. `terms`
6. `expiry`
7. `provenance_hashes` (must be non-empty)

## 6) Disclosure
Required fields:
1. `disclosure_id`
2. `claims`
3. `source_refs`
4. `uncertainty`
5. `compliance_assertions`

## 7) ConsentReceipt
Required when any `data_shared` exists.
Required fields:
1. `consent_receipt_id`
2. `session_id`
3. `user_agent_id`
4. `brand_agent_id`
5. `data_shared`
6. `data_withheld`
7. `consent_timestamp`
8. `consent_hash`
9. `revocation_endpoint`

## 8) OutcomeReceipt
Required fields:
1. `action_id`
2. `session_id`
3. `guardian_verdict`
4. `permit_hash`
5. `policy_hash`
6. `receipt_hash`
7. `prev_receipt_hash`
8. `status`

## 9) RedlineViolation
Required fields:
1. `violation_id`
2. `rule_code`
3. `actor`
4. `decision`
5. `timestamp`
6. `reason`
