# Phase 04: SAP Agentic Ads Marketing Integration

| Field      | Value                                                    |
|------------|----------------------------------------------------------|
| Status     | SPEC                                                     |
| Depends on | Phase 03 (agent-as-marketing frontend)                   |
| Goal       | BIZRA markets itself using its own SAP v0 agentic ads protocol |
| Author     | SPARC spec-pseudocode                                    |
| Date       | 2026-02-21                                               |

---

## 1. Agentic Ad Format

Unlike traditional banner ads, a BIZRA agentic ad is a live sovereign agent
embedded in a page. The visitor interacts with the agent directly. The ad
format is defined as an SAP v0 payload validated against the schemas in
`schemas/sap/v0/`.

```pseudocode
struct AgenticAd:
  ad_id:           blake3::hash(campaign_id + placement_id + timestamp())
  campaign:        "user-zero-bootstrap"
  profile:         "sap-ads-retail-v0"
  agent:           SovereignAgentCard     // from Phase 03, all 8 required fields
  placement:
    format:        "interactive-agent"    // not a banner, an actual agent
    entry_point:   "chat"                 // visitor types, agent responds
    dimensions:    "full-width"           // takes over the interaction space
    embed_allowed: true                   // can be embedded on partner sites
  disclosure:      Disclosure             // pre-populated, validated against disclosure.schema.json
  consent_config:  ConsentConfig

struct ConsentConfig:
  required_before_data_exchange:
    - "data_sharing"
    - "personalization"
    - "recommendation"
  optional:
    - "analytics"
    - "improvement_feedback"
  default_decision:  "deny"              // silence = no consent (SC-02)
  expiry:            Duration::days(30)
  revocable:         true
  revocation_uri:    "/consent/revoke/{session_id}"
```

---

## 2. MeetOpen Session Lifecycle

The 8-phase lifecycle maps the SAP agentic ads flow to BIZRA marketing.
Each phase references the relevant SAP v0 primitive and sovereignty
constraint.

```pseudocode
// Phase 1: Discovery
fn phase_discovery(visitor: Visitor) -> AgenticAdView:
  // Visitor arrives at BIZRA website or partner embed
  ad = load_agentic_ad("user-zero-bootstrap")
  return render_ad_card(ad.agent, ad.disclosure)
  // No data exchange yet. Display only.

// Phase 2: MeetOpen
fn phase_meet_open(visitor: Visitor, ad: AgenticAd) -> SAPSession:
  session = sap::MeetOpen(
    session_id:         blake3::hash(visitor.id + ad.ad_id + timestamp()),
    initiator_agent_id: visitor.agent_id or "anonymous-visitor",
    responder_agent_id: ad.agent.agent_id,
    place_id:           ad.placement.entry_point,
    objective:          "explore-bizra",
    consent_scope:      [],             // empty until explicit consent
    session_limits: SessionLimits {
      max_messages:         50,         // SAP v0 ceiling
      max_duration_seconds: 300,        // 5 minutes
      max_payload_bytes:    65536,      // 64KB
    },
    expires_at: now() + Duration::seconds(300),
  )
  // Deliver initial disclosure (SC-08 mandatory)
  session.disclosure = generate_marketing_disclosure(session)
  return session

// Phase 3: Conversation
fn phase_conversation(session: SAPSession, message: string) -> SAPResponse:
  enforce_session_limits(session)       // SC-03 hard stop
  response = mumo_agent.respond(message, session)
  response.disclosure = refresh_disclosure(session) // per-message refresh
  response.receipt = record_outcome(session, response)
  check_redline(response, session)      // block if violation
  return response

// Phase 4: ConsentRequest (if data exchange needed)
fn phase_consent_request(session: SAPSession, scopes: Vec<string>) -> ConsentReceipt:
  // SC-05: Consent before data sharing
  return request_consent(session, scopes)
  // See Section 4 for full implementation

// Phase 5: Value Exchange
fn phase_value_exchange(session: SAPSession, consent: ConsentReceipt) -> ValuePayload:
  // Only proceeds if consent was granted
  assert consent.granted_scopes.len() > 0
  // Agent provides: architecture walkthrough, demo, custom analysis
  // Visitor provides: feedback, interest signal (scoped by consent)
  payload = generate_value_payload(session, consent.granted_scopes)
  receipt = record_outcome(session, payload)
  return payload

// Phase 6: OutcomeReceipt
// Implicit — every phase generates OutcomeReceipts via record_outcome()
// See Section 5 for receipt chain implementation

// Phase 7: Recommendation (optional, consent-gated)
fn phase_recommendation(session: SAPSession) -> Option<Recommendation>:
  if not session.has_consent_for("recommendation"):
    return None                         // hard gate, not soft
  recommendation = generate_recommendation(session)
  recommendation.disclosure = refresh_disclosure(session)
  return recommendation

// Phase 8: Session Close
fn phase_session_close(session: SAPSession) -> FinalReceipt:
  final_receipt = record_outcome(session, Outcome::SessionClose)
  // Reminder of consent revocation
  final_receipt.revocation_reminder = format!(
    "You can revoke consent at: {}",
    session.consent_config.revocation_uri
  )
  return final_receipt
```

---

## 3. Mandatory Disclosure Implementation

Every response in a marketing session must carry a `Disclosure` that
validates against `schemas/sap/v0/disclosure.schema.json`. The schema
requires: `disclosure_id`, `claims` (array, minItems 1), `source_refs`
(array of `{ref_hash, ref_type}` objects), `uncertainty` (object with
`score`, `method`, `notes`), and `compliance_assertions` (string array,
minItems 1).

```pseudocode
fn generate_marketing_disclosure(session: SAPSession) -> Disclosure:
  return Disclosure {
    disclosure_id: blake3::hash(session.session_id + timestamp()),

    claims: [
      "This agent is compiled from Mumo's real conversations (7000+ across 10 platforms)",
      "BIZRA is in alpha stage (Alpha-100 release)",
      "Agent responses use locally-compiled reflexes, not cloud LLM",
      // Session-specific claims appended dynamically
    ],

    source_refs: [
      {
        ref_hash: blake3::hash("specs/sap-v0/01-core-primitives.md"),
        ref_type: "document",
        ref_uri:  "specs/sap-v0/01-core-primitives.md",
      },
      {
        ref_hash: blake3::hash("tests/conformance/sap_v0/"),
        ref_type: "human_attestation",
        ref_uri:  "tests/conformance/sap_v0/ (24/24 tests)",
      },
      {
        ref_hash: blake3::hash("bizra-omega/"),
        ref_type: "human_attestation",
        ref_uri:  "bizra-omega/ (971+ Rust tests)",
      },
    ],

    uncertainty: {
      score:  0.15,             // 15% uncertainty — alpha software
      method: "manual-assessment",
      notes:  "Compilation score is 0.92. Alpha software: features may change. "
            + "Performance benchmarks are from development hardware (RTX 4090, 128GB RAM).",
    },

    compliance_assertions: [
      "SAP_v0 (24/24 conformance tests)",
      "GDPR_minimal (no PII stored without consent)",
    ],
  }
```

### Disclosure Refresh Strategy

```pseudocode
fn refresh_disclosure(session: SAPSession) -> Disclosure:
  base = generate_marketing_disclosure(session)

  // Append session-specific claims
  if session.message_count > 10:
    base.claims.append("This is a longer conversation. Session limits: "
      + format!("{}/{} messages", session.message_count, 50))

  if session.has_consent:
    base.claims.append("You have granted consent for: "
      + session.granted_scopes.join(", "))

  // Recalculate uncertainty based on conversation quality
  base.uncertainty.score = calculate_session_uncertainty(session)

  return base
```

---

## 4. ConsentReceipt Flow

Consent receipts validate against `schemas/sap/v0/consent_receipt.schema.json`.
Required fields: `consent_receipt_id`, `session_id`, `user_agent_id`,
`brand_agent_id`, `data_shared` (array of `{field, purpose}` objects),
`data_withheld` (string array), `consent_timestamp` (integer),
`consent_hash` (64 hex chars), `revocation_endpoint`.

```pseudocode
fn request_consent(session: SAPSession, scopes: Vec<ConsentScope>) -> ConsentReceipt:
  // Build clear-language consent request
  request = ConsentRequest {
    session_id:       session.session_id,
    requested_scopes: scopes,
    purpose_text:     build_purpose_text(scopes),     // human-readable
    duration:         Duration::days(30),
    revocation_uri:   format!("/consent/revoke/{}", session.session_id),
    data_categories:  scopes.map(|s| s.data_category),
  }

  // Present to visitor — MUST await explicit decision
  // Never timeout into consent. Silence = denial (SC-02 fail-closed).
  decision = await visitor.consent_decision(request)

  if decision.granted:
    data_shared = decision.granted_scopes.map(|s| {
      field:             s.data_category,
      purpose:           s.purpose,
      retention_seconds: Duration::days(30).as_secs(),
      granularity:       "exact",   // or "aggregated" per scope
    })
    data_withheld = decision.denied_scopes.map(|s| s.data_category)

    receipt = ConsentReceipt {
      consent_receipt_id: blake3::hash(session.session_id + scopes_hash + timestamp()),
      session_id:         session.session_id,
      user_agent_id:      session.initiator_agent_id,
      brand_agent_id:     session.responder_agent_id,
      data_shared:        data_shared,
      data_withheld:      data_withheld,
      consent_timestamp:  unix_timestamp_now(),
      consent_hash:       blake3::hash(receipt_id + data_shared + data_withheld + timestamp),
      revocation_endpoint: request.revocation_uri,
    }

    // Chain into session receipts
    outcome_receipt = record_outcome(session, Outcome::ConsentGranted(receipt))
    return receipt

  else:
    // Respect denial — continue session without personalization
    // Same ConsentReceipt shape but data_shared=[], all scopes in data_withheld
    return ConsentReceipt::denied(session, scopes, request.revocation_uri)
```

---

## 5. OutcomeReceipt Chain

OutcomeReceipts form an append-only hash chain per SC-06 (Receipt Chain
Integrity). The chain mirrors the `ActionReceipt` pattern in
`bizra-omega/bizra-node/src/action_executor.rs` where `prev_receipt_hash`
(line 48) links each receipt to its predecessor.

```pseudocode
fn record_outcome(session: SAPSession, outcome: Outcome) -> OutcomeReceipt:
  prev_hash = session.latest_receipt_hash
  if prev_hash is None:
    prev_hash = [0u8; 32]              // genesis hash, same as action_executor.rs line 80

  receipt = OutcomeReceipt {
    action_id:          blake3::hash(session.session_id + outcome.type + timestamp()),
    session_id:         session.session_id,
    guardian_verdict:    "approved",    // or "blocked" if redline triggered
    permit_hash:        session.permit.hash(),
    policy_hash:        session.agent_card.policy_hash,
    receipt_hash:       blake3::hash(prev_hash + action_id + outcome.content_hash),
    prev_receipt_hash:  prev_hash,
    // Extension fields for marketing context
    outcome_type:       outcome.type,  // "response", "consent", "recommendation", "close"
    content_hash:       blake3::hash(outcome.content),
    timestamp:          unix_timestamp_now(),
    ihsan_score:        outcome.ihsan_score,
  }

  session.receipt_chain.append(receipt)
  session.latest_receipt_hash = receipt.receipt_hash

  return receipt

fn verify_receipt_chain(chain: Vec<OutcomeReceipt>) -> bool:
  if chain.is_empty():
    return true

  // First receipt must link to genesis hash
  if chain[0].prev_receipt_hash != [0u8; 32]:
    return false

  for i in 1..chain.len():
    if chain[i].prev_receipt_hash != chain[i-1].receipt_hash:
      return false
    // Verify hash integrity
    expected = blake3::hash(chain[i].prev_receipt_hash + chain[i].action_id + chain[i].content_hash)
    if chain[i].receipt_hash != expected:
      return false

  return true
```

---

## 6. RedlineViolation Recording

Redline violations are constitutional guardrails that block responses
before they reach the visitor. Any violation prevents delivery and
generates an alternative response with enhanced disclosure.

```pseudocode
fn check_redline(response: AgentResponse, session: SAPSession) -> Result<(), RedlineViolation>:
  violations = []

  // Check 1: No unsupported claims (SC-08)
  unsupported = detect_unsupported_claims(response.content)
  if unsupported.len() > 0:
    violations.push(format!("Unsupported claims: {}", unsupported.join(", ")))

  // Check 2: Ihsan gate (>= 0.95 production threshold)
  if response.ihsan_score < 0.95:
    violations.push(format!("Ihsan below threshold: {} < 0.95", response.ihsan_score))

  // Check 3: No pressure tactics
  pressure_patterns = [
    "limited time", "act now", "don't miss", "exclusive offer",
    "best .* ever", "guaranteed", "risk-free", "no-brainer",
  ]
  if matches_any(response.content, pressure_patterns):
    violations.push("Pressure language detected")

  // Check 4: Disclosure present (SC-08)
  if response.disclosure is None:
    violations.push("Missing mandatory disclosure")

  // Check 5: Disclosure validates against disclosure.schema.json
  if response.disclosure is Some:
    if not validate_against_schema(response.disclosure, "schemas/sap/v0/disclosure.schema.json"):
      violations.push("Disclosure schema validation failed")

  if violations.is_empty():
    return Ok(())

  // Block response, generate violation record
  violation = RedlineViolation {
    violation_id:      blake3::hash(session.session_id + violations_hash + timestamp()),
    session_id:        session.session_id,
    violations:        violations,
    original_hash:     blake3::hash(response.content),
    timestamp:         unix_timestamp_now(),
    remediation:       "Response blocked. Alternative generated with enhanced disclosure.",
  }

  // Record violation as an OutcomeReceipt
  record_outcome(session, Outcome::RedlineViolation(violation))

  return Err(violation)
```

---

## 7. Profile Configuration

The `bizra-marketing-v0` profile extends `sap-ads-retail-v0` with
BIZRA-specific policy.

```pseudocode
SAPProfile "bizra-marketing-v0":
  extends: "sap-ads-retail-v0"

  agent_identity:
    name:               "mumo-sovereign-v1"
    type:               "compiled-sovereign-agent"
    compilation_source: "7000+ multi-platform conversations"

  disclosure_policy:
    mode:                "always-on"        // never hidden
    refresh:             "per-message"       // updated each response
    claims_min_items:    1                   // matches schema minItems
    uncertainty_required: true               // score + method + notes
    source_refs_required: true               // ref_hash + ref_type

  consent_policy:
    data_sharing:        "explicit-opt-in"
    personalization:     "explicit-opt-in"
    analytics:           "explicit-opt-in"   // even analytics requires consent
    recommendation:      "explicit-opt-in"
    default:             "deny"              // silence = no consent (SC-02)
    expiry_max:          Duration::days(30)
    revocable:           true

  session_policy:
    max_messages:        50                  // SAP v0 ceiling
    max_duration:        300                 // seconds
    max_payload:         65536               // bytes
    courtesy_gate:       20                  // soft reminder at 20 messages

  redline_config:
    ihsan_threshold:     0.95
    pressure_detection:  true
    claim_verification:  true
    auto_block:          true                // block violating responses, do not just log

  economic_model:
    visitor_cost:        "$0"                // always free for visitors
    agent_compute:       "local-first"       // RTX 4090, no cloud dependency
    cloud_fallback:      "pennies"           // if local unavailable
```

---

## 8. Data Flow Diagram

```
  Visitor arrives
       |
       v
  AgenticAd Display         <-- SovereignAgentCard + badge
       |
       v
  MeetOpen (Phase 2)        --> Disclosure (SC-08), limits: 50/300/64K
       |
       v
  Conversation Loop         --> OutcomeReceipt per message (SC-06)
  (+ Disclosure refresh)        RedlineViolation check each response
  (+ Ihsan gate)
       |
       v (if data exchange)
  ConsentRequest (Phase 4)  --> ConsentReceipt (SC-05, SC-02)
       |
       v (if consent granted)
  Value Exchange (Phase 5)  --> OutcomeReceipt
       |
       v
  Session Close (Phase 8)   --> Final OutcomeReceipt + revocation reminder
```

---

## 9. TDD Anchors

| # | Test Name | Property | Spec Ref |
|---|-----------|----------|----------|
| 1 | `test_disclosure_schema_valid` | Generated disclosure validates against `disclosure.schema.json` | `schemas/sap/v0/disclosure.schema.json` |
| 2 | `test_consent_receipt_schema_valid` | Generated receipt validates against `consent_receipt.schema.json` | `schemas/sap/v0/consent_receipt.schema.json` |
| 3 | `test_disclosure_always_present` | No response in `bizra-marketing-v0` profile lacks disclosure | SC-08, profile policy |
| 4 | `test_consent_default_deny` | No explicit decision = no `data_shared` entries (empty array) | SC-02 |
| 5 | `test_consent_revocation_works` | Revoked consent immediately clears `granted_scopes` on session | SC-05, consent_receipt schema |
| 6 | `test_consent_expiry_honored` | Consent older than 30 days treated as revoked | Profile policy |
| 7 | `test_receipt_chain_integrity` | `verify_receipt_chain()` returns true for valid chain | SC-06 |
| 8 | `test_receipt_chain_tamper_detection` | Modified receipt breaks `verify_receipt_chain()` | SC-06 |
| 9 | `test_receipt_chain_genesis_hash` | First receipt links to `[0u8; 32]` genesis hash | `action_executor.rs` line 80 |
| 10 | `test_redline_blocks_overselling` | "Best AI ever" type claims produce `RedlineViolation` | Redline config |
| 11 | `test_redline_blocks_pressure` | "Limited time offer" language produces `RedlineViolation` | Redline config |
| 12 | `test_redline_blocks_missing_disclosure` | Response without disclosure produces `RedlineViolation` | SC-08, redline config |
| 13 | `test_ihsan_gate_blocks_low_score` | Response with ihsan < 0.95 produces `RedlineViolation` | Ihsan >= 0.95 |
| 14 | `test_profile_extends_sap_ads_retail` | `bizra-marketing-v0` is valid extension of `sap-ads-retail-v0` | Profile config |
| 15 | `test_visitor_zero_cost` | No charge records generated for visitor interactions | Economic model |
| 16 | `test_session_limits_enforced` | Exceeding 50 messages terminates session (SC-03) | SC-03 |
| 17 | `test_consent_receipt_has_revocation_endpoint` | Every receipt has non-empty `revocation_endpoint` | `consent_receipt.schema.json` |

---

## 10. Open Questions

| # | Question | Proposed Resolution |
|---|----------|---------------------|
| 1 | Third-party ad placement (embed agent on partner sites)? | Deferred to post-Alpha. `embed_allowed: true` in ad format is forward-compatible. |
| 2 | Disclosure language localization? | English-first, Arabic second. Schema supports any string content. |
| 3 | Receipt chain storage: local SQLite vs distributed? | Local SQLite first (sovereignty). Federate via `bizra-federation` at Beta-1K. |
| 4 | Should redline violations be visible to the visitor? | Yes, in the Transparency Panel. Honesty about our own guardrails is on-brand. |
| 5 | Consent receipt format: SAP schema only, or also W3C Data Privacy Vocabulary? | SAP schema only for Alpha-100. W3C DPV mapping deferred. |
