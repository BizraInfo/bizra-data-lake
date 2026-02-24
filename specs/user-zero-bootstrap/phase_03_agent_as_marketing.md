# Phase 03: Agent-as-Marketing Frontend

| Field      | Value                                                    |
|------------|----------------------------------------------------------|
| Status     | SPEC                                                     |
| Depends on | Phase 02 (compiled reflexes needed)                      |
| Goal       | Three-layer marketing architecture where the product markets itself |
| Author     | SPARC spec-pseudocode                                    |
| Date       | 2026-02-21                                               |

---

## 1. Three-Layer Marketing Architecture

Three distinct agent personas serve three audiences. Each persona wraps the
same compiled reflex core (Phase 02) but adjusts depth, vocabulary, and
evidence density. All three share a single SAP v0 session protocol, a single
`SovereignAgentCard`, and a single Ihsan gate.

### Layer 1: Website Agent (Public Visitor)

- **Target:** First-time visitors to BIZRA.
- **Protocol:** SAP v0 `MeetOpen` session with profile `sap-ads-retail-v0`.
- **Personality:** Warm, curious, demonstrates sovereignty by example.
- **Constraint:** Never oversells, always discloses limitations.
- **Session limit:** Inherits SAP v0 ceilings (`max_messages <= 50`,
  `max_duration_seconds <= 300`, `max_payload_bytes <= 65536`) per
  `specs/sap-v0/01-core-primitives.md` Section 3.

```pseudocode
fn website_agent_session(visitor_message: string) -> SAPResponse:
  session = sap::MeetOpen(
    session_id:           blake3::hash(visitor_id + timestamp()),
    initiator_agent_id:   visitor.agent_id,
    responder_agent_id:   "mumo-sovereign-v1",
    place_id:             "bizra-website",
    objective:            "explore-bizra",
    consent_scope:        [],           // no data exchange until explicit consent
    session_limits: SessionLimits {
      max_messages:         50,
      max_duration_seconds: 300,
      max_payload_bytes:    65536,
    },
    expires_at: now() + Duration::seconds(300),
  )

  disclosure = generate_layer_disclosure("website", session)
  response   = mumo_agent.respond(visitor_message, session)

  response.attach_disclosure(disclosure)   // mandatory per SAP v0 SC-08
  response.attach_ihsan_score()            // transparency

  receipt = record_outcome(session, response)
  response.attach_receipt(receipt)

  return response
```

### Layer 2: Investor Agent (Technical Depth)

- **Target:** Investors, technical evaluators.
- **Protocol:** SAP v0 with extended evidence chain.
- **Personality:** Precise, evidence-backed, links to conformance tests.
- **Provides:** Architecture diagrams, test results (971+ Rust tests),
  compliance reports (24/24 SAP conformance), Ihsan scores.
- **Constitutional honesty:** Never inflates metrics, always shows
  uncertainty scores alongside claims.

```pseudocode
fn investor_agent_session(evaluator_message: string) -> SAPResponse:
  session = sap::MeetOpen(
    // ... same MeetOpen shape, profile "sap-ads-retail-v0"
    objective: "evaluate-bizra-investment",
  )

  disclosure = generate_layer_disclosure("investor", session)
  // Investor layer adds evidence chain
  disclosure.source_refs.extend([
    { ref_hash: hash("ci.yml"),      ref_type: "document",           ref_uri: ".github/workflows/ci.yml" },
    { ref_hash: hash("cargo-tests"), ref_type: "human_attestation",  ref_uri: "bizra-omega/ (971+ tests)" },
    { ref_hash: hash("sap-conf"),    ref_type: "document",           ref_uri: "tests/conformance/sap_v0/" },
  ])

  response = mumo_agent.respond(evaluator_message, session, depth="technical")
  response.attach_disclosure(disclosure)
  response.attach_ihsan_score()

  return response
```

### Layer 3: Developer Agent (Integration Guide)

- **Target:** Developers wanting to build on BIZRA.
- **Protocol:** SAP v0 `MeetOpen` with code examples.
- **Personality:** Technical, concise, links to source code.
- **Provides:** Integration guides referencing `bizra-omega/` crates,
  conformance test references, bridge protocol documentation from
  `filedfs/bizra-bridge.mjs`.

```pseudocode
fn developer_agent_session(dev_message: string) -> SAPResponse:
  session = sap::MeetOpen(
    objective: "integrate-with-bizra",
  )

  disclosure = generate_layer_disclosure("developer", session)
  // Developer layer adds code reference links
  disclosure.source_refs.extend([
    { ref_hash: hash("bridge"),    ref_type: "document", ref_uri: "filedfs/bizra-bridge.mjs" },
    { ref_hash: hash("useNode"),   ref_type: "document", ref_uri: "filedfs/useNode.js" },
    { ref_hash: hash("executor"),  ref_type: "document", ref_uri: "bizra-omega/bizra-node/src/action_executor.rs" },
  ])

  response = mumo_agent.respond(dev_message, session, depth="code-level")
  response.attach_disclosure(disclosure)
  response.attach_ihsan_score()

  return response
```

---

## 2. Chat UI Extensions

Extend the existing `filedfs/App.jsx` `Bubble` component (line 87) to
display SAP protocol metadata. The current `Bubble` accepts `{role, content,
meta}` where `meta` carries agent count and Ihsan score. The SAP extension
adds disclosure, receipt, and verification data.

```pseudocode
component SAPMessageBubble(message: SAPMessage):
  // Inherits Bubble styling from filedfs/App.jsx line 87-100
  render:
    <Bubble role={message.role} content={message.content} meta={message.meta}>

      if message.disclosure:
        <DisclosurePanel>
          <Claims>
            for claim in message.disclosure.claims:
              <ClaimItem text={claim} />
          </Claims>
          <Uncertainty>
            // Schema requires: score (0-1), method, notes
            // See schemas/sap/v0/disclosure.schema.json
            <Score value={message.disclosure.uncertainty.score} />
            <Method text={message.disclosure.uncertainty.method} />
            <Notes text={message.disclosure.uncertainty.notes} />
          </Uncertainty>
          <SourceRefs>
            for ref in message.disclosure.source_refs:
              <SourceRefLink hash={ref.ref_hash} type={ref.ref_type} uri={ref.ref_uri} />
          </SourceRefs>
        </DisclosurePanel>

      if message.ihsan_score:
        // Mirrors IhsanBar from filedfs/App.jsx line 48-60
        // Green if >= 9500, amber if >= 8000, red otherwise
        <IhsanBadge score={message.ihsan_score} threshold={9500} />

      if message.receipt:
        <ReceiptLink receipt_hash={message.receipt.receipt_hash} />

      if message.sovereign_agent_card:
        <VerificationLink card={message.sovereign_agent_card} />

    </Bubble>
```

---

## 3. Transparency Panel

A collapsible panel shown alongside the chat. Surfaces all SAP session
metadata in human-readable form. Serves the constitutional requirement of
SC-08 (Disclosure Transparency).

```pseudocode
component TransparencyPanel(session: SAPSession):
  render:
    <Panel title="Transparency">

      <Section title="Active Disclosure">
        <ClaimsList claims={session.disclosure.claims} />
        <UncertaintyDisplay
          score={session.disclosure.uncertainty.score}
          method={session.disclosure.uncertainty.method}
          notes={session.disclosure.uncertainty.notes}
        />
      </Section>

      <Section title="Compliance">
        // compliance_assertions is string[] per disclosure.schema.json
        <ComplianceBadges assertions={session.disclosure.compliance_assertions} />
      </Section>

      <Section title="Session Integrity">
        // Receipt chain: each OutcomeReceipt links via prev_receipt_hash
        // See bizra-omega/bizra-node/src/action_executor.rs line 48
        <ReceiptChain receipts={session.outcome_receipts} />
        <HashVerification
          current_hash={session.latest_receipt_hash}
          genesis_hash={[0u8; 32]}
        />
      </Section>

      <Section title="Agent Identity">
        <SovereignAgentCard card={session.agent_card} />
        <CompilationStats>
          <Stat label="Conversations ingested" value={7000} />
          <Stat label="Platforms"              value={10} />
          <Stat label="Compiled reflexes"      value={session.agent_card.compilation.compiled_reflex_count} />
          <Stat label="Compilation coverage"   value={session.agent_card.compilation.compilation_coverage} />
          <Stat label="Ihsan threshold"        value={session.agent_card.compilation.ihsan_threshold} />
        </CompilationStats>
      </Section>

      <Section title="Session Limits (SAP v0)">
        <Limit label="Messages"  used={session.message_count} max={50} />
        <Limit label="Duration"  used={session.elapsed_seconds} max={300} />
        <Limit label="Payload"   used={session.payload_bytes} max={65536} />
      </Section>

    </Panel>
```

---

## 4. Bridge Protocol Extension

Extend `filedfs/useNode.js` (Tauri/browser bridge) and
`filedfs/bizra-bridge.mjs` (WebSocket stdio bridge) with SAP-specific verbs.
The existing bridge uses `{ verb, args }` JSON protocol over WebSocket
(see `bizra-bridge.mjs` line 26-28).

```pseudocode
// New SAP verbs added to the bridge protocol
enum BridgeVerb:
  // Existing verbs (filedfs/useNode.js line 46+)
  RECEIVE           // Send user message, get agent response
  STATUS            // Get node status
  ROSTER            // List active agents

  // New SAP verbs
  SAP_MEET_OPEN     // Initiate SAP session
  SAP_MESSAGE       // Send message within active SAP session
  SAP_DISCLOSURE    // Request current session disclosure
  SAP_CONSENT_REQ   // Request consent for specific scopes
  SAP_CONSENT_REV   // Revoke previously granted consent
  SAP_SESSION_CLOSE // Close SAP session with final receipt

fn handle_sap_verb(verb: BridgeVerb, args: JSON) -> JSON:
  match verb:
    SAP_MEET_OPEN:
      // Validate required MeetOpen fields per 01-core-primitives.md Section 3
      validate_meet_open(args)
      session = create_sap_session(args.profile, args.initiator_agent_id)
      return {
        ok: true,
        fields: {
          session_id: session.id,
          disclosure: session.initial_disclosure,
          agent_card: session.agent_card,
          expires_at: session.expires_at,
        }
      }

    SAP_MESSAGE:
      validate_session_active(args.session_id)
      enforce_session_limits(args.session_id)  // SC-03
      response = process_sap_message(args.session_id, args.content)
      return {
        ok: true,
        fields: {
          content:    response.content,
          disclosure: response.disclosure,
          receipt:    response.receipt,
          ihsan:      response.ihsan_score,
        }
      }

    SAP_DISCLOSURE:
      return {
        ok: true,
        fields: { disclosure: get_current_disclosure(args.session_id) }
      }

    SAP_CONSENT_REQ:
      // SC-05: Consent before data sharing
      receipt = request_consent(args.session_id, args.scopes)
      return {
        ok: true,
        fields: { consent_receipt: receipt }
      }

    SAP_CONSENT_REV:
      revoke_consent(args.session_id, args.consent_receipt_id)
      return { ok: true, fields: { revoked: true } }

    SAP_SESSION_CLOSE:
      final_receipt = close_sap_session(args.session_id)
      return {
        ok: true,
        fields: { final_receipt: final_receipt }
      }
```

Hook integration for `useNode.js`:

```pseudocode
// Extension to useNode hook (filedfs/useNode.js)
fn useNodeSAP():
  { send, status, connected } = useNode()  // existing hook

  sapMeetOpen = async (profile, objective) =>
    return await send("SAP_MEET_OPEN", { profile, objective })

  sapMessage = async (session_id, content) =>
    return await send("SAP_MESSAGE", { session_id, content })

  sapRequestConsent = async (session_id, scopes) =>
    return await send("SAP_CONSENT_REQ", { session_id, scopes })

  sapRevokeConsent = async (session_id, consent_receipt_id) =>
    return await send("SAP_CONSENT_REV", { session_id, consent_receipt_id })

  sapClose = async (session_id) =>
    return await send("SAP_SESSION_CLOSE", { session_id })

  return { sapMeetOpen, sapMessage, sapRequestConsent, sapRevokeConsent, sapClose, status, connected }
```

---

## 5. SovereignAgentCard Display

The `SovereignAgentCard` struct follows the canonical shape from
`specs/sap-v0/01-core-primitives.md` Section 1. Required fields: `agent_id`,
`owner_node_id`, `role`, `policy_hash`, `capabilities`, `endpoints`,
`version`, `compilation`.

```pseudocode
struct SovereignAgentCard:
  agent_id:       "mumo-sovereign-v1"
  owner_node_id:  "node0-genesis"
  role:           "sovereign-marketing-agent"
  policy_hash:    blake3::hash(constitution)
  capabilities:   ["natural_language", "code_review", "architecture", "marketing"]
  endpoints:      [{ protocol: "ws", uri: "ws://localhost:9100" }]
  version:        "alpha-100"
  compilation:
    genesis_version:        "alpha-100"
    ihsan_threshold:        0.95
    compiled_reflex_count:  TBD     // populated at compile time
    compilation_coverage:   0.92
    source_count:           7000    // extension field
    platform_count:         10      // extension field
    last_compiled:          ISO8601 // extension field

component SovereignAgentCardDisplay(card: SovereignAgentCard):
  render:
    <Card>
      <Header>
        <AgentName>{card.agent_id}</AgentName>
        <VersionBadge>{card.version}</VersionBadge>
      </Header>

      <Section title="Capabilities">
        for cap in card.capabilities:
          <CapabilityBadge>{cap}</CapabilityBadge>
      </Section>

      <Section title="Compilation">
        <Stat label="Reflexes"   value={card.compilation.compiled_reflex_count} />
        <Stat label="Coverage"   value={format_percent(card.compilation.compilation_coverage)} />
        <Stat label="Sources"    value={card.compilation.source_count} />
        <Stat label="Platforms"  value={card.compilation.platform_count} />
        <Stat label="Ihsan gate" value={card.compilation.ihsan_threshold} />
      </Section>

      <Section title="Verification">
        <PolicyHash hash={card.policy_hash} />
        <Endpoint protocol={card.endpoints[0].protocol} uri={card.endpoints[0].uri} />
      </Section>
    </Card>
```

---

## 6. Data Flow Diagram

```
  Visitor / Investor / Developer
       |
       v
  +------------------------------+
  |  Layer Selection              | <-- audience detection (explicit or inferred)
  |  (Website / Investor / Dev)   |
  +------------------------------+
       |
       v
  +------------------------------+
  |  SAP v0 MeetOpen             | --> Initial Disclosure (SC-08)
  |  session_id + consent_scope  |     SovereignAgentCard
  |  session_limits (50/300/64K) |
  +------------------------------+
       |
       v
  +------------------------------+
  |  Chat UI (App.jsx extended)  |
  |  +------------------------+  |
  |  | SAPMessageBubble       |  |
  |  | TransparencyPanel      |  |
  |  | IhsanBadge (line 48)   |  |
  |  | VerificationLink       |  |
  |  +------------------------+  |
  +------------------------------+
       |
       v
  +------------------------------+
  |  Bridge (bizra-bridge.mjs)   | <-- SAP verbs over WebSocket
  |  WS protocol: {verb, args}   |     Port 9100 (default)
  +------------------------------+
       |
       v
  +------------------------------+
  |  Mumo's Compiled Agent       | <-- reflexes from Phase 02
  |  (action_executor.rs)        |     PermitUsage tracking
  |  (ActionReceipt chain)       |     prev_receipt_hash linkage
  +------------------------------+
       |
       v
  SAPResponse + Disclosure + OutcomeReceipt
```

---

## 7. Layer Selection Strategy

```pseudocode
enum AudienceLayer:
  Website     // default
  Investor
  Developer

fn detect_audience(first_message: string, url_params: Map) -> AudienceLayer:
  // Priority 1: Explicit URL parameter
  if url_params.contains("layer"):
    return AudienceLayer::from(url_params["layer"])

  // Priority 2: Explicit button selection on landing page
  if url_params.contains("audience"):
    return AudienceLayer::from(url_params["audience"])

  // Priority 3: Keyword inference from first message (optional, deferred)
  // investor_keywords = ["invest", "funding", "valuation", "roi", "traction"]
  // developer_keywords = ["api", "integrate", "sdk", "rust", "crate", "npm"]
  // NOTE: This is an open question — see Section 9

  // Default: Website layer
  return AudienceLayer::Website
```

---

## 8. TDD Anchors

Each anchor specifies the test name, the property under test, and the
relevant specification reference.

| # | Test Name | Property | Spec Ref |
|---|-----------|----------|----------|
| 1 | `test_sap_meet_open_creates_session` | MeetOpen returns valid session with all required fields and initial disclosure | `01-core-primitives.md` Sec 3 |
| 2 | `test_sap_message_includes_disclosure` | Every response carries non-empty `disclosure` field | SC-08 |
| 3 | `test_disclosure_claims_not_empty` | `claims` array has `minItems: 1` per schema | `disclosure.schema.json` |
| 4 | `test_disclosure_uncertainty_present` | `uncertainty` object has `score`, `method`, `notes` per schema | `disclosure.schema.json` |
| 5 | `test_disclosure_source_refs_valid` | Each `source_ref` has `ref_hash` (64 hex chars) and `ref_type` enum | `disclosure.schema.json` |
| 6 | `test_ihsan_badge_renders_above_threshold` | Score >= 9500 renders green badge (matches `IhsanBar` line 50 logic) | `filedfs/App.jsx` |
| 7 | `test_ihsan_badge_renders_below_threshold` | Score < 9500 but >= 8000 renders amber warning | `filedfs/App.jsx` |
| 8 | `test_transparency_panel_shows_receipt_chain` | Panel displays hash-linked OutcomeReceipts with `prev_receipt_hash` | SC-06 |
| 9 | `test_sovereign_agent_card_required_fields` | Card has all 8 required fields from `01-core-primitives.md` Sec 1 | `01-core-primitives.md` |
| 10 | `test_bridge_sap_verbs_round_trip` | SAP commands survive WS serialization via `bizra-bridge.mjs` protocol | `bizra-bridge.mjs` |
| 11 | `test_layer_selection_website_default` | No params and no keywords -> Website layer | This spec Sec 7 |
| 12 | `test_layer_selection_explicit_param` | `?layer=investor` -> Investor layer | This spec Sec 7 |
| 13 | `test_constitutional_honesty_gate` | Response containing unsupported superlatives rejected by Ihsan gate | SC-08, Ihsan >= 0.95 |
| 14 | `test_session_close_produces_final_receipt` | `SAP_SESSION_CLOSE` returns OutcomeReceipt with valid `receipt_hash` | SC-06 |
| 15 | `test_session_limits_enforced` | Message 51 in a session returns error (SC-03 hard stop) | SC-03 |

---

## 9. Open Questions

| # | Question | Proposed Resolution |
|---|----------|---------------------|
| 1 | Audience detection: explicit selection vs LLM-inferred from first message? | Explicit selection via URL params or landing-page buttons. LLM inference deferred to Beta-1K. |
| 2 | Session persistence: localStorage vs server-side? | Both. localStorage primary (sovereignty), server-side optional for cross-device (consent-gated). |
| 3 | Rate limiting for public website agent? | 20 messages per session before requiring consent extension. Matches SAP v0 `max_messages <= 50` ceiling with a softer courtesy gate at 20. |
| 4 | Should the Transparency Panel be collapsed by default? | Yes on mobile, expanded on desktop. Always accessible via toggle. |
| 5 | IhsanBar reuse: extract from App.jsx into shared component? | Yes. `IhsanBar` (line 48-60) and `KnowsMeGauge` (line 26-43) should become shared UI primitives. |
