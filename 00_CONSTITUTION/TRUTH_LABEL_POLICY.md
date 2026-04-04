# BIZRA Truth Label Policy
## Governance Specification for Claim Classification and Transparency

**Document Version:** 1.0  
**Last Updated:** 2026-03-29  
**Status:** LIVE  
**Governance Layer:** Narrative & Transparency

---

## I. Purpose and Rationale

Every statement about BIZRA—whether public, internal, or aspirational—must bear a truth class label that accurately reflects its verification status.

**Why This Matters:**
- Prevents assumption (ظن) from being accepted as proof
- Distinguishes between working features and future plans
- Maintains user trust through honest communication
- Enables systematic detection of canon drift
- Protects governance from mixing fantasy with reality

**Core Principle:** Do not collapse multiple truth classes into a single present-tense claim.

**Example of Violation (rejected):**
> "BIZRA provides Constitutional Trust enforcement"

This claim is ambiguous—it could mean:
- Feature is live and working (LIVE)
- Architecture is designed (PLANNED)
- Proof concept exists (VISION)

**Corrected (accepted):**
> "BIZRA provides Constitutional Trust enforcement [LIVE: local verification subsystem]; node federation compliance is [WIRED: integration path exists]; network-wide constitutional guarantees are [PLANNED: specification complete, implementation pending]"

---

## II. Truth Classes with Definitions

### LIVE — Running Now in Active Service

**Definition:** Feature is operational, tested, and executing in the current production system.

**Criteria for LIVE label:**
- Code is deployed and running
- Tests are passing in production
- Users can invoke the feature right now
- Receipts show actual execution
- Measurement data is actively being generated

**Examples:**
- "Mission Intake accepts natural language input [LIVE]"
- "Receipts are cryptographically signed [LIVE]"
- "Gini monitoring is active [LIVE]"
- "Phase 1 core agents (PAT-7) coordinate on user tasks [LIVE]"

**Update Trigger:** When code ships to production; when feature first executes

**Deprecation:** When feature is disabled or removed

---

### VERIFIED — Proven by Tests or Artifact Inspection

**Definition:** Feature is not yet LIVE, but has been proven to work via rigorous testing or code review.

**Criteria for VERIFIED label:**
- Test suite passes with ≥ 80% code coverage
- External audit has inspected implementation
- Proof-of-concept has been demonstrated
- Specifications have been verified against implementation
- Known issues are documented and mitigated

**Examples:**
- "HDA execution respects permission scope [VERIFIED: 95 test cases pass]"
- "Receipt cryptography produces reproducible hashes [VERIFIED: code audit complete]"
- "Reflex pattern detection identifies recurring tasks [VERIFIED: 10 patterns successfully detected in sandbox]"

**Update Trigger:** When testing is complete and passes threshold; when code review is approved

**Deprecation:** When code changes; when testing results are no longer valid

---

### VALIDATED — Architecture or Harness Evidence Supports It

**Definition:** Design and harness evidence strongly suggests the feature will work, even if not yet tested end-to-end.

**Criteria for VALIDATED label:**
- Architecture design document exists and is approved
- Lower-level components are proven (LIVE or VERIFIED)
- Integration path is clear and feasible
- No known blocking issues
- Cross-component dependency review is complete

**Examples:**
- "Network-wide Byzantine tolerance is achievable [VALIDATED: tolerance math proven; consensus algorithm exists; dependency analysis complete]"
- "Family profiles enable multi-user sovereignty [VALIDATED: identity isolation pattern proven in Phase 1; profile architecture approved]"
- "Multilingual UI supports 10 languages [VALIDATED: localization framework is VERIFIED; translation pipeline exists]"

**Update Trigger:** When architecture is approved and dependencies are proven

**Deprecation:** When architecture changes; when dependency assumptions break

---

### WIRED — Integration Path Exists but Not Yet Live-Proven End-to-End

**Definition:** Feature is partially implemented; integration path exists; but end-to-end execution hasn't been proven.

**Criteria for WIRED label:**
- Subsystems are at least VERIFIED or VALIDATED
- Integration glue code exists
- Integration has not been tested end-to-end
- OR integration has been tested but is not yet in production

**Examples:**
- "URP leasing enables users to monetize spare capacity [WIRED: capacity pooling is LIVE; settlement contract is VERIFIED; leasing UI is not yet built]"
- "Mobile companion syncs with home node [WIRED: mobile app is VERIFIED; sync protocol is designed; end-to-end test not yet done]"
- "Skills market prevents exploitation [WIRED: denial mechanism is VERIFIED; marketplace UI is VERIFIED; integration not yet tested with real skills]"

**Update Trigger:** When subsystems are proven and glued together; before end-to-end verification

**Deprecation:** When end-to-end integration is proven (promote to VERIFIED or LIVE); when integration breaks

---

### PLANNED — Specified, Not Yet Implemented

**Definition:** Feature is fully specified in governance documents, but no code or tests exist yet.

**Criteria for PLANNED label:**
- Requirements document exists and is approved
- Specification is clear enough for implementation
- Acceptance criteria are defined (via DoD)
- No implementation code yet
- Implementation is next in priority queue OR is scheduled for future phase

**Examples:**
- "Phase 3 network federation [PLANNED: specification complete; implementation begins Q2]"
- "Mobile Android companion app [PLANNED: design finalized; engineering estimate: 400 days]"
- "Zakat purification mechanism [PLANNED: logic specified; awaiting smart contract framework]"

**Update Trigger:** When specification is complete and approved; when implementation begins (demote to WIRED/VERIFIED)

**Deprecation:** When implementation completes; when requirement is cancelled

---

### VISION — Directional Future State Only

**Definition:** Feature is aspirational or exploratory; no specification yet; direction is clear but details are fuzzy.

**Criteria for VISION label:**
- Concept is interesting or strategic
- Problem statement is clear
- Solution is not yet designed
- Implementation timeline is uncertain (6+ months out)
- May or may not be built depending on user demand

**Examples:**
- "Integrate with external knowledge systems [VISION: enables BIZRA to reference encyclopedic knowledge; design pending]"
- "GameFi reputation system [VISION: users earn badges for community contribution; business model not finalized]"
- "Interplanetary agents [VISION: extends BIZRA to Mars-based infrastructure; requires unified time systems; long-term research]"

**Update Trigger:** When idea is sufficiently mature to merit discussion; during strategic planning

**Deprecation:** When concept is designed (promote to PLANNED); when idea is rejected

---

## III. Application Rules

### Rule 1: Every Non-Trivial Claim Must Be Labeled

**Scope:** Any statement about BIZRA that (a) describes current functionality, (b) makes a promise, or (c) claims achievement.

**Non-Trivial Examples (require labels):**
- "BIZRA provides cryptocurrency transactions" → must specify LIVE/VERIFIED/VALIDATED/WIRED/PLANNED/VISION
- "Constitutional rules are enforced" → must specify at what layer (local-only? network-wide?)
- "Users maintain complete sovereignty" → must qualify under what conditions

**Trivial Examples (no label needed):**
- "BIZRA is written in Python" (factual, not status-dependent)
- "Gini coefficient is a measure of inequality" (definitional, not feature status)
- "The user can read this document" (logically certain)

**Exception:** Historical statements (e.g., "In Phase 1, we focused on X") do not require labels if context is clear.

---

### Rule 2: Never Collapse Multiple Truth Classes

**Violation Pattern:**
> "BIZRA has autonomous execution capability"

This is ambiguous. It could mean:
- Autonomous code is live (LIVE)
- Autonomous capability is designed (PLANNED)
- Autonomous safety is proven (VERIFIED)
- etc.

**Correction:**
> "BIZRA has autonomous execution capability [PLANNED: execution model specified; safety proofs pending]. Today, users must approve all agent actions [LIVE]."

---

### Rule 3: Truth Class Precedence (LIVE > VERIFIED > VALIDATED > WIRED > PLANNED > VISION)

If a feature has multiple components at different truth levels, label the **least-proven component** as the feature's overall status.

**Example:**
- Mission Intake (LIVE)
- Decomposition (LIVE)
- Execution (LIVE)
- Byzantine fault tolerance (VALIDATED)

**Feature Status:** "Mission execution maintains Byzantine tolerance [VALIDATED: local execution is proven; network-wide consensus is architectural but not yet live-tested]"

---

### Rule 4: Public Vs. Internal Labeling

**Public Communication** (website, marketing, user docs):
- Must be LIVE or VERIFIED only
- PLANNED and VISION are acceptable only if clearly marked "Future" or "Roadmap"
- WIRED and VALIDATED are too technical; avoid in public unless context is clear

**Internal/Governance Docs** (this canon, governance meetings):
- All 6 classes are appropriate
- Use VALIDATED and WIRED liberally to communicate technical state
- Be precise about integration gaps

---

### Rule 5: Truth Class Staleness

Labels decay over time. Revalidate at these intervals:

| Class | Revalidation Interval | Trigger |
|-------|----------------------|---------|
| LIVE | Weekly | Any code change, test failure, or production incident |
| VERIFIED | Monthly | Any spec change or new dependency |
| VALIDATED | Per-phase | Any architecture change or component modification |
| WIRED | Weekly | Any subsystem change or integration difficulty |
| PLANNED | Per-phase | Any requirement change or scheduling shift |
| VISION | Quarterly | Concept exploration and feasibility updates |

**Rule:** If label is not revalidated by deadline, default to lower class (e.g., LIVE → VERIFIED if no weekly review).

---

## IV. Enforcement Mechanisms

### Automated Truth Label Scanner

Tool: Scans all public and governance documents for unlabeled claims.

**Execution:** Daily (UTC midnight)

**Report:** Lists all claims without truth class; flags over-confident language (e.g., "BIZRA is [X]" without label)

**Action:** Herald/Publisher fixes or adds labels within 24 hours

---

### Truth Label Audit Committee

**Composition:** 3 members (Constitutional auditor, Herald/Publisher, Product lead)

**Cadence:** Weekly review of all public statements; monthly review of all internal claims

**Authority:** Can request relabeling of any claim; can reject public statements that lack labels

**Escalation:** If committee disagrees on truth class, take to Consensus/Tank for resolution

---

### Canon Drift Detector

**Definition:** Systematic misalignment between labeled status and actual behavior.

**Detection Method:** 
1. Sample 100 random claims
2. Verify each claim against current code, tests, and user experience
3. Calculate accuracy: (correct labels) / (total labels) × 100%

**Target:** ≥ 95% accuracy

**Cadence:** Monthly

**Action if drift detected:**
- If drift < 95%, identify root cause and fix labels
- If drift < 90%, escalate to product lead; halt marketing claims until resolved
- If drift < 85%, potential canon review (are specs wrong or is code drifting?)

---

### Public Narrative Trust Score

**Measurement:** Post-mission user survey: "Do you believe BIZRA's claims are accurate?"

**Scale:** 1-5 (1 = don't trust, 5 = fully trust)

**Target:** ≥ 4.5/5 average

**If < 4.5/5:**
- Audit all public claims for truth accuracy
- Add citations and evidence links
- Increase truth label visibility

---

## V. Truth Label Reference Card

Use this quick reference when labeling claims:

```
┌─────────────────────────────────────────────────────────────┐
│ TRUTH LABEL DECISION TREE                                   │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│ Q: Is code deployed and running in production?             │
│ ├─ YES → [LIVE]                                            │
│ └─ NO → Continue                                           │
│                                                             │
│ Q: Is code tested with ≥80% coverage and passing?          │
│ ├─ YES → [VERIFIED]                                        │
│ └─ NO → Continue                                           │
│                                                             │
│ Q: Is architecture approved and dependencies proven?        │
│ ├─ YES → [VALIDATED]                                       │
│ └─ NO → Continue                                           │
│                                                             │
│ Q: Is code partially built and glued together?             │
│ ├─ YES → [WIRED]                                           │
│ └─ NO → Continue                                           │
│                                                             │
│ Q: Is the spec complete and approved?                       │
│ ├─ YES → [PLANNED]                                         │
│ └─ NO → [VISION]                                           │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

---

## VI. Examples by Domain

### Constitutional Enforcement

**Claim:** "BIZRA enforces constitutional rules"

**Labeling by Layer:**
- Local-only enforcement: [LIVE] (Crown/Verifier runs on user device)
- Multi-agent coordination: [VERIFIED] (tests prove all 12 agents respect rules)
- Network-wide federation: [WIRED] (A2A protocol exists; end-to-end not proven)
- Distributed Byzantine tolerance: [VALIDATED] (math proven; code not written)

**Complete Claim:**
> "BIZRA enforces constitutional rules locally [LIVE]; all 12-agent coordination respects rules [VERIFIED]; multi-node federation maintains rules [WIRED]; network survives Byzantine faults [VALIDATED]"

---

### Economic System

**Claim:** "BIZRA has a fair economic system"

**Labeling by Layer:**
- PoI token issuance: [LIVE]
- Proof of Impact verification: [LIVE]
- Gini monitoring: [LIVE]
- Zakat automatic redistribution: [PLANNED] (contract logic specified; not deployed)
- Fair payment settlement: [VERIFIED] (tests pass)
- Marketplace royalties: [WIRED] (marketplace VERIFIED; royalty calc VERIFIED; integration untested)
- Extreme inequality prevention: [VALIDATED] (mechanism proven; not yet under load)

**Complete Claim:**
> "BIZRA has a fair economic system: tokens are issued [LIVE], verified [LIVE], monitored [LIVE], and redistributed [PLANNED]. Payments settle fairly [VERIFIED]. Marketplace royalties are prepared [WIRED]. Extreme inequality is prevented [VALIDATED]."

---

### User Experience

**Claim:** "BIZRA is easy to use"

**Labeling:**
- Mission intake: [LIVE] (users can state intent)
- Permission clarity: [VERIFIED] (tested; users understand)
- Receipt explanation: [WIRED] (Herald creates narrative; Herald interface incomplete)
- Onboarding: [PLANNED] (spec complete; UI not built)
- Mobile companion: [VISION] (concept interesting; not designed)

**Complete Claim:**
> "BIZRA aims for ease of use: missions are easy to state [LIVE]; permissions are clear [VERIFIED]; receipts are explained [WIRED]; onboarding is designed [PLANNED]; mobile support is [VISION]"

---

## VII. Special Cases

### Aspirational vs. Marketing Claims

**Aspiration:** "BIZRA will be the operating system for sovereign AI"
- This is [VISION] — long-term direction, not current state
- Do not use in marketing without clarifying timeline

**Marketing Claim:** "BIZRA lets you run AI on your device"
- Specify: local execution [LIVE], remote specialists [WIRED], federation [VALIDATED]
- Do not claim network-wide guarantees until those are LIVE

---

### Competitive Claims

**Avoid:** "BIZRA is better than X"
- This requires proving X's capabilities (external factor)
- Instead: "BIZRA provides [feature] [truth_class]; competitor X provides [feature] [unknown]"

**Acceptable:** "BIZRA provides Constitutional Trust [LIVE]; no other system (to our knowledge) enforces the same guarantees [VISION]"

---

### Negative Claims (What BIZRA Does NOT Do)

**Negative claims must also be labeled:**

**Example:** "BIZRA does not harvest user data"
- This is [LIVE] (architecture proves data stays local)
- Include verification method: "verified by code audit and network monitoring"

**Example:** "BIZRA will not permit interest-bearing lending"
- This is [PLANNED] (RIBA prohibition is specified; enforcement is LIVE locally but PLANNED at network)

---

## VIII. Truth Label Formatting Standards

### In Markdown Documents
```
Claim statement here [CLASS: context]

Example:
BIZRA provides local execution [LIVE: all Phase 1 agents coordinate on user device]
```

### In User-Facing UI
```
Feature: [Badge or icon] Status
Example:
"Encrypted Vault [LIVE]" — lock icon + status indicator
```

### In Code Comments
```
// Mission State is persisted locally [LIVE: Nexus/Integrator tested]
```

### In Meeting Notes
```
Feature: Constitutional Verification
Current Status: [LIVE] local, [WIRED] network, [VISION] distributed
```

---

## IX. Truth Label Policy Change Control

This policy itself is [LIVE] effective 2026-03-29.

**Policy versioning:**
- Version 1.0 [LIVE]: Core 6-class system, enforcement rules, decision tree
- Future versions: Will be versioned; older versions will be archived; changes trigger canon review

**Policy amendments:**
- Require Constitutional auditor + Product lead approval
- Must not retroactively relabel existing claims without review
- Change log is maintained with effective dates

---

## X. Glossary

| Term | Definition |
|------|-----------|
| **Claim** | Statement about BIZRA's capabilities, status, or future direction |
| **Truth Class** | One of six categories: LIVE, VERIFIED, VALIDATED, WIRED, PLANNED, VISION |
| **Canon** | The set of governance documents (SYSTEM_INSTRUCTION_CHAIN, DEFINITION_OF_DONE, etc.) |
| **Canon Drift** | Misalignment between what canon says and what system actually does |
| **Herald/Publisher** | Agent responsible for translating technical claims into user-facing language |
| **Proof Narrative** | Human-readable explanation of how output was derived from input |

---

**End of TRUTH_LABEL_POLICY.md**
