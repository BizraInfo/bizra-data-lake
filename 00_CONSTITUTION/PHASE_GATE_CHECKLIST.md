# BIZRA Phase Gate Checklist
## Governance Specification for Phase Transitions

**Document Version:** 1.0  
**Last Updated:** 2026-03-29  
**Status:** LIVE  
**Governance Layer:** Phase Gate Control

---

## I. Master Gate Rules

These rules apply to ALL phase transitions:

### Rule 1: DoD Completion is Mandatory

Before any phase can proceed to the next phase, ALL Definition of Done criteria for the CURRENT phase must be satisfied.

- [ ] Phase N Definition of Done: 100% complete
- [ ] No exceptions; partial DoD is rejection

### Rule 2: KPI Thresholds Must Be Met

All KPIs tied to the current phase must meet or exceed their stated targets.

- [ ] Phase N KPIs: All at or above threshold
- [ ] If any KPI misses threshold, investigate root cause and remediate before proceeding

### Rule 3: Truth Label Audit Must Pass

All public and internal claims about the system must be accurate and properly labeled.

- [ ] Truth label accuracy audit: ≥ 95% accurate
- [ ] No undeclared truth classes
- [ ] Public narrative trust score: ≥ 4.5/5

### Rule 4: Never Deprioritize Phase N for Phase N+2

It is forbidden to skip ahead to build Phase 3 or Phase 4 features while Phase 1 is unresolved.

**Rule Rationale:** Each phase builds on the prior phase. Prioritizing future vanity over current proof guarantees catastrophic architectural debt.

**Example of Violation:** 
> "We'll build the network federation (Phase 3) while we finish local execution (Phase 1)"
> 
> **Result:** Network layer is broken from day one; federation rules apply to untested local guarantees

**Consequence of Violation:** Entire next phase is rejected; system reverts to prior phase until Phase N is fully resolved.

### Rule 5: External Audit Required for Gate Approval

Phase transitions cannot be approved by internal team alone.

- [ ] External security audit (if phase includes any new trust boundary)
- [ ] External user testing (if phase includes new user-facing features)
- [ ] External economist review (if phase includes economic mechanisms)

---

## II. Phase 1 Gate: Win One User

**Gate Goal:** Prove that one user can be fully sovereign, complete real work, with complete cryptographic proof.

### Phase 1 Deliverables Checklist

#### Core Infrastructure
- [ ] Mission Intake UI (web or desktop app)
  - [ ] Accepts natural language input
  - [ ] Converts input to Mission State Contract
  - [ ] Shows parsed mission back to user for approval
  - [ ] Tested with 5 different user intents

- [ ] PAT-7 Agent Layer (Python/Rust orchestration)
  - [ ] Atlas/Planner: decomposes missions into subtasks ✓
  - [ ] Oracle/Researcher: gathers facts and verifies claims ✓
  - [ ] Forge/Builder: executes tasks and creates artifacts ✓
  - [ ] Judge/Scorer: evaluates work quality ✓
  - [ ] Crown/Verifier: enforces constitutional rules ✓
  - [ ] Herald/Publisher: explains results to user ✓
  - [ ] Nexus/Integrator: updates memory state ✓
  - [ ] All 7 agents successfully coordinate on sample mission

- [ ] Constitutional Chain (9 steps)
  - [ ] Step 1: Receive Intent — implemented
  - [ ] Step 2: Convert to Mission State — implemented
  - [ ] Step 3: Decompose via PAT — implemented
  - [ ] Step 4: Execute with bounded permissions — implemented
  - [ ] Step 5: Verify via constitutional checks — implemented
  - [ ] Step 6: Emit signed proof — implemented
  - [ ] Step 7: Update governed memory — implemented
  - [ ] Step 8: Detect reflex patterns — implemented
  - [ ] Step 9: Connect verified impact to value — implemented
  - [ ] All steps execute in order without skipping

- [ ] Receipt Generation & Cryptography
  - [ ] User identity key generated and stored locally
  - [ ] Receipt hash binds input→process→output
  - [ ] Receipt is cryptographically signed
  - [ ] Signature is verifiable by user
  - [ ] Receipt persists and is retrievable
  - [ ] 100% of missions produce receipts

- [ ] Governed Memory & State Management
  - [ ] Mission state stored locally (not to cloud)
  - [ ] State updates are cross-linked to receipt_id
  - [ ] Prior state is readable (before/after comparison)
  - [ ] Memory persists across session restarts
  - [ ] Memory integrity check passes
  - [ ] User can inspect state changes

- [ ] HDA (Hybrid Desktop Agent) Execution
  - [ ] Tasks execute on user's local machine
  - [ ] Network is disabled; tasks complete successfully
  - [ ] No data leaves device without explicit permission
  - [ ] All actions are logged for audit

#### Bounded Templates
- [ ] Template 1: Task Decomposition
- [ ] Template 2: Receipt Verification
- [ ] Template 3: Memory Inspection
- [ ] Template 4: Reflex Pattern Detection
- [ ] Template 5: Permission Audit
- [ ] All templates tested and documented

#### Constitutional Enforcement
- [ ] Assumption Detector (blocks unverified claims)
  - [ ] Flags when user statement lacks evidence
  - [ ] Suggests verification path
  - [ ] Blocks execution if critical assumption unverified

- [ ] RIBA Compliance Scanner
  - [ ] Detects interest-bearing arrangements
  - [ ] Rejects subscription models
  - [ ] Enforces one-time or Proof-of-Impact payment only

- [ ] Permission Boundary Enforcement
  - [ ] Each agent has explicit scope
  - [ ] No implicit permission elevation
  - [ ] Permission chain is validated
  - [ ] All out-of-scope actions are blocked with explanation

#### Quality & Testing
- [ ] Test Suite
  - [ ] Code coverage ≥ 80% critical paths
  - [ ] All 9 constitutional chain steps tested
  - [ ] Mission Intake: 10 test cases (normal + edge cases)
  - [ ] PAT-7 coordination: 5 multi-agent scenarios
  - [ ] Receipt generation: 20 cases (normal + failure modes)
  - [ ] FATE gate: 15 ethical risk scenarios
  - [ ] Assumption detection: 10 cases
  - [ ] RIBA scanner: 10 cases

- [ ] Documentation
  - [ ] User guide: how to create missions
  - [ ] Receipt guide: how to verify proof
  - [ ] Agent guide: what each agent does
  - [ ] Template guide: pre-built mission examples
  - [ ] API guide: for external integrations (if applicable)

#### User Experience
- [ ] UI Clarity
  - [ ] Mission tree is visually understandable
  - [ ] Permission dialog is clear and minimal
  - [ ] Receipt display explains proof chain
  - [ ] Error messages are actionable

- [ ] Accessibility
  - [ ] Keyboard navigation works
  - [ ] Screen reader compatible (WCAG 2.1 A minimum)
  - [ ] High contrast mode available

### Phase 1 KPI Thresholds

| KPI | Target | Status |
|-----|--------|--------|
| Mission Success Rate | ≥ 70% | ☐ |
| Receipt Coverage | 100% | ☐ |
| Denial Trace Coverage | 100% | ☐ |
| Proof Traceability Rate | ≥ 95% | ☐ |
| Silent Action Rate | 0% | ☐ |
| Truth Label Accuracy | 100% | ☐ |
| Permission Violation Rate | 0% | ☐ |
| Verification Latency | ≤ 5 seconds | ☐ |
| Integrity Hook Coverage | ≥ 90% | ☐ |
| Secret Isolation | 100% | ☐ |

### Phase 1 Exit Criteria

- [ ] All deliverables complete and tested
- [ ] All KPI thresholds met
- [ ] Definition of Done: 100% criteria satisfied
- [ ] Truth label audit: ≥ 95% accuracy
- [ ] External security audit passed (local execution context)
- [ ] Red team test passed (system survives attack attempts)
- [ ] Non-technical user successfully completed 5 missions independently
- [ ] Product lead, Constitutional auditor, and External auditor sign-off

---

## III. Phase 2 Gate: Skills Market

**Gate Goal:** Enable third-party agents to publish and monetize bounded capabilities; users can discover, license, and verify skills with transparent economics.

### Phase 2 Deliverables Checklist

#### Marketplace Infrastructure
- [ ] Skill Publishing System
  - [ ] Agents can upload skill: name, description, capability, cost model
  - [ ] Provenance chain is recorded (creator, date, signature)
  - [ ] Version history is maintained
  - [ ] Skill appears in marketplace within 24 hours

- [ ] Capability Attestation
  - [ ] Creator runs skill to demonstrate capability
  - [ ] Proof of Impact is generated and attached
  - [ ] PoI verification confirms capability
  - [ ] Skill cannot be published without attestation

- [ ] Pricing & Royalty Models
  - [ ] Per-use licensing (micro-payments)
  - [ ] Term-based licensing (lease for X months)
  - [ ] Open-source, no-cost model
  - [ ] All models are implemented and tested

- [ ] Settlement System
  - [ ] User purchases skill → payment executes atomically
  - [ ] Creator receives royalty share immediately
  - [ ] No payment delays or escrow
  - [ ] Payment is cryptographically proof-bearing

- [ ] Skill Denial System
  - [ ] Marketplace can reject skills violating constitution
  - [ ] Rejection criteria: RIBA, unverified claims, exploitation
  - [ ] Creator receives explicit denial reason
  - [ ] Creator can appeal via governance process

- [ ] Community Review & Appeals
  - [ ] Contested skills can be voted on by community
  - [ ] Vote results within 7 days
  - [ ] Appeal process is transparent and documented

#### Quality & Safety
- [ ] Sandbox Execution Environment
  - [ ] Skills run in isolated sandbox
  - [ ] Declared permissions are enforced in sandbox
  - [ ] Actual system calls match declared scope
  - [ ] Any out-of-scope action is blocked

- [ ] Safety Rating System
  - [ ] Users can rate skills (1-5 stars)
  - [ ] Safety ratings are visible on skill listing
  - [ ] Low-rated skills appear lower in search
  - [ ] Rating data is auditable

- [ ] Proof of Impact Verification
  - [ ] Spot audits: 10% of new skills receive PoI audit
  - [ ] Audits verify outcomes are real (not faked)
  - [ ] Fake PoI attempts are logged and sanctioned

- [ ] Exploitative Skill Detection
  - [ ] Red team: attempt to publish 5 exploitative test skills
  - [ ] Marketplace rejects all 5 attempts
  - [ ] Clear denial reasons provided

#### Economic Integrity
- [ ] Gini Coefficient Tracking
  - [ ] Daily calculation of BLOOM distribution
  - [ ] Gini coefficient ≤ 0.35
  - [ ] Automatic rebalancing if exceeds threshold

- [ ] Royalty Calculation
  - [ ] Fair split: creator, agent, infrastructure
  - [ ] No hidden fees or take-rates
  - [ ] Calculation is auditable and transparent

- [ ] Marketplace Governance
  - [ ] Community can vote on royalty rates
  - [ ] Changes require supermajority approval
  - [ ] Changes are transparent and timestamped

### Phase 2 KPI Thresholds

| KPI | Target | Status |
|-----|--------|--------|
| 3rd-Party Skills Published | ≥ 5 | ☐ |
| PoI Eligibility Accuracy | 100% | ☐ |
| Non-Exploitative Revenue Ratio | 100% | ☐ |
| Gini Coefficient | ≤ 0.35 | ☐ |
| Marketplace Denial Rate | ≥ 1% | ☐ |
| Creator Satisfaction | ≥ 4.0/5 | ☐ |
| Skill Success Rate | ≥ 85% | ☐ |
| Royalty Distribution Fairness | Verified | ☐ |

### Phase 2 Exit Criteria

- [ ] All deliverables complete and tested
- [ ] All KPI thresholds met
- [ ] Definition of Done: 100% criteria satisfied
- [ ] Truth label audit: ≥ 95% accuracy
- [ ] External economist review passed (economic mechanisms)
- [ ] External user testing: creators and consumers both satisfied ≥ 4.0/5
- [ ] No exploitative skills live; denial system proven effective
- [ ] Product lead, Constitutional auditor, Economist, and External auditor sign-off

---

## IV. Phase 3 Gate: Network Effect

**Gate Goal:** Enable coordination between sovereign users and agents across network boundaries while maintaining constitutional guarantees.

### Phase 3 Deliverables Checklist

#### Federation Infrastructure
- [ ] Agent Discovery & Publishing
  - [ ] Agents can publish "card" (identity, capabilities, cost, reviews)
  - [ ] Cards are visible to network peers
  - [ ] Cards propagate within 1 minute
  - [ ] Discovery is decentralized (no central registry)

- [ ] Peer-to-Peer Networking
  - [ ] Two nodes can form direct connection
  - [ ] Connection is encrypted and authenticated
  - [ ] Agent cards exchanged upon connection
  - [ ] Bidirectional communication established

- [ ] Remote Task Delegation
  - [ ] User's node requests task from remote specialist
  - [ ] Task executes on remote agent
  - [ ] Receipt and PoI return home
  - [ ] Proof chain includes remote agent signature

- [ ] Federated Constitutional Verification
  - [ ] Remote agent cannot violate user's constitution
  - [ ] Crown/Verifier blocks prohibited actions even at network edge
  - [ ] Verification result is provable to requester

- [ ] Capability Tokens
  - [ ] Agents issue tokens proving past successes
  - [ ] Tokens are cryptographically verifiable
  - [ ] Tokens contain execution history (anonymized)
  - [ ] Requesting node can inspect token before hiring

- [ ] URP (User Resource Pool) Leasing
  - [ ] Users can enable URP to lease spare capacity
  - [ ] Network uses capacity; user earns PoI
  - [ ] Usage is metered and reported
  - [ ] Compensation is automatic and fair

#### Coordination Protocols
- [ ] A2A Messaging (Agent-to-Agent)
  - [ ] Messages are authenticated (sender is verifiable)
  - [ ] Messages are encrypted (content is private)
  - [ ] Message integrity is verifiable (no tampering)
  - [ ] Message freshness is validated (no replay)

- [ ] Distributed Consensus
  - [ ] Multi-agent scenarios: reach consensus without arbitration
  - [ ] Consensus is recorded in canonical receipt
  - [ ] All agents agree on final receipt hash
  - [ ] Consensus survives temporary network partition

- [ ] Byzantine Fault Tolerance
  - [ ] Network survives 33% malicious nodes
  - [ ] Honest nodes detect and isolate bad actors
  - [ ] Consensus remains valid despite adversaries
  - [ ] Red team: introduce malicious node; network survives

- [ ] Conflict Resolution
  - [ ] Network resolves disputes (e.g., "did task succeed?") without human
  - [ ] Voting: honest nodes outvote liars
  - [ ] Loser's evidence is auditable (why dispute resolved that way)
  - [ ] Audit trail is public and verifiable

- [ ] Gossip Protocol
  - [ ] Trust signals (reputation, PoI) shared without central registry
  - [ ] Peer shares reputation with other peers
  - [ ] Update propagates to all nodes within 5 minutes
  - [ ] No delay in reputation update

#### Network Safety
- [ ] Malicious Node Detection
  - [ ] System detects when node claims false capabilities
  - [ ] System detects when node lies about task completion
  - [ ] Detected bad actors are excluded from future tasks
  - [ ] Exclusion is recorded and communicated

- [ ] Network Partitioning
  - [ ] Network survives split into disjoint partitions
  - [ ] Partitions can reconcile when link is restored
  - [ ] Consensus rules prevent divergence
  - [ ] No data loss or contradictory state

### Phase 3 KPI Thresholds

| KPI | Target | Status |
|-----|--------|--------|
| Nodes Coordinating on Shared Mission | ≥ 10 | ☐ |
| Federation Safety Rate | 100% | ☐ |
| Remote Specialist Success Rate | ≥ 90% | ☐ |
| Byzantine Tolerance Verification | Proven | ☐ |
| Malicious Node Detection Rate | 100% | ☐ |
| Network Partition Survival | Verified | ☐ |
| Peer Satisfaction | ≥ 4.0/5 | ☐ |

### Phase 3 Exit Criteria

- [ ] All deliverables complete and tested
- [ ] All KPI thresholds met
- [ ] Definition of Done: 100% criteria satisfied
- [ ] Truth label audit: ≥ 95% accuracy
- [ ] External network security audit passed (Byzantine tolerance, partition safety)
- [ ] Red team test: malicious node successfully isolated and excluded
- [ ] Network stress test: 100 nodes coordinating without degradation
- [ ] Product lead, Constitutional auditor, Network architect, Security auditor, and External auditor sign-off

---

## V. Phase 4 Gate: 8B Reach

**Gate Goal:** Enable low-friction, low-resource deployment to billions of devices while maintaining sovereignty.

### Phase 4 Deliverables Checklist

#### Installation & Deployment
- [ ] 3-Tap Installer
  - [ ] Windows installer: 3 steps max
  - [ ] macOS installer: 3 steps max
  - [ ] Linux installer: 3 steps max
  - [ ] Usability test: non-technical user completes in < 3 minutes

- [ ] Micro-Node Architecture
  - [ ] Runs on 2GB RAM device
  - [ ] Runs on 500MB storage device
  - [ ] All Phase 1 functionality available
  - [ ] Performance acceptable (< 10 second task completion)

- [ ] Offline-First Design
  - [ ] All core functionality works without network
  - [ ] Network is optional enhancement
  - [ ] Offline work is synced upon reconnection
  - [ ] No data loss during offline operation

- [ ] Family Profiles
  - [ ] Multiple users on one device
  - [ ] Separate identities for each user
  - [ ] Data isolation verified (no cross-contamination)
  - [ ] Family member can manage own missions

- [ ] Mobile Companion App
  - [ ] iOS app available
  - [ ] Android app available
  - [ ] Mission creation on phone → execution on home device
  - [ ] Result syncs back to phone within 5 seconds
  - [ ] App works offline and syncs upon connection

#### Localization & Accessibility
- [ ] Multilingual Support
  - [ ] Spanish translation
  - [ ] Mandarin Chinese translation
  - [ ] Arabic translation
  - [ ] French translation
  - [ ] Portuguese translation
  - [ ] Hindi translation
  - [ ] Japanese translation
  - [ ] Swahili translation
  - [ ] Vietnamese translation
  - [ ] Korean translation
  - [ ] All translations tested by native speakers

- [ ] Low-Literacy Support
  - [ ] Usability test: user with < 6th-grade reading level completes mission
  - [ ] Visual icons clearly indicate actions
  - [ ] Proof chains are explained graphically

- [ ] Accessibility (WCAG 2.1 AA)
  - [ ] Keyboard navigation: all features accessible via keyboard
  - [ ] Screen reader: all interactive elements have proper labels
  - [ ] High contrast: available and verified
  - [ ] Text scaling: readable at 200% zoom

- [ ] Color-Blind Accessibility
  - [ ] Truth labels use color-blind safe palette
  - [ ] Proof chain uses shapes/icons, not just color
  - [ ] Tested with color-blindness simulator

#### Performance & Reliability
- [ ] Installation Size
  - [ ] Installer ≤ 200MB (including runtime)
  - [ ] Total installed size ≤ 500MB
  - [ ] All critical dependencies bundled

- [ ] Startup Time
  - [ ] Application becomes functional within 30 seconds
  - [ ] First mission ready within 30 seconds of launch

- [ ] Battery Impact (Mobile)
  - [ ] Idle drain: ≤ 1% per hour
  - [ ] Active mission: ≤ 5% per hour
  - [ ] Tested on iPhone and Android devices

- [ ] Network Resilience
  - [ ] Kill network mid-mission
  - [ ] Resume after reconnection
  - [ ] Data integrity maintained
  - [ ] No data loss

- [ ] Uptime Target
  - [ ] 30-day operation: ≥ 99.5% uptime
  - [ ] < 3.6 hours downtime total per month
  - [ ] Unplanned outages < 30 minutes each

#### Scale & Performance
- [ ] Concurrent Users
  - [ ] System supports 100k concurrent users
  - [ ] Stability verified under load
  - [ ] Response time degrades gracefully (not crashes)

- [ ] Database Performance
  - [ ] Queries return in < 100ms median
  - [ ] Supports 1M receipts per user
  - [ ] Sync completes in < 30 seconds for new user

### Phase 4 KPI Thresholds

| KPI | Target | Status |
|-----|--------|--------|
| Installer Friction | ≤ 3 taps | ☐ |
| Installation Time | ≤ 3 minutes | ☐ |
| Startup Time | ≤ 30 seconds | ☐ |
| Low-RAM Viability | 2GB, full function | ☐ |
| Federation Safety Rate | 100% | ☐ |
| Localization Coverage | ≥ 10 languages | ☐ |
| Accessibility Compliance | WCAG 2.1 AA | ☐ |
| Battery Impact | ≤ 1% idle/hour | ☐ |
| Uptime Target | ≥ 99.5% | ☐ |
| Global User Satisfaction | ≥ 4.5/5 | ☐ |
| Concurrent User Support | 100k+ stable | ☐ |

### Phase 4 Exit Criteria

- [ ] All deliverables complete and tested
- [ ] All KPI thresholds met
- [ ] Definition of Done: 100% criteria satisfied
- [ ] Truth label audit: ≥ 95% accuracy
- [ ] External accessibility audit: WCAG 2.1 AA passed
- [ ] External scale testing: 100k concurrent users verified
- [ ] Global user testing: representative sample from all 10+ languages
- [ ] Product lead, Constitutional auditor, UX auditor, Accessibility expert, Scale engineer, and External auditor sign-off

---

## VI. Post-Gate Checklist

After any phase gate approval, complete these steps:

### Documentation Updates
- [ ] Update SYSTEM_INSTRUCTION_CHAIN.md with new phase capabilities
- [ ] Update DEFINITION_OF_DONE.md with next phase requirements
- [ ] Update KPI_CANON.md with new KPI baselines
- [ ] Update PHASE_GATE_CHECKLIST.md with lessons learned

### Team Communication
- [ ] Announce phase transition to all stakeholders
- [ ] Publish updated roadmap
- [ ] Schedule kickoff for next phase

### Governance Records
- [ ] Record gate approval date and approvers
- [ ] Archive previous phase gate checklist (for history)
- [ ] Note any deferred items from current phase

---

## VII. Gate Reversion Protocol

If at any point a phase gate shows signs of failure, revert to prior phase.

### Reversion Triggers

- [ ] Critical security vulnerability discovered
- [ ] DoD criteria suddenly fail (e.g., regression in code)
- [ ] KPI drops below threshold and remediation is ≥ 2 weeks away
- [ ] Truth label audit shows > 15% inaccuracy
- [ ] External auditor recommends reversion

### Reversion Steps

1. Halt new feature development for next phase
2. Redirect all resources to fixing current phase
3. Retest phase gate criteria
4. Once stable, re-approve gate before advancing

### Post-Reversion

- [ ] Update governance docs with what failed and why
- [ ] Implement process changes to prevent recurrence
- [ ] Reschedule next phase attempt

---

**End of PHASE_GATE_CHECKLIST.md**
