# BIZRA Definition of Done
## Governance Specification for Quality Assurance

**Document Version:** 1.0  
**Last Updated:** 2026-03-29  
**Status:** LIVE  
**Governance Layer:** Quality & Acceptance

---

## I. Master Definition of Done (10 Core Criteria)

Every delivered feature, agent capability, or phase completion must satisfy all 10 criteria below. Partial satisfaction is rejection.

| Criterion | Definition | Verification Method |
|-----------|-----------|---------------------|
| **Invokable** | Feature can be accessed and used by end user or system agent | Functional test; user successfully invokes feature |
| **Real Work** | Produces measurable, auditable improvement to user condition | Proof of Impact verification; baseline-to-outcome comparison |
| **Bounded Permissions** | All execution occurs within declared permission scope; no elevation | Permission trace audit; cross-reference to mission_id scope |
| **Receipt & Proof** | Every execution generates cryptographically signed proof | Receipt hash verification; audit trail complete from input to output |
| **Governed Memory Update** | State change is recorded and cross-linked to receipt | Memory contract inspection; Nexus/Integrator confirmation |
| **Truth-Labeled** | All claims about feature status bear explicit truth class | Claim audit; every non-trivial statement is labeled LIVE/VERIFIED/VALIDATED/WIRED/PLANNED/VISION |
| **Measurable KPIs** | Feature impact is quantifiable against accepted metrics | KPI dashboard shows baseline, actual, delta; within acceptable range |
| **Explicit Failure Mode** | System explicitly refuses when conditions are not met; no silent degradation | Test cases confirm visible refusal; no hidden failures allowed |
| **Tests & Verification** | Feature has automated tests; verification passes prior to release | Test coverage ≥ 80% of code paths; all critical paths tested |
| **No Assumption/RIBA Violation** | Feature does not accept unverified user claims; does not implement interest-based extraction | Assumption detector run; RIBA compliance audit; both pass |

**Rule:** If any single criterion fails, feature is rejected. No exceptions.

---

## II. Phase 1 Definition of Done: Win One User

**Primary Goal:** Make one user fully sovereign over personal compute, able to complete real tasks, with complete proof and audit trail.

### Phase 1 DoD Criteria

#### User-Facing Capabilities

| Capability | Acceptance Criteria | Verification |
|------------|-------------------|--------------|
| **Mission Intake** | User can state intent in natural language; system parses and converts to Mission State Contract | User states 5 different intents; system generates valid Mission State objects for all 5 |
| **Visual Mission Understanding** | User sees decomposed subtasks before execution; can understand why system proposes each step | UI shows task tree; user can explain each node in their own words |
| **Permission Transparency** | Before execution, user sees exactly which agents will run and what they will access | Permission dialog lists all agents, their actions, and scope boundaries |
| **Execution Visibility** | User can watch task execution in real time (with option for speed-up) | Log stream shows all agent steps; user can pause and inspect |
| **Receipt Inspection** | User can view and verify receipt immediately after completion | Receipt interface shows input hash, process hash, output hash; user can copy and validate |
| **Proof Chain Explanation** | User can understand how output was derived from input without needing to read code | Herald/Publisher generates human-readable proof narrative; user comprehends it |
| **State Update Confirmation** | User sees what memory state changed and why | Nexus/Integrator provides before/after state view |
| **Reflex Learning** | User is offered opportunity to automate a pattern they executed manually twice | System detects pattern, shows it explicitly, asks permission before automation |
| **Local Wallet** | User has local private key and can view balance | Wallet UI shows keypair, can export, balance displays accurately |
| **Rejection Clarity** | When system refuses a task, user understands why and what would be required to proceed | Refusal message shows: which rule blocked it, why that rule exists, what would satisfy it |

#### Technical Infrastructure

| Component | Acceptance Criteria | Verification |
|-----------|-------------------|--------------|
| **PAT-7 Decomposition** | All 7 agents (Atlas, Oracle, Forge, Judge, Crown, Herald, Nexus) successfully coordinate on at least one mission | Execution logs show all 7 agents participated; each had decision point |
| **HDA Execution** | Tasks execute on user's local machine without requiring network | Network interface disabled; task completes successfully |
| **Constitutional Checks** | All 9 steps of Constitutional Chain execute in order without skipping | Execution trace shows all 9 steps; no step is omitted; order is maintained |
| **Receipt Generation** | Cryptographic hash successfully binds input→process→output | Receipt hash is reproducible; re-execution produces identical hash |
| **Signature Verification** | User's identity key successfully signs all receipts; signatures are verifiable | Signature verification tool confirms signature is valid |
| **Memory State Tracking** | Mission completion updates internal memory state; state is persistent across sessions | After session restart, prior mission state is correctly recovered |
| **5 Bounded Templates** | System ships with 5 reusable mission templates; each is valid and invocable | Each template successfully executes; produces Proof of Impact |
| **Assumption Detection** | System explicitly flags all unverified user claims | Test: user claims "X is true" without evidence; system responds "ASSUMPTION DETECTED" |
| **RIBA Compliance** | System rejects any feature requesting interest-bearing economics | Test: attempt to charge subscription; system refuses with explanation |
| **Local Database** | All user data is stored locally; no sync to cloud by default | Database file is in user directory; verified to contain user data only |

#### Quality Standards

| Dimension | Target | Verification |
|-----------|--------|--------------|
| **Code Coverage** | ≥ 80% of critical paths tested | Coverage report shows metric met |
| **Documentation** | Every agent role, permission, and decision point documented | Doc review: can trace feature from spec to code to test |
| **Error Messages** | All error outputs are user-facing and actionable | QA: each error message explains the problem and suggests remedy |
| **Uptime** | System runs continuously for 72 hours without crash | Stress test: 72-hour continuous operation, no unplanned restarts |
| **Truth Labels** | 100% of feature claims are labeled with truth class | Claim audit: every claim in UI and docs bears LIVE/VERIFIED/VALIDATED/WIRED/PLANNED/VISION |

### Phase 1 Required Deliverables

- [ ] Mission Intake UI (web or desktop)
- [ ] PAT-7 agent layer (Python/Rust orchestration)
- [ ] Crown/Verifier constitutional check system
- [ ] Receipt generation and cryptographic signing
- [ ] Local memory contract and state management
- [ ] Identity key generation and storage
- [ ] 5 bounded mission templates (with documentation)
- [ ] HDA integration for local execution
- [ ] Assumption detector (blocks unverified claims)
- [ ] RIBA compliance scanner
- [ ] Full test suite (≥80% coverage)
- [ ] User-facing documentation explaining proof system

### Phase 1 Exit Criteria

- [ ] **All 10 Master DoD criteria satisfied**
- [ ] **All Phase 1 DoD capabilities verified**
- [ ] **KPI thresholds met:**
  - Mission Success Rate ≥ 70%
  - Receipt Coverage = 100%
  - Silent Action Rate = 0%
  - Truth Label Accuracy = 100%
- [ ] **Truth label audit passed** (no undeclared truth class)
- [ ] **Red team review passed** (external team attempts to break system; system survives or explicitly refuses)
- [ ] **User acceptance testing passed** (non-developer user successfully completes 5 missions independently)

---

## III. Phase 2 Definition of Done: Skills Market

**Primary Goal:** Enable third-party agents to publish bounded capabilities; users can discover, verify, and license skills with transparent royalty economics.

### Phase 2 DoD Criteria

#### Marketplace Infrastructure

| Component | Acceptance Criteria | Verification |
|-----------|-------------------|--------------|
| **Skill Publishing** | Agent can publish skill with: name, description, capability, permission scope, price (or RIBA-free model) | Test: third-party agent publishes skill; appears in marketplace within 24h |
| **Provenance Chain** | Published skill shows: creator identity, publication date, signature, update history | Skill page displays full provenance; each version is cryptographically linked |
| **Capability Attestation** | Skill demonstrates actual capability via PoI before approval | Attestation flow: creator runs skill, generates PoI, attaches to listing |
| **Price/Royalty Mechanism** | Skill can specify: per-use fee, license term, royalty split, or no-cost-open-source model | Marketplace supports all 4 models; pricing is enforced correctly |
| **Atomic Settlement** | User licenses skill; payment executes; creator receives share immediately | Payment flow: user clicks purchase, token transfer completes in < 1 second |
| **Denial Mechanism** | Marketplace can reject skills that violate constitution (RIBA, exploitation, unverified claims) | Marketplace rejects at least 1 test skill with clear explanation |
| **Review & Appeals** | Rejected skill creator can appeal; community can vote on controversial skills | Appeal process: creator submits appeal, community votes in < 7 days |

#### Quality & Safety

| Dimension | Target | Verification |
|-----------|--------|--------------|
| **Permission Scope Verification** | Skill's declared permissions match actual execution behavior | Sandboxed execution: monitor actual system calls; verify they fall within declared scope |
| **Proof of Impact Authenticity** | PoI claims cannot be faked; attestation requires real outcome | Spot check: 10% of new skills receive PoI audit; verify outcomes are real |
| **No Exploitative Skills** | Marketplace blocks: subscription traps, interest-bearing arrangements, value extraction without production | Red team: attempt to publish 5 exploitative test skills; marketplace rejects all 5 |
| **Safety Rating** | Community can rate skills; low-rated skills appear lower in search | After 50 uses, skill has safety rating visible on listing |

### Phase 2 Required Deliverables

- [ ] Skill publishing interface and backend
- [ ] Provenance tracking system (immutable audit trail)
- [ ] Capability attestation mechanism (PoI verification before approval)
- [ ] Marketplace discovery and search interface
- [ ] License management system (per-use, term-based, free options)
- [ ] Royalty calculation and settlement smart contract
- [ ] Skill denial system with appeal process
- [ ] Community voting mechanism for controversial skills
- [ ] Sandbox environment for skill execution testing
- [ ] Safety rating system
- [ ] Documentation: skill creator guide, marketplace policies

### Phase 2 Exit Criteria

- [ ] **All 10 Master DoD criteria satisfied**
- [ ] **All Phase 2 DoD capabilities verified**
- [ ] **KPI thresholds met:**
  - 5+ third-party skills published
  - PoI Eligibility Accuracy = 100%
  - Non-Exploitative Revenue Ratio = 100%
  - Marketplace Denial Rate ≥ 1% (reject bad skills, not all submissions)
  - Gini coefficient ≤ 0.35 (fair distribution of skill royalties)
- [ ] **No exploitative skills live** (marketplace successfully blocked test attacks)
- [ ] **Creator satisfaction ≥ 4.0/5** (survey: creators feel treated fairly)

---

## IV. Phase 3 Definition of Done: Network Effect

**Primary Goal:** Enable coordination between sovereign users and agents; maintain constitutional guarantees across network boundaries.

### Phase 3 DoD Criteria

#### Federation Infrastructure

| Component | Acceptance Criteria | Verification |
|-----------|-------------------|--------------|
| **Agent Discovery** | Agents can publish a "card" (identity, capabilities, cost, reviews) visible to network | Agent card published; appears in peer discovery within 1 minute |
| **Peer-to-Peer Connection** | Sovereign nodes can form direct connections without central server | Two nodes connect; exchange agent cards; establish bidirectional communication |
| **Remote Task Delegation** | User's node can request task from remote specialist agent; execution happens remotely with full proof | User requests remote skill; remote agent executes; receipt and PoI return home |
| **Byzantine Tolerance** | Network survives introduction of malicious node; bad actors are isolated | Red team: introduce node that lies about capabilities; honest nodes detect and exclude |
| **Federated Verification** | Constitutional rules apply across network boundary; remote execution cannot violate local constitution | Remote agent attempts to execute prohibited action; Crown/Verifier blocks even on network edge |
| **Capability Tokens** | Agents can prove skills via cryptographic token (not just claims) | Agent token can be inspected by requesting node; proves past successful executions |
| **URP Leasing** | Users can lease spare capacity to the network; earn revenue from usage | User enables URP; network uses their compute; user receives proportional PoI |

#### Coordination Protocols

| Dimension | Target | Verification |
|-----------|--------|--------------|
| **A2A Messaging** | Agent-to-Agent messages are authenticated, encrypted, and verifiable | Test: A2A message transmitted; receiver verifies sender, integrity, and freshness |
| **Distributed Consensus** | Multi-agent decisions are made without central arbitration | 3-agent scenario: reach consensus on task outcome; all agree on canonical receipt |
| **Conflict Resolution** | Network resolves disputes (e.g., "did this task actually succeed?") without human intervention | Simulate disputed task; network votes; loser's evidence is auditable |
| **Gossip Protocol** | Network shares trust signals (reputation, PoI) without central registry | Node publishes reputation; peers receive update within 5 minutes |

### Phase 3 Required Deliverables

- [ ] Agent discovery and peer registry
- [ ] Peer-to-peer networking layer (TeleScript protocol)
- [ ] Remote task delegation system
- [ ] Network-wide constitutional rule enforcer
- [ ] Capability token issuance and verification
- [ ] Byzantine fault tolerance (adversarial node handling)
- [ ] URP (User Resource Pool) capacity marketplace
- [ ] Distributed consensus algorithm
- [ ] Reputation and trust signal distribution
- [ ] Network monitoring and health dashboard
- [ ] Federation documentation

### Phase 3 Exit Criteria

- [ ] **All 10 Master DoD criteria satisfied**
- [ ] **All Phase 3 DoD capabilities verified**
- [ ] **KPI thresholds met:**
  - 10+ nodes successfully coordinate on shared mission
  - Federation Safety Rate = 100% (constitutional rules preserved)
  - Remote Specialist Success Rate ≥ 90%
  - Malicious Node Detection Rate = 100%
  - Byzantine Tolerance verified (network survives 33% bad nodes)
- [ ] **Red team test passed** (adversarial nodes cannot break consensus or steal value)
- [ ] **Peer satisfaction ≥ 4.0/5** (survey: network participants feel secure)

---

## V. Phase 4 Definition of Done: 8B Reach

**Primary Goal:** Enable low-friction, low-resource deployment to billions of devices while maintaining constitutional sovereignty.

### Phase 4 DoD Criteria

#### Installation & Deployment

| Component | Acceptance Criteria | Verification |
|-----------|-------------------|--------------|
| **3-Tap Installer** | User goes from zero to working system in 3 taps/clicks maximum | Usability test: non-technical user completes install in < 3 minutes, 3 interactions max |
| **Low-Resource Micro-Node** | System runs on 2GB RAM, 500MB storage device | Deploy on 2GB Raspberry Pi; full Phase 1 functionality works |
| **Offline-First** | All core functionality available without network; network is optional enhancement | Disconnect network; user completes full mission cycle; reconnect syncs automatically |
| **Family Profiles** | Multiple users on one device with isolated identities and data | 4 family members create separate profiles; data isolation verified; no cross-contamination |
| **Mobile Companion** | Smartphone app provides mission creation and receipt inspection; delegates execution to home node | Phone creates mission; home device executes; result syncs back to phone within 5 seconds |

#### Localization & Accessibility

| Dimension | Target | Verification |
|-----------|--------|--------------|
| **Multilingual UI** | System supports ≥ 10 languages (not just English) | UI tested in: Spanish, Mandarin, Arabic, French, Portuguese, Hindi, Japanese, Swahili, Vietnamese, Korean |
| **Low-Literacy Compatibility** | Non-technical users can use system without reading dense documentation | User with < 6th-grade reading level completes mission successfully |
| **Accessibility** | System compliant with WCAG 2.1 AA (screen readers, high contrast, keyboard navigation) | Accessibility audit: all interactive elements accessible via keyboard and screen reader |
| **Visual Clarity** | UI uses simple icons and high contrast; receipt proof chain is visually traceable | Truth label colors match color-blind safe palette; proof chain can be followed visually |

#### Performance & Reliability

| Dimension | Target | Verification |
|-----------|--------|--------------|
| **Installation Size** | Installer ≤ 200MB (including runtime) | Binary size measured; ≤ 200MB including all dependencies |
| **Startup Time** | System becomes functional within 30 seconds | Measure from application launch to first mission intake available |
| **Battery Life Impact** | On mobile, system drains ≤ 1% battery per hour idle | Mobile device: run idle for 100 hours; battery consumption ≤ 1% per hour |
| **Network Resilience** | Network interruption does not lose user data or break constitutional guarantees | Kill network mid-mission; resume after reconnection; data integrity maintained |
| **Uptime Target** | System maintains ≥ 99.5% uptime under normal usage | 30-day run: < 3.6 hours downtime total, planned or unplanned |

### Phase 4 Required Deliverables

- [ ] 3-tap installer (Windows, macOS, Linux)
- [ ] Micro-node architecture (optimized for low resources)
- [ ] Offline-first database and sync engine
- [ ] Multi-user family profile system
- [ ] Mobile companion app (iOS and Android)
- [ ] Localization framework and 10-language translations
- [ ] Accessibility audit and WCAG 2.1 AA compliance
- [ ] Performance optimization and profiling tools
- [ ] Network resilience testing suite
- [ ] Deployment documentation for all platforms
- [ ] User onboarding guide (multilingual)

### Phase 4 Exit Criteria

- [ ] **All 10 Master DoD criteria satisfied**
- [ ] **All Phase 4 DoD capabilities verified**
- [ ] **KPI thresholds met:**
  - Installer Friction ≤ 3 taps
  - Installation Time ≤ 3 minutes
  - Startup Time ≤ 30 seconds
  - Low-RAM Viability verified (2GB device, full functionality)
  - Federation Safety Rate = 100%
  - Localization Coverage ≥ 10 languages
  - Accessibility Compliance: WCAG 2.1 AA
  - Battery Impact ≤ 1% per hour (mobile)
  - Uptime ≥ 99.5%
- [ ] **Scale stress test passed** (100k concurrent users; system stable)
- [ ] **Global user satisfaction ≥ 4.5/5** (survey across all regions and devices)

---

## VI. Subsystem Definition of Done

In addition to phase-level DoD, each subsystem has specific acceptance criteria.

### Mission Intake DoD

- [ ] User can state intent in natural language
- [ ] System parses intent without requiring structured input
- [ ] Mission State Contract is generated with all required fields
- [ ] User can review and approve Mission State before execution
- [ ] Mission can be rejected without penalty
- [ ] Prior missions are visible for reference

### Decomposition DoD

- [ ] Atlas/Planner generates task tree with ≥ 3 and ≤ 10 subtasks
- [ ] Each subtask is assigned to correct agent (Oracle, Forge, Judge, Crown, Herald, Nexus, Consensus, Resource, Proof, Impact, or URP)
- [ ] Subtask dependencies are explicit (no implicit ordering)
- [ ] User can see and understand task tree
- [ ] User can suggest modifications to decomposition

### Execution/HDA DoD

- [ ] Task executes within declared permission scope
- [ ] Execution produces measurable output (artifact or state change)
- [ ] All agent actions are logged
- [ ] Execution can be paused and resumed
- [ ] Execution does not require network (in Phase 1+)
- [ ] Execution fails explicitly (not silently degrading)

### Verification/FATE DoD

- [ ] Crown/Verifier confirms output matches original intent
- [ ] FATE gate calculates ethical risk score (0.0 to 1.0)
- [ ] If FATE score > 0.7, execution halts for user review
- [ ] Assumption detector flags any unverified claims
- [ ] RIBA compliance check passes
- [ ] Proof-of-derivation is generated

### Receipts/Evidence DoD

- [ ] Receipt is generated immediately after execution
- [ ] Receipt hash is reproducible (re-execution = same hash)
- [ ] Receipt is cryptographically signed with user's identity key
- [ ] Receipt signature is verifiable by user or any peer
- [ ] Receipt includes: input hash, process hash, output hash, timestamp, agent list
- [ ] Receipt is stored locally and synced (if network available)

### Memory/Integration DoD

- [ ] Mission completion updates internal state
- [ ] State update is cross-linked to receipt_id
- [ ] Prior state is readable (before/after comparison available)
- [ ] State is persisted across sessions
- [ ] State can be inspected by user
- [ ] Memory corruption is detected and reported

### Reflex Engine DoD

- [ ] Reflex Compiler detects patterns in execution history
- [ ] Pattern matches ≥ 2 prior identical or similar missions
- [ ] Reflex candidate is offered to user explicitly
- [ ] User can approve automation or decline
- [ ] Approved reflex becomes scheduled or triggered task
- [ ] Reflex execution produces receipt same as manual execution

### Marketplace/Economic DoD

- [ ] Users can publish and discover skills
- [ ] Skills have transparent pricing or free-to-use model
- [ ] Royalties are calculated and distributed atomically
- [ ] Gini coefficient is tracked and enforced (≤ 0.35)
- [ ] Zakat threshold is monitored (≥ 2.5% rebalancing)
- [ ] No interest-bearing economics (RIBA-free)

### Security DoD

- [ ] Private keys are generated and stored locally
- [ ] Private keys are never transmitted or logged
- [ ] Permission elevation is explicit and auditable
- [ ] Secrets (API keys, auth tokens) are encrypted at rest
- [ ] Secrets access is logged and attributed
- [ ] Security vulnerabilities are reported via responsible disclosure

### Public Proof Surface DoD

- [ ] Receipts are inspectable by user and peers
- [ ] Proof chains are human-readable
- [ ] Truth labels are visible on all public statements
- [ ] Marketplace reviews and ratings are public
- [ ] PoI distribution is auditable
- [ ] Constitutional rule violations are publicly recorded

---

## VII. Gate Release Criteria Summary

| Gate | Trigger | Approval | Revert Condition |
|------|---------|----------|-----------------|
| **Phase 1 → 2** | Win One User DoD met + KPIs at threshold + truth label audit passed + red team test passed | Product lead + Constitutional auditor | Silent action detected, or unresolved assumption violation |
| **Phase 2 → 3** | Skills Market DoD met + 5+ skills published + PoI attestation working + zero exploitative skills live | Product lead + Economics auditor + Marketplace reviewer | Exploitative skill discovered live, or royalty calculation error |
| **Phase 3 → 4** | Network Effect DoD met + 10+ nodes coordinate + Byzantine tolerance verified + federation safety = 100% | Product lead + Network architect + Security auditor | Consensus break-down, or constitutional rule violation in federation |
| **Phase 4 Go-Live** | 8B Reach DoD met + all KPI thresholds met + global user satisfaction ≥ 4.5/5 | CEO + Board + Community vote | Critical security vulnerability, or undisclosed assumption violation |

---

## VIII. Continuous Definition of Done Compliance

### Review Cadence

- **Per-Mission:** Crown/Verifier confirms Master DoD 10 criteria met before receipt issued
- **Daily:** Subsystem DoD checklist reviewed; any gaps logged as blocking issues
- **Weekly:** Phase gate checklist audited; progress toward current phase gate measured
- **Per-Phase-Gate:** Full DoD audit by external team; gates do not proceed without sign-off

### Enforcement

- **No feature is released until all DoD criteria are met**
- **Partial satisfaction is rejection**
- **Gaps are not deferred; they are blocking**
- **Governance violations are phase-reversions** (if Phase 3 feature violates Phase 1 DoD, system reverts to Phase 2 until fixed)

---

**End of DEFINITION_OF_DONE.md**
