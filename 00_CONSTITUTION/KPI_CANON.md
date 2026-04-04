# BIZRA Key Performance Indicators (KPI) Canon
## Governance Specification for Measurement and Accountability

**Document Version:** 1.0  
**Last Updated:** 2026-03-29  
**Status:** LIVE  
**Governance Layer:** Metrics & Accountability

---

## I. KPI Rules (Binding Constraints)

All KPIs in this document must satisfy these rules:

1. **Measurable** — KPI can be quantified with objective data; no subjective scoring
2. **Phase-Scoped** — KPI is specific to current phase; not mixed with future-phase targets
3. **Attributable** — KPI outcome can be traced to specific agent action or subsystem
4. **Evidence-Tied** — KPI is linked to receipt, proof-of-impact, or audit trail
5. **Reviewable** — KPI data is accessible to user and governance auditors
6. **Actionable** — If KPI misses threshold, there is clear remediation path
7. **Ownership Clear** — Single agent or team is responsible for KPI outcome

**Enforcement:** Any KPI that violates these rules is rejected during canon review.

---

## II. Product Value KPIs

These measure whether BIZRA delivers real user value.

### Mission Success Rate
**Definition:** Percentage of user-initiated missions that complete with verified Proof of Impact.

**Formula:** (Missions completed with PoI verified) / (Total missions initiated) × 100%

**Phase 1 Target:** ≥ 70%  
**Phase 2 Target:** ≥ 75%  
**Phase 3 Target:** ≥ 80%  
**Phase 4 Target:** ≥ 85%

**Measurement:** Count missions reaching VERIFIED status in memory contract

**Actionable Threshold:** If < 70%, investigate: (a) decomposition quality, (b) agent failure rates, (c) user confusion on expectations

**Owner:** Atlas/Planner + Judge/Scorer

---

### Time to First Verified Value
**Definition:** Minutes from mission initiation to first verified Proof of Impact token minted.

**Targets by Phase:**
- Phase 1: ≤ 30 minutes (user sees value quickly)
- Phase 2: ≤ 15 minutes (market reduces friction)
- Phase 3: ≤ 10 minutes (network specialists provide fast solutions)
- Phase 4: ≤ 5 minutes (mass market acceleration)

**Measurement:** Timestamp(mission_received) → Timestamp(PoI_minted)

**Actionable Threshold:** If > target, profile execution time; identify bottleneck agent

**Owner:** Resource/Healer

---

### Real Work Completion Count
**Definition:** Number of distinct user missions producing measurable, auditable improvement in user's life.

**Phase 1 Target:** ≥ 5 missions completed per user per month  
**Phase 2 Target:** ≥ 20 missions completed per user per month  
**Phase 3 Target:** ≥ 50 missions completed per user per month  
**Phase 4 Target:** ≥ 100+ missions completed per user per month

**Measurement:** Count unique mission_ids with PoI_status = VERIFIED and impact_status = IMPACT_SCORED

**Actionable Threshold:** If < target, survey users on: (a) mission clarity, (b) execution confidence, (c) time investment required

**Owner:** Impact/Support

---

### Reflex Candidate Yield
**Definition:** Percentage of missions that generate valid reflex automation candidates.

**Phase 1 Target:** ≥ 1 reflex candidate per user per month  
**Phase 2 Target:** ≥ 5 reflex candidates per user per month  
**Phase 3 Target:** ≥ 20 reflex candidates per user per month  
**Phase 4 Target:** ≥ 50 reflex candidates per user per month

**Measurement:** (Missions generating reflex_candidate = true) / (Total missions) × 100%

**Actionable Threshold:** If < target, analyze: (a) pattern detection sensitivity, (b) user behavior repetition, (c) automation eligibility

**Owner:** Reflex Compiler

---

### N=1 Independence Score
**Definition:** System capability without network dependency; measured as percentage of Phase 1 functionality available offline.

**Target:** 100% (all Phase 1 features work without network)

**Measurement:** Test each feature with network disabled; count successful completions

**Actionable Threshold:** If < 100%, identify features requiring network; migrate to local-first architecture

**Owner:** Nexus/Integrator

---

## III. Constitutional Trust KPIs

These measure whether system maintains governance and transparency.

### Receipt Coverage
**Definition:** Percentage of executed missions producing cryptographically signed receipt.

**Target:** 100% (every mission must produce receipt)

**Measurement:** (Missions with receipt_id present and valid) / (Total missions executed) × 100%

**Actionable Threshold:** If < 100%, halt all execution; investigate receipt generation failure

**Owner:** Proof/DPS + Herald/Publisher

---

### Denial Trace Coverage
**Definition:** Percentage of refused missions producing explicit refusal explanation visible to user.

**Target:** 100% (every refusal must be explained)

**Measurement:** (Missions refused with reason_code and human_explanation present) / (Total missions refused) × 100%

**Actionable Threshold:** If < 100%, review refusal messages; ensure all are user-facing and actionable

**Owner:** Crown/Verifier + Herald/Publisher

---

### Proof Traceability Rate
**Definition:** Percentage of receipts where input→output chain is auditable by user.

**Target:** ≥ 95% (small margin for cryptographic edge cases)

**Measurement:** Test sample of 100 receipts; attempt to verify proof chain; count successful traces

**Actionable Threshold:** If < 95%, audit proof-of-derivation algorithm; regenerate receipts if needed

**Owner:** Proof/DPS

---

### Silent Action Rate
**Definition:** Percentage of system actions taken without explicit user permission or predetermined constitutional rule.

**Target:** 0% (no silent actions permitted)

**Measurement:** Audit log: (Actions where permission_granted = false AND constitutional_rule_applies = false) / (Total actions)

**Actionable Threshold:** If > 0%, halt system immediately; investigate and resolve permission model

**Owner:** Crown/Verifier + URP/Leader

---

### Truth Label Accuracy
**Definition:** Percentage of public/internal claims that bear correct truth class label.

**Target:** 100% (every claim must be labeled and labeled correctly)

**Measurement:** Claim audit: sample 200 claims; verify each has truth class and class is accurate

**Actionable Threshold:** If < 100%, retrain Herald/Publisher; audit all docs for unlabeled claims

**Owner:** Herald/Publisher

---

### Verification Latency
**Definition:** Time from mission completion to Crown/Verifier validation.

**Target:** ≤ 5 seconds (user should see result almost immediately after execution)

**Measurement:** Timestamp(execution_complete) → Timestamp(verification_complete)

**Actionable Threshold:** If > 5 seconds, optimize verification logic; consider parallel verification for independent checks

**Owner:** Crown/Verifier + Proof/DPS

---

## IV. Security KPIs

These measure whether system maintains security boundaries and prevents exploitation.

### Permission Violation Rate
**Definition:** Percentage of agent actions that exceed declared permission scope.

**Target:** 0% (no permission violations permitted)

**Measurement:** (Actions exceeding declared scope) / (Total actions executed) × 100%

**Actionable Threshold:** If > 0%, halt agent; audit permission declaration and execution logic; regenerate permissions

**Owner:** Crown/Verifier

---

### Integrity Hook Coverage
**Definition:** Percentage of critical system state changes protected by automated integrity checks.

**Target:** ≥ 90% (nearly all state changes audited)

**Measurement:** Code audit: identify all state mutations; count those with pre-/post- hooks; calculate coverage

**Actionable Threshold:** If < 90%, add hooks to uncovered state mutations; no new code path without hooks

**Owner:** Proof/DPS + Judge/Scorer

---

### Secret Isolation Score
**Definition:** Percentage of credentials properly encrypted at rest and accessible only via authorized channels.

**Target:** 100% (all secrets encrypted and gated)

**Measurement:** Secrets audit: inspect vault; verify (a) encryption, (b) access control, (c) rotation status; count compliant secrets

**Actionable Threshold:** If < 100%, rotate all non-compliant secrets immediately; audit access logs

**Owner:** Security subsystem

---

### Key Locality Compliance
**Definition:** Percentage of user identity keys stored locally (never transmitted to network).

**Target:** 100% (user retains key custody at all times)

**Measurement:** Network audit: capture all outbound traffic; verify no private keys are sent

**Actionable Threshold:** If < 100%, halt key generation; investigate network leak; implement local-only key management

**Owner:** Proof/DPS

---

### Known Gap Disclosure Rate
**Definition:** Percentage of identified security vulnerabilities that are disclosed to user within 24 hours.

**Target:** 100% (no hidden vulnerabilities)

**Measurement:** (Vulnerabilities disclosed ≤ 24h) / (Total vulnerabilities identified) × 100%

**Actionable Threshold:** If < 100%, establish incident response process; meet 24h SLA on all disclosures

**Owner:** Security subsystem + Herald/Publisher

---

## V. User Experience KPIs

These measure whether users understand and trust the system.

### Mission Clarity Score
**Definition:** User-reported confidence that system understood their intent correctly.

**Scale:** 1-5 (1 = confused, 5 = perfectly clear)

**Target:** ≥ 4.5/5 average

**Measurement:** Post-mission survey: "How clear was the system's understanding of your request?"

**Actionable Threshold:** If < 4.5, analyze failed missions; retrain Atlas/Planner on common misunderstandings

**Owner:** Atlas/Planner + Herald/Publisher

---

### Permission Friction Efficiency
**Definition:** Ratio of permission requests to mission completion.

**Target:** ≤ 2 permission requests per mission

**Measurement:** (Sum of permission_requests across all missions) / (Total missions) = average requests per mission

**Actionable Threshold:** If > 2, audit permission granularity; reduce unnecessary permission gates

**Owner:** URP/Leader + Crown/Verifier

---

### Time to Understand Proof
**Definition:** Minutes required for non-technical user to understand receipt and proof chain.

**Target:** ≤ 60 seconds (user shouldn't need PhD to understand proof)

**Measurement:** Usability test: show user receipt + proof chain; measure time to explain result in own words

**Actionable Threshold:** If > 60 seconds, simplify Herald/Publisher narrative; add diagrams/visuals

**Owner:** Herald/Publisher

---

### Onboarding to First Success
**Definition:** Minutes from system installation to first completed mission with verified PoI.

**Target:** ≤ 15 minutes

**Measurement:** Usability test: new user; measure time from install to PoI minting

**Actionable Threshold:** If > 15 minutes, streamline onboarding flow; reduce initial configuration burden

**Owner:** Atlas/Planner + Herald/Publisher

---

### Public Narrative Trust Score
**Definition:** User-reported trust in system's public statements about itself.

**Scale:** 1-5 (1 = don't trust, 5 = fully trust)

**Target:** ≥ 4.5/5 average

**Measurement:** Survey: "Do you believe BIZRA's public statements are accurate and verifiable?"

**Actionable Threshold:** If < 4.5, audit all public claims; add citations and truth labels

**Owner:** Herald/Publisher + Truth Label auditor

---

## VI. Economic Integrity KPIs

These measure whether economic system is fair and exploitation-free.

### Proof of Impact Eligibility Accuracy
**Definition:** Percentage of minted PoI tokens that meet actual impact criteria (not inflated or false).

**Target:** 100% (only real impact generates tokens)

**Measurement:** Spot audit: randomly select 10% of PoI tokens issued; verify each represents real outcome

**Actionable Threshold:** If < 100%, audit impact scoring logic; regenerate tokens if needed

**Owner:** Impact/Support

---

### Non-Exploitative Revenue Ratio
**Definition:** Percentage of system revenue generated from Proof of Impact (not RIBA, subscriptions, or data sales).

**Target:** 100% (all revenue is PoI-based; no exploitation)

**Measurement:** (PoI-linked revenue) / (Total revenue) × 100%

**Actionable Threshold:** If < 100%, identify non-PoI revenue sources; migrate to PoI model or eliminate

**Owner:** Consensus/Tank + Impact/Support

---

### Gini Coefficient (Economic Fairness)
**Definition:** Measure of BLOOM distribution inequality across users.

**Scale:** 0 (perfect equality) to 1 (perfect inequality)

**Target:** ≤ 0.35 (moderate inequality only)

**Measurement:** Calculate Gini coefficient on BLOOM balances; daily automated rebalancing if > 0.35

**Actionable Threshold:** If > 0.35, trigger automatic Zakat redistribution; investigate if concentration is legitimate

**Owner:** Impact/Support + Economic auditor

---

### Zakat Compliance
**Definition:** Percentage of users with > 2.5% of circulating BLOOM having excess automatically transferred.

**Target:** 100% (all surplus captured and redistributed)

**Measurement:** Daily check: (Users receiving Zakat rebalancing) / (Users exceeding 2.5% threshold) × 100%

**Actionable Threshold:** If < 100%, audit Zakat calculation; ensure daily execution

**Owner:** Smart contract layer + Impact/Support

---

### Marketplace Provenance Coverage
**Definition:** Percentage of published skills with complete provenance chain (creator, publication date, updates).

**Target:** 100% (all skills fully traceable)

**Measurement:** Marketplace audit: sample 50 skills; verify each has full provenance; count compliant

**Actionable Threshold:** If < 100%, enforce provenance requirement at skill publish time

**Owner:** Herald/Publisher + Nexus/Integrator

---

## VII. Operational Readiness KPIs

These measure whether system infrastructure is mature and stable.

### Phase 1 Readiness
**Definition:** Percentage of Phase 1 DoD criteria met.

**Target:** 100% (no phase transition until all criteria met)

**Measurement:** Checklist audit against Definition_of_Done.md; count met criteria

**Actionable Threshold:** If < 100%, address gaps before proceeding; gaps are blocking

**Owner:** Product lead

---

### Bounded Template Availability
**Definition:** Number of pre-built, tested mission templates available to users.

**Phase 1 Target:** ≥ 5 templates  
**Phase 2 Target:** ≥ 20 templates  
**Phase 3 Target:** ≥ 100 templates  
**Phase 4 Target:** ≥ 500 templates

**Measurement:** Count published templates in template library; verify each is documented and tested

**Actionable Threshold:** If < target, create new templates; or retire unused ones to focus on quality

**Owner:** Forge/Builder

---

### Runtime Stability
**Definition:** Percentage uptime of system (system available and functional).

**Target:** ≥ 99% (≤ 3.6 hours downtime per month, planned or unplanned)

**Measurement:** Monitor system availability 24/7; calculate uptime percentage

**Actionable Threshold:** If < 99%, investigate root causes; implement fixes; plan remediation

**Owner:** Infrastructure team + Consensus/Tank

---

### Receipt Lineage Integrity
**Definition:** Percentage of receipts where cryptographic hash chain is unbroken.

**Target:** 100% (every receipt must be verifiable)

**Measurement:** Monthly integrity audit: attempt to verify 100% of receipts; count successful chains

**Actionable Threshold:** If < 100%, investigate receipt generation or storage corruption; restore from backup

**Owner:** Proof/DPS + Nexus/Integrator

---

### Canon Drift Rate
**Definition:** Percentage of system behavior that deviates from written governance canon.

**Target:** ~0% (system should match spec; drift indicates gap between theory and practice)

**Measurement:** Monthly behavior audit: sample 100 agent actions; verify each follows canonical rule; count deviations

**Actionable Threshold:** If > 0.5%, review canon; update either spec or code to eliminate drift

**Owner:** Constitutional auditor

---

## VIII. Scale Readiness KPIs

These measure whether system can handle growth from 1 user to billions.

### Installer Friction
**Definition:** Number of user interactions required to go from zero to working system.

**Target:** ≤ 3 taps/clicks

**Measurement:** Usability test: count distinct user interactions from launch to first working screen

**Actionable Threshold:** If > 3, streamline installer; eliminate optional steps

**Owner:** Forge/Builder

---

### Low-RAM Viability
**Definition:** System functionality verified on 2GB RAM device.

**Target:** 100% Phase 1 features work on 2GB device

**Measurement:** Deploy on 2GB Raspberry Pi; test each Phase 1 capability; count working features

**Actionable Threshold:** If < 100%, optimize memory usage; eliminate unnecessary caches or preload data

**Owner:** Resource/Healer

---

### Federation Safety Rate
**Definition:** Percentage of federated operations (multi-node) that preserve constitutional rules.

**Target:** 100% (constitutional rules must apply across network)

**Measurement:** Test network scenarios; verify each maintains invariants; count safe operations

**Actionable Threshold:** If < 100%, audit federation logic; enforce invariants at network boundary

**Owner:** Nexus/Integrator + Crown/Verifier

---

### Localization Coverage
**Definition:** Number of languages with full UI and documentation support.

**Target:** ≥ 10 languages

**Measurement:** Count languages with: (a) UI translated, (b) docs translated, (c) native speaker QA passed

**Actionable Threshold:** If < 10, prioritize translation; aim for languages representing 5B+ people

**Owner:** Herald/Publisher

---

## IX. Review Cadence and Accountability

### Per-Mission KPI Review
**Trigger:** Every mission completion  
**Scope:** Mission Success Rate, Receipt Coverage, Denial Trace Coverage, Silent Action Rate  
**Owner:** Crown/Verifier  
**Action:** If any KPI fails, block receipt issuance; investigate before proceeding

### Daily KPI Review
**Trigger:** 1x per day (UTC midnight)  
**Scope:** All KPIs listed in sections II–VIII  
**Owner:** Consensus/Tank  
**Action:** Dashboard refresh; alert if any KPI < threshold; auto-escalate critical metrics

### Weekly KPI Review
**Trigger:** Every Monday (UTC)  
**Scope:** Trend analysis; week-over-week deltas  
**Owner:** Product lead + Constitutional auditor  
**Action:** Review trends; identify degradation; plan remediation

### Per-Phase-Gate KPI Review
**Trigger:** Before phase transition (Phase 1→2, 2→3, 3→4)  
**Scope:** All KPIs relevant to current and next phase  
**Owner:** Executive team + External auditor  
**Action:** Gate cannot proceed unless all threshold KPIs are met

---

## X. Minimum Canon Dashboard Fields

Every KPI dashboard MUST display:

```
┌────────────────────────────────────────────────────────────┐
│ KPI DASHBOARD — BIZRA GOVERNANCE METRICS                   │
├────────────────────────────────────────────────────────────┤
│ Current Phase: [Phase 1/2/3/4]                            │
│ Last Updated: [ISO 8601 timestamp]                         │
│                                                            │
│ PRODUCT VALUE KPIs                                         │
│ ├─ Mission Success Rate: 72% [PASS: ≥70%]                │
│ ├─ Avg Time to First PoI: 18 min [PASS: ≤30m]            │
│ ├─ Real Work Completions/mo: 8 [PASS: ≥5]               │
│ ├─ Reflex Candidate Yield: 2/mo [PASS: ≥1]              │
│ └─ N=1 Independence Score: 100% [PASS: 100%]            │
│                                                            │
│ CONSTITUTIONAL TRUST KPIs                                  │
│ ├─ Receipt Coverage: 100% [PASS: 100%]                   │
│ ├─ Denial Trace Coverage: 100% [PASS: 100%]             │
│ ├─ Proof Traceability: 98% [PASS: ≥95%]                │
│ ├─ Silent Action Rate: 0% [PASS: 0%]                    │
│ ├─ Truth Label Accuracy: 100% [PASS: 100%]              │
│ └─ Verification Latency: 2.3s [PASS: ≤5s]              │
│                                                            │
│ SECURITY KPIs                                              │
│ ├─ Permission Violation Rate: 0% [PASS: 0%]             │
│ ├─ Integrity Hook Coverage: 92% [PASS: ≥90%]            │
│ ├─ Secret Isolation: 100% [PASS: 100%]                  │
│ ├─ Key Locality: 100% [PASS: 100%]                      │
│ └─ Known Gap Disclosure: 100% [PASS: 100%]              │
│                                                            │
│ ECONOMIC INTEGRITY KPIs                                    │
│ ├─ PoI Eligibility Accuracy: 100% [PASS: 100%]          │
│ ├─ Non-Exploitative Revenue: 100% [PASS: 100%]          │
│ ├─ Gini Coefficient: 0.31 [PASS: ≤0.35]                │
│ ├─ Zakat Compliance: 100% [PASS: 100%]                  │
│ └─ Marketplace Provenance: 100% [PASS: 100%]            │
│                                                            │
│ OVERALL STATUS: ✅ ALL PASS                              │
│ Next Phase Gate Ready: 2026-04-15                         │
└────────────────────────────────────────────────────────────┘
```

---

**End of KPI_CANON.md**
