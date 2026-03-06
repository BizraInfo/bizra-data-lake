# Phase 04 — ISO 9001: Quality Management System

> Source: BIZRA Quality Standards & Certification Framework
> Standard: ISO 9001:2015 — Quality management systems — Requirements
> Status: SPECIFICATION SEALED | SNR: 0.93

---

ISO 9001 is built on the Process Approach and seven Quality Management Principles. Traditional
QMS implementations bolt processes onto organizations after the fact — documented procedures,
manual audits, corrective action forms filed in binders. BIZRA inverts this entirely. The
seven ISO 9001 principles are not policies to be followed; they are mathematical invariants
enforced by the 15 Native Algorithms, measured by the append-only Event Log, and verified
continuously by the Self-Harness pipeline.

Customer Focus is not a mission statement — it is the Ghazali Equity Factor (A3) and Ihsan
trend analysis. Leadership is not a management commitment — it is the Crown Layer that no
agent or human can override. Engagement is not an HR initiative — it is the Asabiyyah Index
(A12) quantified across mutual attestations, shared actions, and governance participation.
The Process Approach is not a flowchart — it is the five-stage Self-Harness pipeline
(FATE, Ihsan, Gini, Prune, Audit) that every action must traverse. Improvement is not a
suggestion box — it is the Omega Loop compiling high-quality paths into reflexes and the
Shura governance system processing systemic change proposals. Evidence-based Decision Making
is not a quarterly review — it is the Merkle-chained Event Log backing every governance vote.
Relationship Management is not stakeholder meetings — it is Gini convergence toward healthy
equilibrium enforced by Zakat (A5) and Demurrage (A4).

The QMS is the architecture. The architecture is the QMS.

---

## 1. Functional Requirements

### FR-C40: Customer Focus — User Sovereignty and Equity Protection

**ISO 9001 Clause:** 5.1.2 (Customer focus), 9.1.2 (Customer satisfaction).

The sovereign human node is the sole customer. Every subsystem exists to serve that node's
interests, and the Ghazali Equity Factor (A3) ensures newcomers receive the same structural
protection as established participants.

| Dimension | Mechanism | Measurement | Threshold |
|:---|:---|:---|:---|
| **Satisfaction** | Ihsan score trend over rolling window | Weighted mean (W_INTENT=0.25, W_EFFICIENCY=0.25, W_IMPACT=0.30, W_REPRODUCIBILITY=0.20) | >= IHSAN_PRODUCTION (0.95) |
| **Equity** | Ghazali Equity Factor (A3) | Newcomer economic protection coefficient | > 0.0 for all wallets with < 30 days tenure |
| **Personalization** | Living Memory learns user preferences | Episodic recall accuracy on user-relevant queries | Improving or stable trend |
| **Sovereignty** | No data leaves node without Ed25519-signed consent | Unauthorized egress count | 0 |

**User Satisfaction Composite:**

```
user_satisfaction = (
    0.40 * ihsan_trend_slope          # Improving or stable quality
  + 0.25 * equity_factor_coverage    # Newcomer protection active
  + 0.20 * memory_recall_accuracy   # System learns preferences
  + 0.15 * sovereignty_integrity    # Zero unauthorized egress
)
```

**Quality Gate:** `user_satisfaction >= 0.90` for QMS compliance. Below 0.90 triggers a
Shura governance proposal for systemic review.

**Evidence:** Ihsan trend reports, Ghazali Equity Factor per-wallet snapshots, Living Memory
recall accuracy logs, sovereignty egress audit (zero unauthorized events required).

---

### FR-C41: Leadership — Constitutional Invariants as Permanent Quality Policy

**ISO 9001 Clause:** 5.1 (Leadership and commitment), 5.2 (Quality policy).

In ISO 9001, top management establishes and communicates a quality policy. In BIZRA, the
quality policy is the 7 Constitutional Invariants — immutable, machine-enforced, and
unfalsifiable. No agent, user, or governance vote can override them.

| Invariant | Quality Policy Equivalent | Enforcement |
|:---|:---|:---|
| **I-1: Ethical Foundation** | "We will not cause harm" | H0 Crown Verification — safety check on every action |
| **I-2: Performance Floor** | "We will meet quality standards" | H1 Crown Verification — Ihsan >= IHSAN_PRODUCTION |
| **I-3: Accountability** | "Every action is traceable" | Merkle-chained Event Log, Ed25519-signed receipts |
| **I-4: Economic Justice** | "Wealth will not concentrate" | ADL Gini <= ADL_GINI_THRESHOLD (0.35) |
| **I-5: Collective Governance** | "Decisions are democratic" | Shura voting (A8) with BLOOM-weighted quorum |
| **I-6: Data Sovereignty** | "User data is user property" | Local-first, consent-gated egress |
| **I-7: Self-Improvement** | "We will continuously improve" | Omega Loop + Self-RLVR |

**Crown Verification Hierarchy:** H0 (Safety) > H1 (Ethics) > H2 (Performance). A receipt
that passes H2 but fails H0 is rejected. The hierarchy is hardcoded — not configurable.

**Leadership Effectiveness Metric:**

```
leadership_score = invariant_violations == 0 ? 1.0 : 0.0
```

Binary. There is no partial credit for leadership. Either invariants hold, or the system
has failed its quality policy.

**Evidence:** Crown Verification pass/fail log, invariant violation count (target: zero),
constitutional invariant immutability proof (hash of invariant definitions matches genesis
hash).

---

### FR-C42: Engagement of People — Asabiyyah Index and Attestation Network

**ISO 9001 Clause:** 5.1.1(h) (Engaging, directing and supporting persons), 7.1.2 (People).

Engagement is quantified by the Asabiyyah Social Cohesion Index (A12), which decomposes
into three weighted dimensions:

| Dimension | Weight | Source | Measurement |
|:---|:---|:---|:---|
| **Mutual Attestations** | 0.40 | `WalletState.attestations_given` + `attestations_received` | Bidirectional trust links per node |
| **Shared Actions** | 0.30 | Cooperative action receipts | Co-authored receipts / total receipts |
| **Governance Participation** | 0.30 | `WalletState.governance_votes` | Votes cast / proposals eligible |

**Asabiyyah Calculation (Fixed-Point):**

```
asabiyyah = fp(
    W_ATTESTATION * attestation_ratio
  + W_SHARED * shared_action_ratio
  + W_GOVERNANCE * governance_participation_ratio
)
WHERE W_ATTESTATION=0.4, W_SHARED=0.3, W_GOVERNANCE=0.3
```

**Attestation Coverage:** The percentage of node pairs in the federation with at least one
mutual attestation. High coverage indicates distributed trust; low coverage indicates
cluster formation or isolation.

```
attestation_coverage = bidirectional_pairs / (n * (n - 1) / 2)
```

**Engagement Health Classification:**

| Asabiyyah Range | Classification | ISO 9001 Interpretation |
|:---|:---|:---|
| >= 0.80 | THRIVING | Strong engagement, high trust network |
| 0.60 - 0.79 | HEALTHY | Adequate engagement for QMS operation |
| 0.40 - 0.59 | AT_RISK | Engagement declining, investigate causes |
| < 0.40 | CRITICAL | Social fabric compromised, governance effectiveness degraded |

**Quality Gate:** Asabiyyah >= 0.60 and attestation_coverage >= 0.30 for QMS compliance.

**Evidence:** Asabiyyah scores per node (rolling 30-day window), attestation graph snapshots,
governance participation rates, engagement health classification history.

---

### FR-C43: Process Approach — Five-Stage Self-Harness Pipeline

**ISO 9001 Clause:** 4.4 (QMS and its processes), 8.1 (Operational planning and control).

Every action in BIZRA flows through a defined five-stage pipeline. There is no "quick path"
or administrative bypass. The process is the architecture.

| Stage | Gate | Function | Failure Action |
|:---|:---|:---|:---|
| **1. FATE** | Formal (Z3), Alignment, Testing, Ethical | Pre-execution veto | Action blocked, receipt with `fate_pass=False` |
| **2. Ihsan** | Composite quality score | Post-execution quality check | Below IHSAN_PRODUCTION: flagged for causal analysis |
| **3. Gini** | ADL Gini coefficient | Economic impact check | Above ADL_GINI_THRESHOLD (0.35): action rolled back |
| **4. Prune** | Reflex Cache hygiene | Remove stale or degenerate reflexes | Expired reflexes removed, deny-list updated |
| **5. Audit** | Merkle chain append | Tamper-evident evidence trail | Receipt appended to Event Log with prev_hash linkage |

**Process Adherence Rate:**

```
adherence_rate = actions_passing_all_five_stages / total_actions_attempted
```

Target: >= 0.98. Actions that fail at any stage are not silently dropped — they produce a
failure receipt documenting which stage rejected them and why.

**Process Interaction Map:**

```
FATE -> Ihsan -> Gini -> Prune -> Audit
  |       |       |       |        |
  v       v       v       v        v
[Block] [CAR]  [Rollback] [Evict] [Chain]
```

Each stage's output feeds the next stage's input. Stage 5 (Audit) captures the complete
pipeline traversal as a single atomic receipt. If Audit itself fails (disk full, lock
contention), the action is retried with exponential backoff — no action succeeds without
an audit trail.

**Evidence:** Pipeline traversal logs showing stage-by-stage pass/fail for every action,
process adherence rate (rolling 30-day window), failure distribution by stage.

---

### FR-C44: Improvement — Self-RLVR and Governance Proposals

**ISO 9001 Clause:** 10.1 (Improvement — General), 10.2 (Nonconformity and corrective
action), 10.3 (Continual improvement).

Improvement operates at two levels: autonomous (Omega Loop) and deliberative (Shura).

**Autonomous Improvement (Omega Loop / Self-RLVR):**

High-quality action paths (Ihsan >= IHSAN_PRODUCTION, reproducibility >= 0.90, SNR >= SNR_T1)
are compiled into Reflex Cache entries via myelination. This is System-2 to System-1
compression — the system learns from its own successes.

| Metric | Measurement | Target |
|:---|:---|:---|
| **Myelination Ratio** | S1_hits / (S1_hits + S2_invocations) | >= 0.60 (L5 threshold), >= 0.90 (optimizing) |
| **Reflex Compilation Rate** | New reflexes compiled per 1000 actions | Positive and non-decreasing |
| **Degenerate Pruning Rate** | Denied patterns per 1000 actions | Low and stable (high = systemic issue) |

**Deliberative Improvement (Shura Governance):**

Systemic changes that cannot be addressed by myelination require a governance proposal:

```
Proposal:
    proposal_id: str
    proposer: Ed25519PublicKey
    votes_for: int       # BLOOM-weighted
    votes_against: int   # BLOOM-weighted
    status: PENDING | ACCEPTED | REJECTED
    ihsan_impact: float  # Projected Ihsan change
    gini_impact: float   # Projected Gini change
```

Proposals require BLOOM-weighted supermajority. Only wallets with Ihsan >= IHSAN_BLOOM_ELIGIBILITY
(0.90) may vote. This prevents low-quality participants from diluting governance quality.

**Improvement Score:**

```
improvement_score = (
    0.35 * myelination_trend_slope
  + 0.25 * proposal_acceptance_rate
  + 0.25 * corrective_action_closure_rate
  + 0.15 * recurrence_reduction_rate
)
```

**Quality Gate:** `improvement_score > 0.0` — the system must demonstrate net positive
improvement over any rolling 90-day window.

**Evidence:** Myelination Ratio trend, governance proposal log with outcomes, corrective
action receipts linked to root-cause analyses, recurrence tracking per defect category.

---

### FR-C45: Evidence-based Decision Making — Event Log and Proposal Evaluation

**ISO 9001 Clause:** 4.4.1(e) (Evaluate processes and implement changes), 9.1 (Monitoring,
measurement, analysis and evaluation).

Every governance decision is backed by data from the append-only Merkle-chained Event Log.
No proposal is voted on without quantitative impact analysis.

**Proposal Evaluation Pipeline:**

| Step | Input | Output | Verification |
|:---|:---|:---|:---|
| **1. Data Collection** | Event Log query for relevant receipts | Filtered receipt set | Query reproducible from parameters |
| **2. Impact Projection** | Receipts + proposed change | Projected Ihsan delta, Gini delta, Backing Ratio delta | Fixed-point arithmetic (FP_PRECISION=1,000,000) |
| **3. Risk Assessment** | Impact projections + constitutional invariants | Risk classification (LOW/MEDIUM/HIGH/CRITICAL) | Crown Verification hierarchy |
| **4. Presentation** | All above | Proposal with attached evidence bundle | BLAKE3 hash of evidence bundle |
| **5. Vote** | Proposal + evidence | BLOOM-weighted decision | Only IHSAN_BLOOM_ELIGIBILITY (0.90+) wallets eligible |

**Evidence Completeness:**

```
evidence_completeness = proposals_with_full_evidence / total_proposals
```

Target: 1.0. A proposal without attached evidence (impact projections, relevant receipts,
risk classification) is automatically rejected by the governance pipeline — it never reaches
the voting stage.

**Decision Traceability:** Every governance vote produces a receipt linking:
- The proposal_id
- The voter's Ed25519 public key
- The vote (FOR/AGAINST) with BLOOM weight
- The evidence bundle hash the voter reviewed
- Timestamp

This receipt chains into the Merkle log, making every decision auditable back to its
evidence basis.

**Evidence:** Proposal evaluation audit trails, evidence completeness rate, voter participation
metrics, decision outcome tracking (proposal -> implementation -> measured impact).

---

### FR-C46: Relationship Management — Gini Convergence and Federation Health

**ISO 9001 Clause:** 8.4 (Control of externally provided processes), 4.2 (Needs and
expectations of interested parties).

Relationship management in BIZRA operates at two scales: intra-node (economic equity among
wallets) and inter-node (federation health).

**Intra-Node: Economic Equity**

Three algorithms maintain economic balance:

| Algorithm | Function | Mechanism |
|:---|:---|:---|
| **A4: Demurrage** | Prevent token hoarding | Inactive SEED tokens decay over time |
| **A5: Zakat** | Wealth redistribution | 2.5% redistribution threshold on accumulated wealth |
| **A15: Backing Ratio** | Real-value anchor | 1 SEED = compute_hours + storage_gb + bandwidth_mbps |

**Gini Health Classification:**

| Gini Range | Classification | Action |
|:---|:---|:---|
| <= GINI_HEALTHY (0.30) | HEALTHY | None — equilibrium maintained |
| 0.30 - ADL_GINI_THRESHOLD (0.35) | NOMINAL | Monitor — within constitutional bounds |
| 0.35 - GINI_WARNING (0.50) | WARNING | Increase Zakat redistribution rate, trigger governance review |
| 0.50 - GINI_CRISIS (0.70) | DANGER | Emergency Shura session, accelerated Demurrage |
| > GINI_CRISIS (0.70) | CRISIS | Constitutional invariant I-4 violated — system halt on economic operations |

**Inter-Node: Federation Health**

| Metric | Measurement | Threshold |
|:---|:---|:---|
| **Peer Count** | Active federation peers | >= 1 for non-isolated operation |
| **Heartbeat Currency** | Time since last heartbeat from each peer | < 3x heartbeat interval |
| **Asabiyyah Cross-Node** | Mutual attestations across node boundaries | > 0 for federated nodes |
| **Consensus Latency** | Time to reach Shura quorum on cross-node proposals | < 60 seconds |

**Relationship Health Score:**

```
relationship_health = (
    0.40 * (1.0 - gini_distance_from_healthy)   # Economic equity
  + 0.30 * federation_connectivity_ratio         # Peer availability
  + 0.30 * cross_node_asabiyyah                  # Trust across boundaries
)
```

**Quality Gate:** `relationship_health >= 0.70` and `gini < ADL_GINI_THRESHOLD`.

**Evidence:** Gini coefficient time series (convergence toward GINI_HEALTHY), Zakat
redistribution logs, Demurrage decay records, federation peer topology snapshots,
cross-node attestation graph.

---

### FR-C47: QMS Audit Report Generation

**ISO 9001 Clause:** 9.2 (Internal audit), 9.3 (Management review).

The QMS audit report is generated automatically from the continuous monitoring substrate.
It aggregates all seven principles into a single signed evidence pack.

**Report Structure:**

| Section | Content | Source Function |
|:---|:---|:---|
| 1. Quality Policy | 7 Constitutional Invariants + hash proof of immutability | Genesis hash comparison |
| 2. Customer Focus | User satisfaction composite + Ghazali coverage | `assess_customer_focus()` |
| 3. Leadership | Invariant violation count (must be 0) | Crown Verification log |
| 4. Engagement | Asabiyyah scores + attestation coverage | `assess_engagement()` |
| 5. Process Approach | Pipeline adherence rate + failure distribution | Self-Harness pipeline log |
| 6. Improvement | Myelination ratio + proposal outcomes | `assess_improvement()` |
| 7. Evidence-based Decisions | Evidence completeness + decision traceability | `assess_evidence_based_decisions()` |
| 8. Relationship Management | Gini health + federation status | `compute_gini_health()` |
| 9. Nonconformities | All quality gate failures during period | Aggregated from Sections 2-8 |
| 10. Corrective Actions | Linked receipts for each nonconformity | Receipt chain query |
| 11. Merkle Seal | Root hash over all evidence receipts in period | MerkleTree over receipt IDs |

**Report Signing:** The complete report is serialized, hashed with BLAKE3, and signed with
the node's Ed25519 keypair. The signature, Merkle root, and receipt count form the
tamper-evident seal — identical to SOC 2 (FR-C36) report sealing.

**Audit Frequency:** Continuous (every audit tick, default 60 seconds) for monitoring;
full report generation on demand or at configurable intervals (default: monthly).

**Evidence:** Signed QMS report packs, audit tick history, nonconformity-to-corrective-action
linkage chains.

---

## 2. Edge Cases

**EC-C40: Gini Approaching Crisis Threshold.**
The Gini coefficient rises above GINI_WARNING (0.50) and trends toward GINI_CRISIS (0.70).
Resolution: (1) Warning is triggered at 0.50, initiating an automatic governance proposal
for increased Zakat redistribution. (2) Demurrage rates are accelerated on the largest
inactive wallet balances. (3) An emergency Shura session is convened with 24-hour quorum
deadline. (4) If Gini reaches 0.70, constitutional invariant I-4 is violated — all economic
operations (minting, transfer, staking) are halted until Gini returns below ADL_GINI_THRESHOLD
(0.35). (5) The QMS report records the incident as a critical nonconformity with the full
remediation chain. (6) The halt is not discretionary — the Self-Harness Stage 3 (Gini gate)
enforces it automatically. Recovery requires both Gini reduction below threshold AND a signed
governance receipt confirming the root cause was addressed.

**EC-C41: Asabiyyah Collapse — Node Isolation.**
A node's Asabiyyah score drops below 0.40 (CRITICAL), indicating loss of social trust.
Resolution: (1) The node remains operational but loses BLOOM-weighted voting eligibility
(cannot participate in Shura until Asabiyyah recovers). (2) Federation peers reduce
information sharing with the isolated node (trust-proportional disclosure). (3) The node
receives an automatic improvement plan: "increase mutual attestations, participate in
cooperative actions, vote on proposals." (4) If Asabiyyah remains below 0.40 for 30
consecutive days, the node's governance weight is zeroed — it can observe but not influence
decisions. (5) Recovery path: sustained cooperative behavior rebuilds Asabiyyah organically.
There is no administrative override for social trust. (6) The QMS report classifies this
as an engagement nonconformity (FR-C42 gate failure).

**EC-C42: Governance Proposal Spam.**
A wallet submits governance proposals at a rate that degrades deliberation quality.
Resolution: (1) Proposal submission is rate-limited to `max_proposals_per_wallet_per_day`
(default: 3). (2) Each proposal requires a minimum stake of SEED tokens (locked until
vote resolution). (3) Proposals with identical or near-identical content (cosine similarity
> 0.95 against recent proposals) are automatically deduplicated. (4) A wallet whose last
5 proposals were all REJECTED receives a cooldown period (7 days). (5) The rate limit and
stake requirement are themselves governed by Shura — they can be adjusted but not removed.
(6) Evidence completeness gate (FR-C45) rejects proposals without quantitative impact
analysis, naturally filtering low-effort submissions.

**EC-C43: Newcomer Onboarding with Zero History.**
A new wallet joins the federation with no attestations, no governance votes, no action
history. All ratio-based metrics (Asabiyyah, Ihsan trend, myelination) are undefined.
Resolution: (1) Ghazali Equity Factor (A3) activates a protective coefficient — newcomers
receive economic protection that prevents exploitation by established participants.
(2) Asabiyyah is initialized at 0.50 (HEALTHY floor) rather than 0.0 — newcomers start
with benefit of the doubt. (3) Ihsan trend requires a minimum of 50 receipts before slope
calculation; until then, per-receipt Ihsan is evaluated without trend penalties. (4) BLOOM
voting eligibility requires Ihsan >= IHSAN_BLOOM_ELIGIBILITY (0.90), which newcomers must
earn through demonstrated quality — no free governance weight. (5) The QMS report tracks
newcomer cohort metrics separately: time-to-first-attestation, time-to-governance-eligibility,
early-churn rate.

**EC-C44: Zakat Calculation on Very Small Balances.**
A wallet holds 0.001 SEED tokens. The 2.5% Zakat threshold produces a redistribution of
0.000025 SEED — below the minimum transferable unit.
Resolution: (1) Fixed-point arithmetic (FP_PRECISION=1,000,000) ensures the calculation
is exact: `fp(0.001) * fp(0.025) = 25` (in FP units). (2) If the Zakat amount is below
the minimum transferable unit (1 FP unit = 0.000001 SEED), the redistribution is deferred
and accumulated across ticks until it reaches the minimum. (3) Accumulated sub-minimum
Zakat is tracked per-wallet as `pending_zakat_fp` in WalletState. (4) No Zakat is lost —
the accumulation guarantees eventual redistribution. (5) Demurrage (A4) has the same
sub-minimum handling: decay amounts below 1 FP unit accumulate. (6) This is consistent
with the Backing Ratio (A15) — tokens represent real resources, and fractional resource
allocation is meaningful.

**EC-C45: Simultaneous FATE Failure and Gini Violation.**
An action fails the FATE Gate (Stage 1) AND the current Gini coefficient exceeds
ADL_GINI_THRESHOLD (Stage 3) simultaneously.
Resolution: (1) The FATE failure takes priority — Stage 1 rejection stops the pipeline
before Stage 3 is reached. (2) The Gini violation is detected independently by the
continuous audit tick (not dependent on action pipeline). (3) Two separate nonconformities
are recorded: one for the FATE rejection (process nonconformity) and one for the Gini
violation (relationship nonconformity). (4) The corrective actions are independent: FATE
failure triggers causal analysis on the action; Gini violation triggers economic
rebalancing. (5) The QMS report aggregates both nonconformities with their separate
remediation chains.

**EC-C46: Evidence Log Corruption During Report Generation.**
The Event Log becomes unreadable (disk failure, filesystem corruption) while the QMS report
is being generated.
Resolution: (1) Report generation operates on a snapshot — the Event Log is read into memory
at the start of generation, not streamed. (2) If the initial read fails, the report is
marked as INCOMPLETE with the failure reason documented. (3) The last valid Merkle root
(from the most recent audit tick) serves as the integrity anchor — all events up to that
root are known-good. (4) Events between the last Merkle root and the corruption point are
recovered from federation peers (if available) via the Merkle chain reconciliation protocol.
(5) The INCOMPLETE report is itself a valid QMS artifact — it documents the failure and
recovery process, which IS the corrective action.

---

## 3. Pseudocode

### 3.1 assess_customer_focus

```
FUNCTION assess_customer_focus(
    user_model: UserModel,
    ihsan_trend: list[IhsanSample],
    equity_factor: dict[WalletId, float],
    memory_accuracy: float,
    egress_violations: int
) -> CustomerFocusAssessment:
    """Evaluate ISO 9001 Principle 1: Customer Focus.
    Ref: core/constitutional/algorithms.py (A1: Ihsan, A3: Ghazali Equity),
         core/living_memory/ (recall accuracy),
         core/integration/constants.py (IHSAN_PRODUCTION)"""

    # Step 1: Ihsan trend analysis — require minimum sample size
    IF len(ihsan_trend) < 50:
        ihsan_slope = None
        ihsan_status = "INSUFFICIENT_DATA"
        ihsan_mean = mean([s.score FOR s IN ihsan_trend]) IF ihsan_trend ELSE 0.0
    ELSE:
        ihsan_slope = linear_slope([fp_float(s.score) FOR s IN ihsan_trend])
        ihsan_mean = mean([fp_float(s.score) FOR s IN ihsan_trend])

        IF ihsan_slope > 0.001:
            ihsan_status = "IMPROVING"
        ELIF ihsan_slope >= -0.001:
            ihsan_status = "STABLE"
        ELSE:
            ihsan_status = "DECLINING"

    ihsan_gate_pass = ihsan_mean >= IHSAN_PRODUCTION  # 0.95

    # Step 2: Ghazali Equity Factor coverage
    # Every wallet with < 30 days tenure must have equity_factor > 0
    newcomer_wallets = [w FOR w IN user_model.wallets IF w.tenure_days < 30]
    protected_newcomers = [w FOR w IN newcomer_wallets
                           IF equity_factor.get(w.wallet_id, 0.0) > 0.0]
    equity_coverage = len(protected_newcomers) / len(newcomer_wallets)
        IF len(newcomer_wallets) > 0 ELSE 1.0

    # Step 3: Sovereignty integrity — zero tolerance for unauthorized egress
    sovereignty_integrity = 1.0 IF egress_violations == 0 ELSE 0.0

    # Step 4: Composite user satisfaction score
    ihsan_trend_component = max(0.0, min(1.0,
        0.5 + (ihsan_slope * 100) IF ihsan_slope IS NOT None ELSE 0.5))

    user_satisfaction = (
        0.40 * ihsan_trend_component
      + 0.25 * equity_coverage
      + 0.20 * min(1.0, memory_accuracy)
      + 0.15 * sovereignty_integrity
    )

    customer_focus_pass = (
        ihsan_gate_pass
        AND equity_coverage >= 1.0
        AND sovereignty_integrity == 1.0
        AND user_satisfaction >= 0.90
    )

    RETURN CustomerFocusAssessment(
        ihsan_mean=ihsan_mean,
        ihsan_slope=ihsan_slope,
        ihsan_status=ihsan_status,
        ihsan_gate_pass=ihsan_gate_pass,
        newcomer_count=len(newcomer_wallets),
        protected_newcomers=len(protected_newcomers),
        equity_coverage=equity_coverage,
        memory_accuracy=memory_accuracy,
        egress_violations=egress_violations,
        sovereignty_integrity=sovereignty_integrity,
        user_satisfaction=user_satisfaction,
        customer_focus_pass=customer_focus_pass,
        timestamp=now_ms()
    )
```

### 3.2 assess_engagement

```
FUNCTION assess_engagement(
    asabiyyah_scores: dict[NodeId, float],
    attestation_graph: AttestationGraph,
    governance_log: list[GovernanceEvent],
    cooperative_receipts: list[ActionReceipt]
) -> EngagementAssessment:
    """Evaluate ISO 9001 Principle 3: Engagement of People.
    Ref: core/constitutional/algorithms.py (A12: Asabiyyah),
         core/constitutional/types.py (WalletState),
         core/integration/constants.py (ASABIYYAH_WEIGHTS)"""

    W_ATTESTATION, W_SHARED, W_GOVERNANCE = 0.4, 0.3, 0.3

    # Step 1: Compute per-node Asabiyyah decomposition
    node_details = []
    FOR node_id, score IN asabiyyah_scores.items():
        node_details.append(NodeEngagement(
            node_id=node_id,
            asabiyyah=score,
            classification=classify_asabiyyah(score)
        ))

    # Step 2: Network-wide Asabiyyah statistics
    scores = list(asabiyyah_scores.values())
    IF len(scores) == 0:
        RETURN EngagementAssessment(
            network_asabiyyah=0.0, classification="NO_NODES",
            engagement_pass=False, timestamp=now_ms())

    network_asabiyyah = mean(scores)
    min_asabiyyah = min(scores)
    max_asabiyyah = max(scores)
    nodes_at_risk = len([s FOR s IN scores IF s < 0.60])
    nodes_critical = len([s FOR s IN scores IF s < 0.40])

    # Step 3: Attestation coverage — bidirectional trust links
    total_nodes = len(scores)
    max_pairs = total_nodes * (total_nodes - 1) / 2
    bidirectional_pairs = attestation_graph.count_bidirectional()
    attestation_coverage = bidirectional_pairs / max_pairs IF max_pairs > 0 ELSE 0.0

    # Step 4: Governance participation rate
    eligible_voters = len([n FOR n IN asabiyyah_scores.keys()
                           IF get_wallet(n).ihsan_mean >= IHSAN_BLOOM_ELIGIBILITY])
    total_proposals = len(set(e.proposal_id FOR e IN governance_log
                              IF e.event_type == "PROPOSAL_CREATED"))
    total_votes = len([e FOR e IN governance_log IF e.event_type == "VOTE_CAST"])
    max_possible_votes = eligible_voters * total_proposals
    governance_participation = total_votes / max_possible_votes
        IF max_possible_votes > 0 ELSE 0.0

    # Step 5: Cooperative action ratio
    total_receipts = len(cooperative_receipts)
    co_authored = len([r FOR r IN cooperative_receipts IF len(r.co_authors) > 1])
    cooperation_ratio = co_authored / total_receipts IF total_receipts > 0 ELSE 0.0

    # Step 6: Overall engagement health
    network_classification = classify_asabiyyah(network_asabiyyah)

    engagement_pass = (
        network_asabiyyah >= 0.60
        AND attestation_coverage >= 0.30
        AND nodes_critical == 0
    )

    RETURN EngagementAssessment(
        network_asabiyyah=network_asabiyyah,
        min_asabiyyah=min_asabiyyah,
        max_asabiyyah=max_asabiyyah,
        classification=network_classification,
        node_details=node_details,
        nodes_at_risk=nodes_at_risk,
        nodes_critical=nodes_critical,
        bidirectional_pairs=bidirectional_pairs,
        attestation_coverage=attestation_coverage,
        eligible_voters=eligible_voters,
        governance_participation=governance_participation,
        cooperation_ratio=cooperation_ratio,
        engagement_pass=engagement_pass,
        timestamp=now_ms()
    )


FUNCTION classify_asabiyyah(score: float) -> str:
    IF score >= 0.80: RETURN "THRIVING"
    IF score >= 0.60: RETURN "HEALTHY"
    IF score >= 0.40: RETURN "AT_RISK"
    RETURN "CRITICAL"
```

### 3.3 assess_evidence_based_decisions

```
FUNCTION assess_evidence_based_decisions(
    proposals: list[Proposal],
    event_log: EventLog,
    vote_receipts: list[VoteReceipt]
) -> EvidenceDecisionAssessment:
    """Evaluate ISO 9001 Principle 6: Evidence-based Decision Making.
    Ref: core/governance/ (Shura voting, proposal pipeline),
         core/constitutional/types.py (Proposal),
         core/proof_engine/evidence_ledger.py (Merkle chain),
         core/constitutional/fixed_point.py (FP_PRECISION)"""

    # Step 1: Evidence completeness — every proposal must have impact analysis
    proposals_with_evidence = []
    proposals_without_evidence = []

    FOR proposal IN proposals:
        has_ihsan_projection = proposal.ihsan_impact IS NOT None
        has_gini_projection = proposal.gini_impact IS NOT None
        has_receipt_bundle = proposal.evidence_bundle_hash IS NOT None
        has_risk_classification = proposal.risk_level IS NOT None

        evidence_complete = (has_ihsan_projection AND has_gini_projection
                             AND has_receipt_bundle AND has_risk_classification)

        IF evidence_complete:
            proposals_with_evidence.append(proposal)
        ELSE:
            proposals_without_evidence.append(ProposalDeficiency(
                proposal_id=proposal.proposal_id,
                missing_ihsan=NOT has_ihsan_projection,
                missing_gini=NOT has_gini_projection,
                missing_bundle=NOT has_receipt_bundle,
                missing_risk=NOT has_risk_classification))

    evidence_completeness = len(proposals_with_evidence) / len(proposals)
        IF len(proposals) > 0 ELSE 1.0

    # Step 2: Decision traceability — verify vote receipts chain into Merkle log
    unchained_votes = []
    FOR vote IN vote_receipts:
        # Verify vote receipt exists in Event Log
        event = event_log.find_by_receipt_id(vote.receipt_id)
        IF event IS None:
            unchained_votes.append(vote.receipt_id)
            CONTINUE

        # Verify voter signature
        IF NOT Ed25519.verify(vote.signature, vote.content, vote.voter_pubkey):
            unchained_votes.append(vote.receipt_id)
            CONTINUE

        # Verify voter reviewed the evidence bundle
        IF vote.evidence_bundle_hash IS None:
            unchained_votes.append(vote.receipt_id)

    traceability_ratio = 1.0 - (len(unchained_votes) / len(vote_receipts))
        IF len(vote_receipts) > 0 ELSE 1.0

    # Step 3: Voter eligibility verification — BLOOM weight check
    ineligible_votes = []
    FOR vote IN vote_receipts:
        voter_wallet = get_wallet_by_pubkey(vote.voter_pubkey)
        IF voter_wallet IS None:
            ineligible_votes.append(vote.receipt_id)
            CONTINUE
        IF fp_float(voter_wallet.ihsan_mean) < IHSAN_BLOOM_ELIGIBILITY:
            ineligible_votes.append(vote.receipt_id)

    eligibility_compliance = 1.0 - (len(ineligible_votes) / len(vote_receipts))
        IF len(vote_receipts) > 0 ELSE 1.0

    # Step 4: Decision outcome tracking — did accepted proposals deliver?
    accepted = [p FOR p IN proposals IF p.status == "ACCEPTED"]
    measured = [p FOR p IN accepted IF p.measured_ihsan_delta IS NOT None]
    outcome_tracking_ratio = len(measured) / len(accepted) IF len(accepted) > 0 ELSE 1.0

    # Of measured proposals, how many met projections (within 10% tolerance)?
    projection_accuracy = 0.0
    IF len(measured) > 0:
        accurate = [p FOR p IN measured
                    IF abs(fp_float(p.measured_ihsan_delta) - fp_float(p.ihsan_impact))
                       <= 0.10 * abs(fp_float(p.ihsan_impact) + 0.001)]
        projection_accuracy = len(accurate) / len(measured)

    evidence_decision_pass = (
        evidence_completeness >= 1.0
        AND traceability_ratio >= 0.99
        AND eligibility_compliance >= 1.0
    )

    RETURN EvidenceDecisionAssessment(
        total_proposals=len(proposals),
        proposals_with_evidence=len(proposals_with_evidence),
        proposals_without_evidence=proposals_without_evidence,
        evidence_completeness=evidence_completeness,
        total_votes=len(vote_receipts),
        unchained_votes=unchained_votes,
        traceability_ratio=traceability_ratio,
        ineligible_votes=ineligible_votes,
        eligibility_compliance=eligibility_compliance,
        accepted_proposals=len(accepted),
        measured_proposals=len(measured),
        outcome_tracking_ratio=outcome_tracking_ratio,
        projection_accuracy=projection_accuracy,
        evidence_decision_pass=evidence_decision_pass,
        timestamp=now_ms()
    )
```

### 3.4 compute_gini_health

```
FUNCTION compute_gini_health(
    wallets: list[WalletState],
    gini_history: list[GiniSample] = None,
    window_days: int = 30
) -> GiniHealthReport:
    """Compute Gini coefficient with health classification and trend.
    Ref: core/constitutional/algorithms.py (A4: Demurrage, A5: Zakat, A15: Backing Ratio),
         core/integration/constants.py (ADL_GINI_THRESHOLD, GINI_HEALTHY, GINI_WARNING,
                                         GINI_CRISIS)"""

    # Step 1: Compute current Gini coefficient (fixed-point)
    n = len(wallets)
    IF n == 0:
        RETURN GiniHealthReport(gini=0.0, classification="NO_WALLETS",
                                health_pass=True, timestamp=now_ms())

    balances = sorted([fp_float(w.balance) FOR w IN wallets])
    total = sum(balances)

    IF total == 0:
        # Perfect equality (everyone has zero)
        gini = 0.0
    ELSE:
        # Standard Gini formula: G = (2 * sum(i * y_i)) / (n * sum(y_i)) - (n+1)/n
        numerator = sum((i + 1) * balance FOR i, balance IN enumerate(balances))
        gini = (2.0 * numerator) / (n * total) - (n + 1) / n

    # Clamp to [0, 1] — floating-point edge cases
    gini = max(0.0, min(1.0, gini))

    # Step 2: Classify against constitutional thresholds
    IF gini <= GINI_HEALTHY:                        # 0.30
        classification = "HEALTHY"
        action = None
    ELIF gini <= ADL_GINI_THRESHOLD:                # 0.35
        classification = "NOMINAL"
        action = None
    ELIF gini <= GINI_WARNING:                      # 0.50
        classification = "WARNING"
        action = "INCREASE_ZAKAT_RATE"
    ELIF gini <= GINI_CRISIS:                       # 0.70
        classification = "DANGER"
        action = "EMERGENCY_SHURA_AND_ACCELERATED_DEMURRAGE"
    ELSE:
        classification = "CRISIS"
        action = "HALT_ECONOMIC_OPERATIONS"

    # Step 3: Trend analysis
    IF gini_history IS NOT None AND len(gini_history) >= 3:
        cutoff = now_ms() - (window_days * 86_400_000)
        window = [s FOR s IN gini_history IF s.timestamp >= cutoff]

        IF len(window) >= 3:
            gini_slope = linear_slope([s.gini FOR s IN window])
            IF gini_slope < -0.001:
                trend = "CONVERGING"     # Moving toward equality
            ELIF gini_slope <= 0.001:
                trend = "STABLE"
            ELSE:
                trend = "DIVERGING"      # Moving toward inequality
        ELSE:
            gini_slope = None
            trend = "INSUFFICIENT_DATA"
    ELSE:
        gini_slope = None
        trend = "INSUFFICIENT_DATA"

    # Step 4: Distance from healthy equilibrium
    gini_distance = abs(gini - GINI_HEALTHY)

    # Step 5: Zakat and Demurrage effectiveness
    zakat_wallets = [w FOR w IN wallets IF fp_float(w.pending_zakat_fp) > 0]
    demurrage_wallets = [w FOR w IN wallets IF fp_float(w.inactive_days) > 30]

    health_pass = gini <= ADL_GINI_THRESHOLD  # 0.35 — constitutional invariant I-4

    RETURN GiniHealthReport(
        gini=gini,
        classification=classification,
        recommended_action=action,
        trend=trend,
        gini_slope=gini_slope,
        gini_distance_from_healthy=gini_distance,
        wallet_count=n,
        total_supply=total,
        min_balance=balances[0] IF n > 0 ELSE 0.0,
        max_balance=balances[-1] IF n > 0 ELSE 0.0,
        median_balance=balances[n // 2] IF n > 0 ELSE 0.0,
        zakat_eligible_wallets=len(zakat_wallets),
        demurrage_eligible_wallets=len(demurrage_wallets),
        health_pass=health_pass,
        timestamp=now_ms()
    )
```

### 3.5 generate_iso9001_report

```
FUNCTION generate_iso9001_report(
    customer_focus: CustomerFocusAssessment,
    leadership_violations: int,
    engagement: EngagementAssessment,
    process_adherence_rate: float,
    improvement: ImprovementAssessment,
    evidence_decisions: EvidenceDecisionAssessment,
    gini_health: GiniHealthReport,
    period: TimeRange
) -> ISO9001Report:
    """Generate ISO 9001 QMS audit report with signed Merkle seal.
    Ref: core/proof_engine/evidence_ledger.py (Merkle tree),
         core/integration/constants.py (all thresholds),
         core/governance/ (Shura proposal pipeline)"""

    # Section 1: Quality Policy immutability
    current_invariant_hash = BLAKE3.hash(serialize_constitutional_invariants())
    genesis_invariant_hash = load_genesis_hash("constitutional_invariants")
    policy_immutable = current_invariant_hash == genesis_invariant_hash

    # Section 2-8: Aggregate principle assessments
    principles = [
        PrincipleResult(name="Customer Focus", clause="5.1.2",
                        passed=customer_focus.customer_focus_pass,
                        score=customer_focus.user_satisfaction),
        PrincipleResult(name="Leadership", clause="5.1",
                        passed=(leadership_violations == 0 AND policy_immutable),
                        score=1.0 IF leadership_violations == 0 ELSE 0.0),
        PrincipleResult(name="Engagement", clause="5.1.1(h)",
                        passed=engagement.engagement_pass,
                        score=engagement.network_asabiyyah),
        PrincipleResult(name="Process Approach", clause="4.4",
                        passed=process_adherence_rate >= 0.98,
                        score=process_adherence_rate),
        PrincipleResult(name="Improvement", clause="10.1",
                        passed=improvement.improvement_pass,
                        score=improvement.improvement_score),
        PrincipleResult(name="Evidence-based Decisions", clause="9.1",
                        passed=evidence_decisions.evidence_decision_pass,
                        score=evidence_decisions.evidence_completeness),
        PrincipleResult(name="Relationship Management", clause="8.4",
                        passed=gini_health.health_pass,
                        score=1.0 - gini_health.gini_distance_from_healthy),
    ]

    # Section 9: Collect nonconformities from failed principles
    nonconformities = []
    FOR principle IN principles:
        IF NOT principle.passed:
            nonconformities.append(Nonconformity(
                principle=principle.name,
                clause=principle.clause,
                severity="MAJOR" IF principle.score < 0.50 ELSE "MINOR",
                score=principle.score,
                detail=describe_nonconformity(principle)))

    # Section 10: Link corrective actions
    FOR nc IN nonconformities:
        corrective_actions = find_corrective_action_receipts(nc, period)
        nc.corrective_actions = corrective_actions
        nc.remediated = len(corrective_actions) > 0

    # Section 11: Merkle seal over all evidence
    all_receipt_ids = collect_receipt_ids(period)
    merkle_tree = MerkleTree.build(sorted(all_receipt_ids))

    # Compute overall QMS Ihsan
    principle_scores = [p.score FOR p IN principles]
    qms_ihsan = mean(principle_scores) IF principle_scores ELSE 0.0

    # Overall pass: all principles pass AND no unremediated major nonconformities
    major_unremediated = [nc FOR nc IN nonconformities
                          IF nc.severity == "MAJOR" AND NOT nc.remediated]
    all_principles_pass = all(p.passed FOR p IN principles)
    qms_pass = all_principles_pass AND len(major_unremediated) == 0

    report = ISO9001Report(
        standard="ISO_9001:2015",
        period=period,
        policy_immutable=policy_immutable,
        invariant_hash=current_invariant_hash,
        principles=principles,
        principles_passed=len([p FOR p IN principles IF p.passed]),
        principles_total=len(principles),
        nonconformities=nonconformities,
        nonconformity_count=len(nonconformities),
        major_unremediated=len(major_unremediated),
        all_remediated=all(nc.remediated FOR nc IN nonconformities)
            IF nonconformities ELSE True,
        qms_ihsan=qms_ihsan,
        qms_pass=qms_pass,
        gini_current=gini_health.gini,
        gini_classification=gini_health.classification,
        asabiyyah_network=engagement.network_asabiyyah,
        process_adherence=process_adherence_rate,
        merkle_root=merkle_tree.root,
        total_receipts=len(all_receipt_ids),
        generated_at=now_ms()
    )

    # Sign the complete report
    report_hash = BLAKE3.hash(report.serialize())
    report.signature = node_keypair.sign(report_hash)
    report.report_hash = report_hash

    RETURN report
```

---

## 4. TDD Anchors

```
TEST customer_focus_passes_with_high_ihsan_and_full_equity:
    """Stable high Ihsan + full Ghazali coverage = customer focus pass."""
    ihsan_trend = generate_ihsan_samples(count=100, range=(0.95, 0.99))
    equity = {w.id: 0.5 FOR w IN newcomer_wallets}  # All newcomers protected
    result = assess_customer_focus(user_model, ihsan_trend, equity,
                                    memory_accuracy=0.85, egress_violations=0)
    ASSERT result.customer_focus_pass == True
    ASSERT result.equity_coverage == 1.0
    ASSERT result.sovereignty_integrity == 1.0
    ASSERT result.user_satisfaction >= 0.90

TEST customer_focus_fails_on_egress_violation:
    """Any unauthorized egress fails sovereignty integrity."""
    ihsan_trend = generate_ihsan_samples(count=100, range=(0.95, 0.99))
    equity = {w.id: 0.5 FOR w IN newcomer_wallets}
    result = assess_customer_focus(user_model, ihsan_trend, equity,
                                    memory_accuracy=0.85, egress_violations=1)
    ASSERT result.customer_focus_pass == False
    ASSERT result.sovereignty_integrity == 0.0

TEST customer_focus_handles_insufficient_ihsan_data:
    """Less than 50 samples produces INSUFFICIENT_DATA status without crashing."""
    ihsan_trend = generate_ihsan_samples(count=10, range=(0.95, 0.99))
    result = assess_customer_focus(user_model, ihsan_trend, {},
                                    memory_accuracy=0.85, egress_violations=0)
    ASSERT result.ihsan_status == "INSUFFICIENT_DATA"
    ASSERT result.ihsan_slope IS None

TEST customer_focus_fails_on_unprotected_newcomer:
    """Newcomer without Ghazali protection fails equity coverage."""
    newcomers = create_wallets(count=5, tenure_days=10)
    equity = {newcomers[0].id: 0.5, newcomers[1].id: 0.5}  # Only 2 of 5 protected
    ihsan_trend = generate_ihsan_samples(count=100, range=(0.95, 0.99))
    result = assess_customer_focus(user_model_with(newcomers), ihsan_trend, equity,
                                    memory_accuracy=0.85, egress_violations=0)
    ASSERT result.customer_focus_pass == False
    ASSERT result.equity_coverage < 1.0

TEST engagement_detects_critical_asabiyyah_node:
    """Any node below 0.40 Asabiyyah triggers critical classification."""
    scores = {"node-1": 0.85, "node-2": 0.75, "node-3": 0.35}
    graph = create_attestation_graph(3, bidirectional_pairs=2)
    result = assess_engagement(scores, graph, [], [])
    ASSERT result.nodes_critical == 1
    ASSERT result.engagement_pass == False

TEST engagement_passes_healthy_network:
    """All nodes healthy with adequate attestation coverage passes."""
    scores = {"node-1": 0.85, "node-2": 0.75, "node-3": 0.80, "node-4": 0.70}
    graph = create_attestation_graph(4, bidirectional_pairs=3)  # 3 of 6 pairs = 50%
    result = assess_engagement(scores, graph, [], [])
    ASSERT result.network_asabiyyah >= 0.60
    ASSERT result.attestation_coverage >= 0.30
    ASSERT result.nodes_critical == 0
    ASSERT result.engagement_pass == True

TEST engagement_handles_empty_network:
    """Zero nodes returns NO_NODES with engagement_pass=False."""
    result = assess_engagement({}, empty_graph(), [], [])
    ASSERT result.classification == "NO_NODES"
    ASSERT result.engagement_pass == False

TEST evidence_decisions_requires_full_evidence:
    """Proposal without impact analysis fails evidence completeness."""
    proposals = [create_proposal(ihsan_impact=0.02, gini_impact=-0.01,
                                  evidence_hash=b"abc", risk="LOW"),
                 create_proposal(ihsan_impact=None, gini_impact=None,
                                  evidence_hash=None, risk=None)]
    result = assess_evidence_based_decisions(proposals, event_log, [])
    ASSERT result.evidence_completeness == 0.5
    ASSERT result.evidence_decision_pass == False

TEST evidence_decisions_detects_ineligible_voter:
    """Vote from wallet below IHSAN_BLOOM_ELIGIBILITY is flagged."""
    proposals = [create_full_proposal()]
    vote = create_vote(voter_ihsan=0.85)  # Below 0.90 eligibility
    result = assess_evidence_based_decisions(proposals, event_log, [vote])
    ASSERT len(result.ineligible_votes) == 1
    ASSERT result.eligibility_compliance < 1.0

TEST gini_classifies_healthy_distribution:
    """Equal distribution produces Gini near 0 with HEALTHY classification."""
    wallets = create_wallets_with_balances([100, 100, 100, 100, 100])
    result = compute_gini_health(wallets)
    ASSERT result.gini <= GINI_HEALTHY  # 0.30
    ASSERT result.classification == "HEALTHY"
    ASSERT result.health_pass == True

TEST gini_detects_crisis_inequality:
    """Extreme concentration triggers CRISIS classification."""
    wallets = create_wallets_with_balances([0, 0, 0, 0, 10000])
    result = compute_gini_health(wallets)
    ASSERT result.gini > GINI_CRISIS  # 0.70
    ASSERT result.classification == "CRISIS"
    ASSERT result.recommended_action == "HALT_ECONOMIC_OPERATIONS"
    ASSERT result.health_pass == False

TEST gini_handles_zero_total_supply:
    """All wallets at zero balance produces Gini=0 (perfect equality)."""
    wallets = create_wallets_with_balances([0, 0, 0, 0])
    result = compute_gini_health(wallets)
    ASSERT result.gini == 0.0
    ASSERT result.classification == "HEALTHY"
    ASSERT result.health_pass == True

TEST gini_detects_converging_trend:
    """Declining Gini over time is classified as CONVERGING."""
    wallets = create_wallets_with_balances([80, 90, 100, 110, 120])
    history = [GiniSample(gini=0.40, timestamp=day_ago(30)),
               GiniSample(gini=0.35, timestamp=day_ago(15)),
               GiniSample(gini=0.30, timestamp=day_ago(1))]
    result = compute_gini_health(wallets, gini_history=history)
    ASSERT result.trend == "CONVERGING"
    ASSERT result.gini_slope < 0

TEST iso9001_report_passes_when_all_principles_pass:
    """All seven principles passing produces qms_pass=True."""
    report = generate_iso9001_report(
        passing_customer_focus(), leadership_violations=0,
        passing_engagement(), process_adherence_rate=0.99,
        passing_improvement(), passing_evidence_decisions(),
        healthy_gini(), last_30_days)
    ASSERT report.qms_pass == True
    ASSERT report.principles_passed == 7
    ASSERT report.nonconformity_count == 0
    ASSERT report.policy_immutable == True
    ASSERT report.merkle_root IS NOT None AND len(report.merkle_root) == 32
    ASSERT Ed25519.verify(report.signature, report.report_hash, node_pubkey)

TEST iso9001_report_includes_nonconformity_for_failed_principle:
    """Failed principle produces nonconformity with corrective action linkage."""
    report = generate_iso9001_report(
        failing_customer_focus(), leadership_violations=0,
        passing_engagement(), process_adherence_rate=0.99,
        passing_improvement(), passing_evidence_decisions(),
        healthy_gini(), last_30_days)
    ASSERT report.qms_pass == False
    ASSERT report.nonconformity_count >= 1
    ASSERT report.nonconformities[0].principle == "Customer Focus"
    ASSERT report.nonconformities[0].corrective_actions IS NOT None

TEST iso9001_report_fails_on_leadership_violation:
    """Any invariant violation fails the Leadership principle."""
    report = generate_iso9001_report(
        passing_customer_focus(), leadership_violations=1,
        passing_engagement(), process_adherence_rate=0.99,
        passing_improvement(), passing_evidence_decisions(),
        healthy_gini(), last_30_days)
    ASSERT report.qms_pass == False
    leadership_result = [p FOR p IN report.principles IF p.name == "Leadership"][0]
    ASSERT leadership_result.passed == False
    ASSERT leadership_result.score == 0.0
```

---

## 5. Cross-References

### Codebase Modules

| Module | ISO 9001 Principle | Relevance |
|:---|:---|:---|
| `core/constitutional/algorithms.py` | All | A1 (Ihsan), A3 (Ghazali), A4 (Demurrage), A5 (Zakat), A8 (Shura), A12 (Asabiyyah), A15 (Backing Ratio) |
| `core/constitutional/types.py` | 3, 4, 5 | Proposal, WalletState (attestations, governance_votes, cooperative_actions), Reflex |
| `core/constitutional/fixed_point.py` | 6 | FP_PRECISION=1,000,000 — deterministic arithmetic for all Gini and Ihsan calculations |
| `core/integration/constants.py` | All | ADL_GINI_THRESHOLD, GINI_HEALTHY, GINI_WARNING, GINI_CRISIS, ASABIYYAH_WEIGHTS, IHSAN_PRODUCTION, IHSAN_BLOOM_ELIGIBILITY |
| `core/governance/` | 2, 5, 6 | Shura voting pipeline, BLOOM-weighted proposals, proposal evaluation |
| `core/proof_engine/evidence_ledger.py` | 4, 6, 7 | Append-only Merkle chain, audit tick, evidence generation |
| `core/living_memory/` | 1 | User preference learning, episodic recall accuracy |
| `core/iaas/snr_v2_adapter.py` | 5 | SNR calculation for myelination gate |
| `core/federation/` | 7 | Cross-node attestations, federation health, heartbeat protocol |
| `core/auth/` | 1, 6 | Ed25519 identity, consent-gated egress, sovereignty enforcement |
| `bizra-agent/src/omni_kernel.rs` | 5 | ReflexCache myelination, Omega Loop |
| `bizra-ttrl/` | 5 | Self-RLVR training loop for autonomous improvement |
| `bizra-core/` | 2 | Constitutional invariants, Crown Verification hierarchy |

### ISO 9001:2015 Clause Mapping

| ISO 9001 Clause | BIZRA Functional Requirement | Primary Evidence |
|:---|:---|:---|
| 4.4 (QMS Processes) | FR-C43 | Pipeline adherence rate, stage-by-stage pass/fail logs |
| 5.1 (Leadership) | FR-C41 | Invariant violation count, constitutional hash immutability proof |
| 5.1.2 (Customer Focus) | FR-C40 | Ihsan trend, Ghazali coverage, sovereignty egress audit |
| 5.2 (Quality Policy) | FR-C41 | 7 invariants = policy, genesis hash = immutability proof |
| 5.1.1(h) (Engagement) | FR-C42 | Asabiyyah scores, attestation coverage, governance participation |
| 7.1.2 (People) | FR-C42 | Node engagement details, cooperation ratio |
| 8.1 (Operational Control) | FR-C43 | Five-stage Self-Harness, FATE pre-execution veto |
| 8.4 (External Providers) | FR-C46 | Federation health metrics, cross-node Asabiyyah |
| 9.1 (Monitoring) | FR-C45 | Evidence completeness, decision traceability |
| 9.2 (Internal Audit) | FR-C47 | QMS report packs, signed and Merkle-sealed |
| 9.3 (Management Review) | FR-C47 | Principle-by-principle assessment, nonconformity tracking |
| 10.1 (Improvement) | FR-C44 | Myelination ratio, improvement score |
| 10.2 (Corrective Action) | FR-C44 | Causal analysis receipts, corrective action chains |
| 10.3 (Continual Improvement) | FR-C44 | Omega Loop compilation events, Shura proposal outcomes |

### Sibling Specs

- Phase 00 (Framework Overview) -- Unified Evidence Model, cross-standard invariants I-1 through I-7
- Phase 01 (ISO 25010) -- FR-C10 Functional Suitability shares algorithm coverage with FR-C43
- Phase 02 (CMMI Level 5) -- FR-C23 Myelination Ratio reused in FR-C44 Improvement
- Phase 03 (SOC 2 Type II) -- FR-C32 Processing Integrity shares Merkle chain with FR-C43 Stage 5

### Constitutional Thresholds (from `core/integration/constants.py`)

| Constant | Value | ISO 9001 Usage |
|:---|:---|:---|
| IHSAN_PRODUCTION | 0.95 | FR-C40 Customer satisfaction floor, FR-C43 Stage 2 quality gate |
| IHSAN_BLOOM_ELIGIBILITY | 0.90 | FR-C44 Governance voting eligibility, FR-C45 voter verification |
| ADL_GINI_THRESHOLD | 0.35 | FR-C46 Constitutional invariant I-4, FR-C43 Stage 3 gate |
| GINI_HEALTHY | 0.30 | FR-C46 Equilibrium target for Relationship Management |
| GINI_WARNING | 0.50 | FR-C46 Escalation threshold for increased Zakat |
| GINI_CRISIS | 0.70 | FR-C46 Constitutional halt trigger |
| ASABIYYAH_WEIGHTS | (0.4, 0.3, 0.3) | FR-C42 Engagement decomposition (attestation, shared, governance) |
| FP_PRECISION | 1,000,000 | FR-C45 Deterministic proposal impact calculations |
