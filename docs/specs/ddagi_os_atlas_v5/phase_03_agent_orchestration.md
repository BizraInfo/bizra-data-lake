# Phase 03 — Agent Orchestration: PAT-7, SAT-49, Dual-Agentic Protocol

> Source: Atlas v5.0 — Diagrams D4 (PAT-7), D5 (SAT-49), D6 (Dual-Agentic Negotiation)
> Status: SPECIFICATION SEALED | SNR: 0.95

---

## 1. Functional Requirements

### FR-030: PAT-7 — Personal Agent Team

Each sovereign node spawns exactly 7 personal agents that advocate exclusively for
the user. PAT agents share episodic memory but maintain independent inference
state. All 7 are spawned at genesis (Phase 01, FR-010 step 6).

**Canonical Roles** (source: `core/integration/constants.py` `PAT_AGENT_NAMES`):

| # | Role        | Domain                        | Primary Capability                     |
|---|-------------|-------------------------------|----------------------------------------|
| 0 | Planner     | HTN decomposition             | Hierarchical task-network planning     |
| 1 | Researcher  | RAG + Web retrieval           | Knowledge retrieval, fact verification |
| 2 | Coder       | Generation + Test + Debug     | Code synthesis, test generation, debug |
| 3 | Evaluator   | SNR + PoI scoring             | Quality measurement, impact assessment |
| 4 | Ethicist    | Constitutional + Shariah      | Bias audit, alignment, VETO power      |
| 5 | Publisher   | Format + Deliver + Feedback   | Output shaping, delivery, signal capture |
| 6 | Integrator  | Cross-Agent + Context + Memory | Context assembly, memory coordination  |

> **Rust mapping:** `bizra-agent/src/types.rs` uses positional equivalents
> (Navigator, Scholar, Artisan, Guardian, Mentor, Diplomat, Oracle).
> Python canonical names from `constants.py` are authoritative for this spec.

**HTDAG Planning.** The Planner decomposes user intent into a Hierarchical Task
Directed Acyclic Graph. Each node carries: action type, estimated cost, dependency
edges, and an explainability trace serialized into the execution receipt.

**MPC Budget Optimizer.** Resource allocation across PAT agents uses additive
secret sharing so no single agent learns another's internal budget state. The
Integrator aggregates encrypted bids and publishes only the allocation vector.
Guarantees: privacy of agent internals, Pareto-optimal allocation, determinism.

**Trust Stages** (`PAT_TRUST_STAGES`): abstracting, gathering, executing,
attesting, certifying, publishing, chaining.

### FR-031: SAT-49 — System Agent Team

The System Agent Team advocates for infrastructure health, economic stability,
and constitutional compliance. SAT operates at two scales:

**Per-Node Bootstrap (SAT-5):** 5 system agents per node (`SAT_AGENTS_PER_NODE`):
ComputeScheduler, SecurityMonitor, PerformanceAnalyzer, ConsensusValidator,
NetworkOrchestrator.

**Federation Scale (SAT-49):** 1 SAT-CEO + 7 Departments x 7 agents. SAT-49 is
a federation-level logical view; nodes run SAT-5 and participate in distributed
department functions via gossip.

| Department | Code   | Responsibility                                  |
|------------|--------|-------------------------------------------------|
| Growth     | EGO-7  | User acquisition, onboarding, network expansion |
| Product    | PXS-7  | Feature delivery, UX quality, deployment         |
| AI/ML      | AML-7  | Model routing, inference optimization, TTRL      |
| Knowledge  | KNM-7  | Corpus management, RAG, semantic indexing         |
| Risk       | RCE-7  | Security audit, anomaly detection, circuit break  |
| Infra      | IRP-7  | Compute pool, storage, bandwidth, uptime         |
| Treasury   | TTM-7  | SEED/BLOOM economics, Gini attractor, Zakat      |

**SAT-CEO RAE Meta-Loop.** Weekly Resource-Allocation-Evaluation cycle
(`SAT_REBALANCE_INTERVAL_S = 300` for micro-cycles, weekly for macro RAE):
1. Collect department KPIs (latency p95, SNR mean, Gini, uptime).
2. Constrained optimization: maximize Ihsan s.t. Gini <= 0.35, infra >= 20%.
3. Publish new allocation weights as a signed constitutional receipt.
4. Departments adjust agent priorities for the next cycle.

### FR-032: Dual-Agentic Negotiation Protocol

PAT advocates for user. SAT advocates for system. Neither may access the other's
internal state. Every non-trivial action requires negotiated agreement.

**Formal API Contract:**

```
NegotiationRequest {
    request_id  : UUID,      pat_node_id : DID,
    task        : TaskCard,  budget      : ResourceBudget,
    priority    : int(1-10), deadline_ms : int,
    signature   : Ed25519(PAT key, BLAKE3(canonical fields))
}
NegotiationResponse {
    request_id    : UUID,     decision      : ACCEPT | COUNTER | REJECT,
    cost_estimate : ResourceCost, risk_score : float[0,1],
    alignment     : float[0,1],   counter_budget : ResourceBudget?,
    rationale     : str,      signature     : Ed25519(SAT key, BLAKE3(canonical fields))
}
```

**Negotiation State Machine:**

```
PAT_SUBMIT -> SAT_EVALUATE -> ACCEPT -> EXECUTE -> RECEIPT
                  |-> COUNTER -> PAT_REVISE -> SAT_EVALUATE (max 3 rounds)
                  |-> REJECT -> PAT_NOTIFIED (reason logged, receipt emitted)
Deadlock after 3 rounds: escalate to H0/H1/H2 watchdog.
```

**Process Isolation.** PAT and SAT run in separate memory spaces. Communication
via A2A messages only. Violation triggers Sentinel alert and node quarantine.

**Cryptographic Receipts.** Every negotiation step produces an Ed25519-signed,
Merkle-linked receipt. Fields align with `core.a2a.schema.A2AMessage`.

**H0/H1/H2 Watchdog.** Crown Verification (L6) monitors all negotiations.
Default-deny. H0=Ethical/Shariah, H1=Performance/budget, H2=Safety/permissions.

### FR-033: Multi-Timescale Optimization

| Timescale | Actor       | Cadence       | Scope                               |
|-----------|-------------|---------------|--------------------------------------|
| Micro     | PAT agents  | Milliseconds  | Single-task routing, reflex dispatch |
| Meso      | SAT-CEO RAE | Weekly        | Department budget rebalance          |
| Macro     | Human       | Quarterly     | Governance proposals, policy updates |

Micro completes within `TIER_COMPLEX_BUDGET_MS` (15000 ms). Meso is background.
Macro requires BLOOM-weighted vote (Phase 05).

---

## 2. Edge Cases

**EC-030: PAT-SAT Negotiation Deadlock.**
After 3 COUNTER rounds, escalate to H0/H1/H2 watchdog. If unresolvable, suspend
and auto-generate governance proposal. Timeout: `UNIFIED_AGENT_TIMEOUT_MS` (30s)
per round, 90 seconds total maximum.

**EC-031: Budget Exhaustion Mid-Task.**
1. Agent emits `BUDGET_EXHAUSTED` on event bus.
2. Integrator attempts micro-reallocation from idle agents (max 20%).
3. If that fails, suspend with partial result receipt.
4. SAT may grant emergency budget (one-time, logged, max 50% overage).
5. If denied, fail gracefully; partial work preserved in episodic memory.

**EC-032: SAT Department Conflict.**
1. SAT-CEO collects both directives with evidence.
2. RCE (Risk) has constitutional override for safety-critical conflicts.
3. Non-safety: department with higher aggregate Ihsan prevails.
4. Tie-break: more conservative directive wins (fail-safe default).
5. All conflicts logged for quarterly human review.

**EC-033: Agent Crash and Restart.**
1. Roster marks agent `Degraded` (`AgentState::Degraded` in Rust).
2. In-flight tasks re-queued to Integrator.
3. Restart with degraded permit (`SubAgentPermit::degrade()`, factor=0.5).
4. After 3 consecutive successes, promote back to `Idle`.
5. If 3 crashes within 5 minutes, `Suspended` + Sentinel diagnostic.

**EC-034: Stale RAE Optimization.**
Previous allocations retained. `RAE_STALE` flag set. If stale 2 weeks, escalate
to human governance. Emergency: `sat_ceo.force_rebalance()`.

---

## 3. Pseudocode

### 3.1 pat7_dispatch(user_intent)

```
FUNCTION pat7_dispatch(user_intent, context, roster):
    routing = entropy_route(user_intent.text, context)

    # Fast path: reflex cache hit bypasses orchestration
    IF routing.tier IN (TRIVIAL, SIMPLE):
        cached = reflex_cache.get_active(Active, task_hash(user_intent), policy_hash, now_ms())
        IF cached: RETURN ExecutionPlan(single_task=cached, trace="reflex_hit")

    # Planner decomposes into HTDAG
    htdag = planner.decompose(user_intent, context)
    IF htdag.is_empty():
        RETURN ExecutionPlan.error("decomposition_failed")

    # Assign specialists to DAG nodes (topological order)
    plan = ExecutionPlan(intent=user_intent, dag=htdag)
    FOR node IN htdag.topological_sort():
        role = classify_specialist(node)
        agent = roster.get(role)
        IF NOT agent.is_available():
            agent = roster.get(INTEGRATOR)  # fallback
            IF NOT agent.is_available():
                plan.mark_blocked(node, reason="no_available_agent")
                CONTINUE
        roster.mark_busy(role)
        plan.assign(node, agent)

    # MPC budget allocation
    bids = [agent.submit_budget_bid(node) FOR (node, agent) IN plan.assignments]
    allocation = mpc_aggregate(bids, total_budget=context.budget)
    FOR (node, agent), budget IN ZIP(plan.assignments, allocation):
        node.allocated_budget = budget

    # Ethicist pre-screening (VETO gate)
    ethicist = roster.get(ETHICIST)
    FOR node IN plan.all_nodes():
        veto = ethicist.pre_screen(node)
        IF veto.blocked:
            plan.mark_vetoed(node, veto.reason)
            roster.get(veto.node_agent).record_veto(now())

    RETURN plan
```

### 3.2 sat49_evaluate(request)

```
FUNCTION sat49_evaluate(negotiation_request):
    req = negotiation_request

    # Verify PAT signature
    IF NOT verify_ed25519(req.signature, req.pat_node_id.public_key, req.canonical_bytes()):
        RETURN NegotiationResponse(decision=REJECT, rationale="invalid_signature")

    # Parallel department evaluation
    evaluations = PARALLEL_MAP([
        ("risk",  rce7.evaluate_risk(req.task, req.budget)),
        ("cost",  irp7.estimate_cost(req.task, req.budget)),
        ("align", ego7.check_alignment(req.task)),
        ("econ",  ttm7.check_economic_impact(req.budget)),
        ("perf",  pxs7.estimate_latency(req.task)),
    ])

    # Risk gate (fail-closed)
    IF evaluations["risk"].score > 0.70:
        RETURN NegotiationResponse(decision=REJECT, risk_score=evaluations["risk"].score,
                                    rationale=evaluations["risk"].explanation)

    # Budget feasibility — counter if cost > 150% of request
    actual_cost = evaluations["cost"].total
    IF actual_cost > req.budget.total() * 1.50:
        RETURN NegotiationResponse(decision=COUNTER, cost_estimate=actual_cost,
                                    counter_budget=irp7.suggest_reduced_scope(req.task, req.budget),
                                    rationale="requested_budget_insufficient")

    # Gini impact check
    projected_gini = ttm7.project_gini_after(req.budget)
    IF projected_gini > ADL_GINI_THRESHOLD:
        RETURN NegotiationResponse(decision=REJECT,
                                    rationale=f"gini_violation: {projected_gini} > 0.35")

    # Composite alignment score
    alignment = (evaluations["align"].score * 0.40
               + (1.0 - evaluations["risk"].score) * 0.30
               + evaluations["econ"].health * 0.30)
    IF alignment < 0.85:
        RETURN NegotiationResponse(decision=COUNTER, alignment=alignment,
                                    rationale="alignment_below_threshold")

    RETURN NegotiationResponse(decision=ACCEPT, cost_estimate=actual_cost,
                                risk_score=evaluations["risk"].score, alignment=alignment,
                                signature=sign_ed25519(SAT_KEY, response.canonical_bytes()))
```

### 3.3 negotiate(pat_request, sat_evaluation)

```
FUNCTION negotiate(pat_agent, sat_ceo, task, budget, max_rounds=3):
    receipts = []
    current_budget = budget

    FOR round IN 1..max_rounds:
        request = NegotiationRequest(uuid4(), pat_agent.did, task, current_budget,
                                      task.priority, TIER_COMPLEX_BUDGET_MS,
                                      pat_agent.sign(request.canonical_bytes()))
        receipts.append(Receipt("PAT_SUBMIT", round, request))

        response = sat49_evaluate(request)
        receipts.append(Receipt("SAT_EVALUATE", round, response))

        IF response.decision == ACCEPT:
            contract = BilateralContract(request, response)
            contract.pat_sig = pat_agent.sign(contract.canonical_bytes())
            contract.sat_sig = sat_ceo.sign(contract.canonical_bytes())
            receipts.append(Receipt("CONTRACT_SIGNED", round, contract))
            RETURN NegotiationOutcome(AGREED, contract, receipts)

        IF response.decision == REJECT:
            receipts.append(Receipt("REJECTED", round, response.rationale))
            RETURN NegotiationOutcome(REJECTED, None, receipts)

        # COUNTER: PAT evaluates and revises
        IF pat_agent.accepts_counter(response.counter_budget, response.rationale):
            current_budget = response.counter_budget
        ELSE:
            current_budget = pat_agent.revise_budget(budget, response.counter_budget,
                                                      round, max_rounds)

    # Deadlock — escalate to watchdog
    receipts.append(Receipt("DEADLOCK", max_rounds, "escalating_to_watchdog"))
    wd = crown_watchdog.arbitrate(request, response, receipts)

    IF wd.action == FORCE_ACCEPT:
        contract = BilateralContract(request, wd.imposed_budget)
        contract.watchdog_sig = crown_watchdog.sign(contract.canonical_bytes())
        receipts.append(Receipt("WATCHDOG_FORCE_ACCEPT", max_rounds, contract))
        RETURN NegotiationOutcome(ARBITRATED, contract, receipts)
    IF wd.action == FORCE_REJECT:
        receipts.append(Receipt("WATCHDOG_REJECT", max_rounds, wd.rationale))
        RETURN NegotiationOutcome(REJECTED, None, receipts)

    receipts.append(Receipt("SUSPENDED", max_rounds, "pending_human_review"))
    RETURN NegotiationOutcome(SUSPENDED, None, receipts)
```

### 3.4 rae_weekly_optimize()

```
FUNCTION rae_weekly_optimize(sat_ceo, departments):
    # Collect department KPIs
    kpis = {}
    FOR dept IN departments:
        kpis[dept.code] = DepartmentKPI(
            dept.metrics.latency_percentile(95), dept.metrics.snr_rolling_mean(),
            dept.metrics.ihsan_rolling_mean(), dept.metrics.tasks_completed,
            dept.metrics.error_rate(), dept.metrics.budget_utilization())

    # Constrained optimization: max Ihsan s.t. Gini <= 0.35, infra >= 20%
    objective = MAXIMIZE(SUM(w[d] * kpis[d].ihsan_mean FOR d IN departments))
    constraints = [
        gini(allocations) <= ADL_GINI_THRESHOLD,
        allocations["IRP"] >= SAT_INFRASTRUCTURE_FLOOR_PCT / 100.0,
        SUM(allocations.values()) == 1.0,
        ALL(a >= 0.05 FOR a IN allocations.values()),
    ]
    result = convex_optimize(objective, constraints, solver="CLARABEL")

    IF NOT result.converged:
        sat_ceo.set_flag("RAE_STALE")
        RETURN RAEResult(converged=False, allocations=sat_ceo.current_allocations)

    # Publish as signed constitutional receipt and apply
    new_alloc = result.solution
    receipt = ConstitutionalReceipt("RAE_REBALANCE", sat_ceo.current_allocations,
                                     new_alloc, kpis, compute_gini(new_alloc.values()))
    receipt.signature = sat_ceo.sign(receipt.canonical_bytes())

    sat_ceo.current_allocations = new_alloc
    FOR dept IN departments: dept.set_budget_weight(new_alloc[dept.code])
    sat_ceo.clear_flag("RAE_STALE")
    RETURN RAEResult(converged=True, allocations=new_alloc, receipt=receipt)
```

---

## 4. TDD Anchors

```
TEST pat7_dispatch_trivial_uses_reflex:
    roster = make_test_roster(all_idle=True)
    cache.insert(test_rule(mode=Active, trigger=hash("hello")))
    plan = pat7_dispatch(Intent("hello"), ctx, roster)
    ASSERT plan.trace == "reflex_hit"
    ASSERT ALL(a.state == Idle FOR a IN roster.agents)

TEST pat7_dispatch_complex_assigns_specialists:
    roster = make_test_roster(all_idle=True)
    plan = pat7_dispatch(complex_intent, ctx, roster)
    ASSERT plan.task_count >= 3
    ASSERT "Planner" IN {n.agent.role FOR n IN plan.assigned_nodes}

TEST pat7_ethicist_vetoes_unsafe_task:
    plan = pat7_dispatch(unsafe_intent, ctx, make_test_roster())
    vetoed = [n FOR n IN plan.all_nodes() IF n.vetoed]
    ASSERT len(vetoed) >= 1 AND "constitutional_violation" IN vetoed[0].veto_reason

TEST sat49_rejects_invalid_signature:
    req = NegotiationRequest(task=test_task, signature="invalid")
    ASSERT sat49_evaluate(req).decision == REJECT

TEST sat49_rejects_gini_violation:
    mock_gini_projection(0.42)
    req = NegotiationRequest(task=expensive_task, budget=huge_budget)
    ASSERT "gini_violation" IN sat49_evaluate(req).rationale

TEST negotiate_accepts_on_first_round:
    mock_sat_evaluator(always_accept=True)
    outcome, receipts = negotiate(pat, sat_ceo, task, budget)
    ASSERT outcome.status == AGREED AND len(receipts) == 3
    ASSERT outcome.contract.pat_sig IS NOT None

TEST negotiate_deadlock_escalates_to_watchdog:
    mock_sat_evaluator(always_counter=True)
    outcome, receipts = negotiate(pat, sat_ceo, task, budget, max_rounds=3)
    ASSERT outcome.status IN (ARBITRATED, REJECTED, SUSPENDED)
    ASSERT ANY(r.type == "DEADLOCK" FOR r IN receipts)

TEST negotiate_counter_accepted_second_round:
    mock_sat_evaluator(counter_then_accept=True)
    outcome, _ = negotiate(pat, sat_ceo, task, budget)
    ASSERT outcome.status == AGREED

TEST rae_weekly_respects_gini_ceiling:
    result = rae_weekly_optimize(sat_ceo, make_test_departments(7))
    ASSERT result.converged == True
    ASSERT compute_gini(result.allocations.values()) <= ADL_GINI_THRESHOLD

TEST agent_crash_degrades_and_recovers:
    agent = make_test_roster().get(CODER)
    agent.record_failure(now())
    ASSERT agent.state == Degraded
    FOR _ IN 1..3: agent.record_completion(1000, Confidence(0.95), now())
    ASSERT agent.state == Idle
```

---

## 5. Cross-References

### Python Modules
- `core/integration/constants.py` — `PAT_AGENT_COUNT`, `PAT_AGENT_NAMES`, `PAT_TRUST_STAGES`, `SAT_AGENTS_PER_NODE`, `SAT_BOOTSTRAP_ROLES`, `SAT_INFRASTRUCTURE_FLOOR_PCT`, `ADL_GINI_THRESHOLD`, `TIER_COMPLEX_BUDGET_MS`, `UNIFIED_AGENT_TIMEOUT_MS`
- `core/orchestration/` — `EventBus`, `TeamPlanner`, `EnhancedTeamPlanner`, `BackgroundAgentRegistry`, `OpportunityPipeline`, `ProactiveScheduler`
- `core/a2a/` — `AgentCard`, `TaskCard`, `A2AMessage`, `A2AEngine`, `TaskManager`, `A2ATransport`

### Rust Crates
- `bizra-agent/src/types.rs` — `AgentId`, `AgentRole` (7 PAT roles)
- `bizra-agent/src/roster.rs` — `AgentRoster`, `AgentEntry`, `AgentState`, `PAT_SIZE = 7`
- `bizra-agent/src/orchestrator.rs` — `Orchestrator`, `ExecutionPlan`
- `bizra-agent/src/sub_agent.rs` — `SubAgent`, `SubAgentPermit` (degraded permits)
- `bizra-agent/src/omni_kernel.rs` — 8-line sovereign loop
- `bizra-hooks/` — Event bus (8 shards, FNV-1a), `IhsanScore`

### Atlas v5 Phases
- Phase 01 — Sovereign Node (FR-010: genesis spawns PAT-7 + registers with SAT-49)
- Phase 02 — Cognition Engine (entropy routing feeds PAT dispatch)
- Phase 04 — HDA Execution (PAT delegates desktop actions via TeleScript)
- Phase 05 — Blockchain Economics (TTM-7 manages SEED/BLOOM tokens)
- Phase 06 — Governance + Soul (FATE Gate, Ihsan Wall, Crown Verification)

### Standing on Giants
- Sacerdoti (1974): ABSTRIPS hierarchical planning (HTDAG lineage)
- Yao (1982): Secure multi-party computation (MPC budget optimizer)
- Hewitt (1973): Actor Model (agent isolation + message passing)
- Castro & Liskov (1999): PBFT (SAT consensus)
- Nash (1950): Bargaining problem (PAT-SAT negotiation equilibria)
- Al-Ghazali (1095): Ihsan as the measure of excellence
