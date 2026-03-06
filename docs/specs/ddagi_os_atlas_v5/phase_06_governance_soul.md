# Phase 06 — Governance & Soul: RSL, FATE Gate, Crown Verification

> Source: Atlas v5.0 — Diagrams D9 (RSL+FATE), D10 (H0/H1/H2), D13 (Governance Pipeline)
> Status: SPECIFICATION SEALED | SNR: 0.95

---

## 1. Functional Requirements

### FR-060: Runtime Soul Layer (RSL)

The RSL compiles sacred founding documents (Al-Risalah, Al-Bazrah, Ihsan) into
machine-enforceable constraints embedded in every agent instance. The Soul is a
hard runtime boundary -- not advisory.

**Compilation Pipeline:** `PARSE (TOML) -> ENCODE (typed IR) -> HASH (BLAKE3) -> EMBED (agent spawn)`

1. **Parse.** `constitution.toml` with typed sections. Rejects unknown keys.
2. **Encode.** Typed IR nodes: `Threshold(name, value, cmp)`, `Invariant(name, pred)`,
   `Bound(name, min, max)`. Deterministic -- same input = identical bytes.
3. **Hash.** `BLAKE3(canonical_bytes(ir))` -- immutable reference. Anchors the
   BlockGraph genesis block (FR-050). Mutation invalidates the entire proof chain.
4. **Embed.** Read-only injection at PAT-7/SAT-5 spawn. Continuous Auditor
   (FR-013) verifies RSL hash every `GINI_MEASUREMENT_INTERVAL_S` (3600s).

**Three Kernel Invariants** (`KERNEL_INVARIANTS`):
- `RIBA_ZERO` -- No exploitation. No interest. No harm.
- `CLAIM_MUST_BIND` -- No hallucination. Every claim has evidence.
- `IHSAN_FLOOR` -- Excellence is the minimum. 0.99 consensus threshold.

### FR-061: AEGIS Policy Engine

AEGIS classifies every agent action into three zones before execution:

| Zone       | Action               | Example                                        |
|------------|----------------------|------------------------------------------------|
| Rules      | ALLOW (auto-execute) | Read file, cache lookup, format response       |
| Bounds     | DENY (blocked+log)   | Delete system files, leak PII, bypass gate     |
| Gray Zone  | ESCALATE (human)     | Large transfer, irreversible mutation           |

Policies are TOML-encoded, constitution-key-signed, versioned. Gray zone
escalation: emit on action bus, human has `UNIFIED_AGENT_TIMEOUT_MS` (30s).
Timeout applies conservative default (DENY for destructive, ALLOW for read-only).
If > 10 escalations/hour per agent, Sentinel triggers autonomy review.

### FR-062: FATE Gate

4-layer pre-execution verification. Fail mode: closed (`GATE_FAIL_MODE`).

**Layer F -- Formal (Z3 SMT).** Encodes pre/postconditions as Z3 constraints.
Proves no kernel invariant is violated. Implementation: `z3_ihsan.rs`.
Timeout: `GATE_OVERHEAD_BUDGET_MS` (50ms). On timeout: `_conservative_fallback_check()`
(NOT `_manual_constraint_check`) -- default-deny, stricter than Z3. Proofs cached
by `BLAKE3(constraints + constitution_hash)`.

**Layer A -- Alignment (Constitutional).** Scores against 6-dim operational Ihsan
tensor (`IHSAN_OPERATIONAL_WEIGHTS`). Composite must exceed `IHSAN_GATE_MINIMUM`
(0.85) for pass, `UNIFIED_IHSAN_THRESHOLD` (0.95) for production excellence.

**Layer T -- Testing (Sandbox).** Pillar 3 simulation (SNR floor 0.70).
Property-based tests, fuzz, integration checks.

**Layer E -- Ethical (Shariah + Fairness).** Kernel invariants (no riba, no gharar),
ADL Gini projection (`ADL_GINI_THRESHOLD` 0.35), Daughter Test. Any failure = DENY.

**Gate Chains.** Rust (`gate_chain.rs`): Schema > SNR > Ihsan > License.
PCI (`core/pci/gates.py`): Schema > Signature > Timestamp > Replay > Ihsan > SNR > Policy.
PCI places Ihsan before SNR because untrusted peer messages need ethical screening first.

### FR-063: Runtime Enforcement

Every decision is intercepted before dispatch. Synchronous in the action bus hot path:

```
Decision -> [1] AEGIS Zone -> BOUNDS=DENY | GRAY=ESCALATE | RULES=continue
  -> [2] FATE (F-A-T-E) -> fail=DENY
  -> [3] RSL Hash Verify -> mismatch=HALT node
  -> [4] Crown H0/H1/H2 -> HALT=block+governance proposal | WARN=proceed degraded
  -> [5] DISPATCH -> [6] Receipt (Merkle-linked, Ed25519-signed, evidence ledger)
```

Full pipeline targets < 100ms. Unhandled exception in any stage = DENY (fail-closed).

### FR-064: H0/H1/H2 Invariant Verification (Crown Layer)

CROWN (Constitutional Rights Operating With Neutrality) audits three independent
horizons. Implementation: `core/governance/crown_layer.py`.

**H0 -- Ethical:** No Gharar (HALT), No Riba (HALT), Ihsan >= 0.95 (HALT),
Gini <= 0.35 (HALT), RSL hash match (HALT).

**H1 -- Performance:** Latency <= 30s (WARN), SNR >= 0.85 (WARN),
throughput within tier budget (WARN), resource efficiency >= 80% (WARN).

**H2 -- Safety:** Daughter Test (HALT), reversible action (HALT if not),
blast radius bounded (HALT), human override reachable (WARN).

**Verdict:** ACCEPT (all PASS), REJECT (any HALT -- auto-generate governance
proposal after 3 consecutive), REVISE (WARN only -- proceed degraded).
Severity: PASS(0) < WARN(1) < HALT(2). Aggregate = worst across horizons.

### FR-065: Governance Pipeline

No individual may unilaterally modify the constitution.

```
IDEA -> DEPT-CEO -> SAT-CEO (impact+alignment+risk) -> CEO-SIG (Ed25519)
  -> ON-CHAIN (7-day debate) -> BLOOM-WEIGHTED VOTE
  -> PROGRESSIVE GATES (Shadow 48h | Canary 5% 72h | Full 100% 7d)
  -> CROWN VERDICT -> ACCEPTED | REJECTED | REVISION
```

**Proposal types:** ParameterChange, PolicyUpdate, EconomicRule, ConstitutionalAmendment.
Standard threshold: 67% weighted approval. Constitutional amendment: 90%.
Quorum: `sat_frontier_quorum()` (BFT 2f+1). Voting weight = `bloom_balance * reputation`.

**Progressive Gates:**

| Gate   | Duration | Scope | Rollback Trigger                            |
|--------|----------|-------|---------------------------------------------|
| Shadow | 48h      | 0%    | Any H0 HALT in simulation                   |
| Canary | 72h      | 5%    | SNR drop > 15% OR Gini breach OR 2+ HALTs   |
| Full   | 7d       | 100%  | Crown REJECT on any horizon                 |

Rollback constants: `ROLLBACK_CONSECUTIVE_BREACHES` (2), `ROLLBACK_SNR_DROP_THRESHOLD` (0.15).
Final Crown verdict: ACCEPTED = constitution hash updated, REJECTED = full revert,
REVISION = partial revert + 30-day deadline for resubmission.

---

## 2. Edge Cases

**EC-060: Z3 Timeout.** Solver exceeds 50ms. `_conservative_fallback_check()`
applies default-deny. Logged as `fate_layer=F_TIMEOUT`. If timeout rate > 10%
in 5 minutes, Sentinel triggers cache warming. `UNKNOWN` = same as timeout.

**EC-061: Gray Zone Escalation Exhaustion.** Same policy triggers > 5 timeouts
in 24h: auto-promote to Rules:ALLOW with reduced scope (read-only only).
Destructive actions: DENY is absolute, no auto-promotion. Auto-generates
governance proposal to reclassify the zone.

**EC-062: Governance Quorum Failure.** Below quorum after 7 days: author may
extend once (14d max). Still no quorum: archived as LAPSED, 30-day cooldown.
3 consecutive lapses trigger federation health diagnostic.

**EC-063: Crown REVISE Loop.** Max 3 REVISE cycles per proposal lineage. After
third: force-REJECTED, 90-day cooldown in the same policy domain.

**EC-064: Partial FATE Failure.** Gate chain short-circuits: first layer failure
returns DENY, skips remaining layers. Receipt records passed/failed/skipped.
Exception: if Layer F returns UNKNOWN and conservative fallback ALLOWs, all
remaining layers (A, T, E) must still be evaluated.

---

## 3. Pseudocode

### 3.1 fate_gate_check(action)

```
FUNCTION fate_gate_check(action, rsl, crown_layer):
    receipt = FATEReceipt(action_id=action.id, timestamp=now_utc())
    layers_passed = []

    # Layer F: Z3 SMT
    cache_key = BLAKE3(action.preconditions_bytes() + rsl.constitution_hash)
    f_result = z3_proof_cache.get(cache_key)
    IF f_result IS None:
        TRY:
            solver = Z3Solver(timeout_ms=GATE_OVERHEAD_BUDGET_MS)
            FOR inv IN rsl.kernel_invariants: solver.assert(inv.to_z3())
            solver.assert(action.postcondition.to_z3())
            f_result = solver.check()
            IF f_result == SAT: z3_proof_cache.put(cache_key, f_result)
        EXCEPT Z3TimeoutError:
            f_result = UNKNOWN

    IF f_result == UNSAT:
        RETURN FATEResult(DENY, receipt.fail("F", "Z3 proven unsafe"), layers_passed)
    IF f_result == UNKNOWN:
        IF NOT _conservative_fallback_check(action, rsl).safe:
            RETURN FATEResult(DENY, receipt.fail("F", "conservative_fallback_deny"), [])
        receipt.add_warning("F_FALLBACK", "Z3 inconclusive")
    layers_passed.append("F")

    # Layer A: Ihsan Tensor
    scores = {d: evaluate_dimension(action, d) FOR d IN IHSAN_OPERATIONAL_WEIGHTS}
    composite = SUM(scores[d] * IHSAN_OPERATIONAL_WEIGHTS[d] FOR d IN scores)
    IF composite < IHSAN_GATE_MINIMUM:  # 0.85
        RETURN FATEResult(DENY, receipt.fail("A", f"ihsan {composite:.4f} < 0.85"), layers_passed)
    layers_passed.append("A")

    # Layer T: Sandbox Simulation
    sandbox = Sandbox(pillar=3, snr_floor=PILLAR_3_SANDBOX_SNR_FLOOR)
    props = sandbox.run_property_tests(action, iterations=100)
    fuzz  = sandbox.run_fuzz(action, seed_corpus=action.input_samples)
    integ = sandbox.verify_composition(action, system_state)
    IF props.violations > 0 OR fuzz.crashes > 0 OR NOT integ.compatible:
        RETURN FATEResult(DENY, receipt.fail("T", first_failure_reason(props, fuzz, integ)), layers_passed)
    layers_passed.append("T")

    # Layer E: Ethical
    IF action.involves_financial AND (detect_riba(action) OR detect_gharar(action)):
        RETURN FATEResult(DENY, receipt.fail("E", "riba_or_gharar"), layers_passed)
    IF project_gini_after(action) > ADL_GINI_THRESHOLD:  # 0.35
        RETURN FATEResult(DENY, receipt.fail("E", "gini_violation"), layers_passed)
    IF NOT daughter_test(action):
        RETURN FATEResult(DENY, receipt.fail("E", "daughter_test_failed"), layers_passed)
    layers_passed.append("E")

    RETURN FATEResult(ALLOW, receipt.pass(layers_passed), layers_passed)
```

### 3.2 crown_verify(decision)

```
FUNCTION crown_verify(decision, system_state):
    state = SystemState(
        ihsan_score=decision.ihsan_composite, snr_score=decision.snr_score,
        gini_coefficient=decision.projected_gini, latency_ms=decision.latency_ms,
        has_riba=decision.flags.get("riba", False),
        has_gharar=decision.flags.get("gharar", False),
        is_reversible=decision.is_reversible,
        human_override_available=system_state.human_online,
        has_audit_trail=decision.receipt IS NOT None,
    )
    crown = CROWNLayer(ihsan_threshold=0.95, snr_threshold=0.85,
                        gini_threshold=0.35, latency_bound_ms=30000)
    verdict = crown.render_verdict(state)

    IF verdict.halted:
        halts = [h FOR h IN verdict.horizons IF h.status == HALT]
        IF consecutive_halts(decision.agent_id) >= 3:
            blockgraph.submit_governance(auto_generate_proposal("AGENT_REVIEW", halts))
        RETURN CrownResult(REJECT, verdict, halted_horizons=halts)
    IF len(verdict.warnings) > 0:
        RETURN CrownResult(REVISE, verdict, warnings=verdict.warnings)
    RETURN CrownResult(ACCEPT, verdict)
```

### 3.3 governance_pipeline(proposal)

```
FUNCTION governance_pipeline(proposal, sat_ceo, blockgraph, federation):
    VALID_TYPES = {"ParameterChange","PolicyUpdate","EconomicRule","ConstitutionalAmendment"}
    IF proposal.type NOT IN VALID_TYPES: RETURN GovResult(REJECTED, "invalid_type")
    IF NOT verify_ed25519(proposal.signature, proposal.author.public_key, proposal.bytes()):
        RETURN GovResult(REJECTED, "invalid_signature")

    # Stages 2-3: Department + SAT-CEO review
    dept_review = sat_ceo.route_to_department(proposal).evaluate(proposal)
    IF dept_review.risk_score > 0.90: RETURN GovResult(REJECTED, "dept_risk_veto")
    alignment = sat_ceo.score_alignment(proposal, IHSAN_CANONICAL_WEIGHTS)
    IF alignment < IHSAN_GATE_MINIMUM: RETURN GovResult(REJECTED, "alignment_below_gate")

    # Stage 4-5: Sign + on-chain
    threshold = 0.90 IF proposal.type == "ConstitutionalAmendment" ELSE 0.67
    gov_block = GovernanceBlock(proposal, sat_ceo.sign_assessment(proposal),
                                debate_ends=now_utc()+days(7), vote_threshold=threshold)
    blockgraph.append(gov_block)

    # Stage 6-7: Debate + BLOOM-weighted vote
    responses = blockgraph.collect_responses(gov_block.cid, deadline=gov_block.debate_ends)
    quorum = sat_frontier_quorum(federation.node_count)
    total_w, approve_w = 0, 0
    FOR r IN responses:
        IF NOT verify_ed25519(r.signature, r.voter.public_key, r.bytes()): CONTINUE
        w = r.voter.bloom_balance * r.voter.reputation_multiplier
        total_w += w
        IF r.vote == SUPPORT: approve_w += w

    IF total_w == 0 OR len(responses) < quorum:
        RETURN GovResult(QUORUM_FAILED, votes=len(responses), required=quorum)
    IF approve_w / total_w < threshold:
        RETURN GovResult(REJECTED, f"approval below {threshold:.0%}")

    # Stage 8: Progressive gates
    IF run_shadow(proposal, 48h).h0_halts > 0: RETURN GovResult(REJECTED, "shadow_halt")
    canary = run_canary(proposal, 72h, 5%, CANARY_DEFAULT_SALT)
    IF canary.snr_drop > 0.15 OR canary.halts >= 2:
        rollback(proposal); RETURN GovResult(REJECTED, "canary_fail")
    IF run_full(proposal, 7d).crown_rejects > 0:
        rollback(proposal); RETURN GovResult(REJECTED, "full_rollout_fail")

    # Stage 9: Crown verdict
    cv = crown_verify(proposal.as_decision(), federation.current_state())
    IF cv.status == REJECT: rollback(proposal); RETURN GovResult(REJECTED, "crown_reject")
    IF cv.status == REVISE:
        RETURN GovResult(REVISION, feedback=cv.warnings, deadline=now_utc()+days(30))

    new_hash = BLAKE3(apply_proposal(proposal, constitution()).canonical_bytes())
    blockgraph.update_constitution_hash(new_hash)
    RETURN GovResult(ACCEPTED, constitution_hash=new_hash, cid=gov_block.cid)
```

### 3.4 rsl_compile(constitution)

```
FUNCTION rsl_compile(constitution_toml_path):
    raw = parse_toml(constitution_toml_path)
    IF raw IS None OR "kernel_invariants" NOT IN raw:
        RAISE ConstitutionParseError("missing kernel_invariants")

    constraints = []
    FOR name, section IN raw.items():
        MATCH name:
            "kernel_invariants" -> [constraints.append(Invariant(i["name"], i["predicate"])) FOR i IN section]
            "thresholds"        -> [constraints.append(Threshold(t["name"], t["value"], t.get("cmp",">=")))]
            "bounds"            -> [constraints.append(Bound(b["name"], b["min"], b["max"]))]
            _                   -> RAISE ConstitutionParseError(f"unknown section: {name}")

    canonical = canonical_encode(sorted(constraints, key=lambda c: c.name))
    RETURN RSL(constraints=constraints, canonical_bytes=canonical,
               constitution_hash=BLAKE3(canonical), version=raw.get("version", CONSTITUTION_VERSION))
```

---

## 4. TDD Anchors

```
TEST fate_gate_denies_z3_proven_unsafe:
    result = fate_gate_check(make_action(postcondition="balance < 0"), make_rsl(), crown)
    ASSERT result.verdict == DENY AND result.receipt.failed_layer == "F"

TEST fate_gate_falls_back_on_z3_timeout:
    result = fate_gate_check(make_action(postcondition=huge_constraint()), make_rsl(), crown)
    ASSERT result.receipt.warnings[0].code == "Z3_TIMEOUT"

TEST fate_gate_denies_low_ihsan:
    result = fate_gate_check(make_action(ihsan={"moral_clarity": 0.3}), make_rsl(), crown)
    ASSERT result.verdict == DENY AND result.receipt.failed_layer == "A"

TEST fate_gate_denies_gini_breach:
    mock_project_gini(0.42)  # > 0.35
    result = fate_gate_check(make_financial_action(), make_rsl(), crown)
    ASSERT result.verdict == DENY AND "gini" IN result.receipt.reason

TEST crown_verify_halts_on_riba:
    result = crown_verify(make_decision(flags={"riba": True}, ihsan=0.98), state)
    ASSERT result.status == REJECT
    ASSERT ANY(h.horizon == H0_ETHICAL AND h.status == HALT FOR h IN result.verdict.horizons)

TEST crown_verify_warns_on_latency:
    result = crown_verify(make_decision(ihsan=0.97, latency_ms=35000), state)
    ASSERT result.status == REVISE
    ASSERT ANY(h.horizon == H1_PERFORMANCE FOR h IN result.verdict.warnings)

TEST governance_rejects_below_quorum:
    mock_federation(node_count=10, responding=2)
    result = governance_pipeline(make_proposal("PolicyUpdate"), sat_ceo, bg, fed)
    ASSERT result.status == QUORUM_FAILED

TEST governance_constitutional_amendment_needs_90_pct:
    mock_votes(approve_ratio=0.85)  # Below 0.90
    result = governance_pipeline(make_proposal("ConstitutionalAmendment"), sat_ceo, bg, fed)
    ASSERT result.status == REJECTED AND "approval" IN result.reason

TEST governance_canary_rollback_on_snr_drop:
    mock_canary(snr_drop=0.20)  # > ROLLBACK_SNR_DROP_THRESHOLD (0.15)
    result = governance_pipeline(make_proposal("ParameterChange"), sat_ceo, bg, fed)
    ASSERT result.status == REJECTED AND "canary" IN result.reason

TEST rsl_compile_rejects_unknown_section:
    EXPECT_RAISE ConstitutionParseError: rsl_compile(toml_with_unknown_section)
    ASSERT "unknown section" IN str(error)
```

---

## 5. Cross-References

### Python Modules
- `core/governance/crown_layer.py` -- `CROWNLayer`, `CROWNHorizon`, `CROWNStatus`, `CROWNVerdict`, `SystemState`. Worst-of aggregation across H0/H1/H2.
- `core/governance/constitutional_gate.py` -- Re-exports `ConstitutionalGate` (Four Pillars: Runtime, Museum, Sandbox, Cutoff).
- `core/governance/adaptive_ihsan.py` -- Adaptive threshold management.
- `core/governance/model_license_gate.py` -- License gate (SNR before Ihsan for trusted local models).
- `core/pci/gates.py` -- `PCIGateKeeper`, 7-gate chain. Ihsan before SNR for untrusted peers.
- `core/integration/constants.py` -- `UNIFIED_IHSAN_THRESHOLD` (0.95), `IHSAN_GATE_MINIMUM` (0.85), `ADL_GINI_THRESHOLD` (0.35), `GATE_WEIGHTS`, `GATE_FAIL_MODE`, `GATE_OVERHEAD_BUDGET_MS` (50), `KERNEL_INVARIANTS`, `ROLLBACK_*` thresholds.
- `core/proof_engine/evidence_ledger.py` -- Merkle-chained receipt log.

### Rust Crates
- `fate-binding/src/lib.rs` -- `FateValidator`, `GateResult`, `ChallengeResult`. Re-exports thresholds from `bizra-core`.
- `fate-binding/src/z3_ihsan.rs` -- `IhsanVerifier`, `ProofCertificate`. Z3 `Real` constraints, not heuristic.
- `fate-binding/src/gate_chain.rs` -- `GateChain`, `Gate` trait. Schema > SNR > Ihsan > License. Short-circuits.
- `fate-binding/src/dilithium.rs` -- Dilithium-5 post-quantum signatures for CapabilityCards.
- `bizra-core/src/lib.rs` -- `IHSAN_THRESHOLD` (0.95), `SNR_THRESHOLD` (0.85).

### Atlas v5 Phases
- Phase 01 -- FR-010: human accepts Al-Risalah; FR-013: FATE Gate is stage 1 of Constitutional Self-Harness
- Phase 03 -- FR-032: H0/H1/H2 watchdog monitors PAT-SAT negotiations
- Phase 05 -- FR-050: genesis block anchored to Al-Bazrah hash; BLOOM-weighted governance voting

### Standing on Giants
- Al-Ghazali (1095): Ihsan -- 8-dimensional ethical tensor
- de Moura & Bjorner (2008): Z3 SMT Solver -- formal verification
- Shannon (1948): SNR as quality floor
- Lamport (1982): Safety and liveness (H2)
- Rawls (1971): Justice as Fairness (H0, ADL Gini)
- Saltzer & Schroeder (1975): Fail-safe defaults (GATE_FAIL_MODE = "closed")
- Fowler (2010): Canary releases (progressive gates)
- Nygard (2007): Circuit breakers (REVISE loop limit)
