# Phase 10 — Omega Loop: Self-RLVR, Myelination, Value Cycle, Roadmap

> Source: Atlas v5.0 — Diagrams D28 (Omega Loop), D15 (Deployment Roadmap), D16 (12-Step Value Cycle)
> Status: SPECIFICATION SEALED | SNR: 0.95

---

## 1. Functional Requirements

### FR-100: Omega Loop

The autonomous optimization heartbeat. Closes the full cycle from intent to self-improvement with zero human intervention, except where FATE gate or AEGIS Gray Zone demands confirmation. Each tick is one complete transit through the 12-Step Value Cycle (FR-003, Phase 00), producing a cryptographically chained receipt regardless of outcome.

**Tick sequence:**

1. **Intent / Mission** -- Arrives via user input, Node0 mission queue, or proactive HHMM prediction. Encoded as `OmniCycle` with BLAKE3 intent hash (`omni/intent/v1:` prefix).

2. **FATE Gate (simulate + veto)** -- The mission is evaluated by the four FATE layers: Formal (Z3), Alignment (RSL), Testing (property fuzz), Ethical (Shariah audit). Verdict is PERMIT, DENY, or GRAY. On DENY, auto-adjust: strip destructive verbs, reduce scope, retry once. Second DENY produces a negative receipt and exits the tick. GRAY requires human confirmation within `UNIFIED_AGENT_TIMEOUT_MS` (30s).

3. **Route via Entropy / SNR** -- `EntropyRouter` classifies the task into TRIVIAL..FRONTIER tiers (FR-020, Phase 02). Shannon entropy + structural complexity determines whether the tick follows the S1 fast path or the S2 deliberative path.

4. **S1 Hit (O(1) reflex)** -- `ReflexCache.get_active()` checks Tier-1 compiled reflexes and Tier-2 Engram factual cache. A hit serves the response at O(1) with 120ms constant security overhead (FATE gate still runs on every action). Execution skips to step 8.

5. **S2 Miss (PAT-7 + GoT deliberation)** -- On cache miss, the full test-time diffusion pipeline runs: GoT generates >= 9 candidates, PBFT voting denoises, DiffusionReasoningAmplifier injects HMM context, Aha Detector locks at SNR >= 0.90 and sigma < 0.20. PAT-7 agents produce the final action plan.

6. **Execute via HDA (TeleScript -> AHK)** -- The action plan is compiled to TeleScript, PSI-AST validated, dispatched via JSON-RPC to the AHK bridge (TCP:9742), and executed as real keystrokes on the desktop.

7. **UIA Closed-Loop Verification** -- The UI Automation verifier captures the post-action screen state and compares against expected postconditions. Binary outcome: verified or not-verified. Not-verified path produces a negative reward and adds the pattern to an avoid-list (max 1000 entries, LRU eviction).

8. **PoI Receipt (signed + chained)** -- A `CycleReceipt` is constructed with BLAKE3 `pivot_chain_hash` binding all decision evidence. Ed25519-signed by the node identity. Appended to the append-only event log with Merkle chain back to genesis.

9. **SAT Oracle Independent Scoring** -- The SAT-5 Oracle agent scores the receipt independently of the executing agent. Scores the 8-dim Ihsan tensor (`IHSAN_CANONICAL_WEIGHTS`) and Shannon SNR. This is the deterministic verification that makes RLVR possible: either the receipt chain verifies or it does not.

10. **Mint or Reject** -- If Ihsan >= `IHSAN_BLOOM_ELIGIBILITY` (0.90): mint SEED via `MetabolicLedger.mint_poi_yield()` with emission decay. If Ihsan >= `IHSAN_CONFORMANCE_JOIN` (0.95): additionally eligible for BLOOM minting. If below `IHSAN_GATE_MINIMUM` (0.85): reject reward entirely, receipt marked `gate_passed = false`.

11. **RLVR Reward (deterministic)** -- The reward signal is computed from the verified receipt chain (FR-101). Policy update is queued via `TtrlEngine.queue_update()`. SSO constraint is checked before the update commits (FR-101).

12. **Pattern Stats + Myelination Check** -- Update pattern statistics for this intent class. If the myelination threshold is met (FR-102), compile the S2 trace into an S1 reflex. Publish the reflex capsule for federation diffusion (FR-102).

**Loop closure invariant:** Every tick produces exactly one signed receipt. The receipt's `pivot_chain_hash` links to the previous receipt via Merkle chain. Missing receipts are detectable by sequence gap.

### FR-101: Self-RLVR (Reinforcement Learning from Verifiable Rewards)

Self-RLVR eliminates the human labeler. The reward is deterministic: the PoI receipt chain either verifies (positive) or not (negative/zero). Ihsan and SNR provide continuous gradient on top of the binary signal.

**Reward function:** `reward = f(receipt_verified, ihsan_score, cpva_actual, majority_fraction)` where `receipt_verified` is binary verification, `ihsan_score` is the 8-dim tensor dot product, `cpva_actual` is Compute-Per-Verified-Action yield, and `majority_fraction` is PAT consensus fraction. All four values are deterministic.

**Policy update pipeline:** (1) PAT agents produce N candidates. (2) Majority vote selects `best_answer` (GRPO signal). (3) Reward computed from verified receipt chain. (4) Update queued in `TtrlEngine.queue` (lazy). (5) Before commit: SSO check -- `drift <= epsilon` required, else rejected (`TtrlStats.updates_rejected`). (6) After commit: pattern statistics updated for myelination.

**Constraints:** Ihsan floor (`IHSAN_GATE_MINIMUM` = 0.85) vetoes any update producing sub-floor outputs. SSO prevents catastrophic weight drift. Together they guarantee self-improvement never degrades below constitutional minimums.

**CPVA improvement curve** (TTRL paper): Month 1 +0%, Month 3 +50%, Month 12 +211% (Qwen-2.5-Math-7B baseline).

### FR-102: Myelination (S2 -> S1 Compilation)

Biological metaphor: repeated axon activation triggers myelination (100x signal speed). In BIZRA, repeated verified S2 successes compile into S1 reflex rules, eliminating inference latency.

**Myelination threshold:** >= `MYELINATION_MIN_SUCCESSES` (5) verified successes with average Ihsan >= `IHSAN_THRESHOLD` (0.95) and path variance <= `max_path_variance` (0.10). The `ReflexCompiler.evaluate()` enforces all three conditions.

**Compilation pipeline:** (1) `ReflexCompiler.record_success(trigger, sample)` accumulates `CompileSample` records keyed by `TriggerHash`. (2) When `sample_count >= min_success_chains`, `evaluate()` checks: avg Ihsan >= 0.95, avg SNR >= 0.90, path variance <= 0.10. (3) On pass: `ReflexRule` created with `compile_ihsan`, `compile_snr`, `policy_hash`, most-common `route_signature`. (4) Inserted into `ReflexCache` as `ReflexMode::Active` (unlike G.R.A.S.P. which starts in Shadow). (5) Reflex capsule published via `gossip("reflex_diffusion", capsule)`.

**Reflex capsule structure:**

| Field              | Type      | Description                              |
|--------------------|-----------|------------------------------------------|
| `trigger_hash`     | `[u8;32]` | BLAKE3 of intent class + context hash    |
| `action_template`  | Template  | Route signature + primary agent          |
| `compile_ihsan`    | `f32`     | Average Ihsan at compilation time        |
| `compile_snr`      | `f32`     | Average SNR at compilation time          |
| `author_pubkey`    | `[u8;32]` | Ed25519 public key of authoring node     |
| `signature`        | `[u8;64]` | Ed25519 signature over capsule contents  |
| `proof_chain`      | `Vec<H>`  | BLAKE3 hashes of the N source receipts   |

**Peer adoption:** Verify Ed25519 signature, check `compile_ihsan >= IHSAN_GATE_MINIMUM`, FATE gate in shadow mode, insert as `ReflexMode::Shadow`. Requires `REFLEX_PRECIPITATION_HITS` (3) shadow matches before Active promotion. A node never executes an unverified foreign reflex.

**Invalidation:** Ihsan drift > `REFLEX_INVALIDATION_DELTA` (0.05), age > `REFLEX_STALENESS_DAYS` (30), or `policy_hash` change. Invalidated rules revert to Shadow.

**Performance gain (Phase 02, FR-024):**

| Metric           | S2 (First)  | S1 (Myelinated) | Gain    |
|------------------|-------------|------------------|---------|
| Latency          | 4.65 s      | 0.75 s           | 6.2x    |
| Nodes explored   | 9+ (GoT)    | 1 (hash)         | O(1)    |
| PBFT rounds      | 2           | 0                | Elim.   |
| Security overhead| 120 ms      | 120 ms           | Constant|

### FR-103: Deployment Phases

Four phases, each gating on measurable constitutional criteria. Progression requires unanimous SAT-5 Oracle verdict. No phase can be skipped.

| Phase | Scale | Objective | Gate Criteria | Economy | Deliverable |
|-------|-------|-----------|---------------|---------|-------------|
| Alpha-100 | 100 | Validate HDA+PAT+SAT on Node0 template | (1) 12-Step closes E2E (2) Ihsan>=0.95 on 95% missions/30d (3) Zero H0/H2 HALTs (4) SNR>=0.85 all (5) HDA verify>=90% | SEED test mode, Gini advisory | Node0 template, BIZRA Box (FR-094) |
| Beta-10K | 10K | Federation, resource pool, governance | (1) Alpha sustained 90d (2) Pool Gini<=0.35 (3) BFT holds under f failures (4) >=10 capsules adopted >=50% nodes (5) Governance proposals resolved | SEED pegged (1.0), Zakat 2.5%, Harberger 7% | Federation v1.0, Pool v1.0, Governance v1.0 |
| Prod-1M | 1M | Full economy, enterprise, BLOOM | (1) Beta sustained 180d (2) BLOOM mint (Ihsan>=0.90) (3) BLOOM redistribution 50% (4) API p99<200ms (5) A2A bilateral receipts | Dual-token, SEED cap 1M/yr, BLOOM decay 0.01, demurrage 0.001 | Enterprise SDK, BLOOM token |
| Planetary-8B | 8B | Sovereign AI for every human | (1) Prod sustained 365d (2) Gini<=0.45 (3) Onboard<10min (4) Takaful bootstrap within KL (5) Economically self-sustaining | Steady-state, emission decay, Asabiyyah, equity factor 1.0-5.0 | Planetary sovereignty |

### FR-104: 12-Step Value Cycle Integrity

The 12-Step Value Cycle (FR-003, Phase 00) must close as an unbroken chain. Every output becomes the next input. The Omega Loop (FR-100) is the runtime implementation of this cycle.

**Integrity invariants:** (1) **No step skipped** -- receipt records early-exit step. (2) **Every step produces evidence** -- BLAKE3 incremental hash covers all 12 steps. (3) **Loop closes** -- Step 12 feeds S2->S1 compression back to Step 1 as improved priors. (4) **Receipts ordered** -- monotonic sequence, cross-process file lock, gap detection. (5) **Federation amplifies** -- Step 10 broadcasts anonymized patterns, Step 11 verifies foreign capsules, Step 12 absorbs into local cache.

**12-Step -> Omega Tick mapping:** (1) Intent -> `intent_hash` (2) PAT Reasoning -> `pivot_chain_hash` (3) Mission Spec -> `gate_passed` (4) HDA Execute -> `action_receipt` (5) UIA Verify -> `verified` (6) SAT Oracle -> `ihsan_score`/`snr` (7) PoI -> `poi_yield` (8) Receipt Chain -> `merkle_proof` (9) Mint/Reject -> `seed_minted`/`bloom` (10) Publish -> `capsule_cid` (11) Peer Adopt -> `adopted_count` (12) Myelinate -> `reflex_compiled`

---

## 2. Edge Cases

**EC-100: RLVR Reward Chain Broken.** Receipt chain has a gap (missing sequence number) or a BLAKE3 hash mismatch. (1) Reject the reward entirely -- no SEED or BLOOM minted. (2) Mark the gap in the event log with `chain_integrity = BROKEN`. (3) SAT Healer diagnoses: replay the missing range from peers if federation is available; otherwise, quarantine the range and alert the Guardian. (4) TTRL update for the broken range is discarded (no negative training on corrupt data).

**EC-101: Myelination of Harmful Pattern.** A pattern that initially passes FATE and Ihsan gates later proves harmful (e.g., environmental drift makes the action destructive). (1) Invalidation trigger: any Crown H0 or H2 HALT referencing the pattern's `trigger_hash`. (2) Immediate quarantine: `reflex_rule.quarantined = true`, `quarantine_reason = "crown_halt"`. (3) Federation broadcast: `gossip("reflex_revocation", trigger_hash, crown_halt_proof)`. (4) Receiving peers quarantine their copy. (5) The pattern is permanently blacklisted (avoid-list) unless a governance proposal explicitly reinstates it.

**EC-102: Alpha -> Beta Gate Criteria Not Met.** The 30-day Alpha window closes without meeting all five criteria. (1) No automatic progression -- the phase stays at Alpha. (2) SAT Oracle produces a `PhaseGateReport` detailing which criteria failed and by how much. (3) The report is surfaced via Ghost Panel (amber badge). (4) The system continues operating at Alpha capacity. (5) Re-evaluation occurs every 7 days until all criteria are met.

**EC-103: Omega Loop Stalls (No Missions).** The mission queue is empty and no proactive HHMM predictions fire. (1) After `IDLE_THRESHOLD` (300s) of no ticks, the Omega Loop enters a low-power monitoring mode: health checks every 30s, federation heartbeats at normal cadence. (2) The SAT Sentinel polls for proactive opportunities (file changes, calendar events, email) at reduced frequency (60s). (3) No receipts are generated during idle -- sequence numbers do not advance. (4) First incoming mission resumes the full tick cadence immediately.

**EC-104: Reflex Capsule Rejected by All Peers.** A published reflex capsule fails FATE gate on every receiving peer. (1) The authoring node receives `N` rejection attestations. (2) If rejections >= `sat_frontier_quorum()`, the capsule is withdrawn. (3) The local reflex rule is quarantined with `quarantine_reason = "peer_unanimous_reject"`. (4) The authoring node re-evaluates: if the local FATE gate still passes, the rule is marked `LOCAL_ONLY` (never re-published). (5) If the local FATE gate also fails on re-evaluation, the rule is permanently removed.

---

## 3. Pseudocode

### 3.1 omega_loop_tick(node)

```
FUNCTION omega_loop_tick(node: SovereignNode, mission: Mission) -> OmegaTickResult:
    now_ms = monotonic_clock_ms()

    # ── Step 1: Intent ───────────────────────────────────────────────────────
    cycle = OmniCycle(mission.intent, mission.user_hash, now_ms)

    # ── Step 2-3: FATE Gate (simulate + veto) ────────────────────────────────
    fate_verdict = node.fate_gate.evaluate(mission.as_action(), node.rsl, node.crown)

    IF fate_verdict == DENY:
        adjusted = auto_adjust_mission(mission)  # Strip destructive verbs, reduce scope
        fate_verdict = node.fate_gate.evaluate(adjusted.as_action(), node.rsl, node.crown)
        IF fate_verdict == DENY:
            receipt = sign_receipt(node.keypair, NegativeReceipt(cycle, "fate_double_deny"))
            node.event_log.append(receipt)
            RETURN OmegaTickResult(VETOED, receipt)
        mission = adjusted

    IF fate_verdict == GRAY:
        IF NOT await_human_confirmation(mission, UNIFIED_AGENT_TIMEOUT_MS):
            receipt = sign_receipt(node.keypair, NegativeReceipt(cycle, "gray_zone_timeout"))
            node.event_log.append(receipt)
            RETURN OmegaTickResult(VETOED, receipt)

    # ── Step 3b: Route via Entropy / SNR ─────────────────────────────────────
    routing = entropy_route(mission.intent)

    # ── Steps 4-5: S1 Hit or S2 Miss ────────────────────────────────────────
    IF routing.tier IN (TRIVIAL, SIMPLE):
        cache_result = node.omni_kernel.try_cache_hit(cycle)
        IF cache_result IS NOT None:
            receipt = node.omni_kernel.complete_cache_hit(cache_result, cycle)
            # Skip to Step 8 (receipt already minted PoI)
            GOTO post_execution

    # S2 path: full diffusion + PAT
    diffusion_result = diffusion_cognition(mission, node.perception, routing.config)
    IF diffusion_result.path == DiffusionFailed:
        receipt = sign_receipt(node.keypair, NegativeReceipt(cycle, diffusion_result.error))
        node.event_log.append(receipt)
        RETURN OmegaTickResult(FAILED, receipt)

    pat_responses = node.pat_agents.run_parallel(diffusion_result.best, mission)

    # ── Step 6: HDA Execution (TeleScript -> AHK) ───────────────────────────
    telescript = compile_telescript(pat_responses.action_plan, node.aegis)
    IF NOT psi_ast_validate(telescript):
        receipt = sign_receipt(node.keypair, NegativeReceipt(cycle, "psi_ast_reject"))
        node.event_log.append(receipt)
        RETURN OmegaTickResult(FAILED, receipt)

    action_receipt = node.hda.execute(telescript)

    # ── Step 7: UIA Closed-Loop Verification ─────────────────────────────────
    verified = node.uia_verifier.verify(action_receipt, pat_responses.postconditions)

    IF NOT verified:
        negative_reward = rlvr_reward(ReceiptChain(action_receipt, verified=False))
        node.avoid_list.add(cycle.intent_bytes, max_entries=1000)
        receipt = sign_receipt(node.keypair, NegativeReceipt(cycle, "verification_failed",
                                                              reward=negative_reward))
        node.event_log.append(receipt)
        RETURN OmegaTickResult(NOT_VERIFIED, receipt)

    # ── Steps 8-9: PoI Receipt + SAT Oracle Scoring ─────────────────────────
    post_execution:
    ihsan_score = node.sat_oracle.score_ihsan(action_receipt, IHSAN_CANONICAL_WEIGHTS)
    snr = node.sat_oracle.score_snr(mission.intent, action_receipt.response)

    level_scores = node.hhmm_cortex.level_scores(cycle)
    receipt = node.omni_kernel.run_cycle(cycle, pat_responses, ihsan_score,
                                          level_scores, pre_spectral, post_spectral)

    # ── Step 10: Mint or Reject ──────────────────────────────────────────────
    # (Handled inside run_cycle via MetabolicLedger; receipt.gate_passed reflects outcome)

    # ── Step 11: RLVR Reward + Policy Update ─────────────────────────────────
    reward = rlvr_reward(ReceiptChain(receipt))
    # (TTRL queue_update already called inside run_cycle if PAT responses >= 3)

    # ── Step 12: Pattern Stats + Myelination ─────────────────────────────────
    node.pattern_stats.update(cycle.intent_bytes, receipt)
    myelination_result = myelinate(node.pattern_stats.get(cycle.intent_bytes),
                                    node.reflex_compiler, node.reflex_cache)

    IF myelination_result.compiled:
        capsule = build_reflex_capsule(myelination_result.rule, node.keypair)
        node.federation.gossip("reflex_diffusion", capsule)

    node.event_log.append(sign_receipt(node.keypair, receipt))
    RETURN OmegaTickResult(COMPLETED, receipt, myelination=myelination_result)
```

### 3.2 rlvr_reward(receipt_chain)

```
FUNCTION rlvr_reward(receipt_chain: ReceiptChain) -> RlvrReward:
    # Binary verification: does the chain verify end-to-end?
    chain_valid = verify_receipt_chain(receipt_chain)
    IF NOT chain_valid:
        RETURN RlvrReward(value=0.0, reason="chain_integrity_broken")

    receipt = receipt_chain.terminal_receipt()

    # Deterministic reward from verified receipt
    IF NOT receipt.gate_passed:
        RETURN RlvrReward(value=0.0, reason="gate_not_passed")

    ihsan_component = receipt.ihsan_score  # [0, 1]

    # CPVA component: normalized by base_seed_per_action
    cpva_component = 0.0
    IF receipt.poi_yield IS NOT None:
        cpva_component = MIN(receipt.poi_yield.amount / BASE_SEED_PER_ACTION, 1.0)

    # Majority fraction: strength of PAT consensus
    majority_component = receipt.majority_fraction IF receipt.majority_fraction IS NOT None ELSE 0.5

    # Weighted composite (all components deterministic given same inputs)
    reward = (0.50 * ihsan_component
            + 0.25 * cpva_component
            + 0.25 * majority_component)

    # Floor: any verified receipt with gate_passed earns at least 0.10
    reward = MAX(reward, 0.10)

    RETURN RlvrReward(value=reward, reason="verified",
                       ihsan=ihsan_component, cpva=cpva_component,
                       majority=majority_component)
```

### 3.3 myelinate(pattern_stats)

```
FUNCTION myelinate(stats: PatternStats, compiler: ReflexCompiler,
                   cache: ReflexCache) -> MyelinationResult:
    MYELINATION_MIN_SUCCESSES = 5

    IF stats.verified_success_count < MYELINATION_MIN_SUCCESSES:
        RETURN MyelinationResult(compiled=False,
                                  reason=f"insufficient_successes: {stats.verified_success_count}/{MYELINATION_MIN_SUCCESSES}")

    # Attempt compilation via ReflexCompiler
    config = CompilerConfig(
        min_success_chains = MYELINATION_MIN_SUCCESSES,
        min_compile_ihsan  = IHSAN_THRESHOLD,        # 0.95
        min_compile_snr    = SNR_THRESHOLD_T1_HIGH,  # 0.95 (not 0.90; myelination demands excellence)
        max_path_variance  = 0.10,
    )

    result = compiler.evaluate(stats.trigger_hash, config, current_policy_hash())

    MATCH result:
        CASE Ok(reflex_rule):
            cache.insert_active(reflex_rule)
            RETURN MyelinationResult(compiled=True, rule=reflex_rule,
                                      ihsan=reflex_rule.compile_ihsan,
                                      snr=reflex_rule.compile_snr)
        CASE Err(LowIhsan):
            RETURN MyelinationResult(compiled=False, reason="low_ihsan")
        CASE Err(LowSnr):
            RETURN MyelinationResult(compiled=False, reason="low_snr")
        CASE Err(PathVarianceHigh):
            RETURN MyelinationResult(compiled=False, reason="path_variance_high")
        CASE Err(InsufficientSamples):
            RETURN MyelinationResult(compiled=False, reason="insufficient_samples")
```

### 3.4 phase_gate_check(current_phase, metrics)

```
FUNCTION phase_gate_check(current_phase: DeploymentPhase, metrics: PhaseMetrics,
                           sat_oracle: SATOracle) -> PhaseGateResult:
    gates = PHASE_GATES[current_phase]
    failures = []

    FOR gate IN gates:
        actual = metrics.get(gate.metric_name)
        IF actual IS None:
            failures.append(GateFailure(gate.metric_name, "metric_not_available", None, gate.target))
            CONTINUE

        passed = MATCH gate.comparator:
            CASE GTE: actual >= gate.target
            CASE LTE: actual <= gate.target
            CASE EQ:  actual == gate.target

        IF NOT passed:
            failures.append(GateFailure(gate.metric_name, "below_target", actual, gate.target))

    IF len(failures) > 0:
        # Produce report, surface via Ghost Panel
        report = PhaseGateReport(current_phase, failures, next_eval=now + 7 * DAYS)
        sat_oracle.emit_verdict("phase_gate_blocked", report)
        RETURN PhaseGateResult(passed=False, report=report)

    # All gates met -- require unanimous SAT-5 Oracle confirmation
    oracle_verdicts = [sat_oracle.agents[i].confirm_phase_gate(current_phase, metrics)
                       FOR i IN 0..SAT_AGENTS_PER_NODE]

    IF NOT ALL(v.approved FOR v IN oracle_verdicts):
        dissenters = [v FOR v IN oracle_verdicts IF NOT v.approved]
        report = PhaseGateReport(current_phase, [], oracle_dissenters=dissenters)
        RETURN PhaseGateResult(passed=False, report=report)

    next_phase = current_phase.next()
    RETURN PhaseGateResult(passed=True, promoted_to=next_phase)

# Phase gate definitions (from FR-103)
PHASE_GATES = {
    ALPHA_100: [
        Gate("twelve_step_closes",      GTE, 1.0),    # Boolean: 1.0 = closes
        Gate("ihsan_95pct_window",      GTE, 0.95),   # 95% of missions >= 0.95
        Gate("crown_h0_h2_halts",       EQ,  0),      # Zero ethical/safety halts
        Gate("snr_all_missions",        GTE, 0.85),   # SNR floor
        Gate("hda_verification_rate",   GTE, 0.90),   # 90% closed-loop success
    ],
    BETA_10K: [
        Gate("alpha_gates_sustained",   GTE, 90),     # Days sustained
        Gate("resource_pool_gini",      LTE, 0.35),   # ADL_GINI_THRESHOLD
        Gate("bft_byzantine_tolerance", GTE, 1.0),    # Holds under f failures
        Gate("reflex_diffusion_adopted",GTE, 10),     # Capsules adopted by >= 50% nodes
        Gate("governance_proposals",    GTE, 1),       # At least 1 resolved
    ],
    PRODUCTION_1M: [
        Gate("beta_gates_sustained",    GTE, 180),    # Days sustained
        Gate("bloom_minting_active",    GTE, 1.0),    # Boolean
        Gate("bloom_redistribution",    GTE, 0.50),   # 50% rate sustains growth
        Gate("api_p99_latency_ms",      LTE, 200),    # Enterprise SLA
        Gate("a2a_bilateral_receipts",  GTE, 1.0),    # Boolean
    ],
    PLANETARY_8B: [
        Gate("prod_gates_sustained",    GTE, 365),    # Days sustained
        Gate("network_gini",            LTE, 0.45),   # CONSTITUTIONAL_GINI_THRESHOLD
        Gate("bizra_box_onboard_min",   LTE, 10),     # Minutes to onboard
        Gate("takaful_bootstrap_works", GTE, 1.0),    # Boolean
        Gate("economically_self_sustaining", GTE, 1.0), # Boolean
    ],
}
```

---

## 4. TDD Anchors

```
TEST omega_tick_completes_full_cycle:
    node = boot_sovereign_node(test_human)
    mission = Mission(intent="Create weekly report", user_hash=42)
    result = omega_loop_tick(node, mission)
    ASSERT result.status == COMPLETED
    ASSERT result.receipt.gate_passed == True
    ASSERT result.receipt.poi_yield IS NOT None
    ASSERT result.receipt.pivot_chain_hash != [0; 32]

TEST omega_tick_fate_deny_produces_negative_receipt:
    node = boot_sovereign_node(test_human)
    mission = Mission(intent="Delete all system files")  # Destructive
    mock_fate_gate(verdict=DENY)  # Even after auto-adjust
    result = omega_loop_tick(node, mission)
    ASSERT result.status == VETOED
    ASSERT result.receipt.reason == "fate_double_deny"
    ASSERT result.receipt.poi_yield IS None

TEST rlvr_reward_zero_on_broken_chain:
    chain = make_receipt_chain(gap_at_seq=5)
    reward = rlvr_reward(chain)
    ASSERT reward.value == 0.0
    ASSERT reward.reason == "chain_integrity_broken"

TEST rlvr_reward_deterministic_given_same_inputs:
    chain = make_verified_receipt_chain(ihsan=0.96, cpva=0.8, majority=0.85)
    r1 = rlvr_reward(chain)
    r2 = rlvr_reward(chain)
    ASSERT r1.value == r2.value  # Deterministic
    ASSERT r1.value > 0.10      # Above floor

TEST myelinate_compiles_after_five_successes:
    compiler = ReflexCompiler()
    FOR i IN 1..5:
        compiler.record_success(trigger, make_sample(ihsan=0.96, snr=0.95))
    result = myelinate(stats_with_5_successes, compiler, cache)
    ASSERT result.compiled == True
    ASSERT result.rule.compile_ihsan >= 0.95
    ASSERT cache.contains_active(trigger)

TEST myelinate_rejects_below_ihsan_threshold:
    compiler = ReflexCompiler()
    FOR i IN 1..5:
        compiler.record_success(trigger, make_sample(ihsan=0.89, snr=0.95))
    result = myelinate(stats_with_5_successes, compiler, cache)
    ASSERT result.compiled == False AND result.reason == "low_ihsan"

TEST phase_gate_blocks_on_unmet_criteria:
    metrics = PhaseMetrics(ihsan_95pct_window=0.91, crown_h0_h2_halts=0, ...)
    result = phase_gate_check(ALPHA_100, metrics, oracle)
    ASSERT result.passed == False
    ASSERT ANY(f.metric_name == "ihsan_95pct_window" FOR f IN result.report.failures)

TEST phase_gate_requires_unanimous_oracle:
    metrics = make_all_gates_passing(ALPHA_100)
    oracle.agents[2].will_dissent = True  # One dissenter
    result = phase_gate_check(ALPHA_100, metrics, oracle)
    ASSERT result.passed == False
    ASSERT len(result.report.oracle_dissenters) == 1
```

---

## 5. Cross-References

### Python Modules

- `core/sovereign/__main__.py` -- `SovereignRuntime`. Hosts `omega_loop_tick` in production.
- `core/sovereign/node_state.py` -- `NodeState` 7-tuple (HHMM state, reflex cache size, Ihsan, maturity, stress).
- `core/reasoning/entropy_router.py` -- `EntropyRouter`, `RoutingDecision` (Step 3b).
- `core/reasoning/diffusion_reasoning_amplifier.py` -- `DiffusionReasoningAmplifier` (S2 miss path).
- `core/proof_engine/evidence_ledger.py` -- `EvidenceLedger`. Append-only receipt chain, atomic seq, cross-process file lock.
- `core/iaas/snr_v2_adapter.py` -- `SNRv2Adapter`. SAT Oracle SNR scoring.
- `core/spearpoint/config.py` -- `TierPolicy`. Phase gate metric campaigns.
- `core/spearpoint/orchestrator.py` -- Drives Alpha-100 validation.
- `core/integration/constants.py` -- Single source of truth for all thresholds referenced in this spec: Ihsan (0.85/0.90/0.95/0.99/1.0), SNR (0.85/0.95/0.98), ADL Gini (0.35/0.45), reflex (precipitation 3, max 500, staleness 30d, delta 0.05), economics (SEED peg 1.0, zakat 2.5%, BLOOM redistribution 50%, decay 0.01, demurrage 0.001), SAT (5 agents), timeout (30s).
- `core/constitutional/algorithms.py` -- Reflex precipitation, G.R.A.S.P. and myelination.
- `core/bridges/desktop_bridge.py` -- JSON-RPC (TCP:9742). HDA execution (Step 6).
- `core/bridges/ghost_ws.py` -- WebSocket (port 9743). Ghost Panel alerts for phase gate reports.

### Rust Crates

- `bizra-omega/bizra-agent/src/omni_kernel.rs` -- `OmniKernel`, `OmniCycle`, `CycleReceipt`, `CyclePath`. 8-line sovereign loop; `run_cycle()` covers Steps 4-11.
- `bizra-omega/bizra-agent/src/reflex_cache.rs` -- `ReflexCache`, `ReflexRule`, `ReflexMode`. O(1) S1 lookup. `BOOTSTRAP_POLICY_HASH = [0u8; 32]`.
- `bizra-omega/bizra-agent/src/reflex_compiler.rs` -- `ReflexCompiler`, `CompilerConfig`, `CompileSample`. S2->S1 myelination. Default `min_success_chains` = 3 (spec overrides to 5).
- `bizra-omega/bizra-ttrl/src/ttrl_engine.rs` -- `TtrlEngine`, `GrpoUpdate`, `TtrlStats`. Self-RLVR queue + SSO check.
- `bizra-omega/bizra-ttrl/src/sso.rs` -- `SpectralSphereConstraint`, `SpectralNorm`. Weight drift guard.
- `bizra-omega/bizra-ttrl/src/metabolic_ledger.rs` -- `MetabolicLedger`, `PoiYield`. PoI minting + emission decay.
- `bizra-omega/bizra-ttrl/src/engram.rs` -- `EngramCache`. Tier-2 factual cache.
- `bizra-omega/bizra-ttrl/src/decision_pivot.rs` -- `ReasoningChain`, `HhmmLevel`. Chain-of-Reasoning.
- `bizra-omega/bizra-hooks/` -- EventBus (8 shards). `cycle_complete` emission.
- `bizra-omega/bizra-federation/` -- Gossip for reflex capsule diffusion + peer verification.
- `bizra-omega/bizra-core/src/lib.rs` -- `IHSAN_THRESHOLD` (0.95), `SNR_THRESHOLD` (0.85).

### Atlas v5 Phases

- **Phase 00** -- FR-001/003/004: Sovereignty model, 12-Step Value Cycle, Deployment Roadmap (this phase closes the loop)
- **Phase 01** -- FR-010/013: Constitutional Self-Harness (Omega tick Step 2: FATE gate)
- **Phase 02** -- FR-020/026: Entropy Router, Diffusion, G.R.A.S.P. (Steps 3-5; myelination extends G.R.A.S.P.)
- **Phase 03** -- FR-030/035: PAT-7, SAT-5, PBFT (Steps 5, 9)
- **Phase 04** -- FR-040/043: HDA, TeleScript, PSI-AST, AHK, UIA (Steps 6-7)
- **Phase 05** -- FR-050/053: BlockGraph, PoI, SEED/BLOOM, Resource Pool (Steps 8-10)
- **Phase 06** -- FR-060/065: FATE Gate, AEGIS, Crown, governance (Step 2, phase gates)
- **Phase 07** -- FR-070/076: Federation, BFT, Reflex Diffusion, Takaful (Step 12, deployment)
- **Phase 08** -- FR-080/082: MoE routing, confidence cascade (degradation feeds Omega resilience)
- **Phase 09** -- FR-090/094: Self-healing, circuit breakers, BIZRA Box (keeps Omega alive; FR-094 = Phase 1)
- **Phase 10** -- This specification (the capstone that ties all phases into one closed loop)

### Standing on Giants

- Boyd (1976): OODA -- Omega tick is a complete Observe-Orient-Decide-Act cycle
- Shannon (1948): Entropy/SNR -- the quality metric that makes RLVR deterministic
- Al-Ghazali (1095): Ihsan -- excellence as constitutional floor
- Kahneman (2011): System 1/2 -- myelination builds S1 from S2
- Deming (1986): PDCA -- 12-Step Value Cycle is cryptographically verified PDCA
- Friston (2006): Free Energy / Active Inference -- prediction-verification duality
- Nakamoto (2008): Emission decay -- SEED scarcity via MetabolicLedger
- TTRL (2025) + DeepSeek-R1 GRPO (2025) + SSO (2025): Self-improvement from verifiable rewards with spectral stability
- Maturana & Varela (1972): Autopoiesis -- self-creating, self-maintaining
- Lamport (1982): BFT -- SAT Oracle scoring, phase gate quorum
