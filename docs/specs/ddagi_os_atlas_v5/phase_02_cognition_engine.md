# Phase 02 — Cognition Engine: Diffusion, G.R.A.S.P., Ihsan Loop

> Source: Atlas v5.0 — Diagrams D3 (Test-Time Diffusion Cognition), D11 (System 2-1 Transition / G.R.A.S.P.), D23 (Ihsan Feedback Loop)
> Status: SPECIFICATION SEALED | SNR: 0.95

---

## 1. Functional Requirements

### FR-020: Entropy Router

The Entropy Router measures Shannon entropy H(task) and routes to the appropriate
cognition tier. Combines character-distribution entropy with structural complexity
signals (sub-question markers, multi-domain patterns, question density) into a
composite score on [0.0, 1.0]. Aligned with `core/reasoning/entropy_router.py`.

| Score Range  | Tier     | System          | GoT | Quorum | Latency  | SNR Floor |
|--------------|----------|-----------------|-----|--------|----------|-----------|
| [0.00, 0.30) | TRIVIAL  | S1 Reflexive    | No  | 0      | 200 ms   | 0.85      |
| [0.30, 0.50) | SIMPLE   | S1 Reflexive    | No  | 0      | 500 ms   | 0.85      |
| [0.50, 0.70) | MODERATE | S1.5 Moderate   | Yes | 3      | 5000 ms  | 0.95      |
| [0.70, 0.85) | COMPLEX  | S2 Deliberative | Yes | 5      | 30000 ms | 0.95      |
| [0.85, 1.00] | FRONTIER | S2 Deliberative | Yes | 2f+1   | 30000 ms | 0.98      |

Canonical threshold theta = 4.5 bits raw Shannon ~ 0.68 normalized (4.5/6.6 max
printable ASCII). Below theta: System-1. Above: System-2. MODERATE is the transition
zone (GoT engaged, no full orchestrator).

### FR-021: Test-Time Diffusion Cognition

Multi-cycle explore-denoise-amplify pipeline. Mirrors physical diffusion: start noisy,
progressively denoise through PBFT voting, converge on "Aha moment".

**Cycle 0 (Seeding):** GoT generates N >= 9 candidates. Initial SNR ~ 0.62, sigma ~ 0.41.

**Backtrack Gate:** If sigma > 0.30, prune nodes below (median - 1 stddev).

**Cycle 1 (Denoise):** PBFT voting (quorum per tier). Post-vote SNR ~ 0.79, sigma ~ 0.27.
Nodes failing consensus are pruned.

**Cycle 2 (Amplify):** DiffusionReasoningAmplifier injects HMM context. Second PBFT vote.
Post-amplification SNR ~ 0.88, sigma ~ 0.18.

**Aha Detector:** Locks when SNR >= `GOT_CONVERGENCE_SNR` (0.90) AND sigma < 0.20.
If not reached after max_iterations (5), select highest-SNR node with `aha=false`.

Constitutional constraint: final output must pass Ihsan Wall (0.95) before execution.

### FR-022: Atomic Thought Rewards

Every reasoning step, PBFT vote, and execution action receives an individual reward:
- **Step reward:** SNR normalized against task entropy.
- **Vote reward:** 1.0 (aligned with consensus), 0.0 (dissenting against correct majority),
  0.5 (dissenting against incorrect majority).
- **Action reward:** Postcondition verification score.

Mean target: >= 0.85. Rewards accumulate in TTRL queue for self-improvement gradients.

### FR-023: G.R.A.S.P. Skill Learning

System-2 deliberative reasoning compiles into System-1 reflexes via five stages:

1. **Generate** — Extract intent, GoT subgraph, action sequence, outcome from receipt.
2. **Reflect** — Score atomic steps (FR-022). Compute 8-dim Ihsan (`IHSAN_CANONICAL_WEIGHTS`).
3. **Absorb** — Extract pattern: (intent_class, context_hash) -> (action_template,
   expected_outcome). Privacy >= `ABSTRACT_OK` required.
4. **Synthesize** — Skill hash: `BLAKE3("grasp/skill/v1:" + canonical_pattern)`.
   Register in AscentTracker.
5. **Promote** — Cache in ReflexCache as `ReflexMode::Shadow`. After
   `REFLEX_PRECIPITATION_HITS` (3) consecutive shadow matches with Ihsan >= 0.90,
   precipitate to `ReflexMode::Active`.

**Residual Monitor:** 10-execution rolling window. Evict if success < 90%. Invalidate
if Ihsan drift > `REFLEX_INVALIDATION_DELTA` (0.05) or age > `REFLEX_STALENESS_DAYS` (30).

### FR-024: System-2 to System-1 Compression

| Metric          | S2 (First)  | S1 (Cached) | Gain    |
|-----------------|-------------|-------------|---------|
| Latency         | 4.65 s      | 0.75 s      | 6.2x    |
| Nodes explored  | 9+ (GoT)    | 1 (hash)    | O(1)    |
| PBFT rounds     | 2           | 0           | Elim.   |
| Security overhead | 120 ms    | 120 ms      | Constant|

FATE gate runs on every action regardless of tier. ReflexCache capacity:
`REFLEX_MAX_ENTRIES` (500), quality-weighted eviction (lowest Ihsan first).

### FR-025: Ihsan Feedback Loop

Captures implicit signals, converts to reward, updates diffusion parameters.

| User Action | Weight | Interpretation             |
|-------------|--------|----------------------------|
| Hover >2s   | +0.30  | Mild interest              |
| Copy        | +0.80  | Reusing output             |
| Edit        | +1.00  | Building on output         |
| Dismiss     | -0.50  | Not useful                 |
| Regenerate  | -0.80  | Unsatisfactory             |

**Reward:** Sigmoid normalization: `reward = 1/(1 + exp(-3 * aggregate))` over
60-second window. **Param update:** positive (>= 0.70) tightens sigma 5%; negative
(< 0.30) adds +1 GoT hypotheses/depth. **S2-S1 trigger:** 3 consecutive reward >= 0.90
fires G.R.A.S.P. **Federation:** anonymize SHAREABLE rewards, gossip broadcast, Bayesian
prior update. New nodes inherit via Takaful Bootstrap (10 min observation).

### FR-026: Ihsan Excellence Score

```
ihsan_excellence = task_quality * 0.50 + user_satisfaction * 0.30 + efficiency * 0.20
```

Where efficiency = `CLAMP(1 - actual_latency/budget_latency, 0, 1)`. Must meet:
0.95 (prod), 0.90 (CI), 0.99 (strict), 1.0 (runtime/Z3-proven).

---

## 2. Edge Cases

**EC-020: Empty GoT Graph** — Fail closed. Return error receipt
`cycle_path="DIFFUSION_FAILED"`, escalate to Sentinel. No output from zero candidates.

**EC-021: All PBFT Votes Disagree** — Select highest individual SNR, tag
`consensus="NONE"`, reduce confidence by 0.20. If Ihsan falls below threshold, veto.

**EC-022: Aha Not Reached After Max Iterations** — Select best-SNR node, tag
`aha_reached=false`. If best SNR < 0.85, reject entirely.

**EC-023: Skill Hash Collision** — Append 4-byte nonce, re-hash (up to 3 attempts).
After 3 failures, Sentinel alert + Conductor escalation.

**EC-024: Zero User Signals** — Neutral reward 0.50. After 10+ silent outputs,
shift weights: w_q=0.60, w_s=0.10, w_e=0.30 (autonomous mode).

---

## 3. Pseudocode

### 3.1 entropy_route(task)

```
FUNCTION entropy_route(task_text, context = {}):
    char_freq = frequency_count(task_text.lower())
    raw_H = -SUM(p * log2(p) FOR p IN normalized(char_freq))
    entropy_norm = raw_H / log2(len(char_freq)) IF len(char_freq) > 1 ELSE 0

    score = (0.25 * entropy_norm
           + 0.15 * MIN(len(words) / 80, 1)
           + 0.20 * MIN(sub_question_hits / 3, 1)
           + 0.15 * MIN(domain_hits / 2, 1)
           + 0.10 * MIN(q_marks / 3, 1)
           + 0.15 * context.get("complexity_hint", 0))
    score = CLAMP(score, 0, 1)

    tier = TRIVIAL   IF score < 0.30
           SIMPLE    IF score < 0.50
           MODERATE  IF score < 0.70
           COMPLEX   IF score < 0.85
           FRONTIER  OTHERWISE

    RETURN RoutingDecision(tier, TIER_CONFIG[tier])
```

### 3.2 diffusion_cognition(task, perception_data)

```
FUNCTION diffusion_cognition(task, perception_data, config):
    # System-1 fast path
    cached = reflex_cache.get_active(Active, BLAKE3("omni/intent/v1:" + task.text),
                                     Some(policy_hash), now_ms())
    IF cached: RETURN CycleResult(ReflexHit, cached)

    # Cycle 0: Seed N >= 9 candidates via GoT
    candidates = [got.generate_hypothesis(task, perception_data) FOR _ IN 1..N]
    FOR c IN candidates: c.snr = snr_engine.calculate(task.text, c.content)
    IF len(candidates) == 0: RETURN CycleResult(DiffusionFailed, "empty_got_graph")
    sigma = STDDEV(c.snr FOR c IN candidates)

    FOR cycle IN 1..config.max_iterations:
        # Backtrack: prune low-quality if sigma > 0.30
        IF sigma > 0.30:
            cutoff = MEDIAN(snrs) - STDDEV(snrs)
            candidates = [c FOR c IN candidates IF c.snr >= cutoff]
        IF len(candidates) == 0: RETURN CycleResult(DiffusionFailed, "all_pruned")

        # PBFT vote + prune non-consensus
        FOR c IN candidates:
            c.consensus = PBFT_aggregate([v.evaluate(c) FOR v IN validators])
        candidates = [c FOR c IN candidates IF c.consensus >= 0.5]
        IF len(candidates) == 0:
            best = MAX(all_ever, key=snr); best.confidence -= 0.20
            RETURN CycleResult(BestEffort, best, consensus="NONE")

        # Amplify: HMM context injection + re-score
        FOR c IN candidates:
            c.content = amplifier.augment(c.content, hmm_prediction)
            c.snr = snr_engine.calculate(task.text, c.content)
        sigma = STDDEV(c.snr FOR c IN candidates)
        agg_snr = MEAN(c.snr FOR c IN candidates)

        # Aha Moment
        IF agg_snr >= 0.90 AND sigma < 0.20:
            RETURN CycleResult(AhaMoment, MAX(candidates, key=snr), aha=True)

    best = MAX(candidates, key=snr)
    IF best.snr < 0.85: RETURN CycleResult(DiffusionFailed, "snr_below_min")
    RETURN CycleResult(BestEffort, best, aha=False)
```

### 3.3 grasp_learn(execution_receipt)

```
FUNCTION grasp_learn(receipt):
    IF NOT receipt.outcome.success: RETURN GraspResult(False, "execution_failed")

    # Reflect: score every atomic step
    rewards = ([compute_reward(s) FOR s IN receipt.got_nodes]
             + [vote_reward(v, receipt.consensus) FOR v IN receipt.pbft_votes]
             + [a.postcondition_score FOR a IN receipt.actions])
    IF MEAN(rewards) < 0.85:
        ttrl.queue_update(receipt.id, rewards, "sub_threshold")
        RETURN GraspResult(False, "mean_reward_below_target")

    ihsan = DOT(score_ihsan_dims(receipt), IHSAN_CANONICAL_WEIGHTS)
    IF ihsan < 0.95: RETURN GraspResult(False, "ihsan_below_threshold")

    # Absorb: extract reusable pattern
    pattern = CanonicalPattern(receipt.intent_class, BLAKE3(context),
                               receipt.action_template, receipt.outcome.summary)
    IF classify_privacy(receipt) == "LOCAL_ONLY":
        RETURN GraspResult(False, "privacy_local_only")

    # Synthesize: skill hash with collision handling
    skill_hash = BLAKE3("grasp/skill/v1:" + serialize(pattern))
    IF collision_detected(skill_hash, pattern):
        skill_hash = resolve_collision(skill_hash, pattern, max_attempts=3)
        IF skill_hash IS None: RETURN GraspResult(False, "hash_collision")

    ascent_tracker.register(skill_hash, pattern, ihsan)

    # Promote: shadow mode in ReflexCache
    reflex_cache.insert(ReflexRule(
        TriggerHash(BLAKE3(intent + ":" + ctx_hash)),
        policy_hash=current_policy_hash(),
        action=pattern.action_template,
        mode=Shadow, ihsan_score=ihsan))

    ttrl.queue_update(receipt.id, rewards, "grasp_promoted")
    RETURN GraspResult(True, skill_hash, Shadow, ihsan)
```

### 3.4 ihsan_feedback_loop(user_signals, diffusion_params)

```
FUNCTION ihsan_feedback_loop(user_signals, diffusion_params, receipt):
    WEIGHTS = {hover: 0.30, copy: 0.80, edit: 1.00, dismiss: -0.50, regen: -0.80}

    IF len(user_signals) == 0:
        reward = 0.50  # EC-024: neutral default
    ELSE:
        agg = SUM(WEIGHTS[s.action] FOR s IN signals IF s.time <= delivery + 60s)
        reward = 1.0 / (1.0 + exp(-3.0 * agg))  # Sigmoid to [0,1]

    receipt.ihsan_feedback_reward = reward

    # Update diffusion params per intent class
    IF reward >= 0.70: params[intent].sigma_cap *= 0.95       # Tighten
    ELIF reward < 0.30:                                        # Widen
        params[intent].got_hypotheses = MIN(+1, GOT_MAX_HYPOTHESES)
        params[intent].got_depth = MIN(+1, GOT_MAX_DEPTH)

    # S2-S1 compression: 3 consecutive rewards >= 0.90 triggers G.R.A.S.P.
    recent = receipt_store.get_recent(intent, 3)
    IF len(recent) == 3 AND ALL(r.reward >= 0.90 FOR r IN recent):
        grasp_learn(recent[-1])

    # Ihsan Excellence Score (FR-026)
    efficiency = CLAMP(1.0 - receipt.latency_ms / budget_latency, 0, 1)
    ihsan_exc = receipt.snr * 0.50 + reward * 0.30 + efficiency * 0.20
    receipt.ihsan_excellence = ihsan_exc

    # Federate shareable rewards
    IF receipt.privacy_class IN ("ABSTRACT_OK", "SHAREABLE"):
        federation.gossip("reflex_diffusion", anonymize(intent, reward, ctx_hash))

    RETURN ihsan_exc
```

---

## 4. TDD Anchors

```
TEST entropy_route_trivial_below_threshold:
    decision = EntropyRouter().route("What is 2+2?")
    ASSERT decision.system == "S1_REFLEXIVE"
    ASSERT decision.use_got == False
    ASSERT decision.quorum_size == 0

TEST entropy_route_complex_above_threshold:
    query = "Compare and contrast the trade-offs between microservices and "
            "monolithic architectures from a performance and org perspective"
    decision = EntropyRouter().route(query)
    ASSERT decision.system == "S2_DELIBERATIVE"
    ASSERT decision.use_got == True AND decision.quorum_size >= 5

TEST diffusion_reflex_hit_bypasses_pipeline:
    cache.insert(test_rule(mode=Active))
    result = diffusion_cognition(cached_task, data, config)
    ASSERT result.path == "ReflexHit" AND result.latency_ms < 10

TEST diffusion_fails_closed_on_empty_graph:
    config = DiffusionConfig(backend=MockBackend(empty=True))
    result = diffusion_cognition(task, data, config)
    ASSERT result.path == "DiffusionFailed" AND result.error == "empty_got_graph"

TEST diffusion_aha_locks_on_convergence:
    result = diffusion_cognition(convergent_task, data, config)
    ASSERT result.aha == True AND result.snr >= 0.90

TEST grasp_promotes_to_shadow:
    receipt = make_receipt(success=True, mean_reward=0.92, ihsan=0.96)
    result = grasp_learn(receipt)
    ASSERT result.promoted == True AND result.mode == Shadow

TEST grasp_rejects_below_ihsan:
    receipt = make_receipt(success=True, mean_reward=0.92, ihsan=0.89)
    result = grasp_learn(receipt)
    ASSERT result.promoted == False AND result.reason == "ihsan_below_threshold"

TEST feedback_negative_widens_exploration:
    signals = [Signal("dismiss"), Signal("regenerate")]
    ihsan_feedback_loop(signals, params, receipt)
    ASSERT params[intent].got_hypotheses == original + 1

TEST feedback_three_positives_trigger_grasp:
    FOR i IN 1..3: ihsan_feedback_loop([Signal("copy"), Signal("edit")], p, r[i])
    ASSERT reflex_cache.contains_shadow(r[2].intent_class)

TEST residual_monitor_quarantines_degraded_skill:
    skill.rolling_window = [True]*8 + [False]*2  # 80% < 90% threshold
    residual_monitor.evaluate(skill)
    ASSERT skill.mode == Shadow AND skill.quarantine == "RevalidationFailed"
```

---

## 5. Cross-References

### Python Modules
- `core/reasoning/entropy_router.py` — `EntropyRouter`, `QueryComplexity`, `RoutingDecision`
- `core/reasoning/diffusion_reasoning_amplifier.py` — `DiffusionReasoningAmplifier`
- `core/reasoning/graph_core.py` — `GraphOfThoughts`
- `core/iaas/snr_v2_adapter.py` — `SNRv2Adapter` (protocol-conforming SNR)
- `core/integration/constants.py` — All thresholds (Ihsan, SNR, GoT, Reflex)
- `core/uers/entropy.py` — `EntropyCalculator` (5-dim manifold)

### Rust Crates
- `bizra-omega/bizra-agent/src/omni_kernel.rs` — 8-line sovereign loop
- `bizra-omega/bizra-agent/src/reflex_cache.rs` — `ReflexCache`, `ReflexMode`, `ReflexRule`
- `bizra-omega/bizra-agent/src/reflex_compiler.rs` — S2-to-S1 compilation
- `bizra-omega/bizra-ttrl/` — `TtrlEngine`, `EngramCache`, `MetabolicLedger`, `SSO`
- `bizra-omega/bizra-hooks/` — Event bus (8 shards, FNV-1a)

### Atlas v5 Phases
- Phase 00 — System Overview | Phase 01 — Sovereign Node
- **Phase 02** — This specification
- Phase 03 — Agent Orchestration (PAT-7/SAT-5/PBFT)
- Phase 04 — HDA Execution (desktop automation)
- Phase 10 — Omega Loop (self-improvement, federation convergence)

### Standing on Giants
- Kahneman (2011): System 1/2 | Shannon (1948): Entropy
- Boyd (1976): OODA | Besta et al. (2024): Graph of Thoughts
- Al-Ghazali (1095): Ihsan | Friston (2006): Free Energy Principle
- Castro & Liskov (1999): PBFT
