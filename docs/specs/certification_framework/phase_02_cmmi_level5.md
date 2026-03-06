# Phase 02 — CMMI Level 5: Optimizing Process Maturity

> Source: BIZRA Quality Standards & Certification Framework
> Standard: CMMI v2.0 — Maturity Level 5 (Optimizing)
> Status: SPECIFICATION SEALED | SNR: 0.93

---

CMMI Level 5 demands continuous quantitative improvement through feedback loops and causal analysis. BIZRA achieves this natively: every action produces a quantified ActionReceipt, every failure triggers automated root-cause analysis via PAT-7, and the Self-RLVR Omega Loop compiles high-quality paths into O(1) reflexes. The system is its own Six Sigma engine.

---

## 1. Functional Requirements

### FR-C20: Process Performance Baselines (PPB)

**CMMI Practice Areas:** QPM, MPM.

Every ActionReceipt carries four fixed-point scores combined via A1 Ihsan (W_INTENT=0.25, W_EFFICIENCY=0.25, W_IMPACT=0.30, W_REPRODUCIBILITY=0.20). PPBs are computed at three granularities:

| Granularity | Window | Primary Metric | Secondary Metric |
|:---|:---|:---|:---|
| Per-Action | Last 50 receipts of same type | Ihsan mean + stddev | SNR of action class |
| Per-Agent | Last 200 receipts from PAT-7 agent | Agent Ihsan trend | Myelination ratio |
| Per-Node | Last 1000 receipts across all agents | Node-wide Ihsan | Gini coefficient |

**Quality Gates:**

| Metric | Threshold | Interpretation |
|:---|:---|:---|
| Ihsan mean >= IHSAN_PRODUCTION (0.95) | Hard gate | Process is capable |
| Ihsan stddev <= 0.03 | Warning | Process is stable |
| SNR >= UNIFIED_SNR_THRESHOLD (0.85) | Hard gate | Signal quality sufficient |
| SNR >= SNR_T0 (0.98) | Elite gate | Exemplary process |

**Evidence:** Rolling PPB report with SPC charts. All values fixed-point (FP_PRECISION = 1,000,000).

---

### FR-C21: Process Performance Models (PPM)

**CMMI Practice Areas:** QPM.

PPMs predict action quality from three input features before execution:
1. **Task Complexity** -- from Planner (P1), estimated deliberation steps.
2. **Agent Myelination Level** -- ratio of available reflexes to task steps.
3. **Context Freshness** -- time since last semantically similar action.

```
predicted_ihsan = BASELINE_IHSAN
    + BETA_COMPLEXITY * normalized_complexity
    + BETA_MYELINATION * myelination_level
    + BETA_FRESHNESS * context_freshness_score
```

Coefficients calibrated via OLS over last 500 receipts. If |predicted - actual| > 0.10 for >5% of actions, the Optimizer (P7) triggers recalibration.

**Evidence:** PPM coefficients, prediction accuracy histogram, R-squared weekly.

---

### FR-C22: Self-RLVR Optimization Loop (Omega Loop)

**CMMI Practice Areas:** OPM, CAR, PCM.

The Omega Loop replaces human-led process improvement with autonomous self-optimization:

1. **Detect:** After execution, P4 checks if `ihsan >= 0.95`, `snr >= SNR_T1 (0.95)`, `latency <= p50`, `reproducibility >= 0.90`.
2. **Verify:** Reward is the constitutional Ihsan score itself -- deterministic, unhackable.
3. **Compile:** Action path becomes a Reflex: `pattern_hash = BLAKE3(trigger || context)`, `confidence = ihsan`.
4. **Install:** Reflex enters ReflexCache for O(1) future lookups.
5. **Decay:** Reflexes dropping below IHSAN_CI (0.90) on re-evaluation are demoted to System-2.

No human gate required. The system self-optimizes within constitutional bounds.

**Evidence:** Omega Loop activation log, Reflex compilation/pruning events, myelination trend.

---

### FR-C23: Myelination Ratio Tracking

**CMMI Practice Areas:** MPM, OPM.

`Myelination Ratio = S1_cache_hits / (S1_cache_hits + S2_invocations)`. Target: >= 0.90.

| Range | CMMI Equivalent | State |
|:---|:---|:---|
| 0.00-0.30 | Level 3 | Learning phase |
| 0.30-0.60 | Level 4 | Reflexes emerging |
| 0.60-0.90 | Level 5 threshold | Predominantly reflexive |
| 0.90-1.00 | Level 5 optimizing | Steady-state target |

**Trend Detection:** Declining ratio without new task types triggers Causal Analysis. Declining ratio WITH new task types is annotated "learning phase" (no alarm).

**Per-Agent Targets:** P3 Executor (0.95), P6 Chronicler (0.90), P4 Evaluator (0.85), P5 Ethicist (0.80), P7 Optimizer (0.75), P1 Planner (0.70), P2 Researcher (0.60).

**Evidence:** Myelination time series per-agent/per-node, trend slope, S1 latency histogram.

---

### FR-C24: Causal Analysis Pipeline (CAR)

**CMMI Practice Areas:** CAR.

When an action is rejected or a process anomaly detected, a two-stage pipeline activates:

**Stage 1 -- P4 Evaluator:** Identifies deficiency dimension and classifies root cause (INPUT_QUALITY, MODEL_DRIFT, REFLEX_STALENESS, RESOURCE_CONTENTION, or NOVEL_TASK).
**Stage 2 -- P5 Ethicist:** Assigns `ethical_risk_level` (LOW/MEDIUM/HIGH/CRITICAL). CRITICAL triggers Shura.
**Resolution:** Crown hierarchy (H0>H1>H2) picks higher severity. Report feeds back into Planner constraints, Reflex demotions, semantic memory, and PPM recalibration. If P4/P5 disagree >15% over 100 receipts, P7 triggers calibration on 20 shared receipts.

**Evidence:** CausalAnalysisReport log, defect category distribution, recurrence reduction rates.

---

### FR-C25: Organizational Performance Management (OPM)

**CMMI Practice Areas:** OPM, GOV.

"Organizational" maps to "network-wide" via federation:

1. **Anonymized PPB Sharing:** Nodes publish aggregate summaries to federation gossip. Individual receipts never leave.
2. **Network Baselines:** Weighted average of node PPBs, stratified by task domain.
3. **Shura Proposals:** Nodes >2 sigma from domain peers get advisory proposals. BLOOM-weighted voting (IHSAN_BLOOM_ELIGIBILITY >= 0.90) governs policy changes.
4. **Reflex Templates:** Reflexes with confidence >= IHSAN_STRICT (0.99) offered as templates. Receiving nodes evaluate against local PPB.

**Evidence:** Federation PPB summary, Shura proposal log, BLOOM vote records, template adoption rate.

---

### FR-C26: Continuous Improvement Evidence Trail

**CMMI Practice Areas:** II, OPM, CAR.

Immutable trail: (1) Improvement Event Log -- all Omega Loop activations, Reflex compilations/prunings, PPM recalibrations, CAR resolutions; append-only with Merkle chain (A6). (2) Quality Trajectory -- Ihsan trend; negative slope >30 days triggers governance alert. (3) Myelination Growth -- target 0.90 within 90 days. (4) Defect Recurrence -- target 50% reduction within 30 days. (5) Cumulative Score: `0.30*myelination_growth + 0.25*recurrence_reduction + 0.25*ihsan_slope + 0.20*ppb_stability`.

**Evidence:** Improvement Event Log with Merkle proofs, trajectory charts, weekly score.

---

## 2. Edge Cases

**EC-C20: Myelination Ceiling.** System reaches 0.95+ but residual S2 invocations are genuinely novel. Resolution: P7 classifies residual S2 via task diversity entropy. Per-agent targets are adjusted; CMMI pack documents justification.

**EC-C21: Reflex Quality Degradation.** Requirements shift after Reflex compilation. Resolution: Re-evaluation every 100 invocations or 7 days. If intent/impact drops below IHSAN_CI (0.90), Reflex is demoted (confidence halved, S2 for 10 invocations). No recovery triggers pruning + CAR event.

**EC-C22: Cross-Node Aggregation Drift.** Different task distributions skew global PPB. Resolution: Federation PPB stratified by task domain (action-type hash prefix). Deviation measured against domain peer group. Shura proposals use domain-stratified z-scores.

**EC-C23: P4/P5 Disagreement.** P4=LOW, P5=HIGH on same receipt. Resolution: Crown hierarchy (H0>H1>H2) picks higher severity. Both recorded. If disagreement >15% over 100 receipts, P7 triggers calibration session on 20 shared receipts.

**EC-C24: RLVR Reward Hacking.** Degenerate path games Ihsan (no-op with perfect efficiency). Resolution: impact_score (W=0.30) measured against user goals -- no-op scores 0.0. P5 reviews all compilation candidates. Flagged patterns denied + 30-day expiring deny-list.

**EC-C25: Cold Start.** Node has <50 receipts. Resolution: Constitutional Floor Mode -- hard thresholds only, PPM disabled, no myelination, excluded from federation aggregation until 50 receipts.

---

## 3. Pseudocode

### 3.1 compute_process_baseline

```
FUNCTION compute_process_baseline(receipts: list[ActionReceipt], window_size: int) -> PPB:
    """Rolling PPB from receipts. Ref: core/constitutional/algorithms.py (A1), constants.py"""

    IF len(receipts) < 50:
        RETURN PPB(status="COLD_START", ihsan_mean=None, note="Constitutional floor mode")

    window = receipts[-window_size:]

    ihsan_scores = []
    FOR receipt IN window:
        ihsan = fp_mul(fp(0.25), receipt.intent_score)
               + fp_mul(fp(0.25), receipt.efficiency_score)
               + fp_mul(fp(0.30), receipt.impact_score)
               + fp_mul(fp(0.20), receipt.reproducibility_score)
        ihsan_scores.append(ihsan)

    snr_scores = [r.snr FOR r IN window IF r.snr IS NOT None]

    ihsan_mean = fp_mean(ihsan_scores)
    ihsan_stddev = fp_stddev(ihsan_scores, ihsan_mean)
    ihsan_slope = fp_linear_slope(ihsan_scores)
    snr_mean = fp_mean(snr_scores) IF snr_scores ELSE fp(0)

    RETURN PPB(
        status="ACTIVE",
        ihsan_mean=ihsan_mean, ihsan_stddev=ihsan_stddev, ihsan_slope=ihsan_slope,
        snr_mean=snr_mean, sample_size=len(window),
        is_capable=fp_float(ihsan_mean) >= IHSAN_PRODUCTION,
        is_stable=fp_float(ihsan_stddev) <= 0.03,
        is_improving=fp_float(ihsan_slope) >= 0.0,
        snr_sufficient=fp_float(snr_mean) >= UNIFIED_SNR_THRESHOLD,
        timestamp=now_ms()
    )
```

### 3.2 rlvr_optimize

```
FUNCTION rlvr_optimize(receipt: ActionReceipt, reflex_cache: ReflexCache) -> OptimizeResult:
    """Omega Loop: detect high-quality path, compile to reflex.
    Ref: core/constitutional/types.py (Reflex), bizra-agent/src/omni_kernel.rs"""

    ihsan = compute_ihsan_from_receipt(receipt)

    # Gate checks — all must pass for myelination
    IF fp_float(ihsan) < IHSAN_PRODUCTION:
        RETURN OptimizeResult(action="SKIP", reason="Ihsan below 0.95")
    IF receipt.snr IS NOT None AND fp_float(receipt.snr) < SNR_T1:
        RETURN OptimizeResult(action="SKIP", reason="SNR below T1")
    IF fp_float(receipt.reproducibility_score) < fp(0.90):
        RETURN OptimizeResult(action="SKIP", reason="Reproducibility insufficient")

    ppb = get_current_ppb(receipt.action_type)
    IF ppb AND receipt.latency_ms > ppb.p50_latency:
        RETURN OptimizeResult(action="SKIP", reason="Latency above p50")

    # P5 degenerate optimization check
    IF ethicist_p5.is_degenerate(receipt):
        reflex_cache.add_to_deny_list(receipt.pattern_hash, ttl_days=30)
        RETURN OptimizeResult(action="DENY", reason="Degenerate optimization")

    pattern_hash = BLAKE3.hash(receipt.trigger || receipt.context)
    IF reflex_cache.is_denied(pattern_hash):
        RETURN OptimizeResult(action="DENY", reason="Pattern on deny-list")

    # Update existing or compile new
    existing = reflex_cache.get(pattern_hash)
    IF existing AND fp_gt(ihsan, existing.confidence):
        existing.confidence = ihsan
        existing.action_chain = receipt.action_chain
        reflex_cache.update(existing)
        RETURN OptimizeResult(action="UPDATE", reflex=existing)
    ELIF existing:
        RETURN OptimizeResult(action="SKIP", reason="Existing reflex equal or better")

    new_reflex = Reflex(
        pattern_hash=pattern_hash, action_chain=receipt.action_chain,
        confidence=ihsan, use_count=0, compiled_at=now_ms(),
        source_receipt_id=receipt.receipt_id
    )
    reflex_cache.install(new_reflex)
    log_improvement_event("REFLEX_COMPILATION", pattern_hash, ihsan, receipt.receipt_id)
    RETURN OptimizeResult(action="COMPILED", reflex=new_reflex)
```

### 3.3 causal_analysis

```
FUNCTION causal_analysis(rejected_receipt: ActionReceipt) -> CausalAnalysisReport:
    """P4 root-cause + P5 ethical analysis. Ref: core/constitutional/algorithms.py (A1, A8)"""

    # Stage 1: P4 — identify deficiency and root cause
    deficiency = identify_primary_deficiency(rejected_receipt)

    IF deficiency.dimension == "REPRODUCIBILITY" AND has_nondeterminism(rejected_receipt):
        root_cause = "NONDETERMINISM"
    ELIF deficiency.dimension == "EFFICIENCY" AND rejected_receipt.latency_ms > 5000:
        root_cause = "RESOURCE_CONTENTION"
    ELIF is_novel_action_type(rejected_receipt.action_type):
        root_cause = "NOVEL_TASK"
    ELIF has_stale_reflex(rejected_receipt.pattern_hash):
        root_cause = "REFLEX_STALENESS"
    ELSE:
        root_cause = "INPUT_QUALITY"

    p4_severity = evaluate_severity(root_cause, deficiency.score)

    # Stage 2: P5 — ethical dimension
    ethical_review = ethicist_p5.analyze(rejected_receipt)

    # Stage 3: Crown hierarchy resolution (H0 > H1 > H2)
    final_severity = MAX(p4_severity, ethical_review.ethical_risk_level)
    IF ethical_review.ethical_risk_level == "CRITICAL":
        trigger_shura_review(rejected_receipt, ethical_review)

    # Stage 4: Resolution injection
    IF root_cause == "REFLEX_STALENESS":
        reflex_cache.demote(rejected_receipt.pattern_hash)
    IF root_cause IN ["INPUT_QUALITY", "NONDETERMINISM"]:
        planner_p1.add_constraint(rejected_receipt.action_type, determine_resolution(root_cause))

    semantic_memory.store_defect_pattern(
        pattern=rejected_receipt.pattern_hash,
        root_cause=root_cause, resolution=determine_resolution(root_cause, final_severity)
    )

    log_improvement_event("CAUSAL_ANALYSIS", rejected_receipt.receipt_id, root_cause, final_severity)
    RETURN CausalAnalysisReport(
        receipt_id=rejected_receipt.receipt_id,
        root_cause=root_cause, p4_severity=p4_severity,
        p5_ethical_risk=ethical_review.ethical_risk_level,
        final_severity=final_severity,
        disagreement_flag=(p4_severity != ethical_review.ethical_risk_level),
        timestamp=now_ms()
    )
```

### 3.4 compute_myelination_ratio

```
FUNCTION compute_myelination_ratio(
    s1_hits: int, s2_invocations: int,
    history: list[MyelinationSample] = None, window_days: int = 7
) -> MyelinationReport:
    """Myelination ratio with trend detection. Ref: bizra-agent/src/omni_kernel.rs"""

    total = s1_hits + s2_invocations
    IF total == 0:
        RETURN MyelinationReport(ratio=0.0, trend="COLD_START")

    ratio = s1_hits / total
    maturity = "L5_OPTIMIZING" IF ratio >= 0.90
        ELSE "L5_THRESHOLD" IF ratio >= 0.60
        ELSE "L4_EQUIVALENT" IF ratio >= 0.30
        ELSE "L3_EQUIVALENT"

    IF history IS None OR len(history) < 3:
        RETURN MyelinationReport(ratio=ratio, maturity=maturity, trend="INSUFFICIENT_DATA")

    cutoff = now_ms() - (window_days * 86_400_000)
    window = [s FOR s IN history IF s.timestamp >= cutoff]
    IF len(window) < 3:
        RETURN MyelinationReport(ratio=ratio, maturity=maturity, trend="INSUFFICIENT_DATA")

    slope = linear_slope([s.ratio FOR s IN window])

    IF slope > 0.005:
        trend, alert = "IMPROVING", None
    ELIF slope >= -0.005:
        trend, alert = "STABLE", None
    ELSE:
        new_types = count_new_action_types(window)
        IF new_types > 0:
            trend, alert = "LEARNING_PHASE", None
        ELSE:
            trend = "DECLINING"
            alert = MyelinationAlert(severity="WARNING", slope=slope,
                message="Myelination declining without new task types — triggering CAR")
            trigger_causal_analysis_for_declining_myelination(window)

    RETURN MyelinationReport(
        ratio=ratio, maturity=maturity, trend=trend, slope=slope,
        s1_hits=s1_hits, s2_invocations=s2_invocations, alert=alert
    )
```

### 3.5 generate_cmmi_report

```
FUNCTION generate_cmmi_report(
    baselines: list[PPB], models: list[PPM],
    myelination: MyelinationReport, improvements: list[ImprovementEvent],
    time_range: TimeRange
) -> CMMIReport:
    """CMMI Level 5 evidence pack. Ref: core/proof_engine/evidence_ledger.py, constants.py"""

    # Section 1: QPM
    ppb_summary = PPBSummary.from_baselines(baselines)
    qpm_pass = (ppb_summary.ihsan_mean >= IHSAN_PRODUCTION
                AND ppb_summary.ihsan_stddev <= 0.03
                AND ppb_summary.snr_mean >= UNIFIED_SNR_THRESHOLD)

    # Section 2: Self-Optimization
    omega_events = [e FOR e IN improvements IF e.type == "REFLEX_COMPILATION"]
    pruning_events = [e FOR e IN improvements IF e.type == "REFLEX_PRUNING"]

    # Section 3: CAR
    car_events = [e FOR e IN improvements IF e.type == "CAUSAL_ANALYSIS"]
    by_cause = group_by(car_events, key=lambda e: e.root_cause)
    recurrence_reductions = {}
    FOR cause, events IN by_cause.items():
        half = len(events) // 2
        IF half > 0:
            recurrence_reductions[cause] = max(0.0, 1.0 - len(events[half:]) / half)

    # Section 4: Myelination
    myelination_pass = myelination.ratio >= 0.60 AND myelination.trend IN ["STABLE","IMPROVING","LEARNING_PHASE"]

    # Section 5: Cumulative Improvement Score
    score = (0.30 * max(0, myelination.slope or 0)
           + 0.25 * (mean(recurrence_reductions.values()) IF recurrence_reductions ELSE 0)
           + 0.25 * max(0, fp_float(ppb_summary.ihsan_slope or 0))
           + 0.20 * (1.0 - min(1.0, fp_float(ppb_summary.ihsan_stddev) / 0.10)))

    merkle_root = MerkleTree.build(collect_receipt_ids(time_range)).root
    report = CMMIReport(
        standard="CMMI_v2.0_Level_5", time_range=time_range,
        qpm_pass=qpm_pass, ppb_summary=ppb_summary,
        omega_loop_count=len(omega_events), reflex_pruning_count=len(pruning_events),
        car_total=len(car_events), recurrence_reductions=recurrence_reductions,
        myelination=myelination, myelination_pass=myelination_pass,
        cumulative_improvement_score=score,
        merkle_root=merkle_root, total_receipts=len(collect_receipt_ids(time_range))
    )
    report.signature = node_keypair.sign(BLAKE3.hash(report.serialize()))
    RETURN report
```

---

## 4. TDD Anchors

```
TEST ppb_cold_start_returns_insufficient_data:
    receipts = generate_test_receipts(count=30, ihsan_range=(0.90, 1.00))
    ppb = compute_process_baseline(receipts, window_size=50)
    ASSERT ppb.status == "COLD_START"
    ASSERT ppb.ihsan_mean IS None

TEST ppb_detects_stable_high_quality_process:
    receipts = generate_test_receipts(count=200, ihsan_range=(0.95, 0.98))
    ppb = compute_process_baseline(receipts, window_size=100)
    ASSERT ppb.is_capable == True AND ppb.is_stable == True

TEST ppb_detects_declining_trend:
    receipts = generate_declining_receipts(count=200, start=0.98, end=0.91)
    ppb = compute_process_baseline(receipts, window_size=200)
    ASSERT ppb.is_improving == False AND fp_float(ppb.ihsan_slope) < 0.0

TEST rlvr_compiles_high_quality_path_to_reflex:
    receipt = create_test_receipt(intent=0.97, efficiency=0.96, impact=0.98,
                                  reproducibility=0.95, snr=0.96, latency_ms=50)
    result = rlvr_optimize(receipt, empty_reflex_cache())
    ASSERT result.action == "COMPILED"
    ASSERT result.reflex.confidence == compute_ihsan_from_receipt(receipt)

TEST rlvr_rejects_low_reproducibility:
    receipt = create_test_receipt(intent=0.97, efficiency=0.96, impact=0.98,
                                  reproducibility=0.80, snr=0.96, latency_ms=50)
    result = rlvr_optimize(receipt, empty_reflex_cache())
    ASSERT result.action == "SKIP" AND "Reproducibility" IN result.reason

TEST rlvr_blocks_degenerate_optimization:
    receipt = create_degenerate_receipt()  # High efficiency, zero real impact
    cache = empty_reflex_cache()
    result = rlvr_optimize(receipt, cache)
    ASSERT result.action == "DENY" AND cache.is_denied(receipt.pattern_hash)

TEST causal_analysis_identifies_resource_contention:
    receipt = create_test_receipt(intent=0.95, efficiency=0.60, impact=0.95,
                                  reproducibility=0.95, latency_ms=8000)
    report = causal_analysis(receipt)
    ASSERT report.root_cause == "RESOURCE_CONTENTION"

TEST causal_analysis_crown_hierarchy_on_disagreement:
    receipt = create_test_receipt_with_ethical_concern()
    mock_p4_severity("LOW"); mock_p5_severity("HIGH")
    report = causal_analysis(receipt)
    ASSERT report.final_severity == "HIGH" AND report.disagreement_flag == True

TEST myelination_detects_learning_phase:
    history = create_declining_myelination_history(with_new_action_types=True)
    report = compute_myelination_ratio(s1_hits=60, s2_invocations=40, history=history)
    ASSERT report.trend == "LEARNING_PHASE" AND report.alert IS None

TEST myelination_alerts_on_true_degradation:
    history = create_declining_myelination_history(with_new_action_types=False)
    report = compute_myelination_ratio(s1_hits=60, s2_invocations=40, history=history)
    ASSERT report.trend == "DECLINING" AND report.alert.severity == "WARNING"

TEST cmmi_report_includes_all_sections_and_is_signed:
    report = generate_cmmi_report(sample_ppbs, sample_ppms, sample_myelination,
                                   sample_improvements, last_30_days)
    ASSERT report.qpm_pass IS NOT None
    ASSERT report.myelination_pass IS NOT None
    ASSERT report.cumulative_improvement_score >= 0.0
    ASSERT report.merkle_root IS NOT None AND len(report.merkle_root) == 32
    ASSERT Ed25519.verify(report.signature, BLAKE3.hash(report.serialize()), node_pubkey)
```

---

## 5. Cross-References

### Codebase Modules

| Module | CMMI L5 Relevance |
|:---|:---|
| `core/constitutional/algorithms.py` | A1 (Ihsan scoring), A10 (Reflex compilation) |
| `core/constitutional/types.py` | ActionReceipt, Reflex, WalletState (ihsan_history) |
| `core/constitutional/fixed_point.py` | Deterministic arithmetic for all PPB/PPM computation |
| `core/integration/constants.py` | IHSAN_PRODUCTION, IHSAN_CI, SNR_T1, SNR_T0, IHSAN_BLOOM_ELIGIBILITY |
| `core/iaas/snr_v2_adapter.py` | SNR calculation -- parallel quality dimension |
| `core/proof_engine/evidence_ledger.py` | Merkle chain for improvement event trail |
| `core/governance/` | Shura voting, BLOOM-weighted proposals for OPM |
| `core/federation/` | Cross-node PPB aggregation, anonymized metric sharing |
| `core/living_memory/` | Semantic memory for defect pattern storage |
| `bizra-agent/src/omni_kernel.rs` | ReflexCache -- S1 cache hit/miss tracking |
| `bizra-ttrl/` | Self-RLVR engine -- GRPO training loop |

### Sibling Specs

- Phase 00 (Framework Overview) -- unified evidence model and cross-standard invariants
- Phase 01 (ISO 25010) -- FR-C11 shares Myelination Ratio with FR-C23
- Phase 03 (SOC 2 Type II) -- Processing Integrity uses same Merkle chain
- Phase 04 (ISO 9001) -- QMS Process Approach shares PPB/PPM infrastructure

### CMMI v2.0 Practice Area Mapping

QPM -> FR-C20, FR-C21 (PPB/PPM) | MPM -> FR-C20, FR-C23 (Myelination) | OPM -> FR-C22, FR-C25 (Omega Loop, federation) | CAR -> FR-C24 (CausalAnalysisReport) | PCM -> FR-C22, FR-C26 (Improvement Score) | GOV -> FR-C25 (Shura/BLOOM) | II -> FR-C26 (Improvement Event Log + Merkle proofs)
