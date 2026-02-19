# Phase 04 — Impact Measurement & Proof-of-Impact

> **Version:** 0.1.0 | **Status:** Specification + Pseudocode
> **Standing on Giants:** Deming (PDCA, measurement-driven improvement) · Drucker ("what gets measured gets managed") · Kaplan & Norton (balanced scorecard) · Shannon (information gain)

## 4.1 Functional Requirements

| ID | Requirement | Acceptance Criteria |
|----|-------------|---------------------|
| M-01 | Record time saved per task | Timer diff: manual estimate vs. actual |
| M-02 | Record tasks completed (with receipts) | Count of APPROVED receipts per day |
| M-03 | Record errors avoided | Count of SAT veto + risk mitigations |
| M-04 | Record knowledge consolidated | Knowledge graph node growth |
| M-05 | Weekly Impact Report | Automated, auditable, receipt-backed |
| M-06 | Impact Ledger (persistent) | Append-only, hash-chained, local |
| M-07 | Privacy-first: private by default | No transmission without explicit consent |

## 4.2 Baseline Capture (Day 0)

```pseudocode
MODULE BaselineCapture:
  """
  Captures the "before" state for meaningful comparison.
  Called during onboarding (Phase 02, Step 4).
  Standing on Giants: Kaplan & Norton (Balanced Scorecard, 1992)
  """

  STRUCT UserBaseline:
    captured_at:      Timestamp
    node_id:          bytes[16]

    # ── Pain Points (qualitative) ──
    pain_points:      List<String>       # 3 things that waste time
    friction_areas:   List<String>       # Where do you get stuck?

    # ── Quantitative Baselines ──
    tasks_per_week:   int                # Self-reported
    hours_wasted:     float              # Hours/week on busywork
    decisions_per_day: int               # Rough estimate
    clarity_score:    int                # 1-10 self-assessment
    rework_percent:   float              # % of work that needs redoing

    # ── Weekly Goals (for comparison) ──
    weekly_goals:     List<String>       # What they aim to accomplish
    goal_completion_rate: float          # % of last week's goals completed

    # ── Integrity ──
    hash:             bytes[32]          # SHA-256 of all fields
    signature:        bytes[64]          # Ed25519 seal

  FUNCTION capture_baseline(ui: UserInterface) -> UserBaseline:
    baseline = UserBaseline(
      captured_at     = utc_now(),
      node_id         = get_node_id(),
      pain_points     = ui.prompt_list("3 things that waste your time", count=3),
      friction_areas  = ui.prompt_list("Where do you get stuck most?", count=3),
      tasks_per_week  = ui.prompt_number("Tasks you complete in a typical week?"),
      hours_wasted    = ui.prompt_number("Hours/week lost to busywork?"),
      decisions_per_day = ui.prompt_number("Important decisions you make daily?"),
      clarity_score   = ui.prompt_scale("How clear is your daily plan? (1-10)", 1, 10),
      rework_percent  = ui.prompt_percent("% of your work that needs redoing?"),
      weekly_goals    = ui.prompt_list("What do you want to accomplish this week?", count=3),
      goal_completion_rate = ui.prompt_percent("% of last week's goals completed?")
    )

    baseline.hash = sha256(canonical_json(baseline, exclude=["hash", "signature"]))
    baseline.signature = sign_message(load_private_key(), baseline.hash)

    store_baseline(baseline)
    RETURN baseline
```

## 4.3 Daily Metrics Collection

```pseudocode
MODULE DailyMetrics:
  """
  Automatic metrics collection during empowerment loop execution.
  No extra user effort — derived from receipts and task execution.
  """

  STRUCT DailyMetric:
    date:               Date
    node_id:            bytes[16]

    # ── Velocity Metrics ──
    tasks_completed:    int            # Count of APPROVED receipts
    tasks_attempted:    int            # Total tasks started
    tasks_blocked:      int            # BLOCKED by SAT or ZANN
    automations_run:    int            # "Next Action" button presses

    # ── Time Metrics ──
    time_saved_minutes: float          # sum(manual_estimate - actual_duration)
    total_active_minutes: float        # Time spent interacting
    planning_minutes:   float          # Time in goal/planning phase
    execution_minutes:  float          # Time in task execution

    # ── Quality Metrics ──
    avg_snr_score:      float          # Average across all receipts
    avg_ihsan_score:    float          # Average across all receipts
    errors_avoided:     int            # SAT vetoes + risk mitigations applied
    rework_prevented:   int            # Tasks where risk mitigation activated

    # ── Knowledge Metrics ──
    receipts_generated: int            # Total receipts (chain growth)
    evidence_links:     int            # Total evidence pointers created
    memory_nodes_added: int            # Knowledge graph growth

    # ── Seal ──
    receipt_hashes:     List<bytes[32]>  # All receipt hashes for this day
    hash:               bytes[32]
    signature:          bytes[64]

  FUNCTION collect_daily_metrics(date: Date) -> DailyMetric:
    receipts = load_receipts_for_date(date)
    tasks = load_tasks_for_date(date)

    metric = DailyMetric(
      date               = date,
      node_id            = get_node_id(),
      tasks_completed    = count(r FOR r IN receipts IF r.policy_result == APPROVED),
      tasks_attempted    = len(tasks),
      tasks_blocked      = count(r FOR r IN receipts IF r.policy_result == BLOCKED),
      automations_run    = count(t FOR t IN tasks IF t.was_automated),
      time_saved_minutes = sum(
        t.manual_estimate - t.actual_duration
        FOR t IN tasks
        IF t.status == DONE AND t.was_automated
      ),
      total_active_minutes = sum(t.active_time FOR t IN tasks),
      planning_minutes   = sum(s.duration FOR s IN sessions IF s.phase == "planning"),
      execution_minutes  = sum(s.duration FOR s IN sessions IF s.phase == "execution"),
      avg_snr_score      = mean(r.snr_score FOR r IN receipts IF r.policy_result == APPROVED),
      avg_ihsan_score    = mean(r.ihsan_score FOR r IN receipts IF r.policy_result == APPROVED),
      errors_avoided     = count_vetoes(date) + count_risk_mitigations(date),
      rework_prevented   = count_risk_activations(date),
      receipts_generated = len(receipts),
      evidence_links     = sum(len(r.evidence_links) FOR r IN receipts),
      memory_nodes_added = count_new_memory_nodes(date),
      receipt_hashes     = [r.receipt_id FOR r IN receipts],
    )

    metric.hash = sha256(canonical_json(metric, exclude=["hash", "signature"]))
    metric.signature = sign_message(load_private_key(), metric.hash)

    append_to_impact_ledger(metric)
    RETURN metric
```

## 4.4 Weekly Impact Report

```pseudocode
MODULE WeeklyImpactReport:
  """
  Automated weekly report comparing current week to baseline.
  Every claim backed by receipt references.
  Standing on Giants: Deming (PDCA Check phase) · Drucker (measurement)
  """

  STRUCT ImpactReport:
    week_number:     int
    date_range:      (Date, Date)
    node_id:         bytes[16]

    # ── Headline Metrics (vs. Baseline) ──
    hours_saved:         float         # Cumulative time_saved_minutes / 60
    tasks_shipped:       int           # Total APPROVED tasks
    errors_avoided:      int           # Total vetoes + mitigations
    knowledge_growth:    int           # Net new memory nodes

    # ── Comparison to Baseline ──
    velocity_change:     float         # % change in tasks/week
    clarity_improvement: float         # Change in clarity score (self-reported)
    rework_reduction:    float         # % change in rework rate
    goal_completion_change: float      # % change in goal completion rate

    # ── Quality Profile ──
    avg_snr:             float
    avg_ihsan:           float
    receipts_total:      int
    chain_integrity:     bool          # Full chain verified

    # ── Daily Breakdown ──
    daily_metrics:       List<DailyMetric>  # 7 days

    # ── Receipt References ──
    supporting_receipts: List<bytes[32]>    # All receipt hashes backing claims

    # ── Seal ──
    hash:                bytes[32]
    signature:           bytes[64]

  FUNCTION generate_weekly_report(week: int, baseline: UserBaseline) -> ImpactReport:
    start, end = week_date_range(week)
    daily = [collect_daily_metrics(d) FOR d IN date_range(start, end)]

    # Aggregate
    hours_saved     = sum(d.time_saved_minutes FOR d IN daily) / 60
    tasks_shipped   = sum(d.tasks_completed FOR d IN daily)
    errors_avoided  = sum(d.errors_avoided FOR d IN daily)
    knowledge_growth = sum(d.memory_nodes_added FOR d IN daily)

    # Compare to baseline
    current_tasks_week = tasks_shipped
    velocity_change = (
      (current_tasks_week - baseline.tasks_per_week)
      / max(baseline.tasks_per_week, 1) * 100
    )

    # Self-report update (weekly check-in)
    current_clarity = prompt_scale("How clear was your daily plan this week? 1-10")
    clarity_improvement = current_clarity - baseline.clarity_score

    current_rework = prompt_percent("% of work that needed redoing this week?")
    rework_reduction = baseline.rework_percent - current_rework

    current_goal_rate = prompt_percent("% of this week's goals completed?")
    goal_completion_change = current_goal_rate - baseline.goal_completion_rate

    # Quality
    all_receipts = flatten(d.receipt_hashes FOR d IN daily)
    chain_ok = verify_chain(load_chain(start, end), get_public_key()).valid

    report = ImpactReport(
      week_number          = week,
      date_range           = (start, end),
      node_id              = get_node_id(),
      hours_saved          = hours_saved,
      tasks_shipped        = tasks_shipped,
      errors_avoided       = errors_avoided,
      knowledge_growth     = knowledge_growth,
      velocity_change      = velocity_change,
      clarity_improvement  = clarity_improvement,
      rework_reduction     = rework_reduction,
      goal_completion_change = goal_completion_change,
      avg_snr              = mean(d.avg_snr_score FOR d IN daily),
      avg_ihsan            = mean(d.avg_ihsan_score FOR d IN daily),
      receipts_total       = len(all_receipts),
      chain_integrity      = chain_ok,
      daily_metrics        = daily,
      supporting_receipts  = all_receipts,
    )

    report.hash = sha256(canonical_json(report, exclude=["hash", "signature"]))
    report.signature = sign_message(load_private_key(), report.hash)

    store_report(report)
    RETURN report
```

## 4.5 Impact Ledger

```pseudocode
MODULE ImpactLedger:
  """
  Append-only, hash-chained ledger of all impact evidence.
  Private by default — user controls export.
  Standing on Giants: Merkle (hash chain) · Lamport (ordering)
  """

  STRUCT LedgerEntry:
    index:      uint64
    type:       Enum(BASELINE, DAILY_METRIC, WEEKLY_REPORT, MANUAL_NOTE)
    payload:    bytes           # Canonical JSON of the entry
    hash:       bytes[32]      # SHA-256 of (prev_hash || payload)
    prev_hash:  bytes[32]
    timestamp:  ISO8601
    signature:  bytes[64]

  FUNCTION append_to_ledger(entry_type: EntryType, payload: Any):
    prev = get_last_ledger_entry()
    canonical = canonical_json(payload)

    entry = LedgerEntry(
      index     = prev.index + 1 IF prev ELSE 0,
      type      = entry_type,
      payload   = canonical,
      prev_hash = prev.hash IF prev ELSE ZERO_HASH,
      timestamp = utc_now(),
    )
    entry.hash = sha256(entry.prev_hash + canonical)
    entry.signature = sign_message(load_private_key(), entry.hash)

    persist_entry(entry)

  FUNCTION export_ledger(format: "json" | "csv" | "pdf") -> bytes:
    """
    User-initiated export. Always includes chain verification proof.
    """
    ledger = load_full_ledger()
    verification = verify_chain(ledger, get_public_key())

    ASSERT verification.valid, "Cannot export corrupted ledger"

    RETURN serialize(ledger, format, include_verification=true)
```

## 4.6 TDD Anchors

```pseudocode
TEST "baseline_captures_all_fields":
  baseline = capture_baseline(mock_ui_with_answers)
  ASSERT baseline.pain_points IS NOT NONE AND len(baseline.pain_points) == 3
  ASSERT baseline.clarity_score >= 1 AND baseline.clarity_score <= 10
  ASSERT baseline.hash IS NOT NONE
  ASSERT verify_signature(public_key, baseline.hash, baseline.signature)

TEST "daily_metrics_derived_from_receipts":
  receipts = generate_test_receipts(count=10, approved=8, blocked=2)
  metric = collect_daily_metrics(today())
  ASSERT metric.tasks_completed == 8
  ASSERT metric.tasks_blocked == 2
  ASSERT metric.receipts_generated == 10

TEST "time_saved_calculated_correctly":
  tasks = [
    Task(manual_estimate=30, actual_duration=5, automated=true),
    Task(manual_estimate=60, actual_duration=10, automated=true),
  ]
  metric = collect_daily_metrics_from(tasks)
  ASSERT metric.time_saved_minutes == 75  # (30-5) + (60-10)

TEST "weekly_report_compares_to_baseline":
  baseline = UserBaseline(tasks_per_week=10, clarity_score=5)
  # Simulate a week where user completes 15 tasks
  report = generate_weekly_report(week=1, baseline=baseline)
  ASSERT report.velocity_change == 50.0  # (15-10)/10 * 100
  ASSERT report.chain_integrity == true

TEST "impact_ledger_chain_integrity":
  append_to_ledger(BASELINE, baseline_data)
  append_to_ledger(DAILY_METRIC, day1_data)
  append_to_ledger(DAILY_METRIC, day2_data)
  ledger = load_full_ledger()
  ASSERT ledger[1].prev_hash == ledger[0].hash
  ASSERT ledger[2].prev_hash == ledger[1].hash

TEST "ledger_detects_tampering":
  append_to_ledger(BASELINE, baseline_data)
  append_to_ledger(DAILY_METRIC, day1_data)
  tamper_entry(index=0, field="payload", value="fake")
  verification = verify_chain(load_full_ledger(), public_key)
  ASSERT verification.valid == false

TEST "export_fails_on_corrupted_ledger":
  corrupt_ledger_entry(index=2)
  ASSERT_RAISES(IntegrityError, export_ledger("json"))

TEST "report_all_claims_have_receipt_refs":
  report = generate_weekly_report(week=1, baseline)
  ASSERT report.receipts_total > 0
  ASSERT len(report.supporting_receipts) == report.receipts_total

TEST "privacy_default_no_transmission":
  report = generate_weekly_report(week=1, baseline)
  ASSERT report.transmitted == false
  ASSERT report.storage_location == LOCAL_ENCRYPTED_STORE
```
