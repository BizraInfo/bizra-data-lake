# Phase 02 — Daily Empowerment Loop

> **Version:** 0.1.0 | **Status:** Specification + Pseudocode
> **Standing on Giants:** Boyd (OODA loop) · Deming (PDCA) · Besta (Graph-of-Thoughts) · Drucker (management by objectives) · Allen (GTD)

## 2.1 Functional Requirements

| ID | Requirement | Acceptance Criteria |
|----|-------------|---------------------|
| E-01 | Goal intake (1–3 SMART goals + constraints) | Structured capture with validation |
| E-02 | PAT produces daily plan (top 3 outcomes) | Plan generated within 30 seconds |
| E-03 | Task breakdown with time estimates | Each outcome decomposes to ≤ 5 tasks |
| E-04 | Risk list with mitigations | ≥ 1 risk per goal, each with mitigation |
| E-05 | "Next Action" button executes safe automations | Only allowlisted tools, receipt generated |
| E-06 | Task Force executes approved tasks | File ops, summaries, drafting, code gen |
| E-07 | 10-minute daily check-in | Morning + optional evening reflection |

## 2.2 Empowerment Loop Architecture

```
                    ┌─────────────────────┐
                    │    USER INTERFACE    │
                    │  (Tauri Webview)     │
                    └────────┬────────────┘
                             │ goals, approvals, feedback
                             ▼
┌─────────────────────────────────────────────────────────────┐
│                   ORCHESTRATION (Layer 3)                     │
│                                                               │
│  ┌──────────┐   ┌───────────┐   ┌──────────────┐            │
│  │ Extended  │──►│  Team     │──►│ Checkpoint   │            │
│  │ OODA Loop │   │  Planner  │   │ Manager      │            │
│  │ (8 phase) │   │           │   │              │            │
│  └──────────┘   └───────────┘   └──────────────┘            │
│       │                                                       │
│       ▼                                                       │
│  ┌──────────────────────────────────────────────────────┐    │
│  │           PAT — Personal Agent Team (7)               │    │
│  │  Master │ Data │ Planner │ Ethics │ Comms │ Memory │ F│    │
│  └──────────────────────────────┬────────────────────────┘    │
│                                 │ plan + tasks                │
│                                 ▼                             │
│  ┌──────────────────────────────────────────────────────┐    │
│  │           SAT — Validation Team (5)                   │    │
│  │  Security* │ Ethics* │ Perf │ Consistency │ Resources │    │
│  │  (* = VETO power)                                     │    │
│  └──────────────────────────────┬────────────────────────┘    │
│                                 │ approved tasks              │
│                                 ▼                             │
│  ┌──────────────────────────────────────────────────────┐    │
│  │           TASK FORCE — Execution Layer                │    │
│  │  Allowlisted tools only: file_ops, summarize, draft,  │    │
│  │  code_gen, web_search, calendar, email_draft           │    │
│  └──────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## 2.3 Goal Intake

```pseudocode
MODULE GoalIntake:
  """
  Structured SMART goal capture.
  Standing on Giants: Drucker (MBO, 1954) · Locke (Goal Setting Theory, 1968)
  Reuses: core.sovereign.team_planner
  """

  STRUCT Goal:
    id:           UUID
    title:        String          # What
    specific:     String          # Detailed description
    measurable:   String          # How to measure success
    achievable:   bool            # Self-assessed
    relevant:     String          # Why this matters
    time_bound:   Date            # Deadline
    constraints:  List<String>    # Blockers, limitations
    priority:     Enum(HIGH, MEDIUM, LOW)
    created_at:   Timestamp

  FUNCTION capture_goals(ui: UserInterface) -> List<Goal>:
    goals = []
    show_page("What are your goals for today?", hint="1–3 goals")

    WHILE len(goals) < 3:
      input = ui.prompt_goal_form()
      IF input == DONE AND len(goals) >= 1:
        BREAK

      goal = validate_smart(input)
      IF goal.validation_errors:
        ui.show_feedback(goal.validation_errors)
        CONTINUE

      goals.append(goal)
      ui.show_confirmation(f"Goal {len(goals)}: {goal.title}")

    RETURN goals

  FUNCTION validate_smart(input: GoalInput) -> ValidationResult:
    errors = []
    IF len(input.title) < 5:
      errors.append("Title too vague — be specific")
    IF NOT input.measurable:
      errors.append("How will you know you succeeded?")
    IF NOT input.time_bound:
      errors.append("When should this be done?")
    RETURN ValidationResult(goal=input, errors=errors)
```

## 2.4 PAT Planning Pipeline

```pseudocode
MODULE PATPlanningPipeline:
  """
  7-agent Personal Agent Team produces daily plan.
  Standing on Giants: Boyd (OODA) · Kahneman (dual-process) · Besta (GoT)
  Reuses: core.sovereign.collective_intelligence, core.sovereign.team_planner
  """

  STRUCT DailyPlan:
    date:         Date
    goals:        List<Goal>
    top_outcomes: List<Outcome>      # Exactly 3
    task_tree:    List<TaskNode>      # Hierarchical breakdown
    risks:        List<Risk>
    estimated_hours: float
    ihsan_score:  float              # Must be >= UNIFIED_IHSAN_THRESHOLD

  STRUCT Outcome:
    title:        String
    goal_ref:     UUID               # Links to parent goal
    success_criteria: String
    estimated_minutes: int

  STRUCT TaskNode:
    id:           UUID
    title:        String
    outcome_ref:  UUID               # Links to parent outcome
    description:  String
    estimated_minutes: int
    tool_required: Optional<ToolName>  # If automatable
    automatable:  bool
    status:       Enum(PENDING, IN_PROGRESS, DONE, BLOCKED)
    children:     List<TaskNode>     # Sub-tasks (max depth 2)

  STRUCT Risk:
    description:  String
    likelihood:   float              # 0.0 - 1.0
    impact:       float              # 0.0 - 1.0
    mitigation:   String
    goal_ref:     UUID

  FUNCTION generate_daily_plan(goals: List<Goal>, context: UserContext) -> DailyPlan:
    """
    PAT generates plan via collective intelligence.
    Each agent contributes its specialty, then collective synthesizes.
    """
    # Phase 1: Individual PAT agent analysis
    master_analysis    = pat.master_reasoner.analyze(goals, context)
    data_insights      = pat.data_analyzer.analyze(goals, context.history)
    execution_plan     = pat.execution_planner.decompose(goals)
    ethics_review      = pat.ethics_guardian.review(goals, execution_plan)
    communication_plan = pat.communicator.draft_notifications(goals)
    memory_context     = pat.memory_architect.recall_relevant(goals)
    fused_plan         = pat.fusion.synthesize(
      master_analysis, data_insights, execution_plan,
      ethics_review, memory_context
    )

    # Phase 2: Extract top 3 outcomes
    top_outcomes = rank_and_select(
      fused_plan.outcomes,
      count=3,
      criteria="impact * feasibility / time_cost"
    )

    # Phase 3: Task decomposition (max 5 tasks per outcome, max depth 2)
    task_tree = []
    FOR outcome IN top_outcomes:
      tasks = decompose_to_tasks(outcome, max_tasks=5, max_depth=2)
      FOR task IN tasks:
        task.automatable = is_in_allowlist(task.tool_required)
      task_tree.extend(tasks)

    # Phase 4: Risk assessment (≥ 1 risk per goal)
    risks = []
    FOR goal IN goals:
      goal_risks = assess_risks(goal, task_tree, context)
      ASSERT len(goal_risks) >= 1, "Every goal must have ≥1 risk identified"
      risks.extend(goal_risks)

    # Phase 5: Ihsan gate
    plan = DailyPlan(
      date=today(),
      goals=goals,
      top_outcomes=top_outcomes,
      task_tree=task_tree,
      risks=risks,
      estimated_hours=sum(t.estimated_minutes FOR t IN task_tree) / 60,
      ihsan_score=compute_plan_ihsan(fused_plan)
    )

    ASSERT plan.ihsan_score >= UNIFIED_IHSAN_THRESHOLD,
      "Plan does not meet Ihsan threshold — re-plan with better evidence"

    RETURN plan
```

## 2.5 Task Force Execution

```pseudocode
MODULE TaskForce:
  """
  Executes approved tasks using allowlisted tools.
  Every execution produces a Receipt (Phase 03).
  Standing on Giants: Principle of Least Privilege · Allowlist > Denylist
  Reuses: core.sovereign.autonomy_matrix.AutonomyLevel
  """

  CONST TOOL_ALLOWLIST = {
    "file_read":       { risk: LOW,    reversible: true  },
    "file_write":      { risk: MEDIUM, reversible: true  },
    "file_organize":   { risk: LOW,    reversible: true  },
    "text_summarize":  { risk: LOW,    reversible: true  },
    "text_draft":      { risk: LOW,    reversible: true  },
    "code_generate":   { risk: MEDIUM, reversible: true  },
    "web_search":      { risk: LOW,    reversible: true  },
    "calendar_read":   { risk: LOW,    reversible: true  },
    "email_draft":     { risk: MEDIUM, reversible: false },
  }

  FUNCTION execute_task(task: TaskNode, autonomy: AutonomyLevel) -> TaskResult:
    # Gate 1: Tool allowlist
    IF task.tool_required NOT IN TOOL_ALLOWLIST:
      RETURN TaskResult(status=BLOCKED, reason="Tool not in allowlist")

    tool_risk = TOOL_ALLOWLIST[task.tool_required].risk

    # Gate 2: Autonomy matrix check
    IF NOT autonomy_permits(autonomy, tool_risk):
      # Needs user approval
      approval = request_user_approval(task)
      IF NOT approval.granted:
        RETURN TaskResult(status=SKIPPED, reason="User declined")

    # Gate 3: SAT validation (consensus 3-of-5, veto by security/ethics)
    sat_verdict = sat_validate(task)
    IF sat_verdict.vetoed:
      RETURN TaskResult(status=BLOCKED, reason=sat_verdict.veto_reason)

    # Execute
    start_time = now()
    result = run_tool(task.tool_required, task.parameters)
    duration = now() - start_time

    # Generate receipt (Phase 03 pipeline)
    receipt = generate_receipt(
      task=task,
      result=result,
      duration=duration,
      model_version=get_model_version(),
      snr_score=compute_snr(result),
      ihsan_score=compute_ihsan(result)
    )

    RETURN TaskResult(
      status=DONE,
      output=result,
      receipt=receipt,
      duration=duration
    )

  FUNCTION handle_next_action_button(plan: DailyPlan) -> TaskResult:
    """
    The "Next Action" button — finds the highest-priority
    automatable pending task and executes it.
    Standing on Giants: Allen (GTD, 2001) — "What's the next action?"
    """
    next_task = find_next_automatable(plan.task_tree)
    IF next_task IS NONE:
      RETURN TaskResult(status=EMPTY, reason="No automatable tasks pending")

    RETURN execute_task(next_task, current_autonomy_level())
```

## 2.6 Extended OODA Cycle (8-Phase)

```pseudocode
MODULE ExtendedOODA:
  """
  Extended OODA with PREDICT/COORDINATE/LEARN phases.
  Standing on Giants: Boyd (OODA, 1976) · Deming (PDCA, 1950)
  Reuses: core.sovereign.autonomy (extended_ooda=true in proactive_config.yaml)
  """

  FUNCTION run_daily_cycle(entity: ProactiveSovereignEntity, goals: List<Goal>):
    # 1. OBSERVE — Gather state
    state = entity.observe(
      user_goals=goals,
      system_health=check_health(),
      external_signals=scan_environment()
    )

    # 2. ORIENT — Contextualize (GoT multi-path)
    hypotheses = entity.orient(state, method="graph_of_thoughts", min_paths=3)

    # 3. PREDICT — Forecast outcomes for each hypothesis
    predictions = entity.predict(hypotheses, horizon="24h")

    # 4. DECIDE — Select best action plan
    plan = entity.decide(
      predictions,
      constraints=goals_to_constraints(goals),
      ihsan_threshold=UNIFIED_IHSAN_THRESHOLD
    )

    # 5. ACT — Execute approved tasks
    results = entity.act(plan, autonomy=current_autonomy_level())

    # 6. COORDINATE — Sync state across agents
    entity.coordinate(results, pat=entity.pat, sat=entity.sat)

    # 7. CHECK — Verify outcomes (Deming's C in PDCA)
    verification = entity.check(results, expected=plan.expected_outcomes)

    # 8. LEARN — Update models and memory
    entity.learn(
      cycle_results=results,
      verification=verification,
      update_memory=true,
      update_weights=true
    )

    RETURN CycleResult(
      state=state,
      plan=plan,
      results=results,
      verification=verification,
      receipt=generate_cycle_receipt(state, plan, results, verification)
    )
```

## 2.7 Onboarding Flow (First-Time Setup)

```pseudocode
MODULE Onboarding:
  """
  PAT personalization + first goal capture.
  Called from Phase 01 install wizard (Step 5).
  """

  STRUCT OnboardingResult:
    pat_config:     PATConfiguration
    goals:          List<Goal>
    baseline:       UserBaseline       # For Phase 04

  FUNCTION run_onboarding_flow() -> OnboardingResult:
    # Step 1: Personalization
    profile = prompt_profile(
      questions=[
        "What do you do? (role/profession)",
        "What tools do you use daily?",
        "What frustrates you most about your workflow?",
      ]
    )

    # Step 2: PAT configuration (which agents active)
    pat_config = recommend_pat_config(profile)
    pat_config = user_customize(pat_config)  # User can toggle agents

    # Step 3: First goals
    goals = capture_goals(hint="Start with just 1 — you can add more later")

    # Step 4: Baseline capture (Phase 04 integration)
    baseline = capture_baseline(
      pain_points=prompt("Name 3 things that waste your time"),
      weekly_goals=prompt("What did you accomplish last week?"),
      clarity_score=prompt_scale("How clear is your daily plan? 1-10"),
      hours_wasted=prompt_number("Hours/week lost to busywork?")
    )

    RETURN OnboardingResult(pat_config, goals, baseline)
```

## 2.8 TDD Anchors

```pseudocode
TEST "goal_validation_rejects_vague":
  result = validate_smart(GoalInput(title="Do stuff", measurable="", time_bound=null))
  ASSERT len(result.errors) >= 2

TEST "goal_validation_accepts_smart":
  result = validate_smart(GoalInput(
    title="Ship invoice feature",
    measurable="PR merged + 3 tests passing",
    time_bound=tomorrow()
  ))
  ASSERT len(result.errors) == 0

TEST "daily_plan_has_3_outcomes":
  plan = generate_daily_plan(goals=[goal_1, goal_2], context=test_context)
  ASSERT len(plan.top_outcomes) == 3

TEST "every_goal_has_at_least_1_risk":
  plan = generate_daily_plan(goals=[goal_1, goal_2, goal_3], context=test_context)
  FOR goal IN plan.goals:
    ASSERT any(r.goal_ref == goal.id FOR r IN plan.risks)

TEST "task_tree_max_depth_is_2":
  plan = generate_daily_plan(goals=[goal_1], context=test_context)
  ASSERT max_depth(plan.task_tree) <= 2

TEST "plan_ihsan_above_threshold":
  plan = generate_daily_plan(goals=[goal_1], context=test_context)
  ASSERT plan.ihsan_score >= 0.95

TEST "task_force_blocks_unlisted_tool":
  task = TaskNode(tool_required="rm_rf_everything")
  result = execute_task(task, AutonomyLevel.AUTOLOW)
  ASSERT result.status == BLOCKED

TEST "task_force_requires_approval_for_medium_risk":
  task = TaskNode(tool_required="email_draft")
  # With AUTOLOW autonomy, medium-risk needs approval
  result = execute_task(task, AutonomyLevel.OBSERVER)
  ASSERT result.status IN [BLOCKED, SKIPPED]

TEST "next_action_returns_highest_priority_automatable":
  plan = make_plan_with_tasks([
    TaskNode(priority=LOW, automatable=true),
    TaskNode(priority=HIGH, automatable=false),
    TaskNode(priority=HIGH, automatable=true),
  ])
  next = find_next_automatable(plan.task_tree)
  ASSERT next.priority == HIGH AND next.automatable == true

TEST "ooda_cycle_produces_receipt":
  result = run_daily_cycle(entity, goals=[goal_1])
  ASSERT result.receipt IS NOT NONE
  ASSERT result.receipt.snr_score >= 0.85
```
