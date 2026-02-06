# Hooks & Automation Guide

Automate your workflow with event-driven hooks.

## Table of Contents

1. [What Are Hooks?](#what-are-hooks)
2. [Hook Types](#hook-types)
3. [Session Hooks](#session-hooks)
4. [Message Hooks](#message-hooks)
5. [Task Hooks](#task-hooks)
6. [Code Hooks](#code-hooks)
7. [Schedule Hooks](#schedule-hooks)
8. [Guardian Hooks](#guardian-hooks)
9. [Custom Hooks](#custom-hooks)
10. [Hook Conditions](#hook-conditions)
11. [Hook Actions](#hook-actions)
12. [Best Practices](#best-practices)

---

## What Are Hooks?

Hooks are event-driven automations that trigger actions based on system events.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           HOOK SYSTEM                                    │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   Event Occurs                                                          │
│        │                                                                │
│        ▼                                                                │
│   ┌─────────────┐                                                      │
│   │ Hook Engine │ ← Checks registered hooks                            │
│   └──────┬──────┘                                                      │
│          │                                                              │
│          ▼                                                              │
│   ┌─────────────┐                                                      │
│   │ Conditions  │ ← Evaluates if hook should fire                      │
│   └──────┬──────┘                                                      │
│          │ Match                                                        │
│          ▼                                                              │
│   ┌─────────────┐                                                      │
│   │   Actions   │ ← Executes hook actions                              │
│   └─────────────┘                                                      │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Configuration Location

```
~/.bizra/config/hooks.yaml
```

### Basic Hook Structure

```yaml
hooks:
  hook_name:
    event: "event_type"
    conditions:
      - type: "condition_type"
        value: "condition_value"
    actions:
      - type: "action_type"
        params:
          key: "value"
```

---

## Hook Types

| Category | Events | Purpose |
|----------|--------|---------|
| **Session** | start, end, idle | Lifecycle automation |
| **Message** | pre_send, post_receive | Message processing |
| **Task** | create, start, complete, blocked | Task workflow |
| **Code** | edit, commit, pr | Development workflow |
| **Schedule** | cron-based | Time-based automation |
| **Guardian** | review, alert | Safety enforcement |

---

## Session Hooks

Triggered during session lifecycle.

### on_start

Fires when BIZRA starts.

```yaml
session:
  on_start:
    actions:
      - type: "load_context"
        params:
          sources:
            - "recent_tasks"
            - "pending_items"
            - "calendar"
      - type: "greet"
        params:
          style: "morning_brief"
          include_alerts: true
```

### on_end

Fires when session ends.

```yaml
session:
  on_end:
    actions:
      - type: "summarize_session"
        params:
          store: true
          format: "brief"
      - type: "save_context"
        params:
          categories:
            - "decisions"
            - "learnings"
            - "pending"
```

### on_idle

Fires after period of inactivity.

```yaml
session:
  on_idle:
    after: "5m"  # Trigger after 5 minutes idle
    actions:
      - type: "suggest_tasks"
        params:
          based_on: ["goals", "recent_context", "time_of_day"]
          max_suggestions: 3
```

### on_resume

Fires when resuming from idle.

```yaml
session:
  on_resume:
    actions:
      - type: "show_context"
        params:
          what: "where_we_left_off"
```

---

## Message Hooks

Process messages before/after handling.

### pre_send

Before sending to LLM.

```yaml
message:
  pre_send:
    actions:
      - type: "enrich_context"
        params:
          add_recent_files: true
          add_relevant_memory: true
          max_context_tokens: 4000
      - type: "inject_persona"
        params:
          agent: "current"
          style: "professional"
```

### post_receive

After receiving LLM response.

```yaml
message:
  post_receive:
    actions:
      - type: "extract_tasks"
        params:
          auto_add: false
          suggest: true
      - type: "extract_learnings"
        params:
          store_if_significant: true
      - type: "check_fate_gates"
        params:
          block_on_fail: true
```

### on_error

When message processing fails.

```yaml
message:
  on_error:
    actions:
      - type: "retry"
        params:
          max_attempts: 3
          backoff: "exponential"
      - type: "fallback_agent"
        params:
          agent: "guardian"
```

---

## Task Hooks

Automate task workflow.

### on_create

When task is created.

```yaml
task:
  on_create:
    actions:
      - type: "auto_assign"
        params:
          rules:
            - pattern: "research|investigate|find"
              agent: "researcher"
            - pattern: "implement|build|code"
              agent: "developer"
            - pattern: "review|check|audit"
              agent: "reviewer"
            - pattern: "deploy|run|execute"
              agent: "executor"
      - type: "estimate_effort"
        params:
          store: true
      - type: "notify"
        params:
          if: "priority == 'high'"
```

### on_start

When task work begins.

```yaml
task:
  on_start:
    actions:
      - type: "load_context"
        params:
          include_related: true
          include_history: true
      - type: "check_dependencies"
        params:
          block_if_unmet: true
```

### on_complete

When task is marked done.

```yaml
task:
  on_complete:
    actions:
      - type: "extract_learnings"
        params:
          store: true
      - type: "suggest_next"
        params:
          based_on: "goals"
      - type: "update_progress"
        params:
          milestone: "current_quarter"
      - type: "celebrate"
        params:
          if: "is_milestone"
```

### on_blocked

When task is blocked.

```yaml
task:
  on_blocked:
    actions:
      - type: "analyze_blocker"
        params:
          suggest_resolution: true
      - type: "escalate"
        params:
          if: "blocked_duration > 1h"
          to: "guardian"
```

---

## Code Hooks

Development workflow automation.

### on_edit

When code is modified.

```yaml
code:
  on_edit:
    conditions:
      - type: "file_pattern"
        value: "*.rs"
    actions:
      - type: "lint"
        params:
          auto_fix: false
          show_warnings: true
      - type: "check_types"
        params:
          strict: true
```

### on_save

When file is saved.

```yaml
code:
  on_save:
    actions:
      - type: "format"
        params:
          tool: "rustfmt"
      - type: "run_tests"
        params:
          scope: "affected"
          timeout: "30s"
```

### on_commit

Before git commit.

```yaml
code:
  on_commit:
    actions:
      - type: "review"
        params:
          depth: "quick"
          focus: ["security", "logic"]
      - type: "generate_message"
        params:
          style: "conventional"
          include_scope: true
      - type: "guardian_check"
        params:
          require_approval: false
```

### on_pr

When creating pull request.

```yaml
code:
  on_pr:
    actions:
      - type: "full_review"
        params:
          include:
            - "security_audit"
            - "performance_check"
            - "test_coverage"
      - type: "generate_description"
        params:
          include_changes: true
          include_testing: true
      - type: "suggest_reviewers"
        params:
          based_on: "expertise"
```

### on_merge

After PR is merged.

```yaml
code:
  on_merge:
    actions:
      - type: "update_docs"
        params:
          if: "changes_include('api')"
      - type: "notify_deployment"
        params:
          channel: "team"
```

---

## Schedule Hooks

Time-based automation using cron syntax.

### Cron Syntax

```
┌───────────── minute (0 - 59)
│ ┌───────────── hour (0 - 23)
│ │ ┌───────────── day of month (1 - 31)
│ │ │ ┌───────────── month (1 - 12)
│ │ │ │ ┌───────────── day of week (0 - 6) (Sunday = 0)
│ │ │ │ │
* * * * *
```

### Examples

```yaml
schedule:
  morning_brief:
    cron: "0 8 * * *"       # 8:00 AM daily
    actions:
      - type: "run_command"
        params:
          command: "/morning"

  weekly_review:
    cron: "0 9 * * MON"     # 9:00 AM Mondays
    actions:
      - type: "run_command"
        params:
          command: "/weekly"

  hourly_sync:
    cron: "0 * * * *"       # Every hour
    actions:
      - type: "sync_memory"
        params:
          with: "federation"

  backup:
    cron: "0 2 * * *"       # 2:00 AM daily
    actions:
      - type: "backup"
        params:
          targets: ["memory", "config", "tasks"]
```

---

## Guardian Hooks

Safety and ethics enforcement.

### always_review

Patterns that always require Guardian review.

```yaml
guardian:
  always_review:
    patterns:
      - "delete.*data"
      - "production.*deploy"
      - "credentials|secrets|keys"
      - "sudo|rm -rf"
      - "financial.*transaction"
    actions:
      - type: "pause"
        params:
          require: "guardian_approval"
      - type: "log"
        params:
          level: "warn"
          category: "sensitive_action"
```

### anomaly_detection

Alert on unusual patterns.

```yaml
guardian:
  anomaly_detection:
    monitors:
      - type: "rate_limit"
        params:
          action: "exec"
          max_per_hour: 50
      - type: "unusual_time"
        params:
          alert_outside: "06:00-23:00"
      - type: "new_pattern"
        params:
          alert_on: "first_occurrence"
    actions:
      - type: "alert"
        params:
          severity: "medium"
      - type: "require_confirmation"
```

### escalation

When to escalate to human.

```yaml
guardian:
  escalation:
    triggers:
      - "harm_score > 0.25"
      - "confidence < 0.70"
      - "repeated_failures > 3"
    actions:
      - type: "pause"
      - type: "request_human_review"
        params:
          timeout: "1h"
          fallback: "safe_default"
```

---

## Custom Hooks

Create your own hooks.

### Basic Custom Hook

```yaml
custom:
  my_hook:
    event: "message.post_receive"
    conditions:
      - type: "content_matches"
        value: "TODO|FIXME"
    actions:
      - type: "create_task"
        params:
          title: "Address TODO from conversation"
          priority: "low"
```

### Complex Custom Hook

```yaml
custom:
  deployment_workflow:
    event: "code.on_pr"
    conditions:
      - type: "branch_matches"
        value: "release/*"
      - type: "all_checks_pass"
    actions:
      - type: "run_pipeline"
        params:
          steps:
            - name: "Build"
              command: "cargo build --release"
            - name: "Test"
              command: "cargo test"
            - name: "Security Scan"
              command: "/guardian review"
            - name: "Deploy Staging"
              command: "/exec deploy staging"
              require_approval: true
```

### Hook with Multiple Conditions

```yaml
custom:
  urgent_response:
    event: "message.post_receive"
    conditions:
      - type: "and"
        conditions:
          - type: "contains"
            value: "urgent"
          - type: "time_of_day"
            value: "09:00-17:00"
          - type: "not"
            condition:
              type: "agent_is"
              value: "guardian"
    actions:
      - type: "switch_agent"
        params:
          to: "executor"
          reason: "urgent_request"
      - type: "prioritize"
        params:
          level: "high"
```

---

## Hook Conditions

Available condition types.

### Content Conditions

```yaml
conditions:
  - type: "contains"           # Text contains
    value: "deploy"
  - type: "matches"            # Regex match
    value: "^/exec"
  - type: "content_matches"    # Message content
    value: "error|fail"
```

### Context Conditions

```yaml
conditions:
  - type: "agent_is"           # Current agent
    value: "developer"
  - type: "task_has_tag"       # Task tag
    value: "urgent"
  - type: "file_pattern"       # File being edited
    value: "*.rs"
  - type: "branch_matches"     # Git branch
    value: "main|release/*"
```

### Time Conditions

```yaml
conditions:
  - type: "time_of_day"        # Time range
    value: "09:00-17:00"
  - type: "day_of_week"        # Days
    value: "MON-FRI"
  - type: "is_deep_work"       # Deep work hours
```

### State Conditions

```yaml
conditions:
  - type: "idle_duration"      # How long idle
    value: "> 5m"
  - type: "session_length"     # Session duration
    value: "> 2h"
  - type: "fate_gate"          # FATE gate status
    gate: "harm"
    value: "> 0.20"
```

### Logic Conditions

```yaml
conditions:
  - type: "and"
    conditions: [...]
  - type: "or"
    conditions: [...]
  - type: "not"
    condition: {...}
```

---

## Hook Actions

Available action types.

### Agent Actions

```yaml
actions:
  - type: "switch_agent"
    params:
      to: "developer"
  - type: "invoke_agent"
    params:
      agent: "guardian"
      task: "review"
```

### Task Actions

```yaml
actions:
  - type: "create_task"
    params:
      title: "..."
      agent: "developer"
      priority: "high"
  - type: "update_task"
    params:
      status: "completed"
  - type: "suggest_tasks"
    params:
      max: 3
```

### Memory Actions

```yaml
actions:
  - type: "remember"
    params:
      content: "{{response}}"
      category: "learnings"
  - type: "recall"
    params:
      query: "related context"
  - type: "extract_learnings"
    params:
      store: true
```

### Code Actions

```yaml
actions:
  - type: "lint"
    params:
      fix: false
  - type: "format"
  - type: "run_tests"
    params:
      scope: "affected"
  - type: "review"
    params:
      depth: "thorough"
```

### Notification Actions

```yaml
actions:
  - type: "notify"
    params:
      message: "Task completed"
  - type: "alert"
    params:
      severity: "high"
  - type: "speak"  # Voice
    params:
      text: "Completed"
```

### Control Actions

```yaml
actions:
  - type: "pause"
    params:
      require: "approval"
  - type: "retry"
    params:
      max: 3
  - type: "fallback"
    params:
      action: "safe_default"
  - type: "run_command"
    params:
      command: "/guardian status"
```

---

## Best Practices

### 1. Start Simple

```yaml
# Good: Simple, focused hook
task:
  on_complete:
    actions:
      - type: "suggest_next"

# Avoid: Overly complex hook
task:
  on_complete:
    conditions:
      - # Many conditions...
    actions:
      - # Many actions...
```

### 2. Use Conditions Wisely

```yaml
# Good: Targeted condition
code:
  on_commit:
    conditions:
      - type: "file_pattern"
        value: "src/**/*.rs"
    actions:
      - type: "run_tests"

# Avoid: Too broad
code:
  on_commit:
    actions:
      - type: "run_all_tests"  # Slow, unnecessary
```

### 3. Don't Over-Automate

```yaml
# Good: Helpful automation
message:
  post_receive:
    actions:
      - type: "extract_tasks"
        params:
          auto_add: false      # Suggest, don't auto-add
          suggest: true

# Avoid: Intrusive automation
message:
  post_receive:
    actions:
      - type: "extract_tasks"
        params:
          auto_add: true       # Creates tasks without asking
```

### 4. Guardian Integration

```yaml
# Always include Guardian for sensitive operations
code:
  on_pr:
    conditions:
      - type: "branch_matches"
        value: "main"
    actions:
      - type: "guardian_review"  # Safety check
        params:
          required: true
      - type: "deploy"
```

### 5. Test Your Hooks

```bash
# Dry-run a hook
/hook test on_commit --dry-run

# View hook logs
/hook logs --recent 10

# Disable problematic hook
/hook disable my_hook
```

---

## Hook Management Commands

```bash
# List all hooks
/hook list

# View hook details
/hook show on_commit

# Enable/disable hook
/hook enable my_hook
/hook disable my_hook

# Test hook
/hook test on_commit --simulate

# View hook logs
/hook logs
/hook logs --hook on_commit
```

---

## Example: Complete Workflow

```yaml
# Complete development workflow
session:
  on_start:
    actions:
      - type: "load_context"
      - type: "greet"

task:
  on_create:
    actions:
      - type: "auto_assign"
  on_complete:
    actions:
      - type: "suggest_next"

code:
  on_edit:
    conditions:
      - type: "file_pattern"
        value: "*.rs"
    actions:
      - type: "lint"
  on_commit:
    actions:
      - type: "review"
        params:
          depth: "quick"
      - type: "generate_message"
  on_pr:
    conditions:
      - type: "branch_matches"
        value: "main"
    actions:
      - type: "full_review"
      - type: "guardian_check"

guardian:
  always_review:
    patterns:
      - "deploy.*production"

schedule:
  morning_brief:
    cron: "0 8 * * *"
    actions:
      - type: "run_command"
        params:
          command: "/morning"
```

---

## Next Steps

- [Skills System](10-SKILLS-SYSTEM.md) — Multi-step workflows
- [Proactive System](11-PROACTIVE-SYSTEM.md) — Anticipation engine
- [Config Reference](../reference/CONFIG-REFERENCE.md) — All configuration options

---

**Automate the routine, focus on what matters.** ⚡
