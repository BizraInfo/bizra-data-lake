# Proactive System Guide

Your anticipation engine — BIZRA that thinks ahead.

## Table of Contents

1. [What Is Proactive Mode?](#what-is-proactive-mode)
2. [Proactive Modes](#proactive-modes)
3. [Context Awareness](#context-awareness)
4. [Anticipation Engine](#anticipation-engine)
5. [Briefings](#briefings)
6. [Suggestions](#suggestions)
7. [Pattern Learning](#pattern-learning)
8. [Think Tank Mode](#think-tank-mode)
9. [Configuration](#configuration)
10. [Best Practices](#best-practices)

---

## What Is Proactive Mode?

The Proactive System transforms BIZRA from a reactive tool into an anticipatory partner.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        PROACTIVE SYSTEM                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│   ┌───────────────┐                                                    │
│   │    Context    │ ← Gathers: Time, Tasks, Goals, Patterns            │
│   │   Collector   │                                                    │
│   └───────┬───────┘                                                    │
│           │                                                             │
│           ▼                                                             │
│   ┌───────────────┐                                                    │
│   │  Anticipation │ ← Predicts: Needs, Blockers, Opportunities         │
│   │    Engine     │                                                    │
│   └───────┬───────┘                                                    │
│           │                                                             │
│           ▼                                                             │
│   ┌───────────────┐                                                    │
│   │   Suggestion  │ ← Generates: Tasks, Research, Actions              │
│   │   Generator   │                                                    │
│   └───────┬───────┘                                                    │
│           │                                                             │
│           ▼                                                             │
│   ┌───────────────┐                                                    │
│   │    Pattern    │ ← Learns: Your habits, preferences, rhythms        │
│   │    Learner    │                                                    │
│   └───────────────┘                                                    │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Key Principles

1. **Anticipate, don't interrupt** — Suggestions appear when helpful
2. **Learn continuously** — Adapts to your patterns
3. **Respect focus time** — Knows when to be quiet
4. **Align with goals** — Every suggestion ties to your objectives

---

## Proactive Modes

Four modes balance helpfulness with focus.

### Silent Mode

```yaml
proactive:
  mode: "silent"
```

| Feature | Enabled |
|---------|---------|
| Suggestions | ✗ |
| Background research | ✗ |
| Alerts | Critical only |
| Briefings | ✗ |

**Best for:** Deep focus work, minimal distractions.

### Minimal Mode

```yaml
proactive:
  mode: "minimal"
```

| Feature | Enabled |
|---------|---------|
| Suggestions | ✗ |
| Background research | ✗ |
| Alerts | ✓ |
| Briefings | ✓ |

**Best for:** Focused work with essential notifications.

### Balanced Mode (Default)

```yaml
proactive:
  mode: "balanced"
```

| Feature | Enabled |
|---------|---------|
| Suggestions | ✓ (contextual) |
| Background research | On-topic only |
| Alerts | ✓ |
| Briefings | ✓ |

**Best for:** Normal work with helpful assistance.

### Active Mode

```yaml
proactive:
  mode: "active"
```

| Feature | Enabled |
|---------|---------|
| Suggestions | ✓ (proactive) |
| Background research | ✓ (anticipatory) |
| Alerts | ✓ |
| Briefings | ✓ + Predictive |

**Best for:** Exploration, learning, maximum assistance.

### Mode Commands

```bash
/proactive silent     # Switch to silent
/proactive minimal    # Switch to minimal
/proactive balanced   # Switch to balanced
/proactive active     # Switch to active
/proactive status     # Show current mode
```

---

## Context Awareness

The system maintains continuous awareness of your context.

### Context Sources

```yaml
context_awareness:
  sources:
    time:
      - current_time
      - day_of_week
      - time_of_day_category     # morning/afternoon/evening
      - is_deep_work_time
      - is_meeting_time

    tasks:
      - pending_tasks
      - blocked_tasks
      - recent_completions
      - upcoming_deadlines

    goals:
      - quarterly_objectives
      - weekly_targets
      - daily_priorities

    activity:
      - current_agent
      - recent_commands
      - files_being_edited
      - conversation_topic

    external:
      - calendar_events
      - unread_notifications
      - system_alerts

    patterns:
      - typical_activities_now
      - productivity_level
      - energy_estimation
```

### Context Window

```yaml
context_awareness:
  window:
    immediate: "5m"        # Last 5 minutes (high detail)
    recent: "1h"           # Last hour (moderate detail)
    session: "full"        # Full session (summaries)
    historical: "7d"       # Week patterns (trends)
```

### Context Display

```bash
/context                  # Show current context
/context --detail high    # Detailed context view
```

**Output:**
```
📍 Current Context

Time: 14:32 (Afternoon deep work block)
Agent: Developer
Energy: High (based on recent activity)

📋 Active Context:
- Editing: src/consensus.rs
- Topic: Byzantine fault tolerance
- Recent: Implemented voting mechanism

🎯 Relevant Goals:
- Q1: Federation protocol operational
- This week: Complete PBFT implementation

💡 Anticipations:
- May need: Research on network partitions
- Upcoming: Team sync in 1.5 hours
```

---

## Anticipation Engine

Predicts what you'll need before you ask.

### Prediction Types

```yaml
anticipation:
  predictions:
    needs:
      description: "What resources you'll need"
      signals:
        - "discussing topic without recent research"
        - "starting implementation without tests"
        - "approaching deadline without progress"

    blockers:
      description: "What might block progress"
      signals:
        - "dependency on incomplete task"
        - "missing information"
        - "external approval needed"

    opportunities:
      description: "Chances for improvement"
      signals:
        - "related task could be combined"
        - "recent learning applies here"
        - "similar problem solved before"

    transitions:
      description: "Context switches coming"
      signals:
        - "approaching meeting time"
        - "energy level changing"
        - "completing a milestone"
```

### Anticipation Actions

```yaml
anticipation:
  actions:
    pre_research:
      trigger: "discussing unfamiliar topic"
      action: "background research"
      condition: "mode >= balanced"

    dependency_alert:
      trigger: "task depends on blocked item"
      action: "alert with resolution options"
      condition: "always"

    transition_prep:
      trigger: "meeting in 15 minutes"
      action: "prepare context summary"
      condition: "mode >= minimal"

    opportunity_flag:
      trigger: "similar pattern detected"
      action: "suggest connection"
      condition: "mode == active"
```

### Anticipation Commands

```bash
/anticipate             # Show current anticipations
/anticipate --next 2h   # Next 2 hours predictions
```

---

## Briefings

Structured updates at key moments.

### Morning Brief

Triggered at your configured morning time.

```yaml
briefings:
  morning:
    time: "08:00"
    includes:
      - greeting (with Arabic phrase)
      - overnight_alerts
      - today_calendar
      - priority_tasks
      - goal_progress
      - energy_suggestion
    format: "friendly"
```

**Example:**
```
☀️ صباح الخير (Good morning), MoMo!

📅 Wednesday, February 5, 2026

⚠️ Overnight Alerts:
- None (all clear!)

📋 Today's Priorities:
1. Complete PBFT implementation (deadline: Friday)
2. Review PR #42 (blocking: Alice)
3. Team sync at 14:00

📊 Goal Progress:
- Q1 Milestone: 67% complete
- This week: 3/5 tasks done

💡 Suggestion: Start with PBFT while energy is high.
   Consider the research from yesterday on network partitions.
```

### Daily Standup

Quick status format.

```yaml
briefings:
  standup:
    trigger: "on_demand"
    includes:
      - yesterday_completed
      - today_planned
      - blockers
    format: "concise"
```

**Example:**
```
📊 Standup Report

✅ Yesterday:
- Implemented voting mechanism
- Reviewed 3 PRs
- Fixed authentication bug

📋 Today:
- Complete PBFT implementation
- Write federation tests

🚧 Blockers:
- Waiting for API spec from backend team
```

### Evening Review

End-of-day summary.

```yaml
briefings:
  evening:
    time: "20:00"
    includes:
      - accomplishments
      - learnings
      - tomorrow_preview
      - reflection_prompt
    format: "reflective"
```

**Example:**
```
🌙 مساء الخير (Good evening), MoMo!

✅ Today's Accomplishments:
- Completed PBFT implementation ✓
- Reviewed 2 PRs
- Resolved 1 blocker

📚 Learnings Captured:
- Network partition handling patterns
- Testing strategies for distributed systems

📋 Tomorrow Preview:
- Federation integration tests
- Documentation update

💭 Reflection:
Good progress on the milestone. Consider front-loading
the documentation tomorrow to balance the week.
```

### Weekly Review

Comprehensive weekly analysis.

```yaml
briefings:
  weekly:
    day: "friday"
    time: "17:00"
    includes:
      - week_summary
      - goal_progress
      - metrics_analysis
      - learnings
      - next_week_plan
      - retrospective
    format: "comprehensive"
```

### Briefing Commands

```bash
/morning              # Morning briefing
/standup              # Quick standup
/daily-review         # Evening review
/weekly               # Weekly review
/briefing custom      # Custom briefing
```

---

## Suggestions

Context-aware task and action recommendations.

### Suggestion Types

```yaml
suggestions:
  types:
    next_task:
      description: "What to work on next"
      priority: ["goals", "deadlines", "energy", "context"]

    research:
      description: "Topics to explore"
      triggers: ["unfamiliar_term", "blocked_by_unknown"]

    optimization:
      description: "Ways to improve"
      triggers: ["repeated_pattern", "inefficiency_detected"]

    connection:
      description: "Related information"
      triggers: ["similar_past_work", "relevant_memory"]

    break:
      description: "Rest suggestions"
      triggers: ["long_session", "declining_performance"]
```

### Suggestion Delivery

```yaml
suggestions:
  delivery:
    timing:
      idle_threshold: "5m"         # Suggest after 5m idle
      completion_moment: true      # After task completion
      context_switch: true         # When changing focus

    presentation:
      max_suggestions: 3           # Max at once
      style: "non-intrusive"       # Don't interrupt flow
      dismissable: true            # Easy to dismiss

    persistence:
      remember_dismissed: true     # Don't repeat dismissed
      decay_time: "24h"            # Re-suggest after 24h if relevant
```

### Example Suggestions

**After completing a task:**
```
💡 Great work completing the authentication module!

Suggestions:
1. 📝 Write tests for the new auth endpoints (high priority)
2. 📖 Document the authentication flow (goal-aligned)
3. 🔍 Review related: "Authorization patterns you researched last week"

[1] [2] [3] [Dismiss]
```

**During idle:**
```
💡 Based on your goals and current context:

1. 🎯 Continue PBFT implementation (67% → deadline Friday)
2. 📧 Review PR #42 (Alice is waiting)
3. 📚 Research: "Byzantine fault tolerance edge cases"

[1] [2] [3] [Later]
```

### Suggestion Commands

```bash
/suggest              # Get suggestions now
/suggest --goal       # Goal-aligned suggestions
/suggest --context    # Context-based suggestions
/suggest --dismiss 1  # Dismiss suggestion
```

---

## Pattern Learning

The system learns your work patterns.

### What It Learns

```yaml
pattern_learning:
  categories:
    temporal:
      - peak_productivity_hours
      - typical_break_times
      - meeting_patterns
      - task_duration_estimates

    behavioral:
      - preferred_task_order
      - research_before_implement
      - review_frequency
      - collaboration_patterns

    preferences:
      - communication_style
      - detail_level_preference
      - agent_preferences
      - tool_preferences

    performance:
      - task_completion_rates
      - estimation_accuracy
      - focus_duration
      - context_switch_impact
```

### Learning Configuration

```yaml
pattern_learning:
  enabled: true

  sensitivity:
    min_occurrences: 3           # Need 3+ occurrences to learn
    confidence_threshold: 0.7    # 70% confidence to apply

  adaptation:
    gradual: true                # Apply learnings gradually
    confirm_major: true          # Confirm major pattern changes

  privacy:
    store_locally: true          # Never send patterns externally
    anonymize_content: true      # Learn patterns, not content
```

### Pattern Commands

```bash
/patterns             # Show learned patterns
/patterns --category temporal
/patterns --reset     # Reset all patterns (requires confirm)
/patterns --correct   # Correct a learned pattern
```

**Example Output:**
```
📊 Learned Patterns

⏰ Temporal:
- Peak productivity: 09:00-11:00, 15:00-17:00
- Typical breaks: 11:00, 14:00
- Average focus session: 47 minutes

🔄 Behavioral:
- Usually research before implementing (85% confidence)
- Prefer reviewing code in morning (72% confidence)
- Write tests after implementation (91% confidence)

⚙️ Preferences:
- Detailed explanations preferred (78% confidence)
- Frequent Guardian consultations (90% confidence)
- Developer agent for code tasks (95% confidence)
```

---

## Think Tank Mode

Transform BIZRA into your personal think tank.

### What Is Think Tank Mode?

Think Tank mode engages multiple agents collaboratively on complex problems.

```yaml
think_tank:
  description: "Multi-agent collaborative problem solving"

  composition:
    core_team:
      - strategist    # Strategic perspective
      - researcher    # Knowledge and facts
      - developer     # Technical feasibility
      - analyst       # Data and metrics
      - guardian      # Ethics and safety

  process:
    1. frame_problem       # Strategist frames the problem
    2. gather_context      # Researcher gathers information
    3. analyze_data        # Analyst provides metrics
    4. propose_solutions   # All agents contribute
    5. evaluate_options    # Structured comparison
    6. guardian_review     # Ethics check
    7. synthesize_decision # Final recommendation
```

### Activating Think Tank

```bash
/think-tank "Should we open source BIZRA?"
/think-tank "Best architecture for scaling to 1M users"
```

### Think Tank Process

**Step 1: Problem Framing (Strategist)**
```
♟ Strategist Analysis:

Question: "Should we open source BIZRA?"

Key Dimensions:
1. Strategic: Market positioning, competitive advantage
2. Technical: Code quality, security exposure
3. Community: Ecosystem growth, contribution model
4. Business: Revenue impact, support costs
5. Risk: IP protection, reputation

Stakeholders: Users, developers, investors, community
Timeline: Decision needed by Q2
```

**Step 2: Research (Researcher)**
```
🔍 Research Findings:

Precedents Analyzed:
- Similar projects that open-sourced: Results varied
- Market analysis: Open source AI tools growing 40% YoY
- Community sentiment: Strong demand for transparency

Key Data Points:
- 73% of enterprise buyers prefer open source
- Average contribution rate: 2-5% of users
- Support cost increase: 20-40% typical
```

**Step 3: Analysis (Analyst)**
```
📊 Quantitative Analysis:

Revenue Impact Scenarios:
- Conservative: -15% short-term, +25% long-term
- Moderate: -5% short-term, +40% long-term
- Optimistic: +10% short-term, +80% long-term

Community Growth Projection:
- Year 1: 5,000 → 50,000 users
- Year 3: 50,000 → 500,000 users

Cost Analysis:
- Documentation: $50K
- Community management: $80K/year
- Security audits: $30K/year
```

**Step 4: Technical Perspective (Developer)**
```
⚙️ Technical Assessment:

Code Readiness:
- Documentation coverage: 65% (needs improvement)
- Test coverage: 82% (acceptable)
- Security audit status: Pending

Recommended Preparations:
1. Security audit before release
2. API documentation completion
3. Contribution guidelines
4. CI/CD for community PRs
```

**Step 5: Guardian Review**
```
🛡 Guardian Assessment:

FATE Gates Analysis:
- Ihsān (Excellence): Code quality must meet public standards
- Adl (Justice): Ensure equitable access
- Harm (Prevention): Security vulnerabilities exposure risk
- Confidence (Certainty): Decision confidence: 0.78

Ethical Considerations:
- Positive: Democratizes access to sovereign AI
- Concern: Potential misuse without guardrails
- Mitigation: Include FATE gates in open source version

Recommendation: PROCEED with conditions
```

**Step 6: Synthesis**
```
📋 Think Tank Recommendation

Decision: PROCEED with Open Source (Conditional)

Conditions:
1. Complete security audit first
2. Document FATE gates thoroughly
3. Establish community governance
4. Phase release over 6 months

Confidence: 78%
Dissent: Developer notes timeline risk

Next Steps:
1. Commission security audit (Week 1-4)
2. Documentation sprint (Week 5-8)
3. Private beta with select users (Week 9-12)
4. Public release (Q2)
```

### Think Tank Configuration

```yaml
think_tank:
  settings:
    require_unanimous: false     # Majority sufficient
    include_dissent: true        # Show minority views
    time_limit: "30m"            # Max deliberation time
    depth: "thorough"            # Analysis depth

  output:
    format: "structured"
    include:
      - problem_frame
      - research_summary
      - quantitative_analysis
      - technical_assessment
      - ethical_review
      - recommendation
      - dissent
      - next_steps
```

---

## Configuration

Complete proactive configuration reference.

### Full Configuration

```yaml
proactive:
  # Mode setting
  mode: "balanced"    # silent | minimal | balanced | active

  # Context awareness
  context:
    update_interval: "30s"
    sources:
      - time
      - tasks
      - goals
      - activity
      - calendar
    window:
      immediate: "5m"
      recent: "1h"
      session: "full"

  # Anticipation
  anticipation:
    enabled: true
    predictions:
      - needs
      - blockers
      - opportunities
      - transitions
    confidence_threshold: 0.6

  # Suggestions
  suggestions:
    enabled: true
    timing:
      idle_threshold: "5m"
      on_completion: true
    max_suggestions: 3
    persist_dismissed: true

  # Briefings
  briefings:
    morning:
      enabled: true
      time: "08:00"
    evening:
      enabled: true
      time: "20:00"
    weekly:
      enabled: true
      day: "friday"
      time: "17:00"

  # Pattern learning
  patterns:
    enabled: true
    categories:
      - temporal
      - behavioral
      - preferences
    min_occurrences: 3
    confirm_major_changes: true

  # Think tank
  think_tank:
    enabled: true
    require_unanimous: false
    time_limit: "30m"
```

---

## Best Practices

### 1. Start with Balanced Mode

```yaml
# Good: Start balanced, adjust as needed
proactive:
  mode: "balanced"

# Avoid: Starting with active (too noisy)
proactive:
  mode: "active"
```

### 2. Use Focus Time Wisely

```yaml
# Good: Respect deep work
patterns:
  schedule:
    deep_work_hours: ["09:00-12:00", "15:00-18:00"]

proactive:
  respect_deep_work: true    # Go silent during deep work
```

### 3. Let It Learn

Give the system time to learn your patterns (1-2 weeks) before judging effectiveness.

### 4. Provide Feedback

```bash
/feedback suggestion good     # Helpful suggestion
/feedback suggestion poor     # Not helpful
/feedback pattern wrong       # Incorrect pattern detected
```

### 5. Use Think Tank for Big Decisions

```bash
# Good: Strategic decisions
/think-tank "Should we pivot our architecture?"

# Avoid: Simple questions
/think-tank "What color should the button be?"
```

---

## Commands Reference

| Command | Description |
|---------|-------------|
| `/proactive <mode>` | Set proactive mode |
| `/proactive status` | Show current status |
| `/context` | Show current context |
| `/anticipate` | Show anticipations |
| `/suggest` | Get suggestions |
| `/morning` | Morning briefing |
| `/standup` | Quick standup |
| `/daily-review` | Evening review |
| `/weekly` | Weekly review |
| `/patterns` | Show learned patterns |
| `/think-tank "<question>"` | Engage think tank |
| `/feedback` | Provide feedback |

---

## Next Steps

- [Config Reference](../reference/CONFIG-REFERENCE.md) — Full configuration options
- [Skills System](10-SKILLS-SYSTEM.md) — Multi-step workflows
- [PAT Agents](04-PAT-AGENTS.md) — Agent capabilities

---

**Anticipate. Adapt. Excel.** 🎯
