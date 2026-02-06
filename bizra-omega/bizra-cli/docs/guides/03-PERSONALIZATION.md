# Personalization Guide

Make BIZRA truly yours by configuring your sovereign profile.

## Table of Contents

1. [The Sovereign Profile](#the-sovereign-profile)
2. [Identity Configuration](#identity-configuration)
3. [Values & Principles](#values--principles)
4. [Working Patterns](#working-patterns)
5. [Goals & Objectives](#goals--objectives)
6. [PAT Team Customization](#pat-team-customization)
7. [Proactive Behavior](#proactive-behavior)
8. [Memory & Learning](#memory--learning)

---

## The Sovereign Profile

Your profile lives at:
```
~/.bizra/config/sovereign_profile.yaml
```

This file defines **who you are** to the system. Your PAT team learns from this to serve you better.

### Profile Structure

```yaml
identity:       # Who you are
values:         # What you believe
patterns:       # How you work
domains:        # What you know
goals:          # What you're building toward
pat_team:       # Your agent configuration
proactive:      # Anticipation settings
memory:         # What the system remembers
integrations:   # External connections
```

---

## Identity Configuration

### Basic Identity

```yaml
identity:
  name: "MoMo (محمد)"
  title: "Sovereign Node0 Architect"
  location: "Dubai, UAE"
  timezone: "GMT+4"
  languages:
    - code: "ar"
      name: "Arabic"
      proficiency: "native"
    - code: "en"
      name: "English"
      proficiency: "fluent"
```

### Genesis Information

```yaml
identity:
  genesis:
    hash: "a7f68f1f74f2c0898cb1f1db6e83633674f17ee1c0161704ac8d85f8a773c25b"
    timestamp: "2024-01-01T00:00:00Z"
    node_id: "node0_ce5af35c848ce889"
```

This is your node's unique identity on the BIZRA network.

### What This Affects

| Setting | Effect |
|---------|--------|
| `name` | How agents address you |
| `timezone` | When briefings occur |
| `languages` | Response language mixing |
| `genesis.hash` | Your cryptographic identity |

---

## Values & Principles

### FATE Gate Thresholds

```yaml
values:
  fate_gates:
    ihsan_threshold: 0.95      # Excellence minimum
    adl_gini_max: 0.35         # Fairness maximum
    harm_threshold: 0.30       # Harm maximum
    confidence_min: 0.80       # Confidence minimum
```

**Customization Options:**

| Threshold | Conservative | Balanced | Aggressive |
|-----------|-------------|----------|------------|
| Ihsān | 0.98 | 0.95 | 0.90 |
| Adl | 0.25 | 0.35 | 0.45 |
| Harm | 0.20 | 0.30 | 0.40 |
| Confidence | 0.90 | 0.80 | 0.70 |

### Guiding Principles

```yaml
values:
  principles:
    - "Excellence (Ihsān) is not optional — it's the minimum"
    - "Every human is a node. Every node is a seed."
    - "Stand on the shoulders of giants"
    - "Sovereignty through decentralization"
    - "Privacy is a human right"
    - "AI serves humanity, not the reverse"
```

These principles guide agent decision-making.

### Your Giants

```yaml
values:
  giants:
    philosophy: ["Al-Ghazali", "Ibn Rushd", "Rawls"]
    technology: ["Shannon", "Turing", "Lamport", "Torvalds"]
    strategy: ["Sun Tzu", "Clausewitz", "Porter"]
    design: ["Tufte", "Dieter Rams", "Jony Ive"]
```

Add your own inspirations. Agents will reference these in responses.

---

## Working Patterns

### Daily Schedule

```yaml
patterns:
  schedule:
    deep_work_hours: ["09:00-12:00", "15:00-18:00"]
    review_time: "20:00"
    planning_time: "08:00"
    offline_hours: ["00:00-06:00"]
```

**Effects:**
- During deep work: Minimal interruptions
- During offline: No notifications
- At planning time: Morning brief available

### Communication Style

```yaml
patterns:
  communication:
    preferred_style: "concise"          # concise | detailed | visual
    response_format: "structured"       # structured | narrative | bullet
    language_style: "technical"         # technical | casual | formal
    include_arabic: true                # Mix Arabic phrases
```

### Decision Making

```yaml
patterns:
  decision_style:
    risk_tolerance: "calculated"        # conservative | calculated | aggressive
    speed_vs_quality: "quality"         # speed | balanced | quality
    autonomy_level: "high"              # low | medium | high
    requires_confirmation:
      - "financial_transactions"
      - "public_communications"
      - "irreversible_actions"
```

**Autonomy Levels:**

| Level | Behavior |
|-------|----------|
| `low` | Confirms almost everything |
| `medium` | Confirms significant actions |
| `high` | Acts autonomously (Guardian still reviews) |

---

## Goals & Objectives

### Vision

```yaml
goals:
  vision: "8 billion nodes — every human sovereign over their digital life"
```

### Current Quarter OKRs

```yaml
goals:
  current_quarter:
    - objective: "Launch BIZRA Node0 production"
      key_results:
        - "CLI/TUI fully operational"
        - "PAT team integrated with LM Studio"
        - "Voice interface working"
        - "Federation protocol tested"

    - objective: "Establish resource pool foundation"
      key_results:
        - "5 pillars implemented"
        - "Compute commons operational"
        - "First mudarabah partnership"
```

### Milestones

```yaml
goals:
  milestones:
    - date: "2026-Q1"
      target: "Node0 production-ready"
    - date: "2026-Q2"
      target: "First 10 federated nodes"
    - date: "2026-Q4"
      target: "1000 active nodes"
```

**Effects:**
- Proactive system suggests goal-aligned tasks
- Weekly reviews track milestone progress
- Strategist agent focuses on these objectives

---

## PAT Team Customization

### Default Agent

```yaml
pat_team:
  default_agent: "guardian"
```

### Agent-Specific Configuration

```yaml
pat_team:
  agents:
    guardian:
      voice: "NATF3.pt"                    # Voice prompt file
      personality: "protective, ethical, watchful"
      specialties: ["ethics review", "risk monitoring", "anomaly detection"]
      auto_engage_on:
        - "ethical concerns"
        - "unusual patterns"
        - "high-stakes decisions"
      always_consulted: true               # Reviews all major decisions
```

### Auto-Engagement Rules

Each agent can automatically engage when certain topics arise:

```yaml
pat_team:
  agents:
    developer:
      auto_engage_on:
        - "code tasks"
        - "architecture decisions"
        - "performance issues"
```

---

## Proactive Behavior

### Proactive Mode

```yaml
proactive:
  enabled: true
  mode: "balanced"  # silent | minimal | balanced | active
```

| Mode | Suggestions | Research | Alerts | Briefings |
|------|-------------|----------|--------|-----------|
| `silent` | ✗ | ✗ | Critical only | ✗ |
| `minimal` | ✗ | ✗ | ✓ | ✓ |
| `balanced` | ✓ | On topic | ✓ | ✓ |
| `active` | ✓ | ✓ | ✓ | ✓ + Predictive |

### Triggers

```yaml
proactive:
  triggers:
    morning_brief:
      time: "08:00"
      includes:
        - "overnight alerts"
        - "today's priorities"
        - "calendar summary"

    suggest_tasks:
      when: "idle > 5min"
      based_on: ["goals", "recent_context", "pending_items"]
```

---

## Memory & Learning

### Session Memory

```yaml
memory:
  session:
    max_context_tokens: 100000
    summarize_after: 50000
```

### Persistent Memory

```yaml
memory:
  persistent:
    store_path: "~/.bizra/memory"
    categories:
      - "decisions"           # Major decisions and rationale
      - "learnings"           # Lessons learned
      - "preferences"         # Discovered preferences
      - "patterns"            # Behavioral patterns
      - "contacts"            # People and relationships
      - "projects"            # Project states

    retention:
      decisions: "permanent"
      learnings: "permanent"
      preferences: "permanent"
      patterns: "1 year"
```

---

## Quick Customization Recipes

### For Deep Focus Work

```yaml
patterns:
  schedule:
    deep_work_hours: ["08:00-12:00", "14:00-18:00"]
proactive:
  mode: "minimal"
```

### For High Autonomy

```yaml
patterns:
  decision_style:
    autonomy_level: "high"
    requires_confirmation: []
proactive:
  mode: "active"
```

### For Maximum Safety

```yaml
values:
  fate_gates:
    ihsan_threshold: 0.98
    harm_threshold: 0.20
patterns:
  decision_style:
    autonomy_level: "low"
```

---

## Applying Changes

After editing your profile:

```bash
# Restart CLI to apply
bizra

# Or reload in TUI
/reload config
```

---

## Next Steps

- [PAT Agents Guide](04-PAT-AGENTS.md) — Deep dive into your team
- [Commands Reference](05-SLASH-COMMANDS.md) — All available commands
- [Hooks Guide](09-HOOKS-AUTOMATION.md) — Automate your workflow

---

**Your node, your rules, your sovereignty.** 🌟
