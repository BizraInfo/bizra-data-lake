# Slash Commands Guide

Master the command-line interface to unlock the full power of your PAT team.

## Table of Contents

1. [Command Basics](#command-basics)
2. [Agent Commands](#agent-commands)
3. [Task Commands](#task-commands)
4. [Research Commands](#research-commands)
5. [Code Commands](#code-commands)
6. [Analysis Commands](#analysis-commands)
7. [Strategy Commands](#strategy-commands)
8. [Execution Commands](#execution-commands)
9. [Guardian Commands](#guardian-commands)
10. [Memory Commands](#memory-commands)
11. [Voice Commands](#voice-commands)
12. [Proactive Commands](#proactive-commands)
13. [Quick Commands](#quick-commands)
14. [Modifiers](#modifiers)
15. [Aliases](#aliases)
16. [Advanced Usage](#advanced-usage)

---

## Command Basics

### Syntax

All commands follow this pattern:

```
/<command> [subcommand] [arguments] [--modifiers]
```

### Components

| Component | Required | Description |
|-----------|----------|-------------|
| `/` | Yes | Command prefix |
| `command` | Yes | Main action |
| `subcommand` | Sometimes | Specific action |
| `arguments` | Varies | Input data |
| `--modifiers` | No | Behavior flags |

### Examples

```bash
/agent list                          # Simple command
/task add "My task"                  # Command with argument
/research deep "topic" --detailed    # With modifier
/a guardian                          # Using alias
```

---

## Agent Commands

Control your PAT team.

### `/agent list`

Show all agents and their status.

```bash
/agent list
# Alias: /agents
```

**Output:**
```
PAT Agents:
♟ Strategist  │ Ready │ Sun Tzu • Clausewitz • Porter
🔍 Researcher  │ Ready │ Shannon • Turing • Dijkstra
⚙ Developer   │ Ready │ Knuth • Ritchie • Torvalds
📊 Analyst     │ Ready │ Tukey • Tufte • Cleveland
✓ Reviewer    │ Ready │ Fagan • Parnas • Brooks
▶ Executor    │ Ready │ Toyota • Deming • Ohno
🛡 Guardian    │ Ready │ Al-Ghazali • Rawls • Anthropic
```

### `/agent switch <name>`

Change active agent.

```bash
/agent switch developer
/a developer              # Short form
/a dev                    # Even shorter
```

### `/agent status [name]`

Get detailed agent info.

```bash
/agent status             # Current agent
/agent status guardian    # Specific agent
```

**Output:**
```
🛡 Guardian
├─ Status: Active
├─ Giants: Al-Ghazali • Rawls • Anthropic
├─ Specialties: Ethics, Safety, FATE enforcement
├─ Current Task: None
├─ Tasks Completed: 23
└─ FATE Status:
   ├─ Ihsān:      0.97 / 0.95 ✓
   ├─ Adl:        0.28 / 0.35 ✓
   ├─ Harm:       0.12 / 0.30 ✓
   └─ Confidence: 0.91 / 0.80 ✓
```

---

## Task Commands

Manage your work.

### `/task add <title>`

Create a new task.

```bash
/task add "Implement authentication"
/task add "Review PR #42" --agent reviewer
/task add "Urgent fix" --priority high
/t "Quick task"           # Alias
```

**Options:**

| Option | Short | Description |
|--------|-------|-------------|
| `--agent` | `-a` | Assign to agent |
| `--description` | `-d` | Add details |
| `--priority` | `-p` | low/normal/high |
| `--due` | | Due date |
| `--tags` | | Comma-separated tags |

### `/task list [filter]`

View tasks.

```bash
/task list                # All tasks
/task list pending        # Only pending
/task list completed      # Only completed
/task list --agent dev    # By agent
/task list --tag backend  # By tag
```

### `/task show <id>`

View task details.

```bash
/task show 42
/task show "task-uuid"
```

### `/task done <id>`

Mark complete.

```bash
/task done 42
/task done 42 --notes "Completed with tests"
```

### `/task delegate <id> <agent>`

Reassign task.

```bash
/task delegate 42 developer
/task delegate 42 dev     # Short name works
```

### `/task update <id>`

Modify task.

```bash
/task update 42 --priority high
/task update 42 --description "New details"
/task update 42 --due "tomorrow"
```

### `/task delete <id>`

Remove task (requires confirmation).

```bash
/task delete 42
/task delete 42 --force   # Skip confirmation
```

---

## Research Commands

Discover and synthesize knowledge.

### `/research deep <topic>`

Comprehensive multi-source research.

```bash
/research deep "quantum computing applications"
/research deep "Rust async patterns" --sources academic
/research deep "market trends" --depth thorough
```

**Options:**

| Option | Values | Description |
|--------|--------|-------------|
| `--sources` | web, academic, memory, all | Limit sources |
| `--depth` | summary, moderate, deep | Research depth |
| `--format` | markdown, json, outline | Output format |
| `--remember` | | Save to memory |

### `/research quick <query>`

Fast fact lookup.

```bash
/research quick "Rust ownership rules"
/r "PBFT consensus"       # Alias
```

### `/research paper <topic>`

Academic paper search.

```bash
/research paper "transformer architectures"
/research paper "consensus algorithms" --year 2024
/research paper "machine learning" --top 10
```

### `/research compare <a> <b>`

Side-by-side comparison.

```bash
/research compare "Rust" "Go"
/research compare "PBFT" "Raft" --criteria performance,complexity
```

### `/research verify <claim>`

Fact-check a claim.

```bash
/research verify "Rust has no garbage collector"
/research verify "claim" --sources primary
```

---

## Code Commands

Write, review, and understand code.

### `/code implement <description>`

Generate code implementation.

```bash
/code implement "rate limiting middleware"
/code implement "user authentication" --lang rust
/code implement "REST API" --tests --style clean
/c "error handling"       # Alias
```

**Options:**

| Option | Values | Description |
|--------|--------|-------------|
| `--lang` | rust, python, typescript... | Target language |
| `--tests` | | Include unit tests |
| `--style` | minimal, standard, verbose | Code style |
| `--framework` | | Specific framework |

### `/code review <target>`

Review code for quality.

```bash
/code review src/main.rs
/code review PR#123
/code review . --depth thorough
/code review src/ --focus security
```

**Focus Options:**
- `security` — Vulnerability scan
- `performance` — Performance issues
- `style` — Code style/conventions
- `logic` — Logic errors
- `all` — Everything (default)

### `/code debug <issue>`

Diagnose and fix issues.

```bash
/code debug "memory leak in worker pool"
/code debug "authentication failing" --context "error.log"
/code debug "slow queries" --profile
```

### `/code explain <file:lines>`

Understand code.

```bash
/code explain src/consensus.rs
/code explain src/main.rs:42-100
/code explain "fn process_message" --detail high
```

### `/code refactor <file>`

Improve code structure.

```bash
/code refactor src/legacy.rs
/code refactor src/api.rs --focus performance
/code refactor . --dry-run
```

### `/code test <target>`

Generate or run tests.

```bash
/code test src/auth.rs              # Generate tests
/code test run                       # Run all tests
/code test run --coverage           # With coverage
```

---

## Analysis Commands

Extract insights from data.

### `/analyze data <source>`

Analyze datasets.

```bash
/analyze data "04_GOLD/metrics.parquet"
/analyze data "SELECT * FROM users" --db postgres
/analyze data "file.csv" --focus anomalies
/an data "file"           # Alias
```

### `/analyze trend <topic>`

Trend analysis.

```bash
/analyze trend "user growth last 30 days"
/analyze trend "API latency" --period 7d
/analyze trend "errors" --compare "previous_week"
```

### `/analyze compare <a> <b>`

Compare datasets or periods.

```bash
/analyze compare Q1 Q2
/analyze compare "before_change" "after_change"
/analyze compare v1 v2 --metrics "latency,throughput"
```

### `/analyze forecast <metric>`

Predictive analysis.

```bash
/analyze forecast "user_growth" --horizon 30d
/analyze forecast "revenue" --confidence 0.95
```

---

## Strategy Commands

Plan and assess.

### `/strategy plan <objective>`

Create strategic plans.

```bash
/strategy plan "Launch v2.0 by Q2"
/strategy plan "Expand to Asia" --timeline "6 months"
/strategy plan "Increase revenue 50%" --constraints "budget < $100k"
/s plan "Growth"          # Alias
```

### `/strategy assess <scenario>`

Risk and opportunity assessment.

```bash
/strategy assess "Enter new market"
/strategy assess "Open source the project"
/strategy assess "Acquire competitor" --swot
```

### `/strategy roadmap <timeframe>`

Generate roadmap.

```bash
/strategy roadmap "2026"
/strategy roadmap "Q2" --goals "growth,stability"
/strategy roadmap "next 6 months" --format gantt
```

### `/strategy compete <target>`

Competitive analysis.

```bash
/strategy compete "Company X"
/strategy compete "market segment" --depth deep
```

---

## Execution Commands

Run and deploy.

### `/exec run <command>`

Execute system command (Guardian approval required).

```bash
/exec run "cargo test"
/exec run "npm install" --dir ./frontend
/exec run "docker compose up" --background
/x "make build"           # Alias
```

### `/exec deploy <target>`

Deploy to environment.

```bash
/exec deploy staging
/exec deploy production --confirm
/exec deploy canary --percentage 10
```

### `/exec automate <workflow>`

Create automation.

```bash
/exec automate "daily-backup"
/exec automate "test-on-push" --trigger "git push"
/exec automate "weekly-report" --schedule "0 9 * * MON"
```

### `/exec rollback <deployment>`

Revert deployment.

```bash
/exec rollback production
/exec rollback staging --to "v1.2.3"
```

---

## Guardian Commands

Ethics and safety oversight.

### `/guardian status`

View FATE gates status.

```bash
/guardian status
/g                        # Alias
```

**Output:**
```
╔════════════════════════════════════════════════════════════╗
║                    FATE Gates Status                        ║
╠════════════════════════════════════════════════════════════╣
║   ✓ Ihsān:      0.97 / 0.95 (PASS)                         ║
║   ✓ Adl:        0.28 / 0.35 (PASS)                         ║
║   ✓ Harm:       0.12 / 0.30 (PASS)                         ║
║   ✓ Confidence: 0.91 / 0.80 (PASS)                         ║
║                                                            ║
║   Overall: ALL GATES PASSING                               ║
╚════════════════════════════════════════════════════════════╝
```

### `/guardian review <action>`

Request ethics review.

```bash
/guardian review "delete all user data"
/guardian review "send mass email"
/guardian review "deploy to production"
```

### `/guardian alert [severity]`

View system alerts.

```bash
/guardian alert              # All alerts
/guardian alert critical     # Only critical
/guardian alert --since "1h" # Recent alerts
/guardian alert --ack 42     # Acknowledge alert
```

### `/guardian audit [period]`

View audit log.

```bash
/guardian audit              # Recent actions
/guardian audit "today"      # Today's actions
/guardian audit --agent executor  # By agent
```

### `/guardian override <gate>`

Request gate bypass (dangerous, requires justification).

```bash
/guardian override confidence --reason "Experimental research"
```

---

## Memory Commands

Persistent knowledge management.

### `/memory remember <content>`

Store information.

```bash
/memory remember "API key rotation is monthly"
/memory remember "John prefers email" --tags "contacts,preferences"
/memory remember "Decision: Use Rust for backend" --type decision
/m remember "Important"   # Alias
```

### `/memory recall <query>`

Retrieve information.

```bash
/memory recall "security policies"
/memory recall "John" --type contacts
/memory recall "decisions about architecture"
/m "API patterns"         # Alias
```

### `/memory forget <query>`

Remove from memory.

```bash
/memory forget "old API endpoint"
/memory forget --older-than "1 year"
/memory forget --tag "deprecated"
```

### `/memory list [category]`

Browse memories.

```bash
/memory list                 # All categories
/memory list decisions       # Just decisions
/memory list --recent 10     # Recent items
```

### `/memory export [format]`

Export memories.

```bash
/memory export                # JSON
/memory export --format markdown
/memory export --category decisions
```

---

## Voice Commands

Voice interface control.

### `/voice on`

Enable voice mode.

```bash
/voice on
/v on                     # Alias
```

### `/voice off`

Disable voice mode.

```bash
/voice off
```

### `/voice agent <name>`

Set voice agent.

```bash
/voice agent guardian
/voice agent researcher
```

### `/voice config`

Configure voice settings.

```bash
/voice config --speed 1.2
/voice config --volume 0.8
/voice config --model "NATF3.pt"
```

---

## Proactive Commands

Anticipation and briefings.

### `/morning`

Morning briefing.

```bash
/morning
```

**Output:**
```
☀️ صباح الخير (Good morning), MoMo!

📋 Today's Priorities:
1. Complete CLI documentation
2. Review PR #42
3. Team sync at 14:00

⚠️ Alerts:
- None overnight

💡 Suggestion: Start with documentation while energy is high.
```

### `/daily-review`

End-of-day summary.

```bash
/daily-review
```

### `/weekly`

Weekly review and planning.

```bash
/weekly
/weekly --focus "goals"
```

### `/standup`

Quick status report.

```bash
/standup
```

**Output:**
```
📊 Standup Report

Yesterday:
- Completed authentication module
- Reviewed 3 PRs

Today:
- Finish documentation
- Start federation tests

Blockers:
- None
```

### `/suggest`

Get proactive suggestions.

```bash
/suggest                  # General suggestions
/suggest --context "afternoon"
/suggest --goal "complete Q1 milestone"
```

---

## Quick Commands

Common operations with smart defaults.

### `/commit`

Smart git commit.

```bash
/commit
/gc                       # Alias
```

Automatically:
1. Stages changes
2. Generates commit message
3. Requests approval
4. Commits

### `/pr`

Create pull request.

```bash
/pr
/pr --draft
/pr --reviewer "team"
```

### `/summarize <content>`

Summarize content.

```bash
/summarize "long text here"
/summarize https://example.com/article
/summarize file.md
/sum "content"            # Alias
```

### `/translate <text> [lang]`

Translate text.

```bash
/translate "Hello world" ar
/translate "مرحبا" en
/tr "text" ja             # Alias
```

### `/explain <topic>`

Quick explanation.

```bash
/explain "Rust lifetimes"
/explain "Byzantine fault tolerance"
```

---

## Modifiers

Global flags that work with any command.

### Priority Modifiers

```bash
/task add "Fix bug" --urgent     # High priority
/research "topic" --low          # Low priority
```

### Output Modifiers

```bash
/agent list --json               # JSON output
/research "topic" --markdown     # Markdown output
/analyze data --brief            # Concise output
/code review --detailed          # Verbose output
```

### Execution Modifiers

```bash
/exec deploy --dry-run           # Preview only
/task delete --force             # Skip confirmation
/code implement --async          # Run in background
```

### Agent Modifiers

```bash
/research "topic" --agent strategist  # Override agent
/code implement --context "API design" # Add context
```

### Memory Modifiers

```bash
/research "topic" --remember     # Save to memory
/code implement --no-remember    # Don't save
```

### Modifier Combinations

```bash
/research deep "topic" --brief --json --remember --urgent
/exec deploy staging --dry-run --detailed
```

---

## Aliases

Quick shortcuts for power users.

| Alias | Expands To | Example |
|-------|------------|---------|
| `/a` | `/agent switch` | `/a dev` |
| `/t` | `/task add` | `/t "Fix bug"` |
| `/r` | `/research quick` | `/r "Rust traits"` |
| `/c` | `/code implement` | `/c "auth"` |
| `/x` | `/exec run` | `/x "cargo test"` |
| `/m` | `/memory recall` | `/m "API"` |
| `/v` | `/voice` | `/v on` |
| `/g` | `/guardian status` | `/g` |
| `/gc` | `/commit` | `/gc` |
| `/pr` | `/quick pr` | `/pr` |
| `/sum` | `/summarize` | `/sum "text"` |
| `/tr` | `/translate` | `/tr "hello" ar` |

### Agent Shortcuts

| Alias | Agent |
|-------|-------|
| `/a strat` | Strategist |
| `/a res` | Researcher |
| `/a dev` | Developer |
| `/a ana` | Analyst |
| `/a rev` | Reviewer |
| `/a exe` | Executor |
| `/a guard` | Guardian |

---

## Advanced Usage

### Command Chaining

```bash
# Research then implement
/research quick "rate limiting" && /code implement "rate limiter"

# Review then deploy
/code review src/ && /exec deploy staging
```

### Batch Operations

```bash
# Multiple tasks
/task add "Task 1" && /task add "Task 2" && /task add "Task 3"

# Or use batch syntax
/task batch add ["Task 1", "Task 2", "Task 3"]
```

### Context Injection

```bash
# Include file context
/code implement "handler" --context @src/types.rs

# Include memory context
/research "topic" --context @memory:previous_research
```

### Pipeline Mode

```bash
# Multi-agent pipeline
/pipeline start "feature development"
  /research deep "requirements"
  /code implement "solution"
  /code review --thorough
  /exec deploy staging
/pipeline end
```

### Scheduled Commands

```bash
# Run at specific time
/schedule "08:00" /morning

# Run with cron
/schedule "0 9 * * MON" /weekly
```

---

## Command Help

Get help for any command:

```bash
/help                     # All commands
/help agent               # Agent commands
/help task add            # Specific command
/help --search "deploy"   # Search commands
```

---

## Tips

1. **Use aliases** — `/a dev` is faster than `/agent switch developer`
2. **Combine modifiers** — `--brief --json` for API-friendly output
3. **Remember useful results** — Add `--remember` to save insights
4. **Use context** — Add `--context` for better results
5. **Check Guardian** — Run `/g` regularly to monitor FATE gates

---

## Next Steps

- [TUI Navigation](06-TUI-NAVIGATION.md) — Visual interface guide
- [Hooks Automation](09-HOOKS-AUTOMATION.md) — Automate your workflow
- [Command Reference](../reference/COMMAND-REFERENCE.md) — Complete reference

---

**Master the commands, master your workflow.** ⚡
