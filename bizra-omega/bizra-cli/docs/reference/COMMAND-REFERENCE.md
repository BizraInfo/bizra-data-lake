# Command Reference

Complete reference for all BIZRA CLI commands.

## Table of Contents

1. [Command Syntax](#command-syntax)
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

---

## Command Syntax

### Basic Format

```
/<command> [subcommand] [arguments] [--modifiers]
```

### Examples

```bash
/agent list                         # Command with subcommand
/task add "My task"                 # Command with argument
/research deep "topic" --detailed   # Command with modifier
/a guardian                         # Alias (short form)
```

---

## Agent Commands

### `/agent list`

List all PAT agents with their status.

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

Switch the active agent.

```bash
/agent switch guardian
/agent switch developer
# Alias: /a <name>
```

**Arguments:**
| Name | Description |
|------|-------------|
| `strategist` | Strategic planning |
| `researcher` | Knowledge discovery |
| `developer` | Code implementation |
| `analyst` | Data analysis |
| `reviewer` | Quality assurance |
| `executor` | Task execution |
| `guardian` | Ethics oversight |

### `/agent status [name]`

Show detailed agent status.

```bash
/agent status              # Current agent
/agent status developer    # Specific agent
```

---

## Task Commands

### `/task add <title>`

Create a new task.

```bash
/task add "Implement authentication"
/task add "Review PR #42" -a reviewer
/task add "Research competitors" --agent strategist
# Alias: /t <title>
```

**Options:**
| Option | Description |
|--------|-------------|
| `-a, --agent` | Assign to specific agent |
| `-d, --description` | Add description |
| `-p, --priority` | Set priority (low/normal/high) |

### `/task list [filter]`

List tasks.

```bash
/task list                 # All tasks
/task list pending         # Only pending
/task list completed       # Only completed
/task list --agent developer  # By agent
```

### `/task done <id>`

Mark task as complete.

```bash
/task done 42
/task done "task-uuid"
```

### `/task delegate <id> <agent>`

Delegate task to another agent.

```bash
/task delegate 42 developer
```

---

## Research Commands

### `/research deep <topic>`

Deep multi-source research.

```bash
/research deep "quantum computing applications"
/research deep "Rust async patterns" --sources academic
```

**Options:**
| Option | Description |
|--------|-------------|
| `--sources` | Limit sources (web/academic/memory) |
| `--depth` | Research depth (summary/moderate/deep) |
| `--format` | Output format (markdown/json) |

### `/research quick <query>`

Quick fact lookup.

```bash
/research quick "Rust ownership rules"
/r "PBFT consensus"  # Alias
```

### `/research paper <topic>`

Find academic papers.

```bash
/research paper "transformer architectures"
/research paper "distributed systems" --year 2024
```

### `/research compare <a> <b>`

Compare two items.

```bash
/research compare "Rust" "Go"
/research compare "PBFT" "Raft"
```

---

## Code Commands

### `/code implement <description>`

Implement a feature.

```bash
/code implement "rate limiting middleware"
/code implement "user authentication" --lang rust
/c "error handling" # Alias
```

**Options:**
| Option | Description |
|--------|-------------|
| `--lang` | Programming language |
| `--tests` | Include tests |
| `--style` | Code style (minimal/standard/verbose) |

### `/code review <file_or_pr>`

Review code.

```bash
/code review src/main.rs
/code review PR#123
/code review . --depth thorough
```

### `/code debug <issue>`

Debug an issue.

```bash
/code debug "memory leak in worker pool"
/code debug "authentication failing" --context "error log attached"
```

### `/code explain <file:lines>`

Explain code.

```bash
/code explain src/consensus.rs
/code explain src/main.rs:42-100
```

### `/code refactor <file>`

Suggest refactoring.

```bash
/code refactor src/legacy.rs
/code refactor src/api.rs --focus performance
```

---

## Analysis Commands

### `/analyze data <source>`

Analyze data.

```bash
/analyze data "04_GOLD/metrics.parquet"
/analyze data "SELECT * FROM users" --db postgres
/an data "file.csv" # Alias
```

### `/analyze trend <topic>`

Analyze trends.

```bash
/analyze trend "user growth last 30 days"
/analyze trend "API latency" --period "7d"
```

### `/analyze compare <a> <b>`

Compare datasets or periods.

```bash
/analyze compare Q1 Q2
/analyze compare "before_change" "after_change"
```

---

## Strategy Commands

### `/strategy plan <objective>`

Create a strategic plan.

```bash
/strategy plan "Launch v2.0 by Q2"
/strategy plan "Expand to Asia" --timeline "6 months"
/s plan "Growth strategy" # Alias
```

### `/strategy assess <scenario>`

Risk assessment.

```bash
/strategy assess "Enter new market"
/strategy assess "Open source the project"
```

### `/strategy roadmap <timeframe>`

Generate roadmap.

```bash
/strategy roadmap "2026"
/strategy roadmap "Q2" --goals "growth,stability"
```

---

## Execution Commands

### `/exec run <command>`

Run a command (requires Guardian approval).

```bash
/exec run "cargo test"
/exec run "npm install" --dir ./frontend
/x "make build" # Alias
```

### `/exec deploy <target>`

Deploy to environment.

```bash
/exec deploy staging
/exec deploy production --confirm
```

### `/exec automate <workflow>`

Create automation.

```bash
/exec automate "daily-backup"
/exec automate "test-on-push"
```

---

## Guardian Commands

### `/guardian status`

Show FATE gates status.

```bash
/guardian status
/g # Alias
```

**Output:**
```
FATE Gates Status:
✓ Ihsān:      0.97 / 0.95 (PASS)
✓ Adl:        0.28 / 0.35 (PASS)
✓ Harm:       0.15 / 0.30 (PASS)
✓ Confidence: 0.92 / 0.80 (PASS)

All gates passing.
```

### `/guardian review <action>`

Request ethics review.

```bash
/guardian review "delete all user data"
/guardian review "send mass email"
```

### `/guardian alert [severity]`

View alerts.

```bash
/guardian alert            # All alerts
/guardian alert critical   # Only critical
/guardian alert --since "1h"
```

---

## Memory Commands

### `/memory remember <content>`

Store information.

```bash
/memory remember "API key rotation is monthly"
/memory remember "John prefers email" --tags "contacts,preferences"
/m remember "Important fact" # Alias
```

### `/memory recall <query>`

Recall information.

```bash
/memory recall "security policies"
/memory recall "John" --type contacts
/m "API patterns" # Alias
```

### `/memory forget <query>`

Remove from memory.

```bash
/memory forget "old API endpoint"
/memory forget --older-than "1 year"
```

---

## Voice Commands

### `/voice on`

Enable voice mode.

```bash
/voice on
/v on # Alias
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

---

## Proactive Commands

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

End of day review.

```bash
/daily-review
```

### `/weekly`

Weekly review.

```bash
/weekly
```

---

## Quick Commands

### `/commit`

Smart git commit.

```bash
/commit
/gc # Alias
```

### `/pr`

Create pull request.

```bash
/pr
/pr --draft
```

### `/summarize <content>`

Summarize content.

```bash
/summarize "long text here"
/summarize https://example.com/article
/sum "content" # Alias
```

### `/translate <text> [lang]`

Translate text.

```bash
/translate "Hello world" ar
/translate "مرحبا" en
/tr "text" # Alias
```

---

## Modifiers

Apply to any command:

| Modifier | Description |
|----------|-------------|
| `--urgent` | High priority |
| `--low` | Low priority |
| `--json` | JSON output |
| `--markdown` | Markdown output |
| `--brief` | Concise output |
| `--detailed` | Detailed output |
| `--dry-run` | Preview without executing |
| `--force` | Skip confirmations |
| `--async` | Run in background |
| `--agent <name>` | Override agent |
| `--remember` | Store result |
| `--context` | Include context |

**Examples:**

```bash
/research deep "topic" --brief --json
/exec deploy staging --dry-run
/task add "urgent fix" --urgent --agent developer
```

---

## Aliases

Quick shortcuts:

| Alias | Expands To |
|-------|------------|
| `/a` | `/agent switch` |
| `/t` | `/task add` |
| `/r` | `/research quick` |
| `/c` | `/code implement` |
| `/x` | `/exec run` |
| `/m` | `/memory recall` |
| `/v` | `/voice` |
| `/g` | `/guardian status` |
| `/gc` | `/quick commit` |
| `/pr` | `/quick pr` |
| `/sum` | `/quick summarize` |
| `/tr` | `/quick translate` |

---

## Help

### `/help`

Show all commands.

```bash
/help
/help agent        # Help for agent commands
/help --search "task"  # Search commands
```

---

**Command mastery = productivity.** ⚡
