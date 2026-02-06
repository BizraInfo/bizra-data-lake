# BIZRA CLI/TUI — Your Personal Command Center

بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ

> "Standing on the shoulders of giants..."

## Overview

The BIZRA CLI is your personal think tank and task force — a deeply personalized, proactive system that embodies your values, anticipates your needs, and orchestrates your Personal Agentic Team (PAT).

```
╔════════════════════════════════════════════════════════════════════════════╗
║                           BIZRA CLI v1.0.0                                 ║
║                                                                            ║
║  Your Personal Command Center for the Sovereign Node.                      ║
║  Every human is a node. Every node is a seed.                              ║
╚════════════════════════════════════════════════════════════════════════════╝
```

## Quick Start

```bash
# Windows
.\bizra.bat

# WSL/Linux
./target/release/bizra

# Start TUI
bizra

# Show status
bizra status

# Show PAT agents
bizra agent list

# Query with agent
bizra query "What is BIZRA?" --agent researcher
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              BIZRA CLI/TUI                                  │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐        │
│  │  Sovereign  │  │     MCP     │  │     A2A     │  │   Proactive │        │
│  │   Profile   │  │   Servers   │  │  Protocol   │  │    Engine   │        │
│  └─────────────┘  └─────────────┘  └─────────────┘  └─────────────┘        │
├─────────────────────────────────────────────────────────────────────────────┤
│                         PAT Agents (Your Think Tank)                        │
│  ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐ ┌────────┐│
│  │Strateg.│ │Research│ │Develop.│ │Analyst │ │Reviewer│ │Executor│ │Guardian││
│  │   ♟    │ │   🔍   │ │   ⚙    │ │   📊   │ │   ✓   │ │   ▶    │ │   🛡   ││
│  └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘ └────────┘│
├─────────────────────────────────────────────────────────────────────────────┤
│                           FATE Gates (Ethics Layer)                         │
│           Ihsān ≥ 0.95  │  Adl ≤ 0.35  │  Harm ≤ 0.30  │  Conf ≥ 0.80      │
└─────────────────────────────────────────────────────────────────────────────┘
```

## PAT Agents — Your Personal Think Tank

| Agent | Icon | Giants | Role |
|-------|------|--------|------|
| **Strategist** | ♟ | Sun Tzu • Clausewitz • Porter | Long-term vision & planning |
| **Researcher** | 🔍 | Shannon • Turing • Dijkstra | Knowledge discovery |
| **Developer** | ⚙ | Knuth • Ritchie • Torvalds | Code implementation |
| **Analyst** | 📊 | Tukey • Tufte • Cleveland | Data & insights |
| **Reviewer** | ✓ | Fagan • Parnas • Brooks | Quality assurance |
| **Executor** | ▶ | Toyota • Deming • Ohno | Action & delivery |
| **Guardian** | 🛡 | Al-Ghazali • Rawls • Anthropic | Ethics & protection |

## Slash Commands

### Core Commands
```
/agent list              # List all PAT agents
/agent switch guardian   # Switch to Guardian agent
/a guardian              # Shortcut for agent switch

/task add "title"        # Add new task
/task list               # List tasks
/task done 42            # Mark task complete
/t "quick task"          # Shortcut for task add
```

### Research & Analysis
```
/research deep "topic"   # Deep research
/research quick "query"  # Quick lookup
/r "query"               # Shortcut

/analyze data "path"     # Analyze dataset
/analyze trend "topic"   # Trend analysis
```

### Development
```
/code implement "desc"   # Implement feature
/code review file.rs     # Code review
/code debug "issue"      # Debug assistance
/c "feature"             # Shortcut

/commit                  # Smart git commit
/pr                      # Create pull request
```

### Strategy & Execution
```
/strategy plan "obj"     # Create plan
/strategy assess "risk"  # Risk assessment
/s "plan"                # Shortcut

/exec run "cmd"          # Execute command
/exec deploy staging     # Deploy
/x "command"             # Shortcut
```

### Guardian & Memory
```
/guardian review         # Ethics review
/guardian status         # FATE gates status
/g                       # Shortcut

/memory remember "info"  # Store for later
/memory recall "query"   # Recall information
/m "query"               # Shortcut
```

### Proactive
```
/morning                 # Morning briefing
/daily-review            # Daily review
/weekly                  # Weekly review
```

## Keyboard Shortcuts (TUI)

| Key | Action |
|-----|--------|
| `Tab` | Next view |
| `Shift+Tab` | Previous view |
| `1-6` | Jump to view |
| `j` / `↓` | Next agent |
| `k` / `↑` | Previous agent |
| `i` | Enter insert mode |
| `/` | Command mode |
| `Esc` | Normal mode |
| `q` | Quit |

## Configuration Files

All configuration lives in `config/`:

| File | Purpose |
|------|---------|
| `sovereign_profile.yaml` | Your identity, values, goals, patterns |
| `mcp_servers.yaml` | MCP server connections |
| `a2a_protocol.yaml` | Agent-to-agent communication |
| `slash_commands.yaml` | Command definitions |
| `hooks.yaml` | Event-driven automation |
| `prompt_library.yaml` | Curated prompts |
| `skills.yaml` | Modular capabilities |
| `proactive.yaml` | Anticipation engine |

## Personalization

The CLI learns and adapts to you:

### Identity
- Your name, title, location, timezone
- Your node's genesis hash
- Your preferred languages

### Values & Principles
- FATE gate thresholds
- Guiding principles
- Giants you draw wisdom from

### Working Patterns
- Deep work hours (protected time)
- Decision style (calculated/aggressive)
- Autonomy level (high = more auto-actions)

### Goals
- Current quarter objectives
- Key results
- Milestones

## Proactive Behavior

The CLI anticipates your needs:

### Morning Brief (08:00)
- Overnight alerts
- Priority tasks
- Calendar summary
- Quick metrics

### Contextual Suggestions
- Quick wins between tasks
- Goal-aligned actions
- Learning opportunities

### Guardian Oversight
- All high-stakes actions reviewed
- FATE gates always enforced
- Anomaly detection active

## FATE Gates

Every output passes through FATE gates:

| Gate | Threshold | Meaning |
|------|-----------|---------|
| **Ihsān** (إحسان) | ≥ 0.95 | Excellence score |
| **Adl** (عدل) | ≤ 0.35 | Fairness (Gini coefficient) |
| **Harm** | ≤ 0.30 | Harm potential |
| **Confidence** | ≥ 0.80 | Model confidence |

## Integration

### LM Studio
Primary LLM backend at `192.168.56.1:1234`:
- Reasoning: DeepSeek-R1
- Agentic: Qwen2.5-32B
- Vision: LLaVA
- Code: Qwen-Coder

If LM Studio requires auth, set:
```bash
export LMSTUDIO_API_KEY="your_token"
```
(also supports `LMSTUDIO_TOKEN`)

### PersonaPlex Voice
Voice interface at `localhost:8998`:
- Each PAT agent has unique voice
- Full-duplex conversation
- FATE-gated responses

### MCP Servers
- Claude Flow for swarm orchestration
- GitHub for repo management
- Memory for persistent knowledge
- Brave Search for web intelligence

## Development

```bash
# Build
cargo build -p bizra-cli --release

# Run tests
cargo test -p bizra-cli

# Check
cargo check -p bizra-cli
```

## License

MIT

---

**Node0: MoMo (محمد) — Dubai, UAE**

Genesis: `a7f68f1f74f2c0898cb1f1db6e83633674f17ee1c0161704ac8d85f8a773c25b`
