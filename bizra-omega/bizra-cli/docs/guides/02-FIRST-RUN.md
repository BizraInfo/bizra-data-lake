# First Run Guide

Your first experience with BIZRA CLI — what to expect and how to get started.

## Table of Contents

1. [Launching the CLI](#launching-the-cli)
2. [The Welcome Experience](#the-welcome-experience)
3. [Understanding the Interface](#understanding-the-interface)
4. [Your First Commands](#your-first-commands)
5. [Meeting Your PAT Team](#meeting-your-pat-team)
6. [Setting Up Your Day](#setting-up-your-day)

---

## Launching the CLI

### Command Line Mode

```bash
# Show help
bizra --help

# Show status
bizra status

# Show system info
bizra info
```

### TUI Mode (Full Interface)

```bash
# Launch TUI
bizra

# Or explicitly
bizra tui
```

---

## The Welcome Experience

When you first launch the TUI, you'll see:

```
╔════════════════════════════════════════════════════════════════════════════╗
║                     Welcome to BIZRA Node0                                 ║
║                                                                            ║
║  بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ                                    ║
║                                                                            ║
║  Your Personal Agentic Team is ready.                                      ║
║  Type /help for commands, or just start chatting.                          ║
╚════════════════════════════════════════════════════════════════════════════╝
```

### What This Means

- **بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ** — "In the name of God, the Most Gracious, the Most Merciful"
- **PAT Team Ready** — Your 7 Personal Agentic Team members are initialized
- **Node0** — You are the genesis node of your sovereign network

---

## Understanding the Interface

### TUI Layout

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ ✦ BIZRA  MoMo (محمد)  │ [1]Dashboard [2]Agents [3]Chat [4]Tasks │ ● LM 🎤  │  ← Header
├──────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐               │
│  │ ♟ Strategist│ │🔍 Researcher│ │ ⚙ Developer│ │ 📊 Analyst │               │  ← Agent
│  │   Ready    │ │   Ready    │ │   Ready    │ │   Ready    │               │     Cards
│  └────────────┘ └────────────┘ └────────────┘ └────────────┘               │
│  ┌────────────┐ ┌────────────┐ ┌────────────┐                              │
│  │ ✓ Reviewer │ │ ▶ Executor │ │ 🛡 Guardian │ ← Selected                   │
│  │   Ready    │ │   Ready    │ │   Ready    │                              │
│  └────────────┘ └────────────┘ └────────────┘                              │
│                                                                              │
│  ┌─ FATE Gates ─────────────────┐  ┌─ Node Info ─────────────┐             │
│  │ ✓ Ihsān:   0.95 / 0.95      │  │ ID: node0_ce5af35c...   │             │
│  │ ○ Adl:     0.00 / 0.35      │  │ Name: MoMo (محمد)       │             │
│  │ ○ Harm:    0.00 / 0.30      │  │ Genesis: a7f68f1f...    │             │
│  │ ○ Conf:    0.00 / 0.80      │  └─────────────────────────┘             │
│  └──────────────────────────────┘                                          │
│                                                                              │
├──────────────────────────────────────────────────────────────────────────────┤
│ NORMAL │ 🛡 Guardian │ q:Quit  Tab:View  j/k:Nav  i:Insert  /:Command       │  ← Status Bar
└──────────────────────────────────────────────────────────────────────────────┘
```

### Views (Press 1-6 or Tab)

| Key | View | Purpose |
|-----|------|---------|
| `1` | Dashboard | Overview of agents and status |
| `2` | Agents | Detailed agent cards |
| `3` | Chat | Conversation with agents |
| `4` | Tasks | Task management |
| `5` | Treasury | Resource management |
| `6` | Settings | Configuration |

### Status Bar Indicators

| Indicator | Meaning |
|-----------|---------|
| `NORMAL` | Normal mode (navigation) |
| `INSERT` | Editing mode (typing) |
| `COMMAND` | Command mode (/) |
| `● LM` | LM Studio connected |
| `○ LM` | LM Studio disconnected |
| `🎤` | Voice active |
| `🔇` | Voice inactive |

---

## Your First Commands

### Getting Help

Press `i` to enter insert mode, then type:

```
/help
```

This shows all available commands.

### Checking Status

```
/guardian status
```

Shows FATE gate status and any alerts.

### Switching Agents

```
/agent switch developer
```

or use the shortcut:

```
/a developer
```

### Adding a Task

```
/task add "Learn BIZRA CLI"
```

or shortcut:

```
/t "Learn BIZRA CLI"
```

---

## Meeting Your PAT Team

### View All Agents

Press `2` to go to Agents view, or:

```
/agent list
```

### Agent Overview

| Agent | When to Use | Example |
|-------|-------------|---------|
| **Strategist** | Planning, decisions | "Plan Q2 roadmap" |
| **Researcher** | Finding information | "Research quantum computing" |
| **Developer** | Writing code | "Implement auth middleware" |
| **Analyst** | Data questions | "Analyze user growth" |
| **Reviewer** | Quality checks | "Review this PR" |
| **Executor** | Running tasks | "Deploy to staging" |
| **Guardian** | Ethics, safety | "Review this action" |

### Quick Agent Selection

Use `j`/`k` to navigate agents, or:

```
/a strategist   # Switch to Strategist
/a researcher   # Switch to Researcher
/a developer    # Switch to Developer
/a analyst      # Switch to Analyst
/a reviewer     # Switch to Reviewer
/a executor     # Switch to Executor
/a guardian     # Switch to Guardian
```

---

## Setting Up Your Day

### Morning Routine

When you start your day:

```
/morning
```

This gives you:
- Overnight alerts
- Priority tasks
- Calendar preview
- Quick metrics

### Creating Your First Task

```
/task add "Set up development environment" -a developer
```

This:
1. Creates the task
2. Assigns it to Developer agent
3. Developer starts analyzing requirements

### Research Something

```
/research quick "BIZRA architecture"
```

or for deep research:

```
/research deep "distributed consensus algorithms"
```

### End of Day

```
/daily-review
```

This captures:
- Completed tasks
- Learnings
- Tomorrow's preview

---

## Keyboard Reference (Quick)

| Key | Mode | Action |
|-----|------|--------|
| `Tab` | Normal | Next view |
| `1-6` | Normal | Jump to view |
| `j`/`k` | Normal | Navigate agents |
| `i` | Normal | Enter insert mode |
| `/` | Normal | Enter command mode |
| `Esc` | Insert | Return to normal |
| `Enter` | Insert | Send message |
| `q` | Normal | Quit |

---

## What's Next?

Now that you've completed your first run:

1. **[Personalization](03-PERSONALIZATION.md)** — Make it yours
2. **[PAT Agents](04-PAT-AGENTS.md)** — Deep dive into agents
3. **[Commands](05-SLASH-COMMANDS.md)** — Learn all commands

---

## Tips for Beginners

1. **Start with Guardian** — It's the safest default agent
2. **Use /help often** — It's context-aware
3. **Try /morning** — Great way to start each session
4. **Explore with Tab** — Switch views to learn the interface
5. **Read the prompts** — The system guides you

---

**Welcome to your sovereign node!** 🌟
