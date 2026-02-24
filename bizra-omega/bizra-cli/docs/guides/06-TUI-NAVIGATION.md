# TUI Navigation Guide

Master the Terminal User Interface for maximum productivity.

## Table of Contents

1. [Overview](#overview)
2. [Layout](#layout)
3. [Views](#views)
4. [Keyboard Shortcuts](#keyboard-shortcuts)
5. [Modes](#modes)
6. [Panels](#panels)
7. [Navigation Patterns](#navigation-patterns)
8. [Customization](#customization)
9. [Tips & Tricks](#tips--tricks)

---

## Overview

The BIZRA TUI provides a rich terminal interface for interacting with your PAT team.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ BIZRA v0.1.0                                    🛡 Guardian │ GMT+4 │ 14:32 │
├──────────────────────┬──────────────────────────────────────────────────────┤
│                      │                                                      │
│   AGENTS             │   CHAT                                               │
│   ──────────         │   ────                                               │
│   ♟ Strategist       │   🛡 Guardian: How can I help you today?            │
│   🔍 Researcher       │                                                      │
│   ⚙ Developer        │   You: Research Byzantine fault tolerance            │
│   📊 Analyst          │                                                      │
│ > 🛡 Guardian         │   🔍 Researcher: Byzantine Fault Tolerance (BFT)    │
│   ✓ Reviewer         │   allows distributed systems to reach consensus     │
│   ▶ Executor         │   even when some nodes fail or act maliciously...   │
│                      │                                                      │
├──────────────────────┼──────────────────────────────────────────────────────┤
│   FATE GATES         │                                                      │
│   ──────────         │                                                      │
│   Ihsān:    ████████░░ 0.97                                                │
│   Adl:      ██████░░░░ 0.28                                                │
│   Harm:     ███░░░░░░░ 0.12                                                │
│   Confidence████████░░ 0.91                                                │
│                      │                                                      │
├──────────────────────┴──────────────────────────────────────────────────────┤
│ > Type message or /command                                          [i]nsert│
└─────────────────────────────────────────────────────────────────────────────┘
```

### Launch TUI

```bash
bizra tui              # Start TUI mode
bizra                  # TUI is default
```

---

## Layout

The TUI uses a responsive multi-panel layout.

### Standard Layout

```
┌────────────────────────────────────────────────────────────────────────────┐
│                              HEADER BAR                                     │
├─────────────────────┬──────────────────────────────────────────────────────┤
│                     │                                                      │
│    SIDEBAR          │              MAIN CONTENT                            │
│    (Agents/         │              (Chat/Tasks/etc.)                       │
│     Status)         │                                                      │
│                     │                                                      │
├─────────────────────┴──────────────────────────────────────────────────────┤
│                              INPUT BAR                                      │
└────────────────────────────────────────────────────────────────────────────┘
```

### Components

| Component | Description |
|-----------|-------------|
| **Header Bar** | App name, current agent, time, status |
| **Sidebar** | Agent list, FATE gates, quick actions |
| **Main Content** | Primary view (chat, tasks, etc.) |
| **Input Bar** | Message/command input, mode indicator |

### Responsive Behavior

| Width | Layout |
|-------|--------|
| < 80 cols | Sidebar hidden, full-width content |
| 80-120 cols | Narrow sidebar, expanded content |
| > 120 cols | Full sidebar, spacious content |

---

## Views

Switch between views using Tab or number keys.

### Dashboard View (1)

Overview of system status.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ DASHBOARD                                                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   📊 Quick Stats                     🎯 Active Goals                        │
│   ──────────────                     ────────────────                       │
│   Tasks Pending: 5                   Q1: Node0 production (67%)            │
│   Tasks Today: 3                     Week: PBFT implementation             │
│   Streak: 12 days                                                          │
│                                                                             │
│   🛡 FATE Status                      📋 Recent Activity                    │
│   ─────────────                      ────────────────                       │
│   All gates passing                  • Completed: Auth module              │
│   Last check: 2m ago                 • Reviewed: PR #42                    │
│                                      • Research: BFT patterns              │
│                                                                             │
│   💡 Suggestions                                                           │
│   ─────────────                                                            │
│   1. Continue PBFT implementation                                          │
│   2. Review blocked PR #43                                                 │
│   3. Schedule weekly planning                                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Agents View (2)

Agent management and status.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ AGENTS                                                                      │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ♟ STRATEGIST                       Status: Ready                         │
│   ────────────                       Giants: Sun Tzu • Clausewitz • Porter │
│   Strategic planning, market analysis, risk assessment                     │
│   Tasks: 2 pending, 15 completed                                           │
│                                                                             │
│   🔍 RESEARCHER                       Status: Active                        │
│   ───────────                        Giants: Shannon • Turing • Dijkstra   │
│   Knowledge discovery, fact verification, synthesis                        │
│   Current: Researching "Byzantine fault tolerance"                         │
│                                                                             │
│ > 🛡 GUARDIAN                         Status: Watching                      │
│   ─────────                          Giants: Al-Ghazali • Rawls • Anthropic│
│   Ethics, safety, FATE gate enforcement                                    │
│   FATE: Ihsān 0.97 | Adl 0.28 | Harm 0.12 | Conf 0.91                     │
│                                                                             │
│   [Enter] Switch  [s] Status  [c] Configure  [?] Help                      │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Chat View (3)

Primary interaction view.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ CHAT with Guardian                                                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   🛡 Guardian                                                  [14:28]      │
│   ─────────                                                                 │
│   صباح الخير, MoMo! How can I help you today?                              │
│                                                                             │
│   You                                                          [14:30]      │
│   ───                                                                       │
│   Research Byzantine fault tolerance for our consensus module              │
│                                                                             │
│   🔍 Researcher                                                 [14:31]      │
│   ───────────                                                               │
│   ## Byzantine Fault Tolerance                                              │
│                                                                             │
│   BFT allows distributed systems to reach consensus even when              │
│   some nodes fail or behave maliciously.                                   │
│                                                                             │
│   ### Key Algorithms:                                                       │
│   1. **PBFT** - Practical BFT (Castro & Liskov, 1999)                      │
│   2. **Tendermint** - BFT + PoS (Kwon, 2014)                               │
│   ...                                                                       │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ > _                                                              [i]nsert   │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Tasks View (4)

Task management.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ TASKS                                                      Filter: pending  │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   ○ HIGH    Complete PBFT implementation              Developer   Due: Fri │
│   ○ NORMAL  Review PR #42                             Reviewer    Due: Today│
│ > ○ NORMAL  Write federation tests                    Developer   Due: Mon │
│   ○ LOW     Update documentation                      Researcher  Due: Wed │
│   ● DONE    Implement voting mechanism                Developer   ✓ Today  │
│   ● DONE    Research BFT patterns                     Researcher  ✓ Today  │
│                                                                             │
│   ───────────────────────────────────────────────────────────────────────   │
│                                                                             │
│   Selected: Write federation tests                                         │
│   Created: Feb 4, 2026                                                     │
│   Agent: Developer                                                         │
│   Description: Write comprehensive tests for the federation protocol       │
│                including node discovery and consensus...                   │
│                                                                             │
│   [Enter] Start  [d] Done  [e] Edit  [x] Delete  [n] New  [f] Filter       │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Treasury View (5)

Resource management (future).

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ TREASURY                                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   💰 Resource Pool Status                                                  │
│   ───────────────────────                                                  │
│   Status: Initializing...                                                  │
│                                                                             │
│   This feature is under development.                                       │
│   The Resource Pool will enable:                                           │
│                                                                             │
│   • Federated compute sharing                                              │
│   • Mudarabah partnerships                                                 │
│   • Zakat-compliant distributions                                          │
│   • Cross-node resource allocation                                         │
│                                                                             │
│   See: /mnt/c/BIZRA-DATA-LAKE/bizra-omega/bizra-resourcepool/              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Settings View (6)

Configuration.

```
┌─────────────────────────────────────────────────────────────────────────────┐
│ SETTINGS                                                                    │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│   PROFILE                            FATE THRESHOLDS                        │
│   ───────                            ────────────────                       │
│   Name: MoMo (محمد)                   Ihsān:      0.95                       │
│   Location: Dubai, UAE               Adl (Gini): 0.35                       │
│   Timezone: GMT+4                    Harm:       0.30                       │
│   Default Agent: Guardian            Confidence: 0.80                       │
│                                                                             │
│   PROACTIVE MODE                     LLM BACKEND                            │
│   ──────────────                     ───────────                            │
│   Current: Balanced                  Primary: LM Studio                     │
│   [ ] Silent                         URL: 192.168.56.1:1234                 │
│   [ ] Minimal                        Fallback: Ollama                       │
│   [●] Balanced                       Status: Connected                      │
│   [ ] Active                                                                │
│                                                                             │
│   [Enter] Edit  [r] Reset  [s] Save  [?] Help                              │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Keyboard Shortcuts

### Global

| Key | Action |
|-----|--------|
| `Tab` | Cycle through views |
| `1-6` | Jump to specific view |
| `q` / `Ctrl+C` | Quit |
| `?` / `F1` | Help |
| `Ctrl+L` | Clear/refresh screen |
| `Escape` | Exit current mode/cancel |

### Navigation

| Key | Action |
|-----|--------|
| `j` / `↓` | Move down |
| `k` / `↑` | Move up |
| `h` / `←` | Move left / collapse |
| `l` / `→` | Move right / expand |
| `g` | Go to top |
| `G` | Go to bottom |
| `Ctrl+D` | Page down |
| `Ctrl+U` | Page up |
| `Enter` | Select / confirm |
| `Space` | Toggle selection |

### Input Mode

| Key | Action |
|-----|--------|
| `i` | Enter insert mode |
| `Escape` | Exit insert mode |
| `/` | Enter command mode |
| `Ctrl+Enter` | Submit message |
| `Ctrl+K` | Clear input |
| `↑` / `↓` | History navigation |
| `Tab` | Autocomplete |

### Agent Shortcuts

| Key | Action |
|-----|--------|
| `a` | Show agent list |
| `1-7` (in agent list) | Quick switch agent |
| `s` | Strategist |
| `r` | Researcher |
| `d` | Developer |
| `n` | Analyst |
| `v` | Reviewer |
| `x` | Executor |
| `g` | Guardian |

### Task Shortcuts

| Key | Action |
|-----|--------|
| `t` | Go to tasks view |
| `n` | New task |
| `e` | Edit selected task |
| `D` | Mark done |
| `X` | Delete task |
| `f` | Filter tasks |
| `p` | Set priority |

---

## Modes

The TUI operates in different modes.

### Normal Mode

Default navigation mode.

- Navigate with `j/k/h/l` or arrow keys
- Execute actions with shortcuts
- Press `i` to enter Insert mode
- Press `/` to enter Command mode

**Indicator:** None (or `[n]ormal` in status)

### Insert Mode

Text input mode.

- Type freely in input area
- `Escape` returns to Normal mode
- `Ctrl+Enter` submits message
- `Tab` for autocomplete

**Indicator:** `[i]nsert`

### Command Mode

Slash command entry.

- Activated by pressing `/`
- Autocomplete available
- `Enter` executes command
- `Escape` cancels

**Indicator:** `/command`

### Visual Mode

Selection mode (for text/items).

- `v` starts character selection
- `V` starts line selection
- `y` copies selection
- `Escape` exits

**Indicator:** `[v]isual`

### Search Mode

Search within current view.

- `/` + search term in views
- `n` next match
- `N` previous match
- `Escape` exits

**Indicator:** `/search`

---

## Panels

### Sidebar Panel

Shows context-relevant information.

**In Chat View:**
```
┌──────────────────────┐
│ AGENTS               │
│ ♟ Strategist         │
│ 🔍 Researcher         │
│ ⚙ Developer          │
│ 📊 Analyst            │
│ ✓ Reviewer           │
│ ▶ Executor           │
│>🛡 Guardian          │
├──────────────────────┤
│ FATE GATES           │
│ Ihsān:    ████░ 0.97 │
│ Adl:      ███░░ 0.28 │
│ Harm:     █░░░░ 0.12 │
│ Conf:     ████░ 0.91 │
└──────────────────────┘
```

**In Tasks View:**
```
┌──────────────────────┐
│ FILTERS              │
│ ○ All                │
│ ● Pending            │
│ ○ Active             │
│ ○ Completed          │
├──────────────────────┤
│ BY AGENT             │
│ □ Strategist (1)     │
│ □ Developer (3)      │
│ ■ All agents         │
├──────────────────────┤
│ BY PRIORITY          │
│ □ High (1)           │
│ □ Normal (2)         │
│ □ Low (1)            │
└──────────────────────┘
```

### Toggle Sidebar

| Key | Action |
|-----|--------|
| `[` | Hide sidebar |
| `]` | Show sidebar |
| `\` | Toggle sidebar |

---

## Navigation Patterns

### Quick Agent Switch

```
Press: a → 3          # Switch to Developer (3rd in list)
Press: a → d          # Switch to Developer (d shortcut)
Press: /a dev         # Command mode switch
```

### Quick Task Creation

```
Press: t              # Go to Tasks view
Press: n              # New task
Type:  "My new task"
Press: Enter          # Create task
```

### Quick Search

```
Press: /              # Enter command mode
Type:  research BFT   # Query
Press: Enter          # Execute
```

### View History

```
Press: Ctrl+U         # Scroll up in history
Press: Ctrl+D         # Scroll down
Press: g              # Go to beginning
Press: G              # Go to end
```

---

## Customization

### Theme

```yaml
# In sovereign_profile.yaml
tui:
  theme: "dark"        # dark | light | solarized | nord
  colors:
    primary: "#4a9eff"
    secondary: "#ff6b6b"
    background: "#1a1a2e"
    foreground: "#e0e0e0"
```

### Layout

```yaml
tui:
  layout:
    sidebar_width: 25          # Characters
    show_fate_gates: true
    show_clock: true
    compact_mode: false
```

### Keybindings

```yaml
tui:
  keybindings:
    quit: ["q", "Ctrl+C"]
    help: ["?", "F1"]
    insert_mode: ["i"]
    command_mode: ["/"]
    agent_strategist: ["s"]
    agent_researcher: ["r"]
    # ... customize as needed
```

---

## Tips & Tricks

### 1. Quick Commands

Use `/` prefix anywhere:
```
/morning              # Morning briefing
/g                    # Guardian status
/t "Quick task"       # Create task
```

### 2. Agent Hotkeys

In chat, type agent prefix:
```
@researcher Research X    # Direct to researcher
@developer Implement Y    # Direct to developer
```

### 3. Inline Formatting

Markdown in input:
```
**bold**, *italic*, `code`
```

### 4. History Recall

```
↑                     # Previous command
↓                     # Next command
Ctrl+R                # Search history
```

### 5. Split View (Coming Soon)

```
Ctrl+\                # Split horizontal
Ctrl+|                # Split vertical
Ctrl+W                # Switch pane
```

### 6. Focus Mode

```
/focus                # Hide sidebar, full chat
/focus off            # Restore layout
```

### 7. Mouse Support

| Action | Effect |
|--------|--------|
| Click | Select item |
| Double-click | Execute/expand |
| Scroll | Navigate list |
| Drag | Select text |

---

## Status Bar Icons

| Icon | Meaning |
|------|---------|
| 🛡 | Guardian active |
| ⚡ | Processing |
| ✓ | Success |
| ⚠ | Warning |
| ✗ | Error |
| 🔇 | Silent mode |
| 🔊 | Voice active |
| 🌐 | Federation connected |

---

## Next Steps

- [Slash Commands](05-SLASH-COMMANDS.md) — Full command reference
- [Voice Interface](07-VOICE-INTERFACE.md) — Voice interaction
- [Keyboard Customization](../reference/CONFIG-REFERENCE.md#keybindings)

---

**Master the TUI, master your workflow.** ⌨️
