# 01 — Proactive Mode: Your Agents Work While You Sleep

## Standing on Giants
- **OpenClaw**: 24/7 execution, persistent across sessions
- **Agent Zero**: OS-level autonomy, subordinate agent spawning
- **Boyd (OODA)**: Observe→Orient→Decide→Act at machine speed
- **Maturana (Autopoiesis)**: System produces conditions for its own operation

## Problem

Current `bizra` TUI is reactive — user types, agents respond.
OpenClaw proved that 24/7 proactive agents are the killer feature.
BIZRA has the infrastructure (heartbeat loop, EventBus, 12 agents)
but no proactive surface.

## Solution: 4 Proactive Modes

```
MODE 1: reactive           (current) — user asks, agents answer
MODE 2: proactive_suggest  — agents observe and SUGGEST actions
MODE 3: proactive_auto     — agents execute pre-approved action types
MODE 4: proactive_partner  — full autonomy within constitutional gates
```

## Pseudocode: Proactive Engine

```python
class ProactiveEngine:
    def __init__(self, living_memory, rust_bridge, config):
        self.memory = living_memory
        self.bridge = rust_bridge
        self.mode = config.proactive_mode  # reactive|suggest|auto|partner
        self.rules = config.auto_approve_rules  # what's pre-approved

    def tick(self, now):
        """Called by heartbeat every 30s."""
        if self.mode == "reactive":
            return  # Do nothing proactively

        # OBSERVE: scan environment
        observations = self.observe()

        # ORIENT: classify opportunities
        opportunities = self.orient(observations)

        # DECIDE: filter by mode
        for opp in opportunities:
            if self.mode == "proactive_suggest":
                self.suggest(opp)  # Display to user, don't execute
            elif self.mode == "proactive_auto" and self.is_pre_approved(opp):
                self.execute(opp)  # Execute if pre-approved
            elif self.mode == "proactive_partner":
                if self.constitutional_gate(opp):
                    self.execute(opp)  # Full autonomy within gates

    def observe(self):
        """Scan: filesystem changes, calendar, email, code repos, system health."""
        observations = []

        # File system changes (last 30s)
        observations += self.scan_filesystem_changes()

        # Memory: what does the user usually do at this hour?
        hour_pattern = self.memory.user.active_hours
        peak_hour = max(range(24), key=lambda h: hour_pattern[h])

        # Rust feedback: any reinforcement/quarantine signals?
        feedback = self.bridge.poll_feedback() if self.bridge else {}
        if feedback.get("reinforce_pending", 0) > 0:
            observations.append({"type": "quality_improving"})
        if feedback.get("quarantine_pending", 0) > 0:
            observations.append({"type": "quality_declining"})

        return observations

    def suggest(self, opportunity):
        """Display suggestion to user without executing."""
        # Format: [P7 Oracle suggests] Organize your Downloads folder
        print(f"  [P7 Oracle suggests] {opportunity['description']}")

    def execute(self, opportunity):
        """Execute through the mission pipeline with receipt."""
        # Route through bizra-node RECEIVE → receipt chain
        pass

    def constitutional_gate(self, opportunity):
        """Daughter Test + Ihsan gate before any autonomous action."""
        # Guardian (P4) has VETO power
        # Every action must be safe for a child to witness
        return opportunity.get("risk_level", "high") in ("low", "medium")
```

## Observation Sources

| Source | What It Detects | Agent |
|--------|----------------|-------|
| Filesystem | New files in Downloads, modified docs | P2 Scholar |
| Calendar | Upcoming meetings, deadlines | P7 Oracle |
| Git repos | Uncommitted changes, failed CI | P3 Artisan |
| Email/Messages | Unread count, priority items | P6 Diplomat |
| System health | Disk space, memory, services | P4 Guardian |
| Work patterns | "You usually code at 6am Fajr" | P5 Mentor |

## Pre-Approved Actions (Mode 3)

```yaml
auto_approve:
  - type: organize_downloads     # Move files to correct folders
  - type: backup_modified_files  # Snapshot changed files
  - type: summarize_unread       # Digest new messages
  - type: run_tests              # After code changes detected
  - type: morning_briefing       # Auto-generate at wake time

never_auto:
  - type: delete_files           # Always requires confirmation
  - type: send_messages          # Always requires confirmation
  - type: financial_transactions # Always requires confirmation
  - type: modify_constitution    # Immutable without ceremony
```

## TDD Anchors

```python
def test_reactive_mode_does_nothing():
    engine = ProactiveEngine(mode="reactive")
    engine.tick(now())
    assert engine.suggestions == []
    assert engine.executions == []

def test_suggest_mode_suggests_not_executes():
    engine = ProactiveEngine(mode="proactive_suggest")
    engine.tick(now())
    assert len(engine.suggestions) >= 0
    assert engine.executions == []

def test_auto_mode_only_pre_approved():
    engine = ProactiveEngine(mode="proactive_auto")
    engine.auto_approve = ["organize_downloads"]
    engine.tick(now())
    for exec in engine.executions:
        assert exec.type in engine.auto_approve

def test_partner_mode_constitutional_gate():
    engine = ProactiveEngine(mode="proactive_partner")
    dangerous = {"type": "delete_files", "risk_level": "high"}
    assert not engine.constitutional_gate(dangerous)

def test_daughter_test():
    # Every autonomous action must pass the Daughter Test
    engine = ProactiveEngine(mode="proactive_partner")
    for action in engine.pending_actions():
        assert engine.daughter_test(action), f"Action {action} fails Daughter Test"
```

## Integration Points

- Heartbeat loop (`bizra-node/node.rs:heartbeat()`) triggers `tick()`
- Living Memory (`core/living_memory/brain.py`) provides user context
- RustEventBridge (`poll_feedback()`) provides nervous system signals
- Receipt chain records every proactive action
- SEED economy: proactive actions earn/cost SEED
