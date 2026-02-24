# Self-Healing Workflows

> Automatic error detection and recovery for resilient development

## Overview

The self-healing system monitors for errors, recognizes patterns, and suggests or applies fixes automatically. This reduces debugging time and enables continuous development flow.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Command Execution                             │
│  (pytest, cargo, python scripts)                                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Hook Layer                                    │
│  ├─ self-healing.sh      (core detection)                       │
│  ├─ post-test.sh         (pytest analysis)                      │
│  └─ post-rust.sh         (cargo analysis)                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Pattern Database                              │
│  ├─ error-patterns.json  (known errors)                         │
│  └─ recovery-playbook.json (fix procedures)                     │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Recovery Actions                              │
│  ├─ Auto-fix (search/replace)                                   │
│  ├─ Guided steps (manual intervention)                          │
│  └─ Learning (new patterns saved)                               │
└─────────────────────────────────────────────────────────────────┘
```

## Error Patterns

### Import Errors

**Pattern: `import-autonomy-level`**
```
ImportError: cannot import name 'AutonomyLevel' from 'core.sovereign.autonomy'
```

**Auto-Fix:**
```python
# Change from:
from .autonomy import AutonomyLevel

# To:
from .autonomy_matrix import AutonomyLevel
```

---

**Pattern: `event-field-names`**
```
TypeError: Event.__init__() got an unexpected keyword argument 'event_type'
```

**Auto-Fix:**
```python
# Change from:
Event(event_type="topic", data={"key": "value"})

# To:
Event(topic="topic", payload={"key": "value"})
```

### Dependency Errors

**Pattern: `missing-aiohttp`**
```
ModuleNotFoundError: No module named 'aiohttp'
```

**Auto-Fix:**
```python
# Add optional import pattern:
try:
    import aiohttp
except ImportError:
    import httpx  # or use urllib fallback
```

### Async Errors

**Pattern: `eventbus-stop-not-async`**
```
TypeError: object NoneType can't be used in 'await' expression
```

**Auto-Fix:**
```python
# Change from:
await event_bus.stop()

# To:
event_bus.stop()  # stop() is sync, not async
```

### Rust Errors

**Pattern: `rust-unused-import`**
```
warning: unused import: `SomeType`
```

**Auto-Fix:**
```bash
cargo fix --lib -p crate_name --allow-dirty
```

## Recovery Playbooks

### Test Collection Failure

**Trigger:** `ERROR collecting.*ImportError`

**Steps:**
1. Identify missing import in error message
2. Search `__init__.py` for the symbol
3. Find correct source module with `grep`
4. Update import path

### Rust Compilation Failure

**Trigger:** `error[E0XXX]`

**Error Code Reference:**

| Code | Issue | Fix |
|------|-------|-----|
| E0382 | Use after move | Add `.clone()` or use reference |
| E0502 | Borrow conflict | Restructure code |
| E0308 | Type mismatch | Check types |
| E0433 | Unresolved import | Check Cargo.toml |
| E0599 | Method not found | Check trait imports |

### Async/Await Mismatch

**Trigger:** `can't be used in 'await' expression`

**Steps:**
1. Find the method being awaited
2. Check if method signature is `async def` or `def`
3. Remove `await` if sync, or add `async` if should be async

### Missing Dependency

**Trigger:** `ModuleNotFoundError`

**Decision Tree:**
```
Is module required?
├─ Yes → pip install module_name
└─ No  → Add try/except with fallback
```

## Hook Configuration

### self-healing.sh

Core error detection and fix suggestion.

```bash
#!/bin/bash
# Usage: source self-healing.sh
# Then call: main $exit_code "$output"

detect_error_type "$output"
suggest_fix "$error_type"
```

### post-test.sh

Analyzes pytest output.

```bash
# Detects:
# - Collection errors (import issues)
# - Assertion failures
# - Timeout errors

analyze_pytest "$output" "$exit_code"
```

### post-rust.sh

Analyzes cargo output.

```bash
# Detects:
# - Missing crates
# - Type mismatches
# - Borrow checker errors
# - Trait bound issues

analyze_cargo "$output" "$exit_code"
```

## Memory Files

### error-patterns.json

```json
{
  "patterns": [
    {
      "id": "pattern-name",
      "error_regex": "regex to match",
      "category": "import|dependency|async|rust",
      "severity": "high|medium|low",
      "auto_fix": {
        "search": "text to find",
        "replace": "replacement text"
      },
      "occurrences": 1,
      "last_seen": "2026-02-04"
    }
  ]
}
```

### recovery-playbook.json

```json
{
  "playbooks": {
    "playbook-name": {
      "trigger": "regex pattern",
      "severity": "critical|high|medium|low",
      "steps": [
        {"action": "identify", "command": "grep ..."},
        {"action": "fix", "description": "how to fix"}
      ],
      "success_indicators": ["passed", "✓"]
    }
  }
}
```

## Learning Loop

When errors occur multiple times, patterns are automatically saved.

```
Error occurs → Detect pattern → Suggest fix → User applies fix
                                     ↓
                              Fix successful?
                              ├─ Yes → Save pattern to database
                              └─ No  → Log for manual review
```

### Configuration

```json
{
  "learning": {
    "store_new_patterns": true,
    "pattern_file": ".claude-flow/memory/error-patterns.json",
    "min_occurrences_to_save": 2
  }
}
```

## Usage Examples

### Manual Invocation

```bash
# Source the hooks
source .claude/hooks/self-healing.sh

# Analyze test output
output=$(pytest tests/ 2>&1)
exit_code=$?
main $exit_code "$output"
```

### Automatic (with Claude Code hooks)

```json
{
  "PostToolUse": [
    {
      "matcher": "^Bash$",
      "command": "source .claude/hooks/self-healing.sh && main '${tool.result.exitCode}' '${tool.result.output}'"
    }
  ]
}
```

## Supported Error Categories

| Category | Auto-Recovery | Examples |
|----------|---------------|----------|
| Import | ✅ | Wrong module path |
| Dependency | ✅ | Missing module |
| Async | ✅ | await/sync mismatch |
| Rust | 🔧 Partial | Unused imports |
| Test | 🔍 Guided | Assertion failures |
| Timeout | 🔍 Guided | Slow operations |

## File Locations

```
.claude/hooks/
├── self-healing.sh      # Core detection
├── post-test.sh         # Pytest analysis
└── post-rust.sh         # Cargo analysis

.claude-flow/memory/
├── error-patterns.json  # Known patterns
├── recovery-playbook.json # Fix procedures
├── error-log.txt        # Error history
└── test-history.log     # Test run log
```

## Best Practices

1. **Let patterns accumulate** — The system learns from repeated errors
2. **Review auto-fixes** — Verify suggested changes before applying
3. **Add custom patterns** — Extend `error-patterns.json` for project-specific issues
4. **Check logs** — Review `error-log.txt` for recurring problems

---

*Resilient development through automated recovery*
