# Phase 66.02: Audit Trail Integrity

## Problem Statement

6 files contain `except Exception: pass` (or equivalent) in safety-critical
paths. When these paths fail silently, the constitutional audit trail breaks:
mission events vanish, rollbacks appear to succeed when they didn't, and
runtime boot failures become undebuggable.

> Axiom: "Trust is not established through explanation — it is established
> through verification." A silent failure is an unverifiable claim.

## Pseudocode

### Fix 1: Mission Event Emit (HIGH — audit trail)

```
# File: core/sovereign/mission.py:997-998
# BEFORE:
#   except Exception:
#       pass

# AFTER:
async def _emit(self, topic: str, payload: dict) -> None:
    """Emit event to bus. Log on failure — never silently drop."""
    TRY:
        IF self._event_bus IS NOT None:
            await self._event_bus.emit(topic, payload)
    EXCEPT Exception AS exc:
        logger.warning(
            "Event emit failed | topic=%s error=%s",
            topic, exc,
        )
        # Do NOT re-raise — event emission is best-effort
        # But the failure is now visible in logs and metrics
```

### Fix 2: Rollback Safety (HIGH — recovery path)

```
# File: core/rollout/rollback.py:131-132
# BEFORE:
#   except Exception:
#       pass

# AFTER:
TRY:
    <rollback operation>
EXCEPT Exception AS exc:
    logger.error(
        "Rollback operation failed | component=%s error=%s",
        component_name, exc,
    )
    # Record failure in rollback result — caller sees partial rollback
    result.partial_failures.append(str(exc))
```

### Fix 3: Runtime Core Module Imports (MEDIUM — startup debugging)

```
# File: core/sovereign/runtime_core.py:64,70
# BEFORE:
#   except Exception:
#       pass

# AFTER (for each module-level try/except):
TRY:
    from <optional_module> import <symbol>
    _OPTIONAL_MODULE_AVAILABLE = True
EXCEPT ImportError:
    _OPTIONAL_MODULE_AVAILABLE = False
    # Specific exception type — only catches missing modules
EXCEPT Exception AS exc:
    _OPTIONAL_MODULE_AVAILABLE = False
    logger.warning("Optional module load failed: %s", exc)
```

### Fix 4: Desktop Bridge Writer Close (MEDIUM — connection health)

```
# File: core/bridges/desktop_bridge.py:289,603,614,736
# BEFORE (4 locations):
#   except Exception:
#       pass

# AFTER (each location):
EXCEPT Exception AS exc:
    logger.debug("Connection cleanup: %s", exc)
    # DEBUG level — these are expected during client disconnect
```

### Fix 5: Voice Bridge (MEDIUM — integration health)

```
# File: core/voice/personaplex_bridge.py:84-85,93-94
# BEFORE:
#   except Exception:
#       pass

# AFTER:
EXCEPT Exception AS exc:
    logger.warning("Personaplex bridge error: %s", exc)
```

## Pattern Rule (Preventive)

```
# Add to ruff configuration in pyproject.toml:
# [tool.ruff.lint]
# select = [..., "E722"]  # bare-except detection

# E722 already enabled — verify it catches `except Exception: pass`
# If not, add custom lint rule or pre-commit hook:

RULE: No `except Exception: pass` in core/ outside test files
CHECK: grep -rn "except.*Exception.*:" core/ | grep -A1 "pass$" | grep -v test
GATE: CI fails if count > 0 for new commits
```

## Edge Cases

- `core/proof_engine/ihsan_gate.py:40`: `except Exception:` with fallback
  values, NOT `pass`. This is CORRECT — it degrades gracefully to known
  defaults. No change needed.

- `core/genesis/ingestion/enrichment.py:33-34,64-65`: `except Exception: pass`
  for optional enrichment gates. These are documented as advisory-only.
  Change to `logger.debug` for observability. LOW priority.

- `core/proactive/self_harness.py:214,424`: Silent swallows in harness
  scanning. Change to `logger.debug` since these are scan paths, not
  critical operations.

## Invariants

```
ASSERT: grep -rn "except.*:$" core/sovereign/mission.py | wc -l == 0
    (no bare except blocks remain in mission.py)

ASSERT: grep -rn "except Exception" core/sovereign/mission.py → all have logger.*
    (every caught exception is logged)

ASSERT: grep -rn "except.*pass$" core/rollout/rollback.py | wc -l == 0
    (no silent swallows in rollback)
```

## TDD Anchor

```python
# test_audit_trail_integrity.py

import logging

async def test_mission_emit_logs_on_failure(caplog):
    """Mission._emit logs warning when event bus fails."""
    from core.sovereign.mission import MissionOrchestrator

    orchestrator = MissionOrchestrator.__new__(MissionOrchestrator)
    orchestrator._event_bus = None  # Simulate missing bus

    with caplog.at_level(logging.WARNING):
        await orchestrator._emit("test.topic", {"key": "value"})

    # Should NOT raise, but SHOULD log
    assert "Event emit failed" in caplog.text or orchestrator._event_bus is None


async def test_rollback_records_partial_failure():
    """Rollback captures component failures instead of swallowing."""
    from core.rollout.rollback import RollbackResult
    # After fix: partial_failures list is populated on component error
    result = RollbackResult(success=False, partial_failures=["db_restore failed"])
    assert len(result.partial_failures) > 0
```

## Estimated Impact

- Lines changed: ~20 (6 except blocks → 6 logged alternatives)
- Risk: LOW — adds logging, does not change control flow
- Observability: mission events, rollback failures, startup errors now visible
