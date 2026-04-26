# BIZRA Autonomous Flywheel Kernel v0.1

A tiny, stdlib-only engineering-control layer that turns observed execution
lessons into reusable guardrails and adaptive priority signals. Advisory
only. No runtime behavior. Witness decisions, never enact them.

```
Signal → Root Cause → Fix → Test → Validate → Document → Encode → Repeat
```

## Subsystems

| Module | Role |
|--------|------|
| `schemas.py` | Dataclasses: `Trigger`, `Pattern`, `ActionContext`, `GuardDecision`, `PrioritySignal`, `FlywheelResult`. No `pydantic`. |
| `pattern_registry.py` | YAML-subset loader, validation, query helpers (`load_patterns`, `list_patterns`, `get_pattern`, `query_by_trigger`). |
| `pre_action_guard.py` | Evaluates an `ActionContext` against patterns; returns `PROCEED`/`REVALIDATE`/`ABORT`/`NEEDS_OPERATOR_CONFIRMATION`. |
| `priority_engine.py` | Maps observable audit/CI/security/claims state to one of 7 priority lanes. |
| `flywheel_runner.py` | Single-entry CLI: guard + priority in one JSON result. Supports `--explain-summary`. |
| `extract_patterns_from_p0_plus_1.py` | Emits the five seed patterns as JSON (default dry-run). |
| `patterns.yaml` | Human-editable pattern registry. |

## Scope

**Tools-only lane.** Never touches runtime code, canon packs, `MEMORY.md`, PR
files, git state, or the network. Decisions are pure functions of input and
registry.

## Guard decisions

| Decision | Meaning |
|----------|---------|
| `PROCEED` | No matching pattern raised a halt. Safe to continue. |
| `REVALIDATE` | Evidence is incomplete; a specific check (inspect origin, diff YAML, etc.) must run before edit. |
| `NEEDS_OPERATOR_CONFIRMATION` | Critical-severity pattern matched but metadata is ambiguous; human judgment required. |
| `ABORT` | A pattern asserts this edit is known-bad (fix already landed, committed credential fallback, etc.). |

Precedence: `ABORT` > `REVALIDATE` > `NEEDS_OPERATOR_CONFIRMATION` > `PROCEED`.

## Priority lanes

| Lane | Fires when |
|------|------------|
| `SECURITY` | `secret_findings > 0` or `rotation_required` |
| `RUNTIME_HARDENING` | `runtime_defaults_insecure` |
| `CI_BASELINE` | `main_branch_red` or `ci_failing_count > 0` |
| `SUPPLY_CHAIN` | `dependency_vulnerabilities > 0` or `sbom_stale` |
| `PUBLIC_CLAIMS` | secret gate clear + `public_claims_risky` |
| `NODE0_ACTIVATION` | `node0_activation_blocked_rows > 0` |
| `STOP_AND_LAND` | all observable axes clean |

First match wins; the order is operator-reviewed in `priority_engine.py`.

## Quick start

```bash
# Tests
python3 -m pytest tools/execution_flywheel/tests -q          # pytest (if available)
python3 -m unittest discover -s tools/execution_flywheel/tests -v  # fallback

# Pattern registry introspection
python3 -m tools.execution_flywheel.pattern_registry
python3 -m tools.execution_flywheel.pattern_registry --query review_requests_change
python3 -m tools.execution_flywheel.pattern_registry --id DEV_DEFAULT_CREDENTIAL_FALLBACK_TRUTH_DEBT

# Guard a proposed edit
echo '{
  "action_type": "edit_file",
  "target_files": ["core/bus/subscribers.py"],
  "triggers_detected": ["review_requests_change"],
  "metadata": {"fix_already_present": true}
}' | python3 -m tools.execution_flywheel.pre_action_guard --context -

# Priority recommendation
echo '{"secret_findings": 0, "public_claims_risky": true}' \
  | python3 -m tools.execution_flywheel.priority_engine --context -

# Full kernel run (guard + priority)
echo '{
  "triggers_detected": ["review_requests_change"],
  "metadata": {"fix_already_present": true},
  "priority_context": {"secret_findings": 0, "public_claims_risky": true}
}' | python3 -m tools.execution_flywheel.flywheel_runner --context - --explain-summary
```

## Why YAML with a stdlib parser

Patterns are reviewed in PR diffs; YAML is lower-noise than JSON for human
review. Because `tools/` may not depend on third-party packages, the loader
uses a hand-written subset parser (`pattern_registry.parse_minimal_yaml`).
Values with `#` or `:` must be quoted.

## Constitutional posture

- **Law of Assumption** — default trust in what the model "remembers" about a
  PR, a config, or a runtime default is *insufficient*. Guard rules force a
  round-trip to observable evidence before action.
- **Ihsan** — when uncertainty is unavoidable, the kernel requires careful,
  declared, reversible action (`REVALIDATE` over `PROCEED`, `ABORT` over
  "just try").
- **SNR** — *signal* is actionable evidence-backed insight encoded as a
  pattern; *noise* is speculative or duplicated implementation detail. Only
  signals earn a place in the registry.
- **HHMM** (Higher-Higher Meta-Model) — high-level posture → subsystem state →
  failure/opportunity state → action state. Guard/priority separation mirrors
  this stack.

See `docs/engineering/execution_flywheel/` for full specs and the integration
boundary.
