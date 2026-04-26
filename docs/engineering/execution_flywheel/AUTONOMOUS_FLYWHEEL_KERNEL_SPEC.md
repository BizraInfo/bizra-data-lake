# BIZRA Autonomous Flywheel Kernel — v0.1 Specification

## Purpose

Close the loop between engineering events and reusable guardrails:

```
Signal → Root Cause → Fix → Test → Validate → Document → Encode → Repeat
```

The kernel is the *encode* step. Everything upstream is human/AI engineering
work; the kernel persists lessons so they do not need to be re-derived.

## Components

### Pattern Registry (`pattern_registry.py`)

- Loads `patterns.yaml` via a stdlib-only subset parser.
- Validates every pattern against the schema (required fields, valid
  severity, valid `default_decision`).
- Exposes `list_patterns`, `get_pattern`, `query_by_trigger`.
- Advisory only; loading does not execute anything.

### Pre-Action Guard (`pre_action_guard.py`)

- Input: `ActionContext` (action type, target files, triggers detected,
  metadata).
- Output: `GuardDecision` (`PROCEED` | `REVALIDATE` | `NEEDS_OPERATOR_CONFIRMATION` |
  `ABORT`) with reason and matched pattern IDs.
- Never mutates files. Never calls git or GitHub. Never performs I/O beyond
  reading the registry.

### Adaptive Priority Engine (`priority_engine.py`)

- Input: a dict of observable system state (audit, CI, security, claims).
- Output: `PrioritySignal` (priority lane, reason, confidence, evidence).
- Seven lanes, first-match-wins; see `ADAPTIVE_PRIORITY_ENGINE_SPEC.md`.

### Flywheel Runner (`flywheel_runner.py`)

- Combines guard + priority into one `FlywheelResult`.
- Accepts a JSON context with an optional `priority_context` sub-dict.
- CLI: `python3 -m tools.execution_flywheel.flywheel_runner --context <path|->`
  optionally with `--explain-summary` for human-readable output.
- Advisory only. Never executes destructive actions. Never calls external
  services. Never mutates runtime.

## Flow

1. **Signal** — a session observes a new failure/correction/success.
2. **Root cause** — human or AI identifies the mechanism that made the signal
   possible.
3. **Fix** — code change lands via normal PR workflow.
4. **Test** — regression tests lock the fix.
5. **Validate** — CI / audit / operator confirms.
6. **Document** — session recap captures the lesson in plain prose.
7. **Encode** — a new `Pattern` is added to `patterns.yaml` (PR-reviewed)
   with triggers, risks, guard actions, severity, optional
   `default_decision`, and a test in `tests/test_pre_action_guard.py` (or
   `tests/test_priority_engine.py` for priority lanes).
8. **Repeat** — next time the trigger appears, the kernel returns a decision
   without needing to re-derive the lesson.

## Guarantees

- **Purity.** `evaluate()`, `recommend_priority()`, `run_flywheel()` are pure
  functions of input.
- **Schema-validated on load.** `load_patterns()` raises on malformed
  entries; the kernel cannot silently ingest a broken rule.
- **Decision-value validated on construction.** `GuardDecision` and
  `PrioritySignal` reject any string outside their `VALID_*` tuples.
- **No network I/O.** The kernel reads a file; it does not fetch, post, or
  sign anything.

## Non-goals

- Not a runtime gate. Runtime enforcement, if ever wired, is out of v0.1.
- Not a compliance engine. It encodes what the operator wrote; it does not
  prove sufficiency.
- Not a replacement for code review. It accelerates one step (avoiding
  repeat mistakes), not the whole.

## Seed patterns (v0.1)

| ID | Source | Severity | Default decision |
|----|--------|----------|-------------------|
| `PR_REVIEW_STALE_SHA_VERIFY_ORIGIN_BEFORE_EDIT` | PR #49 | critical | (critical fallback) |
| `AUDIT_YAML_INLINE_COMMENT_PARSE_FAILURE` | P0+1 hardening | high | `REVALIDATE` |
| `SECRET_SCANNER_SNR_NOISE_COLLAPSE` | P0+1 hardening | high | `REVALIDATE` |
| `DEV_DEFAULT_CREDENTIAL_FALLBACK_TRUTH_DEBT` | P0+1 hardening | critical | `ABORT` |
| `BOTTLENECK_SHIFT_AFTER_SECRET_GATE_CLEARS` | P0+1 hardening | high | `REVALIDATE` |

## Reserved v0.2+ work

See `NEXT_IMPLEMENTATION_PLAN.md`. Highlights:

- Preflight hook for Claude / Codex sessions.
- Optional CI advisory job (never blocking).
- Operator policy: ABORT counts, REVALIDATE latency, pattern growth rate.
