# Next Implementation Plan

From advisory library (v0.1) → preflight hook → optional CI advisory job →
eventually, a runtime gate the operator explicitly authorizes.

## Step 0 — Bed-down (done in v0.1)

- `tools/execution_flywheel/` scaffold: registry, guard, priority engine,
  runner, extractor, tests.
- `docs/engineering/execution_flywheel/` design notes.
- Five seed patterns captured from PR #49 and the P0+1 hardening session.
- All advisory; no runtime wiring.

## Step 1 — Operator-run evaluation

Wire the runner into the local loop:

```bash
python3 -m tools.execution_flywheel.flywheel_runner \
    --context .claude/flywheel_context.json --explain-summary
```

Context is produced by a thin shell wrapper (proposed location:
`scripts/ops/flywheel_context_build.sh` — **not** in v0.1) that emits JSON
from git state, CI rollup, and scanner output. The runner's output is a
witness the operator can save to a log.

## Step 2 — Preflight hook (Claude Code / Codex)

Register a pre-edit hook in `.claude/settings.json` (or the Codex
equivalent) that calls the runner and halts edits when the decision is
`ABORT` or (opt-in) `NEEDS_OPERATOR_CONFIRMATION`. Constraints:

- Hook is stdlib-only; no venv needed on fresh runners.
- Hook never posts model chain-of-thought to any endpoint; only reads
  observable state and trigger keywords the operator flagged.
- Hook failure is **advisory** by default; opt-in for fail-closed.
- Hook never calls the network; registry path is local.

## Step 3 — CI advisory job (non-blocking)

Add a GitHub Actions job that:

- Builds a `flywheel_context.json` from the PR diff + CI rollup.
- Runs `python3 -m tools.execution_flywheel.flywheel_runner`.
- Posts the result as a sticky PR comment (opt-in per repo).
- Never sets a required-check. Never blocks merge.

Exit criteria: at least four weeks of PR data showing the advisory
surface correlates with review outcomes.

## Step 4 — Pattern growth

Every session that ends with an operator correction is a pattern candidate:

1. Run the triage through `/C` or `/!` to extract the lesson in the
   existing conversation.
2. Add a new entry to `patterns.yaml` via PR.
3. Add at least one test in `tests/test_pre_action_guard.py` or
   `tests/test_priority_engine.py`.

Reject pattern PRs that do not include a test — no untested guardrail should
be allowed to fire.

## Step 5 — Cross-agent reuse

The JSON payload schema is agent-agnostic. Codex, local scripts, and future
multi-agent harnesses can invoke the runner with the same payload. The
`triggers_detected` field is the shared vocabulary; keep it small and
additive.

## Step 6 — MEMORY.md handoff (explicit-auth only)

When the operator authorizes it, a line can be added to `MEMORY.md` pointing
at the pattern registry. Until that authorization, the kernel is strictly
file-based under `tools/` and `docs/` and does not touch auto-memory.

## Exit criteria for v0.1 → v0.2

- At least eight patterns in the registry, each with ≥ 1 test.
- Operator has invoked the runner in a real session and either agreed with
  the decision or refined the pattern in response.
- A shell wrapper exists that produces the JSON context from git state + CI
  rollup; the model no longer hand-writes it.
- Advisory-only posture preserved; no runtime gate promoted.

## What v0.1 explicitly does NOT authorize

- No automated posting of GitHub comments, reviews, approvals, dismissals,
  or merges.
- No edits to `core/`, `bizra-omega/`, `runtime/` triggered by the kernel.
- No writes to `MEMORY.md`, canon packs, Origin Kernel docs, or
  launch/brand directories.
- No serialisation of session-specific chain-of-thought into the pattern
  registry.
