# Integration Boundary

This document names exactly what the kernel WILL and WILL NOT touch, and
what must be true before any further integration.

## What the kernel touches (v0.1)

- Files inside `tools/execution_flywheel/` (source, registry, tests).
- Files inside `docs/engineering/execution_flywheel/` (this documentation).
- Process stdin/stdout (when invoked as a CLI).
- Disk reads of the patterns file passed to `--patterns`.

## What the kernel never touches

- `core/` (Python runtime), `bizra-omega/` (Rust runtime), `runtime/` (any
  subtree). The kernel is advisory; runtime code never imports it.
- `MEMORY.md` and any file under
  `/home/bizra-operating-system/.claude/projects/*/memory/`.
- Canon packs under `docs/canon/`.
- Origin Kernel or `BIZRA_ORIGIN_KERNEL.md`.
- `docs/brand/`, `docs/strategy/`, P0.2 website artefacts.
- The bizra.ai website source or build.
- Git state: no `add`, no `commit`, no `push`, no `branch`, no `tag`, no
  `checkout`.
- GitHub state: no PR comments, reviews, merges, dismissals, or workflow
  dispatches.
- Network: no HTTP, no DNS beyond hostname→IP the OS already resolves.
- Runtime secrets: no env-var reads for secret values; no log output
  containing credentials.

## Required before each subsequent integration step

1. **Pattern PR.** Every new pattern must land via PR with at least one
   paired test.
2. **Operator review.** Every new integration path (preflight hook, CI
   advisory job) must be reviewed *separately* from pattern content.
3. **No fail-closed default without opt-in.** If the kernel is ever wired
   into a hook, its default posture is advisory. Fail-closed mode is
   per-operator opt-in.
4. **Evidence chain preserved.** Every `ABORT` / `REVALIDATE` a hook emits
   must carry its matched pattern IDs and the observable fields that
   triggered the decision.

## Explicit prohibitions

- Do not serialise any of the model's internal deliberation into `patterns.yaml`
  or any artefact the kernel produces.
- Do not write pattern entries whose only evidence is "the model has learned
  this." Patterns are derived from *observable* events.
- Do not let a pattern carry runtime side effects. Guard actions are
  instructions for the operator or the hook, not executable code.
- Do not share raw secret content or redacted previews in pattern source
  fields. If a pattern references a secret-related incident, cite the
  addendum, not the secret.

## What would unlock runtime integration (out of v0.1)

- At least 10 patterns with paired tests.
- An ABORT / REVALIDATE telemetry stream from an advisory hook over a
  multi-week window.
- A signed operator decision to raise the kernel's authority from advisory
  to gate.

None of the above is assumed or requested in v0.1.
