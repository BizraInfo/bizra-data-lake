# BIZRA Autonomous Flywheel Kernel — Engineering Notes

The kernel is a tools-layer, advisory-only engineering-control system for
turning observed execution lessons into reusable guardrails and adaptive
priority signals. The code lives at `tools/execution_flywheel/`; this
directory holds the design notes.

## Index

- [`AUTONOMOUS_FLYWHEEL_KERNEL_SPEC.md`](./AUTONOMOUS_FLYWHEEL_KERNEL_SPEC.md)
  — end-to-end specification for the kernel (schema, loop, guarantees).
- [`PRE_ACTION_GUARD_SPEC.md`](./PRE_ACTION_GUARD_SPEC.md) — guard decision
  rules, data contract, examples.
- [`ADAPTIVE_PRIORITY_ENGINE_SPEC.md`](./ADAPTIVE_PRIORITY_ENGINE_SPEC.md) —
  priority lattice and evidence contracts.
- [`PR49_PATTERN_EXTRACTION.md`](./PR49_PATTERN_EXTRACTION.md) — how the first
  pattern was extracted from the PR #49 triage session.
- [`P0_PLUS_1_PATTERN_EXTRACTION.md`](./P0_PLUS_1_PATTERN_EXTRACTION.md) — how
  four additional patterns were extracted from the P0+1 hardening addendum.
- [`INTEGRATION_BOUNDARY.md`](./INTEGRATION_BOUNDARY.md) — what the kernel
  will NOT touch, what must be true before runtime integration.
- [`NEXT_IMPLEMENTATION_PLAN.md`](./NEXT_IMPLEMENTATION_PLAN.md) — proposed
  path from advisory library to preflight hook to CI advisory job.

## Design tenets

1. **Tools-only.** The kernel produces decisions; runtime never imports it.
2. **Stdlib-only.** No third-party deps → zero supply-chain risk added.
3. **Deterministic.** Given a fixed registry + context, result is
   reproducible. No LLM in the loop; no hidden state.
4. **Witness-first.** Every `ABORT`/`REVALIDATE` carries the matched pattern
   IDs and observable evidence it cites.
5. **No private reasoning exposed.** Patterns describe public preconditions
   and consequences. The kernel never serialises model chain-of-thought.
