# BIZRA Reflex Matrix

Date: 2026-03-12  
Scope: root repo + `bizra-node0`  
Verdict plane: `truth_repetition -> deterministic_reflex`

## Purpose

This matrix is intentionally separate from the enforcement matrix. Deterministic reflex is a second-stage optimization concern and is not a prerequisite for first-stage canonicality.

## Matrix

| Surface | truth_repetition | deterministic_reflex | reflex_verdict | Primary evidence |
| --- | --- | --- | --- | --- |
| LearningLoopOrchestrator | `partial` | `partial` | `partial` | `core/orchestration/learning_loop.py` |
| SDPOReflexBridge | `live` | `partial` | `partial` | `core/sdpo/reflex_bridge.py` |
| `/v1/plan` System-1 fast path | `partial` | `partial` | `partial` | `core/sovereign/api.py`, `tests/integration/test_plan_endpoint.py` |
| Node0 breathe / Helix3 precipitation surfaces | `partial` | `partial` | `partial` | `core/node0/heartbeat.py`, `core/sovereign/runtime_core.py` |
| Aurelle / execution-style transcript narrative | `narrative_only` | `narrative_only` | `contradicted` | Narrative claims outrun stronger code and artifact evidence |

## Surface Notes

### What is clearly real

- Repetition tracking and reflex-candidate extraction are real in `SDPOReflexBridge`.
- The repo does have a meaningful optimization plane rather than only narrative language about reflex.

### What remains partial

- Closed-loop learning is feature-flagged rather than universal default behavior.
- The authoritative path is not yet proven to compile only verified receipts into deterministic reflex under all conditions.
- Deterministic fast-path guarantees remain weaker than the enforcement spine.

### What should be downgraded

- “Reflex canon complete”
- “Deterministic reflex fully proven”
- “Optimization maturity equals enforcement maturity”

## Reflex Verdict

**`reflex_verdict = partial`**

The strongest truthful statement is:

**BIZRA has real repetition-tracking and reflex-candidate infrastructure, but deterministic reflex remains materially less proven than the enforcement spine.**
