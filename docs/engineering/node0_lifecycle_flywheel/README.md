# Node0 Lifecycle Flywheel

This is the operator-facing harness for closing the BIZRA Node0 standalone
lifecycle without turning advisory tooling into hidden runtime authority.

It connects existing assets:

| Source | Role |
| --- | --- |
| `scripts/node0_standalone.py` | Canonical MVSA activation, proof, health, task receipt path |
| `sovereign_state/node0_lifecycle.json` | Lifecycle v2 state and the 11 status-determining gates |
| `tools/audit/flywheel_kernel` | Audit guard state: secrets, truth claims, supply chain, runtime risk |
| `tools/execution_flywheel` | Adaptive priority signal over observable system state |
| `tools/node0_lifecycle_flywheel` | Deterministic loop receipt and next-action selector |

## Loop

```text
Observe -> Guard -> Prioritize -> Recommend -> Act(optional) -> Recheck -> Receipt -> Encode
```

Default mode is read-only. It does not activate Node0, write lifecycle files,
or run missions unless the operator passes `--execute-next`.

## Commands

Dry-run receipt:

```bash
python -m tools.node0_lifecycle_flywheel.closed_loop
```

Write a receipt artifact:

```bash
python -m tools.node0_lifecycle_flywheel.closed_loop \
  --out /tmp/bizra_node0_lifecycle_flywheel_receipt.json
```

Execute exactly one recommended lifecycle action:

```bash
python -m tools.node0_lifecycle_flywheel.closed_loop --execute-next
```

Strict gate for operator scripts:

```bash
python -m tools.node0_lifecycle_flywheel.closed_loop --strict
```

## Decision Order

The harness ranks Node0 lifecycle closure in this order:

| Decision | Condition | Recommended command |
| --- | --- | --- |
| `NODE0_ACTIVATE` | Lifecycle missing or `genesis_authority_valid=false` | `python scripts/node0_standalone.py activate --architect MoMo` |
| `NODE0_PROVE_MVSA` | MVSA bootstrap/self-validation gates are false | `python scripts/node0_standalone.py prove-mvsa` |
| `NODE0_RECEIPT_MISSION` | MVSA is proven but no mission evidence receipt exists | `python scripts/node0_standalone.py task ... --browser-mode mock` |
| `NODE0_REFRESH_RESTART_RECOVERY` | Mission exists but restart recovery is not ready | `python scripts/node0_standalone.py prove-mvsa` |
| `NODE0_MONITOR_AND_RELOOP` | All 11 gates are ready | `python scripts/node0_standalone.py health` |
| `NODE0_RECONCILE_DEGRADED_STATE` | Degraded state does not map to a known gate | inspect health/lifecycle and encode a new pattern |

## Boundaries

- No network calls are required by the harness itself.
- No git, GitHub, or external service state is mutated.
- Dry-run mode only reads JSON artifacts and source-local audit state.
- `--execute-next` delegates to `scripts/node0_standalone.py` for the single
  selected action. The standalone manager remains the authority for lifecycle
  mutation.
- Repeated unresolved gates should be encoded as new
  `tools/execution_flywheel/patterns.yaml` entries with paired tests.

## Why This Closes The Loop

Node0 already had the parts: activation, MVSA proof, mission receipts, restart
recovery, audit guards, and priority signals. The missing piece was a small
deterministic harness that could repeatedly answer:

1. What is the observable lifecycle state now?
2. Which guard or gate blocks closure?
3. What is the single next action?
4. Did that action move the state?
5. What pattern should be encoded if it did not?

That is the self-improving flywheel: evidence first, one action, recheck, then
encode the lesson.
