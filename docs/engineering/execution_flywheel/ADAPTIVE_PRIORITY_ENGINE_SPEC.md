# Adaptive Priority Engine — v0.1 Specification

## Purpose

Detect when one bottleneck is cleared and recommend the next. Advisory only;
never triggers actions.

## Contract

### Input — `priority_context` dict

| Field | Type | Meaning |
|-------|------|---------|
| `secret_findings` | int | Count of scanner-reported secret hits |
| `rotation_required` | bool | A credential rotation is open |
| `runtime_defaults_insecure` | bool | Committed dev-default credential fallback exists |
| `main_branch_red` | bool | Trunk CI is not green |
| `ci_failing_count` | int | Count of failing required checks |
| `dependency_vulnerabilities` | int | Count of high/critical CVEs in deps |
| `sbom_stale` | bool | SBOM artefact is older than policy window |
| `public_claims_risky` | bool | Public claims include prohibited/proof-required items |
| `node0_activation_blocked_rows` | int | Closure scoreboard rows still blocked |

All fields are optional (default: `0` / `False`).

### Output — `PrioritySignal`

```python
@dataclass
class PrioritySignal:
    priority: str       # one of VALID_PRIORITIES
    reason: str
    confidence: float   # 0.0 – 1.0
    evidence: list[str] # which context fields fired the rule
```

## Priority lattice (first match wins)

| # | Lane | Condition | Confidence |
|---|------|-----------|------------|
| 1 | `SECURITY` | `secret_findings > 0` OR `rotation_required` | 0.95 |
| 2 | `RUNTIME_HARDENING` | `runtime_defaults_insecure` | 0.90 |
| 3 | `CI_BASELINE` | `main_branch_red` OR `ci_failing_count > 0` | 0.85 |
| 4 | `SUPPLY_CHAIN` | `dependency_vulnerabilities > 0` OR `sbom_stale` | 0.85 |
| 5 | `PUBLIC_CLAIMS` | Secret gate clear AND `public_claims_risky` | 0.85 |
| 6 | `NODE0_ACTIVATION` | `node0_activation_blocked_rows > 0` | 0.75 |
| 7 | `STOP_AND_LAND` | All axes clean | 0.70 |

## Design rationale

- **Security before everything.** A leaked credential poisons every
  downstream claim; reputational cost scales with time-to-rotation.
- **Runtime hardening before CI/supply.** An insecure default can be shipped
  even when CI is green; CI doesn't catch truth debt that isn't tested.
- **CI before supply.** Trunk CI must be green before dependency work, or
  regressions hide behind red CI.
- **Public claims after secrets clear.** The P0+1 hardening surfaced that
  once secret findings are zero, public-claim discipline becomes the next
  reputational bottleneck.
- **Node0 activation is non-urgent but non-zero.** Closure rows sit above
  stop-and-land because the operator explicitly values that program.
- **Stop-and-land wins over make-work.** If no axis is red, the
  recommendation is to *stop and land*, not to invent new work.

## Changing the lattice

Modify `priority_engine.recommend_priority()`. Every rule change MUST ship
with at least one new test in `tests/test_priority_engine.py`. The
`VALID_PRIORITIES` tuple on `PrioritySignal` must include any new lane name.
