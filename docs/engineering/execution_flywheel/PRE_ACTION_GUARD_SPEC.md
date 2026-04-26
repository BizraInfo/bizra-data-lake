# Pre-Action Guard — v0.1 Specification

## Contract

### Input — `ActionContext`

```python
@dataclass
class ActionContext:
    action_type: str
    target_files: list[str]
    triggers_detected: list[str]
    metadata: dict[str, Any]
```

Canonical JSON payload accepted by the CLI (`pre_action_guard --context`):

```json
{
  "action_type": "edit_file",
  "target_files": ["core/bus/subscribers.py"],
  "triggers_detected": [
    "review_requests_change",
    "pr_has_commits_after_reviewed_sha"
  ],
  "metadata": {
    "fix_already_present": true,
    "reviewed_sha": "34b27bec",
    "head_sha": "c09cc95c",
    "pr_number": 49
  }
}
```

### Output — `GuardDecision`

```python
@dataclass
class GuardDecision:
    decision: str              # PROCEED | REVALIDATE | NEEDS_OPERATOR_CONFIRMATION | ABORT
    reason: str
    matched_patterns: list[str]
```

## Decision rules

Multiple candidates may apply; the strongest by precedence wins.

**Precedence:** `ABORT` > `REVALIDATE` > `NEEDS_OPERATOR_CONFIRMATION` > `PROCEED`.

| # | Condition | Candidate |
|---|-----------|-----------|
| 1 | No pattern's triggers overlap with `triggers_detected` | `PROCEED` (final — short-circuits) |
| 2 | `metadata.fix_already_present == True` | `ABORT` |
| 3 | `metadata.reviewed_sha` and `metadata.head_sha` are both set and differ | `REVALIDATE` |
| 4 | Any matched pattern declares `default_decision` | That value |
| 5 | No candidates yet + any matched pattern has `severity=="critical"` | `NEEDS_OPERATOR_CONFIRMATION` |
| 6 | No candidates at all | `PROCEED` |

## Guarantees

- **Pure function.** Same input → same output.
- **Fail-closed on registry errors.** If `patterns.yaml` fails validation,
  `load_patterns()` raises.
- **Decision validated on construction.** `GuardDecision.__post_init__`
  rejects strings outside `VALID_DECISIONS`.

## Example flows

### A. Fix already upstream (PR #49)

`triggers_detected`: `["review_requests_change"]`,
`metadata.fix_already_present = True`
→ **ABORT** (candidate 2 wins outright).

### B. Stale review SHA

`triggers_detected`: `["review_requests_change"]`,
`metadata.reviewed_sha = "34b27bec"`, `metadata.head_sha = "c09cc95c"`
→ **REVALIDATE** (candidate 3; pattern has no `default_decision` stronger).

### C. Credential fallback detected

`triggers_detected`: `["default_dsn_or_redis_or_neo4j_fallback"]`
→ **ABORT** (pattern's `default_decision = ABORT`, candidate 4).

### D. Scanner noise

`triggers_detected`: `["high_secret_finding_count", "self_scan_matches"]`
→ **REVALIDATE** (pattern's `default_decision = REVALIDATE`).

### E. YAML parse crash + SHA mismatch

`triggers_detected`: `["audit_engine_crash"]`,
`metadata.reviewed_sha`, `metadata.head_sha` set
→ **REVALIDATE** (both SHA-mismatch and pattern default agree).

### F. Critical pattern, ambiguous metadata

`triggers_detected`: `["review_requests_change"]`, no metadata signals, no
pattern-level `default_decision`
→ **NEEDS_OPERATOR_CONFIRMATION** (candidate 5).

### G. Unrelated context

`triggers_detected`: `["docs_update"]`, no pattern matches
→ **PROCEED** (candidate 1, short-circuit).

## Non-goals

- Not a CI gate. Runs *before* model-driven edits, not as a post-commit check.
- Not a substitute for operator judgment. Surfaces known-bad actions;
  judgment lives with the human.
