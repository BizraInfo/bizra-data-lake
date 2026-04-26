# PR #49 Pattern Extraction

## Summary

During triage of PR #49 on branch `prep/node0-closure-replay-fate-gate`, the
model initially planned to edit `core/bus/subscribers.py` and
`tests/core/bus/test_subscriber_integration.py` in response to a CodeRabbit
`CHANGES_REQUESTED` review. Verification against `origin/<branch>` showed the
requested change was already landed on the PR head (commit `c09cc95c`), two
commits past the SHA CodeRabbit had reviewed (`34b27bec`). Applying the edit
would have forked the fix onto a sibling working branch.

The correct behaviour was to:

1. Fetch the origin PR branch.
2. Read `origin/<branch>:<file>` and confirm whether the fix was already in
   place.
3. Compare the `reviewDecision`'s reviewed SHA against the PR head SHA.
4. Abort the edit when the requested change was already present.

## Pattern

```
pattern_id:     PR_REVIEW_STALE_SHA_VERIFY_ORIGIN_BEFORE_EDIT
name:           Verify origin branch before editing from review feedback
severity:       critical

triggers:
  - review_requests_change
  - pr_has_commits_after_reviewed_sha
  - local_branch_differs_from_pr_head

risks:
  - duplicate fix
  - divergent implementation
  - polluted working tree
  - stale review loop

guard_actions:
  - fetch origin PR branch
  - inspect origin/<branch>:<file>
  - compare reviewed SHA against PR head SHA
  - abort edit if requested change already exists

source:
  - PR #49
  - CodeRabbit stale CHANGES_REQUESTED
  - fix commit c09cc95c already pushed
```

Generated programmatically by
`tools/execution_flywheel/extract_patterns_from_p0_plus_1.py` and stored in
`patterns.yaml`.

## What the extraction deliberately excludes

- **No private chain-of-thought.** Patterns describe observable preconditions
  (SHAs differ, review requests a change) and operator-facing outcomes
  (`ABORT` / `REVALIDATE`). The model's internal reasoning about whether to
  believe a review is not captured.
- **No session-specific state.** Commit SHAs and PR numbers appear only under
  `source:` for provenance; they are not part of the trigger contract.
- **No automation authority.** The pattern can only return `ABORT`; taking an
  action based on that verdict (e.g., posting a re-review request) requires
  a separate operator-granted authorization.

## How this becomes a flywheel

Every future PR-review session that matches the trigger set gets a cheap,
deterministic pre-check. If the check `ABORT`s, the model didn't have to
re-derive the lesson. If the operator overrides the `ABORT`, the pattern can
be refined (add another trigger, widen guard actions) so the next session is
sharper. New patterns seed new triggers. The registry compounds.
