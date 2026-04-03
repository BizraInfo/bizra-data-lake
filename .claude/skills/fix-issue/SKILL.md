---
name: fix-issue
description: Fix a GitHub issue with BIZRA compliance
disable-model-invocation: true
---

Analyze and fix the GitHub issue: $ARGUMENTS.

## Workflow

1. Use `gh issue view $ARGUMENTS` to get issue details
2. Understand the problem described in the issue
3. Search the codebase for relevant files
4. Check if issue relates to:
   - PAT enforcement (run `/pat` validation)
   - SAPE probes (run `/sape` validation)
   - Ihsan gate (run `/ihsan` validation)
   - Receipt schema (check src/receipts.rs and core/fate.py sync)
5. Implement necessary changes
6. Write and run tests to verify the fix
7. Ensure code passes linting and type checking
8. Generate evidence receipt for the fix
9. Create a descriptive commit message
10. Push and create a PR

## BIZRA Compliance

Before completing:
- [ ] All tests pass
- [ ] SAPE probes pass (9/9)
- [ ] Ihsan score >= 0.95
- [ ] Receipt emitted for changes
- [ ] No fail-closed violations

## Commit Format

```
type(scope): description

- What was fixed
- How it was fixed
- Evidence receipt ID

Fixes #$ARGUMENTS

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```
