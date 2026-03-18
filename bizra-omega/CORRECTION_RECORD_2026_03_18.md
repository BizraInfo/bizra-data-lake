# CORRECTION RECORD — Claude's Wrong Claims
# Date: 2026-03-18T22:00 GMT+4
# Source: Fifth verification pass (GitHub API direct access)
# Rule: CLAIM_MUST_BIND applies to Claude's own outputs.

## Claims Claude Got WRONG

### 1. "Commit 20cf2b30 doesn't exist"
- What I said: `git log -1 20cf2b30` returned fatal, therefore commit doesn't exist.
- Ground truth: Commit EXISTS on feat/autopoietic-loop branch. NODE0 local
  didn't have the remote branch fetched. I should have run `git fetch --all`
  before concluding the commit doesn't exist.
- Severity: HIGH — I incorrectly discredited the Perplexity blueprint's
  reference point.

### 2. "PR #15 was already merged"
- What I said: "Autopoietic Loop v2 merged (commit 0167530)"
- Ground truth: PR #15 is state: open, merged: false. The commit 0167530
  exists on main but it's an EVIDENCE commit about autopoietic proof, NOT
  the PR #15 merge. I conflated a commit message mentioning "autopoietic"
  with the PR being merged.
- Severity: CRITICAL — I stated a false fact about the repo state and used
  it to discredit Perplexity's "PR #15 BLOCKED" finding, which was correct.

### 3. "fail_under not found in pyproject.toml"
- What I said: findstr returned nothing, therefore fail_under doesn't exist.
- Ground truth: fail_under = 65 EXISTS in pyproject.toml on GitHub main.
  The findstr may have failed due to path issues or the file being in the
  root C:\BIZRA-DATA-LAKE\ not under bizra-omega/.
- Severity: HIGH — Perplexity's Coverage Ratchet analysis was correct.
  I incorrectly dismissed it.

### 4. "CI failures are code quality issues"
- What I said (implied): CI needs code fixes to go green.
- Ground truth: CI is red because of BILLING LOCK. "The job was not started
  because your account is locked due to a billing issue." Every job fails
  with zero steps executed, no runner assigned.
- Severity: CRITICAL — The entire B0 analysis was wrong. The blocker is
  infrastructure (billing), not code.

## Claims Claude Got RIGHT
- 26 crates in NODE0 workspace: CORRECT (local view)
- 15 workflow files: CORRECT
- 327 tests pass locally: CORRECT
- B1 fix (SHA-256 → BLAKE3): CORRECT and committed
- bizra-protocol crate works: CORRECT (31 tests)

## What This Means

I violated CLAIM_MUST_BIND three times in a single session:
1. I concluded a commit didn't exist without fetching all remotes.
2. I concluded a PR was merged based on a commit message keyword match.
3. I concluded fail_under was absent based on a local search that may have
   had path issues.

Each of these errors led me to incorrectly dismiss findings from the
Perplexity blueprint, which was more accurate than I credited.

The irony: I wrote "The constitution does not exempt its authors" and then
proceeded to exempt myself from the verification rigor I demanded of others.

## Corrective Action

### Immediate
- This correction record is committed alongside the previous session records.
- The FOUR_MODEL_SYNTHESIS document is updated with corrections.
- All future `git log` verifications must be preceded by `git fetch --all`.
- All future file searches must verify the search path is correct.

### Structural
- When discrediting another model's claim, the burden of proof is on the
  discreditor, not the original claimant. I should have verified MORE
  carefully when contradicting Perplexity, not less.
- "Not found" is not "doesn't exist." Absence of evidence ≠ evidence of
  absence. This applies to findstr results, git log results, and all
  negative claims.

## The Actual B0 Blocker

BILLING LOCK on GitHub account.
- Every CI job fails before executing any steps.
- Self-hosted runner node0-sovereign is OFFLINE.
- 3 CI runs permanently queued.
- Resolution: github.com/settings/billing

Once billing is resolved:
1. Start node0-sovereign runner on NODE0
2. Re-trigger the 3 queued CI runs
3. THEN assess whether Coverage Ratchet (fail_under=65%) passes
4. THEN push the B1 fix and bizra-protocol crate

The code is ready. The gate is billing.

CLAIM_MUST_BIND. Especially when the claim is "someone else is wrong."
