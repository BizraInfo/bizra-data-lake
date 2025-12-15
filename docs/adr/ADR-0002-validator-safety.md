# ADR-0002: Validator Safety Parameters (N=5, f=1)

**Status:** ACCEPTED  
**Date:** 2025-12-14  
**Scope:** Genesis Cluster

## Decision

We formally set the Genesis BFT parameters to:

* **N = 5** (Total Validators)
* **f = 1** (Maximum Faulty Nodes Tolerated)
* **Q = 3** (Quorum for Commit, calculated as $2f + 1$)

## Rationale

The standard safety condition for BFT consensus under partial synchrony is $N \ge 3f + 1$.
For $f=1$, this requires $N \ge 3(1) + 1 = 4$.
We choose N=5 to allow for 1 faulty node while maintaining a healthy margin above the minimum safety threshold.

## Consequences

* **Quorum Requirement:** Block commitment requires signatures from 3 unique validators.
* **Safety Mode Trigger:** Evidence of $f \ge 2$ (e.g., conflicting blocks signed by >1 validator) must trigger an immediate Safe Mode halt.
* **Scaling:** Expanding the validator set beyond 5 requires a new ADR to re-derive $f$ and $Q$.

## Evidence

* `src/sat.rs` (3/5 quorum rule enforced in code)
* `src/bridge.rs` (execution halts when consensus not reached)
