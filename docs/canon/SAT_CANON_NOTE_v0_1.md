# SAT Canon Restatement v0.1

**Date:** 2026-04-27
**Scope:** SAT placement only. No other canon surface modified by this note.
**Truth boundary:** This note **restates** the 2026-03-25 frozen canon for the
SAT placement and N-scaling rule. It does **not** introduce new canon, and it
is **not** yet a public production proof of multi-node SAT behavior.

---

## Provenance — why this note exists

During PR #59 review, a reading-drift was caught: an earlier framing in this
session implied the 2026-03-25 frozen canon held SAT-5 as a single shared set
of 5 agents network-wide, and that PR #59 introduced the per-node-contribution
model as a canon evolution.

**That framing was inaccurate.** The 2026-03-25 frozen canon already specified
per-node SAT-5 contribution into the shared URP and 5N scaling. The substance
of this note (numbers, scaling, per-node materialization) was **already canon**.

This note exists to (1) confirm the canon at the doc layer for future sessions,
(2) record the reading-drift correction so the same misread isn't repeated, and
(3) link Node0-Local URP Proof v0.1 (`URP_LOCAL_ACTIVE`) back to the canonical
scaling rule it instantiates.

## Reaffirmed canon — verbatim from the 2026-03-25 anchor

The frozen canon (Memory anchor `reference_bizra_topology_canon_frozen_2026_03_25.md`)
states explicitly:

> *"When the first human (Node0) activates:*
> *- System mints PAT-7 on their device (local)*
> *- System mints SAT-5 into the URP (shared)*
> *...*
> *Each subsequent node adds 5 more SAT agents to the shared URP, plus contributed resources."*

And the canonical scaling table:

| Nodes | Local PAT (total) | SAT in shared URP (total) |
|---|---|---|
| 1 | 7 | 5 |
| 1,000 | 7,000 | 5,000 |
| 1,000,000 | 7M | 5M |
| 8,000,000,000 | 56B | 40B |

## Restated rule (this note's only purpose)

```
Each sovereign node materializes 5 SAT agents into the shared URP.
The URP itself is one shared organism (NOT per-node).
URP SAT capacity grows monotonically as 5N for N sovereign nodes.
```

The five SAT agents are canon-named: S1 Validator · S2 Oracle (frozen, immutable
truth axioms) · S3 Mediator · S4 Archivist · S5 Sentinel. PAT-7 placement is
**unchanged**: PAT remains per-node and user-private.

## Truth-label linkage

The first observable proof of this scaling rule at N=1 is the Node0-Local URP
Proof v0.1 with truth label `URP_LOCAL_ACTIVE` (5 SAT registered into the
shared URP). Subsequent stages follow the locked URP truth ladder:

```
URP_LOCAL_ACTIVE → PRIVATE_PILOT_URP → PILOT_SHARED_URP → UNIVERSAL_NETWORK_URP
```

See `artifacts/proofs/node0-local-urp/` for the chain-anchored receipts that
bind this restatement to verifiable Node0 state.

## Bounds

- This note carries **no AGI guarantee**.
- This note carries **no token-value claim**.
- PoI ledger remains `POI_SANDBOX` until measured multi-node evidence exists.
- The 5N scaling rule describes the canon model. Real-world capacity at any
  N requires its own pilot evidence pack before public claim.

## Relationship to the frozen canon

This note **does not supersede** any clause of the 2026-03-25 frozen canon. It
reaffirms the SAT placement and 5N scaling rule that were already canonical.
All other 2026-03-25 canon clauses (PAT-7 names, SAT-5 names, membrane
fail-closed properties, no peer-to-peer-without-membrane, DEMA = P7, ONE URP
organism) remain in force unchanged.

If a conflict ever arises between this note and the frozen canon, **the frozen
canon wins.**
