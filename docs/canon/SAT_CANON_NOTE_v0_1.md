# SAT Canon Evolution Note v0.1

**Date:** 2026-04-27
**Scope:** SAT placement only. No other canon surface modified by this note.
**Truth boundary:** This note records a founder-stated canon evolution. It is
**not** yet a public production proof of multi-node SAT behavior.

---

## Old reading (frozen 2026-03-25)

```
ONE URP shared, NOT per-node.
SAT-5 lives IN URP, NOT between nodes.
```

Interpreted to mean a single set of 5 SAT agents lived once inside a
single shared URP and was consumed by every node from the same fixed
pool.

## Updated founder canon (locked 2026-04-27)

```
Each sovereign node mints/contributes SAT-5 into URP.
URP SAT capacity grows as 5N for N sovereign nodes.
```

Each human-identity node materializes its own SAT-5 instance — Validator,
Oracle (frozen), Mediator, Archivist, Sentinel — and registers them into
the shared URP substrate. URP remains one shared substrate; SAT capacity
inside URP grows monotonically as new sovereign nodes join.

## Scaling rule

| Network state | URP SAT count |
|---|---|
| Node0 alone | 5 |
| Node0 + Node1 | 10 |
| N nodes | 5N |

PAT-7 placement is unchanged: PAT remains per-node and user-private.
Only SAT placement is clarified by this note.

## Truth-label linkage

The first observable proof of this model is the Node0-Local URP Proof
v0.1 with truth label `URP_LOCAL_ACTIVE`. Subsequent stages follow the
locked URP truth ladder:

```
URP_LOCAL_ACTIVE → PRIVATE_PILOT_URP → PILOT_SHARED_URP → UNIVERSAL_NETWORK_URP
```

See `artifacts/proofs/node0-local-urp/` for the chain-anchored receipts
that bind this canon evolution to verifiable Node0 state.

## Bounds

- This note carries **no AGI guarantee**.
- This note carries **no token-value claim**.
- PoI ledger remains `POI_SANDBOX` until measured multi-node evidence
  exists.
- The 5N scaling rule describes the canon model. Real-world capacity at
  any N requires its own pilot evidence pack before public claim.

## Supersedes

This note supersedes the placement clause of the 2026-03-25 frozen
canon ("ONE URP shared, SAT-5 lives IN URP, NOT per-node") **only with
respect to SAT origin and per-node materialization**. All other 2026-03-25
canon clauses (PAT-7 names, SAT-5 names, membrane fail-closed properties,
no peer-to-peer-without-membrane) remain in force unchanged.
