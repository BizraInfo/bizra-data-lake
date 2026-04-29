# Dema Canonical Semantics Layer (CSL) v0.1

**Date:** 2026-04-28 GST
**Status:** PLANNED → first implementation slice
**Scope:** Phase A0.7 of the Dema GTM Masterplan v0.1.
**Truth label:** MEASURED at the file level (CSL is just code + tests);
  DERIVED at the discipline level until consumer modules migrate to
  import from CSL in v0.2.

---

## §1 Purpose

Across A0 → A0.6, the same vocabulary appears in many places:

- `TruthLabel` — `DemaReceipt.truth_label`, `FourStateModel.truth_label`,
  `goal-truth-badge.tsx` UI badge.
- `RiskLevel` — `IntentPrediction.risk`, `interruption_policy` thresholds.
- `ApprovalStatus` — `DemaReceipt.approval_status`.
- `DecisionVerdict` — `interruption_policy.Decision.verdict`.
- Three envelope shapes — `DemaReceipt`, `FourStateModel`, `ProactiveProposal`.

If two consumers fall out of step, one of them silently lies.
A2-A5 (the Proof Surface) will read all of these and surface them to the
operator. Before that surface ships, the meanings need to be locked.

CSL fixes this by declaring each set ONCE in Python and emitting a
synchronised TypeScript mirror. Drift tests on both sides refuse to ship
when anything diverges.

---

## §2 What lands in this slice

### `core/dema/csl/`
- **`labels.py`** — six canonical tuples + matching `str`-Enum types:
  `RECEIPT_TRUTH_LABELS`, `DISPLAY_TRUTH_LABELS`, `MISSION_TRUTH_LABELS`,
  `RISK_LEVELS`, `APPROVAL_STATUSES`, `DECISION_VERDICTS`.
- **`schemas.py`** — three TypedDict envelopes:
  `CanonicalReceiptEnvelope`, `CanonicalFourStateModel`,
  `CanonicalProactiveProposal`.
- **`__init__.py`** — public re-exports + `SCHEMA_VERSION`.

### `scripts/dema/dema_csl.py`
- One CLI command: `emit-ts [--write] [--target <path>]`.
  Emits the canonical TypeScript mirror to stdout, or writes it to
  `frontend/src/lib/dema-csl.ts` (the default target) under `--write`.

### `frontend/src/lib/dema-csl.ts`
- Auto-generated. Holds the TS-side const arrays + literal union types.
- A header comment forbids hand-editing and points to the regenerator.

### Drift tests
- **`tests/scripts/test_dema_csl.py`** — 13 tests covering: enum/tuple
  parity, tier relationships (RECEIPT ⊆ DISPLAY; DISPLAY adds exactly
  UNKNOWN), consumer-module alignment (`DemaReceipt`, `FourStateModel`,
  `intent_model._RULES`, `interruption_policy.DecisionVerdict`), TS
  mirror equality, and TypedDict key coverage.
- **`frontend/tests/dema-csl.test.ts`** — 8 tests independently locking
  the expected string values on the TS side, plus a goal-card
  alignment check.

---

## §3 Tier model — two truth-label sets

```
RECEIPT_TRUTH_LABELS = (MEASURED, DERIVED, PLANNED, SANDBOX)
DISPLAY_TRUTH_LABELS = (MEASURED, DERIVED, PLANNED, SANDBOX, UNKNOWN)
MISSION_TRUTH_LABELS = DISPLAY_TRUTH_LABELS
```

Receipts always describe something real, so `UNKNOWN` is not allowed
inside a receipt. UI surfaces and the mission state machine MAY render
`UNKNOWN` to honestly say "no data yet". The contract is:

> `RECEIPT_TRUTH_LABELS ⊂ DISPLAY_TRUTH_LABELS`, with the difference
> being exactly `{UNKNOWN}`.

A drift test enforces this exact relationship.

---

## §4 Migration strategy (v0.1 → v0.2)

v0.1 **does not** migrate consumer modules to import from CSL. It
declares CSL and adds drift tests against today's local constants. This
keeps the PR scoped and avoids a cascade of cross-module changes during
merge windows.

v0.2 will:

1. Replace `core/dema/receipts.py:VALID_TRUTH_LABELS` with an import
   from CSL.
2. Replace `core/dema/mission_state.py:VALID_TRUTH_LABELS` with CSL.
3. Replace the literal union in
   `frontend/src/components/terminal/goal-truth-badge.tsx` with
   `DisplayTruthLabel` from CSL.
4. Replace `core/dema/proactive/interruption_policy.py:DecisionVerdict`
   Literal with `DecisionVerdict` from CSL.

If any of those replacements drift before v0.2 ships, the drift tests in
v0.1 will catch it.

---

## §5 Storage / surface

CSL writes nothing at runtime. It is a code-only layer. No
`sovereign_state/` paths, no receipts, no network. The drift tests run
in CI under the existing pytest + vitest gates.

---

## §6 Bounds

This layer carries **no AGI guarantee**, **no token-value claim**, and
**no public claim** of any kind. CSL is purely a discipline mechanism
for a vocabulary that already exists. If any clause here conflicts with
the BIZRA Topology Canon (2026-03-25), the Origin Manifest, or the
Brand Canon v0.2, those canonical sources win and this doc must be
amended.

---

## §7 What's NOT shipped in v0.1

- Migration of consumer modules to import from CSL (deliberate; v0.2).
- Pydantic-style runtime validation of envelopes (TypedDicts only).
- A code-generation step in the build (the TS mirror is regenerated
  manually via the CLI; the drift test catches stale files in CI).
- Cross-language schema validation at API boundaries (handled at the
  receipt-write layer today; A2-A5 may add a stricter validator).
- Versioned schema migration tooling for past receipts (the existing
  `schema_version` field is sufficient for v0.1).
