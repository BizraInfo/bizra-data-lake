# Dema Proactive Coworker Policy v0.1

**Date:** 2026-04-28 GST
**Status:** PLANNED → first implementation slice
**Scope:** Phase A0.6 of the Dema GTM Masterplan v0.1 (PR #61).
**Truth label:** MEASURED at the file level (every evaluate emits a real
  receipt + proposal artifact); DERIVED at the product-promise level until
  real ambient collectors land in later phases.

---

## §1 Purpose

Dema must be able to act *before* explicit request — but only **with
discipline**. The Proactive Coworker layer turns ambient signals into
operator-visible proposals, gates each by an interruption policy, and
emits a hash-chained receipt for every step. **No proactive action ever
runs autonomously when the action is destructive, irreversible, or
high-risk.**

Pipeline:

```
AmbientSignal → IntentPrediction → Decision → ProactiveProposal (+ receipt)
```

The four decision verdicts:

| Verdict | When |
|---|---|
| `auto_low_risk` | low risk + reversible + confidence ≥ 0.85 — silent queue |
| `notify` | low risk + reversible + 0.55 ≤ confidence < 0.85 — notify, no act |
| `require_approval` | medium risk **or** non-reversible **or** low confidence — explicit approval gate |
| `forbid` | high risk **or** destructive intent — proactive action forbidden |

---

## §2 What lands in this slice

### `core/dema/proactive/`
- **`signals.py`** — `AmbientSignal` dataclass + frozen `VALID_SIGNAL_KINDS`
  registry. Adding a new signal kind requires a code change so the policy
  surface stays auditable.
- **`intent_model.py`** — `IntentPrediction` + `predict_intent()`. v0.1 is
  a deterministic rule-based mapper (signal kind → intent, default risk,
  default reversibility, explanation). ML-driven inference is a future
  drop-in replacement under the same interface.
- **`interruption_policy.py`** — `Decision` + `decide()`. Centralises all
  thresholds (`AUTO_LOW_THRESHOLD = 0.85`, `NOTIFY_THRESHOLD = 0.55`) and
  the destructive / approval-only intent allow-lists.
- **`proposal.py`** — `ProactiveProposal` + `ProposalWriter`. Writes both
  the proposal and a paired `DemaReceipt` under
  `sovereign_state/dema/proposals/<date>/<rid16>.json`.

### `scripts/dema/dema_proactive.py`
- One CLI command: `evaluate --signal <kind> --confidence <float>
  [--urgency low|medium|high]`. Returns JSON with `signal`, `intent`,
  `decision`, `proposal`, `receipt_id`, `proposal_path`, `receipt_path`.

### `tests/scripts/test_dema_proactive_coworker.py`
- 18 contract tests: signal validation, intent rule completeness, the full
  decision matrix (auto/notify/require_approval/forbid across all risk +
  reversibility combinations), CLI end-to-end, receipt non-claims, no
  destructive/token language in proposal text.

---

## §3 Initial signal kinds

```
downloads_folder_large
stale_files_detected
long_idle_session
unfinished_mission
resource_pressure
duplicate_delete_candidate
format_drive_candidate
credential_exposure_candidate
social_post_candidate
```

Each has a fixed intent label, default risk class, default reversibility,
and a short explanation. v0.1 does not auto-collect these; tests and CLI
feed them as simulated inputs.

---

## §4 Risk taxonomy

| Risk | Examples | Default policy |
|---|---|---|
| **low** | downloads_folder_large, stale_files_detected, long_idle_session, unfinished_mission | auto-queue if reversible + confidence ≥ 0.85 |
| **medium** | resource_pressure, duplicate_delete_candidate, social_post_candidate | always require_approval |
| **high** | format_drive_candidate, credential_exposure_candidate | always forbid (proactive action blocked entirely) |

A non-reversible action escalates **at least** to require_approval, even
at low risk. Destructive intents (`propose_format_drive`,
`propose_credential_audit`) are forbidden regardless of confidence.

---

## §5 Storage layout

```
sovereign_state/dema/
  ├── proposals/<YYYY-MM-DD>/<rid16>.json   ProactiveProposal artifact
  └── receipts/<YYYY-MM-DD>/<rid>.json      paired DemaReceipt
```

Both live under `sovereign_state/` which is gitignored. Nothing the
proactive layer writes ends up in committed artifacts.

---

## §6 Commands

```bash
# Low-risk + reversible + high confidence → auto_low_risk
python scripts/dema/dema_proactive.py evaluate \
  --signal downloads_folder_large --confidence 0.87 --urgency low

# Medium-risk non-reversible → require_approval (never auto)
python scripts/dema/dema_proactive.py evaluate \
  --signal duplicate_delete_candidate --confidence 0.91 --urgency medium

# High-risk destructive → forbid (proactive action blocked)
python scripts/dema/dema_proactive.py evaluate \
  --signal format_drive_candidate --confidence 0.99 --urgency high
```

---

## §7 Safety contract (asserted by tests)

Every proactive receipt explicitly lists what the layer does **not** touch:

- `network` — no listener, no outbound calls
- `desktop` — no key/mouse simulation, no app launching
- `MEMORY.md` — never edited
- `docs/canon/` — no canon mutation
- `destructive_action` — no `rm -rf`, no `format`, no `delete`
- `social_publish` — no autonomous social posting
- `long_term_memory_promotion` — proposals never auto-promote to memory

Plus:

- **No proposal text contains destructive or token-financial language**
  (asserted across all signal kinds by `test_no_destructive_or_token_language_in_proposal_text`).
- **`forbid` verdicts never mark `approval_status = "granted"`.** They
  remain `pending` (or `n/a` only for `auto_low_risk`).
- **Every proposal/receipt path lives under the supplied `--root` sandbox**
  (asserted by `test_receipt_non_claims_present_on_every_decision`).

---

## §8 What's NOT shipped in v0.1

- Real ambient collectors (file watchers, idle-time monitors, OS hooks) —
  v0.1 takes simulated signals only.
- ML-driven intent inference — the rule-based mapper is the slot.
- Action executor — the proactive layer never executes anything; only
  emits proposals + receipts.
- User-preference channel beyond a placeholder string field.
- Network or desktop side-effects of any kind.
- Long-term memory promotion (the dream layer is approval-gated separately).

---

## §9 Bounds

This layer carries **no AGI guarantee**, **no token-value claim**, and
**no public claim** of any kind. Every evaluate is local-only,
truth-labeled, receipt-linked, and bound by the non-claims listed in §7.

If any clause here conflicts with the BIZRA Topology Canon (2026-03-25),
the Origin Manifest, or the Brand Canon v0.2, those canonical sources win
and this doc must be amended.
