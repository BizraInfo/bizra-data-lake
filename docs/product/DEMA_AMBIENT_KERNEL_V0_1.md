# Dema Ambient Kernel v0.1

**Date:** 2026-04-27 GST
**Status:** PLANNED → first implementation slice
**Scope:** Phase A0 of the Dema GTM Masterplan v0.1.
**Truth label:** MEASURED at the file level (every artifact emitted is a real
  receipt over real local state); DERIVED at the product-promise level until
  Phase A1+ surfaces it in the Dema UI.

---

## §1 What this is

The Ambient Kernel is the always-on local presence of Dema on Node0. It does
not call out to the network. It does not control the desktop. It does not
post to social. It is the part of Dema that:

1. **Knows who you are** — preferred name, mother / work language, persona
   tone, memory-consent setting.
2. **Holds today's log** — append-only, summary-only, receipt-linked.
3. **Surfaces Current / Ideal / Gap / Next** — the §9 four-state model from
   the BIZRA Origin Manifest.
4. **Dreams read-only** — five-phase consolidation pass (Orient → Gather →
   Consolidate → Prune → Prepare) with a hard time budget. No automatic
   promotion to long-term memory.
5. **Receipts every meaningful action** — local hash-chained envelopes with
   `truth_label`, `touched_paths`, `not_touched_paths`, and approval status.

It is the foundation Phase A1 (Goal Surface) builds on.

Identity boundary: this kernel currently runs as **DEMA-0 / Node0-DEMA** on
Node0. `DEMA_IDENTITY_MODEL_V0_1.md` is the prerequisite doctrine for keeping
Node0-private memory separate from DEMA Core and URP-shareable knowledge.

---

## §2 Storage layout (all gitignored)

```
sovereign_state/dema/
  ├── profile.json                        DemaProfile
  ├── mission_state.json                  FourStateModel
  ├── logs/<YYYY-MM-DD>.jsonl             append-only DailyLog
  ├── receipts/<YYYY-MM-DD>/<rid16>.json  DemaReceipt store
  └── dreams/<run_id>/
       ├── candidate_notes.jsonl
       └── summary.md
```

`sovereign_state/` is already gitignored. Nothing the Ambient Kernel writes
ends up in committed artifacts.

---

## §3 Components

### `core/dema/profile.py` — DemaProfile + ProfileStore

Captures: `preferred_name`, `mother_language`, `work_language`,
`persona_tone` (default `pragmatic-mystic`), `memory_consent` (one of
`off | local | private | shared_candidates`).

`ProfileStore.init_from_env_or_defaults()` reads `DEMA_PREFERRED_NAME`,
`DEMA_MOTHER_LANGUAGE`, `DEMA_WORK_LANGUAGE`, `DEMA_PERSONA_TONE`,
`DEMA_MEMORY_CONSENT` env vars (all optional) and falls back to safe defaults.

### `core/dema/daily_log.py` — DailyLogEntry + DailyLog

Append-only JSONL files partitioned by UTC date. Each entry carries
`timestamp`, `kind` (`tick | mission | dream | onboarding | import |
action`), `summary`, optional `receipt_id`, and free-form `metadata`.

### `core/dema/mission_state.py` — FourStateModel + MissionStateMachine

The §9 model: `current`, `ideal`, `gap`, `next_admissible_action`,
`truth_label`. Default state is `UNKNOWN` / not actionable. `update()` is
field-wise additive — pass only what you want to change.

### `core/dema/receipts.py` — DemaReceipt + ReceiptWriter

Local audit envelope with `action`, `truth_label`,
`touched_paths`, `not_touched_paths`, `approval_required`,
`approval_status`, `payload`. `receipt_id = BLAKE3(canonical-json)`. Distinct
from the canonical Ed25519 / BLAKE3-chained `CanonicalReceipt` — this is the
lightweight local trail for Ambient Kernel actions.

---

## §4 Commands

```bash
# 1. Onboarding — write profile + receipt + log entry
python scripts/dema/dema_onboarding.py --init
# (Optional: prefill via env)
DEMA_PREFERRED_NAME="Mumu" DEMA_MOTHER_LANGUAGE="ar" \
  python scripts/dema/dema_onboarding.py --init

# 2. Status — Current/Ideal/Gap/Next as JSON
python scripts/dema/dema_status.py --json

# 3. Daemon tick — single heartbeat (no continuous loop in v0.1)
python scripts/dema/dema_daemon.py --once

# 4. Dream — read-only, time-budgeted consolidation
python scripts/dema/dema_dream.py --read-only --max-seconds 15
```

All scripts accept `--root <path>` to override the local state root (used
by the test suite to sandbox each test).

---

## §5 Safety contract (non-claims, enforced by tests)

Every Ambient Kernel receipt explicitly lists the surfaces it does **not**
touch:

- `network` — no listener, no outbound calls
- `desktop` — no key/mouse simulation, no app launching
- `MEMORY.md` — never edited by the kernel
- `docs/canon/` — no canon mutation
- `social` (in daemon receipts) — no autonomous posting
- `long_term_memory_promotion` (in dream receipts) — promotion is
  approval-gated, not automatic

The contract test
`test_dema_writes_only_under_sovereign_state` asserts every path the kernel
returns lives under the sandbox root. The test
`test_no_canon_or_memory_md_is_listed_in_touched_paths` asserts the
non-claims hold.

---

## §6 Truth labels in the kernel

| Surface | Truth label | Why |
|---|---|---|
| Profile | `MEASURED` | The profile fields were really written |
| Daily log entry | `MEASURED` | The entry really lives in the JSONL |
| Mission state | starts `UNKNOWN`, becomes `PLANNED` / `MEASURED` as data lands | Honest about cold-start |
| Dream candidate notes | `MEASURED` (the notes exist) but explicit `promoted_to_long_term: false` | Promotion is approval-gated |
| Tick receipt | `MEASURED` | A real local heartbeat happened |

---

## §7 Out of scope for v0.1

- Continuous daemon loop (only `--once` is supported)
- Network listener of any kind
- Desktop control / actuator (Phase A7)
- Browser automation
- Autonomous social posting
- Long-term memory promotion (Phase A6 / governance)
- Mother-language NLG (only profile capture; no language switching of UI)
- Frontend Goal-tab (Phase A1)

---

## §8 What the operator sees today

After running the four commands above, the local state directory contains:

```
sovereign_state/dema/
├── profile.json                   ← who you are
├── mission_state.json             ← Current/Ideal/Gap/Next
├── logs/2026-04-27.jsonl          ← onboarding + tick + dream entries
├── receipts/2026-04-27/
│   ├── <rid>.json                 ← onboarding receipt
│   ├── <rid>.json                 ← tick receipt
│   └── <rid>.json                 ← dream receipt
└── dreams/dream_<ts>/
    ├── candidate_notes.jsonl
    └── summary.md
```

That's the floor: real artifacts, real receipts, no overclaim. Phase A1
turns this into a UI surface.

---

## §9 Bounds

This kernel carries **no AGI guarantee**, **no token-value claim**, **no
public claim** of any kind. Every action it takes is local-only,
truth-labeled, receipt-linked, and bound by the non-claims listed in §5.

If any clause here conflicts with the BIZRA Topology Canon (2026-03-25),
the Origin Manifest, or the Brand Canon v0.2, those canonical sources win
and this doc must be amended.
