# BIZRA Node0 Entry Point Contract v0.1

**Date:** 2026-05-01 GST
**Status:** DECIDED -> thin wrapper pending
**Scope:** Operator entry model for Node0 / Mumu-DEMA.
**Truth label:** MEASURED for current command inventory and local readiness
checks; PLANNED for future `bizra node0` / `bizra dema` wrappers.

---

## §1 Decision

BIZRA Node0 starts with a CLI/TUI entry point.

The first operator doorway is not web-first, API-first, or hidden
daemon-first. The operator should open a terminal, run one local BIZRA
command, and immediately see DEMA / Node0 status, proof, memory, readiness,
and safe actions.

Canonical first command:

```bash
bizra node0
```

Accepted alias once the TUI exists:

```bash
bizra dema
```

Premium / advanced command-center alias:

```bash
bizra momo
```

The first user-visible face is Mumu-DEMA Relief Mode. Generic dashboards can
exist later, but they must not be the first operator experience.

---

## §2 Current Reality

Today the repo does not yet expose a polished `bizra node0` command. The
actual operational entry points are:

| Current surface | Purpose | Notes |
|---|---|---|
| `scripts/node0_activate.py status` | Node0 backend/model/PAT status | Verified with LM Studio on `http://127.0.0.1:1234`. |
| `scripts/dema/dema_service.py doctor` | DEMA local service health | Read-only; reports profile, lock, tick recency. |
| `scripts/dema/dema_service.py status` | DEMA service status JSON | Read-only; exposes daemon state and last tick. |
| `scripts/dema/dema_service.py start-once` | Single guarded DEMA wake tick | Writes one local tick receipt; does not start a loop. |
| `scripts/dema/dema_status.py --json` | Current / Ideal / Gap / Next | Read-only view over local DEMA state. |
| `core/sovereign/api.py` `/v1/node0/readiness` | API readiness contract | Requires the sovereign API server to be running on the configured port. |
| Node0 frontend terminal shell | Current web-terminal surface | Useful, but not the canonical first doorway. |

Runtime observation on 2026-05-01 GST:

- LM Studio desktop package `lm-studio 0.4.12+1` is installed.
- `/opt/LM-Studio/chrome-sandbox` is `root:root` with mode `4755`.
- LM Studio local API responds at `http://127.0.0.1:1234/v1/models`.
- `scripts/node0_activate.py status` reports LM Studio connected, one model
  loaded, token set, and hardware floor compliant.
- DEMA local profile exists for `Mumu` with `memory_consent=local`.
- DEMA service doctor is healthy after one `start-once` wake tick.

---

## §3 Canonical Future Commands

| Future command | Contract |
|---|---|
| `bizra node0` | Open the Mumu-DEMA Node0 TUI command center. |
| `bizra doctor` | Run guarded preflight/status without starting daemons. |
| `bizra dema status` | Show DEMA daemon, model, memory, readiness, and proof status. |
| `bizra dema start --mode relief` | Start Mumu-DEMA only after preflight and explicit confirmation. |
| `bizra dema stop` | Stop the daemon safely and show final receipt/lock state. |
| `bizra memory` | Show bounded memory import/status, with private/local/shareable boundaries. |
| `bizra proof` | Inspect receipts, proof surface, and readiness evidence. |
| `bizra momo` | Advanced TUI dashboard: live metrics, PAT state, command log, and proof panels. |

These commands are wrappers over existing scripts until the runtime is
consolidated. Current scripts must keep working.

---

## §4 First Screen Contract

`bizra node0` opens a TUI whose first frame answers:

| Panel | Required content |
|---|---|
| Mumu-DEMA status | Profile, Relief Mode state, Current / Ideal / Gap / Next. |
| LM Studio status | API URL, loaded model, model list count, token state. |
| Daemon status | Running/stopped, PID, lock path, last tick, safe start/stop actions. |
| Memory status | Local/private/shareable boundaries and latest import state. |
| Proof panel | Latest receipts, readiness truth label, open proof blockers. |
| Mission inbox | Pending operator intents and next admissible action. |
| Safe actions | Consent-gated local actions only; no hidden automation. |

The TUI must show truth labels and blockers plainly. It must not imply that a
daemon is running, a model is loaded, or a proof exists unless the checked
artifact says so.

---

## §5 Relief Mode Start Contract

`bizra dema start --mode relief` is allowed only after:

1. `bizra doctor` or equivalent preflight passes.
2. LM Studio API is reachable at the configured local URL.
3. At least one local model is available, and loaded-state is shown if known.
4. DEMA profile exists.
5. DEMA service doctor has no findings, or findings are shown before start.
6. Operator confirmation is explicit.
7. A receipt is written for the start action or single wake tick.

Current guarded equivalent:

```bash
.venv/bin/python scripts/dema/dema_service.py doctor --root sovereign_state/dema
.venv/bin/python scripts/dema/dema_service.py start-once --root sovereign_state/dema
.venv/bin/python scripts/dema/dema_service.py doctor --root sovereign_state/dema
```

`start-once` is a wake heartbeat, not a daemon loop. A future loop start must
surface PID, lock state, interval, stop command, and receipt path.

---

## §6 Standing On Giants

The external pattern to adopt is:

- Gateway/runtime first.
- TUI as the operator command surface.
- Sessions as explicit state.
- Slash commands as visible operator protocol.
- Model/session/status visible in the footer or status panel.
- Tool execution shown with consent and output visibility.

OpenClaw demonstrates gateway/local TUI modes, visible agent/session/model
state, slash commands, and local shell execution guarded by prompts.
Hermes demonstrates a modern TUI backed by the same runtime as the CLI, with
shared sessions, shared slash commands, model/session pickers, and richer
overlays.

BIZRA must adapt that pattern to its own law:

| Generic pattern | BIZRA form |
|---|---|
| Gateway/runtime | DEMA daemon / Node0 service |
| TUI | Mumu-DEMA command center |
| Agent/session | DEMA-0, memory, mission sessions |
| Slash commands | BIZRA command protocol |
| Tool execution | Consent-gated local action executor |
| Memory | Mumu private memory, BIZRA canon, URP shareable layer |
| Proof | Proof Surface and receipts |
| Safety | Ihsan, consent, no hidden action |

---

## §7 Non-Goals

This contract does not permit:

- Starting Node1.
- Publishing Third Fact.
- Adding broad automation.
- Hiding daemon/preflight status.
- Auto-starting the DEMA daemon from a status command.
- Making the browser UI the primary entry point.
- Writing to `MEMORY.md` from the DEMA service.
- Claiming readiness without a receipt or measured probe.

---

## §8 Thin Wrapper Plan

The first implementation slice is intentionally narrow and now includes the
read-only `bizra dema status` wrapper. Remaining start/stop/TUI work stays
planned until preflight, receipt signing, and operator confirmation gates are
complete.

1. Add `bizra node0` as a TUI/terminal command-center wrapper.
2. Add `bizra doctor` as a stable wrapper over existing preflight checks.
3. Add `bizra dema status` over `scripts/dema/dema_service.py status`.
4. Add `bizra dema start --mode relief` as an explicit, confirmed wrapper
   over preflight plus the bounded DEMA start path.
5. Add `bizra dema stop` only when there is an implemented loop supervisor to
   stop.

Do not move runtime ownership in this slice. Do not delete or rename current
scripts. Do not change daemon behavior just to make the wrapper pretty.

Truth status:

- `[ENFORCEMENT: WIRED]` `bizra doctor` propagates non-zero exit codes for
  activation blockers.
- `[ENFORCEMENT: WIRED]` `bizra dema status` is a non-starting status wrapper
  and must not create a fresh DEMA root during status reads.
- `[OPTIMIZATION: PLANNED]` `bizra node0`, `bizra dema start`, and
  `bizra dema stop` remain future slices.

---

## §9 Acceptance Tests

Minimum contract tests for the wrapper slice:

- `bizra doctor` exits non-zero when LM Studio is unreachable.
- `bizra doctor` exits non-zero when DEMA profile is missing.
- `bizra dema status` is read-only and does not write a tick.
- `bizra dema start --mode relief` refuses to run when preflight fails.
- `bizra dema start --mode relief` writes a local receipt when it succeeds.
- `bizra node0` renders status without starting a daemon.
- Existing commands continue to work.

---

## §10 Final Rule

Node0 begins at the terminal.

The first face is Mumu-DEMA. The first command is local. The first proof is
readiness. The first mission is relief for Mumu.

If any clause here conflicts with the BIZRA Topology Canon, Origin Manifest,
or DEMA Ambient Kernel / Service contracts, the stricter safety boundary wins
and this document must be amended.
