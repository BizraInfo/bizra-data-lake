# BIZRA Node0 CLI/TUI Source Audit v0.1

**Date:** 2026-05-01 GST
**Status:** DECIDED -> BIZRA-native thin wrapper first
**Scope:** Local CLI/TUI source archives and existing BIZRA command surfaces.
**Truth label:** MEASURED for local archive/license inspection and existing repo
surfaces; PLANNED for future full TUI work.

---

## Decision

Use **Option C** first: build a BIZRA-native thin wrapper inspired by the
downloaded projects, without copying their source code into BIZRA.

The first implemented surface is deliberately small:

- `bizra doctor`
- `bizra dema status`

No full TUI, Node1 work, Third Fact publication, memory bulk import, or
continuous DEMA daemon loop is authorized by this slice.

---

## Existing BIZRA Surfaces Checked First

Before adding code, the repo already had these relevant surfaces:

| Existing surface | Role | Reuse decision |
|---|---|---|
| `scripts/dema/dema_service.py status` | JSON daemon/profile/lock/tick status | Reuse directly |
| `scripts/dema/dema_service.py doctor` | JSON DEMA service health | Reuse directly |
| `scripts/dema/dema_status.py --json` | Current/Ideal/Gap/Next state | Reuse directly |
| `scripts/node0_activate.py status` | Node0 LM Studio/token/PAT status | Reuse as behavior reference |
| `core/sovereign/doctor.py` | Packaged `bizra doctor` implementation | Extend in place |
| `core/sovereign/__main__.py` | Actual `bizra` console entrypoint | Wire `dema status` here |
| `core/cli/commands/*` | Modular command registry | Add matching read-only command |

The wrapper must not replace or rename current scripts. It should call the
existing status/doctor logic wherever possible.

---

## Downloaded Candidate Archives

| Project | Local source | License | Stack | Best reuse |
|---|---|---|---|---|
| OpenClaw | `/home/bizra-operating-system/Downloads/openclaw-2026.4.25.zip` | MIT | Node.js 22+, TypeScript, Swift/Kotlin apps, gateway/plugin SDK | Gateway model, local/remote pairing, approval queues, sandbox/tool-policy separation |
| Hermes Agent | `/home/bizra-operating-system/Downloads/hermes-agent-2026.4.23.zip` | MIT | Python 3.11+, Rich, prompt_toolkit, argparse-style CLI, Ink/React TUI | Python command registry, slash commands, session/status/doctor model, future TUI structure |

Both archives include root MIT licenses. Direct code import remains out of
scope because both projects bring broad architecture and dependency surfaces.

---

## Architecture Findings

### OpenClaw

OpenClaw is strongest as a gateway/process-boundary reference:

- gateway startup and health
- local/remote pairing
- plugin SDK exports
- model/provider setup
- approval handling
- sandbox/tool-policy separation
- native app and messaging routes

OpenClaw should inform later gateway and consent-gated local-action design, not
the first Python CLI wrapper.

### Hermes Agent

Hermes is the closer CLI/TUI fit:

- Python CLI command surface
- central command/slash-command registry
- session lifecycle and resume semantics
- file and terminal safety modules
- richer Ink/React TUI package

Hermes should inform BIZRA command ergonomics and future TUI structure, not be
merged directly.

---

## Recommendation

Implement a native BIZRA wrapper layer:

1. Reuse current DEMA service/status scripts.
2. Keep `bizra doctor` as the packaged health check and extend it read-only.
3. Add `bizra dema status` as a read-only operator doorway.
4. Keep future TUI work separate until the thin wrapper is proven.

Do not fork OpenClaw or Hermes in this slice.

---

## Non-Goals

This audit does not permit:

- starting Node1
- publishing Third Fact
- bulk-ingesting memory
- starting a continuous DEMA daemon loop
- copying OpenClaw or Hermes source into BIZRA
- hiding daemon/preflight state
- claiming readiness without measured probes

