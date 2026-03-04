# Node0 Standalone Readiness

This document is the operating path to activate **one fully usable Node0** before Alpha-100.

## Deliverables

- Unified lifecycle command: `scripts/node0_standalone.py`
- Updated activation portal: `docs/node0_standalone_portal.html`
- Existing runtime entrypoint remains: `scripts/node0_activate.py`

## Unified Installer + Lifecycle

```bash
python scripts/node0_standalone.py activate --architect "MoMo"
python scripts/node0_standalone.py health
```

Activation performs:

1. Identity activation (mint/load) and PAT/SAT roster validation
2. Hardware scan and asset registry publication (`sovereign_state/node0_assets.json`)
3. URP signed pledge + verification (`sovereign_state/urp_pledge.json`)
4. PAT awareness publication (`sovereign_state/pat_awareness.json`)
5. Lifecycle gate update (`sovereign_state/node0_lifecycle.json`)

## Local API for Website/UI

```bash
python scripts/node0_standalone.py serve --host 127.0.0.1 --port 8091
```

Security note:

- Loopback (`127.0.0.1`) can run without an API key for local-only usage.
- Non-loopback hosts require `BIZRA_NODE0_API_KEY` (or `BIZRA_API_KEY`) before startup.
- When a key is set, send it in `X-API-Key` for `POST /activate`, `POST /task`, `GET /assets`, and `GET /lifecycle`.

Endpoints:

- `GET /health`
- `GET /assets`
- `GET /lifecycle`
- `POST /activate`
- `POST /task`

## Runtime Start

```bash
python scripts/node0_activate.py start
python scripts/node0_activate.py status
```

Compatibility flags are also available:

- `python scripts/node0_activate.py --status`
- `python scripts/node0_activate.py --mission "..."`
- `python scripts/node0_activate.py --verify`

## Autonomous Tasks

Browser + desktop mission path:

```bash
python scripts/node0_standalone.py task "research local-first agent runtime patterns" --browser-mode direct
```

Filesystem action path (when HDA is unavailable):

```bash
python scripts/node0_standalone.py task "write file missions/node0_note.md :: PAT wrote this through standalone mission flow."
python scripts/node0_standalone.py task "read file missions/node0_note.md"
python scripts/node0_standalone.py task "list dir missions"
```

Default browser mode is `mock` for deterministic local/offline execution. Use `--browser-mode direct` for live web research.

## Website

Open:

- `docs/node0_standalone_portal.html`

The portal is wired to `http://127.0.0.1:8091` for live health, activation, and task commands.
