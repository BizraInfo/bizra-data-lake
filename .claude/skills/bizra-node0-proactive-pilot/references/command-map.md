# Command Map

> Standing on Giants: Boyd (canonical command routing) · Lamport (deterministic state transitions)

Use this map to select the correct operational path for Node0 proactive pilot tasks.

## Canonical Command Matrix

| Command | Purpose | Mutating | Prerequisites | Source File | Expected Output / Logs |
| --- | --- | --- | --- | --- | --- |
| `python scripts/node0_activate.py status` | Inspect Node0 runtime and LM Studio status | No | Python available | `scripts/node0_activate.py` | Status banner, LM Studio connectivity, token state, mode |
| `python scripts/node0_activate.py start` | Start Node0 proactive kernel loop (foreground) | Yes | LM Studio reachable, `LM_API_TOKEN` set | `scripts/node0_activate.py` | Live loop logs in terminal — does NOT create PID file or log artifacts |
| `./scripts/start_proactive.sh --mode <mode> --config config/proactive_config.yaml` | Start proactive entity as background daemon | Yes | `.venv-linux` exists, config file exists, optional token | `scripts/start_proactive.sh` | `sovereign_state/proactive.pid`, `logs/proactive/sovereign.log`, `logs/proactive/startup.log` |
| `./scripts/stop_proactive.sh` | Gracefully stop running proactive entity (SIGTERM, 30s timeout) | Yes | PID file present or process running | `scripts/stop_proactive.sh` | Stop banner, PID cleanup, startup log stop entry |
| `./scripts/stop_proactive.sh --force` | Force stop after graceful timeout or hung process (SIGKILL) | Yes | Graceful stop already attempted, PID still alive | `scripts/stop_proactive.sh` | Force-kill confirmation, PID cleanup |
| `python scripts/node0_activate.py mission "<task>"` | Execute one mission through PAT agents (in-process, no daemon required) | Yes | LM Studio reachable, `LM_API_TOKEN` set, Python available | `scripts/node0_activate.py` | Mission report with assigned agents, total tokens, Ihsān score |
| `pytest tests/integration/test_autonomous_pilot.py -q` | Validate pilot readiness with 8 smoke pillars | No | Test deps available | `tests/integration/test_autonomous_pilot.py` | pytest pass/fail summary by class/pillar |
| `curl -sS -m 5 http://192.168.56.1:1234/v1/models` | Verify LM Studio API reachability | No | Network path to LM Studio host | — | JSON model list or connection error |
| `test -n "$LM_API_TOKEN"` | Verify token presence for `node0_activate.py` | No | Shell env loaded | — | Exit code 0 if set, non-zero if unset |
| `test -f sovereign_state/proactive.pid && cat sovereign_state/proactive.pid \|\| true` | Inspect proactive PID state | No | None | — | PID value when present or no output |
| `ls -la logs/proactive 2>/dev/null \|\| true` | Inspect proactive log artifacts | No | None | — | Log file listing or empty output |

## Start Path Distinction

| Path | Script | Execution Mode | Creates PID? | Creates Logs? | Use When |
| --- | --- | --- | --- | --- | --- |
| **A** (foreground) | `scripts/node0_activate.py start` | Foreground with signal handling | No | Stdout only | Direct interactive activation |
| **B** (daemon) | `scripts/start_proactive.sh` | Background daemon via `.venv-linux` | Yes (`sovereign_state/proactive.pid`) | Yes (`logs/proactive/`) | Production daemon with mode/config control |

## Mission Execution Note

`python scripts/node0_activate.py mission "<task>"` runs **in-process** — it does NOT require a background daemon to be running. It connects directly to LM Studio, dispatches PAT agents, and returns results to stdout. The background daemon (Path B) is a separate long-running proactive loop.

## Source File Mapping

| Source File | Provides |
| --- | --- |
| `scripts/node0_activate.py` | `start`, `status`, `mission` subcommands; PAT agent dispatch; LLM routing |
| `scripts/start_proactive.sh` | Daemon startup with `--mode`/`--config`; `.venv-linux` activation; PID/log creation |
| `scripts/stop_proactive.sh` | Graceful SIGTERM (30s timeout); force SIGKILL; PID cleanup |
| `scripts/set_lm_studio_key.sh` | Exports `LM_STUDIO_API_KEY` — use if `LM_API_TOKEN` is not set |
| `config/proactive_config.yaml` | Mode defaults, thresholds, autonomy levels, cycle interval (5s) |
| `tests/integration/test_autonomous_pilot.py` | 8 smoke pillars: RuntimeBoot, TokenSystem, EvidenceChain, SNR, SpearPoint, OpportunityPipeline, CLI, FullStack |
| `deploy/node0/health-check.py` | Deployment-level health check (separate from runtime status) |

## Mode Reference

Supported `--mode` values for `start_proactive.sh`:

| Mode | Behavior |
| --- | --- |
| `reactive` | Respond to explicit requests only |
| `proactive_suggest` | Suggest opportunities, await approval |
| `proactive_auto` | Execute low-risk opportunities automatically |
| `proactive_partner` | Full partnership mode (default) |
