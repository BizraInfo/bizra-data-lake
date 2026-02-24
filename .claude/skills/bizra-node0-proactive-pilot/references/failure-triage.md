# Failure Triage

> Standing on Giants: Lamport (failure modes in distributed systems) · Deming (root-cause before remediation)

Use this guide when proactive pilot operations fail or preflight checks block execution.

## Triage Flow

1. Reproduce symptom with one canonical command.
2. Capture first failure signal (exit code, stderr, or missing artifact).
3. Run only the first diagnostic command for that symptom.
4. Apply the minimal safe remediation.
5. Re-run preflight before retrying mutating operations.

## Symptom Matrix

| Symptom | Likely Cause | First Diagnostic Command | Safe Remediation |
| --- | --- | --- | --- |
| LM Studio unreachable | LM Studio not running, host/port mismatch, network route down | `curl -sS -m 5 http://192.168.56.1:1234/v1/models` | Start or restore LM Studio service, verify host/port config, then re-run status; keep diagnostics-only until reachable |
| `LM_API_TOKEN` missing | Token not exported in current shell/session | `test -n "$LM_API_TOKEN"` | Export token in current shell and re-run preflight; if intentionally unset, continue only with non-auth paths |
| Token name mismatch (`LM_API_TOKEN` vs `LM_STUDIO_API_KEY`) | Different scripts expect different env var names | `echo "LM_API_TOKEN=$LM_API_TOKEN"; echo "LM_STUDIO_API_KEY=$LM_STUDIO_API_KEY"` | `node0_activate.py` requires `LM_API_TOKEN`. If only `LM_STUDIO_API_KEY` is set, run `export LM_API_TOKEN="$LM_STUDIO_API_KEY"`. To set from scratch: `source scripts/set_lm_studio_key.sh` then export as `LM_API_TOKEN` |
| Stale proactive PID | Process died without cleanup; PID file left behind | `test -f sovereign_state/proactive.pid && kill -0 $(cat sovereign_state/proactive.pid) 2>/dev/null; echo $?` | If process not running (exit code non-zero), remove stale PID: `rm sovereign_state/proactive.pid`; re-run start preflight |
| Missing `.venv-linux` | Proactive shell start path prerequisites not provisioned | `test -d .venv-linux` | Create venv: `python3.11 -m venv .venv-linux && source .venv-linux/bin/activate && pip install -e .`; use direct Python path only if compatible |
| Start-path confusion (no PID after `node0_activate.py start`) | User expected daemon behavior from foreground command | `test -f sovereign_state/proactive.pid` | `node0_activate.py start` runs foreground — no PID file is created. For daemon mode with PID/log artifacts, use `./scripts/start_proactive.sh --mode <mode> --config config/proactive_config.yaml` instead |
| Stop command hangs | Process ignores SIGTERM or blocked shutdown | `./scripts/stop_proactive.sh` (observe 30s timeout) | Wait for full graceful timeout; only then use `./scripts/stop_proactive.sh --force`; report force-stop event and inspect `logs/proactive/startup.log` |
| Mission fails with connection/auth errors | LM Studio unreachable or token issues during mission call | `curl -sS -m 5 http://192.168.56.1:1234/v1/models` | Resolve reachability/token issues first; retry with explicit confirmation after preflight passes |
| Smoke suite fails | Runtime dependency/config regression in one or more pillars | `pytest tests/integration/test_autonomous_pilot.py -q` | Identify failing pillar/class, run targeted diagnostics below, pause mutating ops unless user explicitly requests continuation |

### Important: No Ollama Fallback for `node0_activate.py`

`scripts/node0_activate.py` hardcodes the LM Studio endpoint at `http://192.168.56.1:1234`. It does NOT fall back to Ollama (`localhost:11434`). If LM Studio is unreachable, all `start`, `status`, and `mission` subcommands will fail. The Ollama fallback exists only in other subsystems (e.g., `tools/engines/unified_model_router.py`).

## Smoke-Pillar Failure Guidance

Map failures from `tests/integration/test_autonomous_pilot.py` to first actions.

| Failing Pillar | Test Class | Signal | First Follow-up | Safe Next Step |
| --- | --- | --- | --- | --- |
| 1 | `TestRuntimeBoot` | Runtime init/status/context manager failures | `pytest tests/integration/test_autonomous_pilot.py -k RuntimeBoot -v` | Inspect `core/sovereign/runtime_core.py` imports and `RuntimeConfig` defaults before any start/mission operation |
| 2 | `TestTokenSystemSmoke` | Ledger/mint/chain integrity failures | `pytest tests/integration/test_autonomous_pilot.py -k TokenSystemSmoke -v` | Verify local ledger state and avoid autonomous mission execution until chain passes |
| 3 | `TestEvidenceChainSmoke` | Evidence append/verify failures | `pytest tests/integration/test_autonomous_pilot.py -k EvidenceChainSmoke -v` | Inspect ledger path permissions and chain consistency before continued ops |
| 4 | `TestSNRSmoke` | SNR facade scoring issues | `pytest tests/integration/test_autonomous_pilot.py -k SNRSmoke -v` | Treat outputs as degraded confidence; keep to diagnostics-only if score checks are unstable |
| 5 | `TestSpearPointSmoke` | Pipeline step failures | `pytest tests/integration/test_autonomous_pilot.py -k SpearPointSmoke -v` | Verify runtime integration dependencies before mission execution |
| 6 | `TestOpportunityPipelineSmoke` | AUTOLOW path fails | `pytest tests/integration/test_autonomous_pilot.py -k OpportunityPipelineSmoke -v` | Verify autonomy pipeline wiring and pause auto mode operations |
| 7 | `TestCLISmoke` | CLI import/version failures | `pytest tests/integration/test_autonomous_pilot.py -k CLISmoke -v` | Treat command surface as unstable; avoid start/mission mutating operations |
| 8 | `TestFullStackSmoke` | Full boot/health summary failures | `pytest tests/integration/test_autonomous_pilot.py -k FullStackSmoke -v` | Keep diagnostics-only until full stack health summary is serializable and stable |

## Escalation Rules

Escalate from remediation to containment when:

1. Same symptom repeats after one safe remediation attempt.
2. Multiple smoke pillars fail in one run.
3. Stop path requires repeated force-stop.

Containment actions:

1. Freeze mutating operations.
2. Collect exact failing command outputs and key logs from `logs/proactive/`.
3. Provide a minimal, reversible next action.
4. Direct user to `docs/PROACTIVE_SOVEREIGN_ENTITY.md` and `deploy/node0/README.md` for manual investigation.
