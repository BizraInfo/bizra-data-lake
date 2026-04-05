# BIZRA Operations Runbook

Last updated: 2026-04-05

This runbook is the operator-focused guide for starting, validating, and troubleshooting BIZRA services.

## SEED Economic Settlement (Phase 91)

Every governed mission now settles SEED tokens via `bizra-node/src/seed_ledger.rs`. The settlement runs synchronously after `execute_governed_mission()` in the handler. Key constants:

| Parameter | Value | Source |
|-----------|-------|--------|
| Zakat rate | 2.5% at mint | `ZAKAT_RATE` in seed_ledger.rs |
| Ihsan floor | 0.95 minimum | `IHSAN_FLOOR` in seed_ledger.rs |
| Adl Gini max | 0.35 hard gate | `ADL_GINI_MAX` (enforced at URP level) |
| Emission decay | TTRL-style (cache efficiency) | `compute_emission_decay()` |

API token verification now uses constant-time comparison (`bizra-api/src/middleware/auth.rs`).

## Unified Stack Orchestration (Phase 89)

The BIZRA service mesh spans two Docker Compose projects joined by a shared `bizra-mesh` network. Use the unified orchestrator for standardized lifecycle management:

```bash
# Start all services in dependency order (PDCA cycle)
./scripts/node0_stack.sh start

# Health dashboard — all services, K8s, GPU, pilot status
./scripts/node0_stack.sh status

# Graceful shutdown (reverse dependency order)
./scripts/node0_stack.sh stop

# Full restart
./scripts/node0_stack.sh restart
```

### Network Architecture

All services join the external `bizra-mesh` network for cross-project DNS resolution:

```
bizra-mesh (external bridge)
├── kernel        (DATA-LAKE)    → resolves: synapse, wisdom, vectors, elite
├── python-api    (DATA-LAKE)
├── synapse       (Dual-Agentic) → Redis 6380
├── wisdom        (Dual-Agentic) → Neo4j 7474/7687
├── vectors       (Dual-Agentic) → ChromaDB 8001
├── postgres      (Dual-Agentic) → pgvector 5433
├── elite         (Dual-Agentic) → Rust PAT+SAT 8080
└── refinery      (Dual-Agentic) → 8081
```

### Dependency Pinning

- `runtime/requirements-kernel.txt` — Python deps with floor pins; `async-timeout>=4.0.3` explicit
- `runtime/Dockerfile.kernel` — smoke test gate: `RUN python -c "import redis, fastapi, torch; print('Smoke: OK')"`
- Rust deps locked via `Cargo.lock` in `bizra-omega/`

## Infrastructure Guardian

The BIZRA Infrastructure Guardian is a self-healing daemon that monitors Docker, memory, disk, EXT4 filesystem errors, critical services, systemd journal health, and port collisions. It implements an OODA (Observe-Orient-Decide-Act) loop with Ihsan quality scoring.

### Quick Commands

```bash
# Single health check
python3 scripts/guardian/infra_guardian.py --check

# Check + auto-correct known issues
python3 scripts/guardian/infra_guardian.py --correct

# JSON health report
python3 scripts/guardian/infra_guardian.py --report

# Install as systemd daemon
sudo bash scripts/guardian/install_guardian.sh
```

### Probes

| Probe | Monitors | Auto-Corrects |
|-------|----------|---------------|
| docker_socket | Socket exists, Docker reachable | Recreates symlink |
| container_health | All containers healthy/running | Restarts unhealthy (5min backoff) |
| memory | RAM usage % | Docker build cache prune at 92% |
| disk | `/` and `/mnt/c` usage | Docker system prune at 95% |
| ext4_errors | dmesg for filesystem errors | Reports (Windows-side fix required) |
| critical_services | Redis, PostgreSQL, ChromaDB | — |
| journal_health | systemd journal size | Vacuum to 200MB |
| banned_services | systemd Redis conflicts | Stop + disable |
| port_collisions | Known conflicts (8081) | Reports |

### EXT4 Disk Repair (Windows-side)

If the guardian reports EXT4 errors, run from Windows PowerShell (Admin):

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force
& "C:\BIZRA-DATA-LAKE\scripts\fix_docker_disk.ps1"
```

### Programmatic Access

```python
from core.proactive import InfraHealthProbe

probe = InfraHealthProbe()
report = probe.check()           # Quick check
report = probe.check_and_fix()   # Check + auto-correct
score  = probe.ihsan_score()     # 0.0 - 1.0
print(probe.summary())           # One-line status
```

## Node0 Performance Recovery (Windows + WSL)

Use this flow when Node0 is slow despite high-end hardware and you need a safe evidence-first cleanup sequence.

### Control Center Path

From `scripts/ops/CONTROL-CENTER.bat`:

- `8` Node0 Performance Snapshot (Analyze)
- `9` Node0 Performance Recovery (Dry Run)
- `10` Node0 Performance Recovery (Execute)
- `14` VHDX Compaction Snapshot (Analyze)
- `15` VHDX Compaction (Dry Run)
- `16` VHDX Compaction (Execute)
- `17` Pagefile Governance Snapshot (Analyze)
- `18` Pagefile Governance (Dry Run Apply)
- `19` Pagefile Governance (Execute Apply)
- `20` Schedule Post-Reboot VHDX Compact (One-Time)

### Direct Script Path

Run from Windows PowerShell (Admin recommended):

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force
& "C:\BIZRA-DATA-LAKE\scripts\ops\node0_performance_recovery.ps1" -Mode Analyze
& "C:\BIZRA-DATA-LAKE\scripts\ops\node0_performance_recovery.ps1" -Mode Remediate -DryRun
& "C:\BIZRA-DATA-LAKE\scripts\ops\node0_performance_recovery.ps1" -Mode Remediate -DryRun:$false
```

### Evidence Artifacts

Each execution writes an immutable snapshot report to:

- `C:\BIZRA-DATA-LAKE\logs\node0_performance_recovery_YYYYMMDD_HHMMSS.json`

Use these reports to compare pressure trends (disk %, Docker VHDX, Ubuntu VHDX, HF cache, `.wslconfig` caps) before and after remediation.
Reports now also include:
- live telemetry (`cpu_total_percent`, `disk_queue_length`, `disk_busy_percent`)
- process hotspots (`top_cpu`, `top_memory`)
- ranked remediation queue (`recommendations`) with `recommended_next_step`
- bottleneck summary (`dominant_bottleneck`, per-severity counts)

## Docker Volume Governance (k3d + Docker)

Use this workflow when active Docker/k3d volumes are the dominant storage pressure source and you need safe, auditable reclaim actions.

### Control Center Path

From `scripts/ops/CONTROL-CENTER.bat`:

- `11` Docker Volume Governance (Inventory)
- `12` Docker Volume Governance (Dry Run Reclaim)
- `13` Docker Volume Governance (Execute k3d Reclaim)

### Direct Script Path

Run from WSL or Windows terminal with Python available:

```bash
python scripts/ops/docker_volume_governance.py inventory
python scripts/ops/docker_volume_governance.py orphans
python scripts/ops/docker_volume_governance.py --dry-run reclaim-k3d
python scripts/ops/docker_volume_governance.py reclaim-k3d --restart-cluster
python scripts/ops/docker_volume_governance.py reclaim-all --dry-run
```

Mutating commands require explicit operator confirmation (`type YES`) unless `--yes` is provided.

### Evidence Artifacts

Each run writes a governance report to:

- `C:\BIZRA-DATA-LAKE\logs\docker_volume_governance_YYYYMMDD_HHMMSS.json`

## VHDX Compaction Governance (Windows Offline)

Use this flow when Docker volume cleanup has already reclaimed logical space but host disk does not improve because `docker_data.vhdx` is still large.

### Direct Script Path

Run from **Windows PowerShell** (Admin recommended, not inside WSL):

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force
& "C:\BIZRA-DATA-LAKE\scripts\ops\VHDX-COMPACTION-LAUNCHER.bat" -Mode Analyze
& "C:\BIZRA-DATA-LAKE\scripts\ops\VHDX-COMPACTION-LAUNCHER.bat" -Mode Compact -DryRun:$true -Target docker
& "C:\BIZRA-DATA-LAKE\scripts\ops\VHDX-COMPACTION-LAUNCHER.bat" -Mode Compact -DryRun:$false -Target docker
```

### Behavior

- Stops Docker service/processes.
- Runs `wsl --shutdown` to force offline compaction window.
- Runs deterministic `diskpart` `compact vdisk` against target VHDX.
- Restarts Docker service afterwards.
- Fails fast if free virtual memory is `< 2 GB` or if compact mode is not elevated.

### Evidence Artifacts

Each run writes a report to:

- `C:\BIZRA-DATA-LAKE\logs\vhdx_compaction_governance_YYYYMMDD_HHMMSS.json`

## Pagefile Governance (Windows Memory Backpressure)

Use this flow when elevated Windows operations fail with low virtual memory or pagefile pressure.

### Direct Script Path

Run from **Windows PowerShell**:

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force
& "C:\BIZRA-DATA-LAKE\scripts\ops\PAGEFILE-GOVERNANCE-LAUNCHER.bat" -Mode Analyze
& "C:\BIZRA-DATA-LAKE\scripts\ops\PAGEFILE-GOVERNANCE-LAUNCHER.bat" -Mode Apply -DryRun:$true -NoPrompt
& "C:\BIZRA-DATA-LAKE\scripts\ops\PAGEFILE-GOVERNANCE-LAUNCHER.bat" -Mode Apply -DryRun:$false
```

### Behavior

- Captures virtual memory + pagefile state before/after.
- Applies deterministic `C:\pagefile.sys` sizing (default 16384 MB initial / 32768 MB max).
- Requires elevation for non-dry-run apply.
- Marks `reboot_required=true` when changes are applied.

### Evidence Artifacts

Each run writes a report to:

- `C:\BIZRA-DATA-LAKE\logs\pagefile_governance_YYYYMMDD_HHMMSS.json`

## Post-Reboot Auto Compaction (One-Time)

Use this flow after pagefile apply when reboot is required and you want compaction to run automatically at next startup.

### Direct Script Path

```powershell
Set-ExecutionPolicy Bypass -Scope Process -Force
& "C:\BIZRA-DATA-LAKE\scripts\ops\schedule_post_reboot_vhdx_compaction.ps1" -Target docker
```

### Behavior

- Registers one-time task `BIZRA-PostReboot-VHDX-Compact` at next user logon with highest privilege.
- At logon, task runs `post_reboot_vhdx_compact_once.ps1`, executes compaction once, writes evidence, and unregisters itself.

### Evidence Artifacts

- `C:\BIZRA-DATA-LAKE\logs\schedule_post_reboot_vhdx_compaction_YYYYMMDD_HHMMSS.json`
- `C:\BIZRA-DATA-LAKE\logs\post_reboot_vhdx_compact_once_YYYYMMDD_HHMMSS.json`

## 1. Prerequisites

- Python 3.11+ with project dependencies installed
- Optional Rust toolchain for Omega services (`rustup`, stable)
- Optional local inference backend (LM Studio, Ollama, or compatible endpoint)

## 1.1 Node0 MVSA Canonical Flow

For Node0 MVSA, the only canonical operator surface is:

```bash
python scripts/node0_standalone.py activate --architect "MoMo"
python scripts/node0_standalone.py prove-mvsa
python scripts/node0_standalone.py task "write file missions/mvsa.txt :: node0 mvsa proof"
python scripts/node0_standalone.py health
```

Node0 documentation hierarchy for operators:

1. `docs/NODE0_STANDALONE_READINESS.md` — MVSA specification
2. `docs/constitutional/BIZRA-Node0-Activation-Planning-Principle-v1.0-DRAFT.md` — planning law and sequencing discipline
3. `docs/constitutional/BIZRA-Node0-Definition-of-Done-v1.0-LOCKED.md` — birth-gate verification
4. `docs/OPERATIONS_RUNBOOK.md` — operator procedure

`Ready Only` is the birth rule. Node0 is not complete unless `sovereign_state/node0_lifecycle.json` reports `status == "ready"`.

The planning principle governs what outranks what before execution.
The locked DoD remains the verification gate at sign-off time.

The verification entrypoint is:

```bash
bash scripts/node0_genesis_ceremony.sh
bash scripts/node0_genesis_ceremony.sh --full
bash scripts/node0_genesis_ceremony.sh --json
```

Authority is fail-closed and comes only from:

- `sovereign_state/node0_genesis.json`
- `sovereign_state/genesis_hash.txt`

The Rust MVSA proof artifact is:

- `sovereign_state/node0_mvsa_proof.json`

The lifecycle single source of truth is:

- `sovereign_state/node0_lifecycle.json`

`health` is read-only. It reports persisted MVSA state and restart recovery; it does not mutate lifecycle files.

Certification path:

- Native Linux is canonical.
- WSL2 is supported for compatibility only when code and state live on the Linux filesystem.
- `/mnt/c` is not a production hot path.

## 2. Start Sequence

### 2.1 Python Sovereign Runtime

```bash
cd /mnt/c/BIZRA-DATA-LAKE
source .venv/bin/activate
python -m core.sovereign
```

### 2.2 Sovereign API Server

```bash
cd /mnt/c/BIZRA-DATA-LAKE
source .venv/bin/activate
python -m core.sovereign.api --host 127.0.0.1 --port 8080
```

### 2.3 Desktop Bridge (TCP 9742)

```bash
cd /mnt/c/BIZRA-DATA-LAKE
source .venv/bin/activate
export BIZRA_BRIDGE_TOKEN=<your_token>
export BIZRA_RECEIPT_PRIVATE_KEY_HEX=<your_64_hex_key>
export BIZRA_NODE_ROLE=node   # use node0 only on genesis home base device
python -m core.bridges.desktop_bridge
```

If `BIZRA_NODE_ROLE=node0`, startup is fail-closed and requires:

- `sovereign_state/node0_genesis.json`
- `sovereign_state/genesis_hash.txt`
- matching genesis hash validation

Or via the full launcher (starts all services including bridge):

```bash
python -m core.sovereign.launch
```

### 2.4 Rust Workspace (Optional, Performance Path)

```bash
cd /mnt/c/BIZRA-DATA-LAKE/bizra-omega
cargo test --workspace
```

## 3. Health and Readiness Checks

### 3.1 API Health

```bash
curl -s http://127.0.0.1:8080/v1/health | jq
curl -s http://127.0.0.1:8080/v1/status | jq
```

### 3.2 Metrics (Prometheus Format)

```bash
curl -s http://127.0.0.1:8080/v1/metrics
```

Expected key signals:

- `sovereign_queries_total`
- `sovereign_query_success_rate`
- `sovereign_snr_score`
- `sovereign_ihsan_score`
- `sovereign_health_score`
- GoT/autonomy/cache counters and gauges

### 3.3 Desktop Bridge Health

```bash
# Ping test (requires BIZRA_BRIDGE_TOKEN)
python3 -c "
import os, socket, json, time, uuid
s = socket.socket(); s.connect(('127.0.0.1', 9742))
token = os.environ['BIZRA_BRIDGE_TOKEN']
msg = json.dumps({
    'jsonrpc': '2.0', 'method': 'ping', 'id': 1,
    'headers': {
        'X-BIZRA-TOKEN': token,
        'X-BIZRA-TS': int(time.time() * 1000),
        'X-BIZRA-NONCE': uuid.uuid4().hex,
    }
}).encode() + b'\n'
s.sendall(msg); print(s.recv(4096).decode()); s.close()
"
```

Expected: `{"jsonrpc":"2.0","result":{"status":"alive","uptime_s":...},"id":1}`

## 4. Smoke Validation

Run these before merging operationally significant changes:

```bash
cd /mnt/c/BIZRA-DATA-LAKE
source .venv/bin/activate
pytest -q tests/core/sovereign/test_runtime_types.py --capture=no
pytest -q tests/core/proof_engine/test_receipt.py --capture=no
pytest -q tests/core/sovereign/test_api_metrics.py --capture=no
```

For sovereign control-plane hardening validation (auth + receipts + latency + secret hygiene + Node0 genesis proof):

```bash
cd /mnt/c/BIZRA-DATA-LAKE
source .venv/bin/activate
export BIZRA_BRIDGE_TOKEN=your_bridge_token
export BIZRA_RECEIPT_PRIVATE_KEY_HEX=your_64_hex_key
python scripts/sape_masterpiece_gate.py --strict --json
```

## 5. Incident Triage

### Scenario A: `/v1/metrics` returns 500

1. Check server logs for `AttributeError` or serialization failures.
2. Confirm `core/sovereign/runtime_types.py` still exposes `RuntimeMetrics.to_prometheus(...)`.
3. Validate both handlers in `core/sovereign/api.py` call `to_prometheus(...)`.
4. Re-run `tests/core/sovereign/test_api_metrics.py`.

### Scenario B: SEL episodes are not committed

1. Inspect `core/sovereign/runtime_core.py` around `_commit_experience_episode`.
2. Verify `SovereignResult.processing_time_ms` is used (not `processing_time`).
3. Verify `result.model_used` is present in `SovereignResult` dataclass.

### Scenario C: Receipt verification starts failing after upgrade

1. Inspect `core/proof_engine/receipt.py` `SimpleSigner.public_key_bytes()`.
2. Confirm public key derivation remains SHA-256 for backward compatibility.
3. Run `tests/core/proof_engine/test_receipt.py`.

### Scenario D: Desktop Bridge connection refused

1. Verify bridge process is running: `ss -tlnp | grep 9742`
2. Check environment variables: `BIZRA_BRIDGE_TOKEN` and `BIZRA_RECEIPT_PRIVATE_KEY_HEX`
3. Bridge refuses startup without both variables set.
4. Re-run `tests/core/bridges/test_desktop_bridge.py`.

### Scenario E: Smart Files skill returns path error

1. Verify `BIZRA_DATA_LAKE_ROOT` resolves to the project root.
2. Smart Files rejects paths outside the data lake root (path traversal protection).
3. Re-run `tests/core/skills/test_smart_file_manager.py`.

### Scenario F: Node0 role startup is blocked

1. Confirm role: `echo $BIZRA_NODE_ROLE` (must be `node0` only on genesis machine).
2. Verify authority files exist:
   - `sovereign_state/node0_genesis.json`
   - `sovereign_state/genesis_hash.txt`
3. Validate chain:
   - `python -c "from pathlib import Path; from core.sovereign.origin_guard import validate_genesis_chain; print(validate_genesis_chain(Path('sovereign_state')))"`.
4. Re-run the canonical proof path:
   - `python scripts/node0_standalone.py prove-mvsa`
5. Inspect persisted artifacts:
   - `sovereign_state/node0_authority_migration.json`
   - `sovereign_state/node0_mvsa_proof.json`
   - `sovereign_state/node0_lifecycle.json`
6. If validation still fails, treat as tamper/corruption incident and stop startup until resolved.

## 6. Safe Rollback Strategy

- Prefer reverting only the smallest offending file set.
- Re-run smoke validation after rollback.
- Keep schema/contract compatibility for:
  - Metrics names
  - Receipt signer public key derivation
  - Runtime result dataclass fields
  - Bridge JSON-RPC error codes

## 7. Operational Logs and Artifacts

- Runtime state: `sovereign_state/`
- Temporary state: `tmp_state/` (should not be committed)
- Proof artifacts: `.proof-forge/`
- Bridge receipts: `sovereign_state/bridge_receipts/`

## 8. On-Call Escalation Inputs

Collect these before escalating:

1. `git rev-parse --short HEAD`
2. Exact failing command / endpoint
3. Relevant traceback
4. Output of:
   - `curl /v1/health`
   - `curl /v1/status`
   - `curl /v1/metrics`
5. Smoke test status (pass/fail list)
6. Bridge status: `ping` response or connection error
7. SEC-001 gate status: `python3 scripts/ci_blake3_gate.py`

---

## 9. Node0 Production Deployment

For production deployment of Node0 on native Linux, see the production repo:

- **Installer:** `bizra-node0/installers/install-node0-linux.sh`
- **Systemd unit:** `bizra-node0/deploy/node0/bizra-node0.service`
- **Logrotate:** `bizra-node0/deploy/node0/bizra-node0.logrotate`
- **Certification:** `bizra-node0/deploy/node0/certify-linux.sh`
- **Full runbook:** `bizra-node0/docs/OPERATIONS_RUNBOOK.md` §9
