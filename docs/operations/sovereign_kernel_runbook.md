# Sovereign Kernel Runbook (Runtime Interface)

Purpose: expose the House of Wisdom (Neo4j) to local apps via a token-gated FastAPI service that enforces Ihsān/ʿAdl/Amānah through a fail-closed FATE gate.

Truth labels: VERIFIED | MEASURED | TARGET | DERIVED

---

## 1) Secure Defaults (Amānah)

- Local-only bind by default: `127.0.0.1` (override via `BIZRA_KERNEL_HOST` only when intended).
- Token required for all endpoints except `GET /healthz` (set `BIZRA_API_TOKEN`).
- Neo4j credentials must come from env vars (set `GRAPH_PASSWORD` / `NEO4J_PASSWORD`).

---

## 2) Preconditions (House of Wisdom)

- Neo4j running locally: `docs/operations/neo4j_runbook.md:1`
- Knowledge graph ingested (creates `:Artifact` nodes): `bizra_synaptic_loader.py:1`

---

## 3) Start the Kernel (Windows / PowerShell)

Install deps (if needed):
```powershell
python -m pip install -r requirements-kernel.txt
```

From repo root (`C:\BIZRA-Dual-Agentic-system--main`):
```powershell
$env:BIZRA_API_TOKEN = \"CHANGE_ME\"
$env:GRAPH_PASSWORD = \"CHANGE_ME\"
$env:BIZRA_KERNEL_HOST = \"127.0.0.1\"
$env:BIZRA_KERNEL_PORT = \"8010\"

python -m core.main
```

Optional: create a junction so `C:\BIZRA-PROJECTS\00-GENESIS` can run the same code:
```powershell
.\scripts\install_sovereign_kernel_junction.ps1
```

If PowerShell script execution is restricted on your machine:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\install_sovereign_kernel_junction.ps1
```

### 3.1 Start via Docker Compose (Citadel)

This mode turns the Kernel + its dependencies into a service-oriented "citadel" (Redis + Neo4j + Chroma + Kernel), with deterministic health probes and acceptance gates.

1) Create your local `.env` (never commit it):
- Copy `.env.example:1` to `.env` and set at minimum:
  - `BIZRA_API_TOKEN`
  - `NEO4J_AUTH`

2) Ignite:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\ignite_node0.ps1
```

3) Verify acceptance gates:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\verify_node0.ps1
```

4) Mint a replayable Genesis Receipt (post-gates; no secrets written):
```powershell
python .\scripts\genesis_receipt.py
```

Kernel URLs (Citadel default):
- `http://127.0.0.1:8010/livez`
- `http://127.0.0.1:8010/healthz`
- `http://127.0.0.1:8010/docs`

### 3.2 One Command Ignition (P0)

Use the unified ignition flow when you need an operator-proof bring-up that seals evidence automatically.

```powershell
pwsh .\scripts\genesis_ignite_all.ps1
```

Expected outcomes:
- Stack resets, ignites, and passes Gates A–E (see [scripts/verify_node0.ps1](scripts/verify_node0.ps1)).
- First SAPE plan yields `BLOCKED_BY_EVIDENCE` before seeding and `PLANNED` after deterministic seeding.
- A timestamped receipt is written under `docs/evidence/receipts/` and validates against `schemas/genesis_receipt_v1.schema.json`.

Failure diagnostics:
- Container state: `docker compose ps`
- Kernel logs: `docker compose logs --tail 200 kernel`
- Gate snapshot: [docs/evidence/gates/node0_gates_latest.json](docs/evidence/gates/node0_gates_latest.json)

---

## 4) Verify

Liveness (no token; always 200):
```powershell
Invoke-RestMethod http://127.0.0.1:8010/livez
```

Readiness (no token; 200 only when core deps are reachable):
```powershell
Invoke-RestMethod http://127.0.0.1:8010/healthz
```

Heartbeat (token required):
```powershell
Invoke-RestMethod http://127.0.0.1:8010/ -Headers @{ \"X-BIZRA-TOKEN\" = $env:BIZRA_API_TOKEN }
```

Example query:
```powershell
$body = @{ agent_id = \"app_01\"; intent = \"tokenomics\"; context = \"\"; limit = 10 } | ConvertTo-Json
Invoke-RestMethod http://127.0.0.1:8010/v1/agent/query -Method Post -Headers @{ \"X-BIZRA-TOKEN\" = $env:BIZRA_API_TOKEN } -ContentType \"application/json\" -Body $body
```

SAPE (plan/execute) endpoints:
- See `docs/operations/sape_runbook.md:1`

---

## 5) Evidence Receipts (Accountability)

The kernel writes sanitized per-request receipts under:
- `docs/evidence/receipts/kernel_request_*/receipt.json`

Disable receipts:
- set `BIZRA_KERNEL_RECEIPTS=0`
