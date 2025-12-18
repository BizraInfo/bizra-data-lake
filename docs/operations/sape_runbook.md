# SAPE Runbook (Kernel-Integrated)

This runbook describes how to use the **SAPE prompt engine** as a **first-class workflow** inside the **BIZRA Sovereign Kernel**.

## What You Get

- `POST /v1/sape/plan`: compiles a deterministic SAPE v1.0 plan (system + user prompts), optionally pulling evidence kernels from Neo4j.
- `POST /v1/sape/execute`: runs the compiled SAPE plan against your **sealed model-family routing**, enforcing **FATE (Ihsān/ʿAdl/Amānah)** and (optionally) **evidence gating**.

## Prerequisites

1) Python deps:

```powershell
pip install -r requirements-kernel.txt
```

2) At least one local LLM provider:

- **Ollama** running at `http://127.0.0.1:11434` (default)
- **LM Studio** OpenAI-compatible server at `http://127.0.0.1:1234/v1` (default)

3) Optional (recommended) graph evidence:

- Neo4j running locally
- `GRAPH_PASSWORD` set

## Required Environment Variables

- `BIZRA_API_TOKEN`: required for all endpoints except `/healthz` and `/livez`

## Optional Environment Variables

- Kernel runtime:
  - `BIZRA_KERNEL_HOST` (default `127.0.0.1`)
  - `BIZRA_KERNEL_PORT` (default `8010`)
  - `BIZRA_ENV` (`development|ci|production`)
  - `BIZRA_FATE_STRICT` (`1` fail-closed if constitution missing)

- Neo4j:
  - `GRAPH_PASSWORD` (required if you want graph evidence or `/v1/agent/query`)
  - `NEO4J_URI` (default `bolt://localhost:7687`)
  - `NEO4J_USER` (default `neo4j`)
  - `NEO4J_DATABASE` (optional)

- Model family (sealed routing):
  - `BIZRA_MODEL_FAMILY_MANIFEST` (default `model-family-genesis-v1-SEALED.yaml`)
  - `BIZRA_ALLOW_UNSEALED_MODEL_FAMILY=1` (dev override; not recommended)

- LLM endpoints:
  - `OLLAMA_URL` or `OLLAMA_HOST` (default `http://127.0.0.1:11434`)
  - `LMSTUDIO_URL` or `BIZRA_LMSTUDIO_URL` (default `http://127.0.0.1:1234/v1`)

- Receipts:
  - `BIZRA_KERNEL_RECEIPTS` (`1` default)
  - `BIZRA_KERNEL_RECEIPT_DIR` (default `docs/evidence/receipts/`)
  - `BIZRA_KERNEL_RECEIPTS_INCLUDE_PROMPTS=1` (stores raw prompts in receipts; default off)

## Start the Kernel

```powershell
$env:BIZRA_API_TOKEN = "genesis_access"
$env:GRAPH_PASSWORD = "bizra_genesis_key"   # optional, but required for Neo4j evidence
python -m core.main
```

Verify:

- `GET http://127.0.0.1:8010/livez`
- `GET http://127.0.0.1:8010/healthz`
- `GET http://127.0.0.1:8010/` (requires token)
- Swagger: `http://127.0.0.1:8010/docs`

## Run SAPE via Docker Compose (Citadel)

If you are running the Kernel inside the Citadel compose stack:

1) Create `.env` from `.env.example:1` (set `BIZRA_API_TOKEN`, `NEO4J_AUTH`).
2) Ignite + verify:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\ignite_node0.ps1
powershell -ExecutionPolicy Bypass -File .\scripts\verify_node0.ps1
```

3) Test SAPE (no browser dependency):
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\test_sape_plan.ps1
```

Citadel defaults:
- Kernel base URL: `http://127.0.0.1:8010`
- SAPE plan endpoint: `POST http://127.0.0.1:8010/v1/sape/plan`

## Use SAPE (Plan)

```powershell
$token = $env:BIZRA_API_TOKEN
$body = @{
  domain = "BIZRA Sovereign Kernel"
  objective = "Integrate SAPE as an evidence-gated runtime workflow across services."
  stakes = "H"
  constraints = "Windows, local-only LLMs, no network calls, fail-closed ethics."
  success_criteria = "Kernel exposes /v1/sape endpoints; receipts+metrics; docs updated."
  evidence_topics = @("kernel", "neo4j", "model-family", "FATE", "SAPE")
  evidence_limit = 8
} | ConvertTo-Json -Depth 6

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8010/v1/sape/plan" `
  -Headers @{ Authorization = "Bearer $token" } `
  -ContentType "application/json" `
  -Body $body
```

Expected:

- `status = PLANNED`
- `seal.verdict = APPROVED`
- `system_prompt` + `user_prompt`
- `candidate_models[]` from the sealed manifest

Fail-closed behavior:

- If `stakes="H"` and graph evidence is required but Neo4j is offline → `status = BLOCKED_BY_EVIDENCE`

## Use SAPE (Execute)

```powershell
$token = $env:BIZRA_API_TOKEN
$body = @{
  domain = "BIZRA System"
  objective = "Produce a prioritized roadmap integrating architecture, security, performance, docs, and Ihsan."
  stakes = "H"
  constraints = "Use only repo + graph evidence; output actionable backlog + gates."
  success_criteria = "Roadmap with owners, risks, CI/CD gates, acceptance criteria."
  lenses = @("Systems Architect","Pragmatic Engineer","Ethicist")
  evidence_topics = @("UNIFIED_ACTION_FRAMEWORK", "kernel", "neo4j", "FATE", "model-family")
  evidence_limit = 10
  include_prompts_in_response = $false
  max_model_attempts = 3
} | ConvertTo-Json -Depth 6

Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8010/v1/sape/execute" `
  -Headers @{ Authorization = "Bearer $token" } `
  -ContentType "application/json" `
  -Body $body
```

Expected:

- `status = SUCCESS`
- `model_used` + `provider_used`
- `attempts[]` (includes fallback attempts)
- `output_text` (the SAPE-filled blueprint)

## Evidence & Observability

- Receipts: `docs/evidence/receipts/` (folders `bizra_sape_*_receipt_v1`)
- Metrics (Prometheus): `GET /metrics`
  - `bizra_kernel_sape_requests_total`
  - `bizra_kernel_sape_llm_calls_total`
  - `bizra_kernel_sape_llm_latency_seconds`
