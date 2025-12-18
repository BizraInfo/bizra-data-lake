# Architecture Map (Evidence-Backed)

Truth: DERIVED (compiled from a repo scan). Anything marked **VERIFIED** includes a `path:line` anchor.

## 0) Fast Orientation (what to run, where truth lives)

- Two runnable HTTP APIs default to port `8080` (avoid running both on the same port):
  - Rust Core (meta-alpha orchestrator) binds `127.0.0.1:8080` via `create_http_server(system, 8080)` in `src/main.rs:37` (listener bind in `src/http.rs:48`).
  - Node0 backend (Genesis Node) defaults to `127.0.0.1:8080` via `API_PORT` fallback in `bizra-genesis-node/backend/src/main.rs:294`.
- Constitution/policy SoT lives in `constitution/` (Ihsan thresholds, Lexicon) and is enforced by CI in `.github/workflows/phase0_integrity.yml:1`.
- "Sim vs Real" honesty is explicit in Rust core via `AdapterModes::current()` returning all `Simulated` in `src/types.rs:160`.

## 1) Top-Level Repo Topology (foundation -> user-facing)

```
BIZRA-Dual-Agentic-system--main/
  src/                         Rust core orchestrator (PAT/SAT/Bridge + HTTP)
  constitution/                Canonical Ihsan + Lexicon + genesis manifest profile
  schemas/                     JSON Schemas for constitution-ledger artifacts
  tools/                       CI-enforced lints + receipt generators + sealing tools
  docs/                        SDLC/ops/security/runtime process pack
  .github/workflows/           Phase-0 integrity gates
  bizra-genesis-node/          Node0 stack (backend + telemetry bridge + dashboard + python)
    backend/                   Rust Axum API (DB + services + PoI + assets)
    bridge/                    WS telemetry bridge (polls backend + broadcasts)
    apps/dashboard/            Next.js UI (user-facing)
    bizra_kernel/              Python kernel utilities (SAPE/SNR/Ihsan tooling)
  ace-framework/               Node.js wrappers (e.g., Ollama generate + Ihsan prompt)
  .bizra-kernel/               Legacy/experimental kernel pack (drift risk)
  evidence/                    Sealed outputs/logs (integrity receipts)
```

## 2) Runtime Surfaces & Call Graphs

### 2.1 Rust Core - "Meta Alpha Dual Agentic" (foundation runtime)

Purpose: PAT/SAT orchestration with an HTTP surface and constitution-based Ihsan scoring.

Entrypoints (VERIFIED):
- Binary entry: `src/main.rs:1`
- Library entry: `src/lib.rs:1`
- HTTP server: `src/http.rs:24` (exported as `create_http_server` in `src/lib.rs:47`)

Core call graph (VERIFIED):

```
HTTP POST /dual/execute
  -> system.execute(request)                         src/http.rs:140
    -> BridgeCoordinator.execute(request)            src/lib.rs:38, src/bridge.rs:30
      -> SAT.validate_request                        src/bridge.rs:39
      -> PAT.execute_parallel                        src/bridge.rs:53
      -> SAT.evaluate_results                        src/bridge.rs:58
      -> Ihsan score + threshold gate                src/bridge.rs:72
      -> DualAgenticResponse(meta.adapter_modes=...) src/bridge.rs:94
```

Key modules (VERIFIED by module declarations): `src/lib.rs:3`
- `src/bridge.rs` - Orchestrates SAT->PAT->SAT and computes Ihsan vector/score.
- `src/pat.rs` / `src/sat.rs` - "teams" (currently simulated confidence outputs).
- `src/ihsan.rs` - Loads `constitution/ihsan_v1.yaml` and computes score/thresholding.
- `src/http.rs` - Axum server, token auth, loopback-only + allowlist CORS.
- `src/types.rs` - Request/response types + `AdapterMode(s)` (truth labeling).
- `src/mcp.rs`, `src/a2a.rs`, `src/reasoning.rs`, `src/pat_enhanced.rs` - capability scaffolds.

Determinism/realism status (VERIFIED):
- PAT confidence uses a time-seeded PRNG (`SystemTime::now()`) in `src/pat.rs:162` -> non-deterministic outputs by design.
- Adapter modes default to simulated in `src/types.rs:160` and are surfaced in API responses in `src/http.rs:76` and `src/bridge.rs:106`.

### 2.2 Node0 Genesis Node (monorepo) - backend + bridge + dashboard

Purpose: a Node0 "Genesis Node" stack: Rust API backend, WS telemetry fanout, and a Next.js dashboard UI.

#### 2.2.1 Node0 Backend (Rust / Axum / SQLx)

Entrypoint (VERIFIED): `bizra-genesis-node/backend/src/main.rs:1`

Services exposed (VERIFIED routes): `bizra-genesis-node/backend/src/main.rs:144`
- Health: `GET /health`
- Service status: `GET /api/services/status`
- Env snapshot: `GET /api/env/snapshot`
- PAT: `POST /api/pat/chat`, `GET /api/pat/agents`, `POST /api/pat/configure`
- PoI: `POST /api/poi/log`, `POST /api/poi/lexicon-receipt`, `GET /api/poi/stats`, `GET /api/poi/timeline`
- Resources: `POST /api/resources/configure`, `GET /api/resources/status`
- Assets: `POST /api/assets/index`, `GET /api/assets/search`, `GET /api/assets/stats`

Module layout (VERIFIED): `bizra-genesis-node/backend/src/lib/mod.rs:1`
- `core/` - circuit breaker, rate limit, caching, scheduler, metrics.
- `services/` - PoI ledger, asset registry, knowledge, resource pool, env snapshot.
- `agents/` - PAT/SAT orchestration.
- `api/` - additional API modules (e.g., knowledge).

Security posture (VERIFIED):
- Secure-by-default bind: loopback unless explicitly exposing with audit reason in `bizra-genesis-node/backend/src/main.rs:277`.
- CORS: loopback + allowlist only (predicate) in `bizra-genesis-node/backend/src/main.rs:132`.

Hotspot (VERIFIED): hard-coded default DB password fallback:
- `DB_PASSWORD` defaults to `bizra_secure_2025` in `bizra-genesis-node/backend/src/main.rs:79` (also in docker-compose env defaults).

#### 2.2.2 Telemetry Bridge (TypeScript / ws)

Entrypoint (VERIFIED): `bizra-genesis-node/bridge/src/index.ts:1`

Integration points (VERIFIED):
- Polls backend API: `config.apiUrl` defaults to `http://localhost:8080` in `bizra-genesis-node/bridge/src/index.ts:19`.
- Calls endpoints: `/api/services/status` and `/api/poi/stats` in `bizra-genesis-node/bridge/src/index.ts:63` and `bizra-genesis-node/bridge/src/index.ts:77`.
- Broadcasts WS telemetry on port `3002` by default in `bizra-genesis-node/bridge/src/index.ts:18`.

Truth note (VERIFIED): resource metrics are currently simulated:
- `Math.random()` used for CPU/memory/GPU/latency/error-rate in `bizra-genesis-node/bridge/src/index.ts:107`.

#### 2.2.3 Dashboard (Next.js / React) - user-facing UI

API client (VERIFIED): `bizra-genesis-node/apps/dashboard/src/lib/api.ts:8`
- `NEXT_PUBLIC_API_URL` default: `http://localhost:8080`

Telemetry hook (VERIFIED): `bizra-genesis-node/apps/dashboard/src/hooks/useGenesisSynapse.ts:78`
- Default WS URL: `ws://localhost:3002`
- Includes basic validation + allowlist checks on protocol/host.

### 2.3 ACE Framework (Node.js) - model runtime wrapper

Purpose: wraps Ollama generation requests with safety keyword blocking + an injected "Ihsan system prompt".

Entrypoint (VERIFIED): `ace-framework/orchestrator-ihsan-wrapper.js:1`
- Calls Ollama `POST /api/generate` at `OLLAMA_HOST` in `ace-framework/orchestrator-ihsan-wrapper.js:82`.

## 3) Constitution, Schemas, Tooling (the enforcement spine)

Constitution (VERIFIED by CI wiring): `.github/workflows/phase0_integrity.yml:1`
- Ihsan policy: `constitution/ihsan_v1.yaml:1`
- Lexicon ledger: `constitution/lexicon_v1.yaml:1`, `constitution/lexicon_ledger_contract_v1.yaml:1`
- Genesis manifest profile: `constitution/genesis_manifest_profile_v1.yaml:1`

Schemas (VERIFIED):
- `schemas/lexicon_v1.schema.json:1`
- `schemas/lexicon_receipt_v1.schema.json:1`

CI-enforced tooling (VERIFIED): `.github/workflows/phase0_integrity.yml:1`
- `tools/truth_lint.py` - prevents docs/claims drift (Truth labels).
- `tools/lexicon_lint.py` + `tools/lexicon_tamper_test.py` - ledger integrity.
- `tools/ihsan_parity_check.py` - Rust/Python parity against the same constitution.
- `tools/node0_secure_defaults_lint.py`, `tools/node0_warning_budget.py` - Node0 hardening ratchets.

## 4) Known Drift / Hotspot Watchlist (highest SNR first)

1) Port collision risk (DERIVED, evidence-backed):
   - Rust core server binds `127.0.0.1:8080` by default (`src/main.rs:37`).
   - Node0 backend binds `*:8080` by default (`bizra-genesis-node/backend/src/main.rs:294`).
   - Dashboard and bridge both default to `http://localhost:8080` (`bizra-genesis-node/apps/dashboard/src/lib/api.ts:8`, `bizra-genesis-node/bridge/src/index.ts:19`).
2) Unsafe DB password default (VERIFIED): `bizra-genesis-node/backend/src/main.rs:79`.
3) Embedding dimension drift (VERIFIED):
   - Slots/config set to `768` in `docs/runtime/slots.yaml:56` and `model-family-genesis-v1-SEALED.yaml:88`.
   - Golden set expects `384` in `golden-set-genesis-v1-DETERMINISTIC.json:72` (likely stale test vector).
4) Telemetry realism gap (VERIFIED): `Math.random()` in `bizra-genesis-node/bridge/src/index.ts:107`.
5) Ihsan scoring drift inside Node0 Python kernel (VERIFIED):
   - `system_protocol_kernel.py` hard-codes weights/threshold (`bizra-genesis-node/system_protocol_kernel.py:46`).
   - Separate constitution-aware implementation exists in `bizra-genesis-node/bizra_kernel/ihsan_vector.py:109`.
6) Non-deterministic PAT confidence (VERIFIED): time-seeded PRNG `src/pat.rs:162`.

## 5) Debugging Pathways (symptom -> fastest file/area)

- "Address already in use" / bind failure
  - Decide which API owns `:8080` and move the other:
    - Rust core: change `create_http_server(system, 8080)` call in `src/main.rs:37`.
    - Node0 backend: set `API_PORT` (see default in `bizra-genesis-node/backend/src/main.rs:294`).
    - Bridge/Dashboard: update `API_URL` / `NEXT_PUBLIC_API_URL` (`bizra-genesis-node/bridge/src/index.ts:19`, `bizra-genesis-node/apps/dashboard/src/lib/api.ts:8`).

- Dashboard loads but shows no telemetry
  - Check WS server is up on `:3002` (`bizra-genesis-node/bridge/src/index.ts:18`).
  - Confirm dashboard hook URL (`bizra-genesis-node/apps/dashboard/src/hooks/useGenesisSynapse.ts:78`).

- Node0 backend fails on startup with DB errors
  - Verify env `DATABASE_URL` or `DB_*` vars (constructed in `bizra-genesis-node/backend/src/main.rs:75`).
  - Confirm Postgres container/user align with compose defaults (see `bizra-genesis-node/docker/`).

- Ihsan gate failures / parity mismatches
  - Canonical policy: `constitution/ihsan_v1.yaml:1`
  - Rust implementation: `src/ihsan.rs:1`
  - Python implementation: `bizra-genesis-node/bizra_kernel/ihsan_vector.py:109`
  - Drift candidate: `bizra-genesis-node/system_protocol_kernel.py:46`

- CI Phase-0 failing
  - Start at `.github/workflows/phase0_integrity.yml:1` to identify which lint/gate failed, then run the same tool locally from `tools/`.

## 6) AI Review Prompt Pack (SAPE, repo-specific)

Use these prompts to drive high-signal reviews without introducing simulation theater.

### Symbolic (invariants, constitutions, contracts)
- "List all constitution-backed invariants (Ihsan thresholds, lexicon append-only, secure defaults). For each, show the enforcing code path and CI gate. Identify any alternate implementations bypassing the constitution."

### Abstraction (boundaries, APIs, dependency hygiene)
- "Draw the boundary between Rust core vs Node0 backend vs dashboard/bridge vs ACE wrapper. Identify duplicated responsibilities (ports, Ihsan scoring, telemetry) and propose one consolidation that reduces drift risk."

### Probe (acceptance tests, failure modes, SNR)
- "Design 5 acceptance tests that increase determinism and truth alignment (e.g., embed dim match, no hard-coded DB secrets, telemetry 'simulated' flag, Ihsan parity, port collision detection). For each test, specify the exact file(s) and a CI step."

### Elevation (architectural evolution with Ihsan)
- "Propose the smallest change that measurably increases auditability (Amanah) without slowing delivery: what receipt/manifest should be emitted, where stored, and how verified in CI?"

