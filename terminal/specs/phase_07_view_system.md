# Phase 07 — View 7: SYSTEM (Health + Security)

> **Purpose:** Node health, security posture, service status.
> **Status:** PARTIAL — sovereign_terminal.py has status(). Rust TUI has FATE gauges.

## 7.1 Content Spec

```
┌── SYSTEM ─────────────────────────────────────────────────────┐
│                                                                │
│  Node: BIZRA-00000000 | Ed25519: d3e4f5a6... | v10.0.0       │
│  Status: SAFE | Uptime: 4h 23m | Backend: LIVE               │
│                                                                │
│  ── Constitutional Metrics ───────────────────────────────    │
│  Ihsan:  0.9587  ███████████████████░  [>= 0.95 MINTING]    │
│  SNR:    0.9234  ██████████████████░░  [>= 0.85 PASS]       │
│  Gini:   0.2134  ████████████████░░░░  [<= 0.35 PASS]       │
│  Myelin: 67.2%   █████████████░░░░░░░  (S1 hit rate)        │
│                                                                │
│  ── Services ─────────────────────────────────────────────    │
│  Sovereign API    :8010   HEALTHY   75 endpoints              │
│  PostgreSQL+pgvec :5433   HEALTHY   vector DB                 │
│  Redis (cache)    :6379   HEALTHY   messaging                 │
│  Redis (synapse)  :6380   HEALTHY   inter-agent               │
│  Neo4j            :7474   HEALTHY   wisdom graph              │
│  ChromaDB         :8001   HEALTHY   vector store              │
│  Ollama           :11434  HEALTHY   local LLM                 │
│  Grafana          :3000   HEALTHY   monitoring                │
│  Prometheus       :9090   HEALTHY   metrics                   │
│                                                                │
│  ── Evidence ─────────────────────────────────────────────    │
│  Chain: 147 blocks | Integrity: VERIFIED | Hash algo: BLAKE2b│
│                                                                │
│  ── Security ─────────────────────────────────────────────    │
│  Headers:     6/6 (HSTS, CSP, X-Frame, Referrer, Perms, Type)│
│  Auth:        JWT + API Key | Rate limit: 100 req/min         │
│  SAST:        Bandit PASS | Cargo-Audit PASS                  │
│  DAST:        ZAP Baseline PASS                               │
│  Vault:       Fernet + PBKDF2 (600K iterations)               │
│                                                                │
│  ── Runtime ──────────────────────────────────────────────    │
│  Model:     qwen2.5-14b-instruct (Ollama)                    │
│  WebSocket: DISCONNECTED                                      │
│  K8s:       k3d-bizra-prod (2 agents, Argo canary)           │
└──────────────────────────────────────────────────────────────┘
```

## 7.2 Data Model

```pseudocode
struct SystemView:
    // Identity
    node_id: str
    public_key_prefix: str
    version: str

    // Overall status
    status: "SAFE" | "DEGRADED" | "OFFLINE" | "CONSTITUTIONAL_VIOLATION"
    uptime: Duration
    backend_mode: "LIVE" | "OFFLINE"

    // Constitutional
    ihsan: float
    snr: float
    gini: float
    myelination: float          # S1 hit rate
    mint_status: "MINTING" | "PAUSED" | "VIOLATION"

    // Services
    services: [{
        name: str
        port: int
        status: "HEALTHY" | "UNHEALTHY" | "UNKNOWN"
        detail: str
    }]

    // Evidence
    chain_height: int
    chain_valid: bool
    hash_algorithm: str         # "BLAKE2b-256"

    // Security posture
    security: {
        headers: str            # "6/6"
        auth: str               # "JWT + API Key"
        rate_limit: str         # "100 req/min"
        sast: str               # "PASS" or "FAIL"
        dast: str               # "PASS" or "FAIL"
        vault: str              # "Fernet + PBKDF2"
    }

    // Runtime
    model: str
    websocket: "CONNECTED" | "DISCONNECTED"
    k8s_context: str

function determine_status(ihsan, snr, gini, services) -> str:
    IF ihsan < 0.85 OR gini > 0.35:
        RETURN "CONSTITUTIONAL_VIOLATION"
    IF any(s.status == "UNHEALTHY" for s in services):
        RETURN "DEGRADED"
    IF all(s.status == "UNKNOWN" for s in services):
        RETURN "OFFLINE"
    RETURN "SAFE"
```

## 7.3 Existing Implementation

**Python (sovereign_terminal.py:178-243):** `status()` — Shows identity, Ihsan, SNR, Gini, myelination, containers, SEED/BLOOM. Missing: services list, security posture, K8s context.

**Rust (widgets/fate_gauge.rs:1-147):** FATE gauge widget — 4 bars (Ihsan, Adl, Harm, Confidence). Can be extended.

**Rust (app.rs):** Dashboard tab renders FATE gauges + agent cards.

## 7.4 What to Build

| Component | Surface | LOC Est | Priority |
|-----------|---------|---------|----------|
| Service health table | Both | 80 | P0 |
| Security posture panel | Both | 60 | P1 |
| K8s/Docker status probe | Python | 40 | P1 |
| WebSocket status indicator | Both | 20 | P1 |
| Constitutional violation alert | Both | 40 | P0 |

## 7.5 TDD Anchors

```
TEST: system_status_safe
  GIVEN ihsan=0.96, gini=0.21, all services healthy
  THEN status = "SAFE"

TEST: system_status_degraded
  GIVEN ihsan=0.96, gini=0.21, redis unhealthy
  THEN status = "DEGRADED"

TEST: system_status_violation
  GIVEN ihsan=0.80 (below 0.85)
  THEN status = "CONSTITUTIONAL_VIOLATION"

TEST: system_mint_status_from_ihsan
  GIVEN ihsan >= 0.95
  THEN mint_status = "MINTING"
  GIVEN ihsan < 0.95 AND ihsan >= 0.85
  THEN mint_status = "PAUSED"

TEST: system_chain_integrity_check
  GIVEN 147 valid chained events
  WHEN verify called
  THEN chain_valid = true
```
