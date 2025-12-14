# BIZRA Node₀ Repository Structure & Master Manifest

**Status:** Architecture v1.0 - Genesis Node Bootstrap  
**Authority:** MoMo - First Architect  
**Last Updated:** 2025-12-11  
**Classification:** Foundation Architecture Blueprint

---

## Table of Contents

1. [Repository Overview](#repository-overview)
2. [Directory Structure](#directory-structure)
3. [Core Services Architecture](#core-services-architecture)
4. [Critical Configuration Files](#critical-configuration-files)
5. [Deployment Models](#deployment-models)
6. [Integration & Data Flow](#integration--data-flow)
7. [Operational Readiness](#operational-readiness)
8. [Next Steps](#next-steps)

---

## Repository Overview

**Repository Name:** `bizra-genesis-node`  
**Primary Function:** Bootstrap kernel for BIZRA Block₀ and Node₀  
**Target Environment:** 
- Local: Docker Compose (single machine - MSI i9, RTX 4090, 128GB RAM)
- Production: Kubernetes cluster (bizra-prod namespace)

**Key Principle:** One codebase, two deployment models. Services are container-agnostic; orchestration changes but architecture is stable.

---

## Directory Structure

```
bizra-genesis-node/
│
├── README.md                           # Project overview & quick-start
├── ARCHITECTURE.md                     # This file (system design)
├── LICENSE                             # MIT or Apache 2.0
├── .gitignore
│
├── bizra-ledger/                       # Blockchain & Consensus Layer
│   ├── genesis.json                    # Genesis configuration (template)
│   ├── genesis.built.json              # Signed genesis (generated)
│   ├── genesis_merkle_root.txt         # Merkle root hash (generated)
│   ├── genesis_build.py                # Genesis builder script
│   ├── genesis_verify.py               # Genesis verification script
│   ├── node0_signing.key               # Node₀ Ed25519 private key (secrets)
│   ├── node0_signing.pub               # Node₀ Ed25519 public key
│   ├── blockgraph/                     # BlockGraph consensus implementation
│   │   ├── consensus.rs                # WQ-refs finality logic
│   │   ├── dag.rs                      # DAG data structures (Block, Header, PoI refs)
│   │   ├── crypto.rs                   # Ed25519, SHA-256, optional BLS
│   │   ├── lib.rs                      # Crate root
│   │   └── Cargo.toml
│   ├── poi-oracle/                     # Proof of Impact attestation service
│   │   ├── poi_spec.md                 # PoI attestation format & validation rules
│   │   ├── poi_cli.py                  # CLI for creating PoI attestations
│   │   ├── attestation_schema.json     # JSON schema for PoI claims
│   │   └── validator.py                # Attestation validator logic
│   └── chain-data/                     # Persistent blockchain state (git-ignored in prod)
│       └── .gitkeep
│
├── services/                           # Microservices Layer (Layer 2)
│   │
│   ├── training/                       # Training Service
│   │   ├── Dockerfile
│   │   ├── docker-compose.override.yml # Local GPU settings
│   │   ├── app.py                      # PyTorch training loop
│   │   ├── model.py                    # BIZRA MoE foundation model
│   │   ├── metrics.py                  # Prometheus metrics export
│   │   ├── config.yaml                 # Training hyperparameters
│   │   ├── requirements.txt
│   │   ├── tensorboard_launcher.sh
│   │   └── datasets/                   # DVC-tracked training data
│   │       └── .gitkeep
│   │
│   ├── aggregation/                    # Federated Aggregation Service
│   │   ├── Dockerfile
│   │   ├── app.py                      # Flask/FastAPI REST API
│   │   ├── server.py                   # ProductionFederatedLearningServer
│   │   ├── aggregation_logic.py        # Byzantine-robust median + DP
│   │   ├── impact_scoring.py           # PoI weight computation
│   │   ├── blockchain_recorder.py      # Event logging to BlockGraph
│   │   ├── policy_engine.py            # Policy evaluation (rate limits, trust)
│   │   ├── requirements.txt
│   │   └── config.yaml
│   │
│   ├── validation-api/                 # Validation & Attestation API (bizra_validation_api)
│   │   ├── Dockerfile
│   │   ├── server.js                   # Express.js or Node.js HTTP server
│   │   ├── routes/
│   │   │   ├── poi.js                  # /api/v1/proof-of-impact/* routes
│   │   │   └── resources.js            # /api/v1/resources/* routes
│   │   ├── validators/
│   │   │   ├── attestation.js          # Attestation schema & crypto validation
│   │   │   └── genesis.js              # Genesis validation against merkle root
│   │   ├── services/
│   │   │   ├── trust_anchor.js         # Load & cache genesis.built.json + pubkeys
│   │   │   └── blockchain_client.js    # RPC calls to local BlockGraph node
│   │   ├── package.json
│   │   └── config.json                 # API ports, TLS cert paths
│   │
│   ├── blockgraph-node/                # BlockGraph Full Node (Validator/Bootstrap)
│   │   ├── Dockerfile
│   │   ├── entrypoint.sh               # Node startup script
│   │   ├── config.toml                 # Node config (peer list, validator mode, genesis)
│   │   ├── Cargo.toml                  # Rust package definition
│   │   ├── src/
│   │   │   ├── main.rs                 # Node daemon & RPC server
│   │   │   ├── p2p.rs                  # Libp2p gossip mesh setup
│   │   │   ├── rpc.rs                  # JSON-RPC endpoints for clients
│   │   │   └── lib.rs
│   │   └── data/                       # Persistent blockchain state
│   │       └── .gitkeep
│   │
│   └── resource-pool-coordinator/      # Resource Pool Manager
│       ├── Dockerfile
│       ├── app.py                      # Resource allocation API
│       ├── scheduler.py                # Task scheduler for contributor jobs
│       ├── accounting.py               # BZS/BZT token balance tracking
│       ├── requirements.txt
│       └── config.yaml
│
├── clients/                            # Client & Contributor Tools
│   │
│   ├── contributor-client/             # BIZRA Contributor Client (BCC)
│   │   ├── Dockerfile
│   │   ├── bccd.py                     # BCC daemon (local job scheduler)
│   │   ├── policy_engine.py            # Policy evaluation for data redaction
│   │   ├── crypto_kms.py               # Local Ed25519 key management
│   │   ├── pack_builder.py             # PoI pack creation (hash + manifest)
│   │   ├── attestation_signer.py       # PoI attestation signing
│   │   ├── uploader.py                 # HTTP client to Node₀ Validation API
│   │   ├── requirements.txt
│   │   └── config.yaml
│   │
│   └── web-ui/                         # Optional web dashboard
│       ├── Dockerfile
│       ├── package.json
│       ├── src/
│       │   ├── index.html
│       │   ├── dashboard.js            # Grafana embed or custom charts
│       │   └── wallet.js               # Token balance & TX history
│       └── .env.local
│
├── infra/                              # Infrastructure & Orchestration
│   │
│   ├── compose/                        # Docker Compose (Local Development & Genesis)
│   │   ├── docker-compose.yml          # Main service stack
│   │   ├── docker-compose.gpu.yml      # GPU override (RTX 4090)
│   │   ├── docker-compose.override.yml # Local development tweaks
│   │   ├── volumes.env                 # Mount paths for datasets, models
│   │   └── .env                        # Environment variables (API keys, secrets)
│   │
│   ├── k8s/                            # Kubernetes Manifests (Production)
│   │   ├── namespace.yaml              # bizra-prod namespace
│   │   ├── configmaps/
│   │   │   ├── genesis-config.yaml     # genesis.built.json as ConfigMap
│   │   │   ├── training-config.yaml    # Training hyperparameters
│   │   │   ├── aggregation-config.yaml # Aggregation policy
│   │   │   └── poi-spec.yaml           # PoI validation rules
│   │   ├── secrets/
│   │   │   ├── node0-keys.yaml         # Ed25519 keys (base64 in prod secret mgmt)
│   │   │   ├── tls-certs.yaml          # TLS certificates for APIs
│   │   │   └── rpc-passwords.yaml      # RPC auth tokens
│   │   ├── deployments/
│   │   │   ├── training.yaml           # Deployment with GPU request
│   │   │   ├── aggregation.yaml        # Deployment (no GPU)
│   │   │   ├── validation-api.yaml     # Deployment
│   │   │   ├── blockgraph-node.yaml    # StatefulSet (persistent data)
│   │   │   └── resource-coordinator.yaml
│   │   ├── services/
│   │   │   ├── training-svc.yaml
│   │   │   ├── aggregation-svc.yaml
│   │   │   ├── blockgraph-rpc.yaml     # LoadBalancer for external RPC
│   │   │   └── validation-api-svc.yaml
│   │   ├── persistentvolumes/
│   │   │   ├── model-storage-pvc.yaml
│   │   │   ├── blockchain-data-pvc.yaml
│   │   │   ├── postgres-data-pvc.yaml
│   │   │   └── redis-data-pvc.yaml
│   │   ├── jobs/
│   │   │   └── genesis-init-job.yaml   # One-time job to initialize genesis & blockchain
│   │   └── hpa.yaml                    # HorizontalPodAutoscaler for training/aggregation
│   │
│   ├── monitoring/                     # Observability Stack
│   │   ├── prometheus/
│   │   │   ├── prometheus.yml          # Scrape config for all services
│   │   │   ├── alert-rules.yaml        # Alert thresholds (GPU OOM, latency, etc.)
│   │   │   └── recording-rules.yaml    # Pre-computed metrics
│   │   ├── grafana/
│   │   │   ├── datasources.yaml        # Prometheus data source
│   │   │   ├── dashboards/
│   │   │   │   ├── training-loss.json
│   │   │   │   ├── aggregation-metrics.json
│   │   │   │   ├── blockchain-health.json
│   │   │   │   └── resource-pool-accounting.json
│   │   │   └── provisioning/
│   │   ├── jaeger/
│   │   │   ├── jaeger-deployment.yaml
│   │   │   └── sampler-config.json
│   │   └── loki/
│   │       └── loki-config.yaml        # Log aggregation
│   │
│   ├── scripts/                        # Operational Scripts
│   │   ├── init-genesis.sh             # Build & verify genesis
│   │   ├── deploy-compose.sh           # Spin up docker-compose stack
│   │   ├── deploy-k8s.sh               # Apply K8s manifests
│   │   ├── health-check.sh             # Validate all services running
│   │   ├── backup-blockchain.sh        # Snapshot chain data
│   │   ├── restore-blockchain.sh       # Restore from backup
│   │   ├── scale-training.sh           # Adjust GPU allocation
│   │   └── incident-response/
│   │       ├── gpu-oom-runbook.md
│   │       ├── latency-spike-runbook.md
│   │       ├── aggregation-failure-runbook.md
│   │       └── consensus-fork-runbook.md
│   │
│   └── terraform/                      # IaC for cloud deployment (optional)
│       ├── main.tf
│       ├── variables.tf
│       └── outputs.tf
│
├── config/                             # Configuration Layer (Layer 3)
│   │
│   ├── node0/                          # Node₀-specific configuration
│   │   ├── genesis.json.example        # Template for genesis.json
│   │   ├── policy.yaml                 # Data scope & resource access policy
│   │   ├── resources.yaml              # Resource allocation ceiling
│   │   ├── networks.yaml               # Peer list & bootstrap nodes
│   │   └── validator-spec.yaml         # Node₀ validator settings (authority, weight)
│   │
│   ├── poi/                            # PoI Configuration
│   │   ├── poi-spec.yaml               # Attestation format & validation rules
│   │   ├── impact-categories.yaml      # Categories (Teaching, Coding, etc.)
│   │   └── reward-multipliers.yaml     # PoI weight to BZS conversion
│   │
│   ├── training/                       # Training Hyperparameters
│   │   ├── base-config.yaml            # LR, batch size, epochs
│   │   ├── moe-config.yaml             # Mixture-of-Experts settings
│   │   └── privacy-config.yaml         # DP epsilon, noise schedule
│   │
│   └── resource-pool/                  # Resource Pool Rules
│       ├── contributor-eligibility.yaml
│       ├── token-minting.yaml          # BZS/BZT issuance rules
│       └── task-priority.yaml
│
├── docs/                               # Documentation
│   │
│   ├── QUICK_START.md                  # "Clone & run in 5 minutes" guide
│   ├── ARCHITECTURE.md                 # (This architecture document)
│   ├── DEPLOYMENT.md                   # Detailed deploy steps (compose + K8s)
│   ├── API_REFERENCE.md                # REST API docs (Aggregation, Validation)
│   ├── CONSENSUS.md                    # BlockGraph consensus deep-dive
│   ├── POI_SPEC.md                     # Proof-of-Impact specification
│   ├── CONTRIBUTOR_GUIDE.md            # How to run BCC & submit attestations
│   ├── OPERATIONALIZATION.md           # SRE guide, runbooks, monitoring
│   ├── SECURITY.md                     # Threat model, key management, TLS setup
│   │
│   └── examples/                       # Example workflows
│       ├── submit_poi_attestation.md
│       ├── federated_learning_cycle.md
│       ├── scale_to_kubernetes.md
│       └── incident_response.md
│
├── tests/                              # Comprehensive Testing
│   │
│   ├── unit/                           # Unit tests (per service)
│   │   ├── test_consensus.rs
│   │   ├── test_poi_validator.py
│   │   ├── test_aggregation_logic.py
│   │   └── test_blockchain_recorder.py
│   │
│   ├── integration/                    # Integration tests (service interactions)
│   │   ├── test_training_to_aggregation.py
│   │   ├── test_aggregation_to_blockchain.py
│   │   ├── test_poi_flow_end_to_end.py
│   │   └── test_resource_pool_accounting.py
│   │
│   ├── chaos/                          # Chaos engineering tests
│   │   ├── test_gpu_oom_recovery.py
│   │   ├── test_network_partition.py
│   │   ├── test_aggregation_byzantine.py
│   │   └── test_consensus_under_load.py
│   │
│   ├── fixtures/                       # Test data
│   │   ├── sample_gradients.npy
│   │   ├── sample_poi_attestations.json
│   │   └── genesis_test.json
│   │
│   └── pytest.ini / cargo.toml         # Test configuration
│
├── benchmarks/                         # Performance benchmarking
│   ├── training_throughput.py          # Tokens/sec, loss convergence
│   ├── aggregation_latency.py          # p50, p95, p99 latencies
│   ├── blockchain_throughput.py        # TPS, finality time
│   └── results/                        # Benchmark outputs (git-ignored)
│
├── .github/                            # GitHub Actions CI/CD
│   ├── workflows/
│   │   ├── build-and-test.yml          # Build, lint, unit tests
│   │   ├── security-scan.yml           # SAST, container scanning
│   │   ├── chaos-tests.yml             # Chaos engineering in CI
│   │   ├── benchmark.yml               # Performance benchmarks
│   │   ├── deploy-staging.yml          # Deploy to staging K8s
│   │   └── deploy-production.yml       # Blue-green deployment to prod
│   └── dependabot.yml                  # Automated dependency updates
│
├── .dockerignore
├── .gitattributes
└── VERSION                             # Semantic versioning (e.g., v0.1.0-alpha)
```

---

## Core Services Architecture

### Layer 1: Infrastructure (Hardware & Containers)
- **Host:** Windows 11 Pro / Linux
- **Container Runtime:** Docker Engine
- **Local Orchestration:** Docker Compose
- **Production Orchestration:** Kubernetes (bizra-prod namespace)

### Layer 2: Service Layer

#### **Training Service (bizra-training)**
- **Language:** Python (PyTorch)
- **Port:** 8080 (HTTP), 6006 (TensorBoard)
- **GPU:** RTX 4090 (4 GPUs in cluster mode)
- **Memory:** 110–120Gi
- **Datastore Mounts:**
  - `/datasets` → Training corpus (DVC-managed)
  - `/models` → Model weights & checkpoints
  - `/metrics` → Prometheus scrape

**Key Functions:**
- Train BIZRA foundation model (Aria 3.9B MoE variant)
- Export metrics for monitoring
- Checkpoint management for fault tolerance
- Gradient export for federation

#### **Aggregation Service (bizra-aggregation)**
- **Language:** Python (Flask/FastAPI)
- **Port:** 5000 (REST API), 5001 (Prometheus)
- **CPU:** 16 cores (production-grade)
- **Memory:** 32Gi

**Key Functions:**
- Receive gradients from contributors
- Byzantine-robust aggregation (median/trimmed-mean)
- Apply differential privacy
- Compute PoI impact scores
- Record events to BlockGraph
- Rate limiting & policy enforcement

**Endpoints:**
```
POST /api/v1/gradients/receive
POST /api/v1/poi/weight
GET  /api/v1/aggregation/status
GET  /metrics (Prometheus)
```

#### **Validation & Attestation API (bizra_validation_api)**
- **Language:** JavaScript/Node.js (Express)
- **Port:** 3006 (HTTPS)
- **Memory:** 8Gi

**Key Functions:**
- Verify PoI attestations (Ed25519 signatures)
- Validate against genesis.built.json
- Ingest contributor metrics
- Trust anchor management

**Endpoints:**
```
POST /api/v1/proof-of-impact/verify
GET  /api/v1/proof-of-impact/status
POST /api/v1/resources/metrics
GET  /api/v1/resources/contributors
```

#### **BlockGraph Node (bizra-blockgraph-node)**
- **Language:** Rust (async)
- **Port:** 30333 (P2P gossip), 9944 (JSON-RPC)
- **Memory:** 16Gi
- **Storage:** 500GB (chain state, pruned)

**Key Functions:**
- Bootstrap BlockGraph DAG
- Validate blocks & PoI references
- Gossip protocol (libp2p)
- JSON-RPC server for clients
- Finality via WQ-refs (PoI-weighted)

#### **Resource Pool Coordinator**
- **Language:** Python (FastAPI)
- **Port:** 5002
- **Memory:** 4Gi

**Key Functions:**
- Track contributor resources (compute, storage, uptime)
- Allocate tasks to contributors
- Manage BZS/BZT token balances
- Enforce resource ceiling per contributor

### Layer 3: Data & Cache
- **PostgreSQL:** Gradient metadata, PoI logs, attestations
- **Redis:** Queues, rate limit state, idempotency
- **Object Storage (S3):** Model checkpoints, gradient snapshots

### Layer 4: Observability
- **Prometheus:** Metrics from all services
- **Grafana:** Dashboards for training loss, latency, TPS, token distribution
- **Jaeger:** Distributed tracing across services
- **Loki:** Log aggregation (optional)

---

## Critical Configuration Files

### 1. **bizra-ledger/genesis.json** (Template)

```json
{
  "chain_id": "bizra-alpha-1",
  "timestamp": "2025-12-11T16:00:00Z",
  "consensus_model": "blockgraph-wq-refs",
  "initial_authority": {
    "node_id": "node0:bizra",
    "public_key": "ed25519:...",
    "initial_weight": 1.0
  },
  "poi_config": {
    "attestation_schema_version": "1.0",
    "impact_categories": [
      { "name": "teaching", "multiplier": 1.5 },
      { "name": "coding", "multiplier": 2.0 },
      { "name": "infrastructure", "multiplier": 2.5 }
    ]
  },
  "token_config": {
    "stable_token": "BZC-Stable",
    "growth_token": "BZT-Growth",
    "initial_supply": "21000000000",
    "inflation_model": "proof-of-impact-driven"
  },
  "upgrade_policy": "governance-vote"
}
```

### 2. **config/node0/policy.yaml**

```yaml
# Node₀ Data Scope & Access Policy

data_scope:
  description: "Node₀ may process training data, attestations, metrics"
  categories:
    - training_data
    - poi_attestations
    - resource_metrics
    - consensus_state

resource_ceilings:
  cpu_cores: 128
  memory_gb: 256
  storage_gb: 3000
  gpu_count: 4
  network_bandwidth_gbps: 10

access_control:
  training_service:
    read: [training_data, model_cache]
    write: [checkpoints, metrics]
  
  aggregation_service:
    read: [poi_attestations, gradients, policies]
    write: [aggregated_gradients, impact_logs, blockchain_events]
  
  blockgraph_node:
    read: [genesis, all_blocks, attestations]
    write: [new_blocks, finality_markers]

rate_limits:
  attestation_ingest: 1000/sec
  gradient_upload: 500/sec
  api_calls_per_contributor: 100/min

trust_boundaries:
  external_inputs: ["attestations", "gradients", "metrics"]
  validation_required: true
  source_verification: "Ed25519-signature"
```

### 3. **config/poi/poi-spec.yaml**

```yaml
# Proof-of-Impact Specification

attestation_format:
  version: "1.0"
  required_fields:
    - attester_id
    - impact_claim
    - genesis_merkle_root
    - timestamp
    - signature_ed25519
  
  impact_claim_schema:
    category: enum [teaching, coding, infrastructure, research, governance]
    description: string (max 1000 chars)
    evidence_hash: sha256
    duration_seconds: integer
    participant_count: integer

validation_rules:
  - signature_must_verify_against_pubkey
  - merkle_root_must_match_current_genesis
  - timestamp_must_be_within_5_minutes
  - evidence_hash_must_be_present
  - impact_category_must_be_recognized

scoring_function: |
  base_score = 1.0
  score = base_score * impact_multiplier[category]
  if participant_count > 1:
    score *= log(participant_count)
  return min(score, 100.0)
```

### 4. **infra/compose/docker-compose.yml** (Excerpt)

```yaml
version: '3.9'

services:
  
  # Training
  training:
    image: bizra-training:latest
    container_name: bizra-training
    ports:
      - "8080:8080"
      - "6006:6006"
    volumes:
      - ./datasets:/datasets
      - ./models:/models
      - ./checkpoints:/checkpoints
    environment:
      PYTORCH_CUDA_ALLOC_CONF: "max_split_size_mb:512"
      METRICS_PORT: 8080
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
    networks:
      - bizra-net
  
  # Aggregation
  aggregation:
    image: bizra-aggregation:latest
    container_name: bizra-aggregation
    ports:
      - "5000:5000"
      - "5001:5001"
    environment:
      FLASK_ENV: production
      POSTGRES_HOST: postgres
      REDIS_HOST: redis
      METRICS_PORT: 5001
    depends_on:
      postgres:
        condition: service_healthy
      redis:
        condition: service_healthy
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:5000/health"]
      interval: 30s
    networks:
      - bizra-net
  
  # BlockGraph Node
  blockgraph:
    image: bizra-blockgraph:latest
    container_name: bizra-blockgraph-node
    ports:
      - "30333:30333"
      - "9944:9944"
    volumes:
      - ./chain-data:/chain-data
      - ./bizra-ledger/genesis.built.json:/genesis.built.json:ro
    environment:
      RUST_LOG: info
      CHAIN_DATA_PATH: /chain-data
    depends_on:
      - postgres
    networks:
      - bizra-net
  
  # Validation API
  validation-api:
    image: bizra-validation-api:latest
    container_name: bizra-validation-api
    ports:
      - "3006:3006"
    environment:
      NODE_ENV: production
      BLOCKGRAPH_RPC: "http://blockgraph:9944"
      GENESIS_PATH: /genesis/genesis.built.json
    volumes:
      - ./bizra-ledger/:/genesis/:ro
    networks:
      - bizra-net
  
  # PostgreSQL
  postgres:
    image: postgres:16-alpine
    container_name: bizra-postgres
    environment:
      POSTGRES_DB: bizra
      POSTGRES_USER: bizra_admin
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    volumes:
      - postgres-data:/var/lib/postgresql/data
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U bizra_admin"]
      interval: 10s
    networks:
      - bizra-net
  
  # Redis
  redis:
    image: redis:7-alpine
    container_name: bizra-redis
    ports:
      - "6379:6379"
    volumes:
      - redis-data:/data
    healthcheck:
      test: ["CMD", "redis-cli", "ping"]
      interval: 10s
    networks:
      - bizra-net
  
  # Prometheus
  prometheus:
    image: prom/prometheus:latest
    container_name: bizra-prometheus
    ports:
      - "9090:9090"
    volumes:
      - ./infra/monitoring/prometheus/prometheus.yml:/etc/prometheus/prometheus.yml:ro
      - prometheus-data:/prometheus
    command:
      - '--config.file=/etc/prometheus/prometheus.yml'
      - '--storage.tsdb.path=/prometheus'
    networks:
      - bizra-net
  
  # Grafana
  grafana:
    image: grafana/grafana:latest
    container_name: bizra-grafana
    ports:
      - "3000:3000"
    environment:
      GF_SECURITY_ADMIN_PASSWORD: ${GRAFANA_ADMIN_PASSWORD}
    volumes:
      - ./infra/monitoring/grafana/datasources.yaml:/etc/grafana/provisioning/datasources/datasources.yaml:ro
      - grafana-data:/var/lib/grafana
    depends_on:
      - prometheus
    networks:
      - bizra-net

volumes:
  postgres-data:
  redis-data:
  prometheus-data:
  grafana-data:

networks:
  bizra-net:
    driver: bridge
```

---

## Deployment Models

### Model A: Docker Compose (Local Genesis)

**Use case:** Development, testing, single-machine Node₀

```bash
# Clone repository
git clone https://github.com/bizra-io/bizra-genesis-node.git
cd bizra-genesis-node

# Setup
cp config/node0/genesis.json.example bizra-ledger/genesis.json
python bizra-ledger/genesis_build.py
python bizra-ledger/genesis_verify.py

# Deploy
docker-compose -f infra/compose/docker-compose.yml \
               -f infra/compose/docker-compose.gpu.yml up -d

# Verify
./infra/scripts/health-check.sh
```

**Stack Size:**
- Training: ~95GB (model + data + memory)
- Aggregation: ~16GB
- BlockGraph: ~16GB
- Supporting: ~8GB
- **Total:** ~135GB RAM, 500GB storage

### Model B: Kubernetes (Production Scale)

**Use case:** Multi-node cluster, geographic distribution, HA

```bash
# Create namespace & secrets
kubectl create namespace bizra-prod
kubectl apply -f infra/k8s/namespace.yaml
kubectl apply -f infra/k8s/secrets/

# Deploy services
kubectl apply -f infra/k8s/configmaps/
kubectl apply -f infra/k8s/deployments/
kubectl apply -f infra/k8s/services/
kubectl apply -f infra/k8s/persistentvolumes/

# Initialize genesis (one-time)
kubectl apply -f infra/k8s/jobs/genesis-init-job.yaml

# Monitor
kubectl logs -f deployment/bizra-training -n bizra-prod
kubectl port-forward svc/prometheus 9090:9090 -n bizra-prod
```

**Scaling:**
- `kubectl scale deployment bizra-training --replicas=8 -n bizra-prod`
- HPA automatically scales based on GPU utilization

---

## Integration & Data Flow

### End-to-End Workflow: PoI Attestation → Aggregation → Block₀

```
┌─────────────────────────────────────────────────────────────────────┐
│ CONTRIBUTOR (edge device)                                           │
│                                                                     │
│  [Local Task]                                                      │
│      │ run, measure impact                                         │
│      ▼                                                              │
│  [BCC: Pack Builder]                                               │
│      │ hash data, create manifest                                  │
│      ▼                                                              │
│  [BCC: Attestation Signer]                                         │
│      │ sign with Ed25519 key                                       │
│      ▼                                                              │
│  [BCC: Uploader]                                                   │
│      │ POST poi_attestation.json → Node₀:3006                      │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ NODE₀: VALIDATION API (port 3006)                                  │
│                                                                     │
│  [Route: POST /api/v1/proof-of-impact/verify]                     │
│      │ parse, schema validate                                      │
│      ▼                                                              │
│  [Validator: Attestation]                                          │
│      │ Ed25519 verify, timestamp check, merkle root match          │
│      ▼                                                              │
│  [Trust Anchor Service]                                            │
│      │ load genesis.built.json, node0 pubkey                       │
│      ▼                                                              │
│  [Blockchain Client]                                               │
│      │ record to /api/blockgraph/poi_log                           │
│      ▼                                                              │
│  Response: { attestation_id, weight, block_height }                │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ NODE₀: AGGREGATION SERVICE (port 5000)                             │
│                                                                     │
│  [Service: Aggregation API]                                        │
│      │ receives poi_weight & attestation                           │
│      ▼                                                              │
│  [Impact Scoring]                                                  │
│      │ compute score from category & participant count             │
│      ▼                                                              │
│  [Ledger Recording]                                                │
│      │ write to Postgres: poi_attestations table                   │
│      ▼                                                              │
│  [Blockchain Recorder]                                             │
│      │ async: record impact event to BlockGraph                    │
│      ▼                                                              │
│  [Policy Engine]                                                   │
│      │ rate limit check, contributor reputation lookup            │
│      ▼                                                              │
│  Response: { aggregation_id, reward_bzt, finality_block }          │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ NODE₀: BLOCKGRAPH NODE (port 9944 RPC)                             │
│                                                                     │
│  [Block Proposal]                                                  │
│      │ node0 proposes block with PoI attestations                  │
│      ▼                                                              │
│  [Consensus: WQ-Refs Finality]                                     │
│      │ weight = sum of PoI scores in block                         │
│      │ finality = gossip of weighted references                    │
│      ▼                                                              │
│  [State Update]                                                    │
│      │ update validator weights, token balances                    │
│      ▼                                                              │
│  [Ledger Persistence]                                              │
│      │ write block to /chain-data/blocks/                          │
│      ▼                                                              │
│  FINALIZED BLOCK₀ with PoI history                                 │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│ MONITORING & VERIFICATION                                          │
│                                                                     │
│  Prometheus scrapes:                                               │
│    - training service metrics                                      │
│    - aggregation latency & throughput                              │
│    - blockgraph finality time, TPS                                 │
│    - contributor attestation count                                 │
│                                                                     │
│  Grafana visualizes:                                               │
│    - Training loss convergence curve                               │
│    - Real-time PoI weight distribution                             │
│    - Block finality histogram                                      │
│    - Token balance evolution                                       │
│                                                                     │
│  Jaeger traces:                                                    │
│    - Latency from attestation submission to finality               │
│    - Span breakdown by service                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Operational Readiness

### Health Check Procedures

```bash
# Service connectivity
curl http://localhost:8080/health        # Training
curl http://localhost:5000/health        # Aggregation
curl http://localhost:3006/health        # Validation API
curl http://localhost:9944/health        # BlockGraph RPC

# Database connectivity
psql -h localhost -U bizra_admin -d bizra -c "SELECT 1;"
redis-cli ping

# Blockchain state
curl -X POST http://localhost:9944 \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"chain_getBlock","params":["0x..."],"id":1}'

# Metrics availability
curl http://localhost:9090/api/v1/query?query=up
```

### Monitoring Dashboards

1. **Training Dashboard:** Loss curve, throughput (tokens/sec), GPU utilization
2. **Aggregation Dashboard:** Request latency (p50, p95, p99), Byzantine robustness score
3. **Blockchain Dashboard:** Block finality time, TPS, validator set changes
4. **Resource Pool Dashboard:** Contributor count, BZS/BZT distribution, task queue depth

### Incident Response

- **GPU OOM:** Reduce batch size in config/training/base-config.yaml, restart training service
- **Aggregation Latency Spike:** Check Byzantine robustness; may need more rounds; monitor network
- **Consensus Fork:** Check validator weights, trigger manual resync of BlockGraph node
- **Attestation Backlog:** Scale aggregation service horizontally or increase rate limits

---

## Next Steps

1. **Freeze this repository structure** as the canonical layout for bizra-genesis-node
2. **Implement core services** in order of dependency:
   - BlockGraph node (Rust)
   - Training service (Python/PyTorch)
   - Aggregation service (Python/FastAPI)
   - Validation API (Node.js/Express)
3. **Wire integration tests** to validate data flow (attestation → aggregation → blockchain)
4. **Deploy to local docker-compose** and achieve health-check green
5. **Benchmark performance** (TPS, latency, throughput) against target specs
6. **Scale to Kubernetes** and measure HA / auto-scaling behavior
7. **Open-source release** with full documentation & contributor onboarding

---

**Authority:** MoMo - First Architect  
**Date:** 2025-12-11  
**Version:** v1.0-alpha  
**Status:** Ready for Implementation