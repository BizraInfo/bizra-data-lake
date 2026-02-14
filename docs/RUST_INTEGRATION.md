# Rust Integration Guide

> **bizra-omega** — High-performance Rust backend for BIZRA sovereign operations

## Overview

The Rust integration provides 10-100x performance improvement for:
- Cryptographic operations (Ed25519 signing, BLAKE3 hashing)
- Constitutional gate validation
- PCI envelope creation and verification
- Federation gossip and consensus

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Python Sovereign Runtime                      │
│  ├─ ProactiveSovereignEntity                                    │
│  ├─ OpportunityPipeline                                         │
│  └─ Constitutional Filters                                       │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
              ▼                         ▼
┌─────────────────────────┐   ┌─────────────────────────┐
│    PyO3 Bindings        │   │    REST API Client      │
│    (In-Process)         │   │    (Inter-Process)      │
│    10-100x faster       │   │    18 endpoints         │
└─────────────────────────┘   └─────────────────────────┘
              │                         │
              └────────────┬────────────┘
                           │
              ┌────────────┴────────────┐
              │     bizra-omega         │
              │     (Rust Binary)       │
              │     Port 3001           │
              └─────────────────────────┘
```

## Quick Start

### 1. Build Rust Binaries

```bash
cd bizra-omega

# Build release binaries
cargo build --release

# Build PyO3 wheel
cd bizra-python
maturin build --release

# Install Python wheel
pip install target/wheels/bizra-*.whl
```

### 2. Start Rust API

```bash
# Using cargo
cargo run --release --bin bizra-api -- --port 3001

# Using binary directly
./target/release/bizra-api --port 3001 --host 0.0.0.0
```

### 3. Use in Python

```python
from core.sovereign import (
    RustLifecycleManager,
    create_rust_lifecycle,
)

# Create lifecycle manager
rust = await create_rust_lifecycle(
    api_port=3001,
    use_pyo3=True,
)

# Check availability
print(f"PyO3: {rust.pyo3_available}")
print(f"API: {rust.api_healthy}")
```

## PyO3 Bindings

Direct in-process calls for maximum performance.

### Cryptographic Operations

```python
# Ihsān threshold check (100x faster)
valid = rust.pyo3_check_ihsan(0.97)

# SNR threshold check (100x faster)
valid = rust.pyo3_check_snr(0.90)

# Domain-separated digest (20x faster)
digest = rust.pyo3_domain_digest(b"message content")

# Ed25519 signing (50x faster)
signature, public_key = rust.pyo3_sign(b"message")
```

### Direct Module Import

```python
# If wheel is installed
import bizra

identity = bizra.NodeIdentity()
print(f"Node ID: {identity.node_id}")

constitution = bizra.Constitution()
print(f"Ihsān: {constitution.ihsan_threshold}")

# Sign a message
signature = identity.sign(b"hello world")
```

## REST API Endpoints

When PyO3 is unavailable, use the REST API.

### Health & Status

```python
# Health check
health = await rust._api_client.health_check()
print(f"Status: {health.status.name}")

# Full status
status = await rust._api_client.get_status()
print(f"Uptime: {status['uptime_seconds']}s")
```

### Gate Validation

```python
# Check content through gates
result = await rust.api_check_gates(
    content=b"content to validate",
    snr_score=0.92,
    ihsan_score=0.97,
)
print(f"Passed: {result['passed']}")
```

### Inference

```python
# Generate via Rust tier selection
response = await rust.api_inference(
    prompt="Explain quantum computing",
    tier="local",  # edge, local, or pool
)
print(f"Response: {response['text']}")
```

### Federation

```python
# Get P2P status
fed_status = await rust.api_federation_status()
print(f"Connected: {fed_status['connected']}")
print(f"Peers: {fed_status['peers']}")
```

## API Endpoint Reference

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/v1/health` | GET | Health check |
| `/api/v1/status` | GET | Node status |
| `/api/v1/metrics` | GET | Prometheus metrics |
| `/api/v1/identity/generate` | POST | Generate identity |
| `/api/v1/identity/sign` | POST | Sign message |
| `/api/v1/identity/verify` | POST | Verify signature |
| `/api/v1/pci/envelope/create` | POST | Create PCI envelope |
| `/api/v1/pci/envelope/verify` | POST | Verify envelope |
| `/api/v1/pci/gates/check` | POST | Check gates |
| `/api/v1/inference/generate` | POST | LLM inference |
| `/api/v1/inference/models` | GET | List models |
| `/api/v1/inference/tier` | POST | Select tier |
| `/api/v1/federation/status` | GET | Federation status |
| `/api/v1/federation/peers` | GET | List peers |
| `/api/v1/federation/propose` | POST | Propose pattern |
| `/api/v1/constitution` | GET | Get constitution |
| `/api/v1/constitution/check` | POST | Check compliance |
| `/api/v1/ws` | GET | WebSocket upgrade |

## Pipeline Integration

Add Rust acceleration to the Opportunity Pipeline.

```python
from core.sovereign import (
    OpportunityPipeline,
    create_rust_gate_filter,
    create_rust_lifecycle,
)

# Create Rust lifecycle
rust = await create_rust_lifecycle()

# Create pipeline
pipeline = OpportunityPipeline()

# Add Rust-accelerated filter
rust_filter = create_rust_gate_filter(rust)
pipeline._filters.append(rust_filter)

await pipeline.start()
```

## Process Management

The `RustProcessManager` can start/stop the Rust API.

```python
from core.sovereign import RustProcessManager

manager = RustProcessManager(
    api_port=3001,
    gossip_port=7946,
)

# Start Rust API
started = await manager.start(wait_for_health=True)

# Check status
print(f"Running: {manager.is_running()}")
print(f"Uptime: {manager.uptime()}s")

# Stop gracefully
manager.stop()
```

## Docker Deployment

```bash
# Build image
cd bizra-omega
docker build -t bizra-omega .

# Run with docker-compose
docker-compose up -d

# GPU support
docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d
```

### Docker Compose Services

| Service | Port | Purpose |
|---------|------|---------|
| bizra-api | 3001 | REST API |
| bizra-api | 7946/udp | Gossip P2P |
| ollama | 11434 | LLM backend |

## Performance Benchmarks

| Operation | Python | Rust | Speedup |
|-----------|--------|------|---------|
| Ed25519 sign | 570/sec | 57K/sec | 100x |
| Ed25519 verify | 280/sec | 28K/sec | 100x |
| BLAKE3 hash | 290K/sec | 5.8M/sec | 20x |
| PCI envelope | 940/sec | 47K/sec | 50x |
| Gate chain | 34K/sec | 1.7M/sec | 50x |
| **Combined** | ~400K/sec | **41M/sec** | **100x** |

## Graceful Fallback

When Rust is unavailable, the system falls back to Python.

```python
from core.sovereign import RustLifecycleManager

rust = RustLifecycleManager(use_pyo3=True)

# These work with or without Rust
valid = rust.pyo3_check_ihsan(0.97)  # Uses Python if no PyO3
digest = rust.pyo3_domain_digest(b"msg")  # Uses hashlib fallback

# Check what's available
stats = rust.stats()
print(f"PyO3: {stats['pyo3_available']}")
print(f"API: {stats['api_healthy']}")
```

## Crate Structure

```
bizra-omega/                    # 14-crate Rust workspace
├── bizra-core/                 # Identity, PCI, Constitution, FATE gates
├── bizra-inference/            # LLM gateway, tier selection
├── bizra-federation/           # Gossip, BFT consensus, signed messages
├── bizra-autopoiesis/          # Pattern memory, preference tracking
├── bizra-api/                  # Axum REST server (health, metrics, query)
├── bizra-python/               # PyO3 bindings (GateChain, InferenceGateway)
├── bizra-installer/            # CLI installer + model cache
├── bizra-cli/                  # Terminal UI dashboard (ratatui)
├── bizra-telescript/           # Mobile agent scripts
├── bizra-resourcepool/         # Compute allocation + genesis ceremony
├── bizra-hunter/               # Anomaly detection + rent-seeking identification
├── bizra-proofspace/           # Block validation + RFC 8785 canonicalization
├── bizra-tests/                # Integration tests + benchmarks
└── Cargo.toml                  # Workspace root (ed25519-dalek, tokio, blake3)
```

## Troubleshooting

### PyO3 Import Fails

```bash
# Rebuild wheel
cd bizra-omega/bizra-python
cargo clean
maturin build --release
pip install --force-reinstall target/wheels/bizra-*.whl
```

### API Connection Refused

```bash
# Check if running
curl http://localhost:3001/api/v1/health

# Start manually
./target/release/bizra-api --port 3001
```

### Cargo Build Fails

```bash
# Update Rust
rustup update stable

# Clean and rebuild
cargo clean
cargo build --release
```

## Testing

```bash
# Rust tests
cd bizra-omega
cargo test

# Python integration tests
pytest tests/core/sovereign/test_rust_lifecycle.py -v
```

---

*Standing on Giants: Ed25519 (Bernstein), BLAKE3 (O'Connor), Axum (Tokio Team)*
