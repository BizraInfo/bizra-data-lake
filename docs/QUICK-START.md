# BIZRA Quick Start

Last updated: 2026-02-14

Get from clone to running sovereign runtime in under 10 minutes.

---

## 1. Prerequisites

| Dependency | Version | Required |
|------------|---------|----------|
| Python | 3.11+ | Yes |
| pip | Latest | Yes |
| Rust toolchain | Stable (1.88+) | Optional (for Omega workspace) |
| LLM backend | LM Studio, Ollama, or OpenAI-compatible | Optional (for live inference) |
| Docker | 20.10+ | Optional (for containerized deployment) |

---

## 2. Clone and Install

```bash
git clone https://github.com/BizraInfo/bizra-data-lake.git
cd bizra-data-lake

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install core + dev dependencies
pip install -e ".[dev]"

# Copy environment template
cp .env.example .env
```

Edit `.env` with your LLM backend settings:

```bash
LM_STUDIO_API_KEY=your_key
LM_STUDIO_BASE_URL=http://localhost:1234
# Or for Ollama:
# OLLAMA_HOST=http://localhost:11434
```

---

## 3. Run Tests (Verify Installation)

```bash
# Fast local gate (no GPU, no network, no Ollama required)
pytest tests/core/ -q --tb=short -m "not requires_ollama and not requires_gpu and not slow and not requires_network"
```

Expected: 1,600+ tests pass in under 5 minutes.

---

## 4. Start the Sovereign Runtime

```bash
# Option A: Full launcher (starts all services including Desktop Bridge)
python -m core.sovereign.launch

# Option B: Sovereign runtime only
python -m core.sovereign

# Option C: API server only
python -m core.sovereign.api --host 127.0.0.1 --port 8080
```

### Verify Health

```bash
curl -s http://127.0.0.1:8080/v1/health | python -m json.tool
curl -s http://127.0.0.1:8080/v1/metrics
```

---

## 5. Build Rust Workspace (Optional)

For high-performance features (PyO3 bindings, native crypto, CLI dashboard):

```bash
cd bizra-omega
cargo build --release
cargo test --workspace

# Build PyO3 Python bindings
cd bizra-python
pip install maturin
maturin develop --release
```

---

## 6. Data Pipeline (Optional)

If you want to use the data ingestion and knowledge pipeline:

```bash
# Layer 1: Build document corpus
python corpus_manager.py

# Layer 2: Generate vector embeddings
python vector_engine.py

# Layer 3: LLM extraction
python langextract_engine.py

# Layer 4: SNR validation
python arte_engine.py
```

Files flow through: `00_INTAKE/` -> `01_RAW/` -> `02_PROCESSED/` -> `03_INDEXED/` -> `04_GOLD/`

Duplicates are quarantined to `99_QUARANTINE/` via SHA-256 detection.

---

## 7. Desktop Bridge (Optional)

Start the TCP bridge for AHK hotkey integration:

```bash
export BIZRA_BRIDGE_TOKEN=your_token_here
export BIZRA_RECEIPT_PRIVATE_KEY_HEX=your_64_hex_key
python -m core.bridges.desktop_bridge
```

Bridge listens on `127.0.0.1:9742` (localhost only). See [Desktop Bridge docs](DESKTOP_BRIDGE.md).

---

## 8. What to Read Next

| Your Role | Start Here |
|-----------|------------|
| New engineer | [Architecture Blueprint](ARCHITECTURE_BLUEPRINT_v2.3.0.md) |
| Operator / SRE | [Operations Runbook](OPERATIONS_RUNBOOK.md) |
| Security reviewer | [DDAGI Constitution](DDAGI_CONSTITUTION_v1.1.0-FINAL.md) |
| Contributor | [Contributing Guide](../CONTRIBUTING.md), [Testing Guide](TESTING.md) |

Full documentation portal: [docs/README.md](README.md)

---

## Troubleshooting

### Import errors after install

```bash
# Ensure you're in the venv
which python  # Should show .venv/bin/python

# Reinstall
pip install -e ".[dev]"
```

### Rust build fails

```bash
rustup update stable
cd bizra-omega && cargo clean && cargo build --release
```

### Tests fail with missing model

Tests that require a live LLM are marked with `@pytest.mark.requires_ollama`. The default test run excludes them. If you want to run them:

```bash
# Start Ollama first
ollama serve

# Then run with the marker included
pytest tests/ -m "requires_ollama"
```

---

*Standing on Giants: Shannon, Lamport, Besta, Vaswani, Anthropic*
