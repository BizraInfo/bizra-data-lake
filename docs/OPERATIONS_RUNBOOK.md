# BIZRA Operations Runbook

Last updated: 2026-02-14

This runbook is the operator-focused guide for starting, validating, and troubleshooting BIZRA services.

## 1. Prerequisites

- Python 3.11+ with project dependencies installed
- Optional Rust toolchain for Omega services (`rustup`, stable)
- Optional local inference backend (LM Studio, Ollama, or compatible endpoint)

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
4. If validation fails, treat as tamper/corruption incident and stop startup until resolved.

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
