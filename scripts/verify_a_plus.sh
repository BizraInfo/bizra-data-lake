#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="$ROOT/.venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="$(command -v python3 || true)"
fi

fail=0

check() {
  local label="$1" file="$2" pattern="$3"
  if [[ ! -f "$file" ]]; then
    echo "[FAIL] $label (missing: $file)"
    fail=1
    return
  fi
  if grep -n "$pattern" "$file" >/dev/null 2>&1; then
    echo "[PASS] $label ($file)"
  else
    echo "[FAIL] $label ($file)"
    fail=1
  fi
}

echo "=== A+ Implementation Verification ==="
echo "[1/3] Checking implementation markers..."
check "SNREnforcer class" "$ROOT/bizra_kernel/snr_enforcer.py" "class SNREnforcer"
check "SNR thresholds" "$ROOT/bizra_kernel/snr_enforcer.py" "class SNRThresholds"
check "Canonical JSON" "$ROOT/core/pci/crypto.py" "def canonical_json"
check "Domain-separated digest" "$ROOT/core/pci/crypto.py" "def domain_separated_digest"
check "Ed25519 signing" "$ROOT/core/pci/crypto.py" "def sign_message"
check "Constitutional Gate" "$ROOT/core/sovereign/integration.py" "class ConstitutionalGate"
check "Execution tiers" "$ROOT/core/sovereign/integration.py" "class ExecutionTier"
check "Receipt writer" "$ROOT/core/main.py" "def _write_receipt"
check "Integrity hash" "$ROOT/core/main.py" "integrity_hash"
check "Synapse TLS default" "$ROOT/core/synapse.py" "rediss://"
check "Node0 identity" "$ROOT/bizra_kernel/node0_identity.py" "class Node0Identity"
check "Hardware fingerprint" "$ROOT/bizra_kernel/hardware_fingerprint.py" "def generate_fingerprint"
check "Genesis sync" "$ROOT/bizra_kernel/genesis_sync.py" "def sync_genesis"

if [[ $fail -ne 0 ]]; then
  echo "Marker check failed."
  exit 1
fi

echo "[2/3] Running tests..."
if [[ -z "$PY" ]]; then
  echo "[FAIL] python not found"
  exit 1
fi

"$PY" - <<'PY'
import importlib, sys
missing = []
for m in ("pytest", "pytest_asyncio", "psutil", "cryptography"):
    try:
        importlib.import_module(m)
    except Exception:
        missing.append(m)
if missing:
    print("Missing modules: " + ", ".join(missing))
    sys.exit(2)
PY

if [[ ${?} -eq 2 ]]; then
  echo "[SKIP] Missing pytest dependencies. Install with:"
  echo "  $ROOT/.venv/bin/pip install pytest pytest-asyncio psutil"
  exit 2
fi

set +e
"$PY" -m pytest -q \
  tests/test_snr_enforcer.py \
  tests/test_kernel_receipt_integrity.py \
  tests/test_synapse_security.py \
  tests/test_node0_sovereignty.py
status=$?
set -e

if [[ $status -ne 0 ]]; then
  echo "[FAIL] Tests failed"
  exit $status
fi

echo "[3/3] Hashes (sha256)"
sha256sum \
  "$ROOT/core/snr.py" \
  "$ROOT/bizra_kernel/snr_enforcer.py" \
  "$ROOT/core/pci/crypto.py" \
  "$ROOT/core/sovereign/integration.py" \
  "$ROOT/core/main.py"

echo "=== Summary ==="
echo "STATUS: ALL CHECKS PASSED"
