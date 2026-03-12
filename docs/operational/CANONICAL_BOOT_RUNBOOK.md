# Node0 Canonical Boot — Operator Runbook

**Version:** 1.0  
**Truth Label:** `[ENFORCEMENT: PROVEN]`  
**Last Updated:** 2026-03-12

Standing on Giants: Deming (PDCA, 1950) — standardize what works before optimizing.

---

## Prerequisites

- Python 3.11+ with `.venv-linux` activated
- `sovereign_state/` directory exists (auto-created on boot)
- Ed25519 signer key available (genesis ceremony generates one)

## Normal Boot Sequence

```bash
# 1. Activate environment
source .venv-linux/bin/activate

# 2. Verify prerequisites
python -c "from core.node0.heartbeat import Node0Heartbeat; print('OK')"

# 3. Boot via API (production path)
# The API lifespan boots Node0 automatically when canonical mode is enabled.
# Set BIZRA_CANONICAL_MODE=1 in environment.

# 4. Verify health
curl -s http://localhost:8000/v1/health | python -m json.tool
# Check: "booted": true, "identity_mode": "genesis_ed25519"
```

## Failure Scenarios

### F1: Boot Fails — Missing Ed25519 Signer

**Symptom:** `RuntimeError: Node0 must have Ed25519 identity in canonical mode`  
**Cause:** `signer_public_key_hex` not passed from organism/runtime  
**Fix:**
```bash
# Check that runtime_core.py has genesis identity:
grep -n '_init_canonical_organism_stack' core/sovereign/runtime_core.py
# Ensure signer_public_key_hex is propagated through organism → Node0
```

### F2: API Returns 503 — Fail-Closed

**Symptom:** POST /v1/plan returns `503 Service Unavailable`  
**Cause:** Canonical mode enabled but no mission authority path  
**This is correct behaviour.** The system is fail-closed by design.  
**Fix:** Ensure the full canonical stack is booted:
```bash
# Check canonical mode detection
python -c "
from core.sovereign.api import _runtime_canonical_mode_enabled
# Returns True if runtime has _canonical_mode=True or status reports canonical
"
```

### F3: GoT Bridge — RuntimeError on Signer

**Symptom:** `RuntimeError: GoT bridge: Ed25519Signer unavailable in canonical mode`  
**Cause:** `canonical_mode=True` but `ed25519` dependency missing  
**Fix:**
```bash
pip install ed25519  # or: pip install PyNaCl
# Verify:
python -c "from core.proof_engine.receipt import Ed25519Signer; Ed25519Signer.generate(); print('OK')"
```

### F4: Hash Chain Broken

**Symptom:** `chain_hash` doesn't link to `prev_chain_hash`  
**Cause:** Node0 restarted without persisting chain state  
**Fix:** Check `sovereign_state/` for state files. Node0 starts a fresh chain on each boot (by design — chain is per-session, not persistent across restarts in current implementation).

### F5: Exception Audit Fails in CI

**Symptom:** SEC-003b gate reports count > 157  
**Cause:** New code added broad `except Exception` in sovereign surfaces  
**Fix:** Replace broad catch with specific exception type:
```python
# Bad (fails SEC-003b):
except Exception:
    logger.error("something broke")

# Good:
except (ValueError, KeyError, ConnectionError) as exc:
    logger.error("Specific failure: %s", exc)
```

## Health Check Interpretation

| Field | Expected | Action if Wrong |
|-------|----------|-----------------|
| `booted` | `true` | Check boot sequence, signer key |
| `identity_mode` | `genesis_ed25519` | Check organism stack wiring |
| `chain_hash` | 64-char hex | Check breathe() is running |
| `avg_ihsan` | ≥ 0.85 | Review mission quality |
| `reflex_compilation_status.enabled` | `false` (default) | Intentional — see E5 |
| `subsystems.helix3` | `true` | Check Helix3Scheduler import |

## Validation Commands

```bash
# Run canonical test suite
python -m pytest tests/core/node0/ tests/integration/test_plan_endpoint.py -x -q

# Run live canonical validation (real objects, no mocks)
python scripts/ops/canonical_empirical_validation.py --live

# Run CPU baseline benchmarks
python scripts/ops/canonical_cpu_baseline.py

# Check exception audit
python scripts/ci_exception_audit.py --scan-dirs core/sovereign core/node0 core/reasoning --baseline 157

# Check docs truth labels
python scripts/ci_docs_truth_gate.py
```

## Escalation

If none of the above resolves the issue:
1. Collect `health()` output
2. Collect last 5 breath receipts from `_breath_history`
3. Check `sovereign_state/node0_lifecycle.json`
4. Open issue with `[Node0-Boot]` tag
