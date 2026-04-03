# BIZRA A+ Evidence Pack — Node0 MSSC (Repo: BIZRA-Dual-Agentic-system--main)

**Generated:** 2026-02-04T02:02:50Z (UTC)

## 1) Summary (Evidence-backed)
- **Grade:** **A (provisional)** — 29 passed, 4 skipped, 0 failed (warnings only). [EVIDENCE]
- **Scope:** Minimal Solvable Special Case (MSSC) for Node0: **SNR enforcement**, **PCI crypto**, **Constitutional Gate**, **Receipt Integrity**, **Synapse TLS default**. [EVIDENCE]
- **Verification:** `./scripts/verify_a_plus.sh` (see §5). [EVIDENCE]

> Note: "A+" title reflects the intended standard. Current grade is **A (provisional)** because 4 TLS/integration checks were skipped (env-gated) and warnings remain. [EVIDENCE]

---

## 2) Evidence Map (Real Paths + Line Anchors)
| Component | Evidence | File (line anchor) |
|---|---|---|
| **SNR Enforcement** | `SNREnforcer` + thresholds | `bizra_kernel/snr_enforcer.py:222` [EVIDENCE] |
| **SNR Budgeting** | SNR budget + estimation | `core/snr.py` [EVIDENCE] |
| **PCI Canonical JSON** | RFC 8785 canonicalization | `core/pci/crypto.py:43` [EVIDENCE] |
| **PCI Domain Separation** | `domain_separated_digest` | `core/pci/crypto.py:85` [EVIDENCE] |
| **PCI Signing** | `sign_message` (Ed25519) | `core/pci/crypto.py:213` [EVIDENCE] |
| **Constitutional Gate** | `ExecutionTier`, `ConstitutionalGate` | `core/sovereign/integration.py:59,135` [EVIDENCE] |
| **Receipt Integrity** | `_write_receipt` + `integrity_hash` | `core/main.py:212,243` [EVIDENCE] |
| **Synapse TLS Default** | `SYNAPSE_URL` uses `rediss://` | `core/synapse.py:83` [EVIDENCE] |

---

## 3) Test Results (Executed)
**Command:**
```
/mnt/c/BIZRA-Dual-Agentic-system--main/.venv/bin/python -m pytest -q \
  tests/test_snr_enforcer.py \
  tests/test_kernel_receipt_integrity.py \
  tests/test_synapse_security.py
```

**Outcome:** **29 passed, 4 skipped, 0 failed** (warnings only). [EVIDENCE]

**Skipped (expected):** TLS/infra checks gated by `BIZRA_TLS_TESTS=1` and integration fixtures. [EVIDENCE]

**Warnings observed:**
- `datetime.utcnow()` deprecation in `bizra_kernel/snr_tracker.py`. [EVIDENCE]
- FastAPI `on_event` deprecation in `core/main.py`. [EVIDENCE]

---

## 4) Gaps / Notes (Honest)
- `tests/test_node0_sovereignty.py` references `bizra_kernel.node0_identity`, which is **missing** in this repo (only `.pyc` remains). [EVIDENCE]
- TLS integration tests require `BIZRA_TLS_TESTS=1` + cert files. [EVIDENCE]

---

## 5) How to Verify (One Command)
```
./scripts/verify_a_plus.sh
```

The script:
- Validates the **key MSSC markers** in source.
- Runs the **same pytest subset** as above.
- Emits a pass/fail summary.

---

## 6) Artifact Hashes (Integrity)
```
core/snr.py                         e50e6191e49721d3e21915a5368ec233b842aaa7b96e6a141a082e63fadf4c1f
bizra_kernel/snr_enforcer.py        6022cf2fd02e9dead70bbb61397db76941a750bf75d13baa8556810e52be0bc7
core/pci/crypto.py                  61e60523ea5f531dc708c2ce18a940ee7577c362544b6a021ea7a6c390124f86
core/sovereign/integration.py       03398d595512e5231bfe7060eb546a048c70b7c6a3953302b47c7152fd89f315
core/main.py                        05ab4618610dc2ff02715ca4829571cb8bcb16901fd553c7f56b045867f42143
```

---

## 7) Standing on Giants (Explicit)
- **Shannon** → SNR thresholds & enforcement. [EVIDENCE]
- **Lamport** → Receipt integrity & replay resistance patterns. [EVIDENCE]
- **Kocher** → Timing-safety emphasis (crypto + receipt integrity). [INFERENCE]
- **Anthropic** → Constitutional gate & Ihsān thresholding. [EVIDENCE]

---

## 8) Repro Checklist (Optional)
- `pip install pytest pytest-asyncio psutil` (already installed in `.venv`). [EVIDENCE]
- Set `BIZRA_TLS_TESTS=1` to include TLS integration checks. [EVIDENCE]
- Run `./scripts/verify_a_plus.sh`. [EVIDENCE]
