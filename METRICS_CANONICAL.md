# BIZRA Metrics — Canonical Source of Truth

**Last updated:** 26 March 2026
**Rule:** If any document disagrees with this file on a number, this file wins.

---

## Codebase

| Metric | Value | Evidence | Date verified |
|---|---|---|---|
| Rust crates | 26 | bizra-omega workspace | March 2026 |
| Rust tests | 1,446 | cargo test output | March 2026 |
| Python tests | 11,216 | pytest output | March 2026 |
| Total tests | 12,662 | Sum of above | March 2026 |
| Git commits | 577 | git log --oneline count | March 2026 |
| Enforceable Spine | v1.1 (6 amendments) | Spine document | March 2026 |
| Phase | 87-88 (v0.88.1) | Git tags | March 2026 |

## Performance (EVIDENCE — not independently verified)

| Metric | Value | Conditions | Status |
|---|---|---|---|
| IHSAN check latency | 90.4 ns | RTX 4090, n=10,000 | Evidence |
| BLAKE3 hash latency | 349 ns | RTX 4090, n=10,000 | Evidence |
| Ed25519 sign latency | 396 ns | RTX 4090, n=10,000 | Evidence |
| Total membrane tax | 3.02 us | RTX 4090, n=10,000 | Evidence |
| Throughput | 237,199 req/s | RTX 4090 | Evidence |
| FAISS query | 5 ms | 84,795 vectors | Verified |
| Reflex speedup | 126.7x | 153.27ms → 1.21ms | Verified (N=1) |

## Stability

| Metric | Value | Status |
|---|---|---|
| Heartbeat longest run | 6.5 hours | Verified |
| Heartbeat errors | 0 | Verified |
| Heartbeat log lines | 9,321 | Verified |
| Constitutional violations | 0 / 12,662 tests | Verified |

## Knowledge

| Metric | Value | Status |
|---|---|---|
| FAISS vectors | 84,795 | Verified |
| Knowledge graph nodes | 577 | Verified |
| Knowledge graph edges | 104,957 | Verified |
| Connection types | 5 | Verified |

## Economic (Block 0)

| Metric | Value | Status |
|---|---|---|
| Block 0 hash | 350d642099bde68b | Verified |
| Block 0 receipts | 10 | Verified |
| Block 0 SEED | 1.1M | Verified |
| Agents in Block 0 | 12 (7 PAT + 5 SAT) | Verified |

## Hardware (NODE0)

| Component | Spec |
|---|---|
| Machine | MSI Titan 18 HX |
| CPU | i9-14900HX |
| RAM | 128GB DDR5 |
| GPU | RTX 4090 (16GB) |
| Storage | 3.8TB RAID 0 |
| OS | Windows 11 |
| Rust | 1.94.0 |
| Python | 3.13.5 |
| Node.js | 24.5.0 |
| Git | 2.53.0 |

## What is NOT measured yet

- Multi-node latency (no federation test run)
- Reverse scaling at N>1 (architectural projection only)
- Morning brief delivery time
- Daily mission count in production use
- Reflex hit ratio over time

---

*Numbers in this file are either VERIFIED (tested, reproducible) or EVIDENCE (measured once, needs independent verification). No number is CLAIMED without measurement.*
