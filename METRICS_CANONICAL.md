# BIZRA Metrics — Canonical Source of Truth

**Last updated:** 26 March 2026
**Rule:** If any document disagrees with this file on a number, this file wins.
**Discrepancies resolved in:** BIZRA_CANONICAL.md

---

## Codebase (VERIFIED)

| Metric | Value | Date verified |
|---|---|---|
| Rust crates | 26 | March 2026 |
| Rust tests | 1,446 | March 2026 |
| Python tests | 11,216 | March 2026 |
| Total tests | 12,662 | March 2026 |
| Git commits | 577+ | March 2026 |
| Phase | 87-88 (v0.88.1) | March 2026 |

## Performance (EVIDENCE — Gold Standard run, n=10,000)

| Operation | Mean | Std Dev | P99 | Complexity |
|---|---|---|---|---|
| IHSAN check | 90.4 ns | 12.3 ns | 145 ns | O(1) |
| BLAKE3 hash | 349 ns | 28.7 ns | 412 ns | O(1) |
| Ed25519 sign | 396 ns | 35.2 ns | 478 ns | O(1) |
| Total membrane | 3.02 us | 0.89 us | 10.14 us | O(1) |
| Throughput | 237,199 req/s | — | — | O(1) |

NOTE: Submission v2 reported Ed25519 at 630 ns (included key gen overhead). Canonical value is 396 ns (sign-only, pre-loaded key). Both are honest measurements of different operations.

## Stability (VERIFIED)

| Metric | Value |
|---|---|
| Heartbeat longest run | 6.5 hours |
| Heartbeat errors | 0 |
| Log lines | 9,321 |
| Constitutional violations | 0 / 12,662 tests |

## Knowledge (VERIFIED)

| Metric | Value |
|---|---|
| FAISS vectors | 84,795 |
| FAISS query | 5 ms |
| Knowledge graph nodes | 577 |
| Knowledge graph edges | 104,957 |

## Self-Improvement (VERIFIED, N=1 only)

| Metric | Value |
|---|---|
| Deliberative latency | 153.27 ms |
| Reflex latency | 1.21 ms |
| Speedup | 126.7x |
| Quality degradation | Zero (Ihsan 0.8662 both) |

## Economic (VERIFIED — Block 0)

| Metric | Value |
|---|---|
| Block 0 hash | 350d642099bde68b |
| Block 0 receipts | 10 (BLAKE3-chained) |
| Block 0 SEED | 1.1M |
| Agents at genesis | 12 (7 PAT + 5 SAT) |

## Constitutional Invariants

| Invariant | Canonical value | Note |
|---|---|---|
| IHSAN_FLOOR | >= 0.95 | Rust newtype, compile-time |
| ZANN_ZERO | No unverified claims | Membrane check |
| RIBA_ZERO | ArithmeticIntegrity AND EconomicPolicyIntegrity | Compound |
| GINI_CEILING | <= 0.35 | CANONICAL. Some older docs say 0.45 — WRONG. |

## Hardware (NODE0)

MSI Titan 18 HX | i9-14900HX | 128GB DDR5 | RTX 4090 | 3.8TB RAID 0
Win11 | Rust 1.94.0 | Python 3.13 | Node 24.5.0 | Git 2.53.0

## NOT YET MEASURED

- Multi-node latency
- Reverse scaling at N>1
- Morning brief delivery time
- Daily mission count (production)
- Reflex hit ratio over time
- 24-hour continuous heartbeat (6.5h proven)
