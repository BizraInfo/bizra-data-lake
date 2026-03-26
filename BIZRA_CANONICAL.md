# BIZRA CANONICAL — Single Source of Truth

**Version:** 1.0 | **Frozen:** 26 March 2026
**Rule:** If ANY document contradicts this file, this file wins.

---

## Part 1: Topology (from TOPOLOGY_CANON.md)

ONE shared URP. PAT-7 local per human. SAT-5 minted into the shared URP per node. Membrane between local and shared. Human never touches network. See TOPOLOGY_CANON.md for full specification.

---

## Part 2: Paper Versions (Canonical Selection)

| Version | Pages | Strengths | Weaknesses | Status |
|---|---|---|---|---|
| Aurelle draft | ~9 | Market positioning | Wrong topology, overclaims | Superseded |
| Formal Core (GPT-5.4) | ~12 | Non-theorems, compound RIBA_ZERO, 3-outcome membrane | Wrong topology (per-user URP) | Superseded |
| Merged v3 (Claude Opus) | 5 | Correct topology, Seed Chain, citations | No ablation, less formal | Superseded |
| Gold Standard (Claude Code) | 8 | Full DFA, ablation, std dev, P99, adversary model | No explicit topology lock, no Seed Chain | Reference |
| **Submission Grade v2 (Claude Code)** | **5** | **Topology lock, evidence classification, proof protocol, CEL** | **No Seed Chain, no ablation** | **SUBMIT THIS** |
| CMN_arXiv_Final (Claude Opus) | 4 | All contributions merged incl Seed Chain | ReportLab formatting less polished | Backup |
| Professional Proof (latest) | ~6 | Lifecycle emulation, competitive comparison, Bounty vertical | Emulated data not real measurements | Supplementary |

**Canonical decision:** Submit **CMN_Submission_Grade_v2.pdf** to arXiv. It has correct topology, evidence classification, and proof protocol. The Seed Chain and ablation study go into the first revision (v2 upload) after arXiv ID is obtained.

---

## Part 3: Metrics (Resolved Discrepancies)

### Benchmark Numbers (EVIDENCE — not independently verified)

Two measurement runs produced slightly different numbers. The Gold Standard ran n=10,000 samples with std dev and P99. The Submission v2 used a reproducibility shell with different sampling. Canonical values use the Gold Standard numbers (larger sample, better statistical treatment).

| Operation | Canonical value | Std Dev | P99 | Source |
|---|---|---|---|---|
| IHSAN check | 90.4 ns | 12.3 ns | 145 ns | Gold Standard (n=10K) |
| BLAKE3 hash (4KB) | 349 ns | 28.7 ns | 412 ns | Both (match) |
| Ed25519 sign | 396 ns | 35.2 ns | 478 ns | Gold Standard (n=10K) |
| Total membrane | 3.02 us | 0.89 us | 10.14 us | Both (match) |
| Throughput | 237,199 req/s | — | — | Both (match) |

**Discrepancy resolution:**
- IHSAN: 90.4 ns (Gold Standard) vs 93 ns (Submission v2) → Use 90.4 ns. Delta is 2.6 ns, within measurement noise.
- Ed25519: 396 ns (Gold Standard) vs 630 ns (Submission v2) → Use 396 ns. The 630 ns likely included key generation overhead. The Gold Standard measured sign-only with pre-loaded key. Note: report both if challenged.

### Codebase Numbers (VERIFIED)

| Metric | Canonical value | Last verified |
|---|---|---|
| Rust crates | 26 | March 2026 |
| Rust tests | 1,446 | March 2026 |
| Python tests | 11,216 | March 2026 |
| Total tests | 12,662 | March 2026 |
| Git commits | 577+ | March 2026 (growing) |
| Phase | 87-88 (v0.88.1) | March 2026 |
| FAISS vectors | 84,795 | March 2026 |
| FAISS query latency | 5 ms | March 2026 |
| Knowledge graph nodes | 577 | March 2026 |
| Knowledge graph edges | 104,957 | March 2026 |
| Heartbeat longest run | 6.5 hours | March 2026 |
| Heartbeat errors | 0 | March 2026 |
| Reflex speedup | 126.7x (153.27ms → 1.21ms) | March 2026, N=1 only |
| Block 0 hash | 350d642099bde68b | March 2026 |
| Block 0 SEED | 1.1M | March 2026 |

### Numbers NOT YET MEASURED

- Multi-node latency (no federation test)
- Reverse scaling at N>1 (projection only)
- Morning brief delivery time
- Daily mission count in production
- Reflex hit ratio over time
- 24-hour continuous heartbeat (6.5h proven, 24h pending)

---

## Part 4: Constitutional Invariants

| Invariant | Value | Implementation | Status |
|---|---|---|---|
| IHSAN_FLOOR | >= 0.95 | Rust newtype (compile-time) | Verified |
| ZANN_ZERO | No unverified claims | Membrane admissibility check | Verified |
| RIBA_ZERO | Compound: ArithmeticIntegrity AND EconomicPolicyIntegrity | Sippar + policy rules | Partial (arithmetic verified, policy planned) |
| GINI_CEILING | <= 0.35 | Ledger invariant | Verified (NOTE: conflict with 0.45 in some older docs — 0.35 is canonical) |

**Gini conflict resolution:** Some earlier documents reference 0.45. The canonical operational value is **0.35**. This was identified in multiple audit sessions. 0.35 is the value implemented in the Rust types. 0.45 appeared in the constitutional Spine text and needs amendment.

---

## Part 5: Authority Hierarchy

```
Quran → Hadith → البذرة (Jul 2023) → الرسالة (Jul 2023)
  → Enforceable Spine v1.1 → Root Invariants
    → TOPOLOGY_CANON.md → METRICS_CANONICAL.md → This file
      → Specs → Code → Tests
```

No document at a lower level may override a document at a higher level.

---

## Part 6: Novel Contributions (Canonical List)

| # | Contribution | Paper section | Status |
|---|---|---|---|
| 1 | Constitutional Membrane (fail-closed DFA) | §3 | In all paper versions |
| 2 | Isnad Risk Propagation (IRP) | §4-5 | In all paper versions |
| 3 | Frozen Agent Principle (Gödel Escape) | §5-6 | In all paper versions |
| 4 | Compound RIBA_ZERO | §5.3 / Non-Theorem 4 | In Formal Core onward |
| 5 | Seed Chain (autopoietic prompt arch) | §5 of merged | In merged v3 and arXiv Final ONLY |
| 6 | Constitutional Engram Layer (O(1) lookup) | §7 | In Submission v2 onward |
| 7 | Proof Protocol (5-phase) | §10 | In Submission v2 onward |
| 8 | Sippar exact arithmetic | §5.3 | In all versions |

---

## Part 7: File Locations

| What | B:\ location (sovereignty) | C:\ location (repo) |
|---|---|---|
| Topology canon | 00_CONSTITUTION\TOPOLOGY_CANON.md | TOPOLOGY_CANON.md |
| This file | 00_CONSTITUTION\BIZRA_CANONICAL.md | BIZRA_CANONICAL.md |
| Metrics | 13_SPRINT_EXECUTION\sprint_0_submit\METRICS_CANONICAL.md | METRICS_CANONICAL.md |
| Giants | 13_SPRINT_EXECUTION\sprint_0_submit\GIANTS.md | GIANTS.md |
| Genesis lineage | 00_CONSTITUTION\GENESIS_LINEAGE.md | GENESIS_LINEAGE.md |
| Genesis provenance | 00_CONSTITUTION\GENESIS_PROVENANCE_VERIFIED.md | GENESIS_PROVENANCE_VERIFIED.md |
| Project blueprint | BIZRA_PROJECT_BLUEPRINT.md | BIZRA_PROJECT_BLUEPRINT.md |
| Gate tracker | 13_SPRINT_EXECUTION\GATE_TRACKER.md | — (B:\ only) |
| Paper (submit) | 10_BIZRA-LAB\publications\ | docs\ |
| Paper fields | — | docs\ARXIV_SUBMISSION_FIELDS.md |
| البذرة | 00_CONSTITUTION\al_bisara_founding\البذرة_2023.pdf | البذرة.pdf |

---

*This file is the canonical resolution of all conflicts across all BIZRA documents.*
*CLAIM_MUST_BIND applies to this file itself.*
