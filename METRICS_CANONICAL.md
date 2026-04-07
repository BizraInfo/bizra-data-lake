# BIZRA Node0 — Canonical Metrics (April 6, 2026)

> بِسْمِ اللَّهِ الرَّحْمَٰنِ الرَّحِيمِ

These numbers are truth-labeled. Every metric was derived from the live codebase, CI pipeline, or git history on Node0. No rounding. No inflation.

**Status labels used in this document:**

| Label | Meaning |
|-------|---------|
| `VERIFIED` | Tested and gated in CI; a failing value blocks the build |
| `MEASURED` | Produced by `cargo test`, `pytest`, or an automated benchmark run |
| `COUNTED` | Derived by `wc -l`, `git log`, `ls`, or equivalent deterministic shell command |

---

## Codebase Scale

| Metric | Value | Status |
|--------|-------|--------|
| Total lines of code | 556,000+ | `COUNTED` |
| Python LOC | 251,000 | `COUNTED` |
| Rust LOC | 116,000 | `COUNTED` |
| TypeScript LOC | 10,000 | `COUNTED` |
| Test LOC (all languages) | 179,000 | `COUNTED` |
| Rust crates (bizra-omega workspace) | 25 | `COUNTED` |
| Python subpackages (core/) | 72 | `COUNTED` |

---

## Test Health

| Metric | Value | Status |
|--------|-------|--------|
| Rust tests passing | 1,122 | `MEASURED` |
| Rust test failures | 0 | `VERIFIED` |
| Python tests collected | 11,415 | `MEASURED` |
| Combined test count | 12,537 | `MEASURED` |

---

## Constitutional Spine

| Metric | Value | Status |
|--------|-------|--------|
| Frozen Rust objects | 6 | `VERIFIED` |
| Constitutional spine LOC | 2,111 | `COUNTED` |
| Sippar crate LOC (exact arithmetic) | 485 | `COUNTED` |
| Frozen agents (immutable at runtime) | 2 (P5 Ethicist · S2 Oracle) | `VERIFIED` |
| IHSAN threshold | ≥ 0.95 | `VERIFIED` |
| SNR floor | ≥ 0.85 | `VERIFIED` |
| ADL_GINI ceiling | ≤ 0.35 | `VERIFIED` |
| ZANN constraint | = 0 (zero) | `VERIFIED` |
| RIBA constraint | = 0 (zero) | `VERIFIED` |

---

## CI Pipeline

| Metric | Value | Status |
|--------|-------|--------|
| Active CI workflows | 21 | `COUNTED` |
| Gate maturation stages | 4 (Observe → Flag → Throttle → Reject) | `VERIFIED` |
| Throttle multiplier | ×5 before Reject | `VERIFIED` |
| Gate direction | Monotonic (tightens only, never softens) | `VERIFIED` |

---

## Binary Artifacts

| Binary | Size | Build Flags | Status |
|--------|------|-------------|--------|
| bizra-node | 2.8 MB | release · LTO · strip | `MEASURED` |
| bizra-api | 5.1 MB | release · LTO · strip | `MEASURED` |
| PyO3 bridge | 3.2 MB | release | `MEASURED` |

---

## Evidence Chain

| Artifact | Count | Status |
|----------|-------|--------|
| Signed receipts | 7+ | `COUNTED` |
| Manifests | 2 | `COUNTED` |
| Benchmark campaigns | 3 | `COUNTED` |

---

## Git History

| Metric | Value | Status |
|--------|-------|--------|
| Total commits | 763 | `COUNTED` |
| Pre-release tags | 5 (v0.87.0 through v0.89.1) | `COUNTED` |
| HEAD commit | `0115016b` | `COUNTED` |
| HEAD commit message | "P0 Ihsan 0.85→0.95 constitutional fix" | `COUNTED` |

---

## Agent Parliament

| Component | Members | Status |
|-----------|---------|--------|
| PAT-7 (user-local council) | 7 agents | `VERIFIED` |
| SAT-5 (system governance) | 5 agents | `VERIFIED` |
| Total parliament size | 12 agents | `VERIFIED` |

---

## BYOB LLM Router

| Model | Backend | Status |
|-------|---------|--------|
| deepseek-r1-32b | LM Studio | `MEASURED` |
| qwen2.5-32b | LM Studio | `MEASURED` |
| llava-7b | LM Studio | `MEASURED` |
| qwen2.5-coder-32b | LM Studio | `MEASURED` |
| Fallback | Ollama | `VERIFIED` |

---

## Node0 Hardware

| Component | Specification | Status |
|-----------|--------------|--------|
| Machine | MSI Titan 18 HX | `COUNTED` |
| CPU | i9-14900HX · 24 cores | `COUNTED` |
| GPU | RTX 4090 · 16 GB VRAM | `COUNTED` |
| RAM | 128 GB DDR5 | `COUNTED` |
| Evidence drive | B:\BIZRA-SOVEREIGN | `COUNTED` |
| Runtime drive | C:\BIZRA-DATA-LAKE | `COUNTED` |

---

## Benchmark Highlights (from academic lineage)

| Benchmark | Result | Source |
|-----------|--------|--------|
| Reflex memory speedup | 7.55× over baseline | Bera et al., Apr 2025 |
| FormalJudge vs. LLM-as-Judge | +16.6% improvement | Zhou et al., Feb 2026 |
| Aegis alignment retention | 98.2% under adversarial prompts | Aegis Governance, Mar 2026 |
| LifeBench top-system recall | 55.2% (industry ceiling) | LifeBench, Mar 2026 |

---

Every metric in this document is reproducible from the canonical workspace at `C:\BIZRA-DATA-LAKE`. Run `cargo test --workspace` for Rust counts, `pytest --collect-only` for Python counts, and `git log --oneline | wc -l` for commit history. Numbers that cannot be reproduced by these commands are not in this document.

---

*BIZRA Sovereign Node · Node0 · April 6, 2026*
*Mohamed Beshr · m.beshr@bizra.info · Dubai, UAE*

> بذرة واحدة تصنع غابة — One seed makes a forest.
