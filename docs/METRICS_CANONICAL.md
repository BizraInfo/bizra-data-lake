# BIZRA — Canonical Metrics (SSOT)

> Every number in this file is verified by running the stated command on Node0.
> Last verified: 2026-04-14
> Any investor document MUST cite numbers from this file, not from memory.

---

## Constitutional Proof Surface: 983 tests

These tests verify that BIZRA's constitutional machinery works correctly:
receipts, evidence audit, FATE gate, cryptographic identity, token economics,
SAT gates, and the zero-proof kernel.

| Module | Tests | What it verifies |
|---|---|---|
| proof_engine | 703 | Receipts, BLAKE3 chains, FATE gate, loop proof, Ihsan scoring, evidence audit |
| pci | 122 | Ed25519 signing, RFC 8785 canonicalization, domain separation |
| token | 92 | SEED/BLOOM minting, emission decay, Zakat distribution, supply caps |
| sat | 31 | SAT-5 gate logic, composite evaluator (5 gates, 59 checks), ceremony |
| urp | 27 | Universal Resource Pool, constitutional membrane, resource management |
| zpk | 8 | Zero-proof kernel |
| **Total** | **983** | `pytest tests/core/{proof_engine,pci,token,sat,urp,zpk}/` |

## ADK Agent Tests: 51 tests

| Category | Tests | What it verifies |
|---|---|---|
| Agent framework | 11 | Charter binding, identity, draft/refuse protocol, tool discovery |
| Mission/Budget | 9 | Budget enforcement, governance class, exhaustion handling |
| Runner lifecycle | 8 | 7-step FATE pipeline, charter drift, fabricated evidence, protocol |
| Researcher | 9 | Full lifecycle through real FATE gate with Ollama |
| Phase C agents | 14 | 5 new agents: import, tools, LOC limits, charter uniqueness, evidence |

## Integration Tests: 638 tests

Cross-module pipelines: FATE pipeline, PAT-SAT provenance, golden path,
threshold enforcement, sovereignty pipeline, token integration, chaos probes.

## Ecosystem Aggregate: 11,605 tests

The full test suite across all modules. Includes the above plus sovereign
runtime, memory, federation, inference, orchestration, and all other subsystems.

Command: `pytest --co -q` → `11,605 tests collected`

---

## Code

| Metric | Value | Command |
|---|---|---|
| Python core/ LOC | 258,987 | `find core/ -name "*.py" \| xargs cat \| wc -l` |
| ADK LOC | 1,526 | 7 agents + framework (13 files) |
| Rust crates | 24 | `grep -c '"bizra-' bizra-omega/Cargo.toml` |
| Rust workspace | compiles clean | `cargo check --workspace` |
| Binary | 3.3 MB | `bizra-omega/target/release/bizra` |

## Agents

| Agent | Type | Model | LOC | Status |
|---|---|---|---|---|
| Researcher | PAT | gemma4:26b-bizra-16k | 133 | EXERCISED |
| Strategist | PAT | gemma4:26b-bizra-16k | 140 | EXERCISED |
| Analyst | PAT | qwen2.5-coder:14b | 153 | EXERCISED |
| Creator | PAT | gemma4:e4b | 116 | EXERCISED |
| Executor | PAT | deepseek-r1:7b | 157 | EXERCISED |
| Coordinator | PAT | gemma4:26b-bizra-16k | 137 | EXERCISED |
| Guardian | PAT | gemma4:26b-bizra-16k | 137 | EXERCISED |

PAT-7: **7/7 EXERCISED** through the ADK lifecycle with FATE gate verification.

## SAT-5 Gates

| Gate | Checks | Status |
|---|---|---|
| Sentinel | 11 | PASS — structural integrity |
| Oracle-S | 14 | PASS — Ihsan/quality scoring via LLM |
| Ledger | 10 | PASS — receipt chain verification |
| Conductor | 12 | PASS — consensus rules |
| Ambassador | 12 | PASS — network boundary |
| **Total** | **59** | All 5 gates passing, fail-closed composite |

## Constitutional Constants

| Constant | Value | Meaning |
|---|---|---|
| IHSAN_THRESHOLD | 0.95 | Minimum excellence score for any output |
| SNR_THRESHOLD | 0.85 | Minimum signal-to-noise ratio |
| ZAKAT_RATE | 0.025 | 2.5% redistribution to community pools |
| ADL_GINI_THRESHOLD | 0.35 | Maximum inequality in token distribution |
| ADL_HARBERGER_TAX | 0.05 | 5% tax on idle compute resources |
| RIBA_ZERO | true | Zero tolerance for interest-based instruments |

## Infrastructure

| Component | Status |
|---|---|
| GitHub repos | 148 (136 public, 12 private) |
| Commits on main | 814 |
| Ollama models | 6 (gemma4:26b-bizra, gemma4:26b, qwen2.5-coder:14b, gemma4:e4b, deepseek-r1:7b, nomic-embed-text) |
| Docker services | 4 (Python API :8000, Rust API :3001, Frontend :3000, Desktop Bridge) |
| Node0 | MSI Titan 18 HX, i9-14900HX, 128GB RAM, RTX 4090 Mobile 16GB |

## Data Asset

| Source | Files | Status |
|---|---|---|
| BIZRA-ASSET (deduplicated) | 64,375 unique | Local on /data2 |
| Cloud drives (9 remotes) | ~432K indexed | Pull in progress |
| Target after full dedup | ~312K unique | ~574 GB |

## Provenance

| Milestone | Date | Evidence |
|---|---|---|
| البذرة (The Seed) written | June 2023 | WhatsApp share timestamp, .docx dated 2023-07-22 |
| First BIZRA video | Aug-Sep 2023 | UpscaleVideo_20230919.mp4 |
| First code (Jupyter notebooks) | Oct 2023 | 231 notebooks in Bizra_Blockchain_System/ |
| Spearpoint seal | 2026-04-12 | commit b08f2208 |
| ADK Phase C complete (6/7 agents) | 2026-04-14 | commit 01625aa9 |
| ADK Phase D complete (7/7 + SAT-5) | 2026-04-14 | commit 0d6aa925 |

---

## Scope Labels

When citing numbers from this document:

- **983** = constitutional proof surface (investor-safe, canonical)
- **51** = ADK agent framework tests (session-built, verified)
- **638** = integration tests (cross-module, reported separately)
- **11,605** = ecosystem aggregate (full suite, label as "total ecosystem")

Do NOT use unlabeled aggregate numbers in investor materials.
Do NOT cite numbers from memory — run the command and verify.
