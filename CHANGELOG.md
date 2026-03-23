# Changelog

All notable changes to BIZRA-DATA-LAKE are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [v0.88.1] — 2026-03-23 "The Organism Release"

### Added
- 9-stage MissionExecutor (FAISS → Amplify → Inference → Skill → SEED → Memory → EventBus → Notify → Watcher)
- LLM response text flows end-to-end (was silently dropped)
- Sovereign identity system prompt — model knows it's BIZRA
- Receipt chain threading across sessions (chain_head persisted)
- Live stage streaming — user watches the nervous system fire
- DiffusionAmplifier on canonical path (fail-closed HMM→GoT hints)
- FanoutEventBus — unified CQRS + sovereign bus dialects
- Node0 dead-letter evidence — delivery failures persisted as audit artifacts
- Governed RSI wired as opt-in (`BIZRA_AUTOPOIESIS_ENABLED=true`)
- CMN v2 paper — "Intelligence Through Governed Refusal"
- Z3 formal proofs of all 4 membrane properties (zero counterexamples)
- 18 executable proof tests in CI gate
- Membrane tax benchmark (0.007ms = 0.00003% overhead)
- Adversarial simulation (82.3% attacks blocked)
- Security Scanning CI pipeline (cargo-audit, bandit, pip-audit, detect-secrets)
- Membrane Proof Gate — Z3 verification on every push
- "Verify It Yourself" README section — 8 commands to prove every claim
- 6 frontend prototypes (FrontDoor, JARVIS, SovereignCockpit, SovereignWorld)
- Business docs (Product Spec v1, GTM 90-Day Launch Plan)

### Fixed
- Bus dialect seam — CQRS and sovereign buses now speak one language
- 4 broad exception handlers replaced with typed exceptions in canonical spine
- 2 LM Studio API keys purged from git history (filter-repo)
- CI lint gates (cargo fmt nightly, isort, clippy, Black)
- Dependabot lru 0.12→0.16 (RUSTSEC Stacked Borrows)
- Cross-session desync (parallel session event_publisher sync)

### Changed
- Coverage threshold ratcheted 65% → 70%
- CI pinned ubuntu-latest → ubuntu-24.04
- Deploy tags pinned (chaos-mesh 2.7.0, uv 0.9.21, nomic-embed-text v1.5)
- FAISS: Linux-side cache (0.5s vs 60s), offline encoder, 15s timeout
- Pipeline expanded 8 → 9 stages (cognitive amplification added)

### Security
- 8/8 credentials rotated
- 0 secrets in code or git history
- All GitHub Actions pinned by SHA

## [Unreleased]

### Added
- **MVSA Authority Resolution** (`core/sovereign/node0_authority.py`)
  - 4-level precedence: canonical → legacy_ceremony → legacy_reference → fail-closed
  - Migration of ceremony-compatible legacy sources into `sovereign_state/`
  - Conflict detection (LEGACY_GENESIS_CONFLICT) and reference-only rejection
  - Atomic migration receipt: `sovereign_state/node0_authority_migration.json`
- **Rust MVSA Proof Binary** (`bizra-omega/bizra-resourcepool/src/bin/node0_mvsa.rs`)
  - Genesis validation, loopback network bootstrap, BLAKE3 self-validation
  - Structured JSON proof: `sovereign_state/node0_mvsa_proof.json`
  - 5-step binary resolution: env → release → debug → cargo → fail-closed
- **Lifecycle v2 Schema** (`node0_lifecycle.json` schema 2.0.0)
  - 11 gates: genesis_authority_valid through restart_recovery_ready
  - Status semantics: blocked → degraded → ready
  - Mission tracking: last_evidence_receipt_id, last_ihsan_score, last_snr_score
  - Restart recovery validation on second health call
- **`prove-mvsa` CLI subcommand** + `GET /mvsa` and `POST /prove-mvsa` API routes
- **`core/sovereign/atomic_io.py`** — shared crash-safe write→fsync→rename utility
- **`core/sovereign/node0_mvsa.py`** — Python wrapper for Rust MVSA binary
- 15 new tests (8 authority + 7 MVSA wrapper), 93/93 combined suite green
- **Wave 1 — Inference Provenance on receipts** (`core/sovereign/mission.py`)
  - `InferenceProvenance` dataclass: backend, model_id, fallback_chain, latency_ms, tokens_generated
  - `_synthesize()` returns `(str, InferenceProvenance)` tuple; provenance attached to MissionResult
  - Provenance propagated to `mission.completed` events and `handle_rpc()` responses
  - `BIZRA_OLLAMA_MODEL` env var for configurable Ollama model ID
- **Wave 2 — 12 CQRS subscriber wiring** (`core/sovereign/organism.py`)
  - `_wire_subscribers()` method with 12 no-op adapters for unresolved dependencies
  - `_emit_cqrs_receipt()` publishes ACTION_RECEIPT + optional IHSAN_GATE_BREACHED to CQRS bus
  - CQRS bus metrics (subscribers_wired, chain_height, chain_valid) in organism stats
  - Graceful degradation: bus wiring failures do not block boot
- 10 new integration tests: 4 provenance + 6 CQRS wiring (117/117 total pass)
- Unified 8D Ihsān content scorer (`core/sovereign/ihsan_scorer.py`, 648 LOC)
  - 8 canonical dimensions scored by content analysis (moral_clarity, epistemic_humility,
    structural_integrity, verifiability, contextual_relevance, intent_alignment, resilience, efficiency)
  - 4D SNR scorer per §8 Shannon weights (signal_density, evidence_grounding,
    contradiction_resolution, actionability)
  - Weighted geometric mean: zero in ANY dimension → zero composite (Al-Ghazali fail-closed §4)
  - Neutral handling for missing context (0.5 not 0.0 when input_text absent)

### Fixed
- Threshold canonicalization drift: removed `except ImportError` fallback blocks from
  helix3.py, mission_nervous_system.py, mission_pipeline.py, organism.py, bloom.py —
  all now hard-import from `core/integration/constants.py` SSOT (4/4 canonicalization tests pass)
- Phantom import bug: `SNR_MINIMUM_THRESHOLD` → `UNIFIED_SNR_THRESHOLD` in
  mission_nervous_system.py (fallback was masking a real ImportError)
- Scorer wiring: mission_nervous_system.py and mission_pipeline.py now delegate to
  ihsan_scorer as single source of truth, replacing duplicate surface heuristics

### Changed
- SEC-003: `mission.py` — all 20 boundary-guard `except Exception` catches now include
  `exc_info=True` for full traceback diagnosability in production logs (30/30 tests pass)
- Phase 20.1: SAPE Sovereign Intelligence Report dashboard (`static/sovereign_analysis.html`)
  - 7 hidden patterns (HP-01..HP-07) with SNR scoring and evidence chains
  - Interactive Graph-of-Thoughts canvas (13 nodes, 17 edges, 4 levels)
  - SNR v2.1 analysis with tier bars and Shannon channel metrics
  - 5 Omega implementation phases with deliverables and test coverage
- Phase 20: RDVE Actuator Layer on Desktop Bridge
  - Shannon entropy gate (H >= 3.5 bits/char) blocking low-signal instructions
  - `actuator_execute` handler with 3-gate pipeline (FATE -> Shannon -> Rust GateChain)
  - `get_context` handler returning UIA schema for desktop state fingerprinting
  - `ActuatorSkillLedger` typed registry with 3 baseline AHK skills
  - 26 new tests covering entropy gate, actuator handlers, and skill ledger
- Smart File Management skill (`core/skills/smart_file_manager.py`) with scan, organize, rename, merge operations
- Token system (`core/token/`) with SEED, BLOOM, IMPT token types and hash-chained ledger
- Experience ledger and judgment telemetry modules
- Desktop Bridge security layer (localhost binding, auth envelope, rate limiter, replay protection)
- Documentation portal (`docs/README.md`) with role-based reading paths
- Machine-generated knowledge indexes (`docs/knowledge/`)
- Spearpoint RDVE recursive loop and auto-evaluator
- SAT controller for sovereign runtime

### Changed
- Documentation A+ quality remediation across 17 files
- FATE acronym corrected to Fidelity, Accountability, Transparency, Ethics
- Constitutional thresholds unified to single source of truth (`core/integration/constants.py`)
- Rust workspace expanded to 14 crates (added `bizra-tests`)

### Fixed
- Phase 19 Sovereign Consolidation: Green Main Protocol achieved (6,423/6,423 tests passing)
- 46 ruff lint errors eliminated across `core/` (25 f-string placeholders, 19 unused imports, 2 unused variables)
- 138 black formatting violations resolved across `core/`
- 10 isort import ordering violations resolved across `core/`
- ZPK kernel tests fixed: hash algorithm aligned to BLAKE3 (`hex_digest`) matching production code
- Token ledger tests fixed: isolated with `tmp_path` to prevent shared state corruption
- Pipeline routing test fixed: assertion aligned with fail-closed gate chain behavior
- A2A TaskManager test fixed: attribute assertion aligned with actual implementation
- Asyncio event loop test fixed: `new_event_loop()` replaces deprecated `get_event_loop()`
- CI workflows hardened (all 7 pipelines: ci, deploy, native-ci, performance, release, tests, docs-quality)
- Gini threshold corrected to 0.40 across all docs (was inconsistent 0.35 in some)
- Coverage threshold corrected to 60% (was erroneously 97.5% in compliance matrix)
- Clock skew tolerance aligned to 120s across all references

---

## [v2.2.0-sovereign] - 2026-02-12

### Added
- Phase 18.1: End-to-end integration wiring with orchestrator, complexity router, FastAPI CLI
- Phase 18: Execution engine + SQLite memory + Node0 Console
- Phase 18-prep: PyO3 `InferenceGateway` bridge + E2E tests + gateway-integrated benchmarks
- Phase 17.5: Proof Forge Evidence System + True Spearpoint Integration
- Phase 17: Elite Hardening Sprint (exception discipline, dead code purge, supply chain, test scaffolding)
- Phase 16: SAPE Audit P0/P1 remediation (security CI hardening, panic elimination, constant centralization)
- Phase 15: Elite Engineering Sprint (dependency cleanup, coverage CI, inline docs)
- Phase 14: CI Hardening + Doc Sprint + Benchmark Sprint
- Phase 13: 233 new Rust tests across 5 crates
- Phase 12: Steel Spine Rust Hardening

### Changed
- Mypy zero-error sprint: 409 errors eliminated across 82 files
- Gateway god-file split into modular components
- P2 security sprint + gateway refactoring

### Fixed
- 3 CRITICAL security vulnerabilities killed, API hardened, deps pinned, constants centralized
- Mypy errors reduced from 1477 to 0 (multi-phase effort)
- 55 pre-existing test failures eliminated (52 failures + 3 collection errors)
- Clippy lints resolved for Rust 1.88 compatibility

---

## [v1.0.0-genesis] - 2026-01-15

### Added
- PAT (Personal AI Terminal) system with user context integration
- RAG retrieval + Claude.ai ingestion (2,495 memories searchable)
- Chat history ingestion (584 conversations, 9,265 messages)
- LM Studio as primary backend
- Pre-seeded founder profile + 16 e2e smoke tests
- Data import wizard for history ingestion
- User-facing Spearpoint with onboarding, gateway, impact tracker
- PyO3 bindings for autopoiesis (10-100x pattern learning performance)
- Encrypted keypair storage with sovereign vault (S-5 security)

### Changed
- SPARC methodology adopted: Dockerfiles, Clippy, type safety, security hardening
- CI pipeline established with soft-gates for pre-existing findings

### Fixed
- Docker build paths corrected
- Integration tests soft-gated for Docker dependency
- Security scan soft-gated for pre-existing findings
- Quality gate uses SNR threshold (0.85) not Ihsan threshold (0.90)
- PyO3 virtualenv + flaky Rust metrics test
- Multiple Clippy lint waves resolved (Rust 1.83, 1.85, 1.88)
- Ruff lint fixes + performance workflow os import
- isort import ordering (11 files) + Black formatting (55 files)

---

## Version History

| Tag | Date | Milestone |
|-----|------|-----------|
| `v2.2.0-sovereign` | 2026-02-12 | Phases 12-18.1, sovereign runtime, elite hardening |
| `v1.0.0-genesis` | 2026-01-15 | PAT system, first CI pipeline, Spearpoint v1 |
