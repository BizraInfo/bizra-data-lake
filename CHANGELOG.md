# Changelog

All notable changes to BIZRA-DATA-LAKE are documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

---

## [Unreleased]

### Added
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
