# Phase 80 — Season Archive Integration Manifest

## Status: SPEC-READY
## Date: 2026-03-09
## Source: `my complete season/` (56 files, Mumo's design archive)

---

## 1. Purpose

The `my complete season/` directory contains 56 files representing the complete
design archive from the BIZRA conception-to-implementation journey. This spec
defines which artifacts integrate into the codebase, where they go, and in
what order.

---

## 2. Classification Summary

| Category | Count | Action |
|----------|-------|--------|
| Code to integrate | 11 | Copy to target, write tests |
| Schemas & config | 4 | Compare + merge |
| Terminal TSX | 9 | Copy to `frontend/src/` |
| Constitutional docs | 10 | Archive to `docs/constitutional/` |
| Sprint plans | 5 | Archive to `docs/sprints/` |
| Installer scripts | 3 | Copy to `scripts/` |
| Skipped files | 14 | **SKIP** — bytecode, cache, superseded/duplicate docs (see §8) |

---

## 3. Code Integration (Priority Order)

### 3.1 Priority 1 — Enables Week 2 MOE Engine

| Source | Target | LOC | Action | TDD Anchor |
|--------|--------|-----|--------|------------|
| `reflex_compiler.py` | `core/sovereign/reflex_compiler.py` | 658 | **UPGRADE** — adds HHMM engine, evidence recorder, gossip export, forest import, revalidation (+181 LOC over current 477) | Existing 43 tests + 12 new for HHMM paths |
| `subscribers.py` | `core/bus/subscribers.py` | 810 | **NEW** — 12 EventBus subscribers (Phase 1: learning loop, Phase 2: safety, Phase 3: economics) | 24 tests (2 per subscriber) |
| `action_schema_v1.json` | `docs/schemas/action_schema_v1.json` | 281 | **NEW** — JSON Schema for action bus contracts | Schema validation test |
| `event_schema_v1.json` | `docs/schemas/event_schema_v1.json` | 271 | **NEW** — JSON Schema for event bus contracts | Schema validation test |

### 3.2 Priority 2 — Enables SDPO + Token System (Week 3-4)

| Source | Target | LOC | Action | TDD Anchor |
|--------|--------|-----|--------|------------|
| `bloom.py` | `core/token/bloom.py` | 453 | **NEW** — BLOOM governance token, community pool, Gini enforcement, decay | 15 tests (mint, decay, Gini, pool split, soulbound rejection) |
| `bizra_test.py` | `bizra_test.py` (root) | 468 | **UPGRADE** — diff and merge new features (Delta test runner + version lock tool, T0-T4 tiers) | Self-testing (--status mode) |
| `conftest_tiers.py` | `tests/conftest_tiers.py` | 46 | **NEW** — supplemental tier markers plugin (does not replace `tests/conftest.py`); pytest plugin for tier markers (T0 smoke → T4 genesis gate) | Marker registration test |

### 3.3 Priority 3 — Genesis Gate Readiness (Week 5+)

| Source | Target | LOC | Action | TDD Anchor |
|--------|--------|-----|--------|------------|
| `genesis_gate.py` | `scripts/genesis_gate.py` | 567 | **NEW** — 68-check gate runner (5 SAT layers) | Integration test against running API |
| `empirical_validation.py` | `scripts/empirical_validation.py` | 1188 | **NEW** — V1-V10 validation suite (economics, scaling, latency, Gini, precipitation, HHMM, chain, self-critique, impossibility, P5) | 10 validation functions, each self-reporting |
| `sovereign_lifecycle_proof.py` | `scripts/sovereign_lifecycle_proof.py` | 1115 | **NEW** — Full lifecycle proof harness (6 claims: self-sustainable, self-critique, self-correct, self-optimize, complete lifecycle, one:one) | Proof receipt generation |
| `h_neuron_proof.py` | `scripts/h_neuron_proof.py` | 922 | **NEW** — H-Neuron hallucination localization experiment (requires GPU + torch) | GPU-dependent, mark `@requires_gpu` |

### 3.4 Priority 4 — CLI & Installers

| Source | Target | LOC | Action |
|--------|--------|-----|--------|
| `bizra-cli.py` | `bizra_cli.py` (root) | 770 | **NEW** — Unified CLI entry point (start, stop, status, mission, briefing, wallet, identity, doctor, reset, launch) |
| `sovereign_terminal.py` | `core/sovereign/terminal.py` | 681 | **NEW** — Rich TUI REPL with DEMA persona |
| `install-bizra-cli.ps1` | `scripts/install-bizra-cli.ps1` | 130 | **NEW** — Windows installer |
| `install-bizra-cli.sh` | `scripts/install-bizra-cli.sh` | 152 | **NEW** — Linux/macOS installer |

---

## 4. Terminal TSX Integration

All 9 files target `frontend/src/components/terminal/` and `frontend/src/`:

| Source | Target | LOC | Component |
|--------|--------|-----|-----------|
| `terminal-shell.tsx` | `components/terminal/terminal-shell.tsx` | 197 | Shell container (7 views + nav + status bar) |
| `terminal-memory.tsx` | `components/terminal/terminal-memory.tsx` | 355 | Memory/briefing view |
| `terminal-network.tsx` | `components/terminal/terminal-network.tsx` | 299 | Network/lifecycle view |
| `terminal-settings.tsx` | `components/terminal/terminal-settings.tsx` | 341 | Settings/identity view |
| `terminal-skills.tsx` | `components/terminal/terminal-skills.tsx` | 401 | Skills/agents/reflexes view |
| `terminal-timeline.tsx` | `components/terminal/terminal-timeline.tsx` | 504 | Timeline/event proof chain |
| `economic.ts` | `lib/economic.ts` | 236 | Economic calculations (PoI, receipts) |
| `useWallet.ts` | `hooks/useWallet.ts` | 245 | Wallet state hook |
| `wallet-hardening.test.ts` | `tests/wallet-hardening.test.ts` | 224 | Wallet edge case tests |

**Prerequisite:** Frontend must have `useSovereignApi` hook and
`constitutional-constants.ts` (both shipped in Phase 78).

---

## 5. Constitutional Docs (Archive Only)

These are LOCKED reference documents. Copy to `docs/constitutional/`:

```
docs/constitutional/
├── BIZRA-Constitutional-Sources-v1.0.docx
├── BIZRA-Identity-Canon-v1.0.docx
├── BIZRA-Proof-Canon-v1.0.docx
├── BIZRA-Proof-Canon-v1.1.docx
├── BIZRA-Elite-Engineering-Blueprint.docx
├── BIZRA-Omega-Infinity-Peak-Synthesis.docx
├── BIZRA-Omega2-Definitive-Synthesis.docx
├── BIZRA-Enforceable-Spine-v1.0.md
├── BIZRA-Definition-of-Done-Genesis-100.md
└── BIZRA-Peak-Synthesis-Omega-Infinity.md
```

---

## 6. Sprint Plans (Archive Only)

Copy to `docs/sprints/`:

```
docs/sprints/
├── BIZRA-Genesis-Sprint-Blueprint.md
├── BIZRA-Sprint2-4-Execution-Playbook.md
├── BIZRA-Sprint2-4-Execution-Playbook-v1.1.md
├── BIZRA-Evidence-Delta-v1.0-to-v1.1.md
└── BIZRA-Versioned-Test-Strategy-v1.0.md
```

---

## 7. Config Comparison

| Source | Codebase | Action |
|--------|----------|--------|
| `bizra-ci-pipeline.yml` | `.github/workflows/ci.yml` | **COMPARE** — season version may have updated gates. Diff and cherry-pick improvements. |
| `bizra-pyproject.toml` | `pyproject.toml` | **COMPARE** — season version may have updated metadata. Do NOT overwrite coverage settings. |

---

## 8. Files to SKIP

| File | Reason |
|------|--------|
| `reflex_compiler.cpython-312.pyc` | Compiled bytecode |
| `test_reflex_compiler.cpython-312-pytest-9.0.2.pyc` | Compiled bytecode |
| `CACHEDIR.TAG` | Cache marker |
| `nodeids` | pytest cache |
| `README.md` | pytest cache readme |
| `test_reflex_compiler.py` | Already at `tests/core/sovereign/test_reflex_compiler.py` |
| `CLAUDE-CODE-EXECUTE-NOW.md` | One-shot session directive |
| `TERMINAL-INTEGRATION-DIRECTIVE.md` | One-shot session directive |
| `terminal_information_architecture.md` | Superseded by `_terminal/` specs |
| `node0_terminal_mission_loop.md` | Superseded by Phase 57 spec |
| `BIZRA-Terminal-v1-Locked-Build-Contract.md` | Already in specs |
| `BIZRA-ddagi-pilot-v2.0.md` | Already loaded as agent mode |
| `BIZRA-Universal-Installer-Spec-v1.0.md` | Superseded by v2.0 |
| `BIZRA-Universal-Installer-Spec-v1.1.md` | Superseded by v2.0 |

---

## 9. ReflexCompiler Upgrade Detail

The season version adds 5 capabilities absent from the codebase version:

| Feature | Season (658 LOC) | Codebase (477 LOC) | Migration |
|---------|------------------|---------------------|-----------|
| HHMM engine integration | `HHMMEngine` protocol + `predict_state()` | None | Add protocol, wire into `_hash_input()` |
| Evidence recorder | `EvidenceRecorder` protocol | None | Add protocol, wire into `compile_from_candidate()` |
| Gossip export | `export_gossip_payload()` | None | New method, no conflicts |
| Forest import | `import_from_forest()` | None | New method, no conflicts |
| Revalidation | `revalidate_entry()` | None | New method, no conflicts |

**Migration strategy:** The season version's class structure is a SUPERSET.
Diff the two files, apply the additions as patches, preserve the codebase's
`OrderedDict` fix (which the season version lacks).

---

## 10. Execution Schedule

| Week | Artifacts Integrated | Tests Added |
|------|---------------------|-------------|
| 2 | reflex_compiler upgrade, subscribers.py, schemas | 36 |
| 3 | bloom.py, bizra_test.py, conftest_tiers.py | 18 |
| 4 | Terminal TSX (9 files) | 9 (Vitest) |
| 5 | genesis_gate.py, empirical_validation.py, lifecycle_proof.py | 20 |
| 6 | bizra-cli.py, sovereign_terminal.py, installers | 12 |
| 7 | Constitutional docs + sprint plans (archive) | 0 |
| 8 | CI pipeline comparison, pyproject merge, cleanup | 5 |

**Total:** 56 files processed, ~7,678 LOC integrated, ~100 new tests.

---

## 11. Standing on Giants

- **Brooks (1975):** "Plan to throw one away" — archive the design artifacts, build from specs
- **Deming (1986):** PDCA — this manifest IS the Plan; integration IS the Do
- **PMBOK (2021):** Organizational process assets — season archive is retained knowledge
- **IEEE 830:** Traceability — every artifact maps to a target path and test anchor
