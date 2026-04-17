# Downloads/files (7) — 2026-04-17 Ingest Inventory

بسم الله الرحمن الرحيم

**Source:** `~/Downloads/files (7)/` — 28 files, 556 KB
**Ingested:** 2026-04-17 Dubai GST
**Action posture:** **Archive-first, halt-on-integration.** Per cycle-6/execution-canon.md hard rule ("No widening scope beyond G1"), nothing in Category A / A' / B was merged into canonical paths this pass. All 28 files preserved here for founder-gated decision.

## Critical context

**These files reflect a DIFFERENT Cycle-6 plan than the one sealed on origin.**

- Their `cycle-6-execution-spec.md` niyyah: *"First real impact receipt on Mumo's Downloads folder via MCP tool transport"*
- The origin-committed `cycle-6/niyyah.md` names: *"Persistence + Authority Unification"*

The Downloads folder appears to be an alternate or earlier parallel-session draft that introduces scope-widening modules: `trust_compiler.rs`, `lawful_loop.rs`, `gateway_v03_compile.rs`, `dema_cli_v02_organize.rs`, plus a new E2E test harness and CI workflow. Merging them blindly would fork the canon.

## Classification matrix

Category legend:
- **A** — fork / scope-widening (new kernel modules, new CI) that differ from sealed cycle
- **A'** — fork-framing docs (different niyyah, landing guide, manifest version, handover)
- **B** — variant of existing canonical Rust file (origin already has it, downloads differs)
- **C** — historical / reference / adjunct (past retros, audits, assets, installers)
- **D** — identical to in-repo canon (skip; in-repo is authoritative)

### Category A — fork, scope-widening code (6 files)

| File | Size | Notes |
|---|---|---|
| `trust_compiler.rs` | 28,563 B | New kernel module — generalizes `submit_mission()` into universal trust compilation. Header cites "Cycle: 6 — first impact receipt" (different Cycle-6 plan). |
| `lawful_loop.rs` | 18,943 B | §6 End-to-End Connector + §16 Minimum Undeniable Loop. Not currently in kernel; would be a new module. |
| `gateway_v03_compile.rs` | 8,540 B | Gateway v0.3 compilation scaffold (current on origin is v0.2). |
| `dema_cli_v02_organize.rs` | 9,769 B | Dema CLI v0.2 `organize` subcommand (MCP tool transport surface). |
| `e2e-trust-compiler-test.sh` | 9,801 B | End-to-end test: CLI → Gateway → TrustCompiler → FilesystemExecutor → ReceiptChain. |
| `trust-compiler-e2e.yml` | 2,426 B | GitHub Actions workflow for above. Targets `bizra-omega/bizra-cognition/src/trust_compiler.rs` path. |

**Integration requires founder gate** — these introduce a parallel architecture to the sealed G1 Phase 1/2 landed on origin.

### Category A' — fork, alternate-cycle framing docs (4 files)

| File | Size | Notes |
|---|---|---|
| `cycle-6-execution-spec.md` | 5,996 B | Different Cycle-6 niyyah ("First real impact receipt on Downloads folder via MCP"). Not superseding origin's `cycle-6/execution-canon.md` + `g1-*.md` canon chain. |
| `cycle-6-landing-guide.md` | 3,763 B | Landing guide for the Downloads-impact cycle variant. |
| `manifest-5.md` | 2,904 B | Possible Manifesto v5 draft. Origin has doctrine sealed at `docs/dema-cli-manifesto-v1.md` (commit 8b7adec9). |
| `HANDOVER.md` | 10,485 B | Alt handover for different scope. Origin has `docs/BIZRA-Handover-v1.md` + `docs/BIZRA-Repo-Inventory-v1.md`. |

**Integration requires founder gate.**

### Category B — variants of canonical kernel files (4 files)

All four have in-repo counterparts in `bizra-omega/bizra-cognition/src/`. Sizes differ by 2–1105 bytes. **Copying over the canonical versions risks regressing the 98 green G1 Phase 1+2 tests.**

| File | Size diff vs in-repo | Action |
|---|---|---|
| `admissibility_freeze_v1.rs` | 34,436 (−7 B) | Diff-check required |
| `eval_v1_integrated.rs` | 27,755 (−102 B) | Diff-check required |
| `eval_v1.rs` | 39,410 (−1,105 B) | Diff-check required — largest delta |
| `mission_freeze_v1.rs` | 19,871 (+2 B) | Trivial diff likely whitespace |

**Recommendation:** Do NOT overwrite in-repo versions without running full cognition test suite against the variant and verifying no regression. Founder gate.

### Category C — historical / reference / adjunct (11 files)

| File | Size | Proposed destination |
|---|---|---|
| `al-mithaq-al-tasisi.md` | 12,837 B | **PROMOTED this pass** → `docs/al-mithaq-al-tasisi.md` (founding covenant, new) |
| `cycle-3-retrospective.md` | 16,523 B | **SKIPPED** — `cycle-3/retrospective.md` already exists with different content; diff check needed |
| `cycle-5-retrospective.md` | 2,969 B | **SKIPPED** — `cycle-5/retrospective.md` already exists; variants differ |
| `cycle-5-phase-7-retrospective.md` | 5,104 B | Archive only — references Cycle-5 phase 7 (not in current cycle-5/ structure) |
| `g2-patches-abc.md` | 7,450 B | Archive only — already in cycle-5 G2-hardening-acceptance-note.md history |
| `bizra_audit_15-April-2026.pdf` | 20,706 B | Archive only — audit artifact (PDF) |
| `bizra_peak_synthesis_cycle_2.pdf` | 41,832 B | Archive only — synthesis PDF |
| `bizra_peak_synthesis_cycle_2.py` | 67,319 B | Archive only — synthesis script |
| `bizra_constitutional_audit.py` | 34,734 B | Archive only — audit script |
| `install.sh` | 8,954 B | Archive only — repo already has `scripts/install.sh` + `UNIFIED-NODE-INSTALLER/installers/linux/install.sh` |
| `dema-overlay.jsx` | 32,689 B | Archive only — React component for Dema overlay; needs frontend architecture review |

### Category D — identical to canon (3 files)

| File | In-repo location | Action |
|---|---|---|
| `manifest_artifact.rs` | `bizra-omega/bizra-cognition/src/manifest_artifact.rs` | Skip (identical) |
| `receipt_freeze_v1.rs` | `bizra-omega/bizra-cognition/src/receipt_freeze_v1.rs` | Skip (identical) |
| `why-dema-wins.md` | `docs/why-dema-wins.md` | Skip (already present) |

## Summary

| Category | Count | Action taken this pass |
|---|---|---|
| A (fork code) | 6 | Archived; founder gate needed |
| A' (fork docs) | 4 | Archived; founder gate needed |
| B (variants) | 4 | Archived; diff-check + test-regression gate needed |
| C (historical/reference) | 11 | Archived; 1 promoted (`al-mithaq-al-tasisi.md` → `docs/`) |
| D (identical) | 3 | Archived for completeness; skipped in-repo moves |
| **Total** | **28** | **All preserved in this archive** |

## Why archive-first

The `/home/bizra-operating-system/Downloads/files (7)/` bundle has high signal AND high canon-fork risk. The elite move is:
1. Preserve everything (discoverable, non-destructive)
2. Classify everything (per-file actionability)
3. Halt on scope-widening (respect G1 canon just sealed on origin)
4. Promote only clear-fit additions (al-mithaq-al-tasisi founding covenant)
5. Leave the rest as a staging area for founder-gated decisions

This matches the cycle's Path-1 discipline applied at ingest-time.

## Decision gates queued for founder

| Gate | Scope | Suggested command |
|---|---|---|
| **A1** Trust Compiler integration | Merge `trust_compiler.rs` + `lawful_loop.rs` as NEW cognition modules | Needs Cycle-7 niyyah amendment or separate cycle opening |
| **A2** Gateway v0.3 + dema organize | Upgrade gateway + add `dema organize` MCP surface | Needs niyyah scope expansion |
| **A3** E2E trust-compiler test + CI | Requires A1 first (test targets trust_compiler.rs) | Follows A1 |
| **B1** Variant Rust diff review | Diff 4 variant files; decide per-file keep/overwrite | Low-risk docs-only pass if identical-after-diff |
| **C1** cycle-3/5 retrospective diff | Compare variant retros vs in-repo; merge or defer | Docs-only |
| **C2** Dema overlay JSX wiring | Integrate React component into frontend architecture | Needs G3 (frontend authority) first |
| **C3** Audit scripts (`bizra_constitutional_audit.py`, synthesis) | Wire into `just audit-constitution` recipe | Independent arc |

## Provenance

- Original location: `/home/bizra-operating-system/Downloads/files (7)/`
- All file modification times preserved via `cp` (timestamps in archive match originals)
- Original folder retained by user; this archive is a repo-committed copy, not a move

الحمد لله.
