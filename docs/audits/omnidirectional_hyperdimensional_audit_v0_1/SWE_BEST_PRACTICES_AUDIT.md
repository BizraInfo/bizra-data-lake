# Software Engineering Best Practices Audit — BIZRA v0.1

**Scope:** module boundaries, testability, typing / schema discipline, CI/CD readiness, coding standards, duplication, repo hygiene.

---

## 1. Module boundaries

- **Python:** `core/` has ~58 subpackages. CLAUDE.md explicitly prefers decomposed modules (governance, reasoning, orchestration, treasury) over the monolithic `sovereign/` equivalents. Enforcement: convention.
- **Rust:** 25 crates in `bizra-omega/` workspace (v2.0.0) with 8 layers (Platform, Cognitive, Action, TTRL, Desktop, Mission, Numeric, Protocol). Clean layering.
- **Frontend:** React + Vite, phase state machine, 6-tab dashboard. Clean.

**Assessment:** ✅ Boundaries are well-drawn. Python's `core/sovereign/` is the one known-loose module — decomposition in progress.

## 2. Testability

- **Python:** pytest with markers (`slow`, `integration`, `requires_ollama`, etc.); asyncio_mode `auto`; 60 s per-test timeout. Coverage floor 65%.
- **Rust:** 1 603+ tests across bizra-omega. Cross-language parity tests for the 5 constitutional objects (246 tests).
- **Frontend:** Vitest with watch mode + CI single-run.

**Assessment:** ✅ Testing infra is mature.

**Issue:** `cargo test --workspace` contaminates `action_receipts.jsonl` — operational hygiene concern; documented workaround exists.

## 3. Typing / schema discipline

- **Python:** PEP 484 type hints throughout; mypy in CI (relaxed for `core.*` / `tests.*`, strict elsewhere).
- **Rust:** `#[derive(...)]` types + `ts-rs` for frontend interop. Memory note `project_ts_rs_serde_rename_drift.md` records a historical drift bug pattern — known issue class.
- **Python-Rust bridge:** PyO3 via `bizra-python/`. Gated by `VIRTUAL_ENV` + maturin build step.

**Assessment:** ✅ Strong typing surface in both languages. The cross-language interop is where drift lives.

## 4. CI/CD readiness

Per CLAUDE.md: 24 gates across 7 stages:

1. Lint (ruff, black, isort, mypy / cargo fmt, clippy)
2. Schema + Sync (cross-language constants sync)
3. Test (pytest matrix 3.11/3.12 + cargo test + PyO3 smoke)
4. Frontend (lint, typecheck, test, build, bundle budget)
5. Quality Gates (FATE, MVSA, Phase65, Phase56 security)
6. Security (bandit, pip-audit, cargo-audit, Trivy, SBOM, container signing)
7. Docker Build (deploy/Dockerfile.elite, bizra-omega/Dockerfile)

All GitHub Actions pinned by SHA. No `|| true` on security-critical gates.

**Assessment:** ✅ **Mature CI/CD.** Stage 6 calls out SBOM — this audit found no SBOM artifact in repo. Either CI generates but doesn't commit, or the step is present but inactive. Verify.

## 5. Coding standards

- Ruff primary linter; ignores `E402` (deferred imports) and `E501` (black).
- `# noqa: SEC-001` marks intentional legacy SHA-256 (BLAKE3 gate).
- Bare `except:` forbidden; convention enforced.

**Assessment:** ✅ Consistent.

## 6. Duplication

- **Cognitive Foundry canon packs:** 5 packs on disk for the same origin run — all honest snapshots. Explicit, documented, not noise.
- **Media kit:** ready_to_post/ is intentional raster exports of editable_svg/ templates. Not duplication.
- **Code:** no duplication audit performed by this pass. Recommend a targeted `similarity`-style scan for large copy-paste blocks across `core/` (which has been subject to decomposition).

## 7. Repo hygiene

| Check | State |
|---|---|
| `.gitignore` present | ✅ |
| Large binaries in repo | Some (media kit 42 MB — acceptable as an asset pack, not in core runtime paths) |
| Secret sweep | 0 current secret-pattern matches (see `SECURITY_AUDIT.md`) |
| Dead code / unused dirs | Not audited |
| Git history cleanliness | Not audited |
| Branch protections | Not visible from repo scan |

## 8. Repo-hygiene notes observed

- `.venv-linux` + `.venv` coexist. CLAUDE.md explicitly pins `.venv-linux` for WSL; `.venv` is Windows. Clear. ✅
- `bizra-omega/bizra-python/python/bizra/` gitignored; `__init__.py` must be force-added. Per CLAUDE.md. ✅
- `native/` deprecated; all Rust development in `bizra-omega/`. Per CLAUDE.md. ⚠️ suggest removing `native/` or adding a DEPRECATED.md to its root.

## 9. SWE-practice debts (ranked)

| # | Debt | Severity | Action |
|---|---|---|---|
| SW1 | SBOM verification — is it being generated but not committed? | MEDIUM | Check CI logs; ensure artifact lands as release asset |
| SW2 | `native/` deprecation marker | LOW | DEPRECATED.md in `native/` or removal |
| SW3 | Python strict-pin via `uv pip compile` | MEDIUM | (also in Dependency Audit) |
| SW4 | Hot-path `.unwrap()` audit | MEDIUM | (also in Architecture / Error Handling) |
| SW5 | Code-duplication scan over `core/` | LOW | Targeted `jscpd` / `simian` run |
| SW6 | `cargo test` audit-log contamination guard as CI | MEDIUM | (also in Error Handling) |

---

**SWE verdict:** BIZRA is a well-engineered codebase by most conventional measures — strong typing, strong testing, strong CI, clean module boundaries. The principal tech-debt is panic-surface (unwrap) and supply-chain attestation (SBOM / lockfiles).
