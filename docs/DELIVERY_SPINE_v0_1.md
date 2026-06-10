# BIZRA Delivery Spine v0.1

بسم الله الرحمن الرحيم

**Purpose:** One-page control plane mapping PMBOK delivery areas to **repo-proven** gates. This is an index, not a vision doc. Full program frame: `bizra-omega/docs/BIZRA_UNIFIED_EXECUTION_BLUEPRINT.md`.

**Truth rule:** SHIPPED only when a workflow or command in this repo passes on `main`. ASPIRATIONAL items are named explicitly.

---

## Spine flow

```text
design → implement → verify → attest → observe → learn
```

| Stage | Repo artifact | Status |
|---|---|---|
| Design | `cycle-6/*-authority-adr.md`, `TOPOLOGY_CANON.md` | SHIPPED (Cycle-6 ADRs) |
| Implement | `bizra-omega/` canonical Rust | SHIPPED |
| Verify | `.github/workflows/*`, `Justfile`, `scripts/ci_*` | SHIPPED (multi-gate) |
| Attest | `bizra-omega/evidence/*.json`, Proof Forge (local) | WIRED_PARTIAL |
| Observe | Gateway `/health`, `/chain`; operator `dema` CLI | SHIPPED |
| Learn | `cycle-6/retrospective.md`, witness bumps | WIRED_PARTIAL |

---

## PMBOK → gate map

| PMBOK area | BIZRA enforcement |
|---|---|
| Integration | Single repo; `bizra-omega/` authority ADR; no shadow runtime |
| Scope | Cycle arcs (G1–G4, Arc 3) — one gate per PR where possible |
| Schedule | Merge train via PR; no force-push |
| **Quality** | `Quality Spine`, `Tests`, `Canonical Validation Gate`, `e2e-polyglot.yml` |
| **Resource** | PAT roles in blueprint; operator `dema` as face |
| Communications | ADRs + evidence JSON witnesses |
| Risk | Fail-closed admissibility; corrupt persist store aborts boot |
| Procurement | `cargo audit` / dependabot (known noise on openssl — hygiene lane) |
| Stakeholders | External `award-winner-design` (G3 ADR); gateway-direct CI |

---

## CI/CD workflows (operator-relevant)

| Workflow | Proves |
|---|---|
| `e2e-polyglot.yml` | G4 lawful loop + **Arc 3 restart persistence (test 8)** |
| `canonical-validation-gate.yml` | Cross-stack canon |
| `quality-spine.yml` / `tests.yml` | Rust/Python test matrix |
| `wire-completeness-audit.yml` | Integration wiring |

---

## Local verification (cold agent path)

```bash
cd bizra-omega && cargo fmt --all -- --check
cd bizra-omega && cargo clippy --workspace --all-targets -- -D warnings
cd bizra-omega && cargo test --workspace
bash scripts/e2e-polyglot/test.sh
```

---

## Current arc pointer

**Cycle-6 Arc 3.1** — post-merge operationalization: `cycle-6/arc3-post-merge-canon.md`
