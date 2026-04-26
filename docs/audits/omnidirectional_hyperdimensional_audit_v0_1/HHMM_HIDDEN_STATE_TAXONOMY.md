# HHMM Hidden-State Taxonomy — BIZRA v0.1

**Definition:** 4-level hierarchical hidden-state taxonomy derived from audit findings.

- **Level 0 — Domain state:** high-level audit domain.
- **Level 1 — Subsystem state:** specific subsystem under that domain.
- **Level 2 — Failure mode / opportunity state:** the observed condition.
- **Level 3 — Action state:** evidence-backed next action + transition trigger.

**Source:** `artifacts/hhmm_taxonomy.json`. Counts by domain:

| Domain | Count |
|---|---:|
| SECURITY | 1 |
| DEPENDENCY | 3 |
| CODE_QUALITY | 5 |
| PUBLIC_CLAIMS | 6 |
| DOCUMENTATION | 2 |

---

## Domain: SECURITY

### Subsystem: secrets

| Failure mode / opportunity | Action | Transition trigger |
|---|---|---|
| No matches from secret-pattern scanner in configured scan roots | Expand scan roots / add CI gate for ongoing coverage | Scanner gate runs in pre-commit or CI and current artifact remains at 0 findings |

## Domain: DEPENDENCY

### Subsystem: lockfiles

| Failure mode / opportunity | Action | Transition trigger |
|---|---|---|
| `filedfs/Cargo.toml` has no Cargo.lock | Generate and commit Cargo.lock | `cargo generate-lockfile` in that workspace |
| `desktop/rust/Cargo.toml` has no Cargo.lock | Generate and commit Cargo.lock | same |
| SBOM artifact not located anywhere in repo | Emit SBOM on release | CI pipeline adds SBOM step |

## Domain: CODE_QUALITY

### Subsystem: RS_UNWRAP

| Failure mode / opportunity | Action | Transition trigger |
|---|---|---|
| 806 `.unwrap()` sites → panic surface on hot paths | Triage hot paths; replace with `Result` + explicit error receipts | Audit log of hot-path vs cold-path classification complete |

### Subsystem: PY_SHELL_TRUE

| Failure mode / opportunity | Action | Transition trigger |
|---|---|---|
| 1 `subprocess(shell=True)` — command-injection surface | Remove; add lint rule forbidding new sites | Lint rule green in CI |

### Subsystem: PY_BROAD_EXCEPT

| Failure mode / opportunity | Action | Transition trigger |
|---|---|---|
| 126 broad `except Exception` — may mask errors | Sweep; tighten exception classes | Coverage of tightened paths in CI |

### Subsystem: RS_TODO / PY_TODO

| Failure mode / opportunity | Action | Transition trigger |
|---|---|---|
| 1 Rust TODO/FIXME and 4 Python TODO/FIXME markers present | Triage; convert into tickets or remove | Backlog merged in `IMPLEMENTATION_BACKLOG.md` |

## Domain: PUBLIC_CLAIMS

### Subsystem: prohibited

| Failure mode / opportunity | Action | Transition trigger |
|---|---|---|
| 20 PROHIBITED-class claim patterns (AGI, first-in-world, benchmark-superiority) | Rewrite or remove each before any public reuse | Rewrite PR merged on bizra.ai source |

### Subsystem: needs_rewrite

| Failure mode / opportunity | Action | Transition trigger |
|---|---|---|
| 94 NEEDS_REWRITE patterns (production-ready, exact cost, SNR number, 100% pass, manufactured scarcity) | Remove from hero; move to under-the-hood page with receipts | `bizra.ai` hero replaced per `CLAIM_SAFE_LAUNCH_COPY.md` |

### Subsystem: proof_required

| Failure mode / opportunity | Action | Transition trigger |
|---|---|---|
| 367 PROOF_REQUIRED patterns (Ed25519, BLAKE3, Ihsan threshold, no-telemetry, post-quantum) | Publish receipt per claim OR soften to directional | Receipt index published at `bizra.ai/receipts/` or claim softened |

### Subsystem: website_rendering

| Failure mode / opportunity | Action | Transition trigger |
|---|---|---|
| bizra.ai SPA returns a shell to non-JS fetchers | Add OG meta tags in shell HTML; consider SSR/prerender | Link preview renders hero image + description |
| bizra.info redirects to bizra.ai; destination shell remains weak for non-JS fetchers | Keep redirect; improve destination shell metadata | bizra.info preview inherits corrected bizra.ai metadata |

### Subsystem: redirects

| Failure mode / opportunity | Action | Transition trigger |
|---|---|---|
| bizra.info 302 → bizra.ai confirmed; no split surface | None (keep brand-defense redirect) | — |

## Domain: DOCUMENTATION

### Subsystem: doctrine_surface_area

| Failure mode / opportunity | Action | Transition trigger |
|---|---|---|
| 132 doctrine-class documents present | Index + deduplicate; single forward path is Canon Store Ingestion Gate | Gate spec drafted |

### Subsystem: agent_instructions

| Failure mode / opportunity | Action | Transition trigger |
|---|---|---|
| Top-level CLAUDE.md stable | Review quarterly; keep in sync with module decomposition | Quarterly review completed |

---

## Traversal hints

Read this taxonomy top-down when triaging work:

1. Start at the highest-signal domain (PUBLIC_CLAIMS or DEPENDENCY).
2. Pick the subsystem with clearest evidence.
3. Execute the listed action.
4. Verify via the transition trigger.
5. Re-run audit; the row either drops, weakens, or reveals a new subsystem.

If the audit is re-run and a subsystem doesn't shrink despite action, the action was wrong or incomplete — surface as a new finding.
