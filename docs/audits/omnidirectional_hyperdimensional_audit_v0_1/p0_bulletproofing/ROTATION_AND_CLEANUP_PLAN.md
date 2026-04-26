# Rotation & Cleanup Plan

**Input:** `SECRET_TRIAGE_REDACTED.md` + `SECRET_TRIAGE_REGISTER.json`.
**Triage verdict:** 0 real secrets. **Nothing to rotate.** This plan is for hygiene and future-drift prevention, not incident response.

---

## Part A — Rotation (nothing to rotate)

**No credentials require rotation.** The 35 scanner matches decompose into:

- 0 real production credentials.
- 4 dev-default credentials (localhost fallbacks — see Part B).
- 31 false positives / placeholders / env-substitution templates / detection-event log / scanner self-reference.

No remove-from-git-history operation is required. No BFG / `git filter-repo` run is needed.

## Part B — Dev-default credential cleanup (5 sites, 4 files)

These committed localhost-only fallback credentials should be refactored to **fail-fast without a fallback** (or point to `.env.example`). This is an anti-pattern cleanup, not a rotation.

| # | File | Line | Change |
|---|---|---:|---|
| B1 | `runtime/tools/kg_seed_from_concept_graph.py` | 12 | In docstring `Usage:`, replace literal dev DSN with `export BIZRA_PG_DSN="$(cat .env.local \| grep BIZRA_PG_DSN)"` style reference, or point to `.env.example`. |
| B2 | `runtime/tools/kg_seed_from_concept_graph.py` | 36 | Remove the string fallback from `os.environ.get(...)` — fail-fast with `sys.exit(1)` + error message naming the required env var. |
| B3 | `runtime/core/autoconfig.py` | 330 | Remove fallback; raise `RuntimeError("DATABASE_URL required")`. |
| B4 | `runtime/core/pci/receipt_store_persistent.py` | 1148 | Same — remove fallback, raise on missing env. |
| B5 | `runtime/config/substrate_v1.yaml` | 19 | Remove `default_dsn`; require operator to provide via env. |

**Dev ergonomics preserved via a new `.env.example` file** that documents the expected environment variables with placeholder values, checked in. Actual values live in `.env.local` (git-ignored).

**Effort:** 1–2 hours total (5 edits, 1 new `.env.example`, 1 entry in `.gitignore` if not already excluded).

## Part C — Scanner tuning (P0.3 prerequisite)

Tune the secret scanner to reduce expected noise before wiring it as a pre-commit gate.

| # | Change | File | Effort |
|---|---|---|---|
| C1 | Add scanner **self-exclusion** for `tools/audit/omni_audit/secret_pattern_scanner.py` | `tools/audit/omni_audit/secret_pattern_scanner.py` (ALWAYS_SKIP_PARTS) | XS |
| C2 | Add `.claude/logs/` to `ALWAYS_SKIP_PARTS` | same | XS |
| C3 | Add `.tmp_prod_artifacts_v2/` to `ALWAYS_SKIP_PARTS` | same | XS |
| C4 | Respect `# nosec B105` / `# nosec B106` markers in DOTENV_LIKE matches (skip flagged line) | same | S |
| C5 | Tighten `DOTENV_LIKE` regex so values that are **string constants** (`= "sha256:"`, `= "apiKey"`, `= "/etc/..."`) are not flagged; require at least one non-letter / non-underscore char or `{` / `[` | same | S |
| C6 | Add an **allow-list** mechanism (line comment `# scanner:allow <class>`) | same | S |

After C1-C6, re-running the scanner should collapse the 35 findings to **~4–5** items (the dev-default credentials + possibly 1–2 genuine edge cases). This is the target noise level for a pre-commit gate.

## Part D — Optional log-hygiene policy

The `.claude/logs/audit.jsonl` file stores `user_prompt` previews verbatim. Even though the S0015 match was a false positive, the pattern means the log *could* in the future store a real leaked value if the user pasted one.

**Optional enhancement (NOT P0, not required):**

- Pre-save redaction in the audit hook that writes `prompt_preview`: run the secret-scan patterns against the preview before persisting; replace matches with `[REDACTED:<class>:<len>]`.
- This is a separate lane; no file is modified in this pass.

## Part E — `.gitignore` / scope review

| Check | State | Action |
|---|---|---|
| `.env`, `.env.local`, `.env.*.local` in `.gitignore`? | Not verified in this pass | Add if missing |
| `.claude/logs/` in `.gitignore`? | Depends on project policy | Depends — logs are per-machine artifacts; usually yes |
| `.tmp_prod_artifacts_v2/` is gitignored? | Unclear (present on disk + scanned) | Add if temp-generated |
| `deploy/node0/.env.local` (referenced in `scripts/start_mission_bridge.sh:25`) | Assumed per convention | Verify existence + gitignore |

**Action:** run a quick grep for `^\.env` in `.gitignore`; add missing patterns in a single `.gitignore` edit (out of scope for this P0 pass; schedule for cleanup).

## Part F — No destructive actions in this pass

This plan documents intended changes. **Nothing has been modified in this pass.** Specifically:

- No source file was edited.
- No log record was deleted or rewritten.
- No git history was rewritten.
- No credential was rotated (none real).
- No CI / pre-commit config was installed yet.

All proposed changes in Parts B and C are **independent, reviewable edits** — each is ~1–30 lines — that the operator can choose to execute, defer, or discard.

---

## Summary by severity and proposed action

| Severity | Count | Action category |
|---|---:|---|
| HIGH | 0 | — |
| MEDIUM | 4 | Refactor dev-default fallbacks (Part B) |
| LOW | 1 | Log-scan exclusion (Part C2) |
| INFORMATIONAL | 30 | Scanner tuning (Part C) reduces noise |

## Execution order (when authorized)

1. Apply Part C (scanner tuning) first — no behavioral change to the runtime, collapses the noise floor.
2. Apply Part B (dev-default refactor) second — small blast radius, local dev impact only.
3. (Optional) Part D — log-hygiene policy — separate lane.
4. Re-run audit engine; expect ≤ 5 findings.
5. Wire pre-commit scanner (see `PRECOMMIT_SECRET_SCANNER_PLAN.md`).
