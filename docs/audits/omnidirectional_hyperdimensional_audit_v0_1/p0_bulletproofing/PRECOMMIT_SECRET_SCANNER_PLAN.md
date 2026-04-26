# Pre-commit Secret Scanner Plan

**Goal:** wire `tools/audit/omni_audit/secret_pattern_scanner.py` as a pre-commit gate so that future drift is caught at commit time, not discovered in audits.

**Prerequisite:** `ROTATION_AND_CLEANUP_PLAN.md` Part C (scanner tuning) must land first — otherwise the 30 existing informational findings cause noise on every commit.

---

## 1. Design principles

- **Fast.** Pre-commit hooks must run in seconds on a `git diff` scope, not a full-repo scan.
- **Redacted.** The hook must never print matched values to stdout/stderr — redacted preview only.
- **Fail-closed.** Any HIGH / MEDIUM finding blocks the commit. INFORMATIONAL does not.
- **Overrideable.** Operator can bypass with `SKIP=bizra-secret-scan git commit ...` for explicit exceptions. Every skip is logged to `.git/hooks/skip.log`.
- **Self-excluding.** Scanner cannot match its own regex literals (Part C1).

## 2. Hook modes

| Mode | Trigger | Scope | Severity block |
|---|---|---|---|
| `pre-commit` | Local commit | Changed files only (`git diff --cached --name-only`) | HIGH + MEDIUM |
| `pre-push` | Local push | Changed files since last push | HIGH + MEDIUM |
| `ci` | GitHub Actions | Full repo | HIGH + MEDIUM + regression baseline |

## 3. Proposed wiring

### 3.1 Add a small CLI entry point to the scanner

Add to `tools/audit/omni_audit/secret_pattern_scanner.py`:

```python
def cli_main(argv=None):
    """Standalone CLI for pre-commit usage.

    Args:
      --paths PATH [PATH ...]   files to scan (else read from stdin)
      --format text|json        output format
      --block-severity HIGH|MEDIUM  exit-code threshold (default MEDIUM)

    Exit: 0 if no blocking findings, 1 if blocking, 2 if invocation error.
    """
    ...
```

This keeps the main scanner API unchanged and adds a thin wrapper for hook use.

### 3.2 `.pre-commit-config.yaml` entry

```yaml
repos:
  - repo: local
    hooks:
      - id: bizra-secret-scan
        name: BIZRA secret-pattern scanner (redacted)
        entry: python3 -m tools.audit.omni_audit.secret_pattern_scanner --format text --block-severity MEDIUM --paths
        language: system
        types: [text]
        pass_filenames: true
        stages: [commit, push]
        exclude: |
          (?x)^(
            tools/audit/omni_audit/secret_pattern_scanner\.py|
            \.claude/logs/.*|
            \.tmp_prod_artifacts_v2/.*|
            tests/fixtures/.*
          )$
```

The `exclude` mirrors the scanner's own `ALWAYS_SKIP_PARTS` for belt-and-suspenders safety.

### 3.3 CI stage wiring (GitHub Actions)

In `.github/workflows/ci.yml` — extend **Stage 6 Security** with:

```yaml
- name: Secret-pattern scanner
  run: |
    python3 -m tools.audit.omni_audit.secret_pattern_scanner \
      --format json \
      --block-severity MEDIUM \
      > secret_findings.json
  # Fails the build if exit code != 0
```

Plus a regression-baseline check: diff against a committed `audit_baseline.json` so newly introduced findings fail the build even if they're INFORMATIONAL-class.

## 4. Operator UX

- **First-time setup:** `pip install pre-commit && pre-commit install` (standard).
- **Normal commit:** no visible change unless a match is found.
- **Match found:** hook prints `{path}:{line} {pattern_class}  redacted_preview=[REDACTED:N]` + decision hint (`classify via docs/audits/omni_audit_v0_1/p0_bulletproofing/SECRET_TRIAGE_REGISTER.json`).
- **Explicit exception:** add `# bizra-secret-scan: allow <pattern_class>` to the line OR prepend `SKIP=bizra-secret-scan` for a one-off.
- **False positive:** update scanner tuning (Part C) and re-commit.

## 5. Audit trail

Every run appends one line to `.git/hooks/secret_scan.log`:

```
{ "ts": "<ISO8601>", "mode": "pre-commit|pre-push|ci",
  "files_scanned": N, "findings": N,
  "blocked": true|false, "commit": "<sha-placeholder>" }
```

Never write matched values to the log. Redacted-preview only.

## 6. Deliverables (when this plan executes — NOT in this pass)

1. `tools/audit/omni_audit/secret_pattern_scanner.py` — add `cli_main()` + Part C tuning.
2. `.pre-commit-config.yaml` — add `bizra-secret-scan` hook.
3. `.github/workflows/ci.yml` — add secret-scan step to Stage 6.
4. `docs/audits/omnidirectional_hyperdimensional_audit_v0_1/p0_bulletproofing/SECRET_SCAN_BASELINE.json` — committed baseline (post-Part-B cleanup) for regression detection.
5. Update `CLAUDE.md` with hook notes.

Each of those is an individual, reviewable change. None is done in this pass.

## 7. Why this is P0 but the wiring is P0+1

**P0 triage** = know what's there (this pass, done).
**P0 cleanup** = Part B dev-default refactor (small, local, ~1–2 h).
**P0 tuning** = Part C regex tightening (~1 h).
**P0+1 wiring** = the pre-commit + CI integration (this document) — belongs in a follow-up because it changes `.github/workflows/ci.yml` which has implications beyond just this lane.

## 8. Exit criteria

The secret-scanner lane is "done" when:

- [ ] Part B: 4 dev-default fallbacks refactored.
- [ ] Part C: scanner tuning landed.
- [ ] Re-run of audit engine shows ≤ 5 findings, all INFORMATIONAL or explicitly allow-listed.
- [ ] `.pre-commit-config.yaml` entry added and tested on a trivial commit.
- [ ] CI stage 6 enforces scanner + baseline regression.
- [ ] `SECRET_SCAN_BASELINE.json` committed.

At that point, any future secret introduction is blocked at commit time.
