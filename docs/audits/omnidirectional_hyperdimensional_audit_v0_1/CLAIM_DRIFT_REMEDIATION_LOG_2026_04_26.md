# Claim-Discipline Drift Remediation Log — 2026-04-26

**Status:** docs-only remediation, runtime-evidence-backed, instrumentation retained until user confirms removal.
**Scope:** top-level masterpiece briefs only.
**Audit branch:** `prep/node0-closure-receipt-lineage` (pre-merge).
**Probe:** `tools/audit/claim_drift_probe.py` (read-only, no canon ingestion, no network).

---

## 1. Why this log exists

The `bizra-pilot` self-evaluation flagged an overclaim in `ULTIMATE_MASTERPIECE_EXECUTIVE_BRIEF.md` ("Status: READY FOR PRODUCTION") as a BLOCKER for PR #58 because it contradicts the project's own claim-discipline framework.

This remediation was performed under Debug-mode discipline: no fix without runtime evidence. The evidence is an NDJSON log produced by a read-only probe that scans the 24 most-touched docs against the patterns derived from `p1_truth_integrity/PROHIBITED_CLAIMS_REGISTER.csv` and `p1_truth_integrity/NEEDS_REWRITE_REGISTER.csv`.

---

## 2. Hypothesis battery and outcome

| ID | Hypothesis | Outcome | Evidence |
|----|------------|---------|----------|
| H1 | Top-level briefs carry explicit production-ready / guaranteed / trustless / AGI / world-first language without truth-label qualifiers. | CONFIRMED (10 true positives in 3 files). | Probe-initial log, lines 2–11. |
| H2 | C-class numerics (SNR 0.974, $0.10→$0.008, 100% pass, 73/100 nodes) live in docs without linked receipts. | INCONCLUSIVE (probe regex too narrow; however the top-3 briefs do carry unreceipted `2.2x`, `112x`, `352ms→160ms`, `8→900 req/sec`, `60–80% cache hit`, `0.95 Ihsan`, which the added banner contextualizes). | No positive log lines. |
| H3 | Origin Kernel §6.3 discipline drift (Kernel treated as ingested / runtime canon). | REJECTED. Both log hits are inside `CANON_STORE_INGESTION_GATE_DESIGN.md` §1 `Non-authorization` and §15 `Stop Line`, which negate the pattern rather than assert it. | Probe-initial log, lines 32–33; verified by reading source at `docs/audits/omnidirectional_hyperdimensional_audit_v0_1/CANON_STORE_INGESTION_GATE_DESIGN.md:6` and `:307–319`. |
| H4 | A single doc carries both the legacy "Node0 proves the seed can live alone…" sentence and the Topology Canon "Each human node mints PAT-7…" sentence. | REJECTED (no hits). | Probe-initial log, no H4 entries. |
| H5 | `PILOT_EVIDENCE_REGISTER.md` or related docs upgrade a `MEASURED_LOCAL_ARTIFACT` to a cross-device / multi-node `MEASURED` claim. | REJECTED (all 6 hits are inside `PLANNED`, `NO-GO`, `unproven`, `Red Lines`, or `until proven` contexts; cross-checked against source). | Probe-initial log, lines 12, 15, 18, 19, 22, 25; verified against `STATUS.md:83`, `INVESTOR_OPERATOR_HANDOVER.md:63–78`, `BUSINESS_MODEL_AND_PRICING_OPTIONS.md:87–94`, `PRODUCTION_READINESS_AND_GTM_CLOSURE_SPRINT.md:57, 175`. |

No rejected-hypothesis code changes accumulated; only the H1-driven patches were applied.

---

## 3. Lines patched (runtime-evidence backed)

All patches are docs-only, confined to three files at the repo root. No canon, no code, no schema, no runtime state.

| File | Line (pre-fix) | Before | After |
|------|----------------|--------|-------|
| `ULTIMATE_MASTERPIECE_EXECUTIVE_BRIEF.md` | 8 | `Status: READY FOR PRODUCTION` | `Status: HISTORICAL ASPIRATIONAL DRAFT — NOT A PRODUCTION RUNTIME` |
| `ULTIMATE_MASTERPIECE_EXECUTIVE_BRIEF.md` | 327 | `✓ Byzantine fault tolerance guaranteed` | `✓ Byzantine fault tolerance: design target (not independently verified)` |
| `ULTIMATE_MASTERPIECE_EXECUTIVE_BRIEF.md` | 434 | `READY FOR PRODUCTION. READY TO SCALE. READY TO TEACH.` | `Not a production runtime. Single-node architecture documented; private pilot is the next verifiable milestone. See the claim-discipline registers.` |
| `ULTIMATE_MASTERPIECE_MANIFESTO.md` | 41 | `✅ 0.95 Ihsān compliance (production-ready)` | `✅ 0.95 Ihsān compliance: DESIGN TARGET (PREPARATION; not a production runtime)` |
| `ULTIMATE_MASTERPIECE_MANIFESTO.md` | 331 | `└─ Byzantine tolerance guaranteed` | `└─ Byzantine tolerance: design target (not independently verified)` |
| `ULTIMATE_MASTERPIECE_MANIFESTO.md` | 362 | `└─ 0.95 Ihsān compliance (production ready)` | `└─ 0.95 Ihsān compliance: DESIGN TARGET (PREPARATION; not a production runtime)` |
| `ULTIMATE_MASTERPIECE_MANIFESTO.md` | 532 | `Ready for production NOW.` | `Not a production runtime now. Private pilot is the next verifiable milestone.` |
| `ULTIMATE_MASTERPIECE_POLYMATH_SYNTHESIS.md` | 573 | `✓ Byzantine tolerance guaranteed (consensus theory)` | `✓ Byzantine tolerance: design target informed by consensus theory (not independently verified)` |
| `ULTIMATE_MASTERPIECE_POLYMATH_SYNTHESIS.md` | 599 | `✓ 0.95 Ihsān compliance (production ready)` | `✓ 0.95 Ihsān compliance: DESIGN TARGET (PREPARATION; not a production runtime)` |
| `ULTIMATE_MASTERPIECE_POLYMATH_SYNTHESIS.md` | 634 | `Ready for production NOW.` | `Not a production runtime now. Private pilot is the next verifiable milestone.` |

In addition, each of the three files received a top-of-file **Claim-Discipline Banner** (HTML comment + visible Markdown blockquote) that declares the document as HISTORICAL ASPIRATIONAL DRAFT, not a measured production-readiness report, and links to the authoritative claim registers.

---

## 4. Proof-of-Truth Convergence for this remediation

- **Formal.** The patterns under test are derived from the already-committed `p1_truth_integrity/PROHIBITED_CLAIMS_REGISTER.csv` and `NEEDS_REWRITE_REGISTER.csv`. The probe is a deterministic regex scan with no hidden state.
- **Cryptographic.** Evidence is persisted as NDJSON at `.cursor/debug-c98f9f.log`; each line carries `sessionId`, `runId`, `timestamp`, `hypothesisId`, `location`, and the raw matched line for reproducibility by any reviewer running the same probe.
- **Empirical.** Line-count falsifier: pre-fix 34, post-fix 24, delta exactly 10, matching the 10 patched lines. Zero residual hits in the three target files in the post-fix run.
- **Economic.** No production cost paid for this fix (docs-only). Public-claim liability reduced, because the three most-likely-to-be-quoted files no longer present a `READY FOR PRODUCTION` banner that would contradict the project's own `CLAIM_DISCIPLINE_FOR_NODE0_AND_URP.md`.

---

## 5. What was NOT changed (by design)

- No edit to `docs/canon/BIZRA_ORIGIN_KERNEL.md`.
- No edit to `MEMORY.md`.
- No edit to Rust or Python runtime canon stores.
- No edit to any public-website file (that remains gated by the website patch plan in `p1_truth_integrity/`).
- No new runtime dependency. The probe uses only Python stdlib (`re`, `json`, `pathlib`, `time`).
- No rewrite of the aspirational voice of the three briefs; only the overclaim lines flagged by the probe plus a context banner.

---

## 6. Instrumentation status

- `tools/audit/claim_drift_probe.py` is retained until the user confirms removal of debug-session instrumentation. Because it is read-only, deterministic, and has no external effects, it may also remain as a permanent claim-discipline CI check if desired. That decision is left to the user.
- Log path `.cursor/debug-c98f9f.log` is re-created on each probe run and is not tracked by git.

---

## 7. Residual findings worth escalating (not fixed in this log)

These are the 20 remaining log lines in the post-fix run. All are expected-good discipline rather than true overclaims, but they deserve a labeled note:

- `STATUS.md:83` — `multi-node ordering unproven` (explicit disclaimer inside a risk table). KEEP.
- `docs/architecture/BIZRA_NODE0_TO_URP_ECOSYSTEM_TRANSITION_v0_1.md:166-167` — `❌ Not AGI`, `❌ Not "world-first"` (explicit negation). KEEP.
- `docs/gtm/node0_activation_go_to_market_v0_1/README.md:37` — `Multi-node URP | PLANNED | … not proven yet`. KEEP.
- `docs/gtm/node0_activation_go_to_market_v0_1/PRODUCTION_READINESS_AND_GTM_CLOSURE_SPRINT.md:57, 149, 175` — `PLANNED … not proven`, stop-line list, `cross-device proof without overstating production`. KEEP.
- `docs/gtm/node0_activation_go_to_market_v0_1/INVESTOR_OPERATOR_HANDOVER.md:5, 68, 71` — framing sentence and `## Red Lines` list. KEEP.
- `docs/gtm/node0_activation_go_to_market_v0_1/BUSINESS_MODEL_AND_PRICING_OPTIONS.md:71, 89, 91` — `Directional until proven`, `## Claims to Avoid in Sales`. KEEP.
- `docs/gtm/node0_activation_go_to_market_v0_1/CLAIM_DISCIPLINE_FOR_NODE0_AND_URP.md:48, 49, 50, 62` — the register itself, quoting prohibited phrasings in context. KEEP.
- `docs/audits/omnidirectional_hyperdimensional_audit_v0_1/CANON_STORE_INGESTION_GATE_DESIGN.md:6, 312` — `Non-authorization` and `Stop Line`. KEEP.

Future probe hardening (non-blocking): make the probe aware of `## Red Lines`, `## Claims to Avoid in Sales`, `## Stop Line`, `### Non-authorization`, and `❌ Not …` sections, so these false positives are suppressed automatically. Out of scope for this remediation because the current false-positive rate is auditable by eye and does not contaminate the signal.

---

## 8. Proposed next logical step (for explicit operator approval)

Two options, pick one:

- **A. Minimal close-out.** Commit this remediation log plus the three patched briefs plus the probe onto the current branch, then re-mark PR #58 ready-for-review with a pointer to this log as the claim-discipline evidence.
- **B. Upgrade to CI gate.** Add a GitHub Action that runs `tools/audit/claim_drift_probe.py` on every PR and fails the build if any H1 finding lands in a file that is NOT in an allowlist (the registers, stop-line docs, and anti-claim documents listed in §7). This converts claim-discipline from an author-time habit into a mergetime gate.

Both options are docs/CI-only. Neither mutates canon, runtime state, or website claims. Authorization required before execution.

---

## 9. Option B executed — 2026-04-26 addendum

Operator chose to proceed with professional peak implementation, authorizing Option B. This section records the activation of the claim-discipline CI gate.

### 9.1 Probe hardening for CI use

`tools/audit/claim_drift_probe.py` gained four additions that preserve backward compatibility:

| Flag / feature | Purpose |
|----------------|---------|
| `--ci` | Fail-closed mode. Exits non-zero when any H1 or H4 finding lands in the CLEAN_SET. |
| `--log-path PATH` | Override for NDJSON output (CI writes into `artifacts/claim-drift/`). |
| `--summary-json PATH` | Machine-readable verdict for downstream dashboards. |
| `--verbose` | Prints every WATCH_SET finding to stdout for CI log inspection. |
| `--run-id` | Tags NDJSON records for this run (CI uses `ci-${GITHUB_SHA::8}`). |
| `<!-- claim-probe: allow -->` | Line-scoped suppression marker for verbatim register quotations. Use sparingly. |

The probe now partitions its targets into two explicit sets:

- **CLEAN_SET** (gated) — the three masterpiece briefs asserted claim-clean by §3 of this log: `ULTIMATE_MASTERPIECE_EXECUTIVE_BRIEF.md`, `ULTIMATE_MASTERPIECE_MANIFESTO.md`, `ULTIMATE_MASTERPIECE_POLYMATH_SYNTHESIS.md`. Any H1/H4 hit here fails CI.
- **WATCH_SET** (report-only) — 21 supporting files (strategy, GTM, audit, business). Findings are logged to NDJSON and printed in verbose output but never fail the build; these files legitimately *quote* prohibited phrases while explaining the discipline rule.

This partition lets the gate start at zero false positives today while keeping claim-drift telemetry across the whole doc corpus.

### 9.2 CI wiring

Inserted as Gate 6 in the `layer2-python-sovereign` job of `.github/workflows/canonical-validation-gate.yml`, directly after Gate 5 (Truth Label Enforcement). Two steps:

1. **Gate 6: Claim-Discipline Drift Probe (CLEAN_SET fail-closed)** — runs `tools/audit/claim_drift_probe.py --ci --verbose --log-path artifacts/claim-drift/findings.ndjson --summary-json artifacts/claim-drift/summary.json --run-id "ci-${GITHUB_SHA::8}"`.
2. **Gate 6: Upload claim-drift artifacts** — `if: always()` upload of the NDJSON log and `summary.json` as the run artifact `claim-drift-${{ github.run_id }}`, 30-day retention. Evidence is retained even on gate failure.

Rationale for placement: `canonical-validation-gate.yml` already enforces constants-vs-docs coherence (Gate 5). Claim discipline is the docs-vs-rule coherence counterpart. Co-locating the two gates keeps the canon-integrity surface in one workflow.

### 9.3 Local verification evidence (pre-merge proof)

All four gate behaviors were verified on branch `prep/node0-closure-receipt-lineage` against the current working tree. Logs written to `.cursor/debug-c98f9f.log` under `runId: "gate-verify"` and `"regression-test"`.

| Test | Command | Expected rc | Actual rc | Verdict |
|------|---------|-------------|-----------|---------|
| G1 | `python3 tools/audit/claim_drift_probe.py --ci` | 0 | 0 | PASS — CLEAN_SET 0 gating findings |
| G2 | `python3 tools/audit/claim_drift_probe.py --verbose --ci` | 0 | 0 | PASS — 22 WATCH_SET findings reported, gate untouched |
| G3 | `python3 tools/audit/claim_drift_probe.py` (default debug) | 0 | 0 | NDJSON grew by 10,424 bytes to `.cursor/debug-c98f9f.log` |
| G4 | `python3 tools/audit/claim_drift_probe.py --ci --log-path /tmp/probe_ci.ndjson` | 0 | 0 | 10,519 bytes written to the explicit path |
| R0 | Baseline `--ci` on clean tree | 0 | 0 | PASS |
| R1 | `--ci` after injecting `"READY FOR PRODUCTION"` into a CLEAN_SET file | 1 | 1 | FAIL, correctly flags the injected line |
| R2 | `--ci` with the injected phrase tagged `<!-- claim-probe: allow -->` | 0 | 0 | Suppression works on a single line |
| R3 | `--ci` after restoring the file from backup | 0 | 0 | PASS |
| R4 | `git diff --stat` on the mutated file after R3 | empty | empty | No residue |

Outcomes: the gate **provably passes on the current clean tree** (R0, G1), **provably fails on a regression** (R1), **respects per-line suppression** (R2), and **returns to green after revert** (R3, R4). The supporting NDJSON log lines are retained under session `c98f9f`.

### 9.4 What is NOT changed by this CI gate

- No runtime code touched.
- No existing CI gate is weakened or removed; Gate 6 is strictly additive.
- No change to `docs/canon/BIZRA_ORIGIN_KERNEL.md`, `MEMORY.md`, or any Rust/Python constant.
- No change to the website patch plan or to any public site copy.
- WATCH_SET files are not forced to rewrite anything; they are only monitored. Future expansion of CLEAN_SET requires an explicit follow-up PR that first cleans the target file.

### 9.5 Operator note — promoting a file into CLEAN_SET

1. Manually review every H1 / H4 finding reported by `python3 tools/audit/claim_drift_probe.py --verbose` for that file.
2. Either rewrite each finding to a truth-labeled form, or annotate the specific line with `<!-- claim-probe: allow -->` **only** when the prohibited phrase is a verbatim register quotation.
3. Move the file from `WATCH_SET` to `CLEAN_SET` in `tools/audit/claim_drift_probe.py`.
4. Re-run `python3 tools/audit/claim_drift_probe.py --ci` locally; it must return 0 before committing.
5. Reference this §9 in the commit message as the gate-expansion authority.

No file should be added to `CLEAN_SET` without this sequence. The gate is only meaningful if CLEAN_SET is kept credible.

---

## 10. Post-landing fix — H4 suppression semantics (2026-04-26)

A reviewer flagged a mismatch between the module docstring ("any line containing `claim-probe: allow` is exempted from gating") and the H4 coexistence check, which was gating on a **file-scope** marker test (`ALLOW_MARKER not in text`). This meant that a `claim-probe: allow` marker placed on *any* line anywhere in a file suppressed the H4 coexistence finding, even when neither of the two canonical sentences carried the marker on its own line.

### 10.1 Runtime evidence (debug session `c98f9f`, runId `h4-bug-probe`)

Six synthetic fixtures were built and fed to `scan_file`:

| Case | Fixture | Expected H4 | Pre-fix | Post-fix |
|------|---------|-------------|---------|----------|
| A | both canonical sentences, no marker | 1 | 1 | 1 |
| B | marker on an unrelated line, sentences clean | 1 | **0 (BUG)** | **1** |
| C | marker on the H4_A line only | 0 | 0 | 0 |
| D | marker on both H4 sentence lines | 0 | 0 | 0 |
| E | H1 per-line control (`READY FOR PRODUCTION`) | 1 | 1 | 1 |
| F | only H4_A present (no H4_B) | 0 | 0 | 0 |

Pre-fix NDJSON proof: `.cursor/debug-c98f9f.log` line 2 (Case B: `"findings": []`).
Post-fix NDJSON proof: same log, runId `h4-post-fix`, Case B: `H4 line=3 matched=double-canonical`.

### 10.2 Fix

Replaced the two file-scope predicates with a helper that collects unsuppressed line numbers per pattern and fires H4 only if both sets are non-empty:

```python
def _unsuppressed_line_matches(pattern: re.Pattern, lines: List[str]) -> List[int]:
    matches: List[int] = []
    for lineno, line in enumerate(lines, 1):
        if ALLOW_MARKER in line:
            continue
        if pattern.search(line):
            matches.append(lineno)
    return matches

legacy_hits    = _unsuppressed_line_matches(H4_A, lines)
canonical_hits = _unsuppressed_line_matches(H4_B, lines)
if legacy_hits and canonical_hits:
    # emit H4 with data={"legacy_lines": ..., "canonical_lines": ...}
```

Suppression semantics are now consistent across H1, H2, H3, H4, and H5: a line carrying `claim-probe: allow` is exempt for that occurrence only; no file-scope suppression exists. The H4 finding also now reports the line number of the first unsuppressed legacy-sentence match (previously `line: 0`), making it easier to locate in CI logs.

### 10.3 Impact on the live tree

Running the full CI invocation (`--ci --verbose --log-path … --summary-json …`) against the current branch before and after the fix yields the **same** verdict (PASS) and the **same** counts (CLEAN_SET 3 files / 0 findings / 0 gating; WATCH_SET 21 files / 22 findings). No production file relied on the buggy file-scope behavior, so the fix has zero observable impact on the current gate while tightening its specification.

---

## 11. Audit v0.3 — activation of the canonical truth pack (2026-04-26)

This pass closes two hidden-golden-gem gaps surfaced by the Omnidirectional pass v0.3:

### 11.1 G1 — `sys.modules` registration defect in `scripts/generate_canonical_truth_pack.py`

The generator loads `core/sovereign/api_exposure_policy.py` dynamically via
`importlib.util.spec_from_file_location` + `module_from_spec` + `exec_module`. On
Python 3.12, `@dataclass(frozen=True)` internally calls
`sys.modules.get(cls.__module__).__dict__`; when the module is not registered in
`sys.modules`, the lookup returns `None` and fails with
`AttributeError: 'NoneType' object has no attribute '__dict__'`.

**Reproduction (pre-fix):** `python3 scripts/generate_canonical_truth_pack.py`
→ traceback at `core/sovereign/api_exposure_policy.py:30`.

**Fix:** register the dynamic module in `sys.modules` before `exec_module`,
with best-effort cleanup on exception. The fix is strictly scoped to
`_load_api_policy_module`.

**Regression lock:** `tests/scripts/test_generate_canonical_truth_pack.py::test_load_api_policy_module_resolves_dataclass_classes`.

### 11.2 G2 — probe contract had zero unit tests

Gate 6 (claim-drift probe) had been load-bearing since commit `5699173b` but
lacked any unit tests; the H4 file-scope-suppression bug slipped past review
because the probe's contract lived only in comments.

**Closed by:** `tests/tools/audit/test_claim_drift_probe.py` — 26 tests
covering H0/H1/H4 firing rules, per-line suppression (including the explicit
regression lock for the file-scope bug fixed in §10), `LogSink` behaviour,
`_resolve_log_path` policy, and end-to-end `main()` CI exit codes.

### 11.3 G3 — canonical truth pack was orphaned

The 151-line generator committed as `dc732128` was functional (once G1 was
fixed) but unwired: no committed JSON artifact, no CI reference, no docs
consumer. Thresholds were still copied by hand into ~20 markdown files.

**Closed by three co-ordinated changes:**

1. **Artifact committed:** `docs/knowledge/canonical_truth_pack.json` —
   machine-readable snapshot of volatile-but-governed facts
   (IHSAN / SNR / ADL Gini thresholds; public / bootstrap / authenticated
   route counts; Rust workspace crate count; workflow file count).

2. **Gate 7 wired into `.github/workflows/canonical-validation-gate.yml`:**
   regenerates the pack in CI, diffs against the committed copy, fails the
   build on any drift, and uploads the diff as a 30-day artifact. Pass- and
   fail-path both verified locally.

3. **Deterministic generation proof:**
   `test_write_truth_pack_is_deterministic` asserts two back-to-back
   generations produce byte-identical output — the property Gate 7 relies on.

### 11.4 Outcomes at a glance

| Metric | Before this pass | After this pass |
|---|---|---|
| Truth-pack generator | crashes on `@dataclass` | runs clean |
| Truth-pack JSON in repo | missing | committed |
| CI freshness enforcement | none | Gate 7 fail-closed |
| Claim-drift probe tests | 0 | 26 |
| Truth-pack generator tests | 0 | 12 |
| Pytest lines added | 0 | 38 tests, <0.3 s runtime |
| Ruff findings on new code | — | 0 |

### 11.5 Evidence pointers

* Generator fix: `scripts/generate_canonical_truth_pack.py:61-79`
* Truth-pack JSON: `docs/knowledge/canonical_truth_pack.json`
* Gate 7 YAML: `.github/workflows/canonical-validation-gate.yml` (§"Gate 7: Canonical Truth Pack Freshness")
* Tests: `tests/tools/audit/test_claim_drift_probe.py`, `tests/scripts/test_generate_canonical_truth_pack.py`

No constitutional claim or public statement was modified by this pass; the
delta is pure enforcement infrastructure plus a previously-absent evidence
artifact.
