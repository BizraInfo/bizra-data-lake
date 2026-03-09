# BIZRA Quality Management Pipeline — Integration Guide

> Standing on Giants: Deming (PDCA, 1950) · Shewhart (SPC, 1924) · Crosby (Zero Defects, 1979) · PMI/PMBOK 7th Ed (2021)

## Overview

This deliverable implements the **Quality Management Orchestration Engine** — the PMBOK backbone that connects CI gates to CD readiness with continuous ratcheting, trend tracking, and evidence-chained release certification.

### Architecture

```
┌──────────────────────────────────────────────────────────────────────────┐
│                    BIZRA Quality Management Pipeline                      │
├──────────────────────────────────────────────────────────────────────────┤
│                                                                          │
│  COVERAGE RATCHET ──→ QUALITY TREND ──→ PR SUMMARY                      │
│       │                     │                │                           │
│       ▼                     ▼                ▼                           │
│  pyproject.toml         04_GOLD/          PR Comment                     │
│  (fail_under ↑)    quality_trend.jsonl   (sticky)                       │
│                                                                          │
│  ──── On Tag Push ────────────────────────────                          │
│                                                                          │
│  CHANGELOG GEN ──→ RELEASE READINESS ──→ Evidence Ledger                │
│       │                     │                    │                       │
│       ▼                     ▼                    ▼                       │
│  CHANGELOG.md        04_GOLD/             04_GOLD/                      │
│                   release_readiness    coverage_ratchet_log              │
│                      _log.jsonl            .jsonl                        │
└──────────────────────────────────────────────────────────────────────────┘
```

## Files Created

### Core Engine (5 files)

| File | Purpose | Lines |
|------|---------|-------|
| `scripts/ci_coverage_ratchet.py` | Coverage floor auto-ratcheting with multi-language aggregation | ~250 |
| `scripts/ci_release_readiness.py` | Release gate orchestrator (8 gates, weighted scoring) | ~350 |
| `scripts/ci_changelog_gen.py` | Conventional commit parser + Markdown changelog generator | ~300 |
| `scripts/ci_pr_quality_summary.py` | PR comment generator with coverage delta, trend, badges | ~120 |
| `core/devops/quality_trend.py` | Quality snapshot persistence, hash-chaining, trend analysis | ~420 |

### CI Workflow (1 file)

| File | Purpose |
|------|---------|
| `.github/workflows/quality-management.yml` | 4-stage pipeline: Ratchet → Trend → PR Summary → Changelog |

### Test Suites (4 files, ~90 tests)

| File | Tests | Coverage |
|------|-------|----------|
| `tests/scripts/test_ci_coverage_ratchet.py` | 18 tests | XML parsing, floor R/W, ratchet logic, evidence chain, multi-lang |
| `tests/core/devops/test_quality_trend.py` | 20 tests | Snapshots, hash-chaining, store operations, trend analysis |
| `tests/scripts/test_ci_changelog_gen.py` | 22 tests | Commit parsing, changelog generation, Markdown rendering |
| `tests/scripts/test_ci_pr_quality_summary.py` | 10 tests | Summary generation, badge logic, regression warnings |

### Supporting Files (3 files)

| File | Purpose |
|------|---------|
| `core/devops/__init__.py` | Module init |
| `tests/scripts/__init__.py` | Test package init |
| `tests/core/devops/__init__.py` | Test package init |

**Total: 13 files created**

## Required Manual Edits

### 1. Fix missing `sys` import in `core/devops/quality_trend.py`

Add `import sys` to the imports section (after `import statistics`):

```python
# Line ~37, add after 'import statistics':
import sys
```

### 2. Run tests

```bash
# Activate the Linux venv
source .venv-linux/bin/activate

# Run all new tests
pytest tests/scripts/test_ci_coverage_ratchet.py -v
pytest tests/core/devops/test_quality_trend.py -v
pytest tests/scripts/test_ci_changelog_gen.py -v
pytest tests/scripts/test_ci_pr_quality_summary.py -v

# Or all together
pytest tests/scripts/ tests/core/devops/ -v
```

### 3. Git operations

```powershell
# From the repo root
cd C:\BIZRA-DATA-LAKE

git add scripts/ci_coverage_ratchet.py
git add scripts/ci_release_readiness.py
git add scripts/ci_changelog_gen.py
git add scripts/ci_pr_quality_summary.py
git add core/devops/__init__.py
git add core/devops/quality_trend.py
git add .github/workflows/quality-management.yml
git add tests/scripts/__init__.py
git add tests/scripts/test_ci_coverage_ratchet.py
git add tests/scripts/test_ci_changelog_gen.py
git add tests/scripts/test_ci_pr_quality_summary.py
git add tests/core/devops/__init__.py
git add tests/core/devops/test_quality_trend.py
git add docs/QUALITY_MANAGEMENT_GUIDE.md

git commit -m "ci(quality): add Quality Management Orchestration Engine

- Coverage ratchet: auto-raise fail_under floor on coverage gains
- Quality trend: hash-chained JSONL snapshots with SPC trend analysis
- Release readiness: 8-gate weighted orchestrator with evidence receipts
- Changelog generator: conventional commit parser + Markdown renderer
- PR quality summary: sticky PR comment with coverage delta and badges
- CI workflow: 4-stage pipeline (ratchet→trend→summary→changelog)
- 90 tests across 4 test suites

Standing on Giants: Deming (PDCA, 1950) · Shewhart (SPC, 1924)
Constitutional: Ihsān ≥ 0.95, SNR ≥ 0.85, ADL Gini ≤ 0.35"
```

## Tool Usage

### Coverage Ratchet

```bash
# Check (CI default — exits 1 on regression, 0 on pass)
python scripts/ci_coverage_ratchet.py --coverage-xml coverage.xml

# Apply ratchet (post-merge — bumps fail_under in pyproject.toml)
python scripts/ci_coverage_ratchet.py --coverage-xml coverage.xml --apply

# Multi-language aggregation
python scripts/ci_coverage_ratchet.py \
  --coverage-xml coverage.xml \
  --rust-lcov bizra-omega/target/lcov.info \
  --frontend-json frontend/coverage/coverage-final.json \
  --json
```

### Quality Trend Tracker

```bash
# Record a snapshot after CI run
python -m core.devops.quality_trend record \
  --snr 0.92 --coverage 42 --mypy-errors 1580 \
  --tests-total 200 --tests-passed 195

# Analyze trend
python -m core.devops.quality_trend analyze --last 30

# Export for dashboard
python -m core.devops.quality_trend export --format json --output /tmp/trend.json
```

### Release Readiness

```bash
# Full gate check
python scripts/ci_release_readiness.py --commit-sha $(git rev-parse HEAD)

# Fast mode (skip slow gates)
python scripts/ci_release_readiness.py --fast --json

# Strict mode (all warnings become blockers)
python scripts/ci_release_readiness.py --strict
```

### Changelog Generator

```bash
# From last tag
python scripts/ci_changelog_gen.py --from-tag v2.0.0

# Append to CHANGELOG.md
python scripts/ci_changelog_gen.py --from-tag v2.0.0 --append CHANGELOG.md --version v2.1.0

# JSON output
python scripts/ci_changelog_gen.py --from-tag v2.0.0 --json
```

## How It Connects to Existing CI

The new `quality-management.yml` workflow runs **in parallel** with the existing `ci.yml` pipeline. They are complementary:

| Concern | `ci.yml` (existing) | `quality-management.yml` (new) |
|---------|---------------------|-------------------------------|
| Tests | Run + fail on failure | Uses test coverage as input |
| Coverage | Report to Codecov | Ratchet floor + trend tracking |
| Quality | Point-in-time gates | Longitudinal trend analysis |
| Release | Build artifacts | Gate + evidence + changelog |
| PR Feedback | Status checks | Rich Markdown summary comment |

## Evidence Chain

All operations produce append-only, hash-chained JSONL evidence:

- `04_GOLD/coverage_ratchet_log.jsonl` — Every ratchet evaluation
- `04_GOLD/quality_trend.jsonl` — Every quality snapshot (hash-chained)
- `04_GOLD/release_readiness_log.jsonl` — Every release gate evaluation
- `04_GOLD/changelog_evidence.jsonl` — Every changelog generation

This satisfies the constitutional requirement: **Receipts as Truth**.

## PMBOK Alignment

| PMBOK Process Group | Implementation |
|---------------------|---------------|
| Planning | Release readiness gates define quality criteria |
| Executing | CI workflow runs automatically on push/PR |
| Monitoring & Controlling | Quality trend analysis + anomaly detection |
| Closing | Release changelog + evidence receipt |

| PMBOK Knowledge Area | Implementation |
|-----------------------|---------------|
| Quality Management | Coverage ratchet + trend SPC + gate orchestration |
| Risk Management | Regression detection + anomaly alerts |
| Communications | PR summary comments + trend reports |
| Integration Management | Cross-language coverage aggregation |

---

*Standing on Giants: Deming (PDCA quality cycle, 1950) · Shewhart (control charts, 1924) · Crosby (Zero Defects, 1979) · PMI/PMBOK 7th Ed (Quality Management, 2021) · Shannon (signal theory, 1948) · Al-Ghazali (Ihsān, 1095)*

*Evidence: `scripts/ci_coverage_ratchet.py`, `core/devops/quality_trend.py`, `.github/workflows/quality-management.yml`*
