# BIZRA Next Step Decision

**Generated**: 2025-12-21T12:45:00+04:00 (Dubai)  
**Decision type**: Canonical baseline establishment  
**Evidence bundle**: `docs/execution/evidence/v2/`

---

## Selected Step

### Commit and push all changes to establish canonical baseline

**Why this step?**
- Closes the "multiple truth snapshots" drift identified in SAPE analysis
- Triggers CI validation to confirm all gates pass
- Enables downstream work (elite service, pytest, SAPE probes)
- Minimal risk with immediate rollback available

---

## Command Sequence

```bash
# Stage all changes
git add -A

# Commit with descriptive message
git commit -m "fix: resolve CI YAML, Dockerfile.refinery, test warnings, copilot-instructions links

- Fix elite-ci-cd.yml line 180 indentation (YAML parse error)
- Fix Dockerfile.refinery CMD to use shell-form for env expansion
- Remove unused imports in tests (clippy warnings)
- Fix useless comparison in integration_harness.rs
- Fix copilot-instructions.md relative links (../ prefix)
- Add audit deliverables:
  - docs/audit/ (REVIEW_SUMMARY, FACT_BACKLOG, NEXT_STEP_DECISION)
  - docs/execution/evidence/v2/ (fresh truth snapshot)
  - Updated CODEBASE_STATE_REPORT.md (zero UNKNOWNs for build/tests)

Test Evidence:
- Rust: 73/73 tests pass (see docs/execution/evidence/v2/rust_build_test.txt)
- Docker: 5/5 services healthy (see docs/execution/evidence/v2/docker_ps.txt)
- CI YAML: Validated (see docs/execution/evidence/v2/metrics_probe.txt)

BIZRA-Specific:
- Ihsan (ethical excellence score): constitution threshold enforced
- FATE (Fail-safe Agentic Trust Escalation): rejection routing wired
- Receipts: audit trail active for all decisions

See GLOSSARY section in docs/execution/NEXT_STEP_DECISION.md for term definitions."

# Push (branch upstream is gone, need force)
git push origin feature/coderabbit-integration-dual-system --force-with-lease
```

> **⚠️ Branch Verification**: Before pushing, confirm your branch with:  
> ```bash
> git branch -vv  # Should show: * feature/coderabbit-integration-dual-system
> ```

---

## Acceptance Criteria

| Criterion | Verification |
|-----------|--------------|
| Git status clean | `git status --porcelain` returns empty |
| CI workflow triggered | GitHub Actions shows new run |
| Security Gate passes | No secret leaks detected |
| Quality Gate passes | Rust tests + clippy (warnings allowed) |
| Ihsān Gate passes | Score ≥ threshold for env |
| Performance Gate passes | Build time + binary size within bounds |
| Container Gate passes | Docker builds + scans pass |

---

## Rollback Plan

If CI fails or issues arise:

```bash
# Option 1: Soft reset (preserve changes locally)
git reset --soft HEAD~1

# Option 2: Hard reset (discard changes)
git reset --hard HEAD~1

# Option 3: Revert commit (keep history)
git revert HEAD
git push origin feature/coderabbit-integration-dual-system --force-with-lease
```

---

## Dependencies

| Dependency | Status | Notes |
|------------|--------|-------|
| GitHub access | ✅ Ready | origin remote configured |
| Branch exists | ⚠️ Gone | Will be recreated on push |
| CI workflow | ✅ Valid | YAML parses correctly |
| Docker services | ✅ Healthy | 5/5 running |

---

## Post-Commit Next Steps

After successful push and CI run:

1. **Capture CI artifacts** → Add to `docs/execution/evidence/v2/ci_run.txt`
2. **Fix clippy warnings** → Remove `assert!(true)` from integration_harness.rs
3. **Add elite to compose** → Enable `:8080` metrics endpoint
4. **Install pytest** → Add to requirements-kernel.txt
5. **SAPE probes** → Implement 7-12 rare-circuit probes per framework

---

## Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|------------|--------|------------|
| CI gate failure | Low | Medium | Rollback + fix specific gate |
| Force push conflict | Low | Low | `--force-with-lease` protects |
| Missing secrets | Low | High | Template in .env.example |
| Test regression | Very Low | High | 73/73 pass locally |

---

## Decision Rationale (SAPE Alignment)

**Symbolic**: Constitution + gates define what "pass" means  
**Abstraction**: Single commit represents unified truth state  
**Probe**: CI run will probe all 5 gates (rare circuits)  
**Elevation**: If CI passes, this becomes the canonical baseline for all future work

This decision directly implements the **Phase 0 — Converge truth** step from the SAPE framework application.

---

## GLOSSARY: BIZRA-Specific Terms

| Term | Definition | Reference |
|------|------------|----------|
| **Ihsān** (إحسان) | Ethical excellence score — weighted composite across 8 dimensions (correctness, safety, user_benefit, etc.). Threshold: 0.95 production, 0.90 CI, 0.80 dev. | [constitution/ihsan_v1.yaml](../../constitution/ihsan_v1.yaml) |
| **FATE** | Fail-safe Agentic Trust Escalation — handles quarantine, human review routing, rejection receipts when requests fail validation. | [src/fate.rs](../../src/fate.rs), [core/fate.py](../../core/fate.py) |
| **SAPE** | Symbolic-Abstraction Probe Elevation — 9-probe verification system that elevates recurring patterns into optimized shortcuts. | [src/sape.rs](../../src/sape.rs), [core/sape.py](../../core/sape.py) |
| **Receipts** | Append-only audit trail for all decisions. Every PAT/SAT action emits a structured receipt with SHA-256 integrity hash. | [src/receipts.rs](../../src/receipts.rs) |
| **PAT** | Personal Agentic Team — 7 specialized agents for task execution. | [.github/copilot-instructions.md](../../.github/copilot-instructions.md) |
| **SAT** | System Agentic Team — 5 guardian agents for validation (requires 3/5 consensus). | [.github/copilot-instructions.md](../../.github/copilot-instructions.md) |

---

## Evidence Links

| Claim | Evidence File |
|-------|---------------|
| 73/73 Rust tests pass | [rust_build_test.txt](evidence/v2/rust_build_test.txt) |
| 5/5 Docker services healthy | [docker_ps.txt](evidence/v2/docker_ps.txt) |
| CI YAML valid | [metrics_probe.txt](evidence/v2/metrics_probe.txt) |
| Refinery daemon healthy | [refinery_logs.txt](evidence/v2/refinery_logs.txt) |
| Branch identity confirmed | [identity_git.txt](evidence/v2/identity_git.txt) |
