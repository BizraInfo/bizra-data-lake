# BIZRA Fact-Based Backlog (v2)

**Generated**: 2025-12-21T12:45:00+04:00 (Dubai)  
**Evidence bundle**: `docs/execution/evidence/v2/`  
**Scoring**: Impact + RiskReduction + EthicalCriticality − Effort − DependencyPenalty

---

## Top 15 Items by Leverage

| # | Score | Item | Evidence | Acceptance Criteria |
|---|-------|------|----------|---------------------|
| 1 | **9.5** | Commit + push all changes | [identity_git.txt](evidence/v2/identity_git.txt) — 14 modified, 9 untracked | Clean `git status`; CI triggered |
| 2 | **8.0** | Fix 5 clippy `assert!(true)` warnings | [rust_clippy.txt](evidence/v2/rust_clippy.txt) — lines 289-292 | `cargo clippy` zero warnings |
| 3 | **7.5** | Add elite service to default compose | [metrics_probe.txt](evidence/v2/metrics_probe.txt) — :8080 returns 404 | `docker compose up` starts elite |
| 4 | **7.0** | Install pytest + run Python tests | [python_compileall.txt](evidence/v2/python_compileall.txt) — pytest NOT FOUND | `pytest -q` passes |
| 5 | **6.5** | Capture CI run artifacts | [metrics_probe.txt](evidence/v2/metrics_probe.txt) — no logs | GitHub Actions artifacts exist |
| 6 | **6.0** | Fix branch upstream "gone" | [identity_git.txt](evidence/v2/identity_git.txt) — tracking lost | `git push` succeeds |
| 7 | **5.5** | Add benchmark for P99 latency claim | README claims "Sub-100ms P99" | `cargo bench` or k6 results |
| 8 | **5.0** | Document .env.example completeness | docker-compose env vars | All vars have defaults |
| 9 | **4.5** | Add Python tests to CI | `tests/test_kg_receipts.py` exists | pytest runs in quality gate |
| 10 | **4.0** | Template k8s secrets | [security posture in CODEBASE_STATE_REPORT](CODEBASE_STATE_REPORT.md#8-security-posture) | No literal secrets in repo |
| 11 | **3.5** | Add env contract checks | SAPE probe requirement | Startup fails with actionable error if env missing |
| 12 | **3.0** | Add secret hygiene gate | Security lens | CI blocks on detected secrets |
| 13 | **2.5** | Archive old activation receipts | 18 folders in `docs/evidence/receipts/` | Archival policy applied |
| 14 | **2.0** | Add Node subproject tests | `bizra-genesis-node/bridge/` | npm test passes |
| 15 | **1.5** | Replace tree fallback | [tree.txt missing hierarchy](../evidence/tree.txt) | PowerShell native tree |

---

## Evidence Links

| Item | File Path |
|------|-----------|
| #1 | [identity_git.txt](evidence/v2/identity_git.txt) |
| #2 | [rust_clippy.txt](evidence/v2/rust_clippy.txt) |
| #3 | [metrics_probe.txt](evidence/v2/metrics_probe.txt) |
| #4 | [python_compileall.txt](evidence/v2/python_compileall.txt) |
| #5 | [metrics_probe.txt](evidence/v2/metrics_probe.txt) |
| #6 | [identity_git.txt](evidence/v2/identity_git.txt) |
| #7 | [README.md#L32](../../README.md) |
| #8 | [.env.example](../../.env.example) |
| #9 | [tests/test_kg_receipts.py](../../tests/test_kg_receipts.py) |
| #10 | [k8s/base/secrets.yaml](../../bizra-genesis-node/k8s/base/secrets.yaml) |
| #11 | SAPE architecture requirement |
| #12 | Security lens recommendation |
| #13 | [docs/evidence/receipts/](../evidence/receipts/) |
| #14 | [bizra-genesis-node/bridge/](../../bizra-genesis-node/bridge/) |
| #15 | [tree.txt](../evidence/tree.txt) |

---

## Scoring Formula

```
Score = Impact(1-5) + RiskReduction(1-3) + EthicalCriticality(0-2) − Effort(1-3) − DependencyPenalty(0-2)
```

| Item | Impact | RiskRed | Ethics | Effort | Deps | Total |
|------|--------|---------|--------|--------|------|-------|
| #1 | 5 | 3 | 2 | 0.5 | 0 | **9.5** |
| #2 | 4 | 2 | 1 | 0.5 | 0.5 | **6.0** → 8.0* |
| #3 | 4 | 2 | 1 | 1 | 1 | **5.0** → 7.5* |
| #4 | 3 | 3 | 1 | 0.5 | 0 | **6.5** → 7.0* |

*Adjusted for unlock multiplier (unblocks downstream items)

---

## Closed Items (from v1)

| Item | v1 Score | Resolution |
|------|----------|------------|
| Fix CI YAML syntax error | 9.5 | ✅ Fixed line 180 indentation |
| Fix Dockerfile.refinery CMD | 8.5 | ✅ Changed to shell-form |
| Remove unused test imports | 7.0 | ✅ Removed std::fs, Duration |
| Fix copilot-instructions links | 6.5 | ✅ Added ../ prefix |
| Fix useless comparison | 4.0 | ✅ Changed to assert_eq |
