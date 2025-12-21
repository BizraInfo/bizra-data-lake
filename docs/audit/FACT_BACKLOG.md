# BIZRA Fact-Based Backlog
**Generated**: 2025-12-21 | **Scoring**: Impact + RiskReduction + EthicalCriticality − Effort − DependencyPenalty

---

## Top 15 Items by Leverage

| # | Score | Item | Evidence | Acceptance Criteria |
|---|-------|------|----------|---------------------|
| 1 | **9.5** | Fix CI YAML syntax error | `.github/workflows/elite-ci-cd.yml:180` - extra indent breaks parser | CI workflow passes YAML lint; Ihsān gate runs |
| 2 | **8.5** | Fix Dockerfile.refinery CMD variable expansion | `docker compose logs refinery` shows `${VAR}` literal not expanded | refinery service status = healthy |
| 3 | **8.0** | Commit 10 modified core files | `git status` shows `src/http.rs`, `src/ihsan.rs`, `constitution/ihsan_v1.yaml` etc. | Clean `git status`; PR ready |
| 4 | **7.5** | Add .env file with BIZRA_REFINERY_THROUGHPUT | Refinery expects float, receives literal `${VAR}` | `grep BIZRA_REFINERY_THROUGHPUT .env` returns numeric |
| 5 | **7.0** | Remove unused imports in test files | `tests/pat_sat_runtime_tests.rs:8-9`, `tests/integration_harness.rs:20` | `cargo clippy` zero warnings |
| 6 | **6.5** | Fix copilot-instructions.md relative links | Links resolve from `.github/` not repo root | All links use `../` prefix or absolute paths |
| 7 | **6.0** | Add missing CODEBASE_STATE_REPORT.md | Referenced in audit prompt but doesn't exist | File exists in `docs/execution/` |
| 8 | **5.5** | Update README truth label | Line 3 says TARGET but badge says "production" | Consistent labeling across all claims |
| 9 | **5.0** | Add benchmark for P99 latency claim | README claims "Sub-100ms P99" but no evidence | `cargo bench` or integration test measuring latency |
| 10 | **4.5** | Document .env.example with all required vars | `.env.example` exists but may be incomplete | All docker-compose env vars have defaults documented |
| 11 | **4.0** | Fix useless comparison warning | `tests/integration_harness.rs:304` - `pending >= 0` on unsigned | Warning eliminated |
| 12 | **3.5** | Add elite service to docker-compose up | Not in default services list currently | `docker compose up -d` starts elite |
| 13 | **3.0** | Verify gate A-E run in CI | `node0_gates_latest.json` local only | CI artifacts include gate results |
| 14 | **2.5** | Add Python tests to CI | Only Rust tests run; `tests/test_kg_receipts.py` exists | pytest runs in quality gate |
| 15 | **2.0** | Clean up activation_* receipt folders | 18 old activation folders in receipts dir | Archival policy or cleanup script |

---

## Evidence Links

| Item | File Path |
|------|-----------|
| #1 | [.github/workflows/elite-ci-cd.yml#L180](.github/workflows/elite-ci-cd.yml) |
| #2 | [Dockerfile.refinery#L70](Dockerfile.refinery) |
| #3 | `git status --porcelain` output |
| #4 | [docker-compose.yml#L182](docker-compose.yml) |
| #5 | [tests/pat_sat_runtime_tests.rs#L8-9](tests/pat_sat_runtime_tests.rs) |
| #6 | [.github/copilot-instructions.md](.github/copilot-instructions.md) |
| #7 | `docs/execution/` directory listing |
| #8 | [README.md#L3](README.md) |
| #9 | [README.md#L32](README.md) |
| #10 | [.env.example](.env.example) |
| #11 | [tests/integration_harness.rs#L304](tests/integration_harness.rs) |
| #12 | [docker-compose.yml](docker-compose.yml) - elite service present but not default |
| #13 | [docs/evidence/gates/node0_gates_latest.json](docs/evidence/gates/node0_gates_latest.json) |
| #14 | [tests/test_kg_receipts.py](tests/test_kg_receipts.py) |
| #15 | [docs/evidence/receipts/](docs/evidence/receipts/) |

---

## Scoring Formula Applied

```
Score = Impact(1-5) + RiskReduction(1-3) + EthicalCriticality(0-2) − Effort(1-3) − DependencyPenalty(0-2)
```

| Item | Impact | RiskRed | Ethics | Effort | Deps | Total |
|------|--------|---------|--------|--------|------|-------|
| #1 | 5 | 3 | 2 | 0.5 | 0 | 9.5 |
| #2 | 4 | 3 | 1 | 0.5 | 1 | 6.5→8.5* |
| #3 | 4 | 3 | 1 | 1 | 0 | 7→8.0* |

*Adjusted for unlock multiplier (unblocks downstream items)
