# scripts/e2e-polyglot — Cycle-6 G4 Polyglot E2E Test

بسم الله الرحمن الرحيم

**Status:** GREEN — real end-to-end test. Replaces the intentional-red scaffold as of 2026-04-17.

## What it proves

Per `cycle-6/niyyah.md` §G4:

> "`scripts/e2e-polyglot/` contains the full-stack smoke test; one CI workflow runs it on every push; the test proves a real receipt sealed through the polyglot chain."

The harness exercises the polyglot vertical:

```
bash harness → HTTP (Rust gateway v0.2) → admissibility chain (5 gates) →
receipt artifact → in-memory ReceiptChain → HTTP read-back
```

## Pre-conditions (all satisfied)

- **G1 persistence** — ✅ live-verified (`cycle-6/g1-live-verification.md`)
- **G2 gateway authority** — ✅ sealed (`cycle-6/g2-authority-adr.md`)
- **G3 frontend authority** — ✅ sealed (`cycle-6/g3-authority-adr.md`)

## Why gateway-direct, not through the external Next.js proxy

G3 ADR designates `award-winner-design` as the authoritative operator face. That repo is external; CI runs in this repo. Making every push clone an external repo is fragile. This harness is deliberately **gateway-direct** so it runs in any CI without external dependency. External-proxy verification is a **disaster-recovery drill** concern covered by `docs/ROLLBACK-RUNBOOK-Cycle-5.md`, not per-push CI.

That split is itself part of the G3 three-tier rollback model: the external Next.js is the production operator face; the gateway-direct path is the always-available substrate that the dema CLI already uses and the test harness exercises.

## 8 assertions

| # | Assertion | Why |
|---|---|---|
| 1 | Gateway boots + `/health` responds | Infrastructure alive |
| 2 | `/chain` starts empty (length=0) | Clean initial state |
| 3 | `POST /mission` with quality 0.98 returns `verdict: Permit` | Full 5-gate admissibility PASS |
| 4 | Response includes 64-char hex `receiptId` | Receipt was sealed |
| 5 | `/chain` length advanced | In-memory chain updated atomically |
| 6 | `GET /chain/{receiptId}` returns receipt metadata | Round-trip read works |
| 7 | `POST /mission` with quality 0.50 returns HTTP 422 | Fail-closed on IHSAN_FLOOR violation |
| 8 | `GET /chain/{fff...}` (unknown) returns HTTP 404 | Fail-closed on unknown hash |

## Environment

- `BIZRA_E2E_PORT` — custom port (default 7431) to avoid clashes with a running local dev gateway
- `BIZRA_E2E_SKIP_BUILD=1` — skip cargo build if gateway binary is already present (CI sets this after its own build step)

## Local run

```bash
just build-rust              # or: cargo build --release -p bizra-cognition-gateway (bizra-omega/)
bash scripts/e2e-polyglot/test.sh
```

Expected: `═══ RESULTS: 8 passed / 0 failed ═══` → exit 0.

## CI

See `.github/workflows/e2e-polyglot.yml`. Runs on every push / pull_request to `main`. Intentional-red-on-red has been retired; any future failure is a **real regression** to investigate, not a visible pressure gauge.

## Exit codes

- `0` — all 8 assertions passed
- `1` — ≥1 assertion failed (real regression)
- `2` — infrastructure failure (gateway won't boot, port conflict, binary missing)

## References

- Cycle-6 niyyah §G4: `cycle-6/niyyah.md`
- G1 persistence: `cycle-6/g1-authority-adr.md`, `cycle-6/g1-live-verification.md`
- G2 gateway authority: `cycle-6/g2-authority-adr.md`
- G3 frontend authority: `cycle-6/g3-authority-adr.md`
- CI workflow: `.github/workflows/e2e-polyglot.yml`
