# scripts/e2e-polyglot — Cycle-6 G4 scaffold

بسم الله الرحمن الرحيم

**Purpose:** End-to-end polyglot smoke test. Proves a real principal-activation receipt seals through the full Rust gateway → external Next.js proxy → internal Vite loop, as specified in `cycle-6/niyyah.md` §G4.

**Status:** SCAFFOLD — intentionally red until Cycle-6 G4 closes.

## Why red now, by design

Per DevOps BOK (make-the-unfinished-visible): the red CI run on `e2e-polyglot` workflow is the visible pressure gauge that Cycle-6 G4 is open. Every commit until Cycle-6 closes will see the red reminder. **Do not "fix" this by making `test.sh` return 0 without implementing the real contract below** — that would be NO_SHADOW_STATE violation by CI spoofing.

## Contract (from niyyah §G4)

> "`scripts/e2e-polyglot/` contains the full-stack smoke test; one CI workflow runs it on every push; the test proves a real receipt sealed through the polyglot chain."

Verification on green CI:
- `bizra-cognition-gateway` binary starts cleanly
- mission POST through external Next.js proxy returns 200 with real receipt
- `dema chain --since today` reads the sealed receipt via the **persistent** chain (requires G1)
- All 5 admissibility gates (ZANN_ZERO, CLAIM_MUST_BIND, RIBA_ZERO, NO_SHADOW_STATE, IHSAN_FLOOR) verdicted

## Pre-conditions (must close before G4 passes)

- **G1 persistence** — chain survives gateway restart (otherwise `dema chain` is ephemeral and this test is meaningless)
- **G2 gateway authority** — ✅ SEALED by `cycle-6/g2-authority-adr.md` (gateway under test = `bizra-cognition-gateway`)
- **G3 frontend authority** — determines whether the test uses external `award-winner-design` or in-repo `frontend/` as proxy path

## What lives here now (pre-G4)

- `README.md` — this file
- `test.sh` — placeholder that exits 1 with scope message

## What will live here at G4 close

- `test.sh` — real end-to-end script (bash orchestration)
- receipt verification logic (signature check against `sovereign_state/key_registry.json`)
- gateway lifecycle (spawn / health-check / teardown)
- optional Python variant (`test.py`) for polyglot coverage of the harness itself

## References

- Cycle-6 niyyah: `cycle-6/niyyah.md` §G4
- G2 ADR (gateway under test): `cycle-6/g2-authority-adr.md`
- Prototype walk-through: `/tmp/g4-mumo-walk.sh` (Cycle-5, ephemeral — to be promoted)
- CI workflow: `.github/workflows/e2e-polyglot.yml`
