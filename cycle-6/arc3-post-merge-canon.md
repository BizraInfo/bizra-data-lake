# Cycle-6 — Arc 3 Post-Merge Canon (Operational Closure)

بسم الله الرحمن الرحيم

**Arc:** 3 — Authoritative receipt chain persistence (`BIZRA_RECEIPT_STORE_PATH`)
**Merge:** PR #108 → `main` @ `37b8d114` (2026-06-10)
**Status:** MERGED — Arc 3.1 closes the CI proof gap (G4 test 8)

---

## What shipped (Arc 3)

| Invariant | Behavior |
|---|---|
| `BIZRA_RECEIPT_STORE_PATH` unset | In-memory chain — ephemeral across restarts |
| Explicit path | sled payloads + `chain_snapshot.json` under store root |
| `BIZRA_RECEIPT_STORE_PATH=default` | Opt-in operator canonical path (never implicit) |
| Corrupt store | Fail-closed at gateway bootstrap |

**Code:** `bizra-omega/bizra-cognition/src/receipt_chain_store.rs`, gateway `sled-store` feature enabled.

**Remote witness (pre–Arc 3.1):** `bizra-omega/evidence/CYCLE6_ARC3_PERSISTENCE_REMOTE_WITNESSED.json`

---

## Arc 3.1 — what this pass adds

1. **G4 test 8** — `scripts/e2e-polyglot/test.sh` proves restart survival with a temp `BIZRA_RECEIPT_STORE_PATH` on every push (workflow `e2e-polyglot.yml`).
2. **Delivery spine index** — `docs/DELIVERY_SPINE_v0_1.md` maps PMBOK areas to repo gates (no new vision; wiring only).
3. **Witness bump** — evidence artifact updated to post-merge `main` + G4 assertion.

---

## Operator golden path

**CI (G4 test 9):** `scripts/operator-smoke-arc3.sh` with `BIZRA_RECEIPT_STORE_PATH=default` and isolated `BIZRA_DATA_LAKE_ROOT` — invoked from `scripts/e2e-polyglot/test.sh` on every push.

**Local (authoritative store):**

```bash
cd bizra-omega && cargo build --release -p bizra-cognition-gateway

export BIZRA_RECEIPT_STORE_PATH=default
export BIZRA_DATA_LAKE_ROOT=/data/bizra   # optional anchor
./target/release/bizra-cognition-gateway

# separate terminal — seal + verify
dema mission submit --intent "operator smoke" ...
dema chain
# restart gateway; dema chain must show same head
```

**Local (isolated, no shared store):**

```bash
BIZRA_OPERATOR_SMOKE_ISOLATED=1 bash scripts/operator-smoke-arc3.sh
```

Local deep witness (optional): `/data/bizra/logs/node0-persist-witness-final-20260610-v3/run_witness.py`

---

## Verification commands (minimum undeniable loop)

```bash
# Unit + integration (omega workspace)
cd bizra-omega && cargo test -p bizra-cognition receipt_chain_store

# G4 polyglot (includes Arc 3 test 8 after Arc 3.1)
bash scripts/e2e-polyglot/test.sh

# Remote gates (on PR)
# — E2E Polyglot (Cycle-6 G4)
# — Canonical Validation Gate
# — Quality Spine / Tests
```

---

## Out of scope (separate lanes)

- Dependabot openssl/cargo failures on `main`
- Vercel account block
- Proof Forge index legacy entries (local only; needs `GO commit proof-forge`)
- Auto-exporting `BIZRA_RECEIPT_STORE_PATH` in shell profiles

---

## Authority links

- Program frame: `bizra-omega/docs/BIZRA_UNIFIED_EXECUTION_BLUEPRINT.md` §4–§5
- Delivery spine: `docs/DELIVERY_SPINE_v0_1.md`
- Cycle-6 execution: `cycle-6/execution-canon.md`
- G4 contract: `cycle-6/niyyah.md` §G4
