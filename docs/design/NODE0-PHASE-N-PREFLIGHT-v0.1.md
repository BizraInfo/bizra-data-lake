# NODE0 Phase N Pre-Flight — v0.1

Reality-bridge checklist for Mumo's real Node0 principal activation.
**Irreversible.** One-shot. No rollback. Designed to be readable cold.

Canon anchor: `CLAUDE.md` §6 (Phase G / Phase N discipline) · §10 (stop conditions) · §1 (five invariants).

---

## 0 · What this document is (and is not)

**Is:** The exact pre-flight the operator must clear before typing `GO Node0`. Covers inputs, environment, canon stop conditions, exact commands, post-activation validation.

**Is not:** An execution plan to be run by an agent. Phase N **halts under any auto-mode** (canon §6). The operator types `GO Node0` standalone, without continuation flags, after clearing this checklist.

---

## 1 · The Three Binding Inputs

Phase N requires three operator-provided values. None have defaults. All three must be set before the `dema activate-principal` invocation.

| # | Input | Shape | Example (smoke) | Real (Phase N) |
|---|---|---|---|---|
| **I1** | **Identity key path** — path to node identity anchor JSON | absolute filesystem path | `/tmp/smoke-anchor/credentials.json` | **operator decides** (e.g. `~/.bizra/mumo/credentials.json`) |
| **I2** | **Dema cache root** — parent directory under which `dema_cache/` will be created | absolute filesystem path | `/tmp/smoke-cache` | **operator decides** (e.g. `~/.local/share/bizra/sovereign_state`) |
| **I3** | **Principal name + declared role** — human-readable operator identity | `--name <N> --role <R>` flags | `--name mumo-smoke --role node0_principal` | **operator decides** — real name binds the sovereign origin |

**I1 JSON shape** (validated by `NodeIdentityAnchor::load` at `bizra-cognition/src/principal_activation.rs:93`):

```json
{
  "node_id": "<stable identifier>",
  "public_key": "<64-hex ed25519 public key>",
  "created_at": "<ISO-8601 timestamp>"
}
```

**I1 key generation** — operator is responsible. Suggested:

```bash
# Generate a fresh ed25519 keypair (do this ONCE; store securely)
openssl genpkey -algorithm ed25519 -out ~/.bizra/mumo/key.pem
openssl pkey -in ~/.bizra/mumo/key.pem -pubout -outform DER \
  | tail -c 32 | xxd -p -c 64   # produces the 64-hex pubkey
```

The private key never touches the chain. Only the pubkey is anchored.

---

## 2 · Pre-flight checklist (canon stop conditions)

All items MUST be YES before `GO Node0`. Any NO halts per canon §10.

### A. Repo & CI hygiene
- [ ] `git status` clean on the branch where Node0 will be activated
- [ ] All open PRs relevant to Phase N are merged (e.g. PR #41 `effectiveCacheDir`)
- [ ] `cargo test --workspace` green (last run on activation branch, not cached)
- [ ] `cargo clippy --workspace --all-targets -- -D warnings` clean
- [ ] `Test Python (3.11)` on main is **green** (issue #40 resolved — Phase N should not proceed while Python activation stack is red)

### B. Environment isolation
- [ ] `BIZRA_IDENTITY_ANCHOR=<I1 absolute path>` exported
- [ ] `BIZRA_DEMA_CACHE_ROOT=<I2 absolute path>` exported (non-empty — gateway bootstrap treats empty as unset since PR #41)
- [ ] `BIZRA_SOVEREIGN_STATE_PATH` either unset OR pointed at the same real sovereign root you plan to use post-Phase-N
- [ ] Gateway binary freshly built from the post-#41 main: `cargo build --release -p bizra-cognition-gateway`

### C. Identity anchor file
- [ ] `<I1 path>` exists, readable
- [ ] `node_id` set to a stable, operator-chosen identifier
- [ ] `public_key` is 64 hex chars (32 bytes) derived from a real ed25519 key the operator controls
- [ ] Private key backed up AND stored separately from the repo (never in working tree)

### D. Daughter Test (canon §10 / NO_SHADOW_STATE)
- [ ] `~/Downloads` and any path that `dema organize` will touch contains **no shadow state** (no unverified PDFs, no untrusted binaries, no conflicting prior receipts)
- [ ] Any existing `sovereign_state/dema_cache/` on disk is either expected rehydration content OR has been archived off path
- [ ] No zombie gateway process from earlier smoke runs: `pgrep -af bizra-cognition-gateway` returns empty

### E. Operator alignment
- [ ] Operator is rested, not fatigued, not on hour 15 of a session
- [ ] Operator has typed `GO Node0` as a **standalone** phrase (no `/A`, `/@`, `/L` flags chained)
- [ ] Operator understands: this mints the first PrincipalActivationReceipt under the real anchor. **It cannot be undone** short of `rm -rf`-ing the cache root (which leaves the chain head but loses profile rehydration).

### F. Escalation reachability
- [ ] At least one trusted witness can be pinged if Phase N produces an unexpected verdict
- [ ] `dema chain` and `dema poi` commands are understood — operator can read the receipt chain independently to audit the outcome

---

## 3 · Execution sequence (operator-run, not agent-run)

Exact commands. No phantoms — every flag verified against `dema --help` (canon §5).

```bash
# 0. Sanity
which dema                     # must point to the post-#41 release binary
dema --version                 # confirm build

# 1. Start a fresh gateway with the real env
export BIZRA_IDENTITY_ANCHOR=<I1 absolute path>
export BIZRA_DEMA_CACHE_ROOT=<I2 absolute path>
unset BIZRA_SOVEREIGN_STATE_PATH      # or set to real if bootstrapping durable
nohup bizra-cognition-gateway > ~/.bizra/gateway.log 2>&1 &
sleep 2
dema health                    # must return: gateway: ok (bizra-cognition-gateway-v1)

# 2. THE IRREVERSIBLE ACT — mint the real PrincipalActivationReceipt
#    Quality 1.00 = runtime IHSAN floor. Not negotiable for real Node0.
dema activate-principal \
  --name "<I3 name>" \
  --role "<I3 role, default: node0_principal>" \
  --quality 1.00 \
  --anchor "<I1 absolute path>"

# 3. Capture the receipt hash immediately (operator records this offline)
dema chain --json | jq -r '.head' > ~/.bizra/node0-genesis-receipt.hash

# 4. Verify all five invariants Permitted at runtime floor
# Expected from step 2 output:
#   ✓ ZANN_ZERO          Permit    score=1.0000
#   ✓ CLAIM_MUST_BIND    Permit    score=1.0000
#   ✓ RIBA_ZERO          Permit    score=1.0000
#   ✓ NO_SHADOW_STATE    Permit    score=1.0000
#   ✓ IHSAN_FLOOR        Permit    score=1.0000
#   verdict: Permit
#   ✓ chain head equals principal activation receipt — sealed
#   ✓ profile persisted to <I2>/dema_cache/        ← (from PR #41: server-reported)
```

**If ANY invariant scores < 1.0000 or returns Reject:** abort. The chain is unchanged per canon §10 Proof Law ("refused intents leave no chain trace"). Diagnose, fix, retry. Do not escalate quality by flag.

---

## 4 · Post-activation validation

After step 2 returns Permit, operator runs these to confirm the chain is sealed as expected:

```bash
# A. Chain length grew by 9 (per bizra-cognition-gateway tests):
dema chain
# Expected: length increased by 9 from its pre-activation value
#           head == principalActivationReceiptId from step 2 output

# B. Profile is on disk at the server-reported path (post-#41):
cat <I2>/dema_cache/principal.json | jq '.principal_id, .name'

# C. POI ledger shows the genesis activation:
dema poi
# Expected: PrincipalActivation bucket count=1, total≈0.9604-1.0000

# D. Rehydration smoke (prove durability):
kill %1                        # stop gateway
nohup bizra-cognition-gateway > ~/.bizra/gateway2.log 2>&1 &
sleep 2
grep "principal profile rehydrated from disk" ~/.bizra/gateway2.log
# Expected match: proves the real profile survives restart
```

---

## 5 · Abort paths

Phase N is **not resumable** mid-flight. "Abort" means refuse-to-launch before step 2.

| Failure Mode | Detection | Response |
|---|---|---|
| Pre-flight §2 item NO | checklist | Do not type `GO Node0` until cleared |
| Gateway health returns error | step 1 `dema health` | Kill gateway, diagnose log, rebuild, retry |
| Anchor load error | step 2 returns `IDENTITY_ANCHOR_LOAD` | Fix `<I1>` JSON shape (per §1 I1 table), retry step 2 |
| IHSAN score < 1.0000 | step 2 admissibility output | Do NOT raise quality flag. Fix the real underlying weakness (the gate caught something real) |
| Verdict: Reject | step 2 admissibility | Chain unchanged. Read rejection.remediation_path, apply fix, retry |
| `cache_warning` present in step 2 output | step 2 output | Chain IS sealed. Profile write failed but is rebuildable from chain. Investigate disk issue, re-persist via rehydrate flow |

If step 2 completes with Permit + sealed verdict: **Node0 is live**. No abort possible. The sovereign origin now binds to `<I1>`'s pubkey. Proceed to §4 validation.

---

## 6 · What Phase N seals

Per canon §2 (Constitutional Spine):
- **CanonicalReceipt** — the PrincipalActivationReceipt, signed, chained to genesis
- **MissionState** — NodeLifecycle mission record at stage `Replayability`
- **ReceiptStateMachine** — transition from HYPOTHESIS → ... → MARKETABLE for the activation mission
- **GenesisSeal** — this receipt becomes the sovereign origin; all future receipts chain from here

After Phase N, the following canon truths are locked:
- The operator's pubkey is the sovereign identity
- The `<I2>/dema_cache/` path is the authoritative profile persistence location
- Every subsequent mission receipt chains from this activation's chain head

---

## 7 · What Phase N does NOT do

- **Does NOT** start libp2p networking or bind to any port beyond localhost:7421
- **Does NOT** mint SAT/SEED/BLOOM tokens (canon §4 aspirational horizon — not part of Phase N)
- **Does NOT** invite peer nodes or establish a URP flywheel
- **Does NOT** expose the gateway beyond localhost
- **Does NOT** require or produce external commitments

Network activation, multi-node coordination, and economic flywheel are **separate future arcs** with their own pre-flights. Phase N is sovereign-origin-only.

---

## 8 · Proof-of-Truth convergence (canon §9)

On successful Phase N, all four modalities must cross positive:

| Modality | Evidence captured |
|---|---|
| **Formal** | Five invariants Permit at IHSAN score 1.0000; FATE predicates satisfied |
| **Cryptographic** | BLAKE3-chained receipt with ed25519 pubkey binding; chain head matches activation_receipt_id |
| **Empirical** | Chain length grew by 9; `principal.json` on disk at server-reported path; rehydration on restart works |
| **Economic** | n/a for Phase N (no tokens minted; economic gate applies to later arcs) |

If any required modality is negative, Phase N has not truly happened regardless of what the CLI printed. Re-audit before claiming SHIPPED status.

---

## 9 · Session record

Every Phase N run produces:

```
~/.bizra/
├── node0-genesis-receipt.hash      # 64-hex, offline record
├── gateway.log                      # first-boot output (keep for audit)
└── gateway2.log                     # rehydration smoke (keep for audit)

<I2>/dema_cache/
├── principal.json                   # the sealed profile
├── receipt_history.json             # append-only chain cache
├── manifest_history.json
├── mission_log.json
├── state_snapshots.json
├── resource_registry.json
└── poi_ledger.json
```

Archive these. The `.hash` file is the canonical "did it happen" witness Mumo keeps off-repo.

---

## 10 · Version discipline

This document is `v0.1` — the first formal pre-flight. Revisions required when:
- New invariants added to canon §1
- New required inputs added (e.g. SAT bond when Economic modality graduates)
- Gateway DTO changes (e.g. new fields on `ActivatePrincipalResponse`)
- Any canon §10 stop condition added or removed

Bump version, preserve history. Do not rewrite in place.

---

**Refuse to fire until every §2 checkbox is YES. The sovereign origin cannot be minted twice.**
