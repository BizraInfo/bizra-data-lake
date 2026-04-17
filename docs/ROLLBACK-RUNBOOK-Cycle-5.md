# Rollback Runbook — Cycle-5 Session Commits

بسم الله الرحمن الرحيم

**Purpose:** Operator runbook for reverting any Cycle-5 session commit safely. Per DevOps discipline, every shipped commit needs a documented forward-only rollback path.

**Scope:** 17 commits on `bizra-data-lake/main` + 2 on `award-winner-design/main` — shipped 2026-04-17 during the Cycle-5 Principal Activation cycle.

**Principle:** Forward-only revert (new commit that undoes the prior) — no `git push --force`, no history rewrite. Every revert is itself auditable.

---

## Quick reference — full rollback commands

### bizra-data-lake (17 Cycle-5 commits)

To revert **the entire Cycle-5 session** back to the pre-Cycle-5 state:

```bash
cd /data/bizra/repos/bizra-data-lake
git revert --no-commit ad303bb2^..HEAD
git commit -m "revert: Cycle-5 session (ad303bb2..HEAD) per rollback runbook"
git push origin main
```

This creates ONE revert commit that undoes all 17 commits. The revert commit itself lands forward-only — no history rewrite, no force-push.

### award-winner-design (2 Cycle-5 commits)

```bash
cd /data/bizra/repos/award-winner-design
git revert --no-commit d4eec8b^..HEAD
git commit -m "revert: Cycle-5 /api/chain + /api/missions proxies"
git push origin main
```

---

## Per-commit revert table (bizra-data-lake)

For selective rollback, revert individual commits. Order matters — revert in **reverse chronological order** to avoid conflicts.

| # | Commit | What it added | Revert impact |
|---|---|---|---|
| 17 | `776c5754` | Justfile + CI policy audit | Removes `Justfile` and `docs/CI-POLICY-AUDIT-v1.md`. No code impact. |
| 16 | `755ac8fa` | Full polyglot repo inventory | Removes `docs/BIZRA-Repo-Inventory-v1.md` + SCOPE CORRECTION banner on v1 handover. |
| 15 | `39df97d9` | Handover v1 + v0.1 preserved | Removes both handover docs. Prior state had no handover on disk. |
| 14 | `bb230fd9` | Proof-forge evidence kernel | Removes `.proof-forge/` (new receipt + script only — pre-existing receipts in same dir are untouched) + `PROOF_SUMMARY.md`. |
| 13 | `24346b67` | Why Dema Wins product thesis | Removes `docs/why-dema-wins.md`. |
| 12 | `b4065025` | Cycle-5 retrospective + gateway README | Removes `cycle-5/retrospective.md` + `bizra-omega/bizra-cognition-gateway/README.md`. |
| 11 | `8b7adec9` | Two-layer doctrine + manifesto v0→v1 | Removes thesis, manifesto v1, FTAP seed, amendment record. |
| 10 | `1bf5dbb0` | Dema CLI manifesto v0 | Removes `docs/dema-cli-manifesto-v0.md`. |
| 9 | `f3f2c774` | dema CLI binary + source | Removes `bizra-omega/bizra-cognition-gateway/src/bin/dema.rs` + Cargo.toml target. |
| 8 | `77721f42` | G3 acceptance note | Removes `cycle-5/g3-acceptance-note.md`. |
| 7 | `229bd323` | G2-hardening acceptance note | Removes `cycle-5/g2-hardening-acceptance-note.md`. |
| 6 | **`8b16762a`** | **G2-hardening per founder spec** | 🚩 **HIGH IMPACT** — reverts the reject-path fix. A revert would restore the NO_SHADOW_STATE bug. **Do NOT revert in isolation.** |
| 5 | `b031fec8` | Gateway v0.2 POST /mission | Removes write-path endpoint + mission submission flow. |
| 4 | `1b2bccc5` | G1 + G2 acceptance notes | Removes `cycle-5/d5-acceptance-note.md` + `cycle-5/g2-acceptance-note.md`. |
| 3 | **`80c41602`** | **Cycle-5 G2 mission-runtime + manifest** | 🚩 **CRITICAL** — reverts `submit_mission`, `manifest_artifact.rs`, `MissionRuntimeRecord`. Kernel-level change. |
| 2 | `afe9cc30` | Cycle-4 retrospective | Removes `cycle-4/retrospective.md`. |
| 1 | `ad303bb2` | cognition + gateway crate v0.1 | 🚩 **CRITICAL** — reverts `latest_timestamp` accessor, `bizra-cognition-gateway` crate, 28th workspace member. |

### Commits marked 🚩 — do not revert in isolation

- `ad303bb2` (crate) — reverting removes the entire `bizra-cognition-gateway` crate. Subsequent commits (`b031fec8`, `8b16762a`, `f3f2c774`) all build on it.
- `80c41602` (G2) — reverting breaks the mission-runtime surface that `b031fec8` gateway depends on.
- `8b16762a` (G2-hardening) — reverting restores the reject-path NO_SHADOW_STATE bug explicitly fixed per founder's `g2-patches-abc.md` spec.

**If reverting any of the three:** revert the full stack from that commit forward (use range revert `<hash>^..HEAD`), don't cherry-pick.

---

## Per-commit revert table (award-winner-design)

| # | Commit | What it added | Revert impact |
|---|---|---|---|
| 2 | `40a6832` | `/api/missions` proxy + MissionStage reconcile | Removes mission-submit frontend path. Reverting alone still leaves `d4eec8b` chain routes wired to gateway. |
| 1 | `d4eec8b` | `/api/chain` + `/api/chain/:hash` proxies | Removes chain-read frontend paths. |

---

## Per-commit test verification after revert

After any revert, run the matching tests to confirm the system returns to a known-good state:

| After reverting | Run to verify |
|---|---|
| any bizra-cognition commit | `cargo test -p bizra-cognition --lib` (should still pass at whatever count the prior state had) |
| any bizra-cognition-gateway commit | `cargo test -p bizra-cognition-gateway` |
| full Cycle-5 | `cargo test --workspace` on bizra-omega; `pnpm test` on award-winner-design |
| proof-forge kernel | `python3 .proof-forge/scripts/forge_evidence.py --verify --project-dir .` (after revert the kernel is gone; verify pre-existing `proof-forge-v0` receipts are still present and unchanged) |

---

## Known cross-repo dependencies

Commits in the two repos are **not co-versioned**. If reverting one repo, the other may need a matching revert:

| If you revert in bizra-data-lake | Consider also reverting in award-winner-design |
|---|---|
| `b031fec8` (gateway /mission) | `40a6832` (frontend /api/missions proxy will 503 without gateway endpoint) |
| `ad303bb2` (gateway crate) | `d4eec8b` (frontend /api/chain will 503 — gateway gone entirely) |

For clean full-rollback, **revert both repos to pre-Cycle-5 state in this order:**

1. `git revert` the award-winner-design commits first (removes the dependent)
2. `git revert` the bizra-data-lake commits next (removes the dependency)
3. Push both

---

## Emergency procedures

### CI turns red on push after session

```bash
git log --oneline -3                      # identify which commit broke CI
gh run list --repo BizraInfo/bizra-data-lake --limit 5  # see which workflow failed
gh run view <databaseId> --log-failed     # read the specific failure
# If trivially fixable → fix-forward commit
# If not → git revert the commit and re-push
```

### Chain corruption detected

The proof-forge chain is tamper-evident. To diagnose:

```bash
python3 .proof-forge/scripts/forge_evidence.py --verify --project-dir .
# Walks genesis → latest, recomputes each receipt_hash, reports BROKEN or OK
```

If BROKEN: the offending receipt's `previous_hash` doesn't match the recomputed hash of the prior receipt. Two possible causes:
1. Schema mismatch across proof-forge versions (known issue — pre-existing v0 receipts vs v1 receipts don't chain end-to-end; documented in `PROOF_SUMMARY.md`)
2. File tampering (critical — investigate immediately)

### Gateway won't start

```bash
# Check port not already bound
ss -ltn | grep 7421
# Kill any stuck process
pkill -f target/release/bizra-cognition-gateway
# Restart
just dev-gateway
```

### Full nuclear rollback

If everything needs to return to pre-Cycle-5 state:

```bash
cd /data/bizra/repos/award-winner-design
git revert --no-commit d4eec8b^..HEAD
git commit -m "revert: Cycle-5 frontend proxies (nuclear rollback)"
git push origin main

cd /data/bizra/repos/bizra-data-lake
git revert --no-commit ad303bb2^..HEAD
git commit -m "revert: Cycle-5 entire session (nuclear rollback)"
git push origin main

# CI will confirm both repos are at pre-Cycle-5 state
# Dema web console at /dema returns to its pre-D5 state
```

**After nuclear rollback:**
- first principal activation receipt history in `.proof-forge/receipts/2026-04-17_074432.json` remains as historical artifact (revert doesn't delete pre-existing files that were committed earlier — .proof-forge/ existed before my session in proof-forge-v0 state)
- all Cycle-5 acceptance notes in `cycle-5/` are removed
- doctrine canon is removed (thesis, manifesto v1, FTAP seed)
- `dema` CLI binary becomes unavailable (not distributed; only built locally)

---

## What this runbook does NOT cover

- Rollback of pre-existing (non-Cycle-5) work in the repo's dirty tree — that's the original author's responsibility
- Rollback of external CI run artifacts on GitHub (those are immutable per GitHub)
- Rollback of receipt hashes already exported / shared externally (once exported, the receipt IS the record of what happened at that moment — it's immutable-by-cryptography, not by policy)
- Rollback of `.proof-forge-v0/` pre-existing receipts from 2026-03 sessions — not touched by this rollback procedure

---

## Section audit trail

| Field | Value |
|---|---|
| Generated | 2026-04-17, same-day as the cycle being rolled back |
| Verified scope | 17 bizra-data-lake commits + 2 award-winner-design commits |
| Secrets scan | 0 matches on common-secret-pattern regex across all 17 commits |
| Critical-dependency commits | 3 (`ad303bb2`, `80c41602`, `8b16762a`) — do not revert in isolation |
| Cross-repo dependencies | 2 pairs identified |
| Nuclear rollback tested | NOT tested against live repo (procedure documented only) |

---

*Filed per DevOps discipline. Every commit on origin/main has a documented revert path. Every revert path is forward-only (no history rewrite). Every critical dependency is flagged. Secrets scan clean.*

الحمد لله.
