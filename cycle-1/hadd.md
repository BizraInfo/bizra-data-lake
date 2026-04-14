# Hadd — Cycle #1

**Date:** 2026-04-14

---

## IN SCOPE

1. **Fix the syntax error** in `core/inference/_connection_pool.py:354` — the `raise RuntimeError( from None` misplacement
2. **Re-run integration test** to verify 9/9 pass after fix
3. **Verify smoke tests** still 11/11 after fix (no regressions)
4. **Record BLAKE3 hash** of the canonical activation artifact set
5. **Update TOPOLOGY_CANON.md** with Node0 Activation canonical entry

## OUT OF SCOPE

- Ed25519 credential setup (`identity/credentials.json` missing) — separate concern, HMAC fallback is functional
- Glass Cockpit fictional code cleanup — separate cycle
- Any new features or architectural changes
- Rust codebase (`cargo test`) — not in the Node0 activation path
- Any changes to frozen anchors or constitutional parameters
- Paper submission workflow

## Daughter Test

**"We fix one typo in a helper file so the full system boots and we can stamp the activation as proven."**

Simple. One change. Verifiable.

## Constraints Applied

- ONE architectural change maximum: ✓ (zero architectural changes — this is a syntax fix)
- Frozen anchors unmodified: ✓
- Scope is bounded to a single file fix + re-verification
