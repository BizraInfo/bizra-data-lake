# BIZRA Agent Instruction — Cycle-6 G1 Code Canon (Post-Writer Resolution)

بسم الله الرحمن الرحيم

**Filed:** 2026-04-17 (Friday), Dubai GST
**Status:** ACTIVE — G1 Phase 1 code window
**Authority:** Founder `/@ b`, 2026-04-17 14:37 GST

## Status

Cycle-6 G1 is **UNBLOCKED**. The Python writer format is resolved and verified 4/4 against live data (`cycle-6/g1-writer-format-found.md`).

## Read first

1. `cycle-6/execution-canon.md`
2. `cycle-6/g1-authority-adr.md`
3. `cycle-6/g1-execution-brief.md`
4. `cycle-6/g1-blocker-resolution-canon.md`
5. `cycle-6/g1-writer-format-found.md`

## Current canonical truth

- G1 scope is **durable-read persistence only**
- Python remains the authoritative writer of `sovereign_state/`
- Rust must implement a **read-only projection**
- `block_zero` is a **genealogical anchor**, not the live activation-chain head
- envelope integrity must be verified internally
- no signer audit in this pass
- no threshold drift check in this pass

## Resolved hash rule

For each chain entry:

```
entry.hash = BLAKE3( prev_hash_ascii_hex || json.dumps(data, sort_keys=True).encode() ).hexdigest()
```

Notes:
- `prev_hash_ascii_hex` is the 64-char ASCII hex string
- Genesis prev-hash is `"0" * 64`
- Python JSON formatting uses default separators with spaces (`", "`, `": "`)
- Rust `serde_json` default output does **NOT** match by byte parity
- Implement a custom formatter to reproduce Python output exactly
- SHA-256 fallback exists in the writer if Python `blake3` import fails; Rust supports BLAKE3 primary and fails closed if verification cannot be established

## Execution mode — staged 3-commit Phase 1

### Commit A — custom formatter + property tests

- Reproduce Python `json.dumps(..., sort_keys=True)` byte output
- Test byte-equality against fixture examples
- End-to-end blake3 chain-hash test against the first live activation_chain entry
- Catches separator / spacing mismatch before any runtime code

### Commit B — snapshot loader + verification tests

- Implement `SovereignStateSnapshot::load(path)` reading
- Load activation-chain envelopes from `sovereign_state/receipts/`
- Verify per-entry hash + prev-hash linkage end-to-end
- Fail closed on mismatch (`SovereignStateError` enum)
- Tests against the live fixture + tampered-fixture regression

### Commit C — runtime constructor + gateway wiring

- Implement `CognitionRuntime::from_sovereign_state(path: &Path) -> Result<Self, BootstrapError>`
- Add `SovereignStatePayloadStore { root: PathBuf }` (read-only; write attempts return `Err`)
- Preserve in-memory fallback for dev mode when path is absent
- Wire gateway bootstrap to use `BIZRA_SOVEREIGN_STATE_PATH` env var

## Hard rules

- No widening scope beyond G1
- No signer audit
- No constitutional-threshold drift check
- No G3 ADR work
- No G4 implementation
- No P0 security patch arc in this pass
- No speculative crypto logic
- Fail closed on any verification mismatch

## Success condition for G1 Phase 1

- Rust can read authoritative Python-authored sovereign state
- Live activation-chain envelopes self-verify correctly
- Runtime can bootstrap from sovereign state in read-only mode
- Dev fallback to in-memory still works
- All of this is covered by tests before gateway wiring is claimed complete

## Reporting format

When finished, report:
1. Which of A/B/C completed
2. Files changed
3. Tests added and passing
4. What remains open after G1 Phase 1

---

الحمد لله.
