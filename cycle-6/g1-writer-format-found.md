# Cycle-6 G1 — Writer Format Resolution

بسم الله الرحمن الرحيم

**Filed:** 2026-04-17 (Friday), Dubai GST
**Status:** RESOLVED — algorithm tool-verified against live data
**Resolves:** `cycle-6/g1-blocker-resolution-canon.md` mandatory-next-action
**Enables:** G1 code per `/@ C` decision rule (writer simple and reproducible → implement compatible Rust)

---

## Writer identified

`deploy/node0/bizra_node_activate.sh:400-407` — Bash script with inline Python that writes `activation_chain_<TIMESTAMP>.json` envelopes.

This is not a library module; it's a one-shot activation orchestrator. The chain envelope is produced during Node0 activation ceremony (`2026-04-13T23:55:26Z`), then read subsequently by `core/cockpit/server.py:196`.

## Exact hash algorithm (verified)

```python
# From deploy/node0/bizra_node_activate.sh:400-407
import blake3, json

content = json.dumps(data, sort_keys=True).encode()  # DEFAULT separators
current_hash = blake3.blake3(prev_hash.encode() + content).hexdigest()
```

### Verification (4/4 entries match)

Ran against `sovereign_state/receipts/activation_chain_2026-04-13T23:55:26Z.json`:

| Entry | File | Expected | Computed | Match |
|---|---|---|---|---|
| 0 | `agent_activation_…Z.json` | `89035bdc…b2b7584b` | `89035bdc…b2b7584b` | ✅ |
| 1 | `fate_validation_…Z.json` | `82a3a599…47b85ffe` | `82a3a599…47b85ffe` | ✅ |
| 2 | `genesis_urp_…Z.json` | `783aad60…b31dee50` | `783aad60…b31dee50` | ✅ |
| 3 | `onboard_founder_…Z.json` | `b98d2031…34d7649e` | `b98d2031…34d7649e` | ✅ |
| envelope `head_hash` | — | `b98d2031…34d7649e` | == entry[3].hash | ✅ |

## Rust-reproduction contract

To compute a compatible hash in Rust, the implementation **must** match Python's `json.dumps(data, sort_keys=True)` default exactly:

| Python default | Rust requirement |
|---|---|
| Keys sorted recursively (all nested objects) | Use `BTreeMap` or sort keys pre-serialization |
| `", "` between array elements / object pairs (comma + space) | Custom `serde_json::ser::Formatter` — CompactFormatter has no spaces |
| `": "` between key and value (colon + space) | Same — custom formatter |
| No outer whitespace; no indent | No `PrettyFormatter` |
| UTF-8 encoding | `.to_string().into_bytes()` |
| No trailing newline | Default `to_string()` adds none |
| `ensure_ascii` is **default True** in the writer *(not verified — test with non-ASCII input before extending scope)* | Most BIZRA events are ASCII; defer non-ASCII handling until a real case surfaces |

### Minimal Rust formatter (narrow-real, ASCII-only)

A custom `Formatter` with ~25 lines: writes `", "`, `": "`, no newlines, no indent. Keys sorted via `BTreeMap` or explicit `sort_by_key`. Sufficient for the current 4-entry fixture + future ASCII event receipts.

Non-ASCII edge cases (emoji, Arabic, etc.) can arise in future receipts. The Python writer uses `ensure_ascii=True` implicitly (default) which escapes non-ASCII as `\uXXXX`. Rust `serde_json` also escapes non-ASCII as `\uXXXX` by default. Byte-for-byte parity requires verifying escape-form consistency — **deferred to when a non-ASCII receipt actually exists.**

## G1 ADR impact — what changes

| ADR section | Original | Updated (post-finding) |
|---|---|---|
| §Bootstrap flow step 4 ("verify referenced receipt file hash") | Unspecified algorithm | **SPECIFIED:** `BLAKE3(prev_hash_hex_ascii + python_sortk_json(receipt_data))` |
| §Bootstrap flow step 5 ("assert computed chain head equals `block_zero.receipt_chain.chain_hash`") | Assumed block_zero is the chain root | **CORRECTED:** `block_zero` and live `activation_chain_*.json` are independent chain surfaces with zero hash overlap. `block_zero` is a sealed genealogical anchor (2026-03-19 genesis ceremony); live envelopes are later activation-specific chains. G1 Phase 1 verifies **each envelope internally** (first entry prev_hash == 0^64, last entry hash == envelope.head_hash). `block_zero` reconciliation is a separate read (genealogical anchor, not chain head). |

These are not scope widenings — they are *clarifications* of the ADR that the original draft left implicit. They do not require a new founder gate.

## Authoritative writer has `SHA-256` fallback — decision

`bizra_node_activate.sh:405-407` falls back to SHA-256 if `blake3` import fails:

```python
except ImportError:
    content = json.dumps(data, sort_keys=True).encode()
    current_hash = hashlib.sha256(prev_hash.encode() + content).hexdigest()
```

**Decision:** Rust G1 verifies BLAKE3 only. If an envelope's head_hash doesn't verify under BLAKE3, Rust retries under SHA-256. If neither matches, fail closed with `HashAlgorithmUnknown` error.

Rationale: allows forward compatibility with envelopes produced on hosts lacking `blake3` — honest, bounded, fail-closed.

## What this enables

Cycle-6 G1 Phase 1 is now unblocked under Path (C) resolution:

- Rust `SovereignStateSnapshot::load(path) -> Result<..., SovereignStateError>` can verify envelope integrity end-to-end per the exact algorithm
- `CognitionRuntime::from_sovereign_state(&Path)` composes over the snapshot
- Gateway bootstrap wiring via `BIZRA_SOVEREIGN_STATE_PATH` env var per ADR
- Verification gate from niyyah §G1 (seal → restart → `dema chain` returns same receipt) becomes mechanically satisfiable

## Non-goals this commit

- Rust code is **not** in this commit. This is findings + spec update only. Code lands in a subsequent commit.
- `block_zero` verification (its internal `chain_hash` over its own 10 receipts) is **not** part of G1 Phase 1. That is a separate algorithm likely in `genesis_engine*.py` or the genesis ceremony code — a separate finding if genealogical verification is later scoped in.
- Non-ASCII encoding parity — deferred until first non-ASCII event receipt.

## Constitutional filter

| Invariant | Upheld by |
|---|---|
| ZANN_ZERO | Findings introduce no economic surface |
| **CLAIM_MUST_BIND** | Algorithm is not speculated — it is derived from the actual writer code and tool-verified 4/4 against live data |
| RIBA_ZERO | No extractive pattern |
| **NO_SHADOW_STATE** | Eliminates "what algorithm does Python use" ambiguity; binds Rust to Python's precise rule |
| IHSAN_FLOOR | Preserved; unchanged by this finding |

## References

- Writer source: `deploy/node0/bizra_node_activate.sh:400-407`
- Canonicalization library (not used by writer directly but related): `core/proof_engine/canonical.py`
- Live fixture: `sovereign_state/receipts/activation_chain_2026-04-13T23:55:26Z.json`
- Blocker record: `cycle-6/g1-blocker-resolution-canon.md`
- G1 ADR: `cycle-6/g1-authority-adr.md`
- Niyyah: `cycle-6/niyyah.md` §G1

## Signature

Filed: Mumo (Muhammad Beshr) — 2026-04-17 Dubai GST
Cycle chain position: 6 / G1 / writer-format-resolution
Canon status: **FINDING** — unblocks G1 code. Subsequent Rust commit implements against this spec.

الحمد لله.
