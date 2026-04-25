# Node0 Genesis Asset Manifest

**Scope**: `tools/node0_genesis_manifest.py` + `evidence/node0_genesis_manifest/`
**Status**: side-track v0.1 · DECOUPLED from runtime PR queue · NOT in Claim Registry · NOT signed
**Generated**: 2026-04-25 GST

The Genesis Manifest is the founder-stated → hash-verified bridge for Node0. It does **not** prove Mumu's claims true. It freezes the digest of named artifacts at a point in time so that:

- the existence of an artifact at that moment is provable
- the content cannot be silently mutated afterward without breaking the hash
- private artifacts can be acknowledged without exposing their bytes

What it deliberately is **not**:

- not a Claim Registry (Phase 3, deferred)
- not an attestation service (no third-party signer)
- not a Proof-of-Impact engine (no scoring)
- not Ed25519-signed (signing belongs to the runtime spine; this is a content-addressed manifest only)
- not a runtime change — touches no module under `core/`, `src/`, `proof_engine/`, `bus/`, `identity/`, `FATE`, or CI

## Files in this directory

| File | Role |
|---|---|
| `README.md` | This file |
| `REDACTION_POLICY.md` | Hard rules + soft rules governing what the script may read/emit |
| `manifest.schema.json` | JSON Schema for the manifest input format (asset declarations) |
| `assets.example.json` | Example manifest with a few assets at each visibility level — for testing |
| `NODE0_GENESIS_ASSET_MANIFEST.json` | Canonical Node0 manifest — the actual one the script processes |
| `NODE0_GENESIS_HASH_LEDGER.jsonl` | Append-only output: one JSON line per asset processed |
| `RUN_REPORT.json` | Output: stats, warnings, hash algorithm used, per-visibility counts |

## How to run

```bash
# default: process the canonical manifest
/data/bizra/repos/bizra-data-lake/.venv/bin/python tools/node0_genesis_manifest.py

# explicit:
python tools/node0_genesis_manifest.py \
    --manifest evidence/node0_genesis_manifest/NODE0_GENESIS_ASSET_MANIFEST.json \
    --output-dir evidence/node0_genesis_manifest/
```

The script:

1. Validates each asset against `manifest.schema.json` (basic structural checks; full JSON Schema draft-07 validation is best-effort).
2. For each asset, honors the `visibility` field per `REDACTION_POLICY.md`.
3. Computes a `content_hash` (BLAKE3 by default; SHA-256 fallback) for `public` and `private`/`hash_only` assets when the file exists.
4. Computes a `metadata_hash` for every asset (always — even redacted ones).
5. Appends one JSONL line per asset to `NODE0_GENESIS_HASH_LEDGER.jsonl`.
6. Writes `RUN_REPORT.json` with stats.

The script is read-only against the source files. It writes only inside `evidence/node0_genesis_manifest/`.

## What goes in the manifest

The first canonical roots:

1. **الرسالة** (Al-Risāla) — Mumu's first origin document, Ramadan 2023. Status: `private` + `FOUNDER_STATED` + `PLANNED` until the operator upload anchors it.
2. **البذرة** (Al-Bidhra) — Mumu's second origin document, Ramadan 2023. Same handling.
3. **`docs/canon/bizra-origin-canon-v1.md`** — in-tree canon-v1 document (closest verifiable analogue to the origin kernel until BIZRA_ORIGIN_KERNEL.md is committed). Status: `public` + `VERIFIED`.
4. **Repo HEAD index** — `git ls-tree HEAD` digest. Status: `public` + `VERIFIED`.
5. **Founder-stated metrics** (15k+ hours, 1601 conversations, etc.) — recorded as `PLANNED` placeholder entries until backed by timestamped artifacts (commit logs, calendar exports, etc.). The manifest acknowledges them; it does NOT validate them.

## Truth-label discipline

Each asset declares a `proof_status` per BIZRA claim discipline:

- `VERIFIED` — file present, hashable, origin traceable in git or signed receipts
- `MEASURED` — instrumented metric with reproducible measurement (no current entries; this label belongs in Phase 3 Claim Registry)
- `DERIVED` — computed from a verified source via a deterministic function
- `FOUNDER_STATED` — Mumu's testimony only; no automated verification yet
- `PLANNED` — declared but file/data not yet present in the manifest's reachable scope

The script never upgrades a label. Upgrades require operator action and, in Phase 3, an `IhsanDecision` trace.

## What this side-track unlocks

Once the Genesis Manifest is committed to the repo (separate operator decision, NOT in this v0.1 scope), Mumu can publicly say:

> Every Node0 founder-stated claim is recorded with a manifest entry, a hash, a visibility level, and a proof status. Anything labeled `VERIFIED` is hash-verifiable from this repo right now. Anything labeled `FOUNDER_STATED` or `PLANNED` is testimony, marked as such, and pending verification.

This is the cleanest implementation of `feedback_audit_label_inflation_guard` for the founder claim surface.

## Hard stops respected

- No Claim Registry implementation
- No attestation service
- No Proof-of-Impact engine
- No edits to `core/`, `src/`, `proof_engine/`, `bus/`, `identity/`, `FATE`, CI
- No raw private content exposed
- No commit, no push, no PR
