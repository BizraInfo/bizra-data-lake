# MSSC — One Node, One Proof, One Reward

**Goal:** From a clean machine, generate **Block 0**, run **Node‑0 Validation API**, produce **one PoI attestation**, verify it deterministically, and compute a **testnet reward quote** (off‑ledger).  
This is the smallest slice that proves: **privacy‑preserving impact → cryptographically verifiable → rewardable**.

## Commands (golden path)
```bash
python3 mssc/mssc.py genesis build
python3 mssc/mssc.py api up
python3 mssc/mssc.py contribute run
python3 mssc/mssc.py poi attest
python3 mssc/mssc.py poi verify
```

## Artifacts (canonical)
- `mssc/artifacts/genesis.built.json`
- `mssc/artifacts/genesis_merkle_root.txt`
- `mssc/artifacts/pack.manifest.json`
- `mssc/artifacts/pack.sha256`
- `mssc/artifacts/poi_attestation.json`

## Determinism Notes
- JSON is canonicalized with sorted keys + no whitespace.
- Ed25519 signatures are deterministic **for a given keypair**.
- The keypair is generated once in `mssc/keys/` and reused.

## API
- POST `http://127.0.0.1:8808/api/v1/proof-of-impact/verify`
- Body: the `poi_attestation.json` content.
- Response: `{valid, score, reward_quote}`

## Privacy
- Raw evidence stays local (only hashes are included in the pack).

## Tamper Tests
- Flip one byte in attestation → verify must fail.
- Flip one byte in pack hash → verify must fail.
- Replace genesis root → verify must fail.
