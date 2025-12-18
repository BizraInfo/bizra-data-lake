# Genesis Closeout (Node0)

- Commit SHA: 24d743681be2141fe4b73a31159a47e2863a3b55
- Receipt: genesis_receipt_v1_20251218_155547Z.json
- Receipt sha256 (canonical, excluding receipt_sha256 field): c6e75b10a5918a1b89cea17896bb18a1b366ce22aafd0de0724e8da1bc661edd
- Gates snapshot: docs/evidence/gates/node0_gates_latest.json
- SAPE prompt_sha256: 19515b309196c65c578365172fdcad6cfc7ff621e3c1257a9c8b5b7dacf60c26
- Kernel image digest: bizra-dual-agentic-system--main-kernel@sha256:8f0820adef6956286d50b8a0b05fe805e693375ddb81dd92218f0fee4afb00ca
- Ignition command: pwsh ./scripts/genesis_ignite_all.ps1
- Ignition timestamp (UTC): 2025-12-18 15:55:47

Verification check-list:
- python -m py_compile scripts/genesis_receipt.py
- docker compose config
- Canonical receipt hash recomputed locally and matched recorded value
