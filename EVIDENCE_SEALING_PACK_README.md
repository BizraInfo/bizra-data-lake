# BIZRA Evidence Sealing Pack v1

Contains templates and scripts to:

- capture local model + GPU evidence (Ollama digests, env, smoke test)
- pin capability slots to exact artifacts
- run a deterministic golden set
- seal the evidence with signed git tags

Also included:
- Exportable dual-provider seal pack (Ollama + LM Studio): `bizra_phase0_week1_seal.zip`
  - Source folder: `bizra_phase0_week1_seal/`
  - Includes its own audit script and manifests for portability.

## Quick start (Windows PowerShell)

1) Capture evidence:

```powershell
./capture_evidence.ps1
```

2) Fill digests into `model-family-genesis-v1-SEALED.yaml` from `evidence/audit-results-node0.json`.

3) Seal (requires GPG configured for `git tag -s`):

```powershell
./seal_evidence.ps1 -Tag "evidence-seal-v1" -Message "Seal evidence pack v1 (Phase 0 Week 1)"
```

## Notes

- For true 3-of-5 multisig, you can either:
  - repeat the tag signing with each validator key and store signatures in `evidence/seal-attestations.json`, or
  - use a dedicated signing/attestation tool later (cosign/rekor/etc.).
- Golden set runner is intentionally not included here (language-agnostic). The JSON is designed so any runner can implement it.
