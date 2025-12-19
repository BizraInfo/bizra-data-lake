# Phase 0 Week 1 — Seal & Gate Pack (Dual Provider: Ollama + LM Studio)

This pack standardizes and *seals* your local model fleet so routing becomes deterministic and auditable across **Ollama + LM Studio**.

## What this pack contains
- `manifests/model-family.capabilities.v1.yaml`: provider-agnostic capability policy (slots, SLOs, gates).
- `manifests/model-family.artifacts.v1.yaml`: pinned artifact identities (endpoints + hashes).
- `scripts/audit-model-family-dual.ps1`: evidence capture script for Ollama + LM Studio (plus GPU snapshot).
- `docs/adr/ADR-0002-validator-safety.md`: Genesis quorum decision record (copied for pack portability).
- `evidence/model_family/`: where audit outputs are written (empty until you run the audit).

## How sealing works (high-level)
1) **Audit (MEASURED):** capture *exact* model identities and runtime details as evidence.
2) **Pin (DERIVED):** copy hashes/IDs into the artifacts manifest.
3) **Seal (VERIFIED once committed):** set `sealed: true`, set timestamp + signer, and commit/tag.

## Step-by-step (Windows / PowerShell)

### 1) Run the dual-provider audit
From the pack root:
```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\audit-model-family-dual.ps1
```

This writes evidence files under `evidence/model_family/` (relative to the pack root):
- `evidence/model_family/host_info.txt`
- `evidence/model_family/gpu_memory.txt`
- `evidence/model_family/ollama_list.txt`
- `evidence/model_family/ollama_show_modelfile_*.txt`
- `evidence/model_family/ollama_store_snapshot.txt`
- `evidence/model_family/lmstudio_models.json`
- `evidence/model_family/lmstudio_file_hashes.json`

### 2) Fill the artifacts manifest
Edit:
- `manifests/model-family.artifacts.v1.yaml`

Replace:
- `AUDIT_FILL` fields (runtime version/system hardware)
- `UNKNOWN_SHA256` fields (pins)

### 3) Seal
In both manifest files set:
```yaml
sealed: true
sealed_date_utc: "YYYY-MM-DDTHH:MM:SSZ"
sealed_by: "your_signer_id"
```

### 4) Commit + tag (recommended)
Commit the pack (and evidence if you choose to track it), then tag the seal event.

## Notes
- Keep secrets out of manifests. Credentials belong in env vars or secret managers.
- Avoid committing full local file paths if you plan to publish the repo.

