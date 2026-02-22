# Evidence Package Runbook

## Canonical Order
1. `python scripts/evidence/preflight_evidence.py --stage scaffold --tier private_full`
2. `python scripts/evidence/import_external_assets.py --config scripts/evidence/config/evidence_package.yaml`
3. `python scripts/evidence/build_evidence_package.py --stage scaffold --tier private_full`
4. `python scripts/evidence/export_public_redacted.py --from private_full --to public_redacted`
5. `python scripts/evidence/sign_evidence_package.py --tier private_full`
6. `python scripts/evidence/verify_evidence_package.py --stage scaffold --tier private_full`
7. `python scripts/evidence/release_pack.py --stage final`

## Final Mode
- Use `--stage final` on preflight/build/verify.
- Final gate is fail-closed and requires:
  - founding docs present
  - required artifacts present
  - `ihsan_as_architecture.md` present
  - research manifest completeness (`discovered == indexed`, `unindexed == 0`)

## Single Release Pack Command
- Command:
  - `python scripts/evidence/release_pack.py --stage final`
- Output location:
  - `artifacts/evidence/BIZRA-EVIDENCE-PACKAGE-v1.0-GENESIS/release/`
- Per tier outputs:
  - `.tar.gz`
  - `.tar.gz.blake3`
  - `.tar.gz.sha256`
  - `.tar.gz.sig`
