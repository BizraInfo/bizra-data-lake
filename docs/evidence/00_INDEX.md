# Evidence Ledger (Index)

Purpose: Store reproducible evidence bundles for key claims (security posture, SLOs, deployments, migrations, audits).

## Structure (recommended)
- `docs/evidence/<timestamp>/...` for point-in-time bundles
- `docs/evidence/receipts/...` for per-request receipts (machine-readable, local-only by default)

## Sealing
- Use your evidence sealing flow to hash and tag evidence packs.
