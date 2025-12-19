# Evidence Ledger – Folder Convention (v0.1)

Recommended:
- `evidence/receipts/` → JSON receipts (use `schemas/receipt.schema.json`)
- `evidence/runs/` → raw command outputs (txt, json)
- `evidence/reports/` → compiled reports (md, pdf)
- `evidence/logs/` → exported logs/traces as needed

## Naming
Use ISO timestamps and short action labels:
- `2025-12-16T00-00-00Z_inventory_receipt.json`
- `2025-12-16T00-10-00Z_repo_health_receipt.json`

## Truth labels
- **Measured**: produced by a script/tool output stored in evidence
- **Estimated**: reasoned approximation
- **Target**: goal or SLO
- **Unknown**: not verified yet
