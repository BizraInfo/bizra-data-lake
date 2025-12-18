# Data Migration Playbook: Alpha -> Beta (Template)

## Scope
- Source schema: TBD
- Target schema: TBD
- Transformations: TBD

## Plan
1. Inventory and classify data (PII, prompts, logs)
2. Create migration scripts (idempotent)
3. Dry-run on staging with realistic volumes
4. Measure duration + error rates + resource consumption
5. Define cut-over strategy (downtime vs dual-write)
6. Define rollback strategy (snapshots + restore)
7. Produce evidence bundle + sign-off (Product + Security + Data owner)

