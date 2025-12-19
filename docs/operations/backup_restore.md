# Backup / Restore / DR (Template)

## Define targets
- RPO (data loss tolerance): TBD
- RTO (time to recovery): TBD

## Backup plan
- Database backups: schedule + retention
- Logs/evidence retention
- Restore verification cadence (quarterly minimum)

## Restore drill checklist
1. Create fresh environment
2. Restore from latest backup
3. Run verification queries
4. Validate service readiness
5. Record metrics + seal evidence

