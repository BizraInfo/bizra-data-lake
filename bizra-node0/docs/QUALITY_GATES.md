# Node0 Quality Gates

Status: release-control document

## Gate Hierarchy

1. Specification gates
   - `docs/NODE0_STANDALONE_READINESS.md`
2. Birth verification gates
   - `docs/constitutional/BIZRA-Node0-Definition-of-Done-v1.0-LOCKED.md`
3. CI/CD gates
   - `.github/workflows/node0-ci.yml`
   - `.github/workflows/node0-canon.yml`
4. Native Linux certification gate
   - `deploy/node0/certify-linux.sh`
5. Release gate
   - `RELEASE.md`

## CI/CD Gates

### Docs parity

Must prove:
- spec, DoD, audit trail, roadmap, risk, and quality docs all exist
- import manifest exists
- release policy exists

### Security fail-closed

Must prove:
- production JWT secret requirement
- production API auth requirement
- websocket auth contract stability
- Ghost bridge production defaults

### Node0 operator smoke

Must prove:
- canonical tests for authority, MVSA, and acceptance pass

### Native Linux certification

Must prove:
- installer, service unit, logrotate, and certification scripts are valid
- installed host satisfies certification checks

### Provenance

Must produce:
- import manifest
- release policy
- certification receipt or explicit hold

## Performance Ratchets

Initial targets:

| Metric | Target |
|---|---|
| `health` median latency | <= 100 ms |
| `health` p95 latency | <= 250 ms |
| `prove-mvsa` wall time | <= 60 s |
| first receipted `task` wall time | <= 120 s |
| ceremony hard-gate success | 100% |
| anonymous production exposure | 0 |

These are ratchets, not suggestions. They may be tightened, not weakened.

## Release Blocking Rules

A release is blocked if any of the following are true:

- lifecycle truth is contradictory
- birth gate does not pass
- production auth does not fail closed
- native Linux certification does not pass
- provenance or signing artifacts are missing

## Ethical Quality Rules

- Ihsan: do not accept weakened evidence for a green status
- Adl: do not hide operator requirements or ambiguous behavior
- Amanah: do not release without clear custody of secrets and artifacts
