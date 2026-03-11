# bizra-node0

Purpose: dedicated production repository for the canonical Node0 release surface.

`bizra-node0` is the future shipping surface for Node0. It exists to separate
production truth from the wider `bizra-data-lake` knowledge lake.

## Contract

- `bizra-data-lake` remains the knowledge lake and upstream import source.
- `bizra-node0` becomes the only production release surface.
- only dependency-closure imports from Node0 canonical entrypoints are allowed.
- signed artifacts may only be produced from this repo once extraction completes.
- native Linux is the certification path; WSL2 is compatibility-only.

## Canonical Entry Points

- `scripts/node0_standalone.py`
- `scripts/node0_genesis_ceremony.sh`
- `core/sovereign/node0_authority.py`
- `core/sovereign/node0_mvsa.py`
- `core/sovereign/genesis_identity.py`
- Rust MVSA proof binary under `bizra-omega`

## Operator Contract

The only canonical Node0 operator path is:

```bash
python scripts/node0_standalone.py activate --architect "MoMo"
python scripts/node0_standalone.py prove-mvsa
python scripts/node0_standalone.py task "write file missions/mvsa.txt :: node0 mvsa proof"
python scripts/node0_standalone.py health
bash scripts/node0_genesis_ceremony.sh
```

No alternate birth path is allowed outside `node0_standalone.py` and the
ceremony verifier.

## Documentation Stack

Read in this order:

1. `docs/NODE0_STANDALONE_READINESS.md`
2. `docs/constitutional/BIZRA-Node0-Definition-of-Done-v1.0-LOCKED.md`
3. `docs/OPERATIONS_RUNBOOK.md`
4. `docs/PROGRAM_BLUEPRINT.md`
5. `docs/QUALITY_GATES.md`
6. `docs/RISK_REGISTER.md`
7. `docs/ROADMAP.md`

## Program Package

This repo includes the minimum governance package for production canon work:

- `UPSTREAM_IMPORT_MANIFEST.yaml` - upstream extraction contract
- `RELEASE.md` - release and signing policy
- `docs/PROGRAM_BLUEPRINT.md` - PMBOK x DevOps x SAPE execution framework
- `docs/QUALITY_GATES.md` - CI/CD, verification, performance, and release gates
- `docs/RISK_REGISTER.md` - cascading risk and mitigation register
- `docs/ROADMAP.md` - phased delivery sequence
- `.github/CODEOWNERS` - ownership and review boundaries
- `.github/PULL_REQUEST_TEMPLATE.md` - evidence-first merge discipline

## Required Top-Level Layout

```text
bizra-node0/
  .github/
  bizra-omega/
  core/
  deploy/
  docs/
  installers/
  scripts/
  tests/
  pyproject.toml
  README.md
  RELEASE.md
  UPSTREAM_IMPORT_MANIFEST.yaml
```

## Acceptance Target

This repo is ready to replace the lake as the shipping surface only when it can:

1. activate Node0
2. prove MVSA
3. receipt a mission
4. report `ready`
5. pass `scripts/node0_genesis_ceremony.sh`
6. pass native Linux certification
7. pass Genesis-100 preflight gate (68 checks across 5 SAT domains)
8. emit signed, provable release artifacts

without importing undocumented behavior from the lake.
