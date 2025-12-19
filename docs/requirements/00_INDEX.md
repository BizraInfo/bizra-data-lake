# Requirements Pack (Index)

This folder holds the requirements baseline and traceability for the BIZRA system.

## Files
- Baseline requirements: `docs/requirements/requirements_v1.yaml`
- Traceability matrix: `docs/requirements/traceability_v1.yaml`

## Rules
- Every requirement has an ID (e.g., `NFR-001`).
- Every "DONE" requirement must map to:
  - design reference (doc or ADR),
  - implementation evidence (`path:line` or module),
  - verification evidence (test name or command),
  - operational evidence (runbook / dashboard).
- If a requirement cannot be verified yet, it stays `status: planned`.

