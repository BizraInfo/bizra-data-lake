# SAPE in the BIZRA Workflow Lifecycle

SAPE (Symbolic → Abstraction → Probe → Elevation) is the **standard reasoning workflow** for high-impact work in BIZRA. It turns “analysis” into **buildable artifacts** (specs, invariants, tests, risks, and acceptance criteria) under **Ihsān/ʿAdl/Amānah**.

## Where SAPE Runs (Lifecycle Insertion Points)

### 1) Requirements → Scope Control (PMBOK)
- Use SAPE to convert raw intent into:
  - measurable success criteria
  - constraints + forbidden moves
  - acceptance tests and verification steps
- Output is attached to:
  - `docs/requirements/*`
  - `docs/blueprints/*`

### 2) Architecture Decisions (ADR Discipline)
- Use SAPE to produce an ADR draft with:
  - alternatives (I/C/O paths)
  - invariants and proof obligations
  - operational consequences + rollback strategy
- Final ADRs live under:
  - `docs/adr/*`

### 3) Security Reviews (Fail-Closed Ethics + Threat Model)
- Every “H” stakes security change must be:
  - FATE-approved (ethics gate)
  - evidence-backed (graph or repo evidence)
  - tested (negative tests, abuse cases, red-team scenarios)
- Track outcomes in:
  - `docs/security/*`

### 4) Performance & Reliability Work (SLO/SLA + QA)
- Use SAPE to define:
  - p95/p99 targets
  - load test plan and regression gates
  - observability requirements (metrics/logs/traces)
- Track in:
  - `docs/slo/*` and `docs/operations/*`

### 5) CI/CD & Release Gates (DevOps)
- Use SAPE to generate a **release checklist** and **risk register** for changes that:
  - touch auth/keys
  - touch ingestion/ledger
  - touch kernel interfaces
  - change model routing/manifests

## Kernel Integration (Runtime)

SAPE is implemented as API workflows in the Sovereign Kernel:

- `POST /v1/sape/plan` → compile deterministic SAPE prompts + evidence kernels
- `POST /v1/sape/execute` → run prompts via **sealed model-family routing**, gated by FATE

Runbook:
- `docs/operations/sape_runbook.md`

Evidence & audit:
- Each call emits a receipt under `docs/evidence/receipts/` (configurable).

## Governance Rules (Ihsān/ʿAdl/Amānah)

- **Fail closed:** if FATE cannot validate ethics, the kernel rejects.
- **Evidence-first for high stakes:** if `stakes="H"` and graph evidence is required but unavailable, the request is rejected (BLOCKED).
- **Traceability:** roadmap items produced by SAPE must map to:
  - a finding, requirement, ADR, or receipt
  - a verification step (test/metric) and owner

