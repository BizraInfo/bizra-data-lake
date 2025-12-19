# PAT (Personal Autonomy Team) – Roles (v0.1)

Use this as a **human-friendly** mapping for your local workflow. You can later automate each role with scripts.

## PAT – Builders
1) **Architect**: defines modules, interfaces, constraints, acceptance criteria
2) **Integrator**: wires UI/API/DB/observability with minimal coupling
3) **Data Curator**: catalogs datasets, cleans, labels, classifies, retention
4) **Automation Engineer**: makes scripts, pipelines, reproducible runs
5) **Performance Engineer**: load tests, profiling, caching, budgets
6) **UX Engineer**: console flows, accessibility, RUM, design tokens
7) **Release Captain**: versioning, changelog, rollout, rollback

## SAT – Auditors
1) **Security Auditor**: secret scanning, dependency audit, threat model updates
2) **QA Lead**: test coverage, E2E, regression gates
3) **SRE**: SLOs, error budgets, DR drills, backup/restore proof
4) **Policy Gate**: truth labels, evidence receipts, compliance checklist

## “Receipts” principle
Every critical action produces a receipt:
- who (actor)
- what (action)
- why (intent)
- inputs/outputs
- evidence links (logs, command output, artifacts)
