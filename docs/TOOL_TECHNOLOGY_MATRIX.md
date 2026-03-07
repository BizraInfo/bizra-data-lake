# BIZRA Tool and Technology Matrix

Last updated: 2026-03-06
Status: recommended enterprise stack

| Domain | Tool / Platform | Version Target | Why |
|---|---|---|---|
| Core runtime | Python | 3.12 | align local and CI baseline |
| Systems core | Rust | 1.88+ | performance and memory safety |
| API framework | FastAPI | 0.115+ class | existing repo alignment |
| ASGI server | Uvicorn | 0.30+ class | current Python edge fit |
| Validation | Pydantic | 2.x | typed contracts |
| Frontend | React | 19 | enterprise UI baseline |
| Frontend build | Vite | 6 | fast build/dev loop |
| Frontend typing | TypeScript | 5.7 | strict typing |
| Relational DB | PostgreSQL | 16 | OLTP source of truth |
| Cache | Redis | 7.2 | ephemeral state and rate limits |
| Messaging | NATS JetStream | 2.10 | durable async eventing |
| Object storage | S3-compatible | current | immutable artifacts |
| Vector retrieval | pgvector | current | operational simplicity first |
| Optional retrieval scale-out | Qdrant | current | only if pgvector ceiling is hit |
| Containerization | Docker | 29.x | existing local/prod parity |
| Orchestration | Kubernetes | 1.30 class | enterprise deployment model |
| Package manager | uv + pip | current | deterministic Python workflows |
| IaC | Terraform | 1.9 | cloud and environment provisioning |
| GitOps | ArgoCD | 2.12 | controlled promotion |
| Packaging | Helm + Kustomize | 3.14 + current | charting + overlays |
| Metrics | Prometheus | 2.51 | existing repo alignment |
| Dashboards | Grafana | 11 | observability and exec views |
| Logs | Loki | 3.x | cost-efficient structured logs |
| Traces | Tempo | 2.x | distributed tracing |
| Telemetry | OpenTelemetry | current | vendor-neutral signals |
| Python tests | pytest | 9.x class | existing repo alignment |
| Property tests | Hypothesis | 6.x | deterministic edge validation |
| UI tests | Playwright | current | browser E2E |
| Perf tests | k6 | current | release benchmark gate |
| Python SAST | Bandit, Semgrep | current | code scanning |
| Dependency audit | pip-audit, cargo-audit | current | supply chain control |
| Container security | Trivy | current | image and FS scanning |
| Secrets management | Vault or cloud KMS + SOPS | current | enterprise secret lifecycle |
| Identity | Entra ID / Okta / Keycloak | current LTS | OIDC/SAML enterprise auth |
| Documentation | MkDocs Material or Docusaurus | current | governed docs portal if expanded |

Licensing notes:

- Terraform BUSL must be reviewed against organizational policy; OpenTofu is acceptable if required.
- Grafana/Loki/Tempo licensing should be reviewed for commercial distribution contexts.
- Redis licensing should be reviewed if commercial redistribution is in scope; operational use is generally acceptable.
