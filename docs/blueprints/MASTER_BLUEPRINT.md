# BIZRA Master Blueprint (SAPE × PMBOK × DevOps × Ihsān)

**High-SNR entrypoint:** `docs/blueprints/UNIFIED_ACTION_FRAMEWORK.md`

**Scope:** Convert the current repo from a demo scaffold into an audit-grade, production-hardened dual-agentic runtime with verifiable ethics, evidence, and delivery discipline.  
**Ihsān constraint:** “No assumptions—only verified excellence.” Every “VERIFIED” claim must link to evidence (`path:line`) or to a reproducible command output captured into the evidence ledger.

---

## 0) Intent Gate (SAPE Module 1)

### What exists (VERIFIED)
- Rust dual-agentic scaffold with HTTP API and PAT/SAT orchestration (`src/lib.rs:20`, `src/http.rs:22`, `src/bridge.rs:27`).
- SAT quorum threshold is implemented as `approvals >= 3` (`src/sat.rs:70`) and Bridge halts on failed consensus (`src/bridge.rs:40`).
- Core “agents/tools/reasoning” are currently **simulated** (non-binding validators, stubbed MCP/A2A, canned reasoning) (`src/sat.rs:118`, `src/mcp.rs:66`, `src/a2a.rs:62`, `src/reasoning.rs:58`).
- Workspace + vault contracts exist to prevent repo split-brain and path drift (`.bizra/workspace.yaml:1`, `.bizra/vault.yaml:1`, `scripts/resolve-bizra-root.ps1:1`, `scripts/resolve-bizra-vault.ps1:1`).
- Tracked configs avoid plaintext DB passwords (passwordless DSN + explicit warning) (`.bizra/workspace.yaml:12`, `scripts/resolve-bizra-root.ps1:94`).
- Hook execution is allowlisted and runs via `execFile` (no arbitrary shell execution) (`.bizra-kernel/kernel-entry.ts:124`, `.bizra-kernel/kernel-entry.ts:159`).
- Evidence tooling blocks `Invoke-Expression` and uses safe scriptblocks (`tools/phase0_audit.ps1:48`).

### Recent hardening + activation tooling (MEASURED)
- HTTP server binds to loopback, requires token for execution endpoints, and restricts CORS by default (`src/http.rs:40`, `src/http.rs:120`, `src/http.rs:61`).
- Personal agentic team “evidence run” exists: deterministic audits + safe ingests + LLM role synthesis (`tools/run_master.ps1:1`, `tools/activate_team.ps1:1`, `ace-framework/team-runner.js:1`).

### What this blueprint builds (PLANNED → to be verified)
- A **verifiable** dual-agentic runtime where:
  - PAT produces structured claims + artifacts (not just text).
  - SAT verifies using tools, policies, and evidence receipts.
  - Ihsān scoring is **definitionally aligned** to a published formula and enforced as a gate (not a label).

---

## 1) Architecture: “As-Is” vs “To-Be”

### 1.1 As-Is Runtime Graph (VERIFIED)
```mermaid
flowchart LR
  HTTP[Axum HTTP API<br/>src/http.rs:22] --> CORE[MetaAlphaDualAgentic<br/>src/lib.rs:20]
  CORE --> BRIDGE[BridgeCoordinator<br/>src/bridge.rs:12]
  BRIDGE --> SAT[SATOrchestrator<br/>src/sat.rs:19]
  BRIDGE --> PAT[PATOrchestrator<br/>src/pat.rs:19]
  HTTP --> EPAT[EnhancedPATOrchestrator<br/>src/pat_enhanced.rs:14]
  EPAT --> MCP[MCPClient (stubbed)<br/>src/mcp.rs:42]
  EPAT --> A2A[A2AServer (stubbed)<br/>src/a2a.rs:30]
  EPAT --> REASON[MultiMethodReasoning (stubbed)<br/>src/reasoning.rs:11]
```

### 1.2 Key architecture gaps (VERIFIED)
- “Parallel” execution is sequential loop today (`src/pat.rs:65`, `src/pat.rs:76`).
- SAT validators always approve (`src/sat.rs:118`) → quorum exists but doesn’t protect.
- HTTP server is **localhost-only** and token-gated by default, but still lacks a production edge (TLS, rate limits, WAF) for public exposure (`src/http.rs:40`, `src/http.rs:120`).
- “Ihsān score” is currently `(avg_confidence + consistency)/2` (`src/bridge.rs:111`) and **does not match** kernel’s published formula (`.bizra-kernel/memory.json:26`).

### 1.3 To‑Be Reference Architecture (Blueprint)
**Design goals:** Replace simulated components with real adapters, explicit trust boundaries, and verifiable gates.

**Core boundaries**
- `core-orchestrator` (Rust): request routing, workflow engine, state machine, deadlines.
- `policy-engine` (Rust): Ihsān + safety rules as explicit, testable predicates.
- `tool-runtime` (Rust): MCP/A2A/FS/DB adapters behind allowlisted capabilities.
- `evidence-ledger` (Rust+PS): append-only event receipts + deterministic sealing.
- `api-gateway` (Rust/Reverse proxy): auth, rate limits, request size/time budgets.
- `ui` (optional): dashboards and evidence browser.

**Non-negotiable contracts**
- **Workspace Contract:** one canonical `.bizra/workspace.yaml` format, with environment overrides, never hardcoded secrets.
- **Evidence Contract:** every critical decision emits a signed receipt: `{inputs, outputs, policy_version, tool_calls, hashes}`.
- **Policy Contract:** “allowed” is provable; “refused” is explainable; “uncertain” is quarantined.

---

## 2) Security Blueprint (Ihsān = Amanah + Adl)

### 2.1 Immediate blockers (fix first)
- Remove secrets from repo and scripts; keep only passwordless DSNs and env-based secrets (`.bizra/workspace.yaml:12`).
- Keep HTTP API localhost-only by default; use token-gated execution endpoints and restrictive CORS (`src/http.rs:40`, `src/http.rs:120`, `src/http.rs:61`).
- Lock down hook execution to an allowlist and safe runner (`.bizra-kernel/kernel-entry.ts:120`).

### 2.2 Threat model (minimum)
- **Assets:** evidence packs, DSNs/keys, model prompts, user content, policy configs.
- **Attack surfaces:** HTTP endpoints, hook execution, workspace path resolution, tool adapters, logs.
- **Primary risks:** prompt injection → unsafe tool calls; RCE via hooks; secret exfiltration; evidence tampering.

### 2.3 Security controls (target state)
- Secrets: environment-only + secret scanning in CI; no plaintext in YAML.
- AuthN/Z: API key or OAuth for gateway; RBAC for tool scopes.
- Tooling sandbox: capability allowlists; deny-by-default; per-tool timeouts.
- Auditability: immutable logs + seal tags (build on `seal_evidence.ps1:1`).

---

## 3) Performance & Reliability Blueprint (SLO-driven)

### 3.1 Principles
- Define SLOs first; optimize to them; prove via load tests + profiling.
- Enforce budgets: payload size, max tool calls, total execution time, concurrency caps.

### 3.2 Minimum SLIs/SLOs (v0)
- API availability: `>= 99.5%` monthly (local single-node is fine; document limitations).
- P95 latency (non-tool requests): target `<= 200ms` (measured).
- Tool-call success rate: `>= 99%` for allowlisted tools (measured).
- Evidence completeness: `100%` of requests emit a receipt (measured).

### 3.3 Engineering tasks
- Implement real concurrency for PAT using bounded task spawning + timeouts (replace sequential loop in `src/pat.rs:65`).
- Add request deadlines propagated across SAT/PAT/tool calls.
- Backpressure: per-user + global rate limiting at gateway; queue with bounded memory.

---

## 4) Documentation & Evidence Blueprint (High‑SNR)

### 4.1 Fix drift first (VERIFIED drift)
- Kernel docs reference paths not in this repo (`.bizra-kernel/README.md:5`, `.bizra-kernel/SYSTEM-OVERVIEW.md:5`).
- Canonical config says `git_toplevel` but workspace points outside (`.bizra/config/node.yaml:9`, `.bizra/workspace.yaml:5`).

### 4.2 “Truth labeling” rule
- Every diagram/page must mark each claim as one of: `VERIFIED`, `MEASURED`, `DERIVED`, `PLANNED`.
- “Architecture Atlas” statuses must be backed by evidence links, not adjectives.

### 4.3 Doc spine (recommended)
- `docs/devops/00_INDEX.md` (runbooks + contracts)
- `docs/security/00_THREAT_MODEL.md`
- `docs/architecture/00_REFERENCE_ARCHITECTURE.md`
- `docs/quality/00_SLOS_AND_GATES.md`
- `docs/evidence/00_EVIDENCE_LEDGER.md`
- `docs/adr/` (keep; expand)

---

## 5) DevOps + CI/CD Blueprint (Pipeline as Policy)

### 5.1 Pipeline stages (recommended)
1. **Pre-merge (CI):** format, lint, unit tests, dependency audit, secret scan.
2. **Build:** reproducible builds; artifacts + SBOM.
3. **Attest:** sign artifacts + attach evidence tag/attestation.
4. **Deploy:** progressive rollout gates (shadow → canary → prod).
5. **Observe:** SLO checks + auto-rollback + incident capture.

### 5.2 “Rare-path prober” automation (SAPE Module 4)
- Negative tests: invalid payloads, timeouts, partial tool failures.
- Adversarial tests: prompt injection attempts against tool runtime.
- Chaos tests: tool adapter unavailable; DB latency spikes; disk full.

---

## 6) Ihsān Implementation (Make it Executable)

### 6.1 Define Ihsān as a gate, not a slogan
- Publish the scoring function and version it.
- Enforce a threshold: requests below threshold are quarantined with a receipt.

**Current mismatch to resolve**
- Kernel formula: `ihsan = 0.3*correctness + 0.3*safety + 0.2*efficiency + 0.2*user_benefit` (`.bizra-kernel/memory.json:26`).
- Rust runtime currently uses confidence/variance proxy (`src/bridge.rs:111`).

### 6.2 Adl/Amanah guardrails (minimum)
- **Adl (justice):** log and monitor refusal/approval rates by request class; detect bias.
- **Amanah (trust):** never store secrets in plain text; sign evidence; minimize data retention.

---

## 7) SAPE → Engineering Translation (How to operationalize “untapped capacity”)

- **Intent Gate →** PRD + threat model + explicit non-goals.
- **Cognitive Lenses →** review roles: Arch / Sec / Perf / QA / Docs / Ops / Ethics.
- **Knowledge Kernels →** ADRs + evidence ledger receipts.
- **Rare-Path Prober →** adversarial + chaos + negative testing gates.
- **Symbolic Harness →** typed policies, invariants, and tool capability contracts.
- **Abstraction Elevator →** “micro→meso→macro” dashboards: code metrics → service SLO → governance KPIs.
- **Tension Studio →** track tradeoffs explicitly: speed vs safety; autonomy vs control; openness vs privacy.

---

## 8) Prioritized Roadmap (High‑SNR, dependency-aware)

### Phase 0 (0–7 days): Stop the bleeding (Critical)
- Remove secrets from repo (done) + add CI secret scanning gate (todo) (ref: `.bizra/workspace.yaml:12`, `tools/activate_team.ps1:60`).
- Bind API to localhost + add minimal auth token + restrictive CORS (`src/http.rs:34`, `src/http.rs:38`).
- Disable/lockdown hook execution until allowlisted runner exists (`.bizra-kernel/kernel-entry.ts:120`).
- Replace `Invoke-Expression` evidence capture with safe command invocation (`tools/phase0_audit.ps1:42`).

### Phase 1 (1–3 weeks): Make SAT real (High)
- SAT validators must be functional checks (not `true`) (`src/sat.rs:118`).
- Implement a policy engine with versioned rules and deterministic evaluation receipts.
- Align Ihsān scoring to published formula and enforce threshold (see mismatch above).

### Phase 2 (3–6 weeks): Tool runtime + evidence ledger (High)
- Implement MCP tool calls as real adapters with allowlists + timeouts (replace stubs in `src/mcp.rs:66`).
- Evidence ledger: append-only receipts + sealing automation (`seal_evidence.ps1:1`).

### Phase 3 (6–10 weeks): Performance + Ops excellence (Medium)
- True PAT concurrency + backpressure + deadlines (replace sequential loop `src/pat.rs:65`).
- Observability: structured tracing, metrics, SLO dashboards.
- Resilience tests + chaos suite (Rare-Path Prober).

### Phase 4 (10–16 weeks): Governance + progressive delivery (Medium)
- ADR discipline + release gates + canary workflow.
- “Atlas” becomes evidence-linked (no “verified” without proofs).

---

## 9) PMBOK Alignment (Project Governance)

### Initiating
- Project Charter: scope, success metrics (SLOs), ethical constraints, risk appetite.

### Planning
- WBS by workstream: Architecture / Security / Tooling / Evidence / QA / Docs.
- Risk Register: split-brain roots (`.bizra/workspace.yaml:5`), RCE hooks (`.bizra-kernel/kernel-entry.ts:132`), secret leakage, public API exposure.
- Quality Plan: gates, tests, SLO checks, evidence requirements.

### Executing
- Deliver per phase, ship behind gates, generate evidence automatically.

### Monitoring & Controlling
- Track SLIs/SLOs; regression budgets; security findings; evidence completeness.

### Closing
- Seal releases; publish attestations; finalize runbooks; postmortems for incidents.

---

## 10) “First 5 Commits” (Concrete next step)
1. Secrets removal + `.env.example` + CI secret scan gate.
2. HTTP hardening (localhost bind, CORS, auth token).
3. Hook runner lockdown (allowlist; no arbitrary `exec`).
4. Ihsān score alignment + enforcement + receipt schema.
5. Replace simulated SAT validation with real checks + negative tests.

---

## 11) Agent Experts (Runtime Learning)

To operationalize “agents that execute and learn” (not just execute and forget), adopt the **Agent Expert** pattern: a high‑SNR expertise model per domain + an Ihsān‑gated self‑improvement loop that is continuously validated against the codebase and sealed evidence.

See: `docs/blueprints/AGENT_EXPERTS_RUNTIME_LEARNING.md`

---

## 12) Execution Backlog (Roadmap)

Use the evidence-gated roadmap and machine-readable backlog to drive implementation without drift:
- `docs/blueprints/ROADMAP_EXECUTION_BACKLOG.md`
- `docs/blueprints/backlog_v1.yaml`

---

## 13) Federated Hybrid Stance (Centralization Dilemma)

Translate the centralization-vs-decentralization tension into implementable contracts (identity, policy, receipts, quarantine, interop) and an architecture that survives multiple futures:
- `docs/blueprints/FEDERATED_AGI_HYBRID.md`

---

## 14) SDLC Artifact Pack (ISO/IEEE/CMMI Alignment)

Turn audits and "enterprise-grade" claims into reproducible artifacts:
- `docs/process/00_INDEX.md`
- SDLC audit intake and evidence labeling: `docs/blueprints/SDLC_AUDIT_RECONCILIATION.md`
- `docs/requirements/requirements_v1.yaml`
- `docs/slo/service_level_objectives_v1.yaml`
- `docs/operations/00_INDEX.md`
- `docs/security/threat_model.md`
- `docs/privacy/data_retention_policy.md`
- `docs/data-migration/alpha_to_beta.md`
- `docs/finops/cost_management.md`
- `docs/accessibility/wcag_2_1_aa.md`
- `docs/retirement/sunset_policy.md`
