# BIZRA Unified Action Framework (Master Synthesis)

**Purpose:** A single high-SNR execution blueprint that fuses architecture, security, performance, documentation, DevOps, and ethics (Ihsān / ʿAdl / Amānah) into one dependency-aware roadmap and delivery system.

**Truth labels:** VERIFIED | MEASURED | DERIVED | TARGET  
**Ihsān rule:** Only mark **VERIFIED** when backed by evidence (`path:line`) or a reproducible command output captured in evidence.

---

## 0) The Pinnacle Next Step (Executive Summary)

**Next step:** Ignite the Brain (Neo4j knowledge graph) **and** expose it as a governed runtime service via the Sovereign Kernel API (token-gated, FATE-enforced), while keeping an evidence-gated SDLC that prevents drift.

**Why this is the correct next step (SNR rationale):**
1) **We already minted roots** (Genesis + ledger chain), so the next high-leverage move is turning static assets into queryable memory.  
2) **Apps need a contract**, not a script: a stable HTTP API (Kernel) is the “nervous system” that lets the 10 apps query wisdom safely.  
3) **We already have delivery gates** (CI + lints + truth-label enforcement), so the next maturity step is binding both the graph *and* the API into those gates and operating them as auditable subsystems.

---

## 1) System Anchors (Immutable / Evidence-Linked)

### 1.1 Genesis Anchor (MEASURED)
- Genesis Block hash: `BIZRA_GENESIS_BLOCK_0.json:2`

### 1.2 Knowledge Ledger Anchor (MEASURED)
- Ledger chain hash: `BIZRA_KNOWLEDGE_MANIFEST.json:3`
- Total artifacts: `BIZRA_KNOWLEDGE_MANIFEST.json:131`
- Total intrinsic value (BZR‑G): `BIZRA_KNOWLEDGE_MANIFEST.json:132`
- Sanitized receipt (aggregates only; no file paths): `evidence/genesis/KNOWLEDGE_MANIFEST_RECEIPT.json:1`
- Tokenomics constitution (declares the same fixed supply): `BIZRA_TOKENOMICS_GENESIS.yaml:11`, `BIZRA_TOKENOMICS_GENESIS.yaml:12`, `BIZRA_TOKENOMICS_GENESIS.yaml:14`

### 1.3 Live Node Capability Snapshot (MEASURED)
Captured host + inference inventory:
- Hardware/OS snapshot: `evidence/audit-results-node0.json:5`
- Ollama reachable + model inventory: `evidence/audit-results-node0.json:21`
- LM Studio reachable + model inventory: `evidence/audit-results-node0.json:69`

### 1.4 Model Fleet Seal (VERIFIED/MEASURED)
- Sealed “model family” routing + pinned identities (Ollama + LM Studio): `model-family-genesis-v1-SEALED.yaml:1`
- CI lint that enforces sealed + pinned + routing consistency: `tools/model_family_lint.py:1`
- Evidence capture script (host/GPU + provider inventories): `capture_evidence.ps1:1`, `evidence/audit-results-node0.json:1`
- Exportable dual-provider seal pack (portable manifests + audit script): `bizra_phase0_week1_seal/README.md:1`, `bizra_phase0_week1_seal.zip`

### 1.5 Citadel Runtime (Docker) + Genesis Receipt (VERIFIED/MEASURED)
This is the bridge from "scripts" to "services": deterministic ignition + acceptance gates + a cryptographically replayable receipt.

- Citadel compose (service-name-first wiring): `docker-compose.yml:1`
- Production-minimal kernel image: `Dockerfile:1`
- Secure env template (no secrets committed): `.env.example:1`, `.gitignore:1`, `.dockerignore:1`
- Deterministic ignition + acceptance gates:
  - Ignite: `scripts/ignite_node0.ps1:1`
  - Verify gates (health stability, restart storm, RO mount, LLM degraded visibility): `scripts/verify_node0.ps1:1`
- Genesis Receipt (replayable ignition evidence):
  - Generator: `scripts/genesis_receipt.py:1`
  - Schema: `schemas/genesis_receipt_v1.schema.json:1`
  - Example: `docs/evidence/genesis_receipt_v1.example.json:1`

---

## 2) Multi-Lens Findings (Synthesis)

This section consolidates what matters across lenses into “decision-grade” insights.

### 2.1 Architecture Lens (VERIFIED/MEASURED/TARGET)
**What is verified in this repo (VERIFIED):**
- There is a master blueprint with evidence gates and explicit gap analysis: `docs/blueprints/MASTER_BLUEPRINT.md:1`.
- There is a dependency-aware execution backlog: `docs/blueprints/backlog_v1.yaml:1`.

**What is measured (MEASURED):**
- The workspace contains a very large mixed asset surface (including binaries and a linked vault); inventory and security signals were captured: `evidence/phase0/codebase_inventory_summary.txt:1`.

**Primary architectural truth (VERIFIED):**
- This is a **system-of-systems**: a runtime repo + a data vault + external node stacks (e.g., Neo4j/Ollama/LM Studio). The roadmap must preserve clear boundaries so audits remain meaningful.

**Architectural gaps to close (TARGET):**
- Make SAT validation non-simulated and enforceable.
- Align Ihsān scoring to the constitution and enforce it as a gate (not a label).
- Turn the knowledge ledger into a queryable graph + retrieval layer.

### 2.2 Security Lens (DERIVED → ACTIONABLE)
**Threat model baseline exists (VERIFIED):**
- Template controls and threats: `docs/security/threat_model.md:1`.

**Heuristic risk signals (MEASURED):**
- Large workspace surface with many risky constructs (signals are *prioritization*, not proof): `evidence/phase0/codebase_inventory_summary.txt:49`.
- Notable counts: `node_eval` and `ps_invoke_expression` appear in the scanned workspace: `evidence/phase0/codebase_inventory_summary.txt:51`, `evidence/phase0/codebase_inventory_summary.txt:53`.

**Security synthesis (DERIVED):**
- The **largest realistic failure mode** is not a single bug; it is *blast-radius coupling*:
  - data vault ↔ code workspace ↔ tool execution surfaces ↔ secrets.
- Therefore, the professional next step is to:
  1) tighten boundaries,  
  2) constrain execution surfaces, and  
  3) require receipts for every side effect.

### 2.3 Performance Lens (TARGET → MEASURED)
**SLO template exists (VERIFIED):**
- Baseline SLO file: `docs/slo/service_level_objectives_v1.yaml:1`.

**Performance synthesis (DERIVED):**
- “State-of-the-art” in this context means **encoded budgets + continuous regression protection**:
  - timeouts per tool call,
  - backpressure limits,
  - known workloads for p95/p99,
  - automated perf smoke checks on every PR (at least “does not regress”).

### 2.4 Documentation & Governance Lens (VERIFIED)
**Evidence‑gated SDLC pack exists (VERIFIED):**
- SDLC index: `docs/process/00_INDEX.md:1`.
- Audit reconciliation policy (avoid false “VERIFIED” claims): `docs/blueprints/SDLC_AUDIT_RECONCILIATION.md:1`.

**Documentation synthesis (DERIVED):**
- The documentation is already structured for audit‑grade delivery; the next step is unifying:
  - “what we believe” (TARGET),
  - “what we measured” (MEASURED),
  - “what we can prove” (VERIFIED),
  into a single operating rhythm (CI + receipts + release seals).

### 2.5 Ethics Lens (Ihsān / ʿAdl / Amānah) (VERIFIED)
**Ihsān constitution exists (VERIFIED):**
- Canonical Ihsān dimensions + weights: `constitution/ihsan_v1.yaml:1`.

**Ethical synthesis (DERIVED):**
- Ihsān is implemented professionally when:
  - constraints are encoded as policy,
  - violations are measurable,
  - “uncertain” routes to quarantine,
  - and receipts preserve accountability (Amānah) and fairness (ʿAdl).

---

## 3) Unified Execution Model (PMBOK × DevOps × Ihsān)

### 3.1 PMBOK Integration Map (Practical)
**Initiating**
- Project charter (scope, success metrics, risk appetite) → `docs/requirements/requirements_v1.yaml:1` (baseline template).

**Planning**
- WBS + backlog + dependencies → `docs/blueprints/backlog_v1.yaml:1`.
- Risk register and cascading risk map → `docs/blueprints/ROADMAP_EXECUTION_BACKLOG.md:1`.

**Executing**
- Trunk-based development with evidence-gated PRs (CI is the gate; receipts are the record).

**Monitoring & Controlling**
- SLOs + error budgets: `docs/slo/service_level_objectives_v1.yaml:1`.
- Continuous integrity checks in CI (see Section 3.2).

**Closing**
- Seal releases and evidence packs (tag + hashes) and publish a release receipt.

### 3.2 DevOps Pipeline Baseline (CI/CD)
**Integrity pipeline exists (VERIFIED):**
- `phase0-integrity` workflow: `.github/workflows/phase0_integrity.yml:1`.

**Blueprint upgrade path (TARGET):**
- Add performance smoke tests (small, deterministic workloads).
- Add supply-chain SBOM + dependency pinning policies.
- Add release workflow (signed tags + manifest receipts + provenance).

---

## 4) “Ignite the Brain” (Knowledge Graph Activation)

### 4.1 Data-to-Brain Pipeline (MEASURED/VERIFIED)
**Refinery outputs (MEASURED):**
- Manifest + ledger chain are already minted: `BIZRA_KNOWLEDGE_MANIFEST.json:3`.

**Graph loader exists (VERIFIED):**
- Neo4j synaptic loader script: `bizra_synaptic_loader.py:1`.
- Neo4j ops runbook (secure defaults + ingestion + queries): `docs/operations/neo4j_runbook.md:1`.

### 4.2 Neo4j Graph Model (Minimal, SNR-first)
**Nodes**
- `:Artifact` — one per ledger record (keyed by `hash`).
- `:KnowledgeManifest` — one per ledger chain (`ledger_chain_sha256`).
- `:GenesisBlock` — one per genesis hash.
- `:FileExtension`, `:AssetClass` — normalized dimensions for query speed.

**Edges**
- `(Artifact)-[:IN_MANIFEST]->(KnowledgeManifest)`
- `(KnowledgeManifest)-[:ANCHORED_TO]->(GenesisBlock)`
- `(Artifact)-[:HAS_EXTENSION]->(FileExtension)`
- `(Artifact)-[:CLASSIFIED_AS]->(AssetClass)`

**Why this model (SNR):**
- It preserves traceability (receipts), enables fast filtering (type/ext), and avoids premature ontology design.

### 4.3 Retrieval Strategy (GoT-ready, evidence-gated)
**Graph-of-Thoughts (GoT) without hidden reasoning:**
- Represent reasoning as a **DAG of claims**, where:
  - each claim node links to artifacts (evidence),
  - counter-claims exist explicitly,
  - SAT validators check claim validity against evidence.

This enables advanced reasoning while preserving Amānah: the system can show *what evidence* supported the output without disclosing private scratch reasoning.

### 4.4 Runtime Interface (Sovereign Kernel) (VERIFIED)
**Kernel service exists (VERIFIED):**
- FastAPI gateway (token-gated): `core/main.py:1`
- FATE gate (fail-closed; ihsan_v1-aligned thresholds): `core/fate.py:1`, `constitution/ihsan_v1.yaml:1`
- Neo4j interface (House of Wisdom): `core/wisdom.py:1`
- Ops runbook: `docs/operations/sovereign_kernel_runbook.md:1`

---

## 5) SAPE Operationalization (Symbolic-Abstraction Probe Elevation)

SAPE is the method for extracting “untapped capacity” from LLMs **safely**:

1) **Symbolic harness:** backlogs, policies, schemas, receipts (machine-checkable).
2) **Abstraction elevator:** link micro (code) ↔ meso (services) ↔ macro (governance + economy).
3) **Probe elevation:** adversarial tests, negative tests, chaos drills, and red-team prompts.
4) **SNR maximization:** ban unverifiable claims; require evidence links; keep docs short and composable.

Primary SAPE artifacts:
- Master blueprint: `docs/blueprints/MASTER_BLUEPRINT.md:1`
- Backlog: `docs/blueprints/backlog_v1.yaml:1`
- Roadmap: `docs/blueprints/ROADMAP_EXECUTION_BACKLOG.md:1`
- Kernel-integrated SAPE API: `core/sape.py:1`, `core/main.py:1`
- SAPE ops runbook: `docs/operations/sape_runbook.md:1`
- Sealed routing (model family): `model-family-genesis-v1-SEALED.yaml:1`

---

## 6) Prioritized Optimization Roadmap (Unified)

This is the unified roadmap across architecture, security, performance, docs, DevOps, ethics, and knowledge.

### Phase 0 (P0): Hardening + Truth Alignment
Canonical backlog epics:
- `EPIC-PH0-SEC` (secrets + scanning gates)
- `EPIC-PH0-API` (harden API defaults)
- `EPIC-PH0-REQ` (requirements/traceability/SLO/RACI)

### Phase 1 (P1): Make SAT Real
Canonical epic:
- `EPIC-PH1-SAT` (real validators + negative tests + quarantine)

### Phase 2 (P1): Evidence Ledger + Tool Runtime + Knowledge Activation
Canonical epics:
- `EPIC-PH2-EVID` (receipt schema + sealing)
- Knowledge activation (add to backlog; see Section 7)

### Phase 3 (P2): Performance + Observability
Canonical epics:
- `EPIC-PH3-OPS` and related SLO/perf regression gates.

### Phase 4 (P2): Agent Experts (Runtime Learning)
Canonical epic:
- `PH4` expert loop (self-improve gated by Ihsān + SAT).

---

## 7) The Immediate “Elite Practitioner” Work Package (Next 7 Days)

**Objective:** Convert the system from “functional + evidence-aware” to “audit-grade + graph-activated”.

0) **Ignite the Citadel (Docker) and mint a Genesis Receipt**:
   - Bring up services: `scripts/ignite_node0.ps1:1`
   - Run deterministic acceptance gates: `scripts/verify_node0.ps1:1`
   - Generate replayable receipt (no secrets): `scripts/genesis_receipt.py:1`
1) **Operationalize Neo4j** (local-only; secure defaults; backups plan) and ingest the ledger via `bizra_synaptic_loader.py`.
2) **Bind graph ingestion to evidence**:
   - verifies ledger chain before write: `bizra_synaptic_loader.py:225`,
   - creates uniqueness constraints by default: `bizra_synaptic_loader.py:249`,
   - emits an ingestion receipt by default: `bizra_synaptic_loader.py:341`, `docs/operations/neo4j_runbook.md:82`.
3) **Activate the Sovereign Kernel (runtime interface)**:
   - token-gated query endpoint + metrics: `core/main.py:1`
   - FATE gate (fail-closed; ihsan_v1-aligned): `core/fate.py:1`, `constitution/ihsan_v1.yaml:1`
   - runbook (start/verify + receipts): `docs/operations/sovereign_kernel_runbook.md:1`
4) **Upgrade gates**:
   - make “uncertain” a first-class outcome,
   - require receipts for every side-effect tool call,
   - start capturing perf smoke metrics into `docs/evidence/`.

---

## 8) Definition of Done (Unified, World-Class)

**Security**
- No tracked secrets; secret scan gate enforced in CI.
- Tool execution surfaces allowlisted and time-bounded.

**Quality**
- Negative tests cover SAT rejection + quarantine + timeout paths.
- Evidence receipts validate against schema and can be sealed.

**Operations (DevOps)**
- `/livez` always 200 and `/healthz` fails-closed for core deps under strict time budget: `core/main.py:317`, `core/main.py:576`.
- Citadel acceptance gates pass deterministically: `scripts/verify_node0.ps1:1`.
- A replayable genesis receipt is minted post-gates (no secrets): `scripts/genesis_receipt.py:1`.

**Performance**
- SLOs defined; at least one workload is MEASURED and captured.
- Perf regressions are detected automatically.

**Documentation**
- Truth labels used consistently; VERIFIED claims always have evidence.

**Ethics**
- Ihsān scoring is definitionally aligned to constitution; enforced as a gate.
- Adl metrics exist (approval/refusal distributions by request class).
- Amānah is operational: secrets protected, receipts signed, retention minimized.
