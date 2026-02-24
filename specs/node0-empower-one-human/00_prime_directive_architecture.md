# Phase 00 — Prime Directive & Architecture Overview

> **Version:** 0.1.0 | **Status:** Specification
> **Standing on Giants:** Shannon (information theory) · Lamport (distributed ordering) · Boyd (OODA) · Al-Ghazali (Ihsan ethics) · Anthropic (constitutional AI) · Besta (Graph-of-Thoughts)

## 0.1 Prime Objective

**Before "8 billion humans," prove "1 node → 1 human."**

Node0 succeeds if and only if it **measurably** increases one user's:

| Dimension | Metric | Baseline Source |
|-----------|--------|-----------------|
| Clarity | Decisions made with evidence vs. intuition | Day 0 self-report |
| Execution Velocity | Tasks shipped / week | Day 0 task log |
| Truthfulness | % outputs with receipts, 0% unverified speculation | Receipt pipeline |

**Constraint:** Zero cloud dependency for core empowerment loop.

## 0.2 System Boundary Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         USER DEVICE (any OS)                           │
│                                                                        │
│  ┌──────────────────┐    ┌──────────────────────────────────────────┐  │
│  │   HOST APP        │    │         RUNTIME CAPSULE                  │  │
│  │   (Tauri + Rust)  │    │  ┌────────────────────────────────────┐  │  │
│  │                   │    │  │     SOVEREIGN CORE (Layer 0)       │  │  │
│  │  - Installer      │◄──►│  │  Ihsan Gate │ SNR Engine │ PCI    │  │  │
│  │  - Tray App       │    │  ├────────────────────────────────────┤  │  │
│  │  - Wallet UI      │    │  │     PROACTIVE ENGINE (Layer 1)    │  │  │
│  │  - Resource Slider│    │  │  Muraqabah │ Opportunity │ Autonomy│  │  │
│  │  - Onboarding     │    │  ├────────────────────────────────────┤  │  │
│  │                   │    │  │     DUAL-AGENTIC TEAM (Layer 2)   │  │  │
│  │                   │    │  │  PAT (7 agents) │ SAT (5 validators│  │  │
│  │                   │    │  ├────────────────────────────────────┤  │  │
│  │                   │    │  │     NODE0 ORCHESTRATION (Layer 3)  │  │  │
│  │                   │    │  │  OODA Loop │ Planner │ Checkpoint  │  │  │
│  │                   │    │  └────────────────────────────────────┘  │  │
│  └──────────────────┘    └──────────────────────────────────────────┘  │
│                                                                        │
│  ┌──────────────────┐    ┌──────────────────────────────────────────┐  │
│  │  LOCAL LLM        │    │  ENCRYPTED LOCAL STORE                   │  │
│  │  (Ollama/LMStudio)│    │  Wallet │ Receipts │ Impact Ledger      │  │
│  └──────────────────┘    └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
```

## 0.3 Constraint Lattice (Non-Negotiables)

```
CONSTRAINT_LATTICE:
  constitutional:
    IHSAN_FLOOR:        import from core.integration.constants.UNIFIED_IHSAN_THRESHOLD  # 0.95
    IHSAN_STRICT:       import from core.integration.constants.STRICT_IHSAN_THRESHOLD   # 0.99
    SNR_FLOOR:          import from core.integration.constants.UNIFIED_SNR_THRESHOLD     # 0.85
    SNR_T1:             import from core.integration.constants.SNR_THRESHOLD_T1_HIGH     # 0.95
    DAUGHTER_TEST:      enabled = true
    ZANN_ZERO:          speculation_blocked = true, unverified_downgraded = true

  operational:
    INSTALL_TIME_MAX:   600 seconds (10 minutes)
    TERMINAL_REQUIRED:  false
    CLOUD_DEPENDENCY:   none (for core loop)
    LOCAL_FIRST:        true (inference, storage, receipts)

  security:
    SECRETS_HARDCODED:  never
    STORE_ENCRYPTED:    AES-256-GCM or XChaCha20-Poly1305
    WALLET_LOCAL:       Ed25519 keypair, never transmitted
    SANDBOX_ENFORCED:   WASM primary, microVM secondary, Docker fallback
```

## 0.4 Module Dependency Graph

```
DEPENDENCY_GRAPH:
  phase_01_installer
    ├── depends_on: nothing (first build target)
    ├── produces:   host_app_binary, runtime_capsule, system_service
    └── enables:    phase_02, phase_03, phase_04

  phase_02_empowerment_loop
    ├── depends_on: phase_01 (runtime must be running)
    ├── consumes:   core.sovereign.ProactiveSovereignEntity
    │               core.sovereign.autonomy_matrix.AutonomyLevel
    │               config.proactive_config.yaml
    ├── produces:   daily_plan, task_breakdown, executed_tasks
    └── enables:    phase_03 (receipts), phase_04 (metrics)

  phase_03_receipt_pipeline
    ├── depends_on: phase_02 (needs outputs to receipt)
    ├── consumes:   core.pci.envelope (PCI protocol)
    │               core.pci.crypto (Ed25519, SHA-256)
    │               core.integration.constants (thresholds)
    ├── produces:   receipt_chain, evidence_pointers
    └── enables:    phase_04 (auditable metrics)

  phase_04_impact_measurement
    ├── depends_on: phase_02 + phase_03
    ├── consumes:   receipt_chain, task_metrics, user_baseline
    ├── produces:   weekly_impact_report, impact_ledger
    └── enables:    "Proof-of-Impact" (the value claim)

  phase_05_build_sequence
    ├── P0: installer → capsule → onboarding → receipts → impact
    └── P1: wallet_ui → resource_slider → p2p_mesh → content_gateway
```

## 0.5 Existing Code Reuse Map

| New Component | Reuses From | Path |
|---------------|-------------|------|
| Ihsan Gate | `core.sovereign.constitutional_gate` | `core/governance/constitutional_gate.py` |
| Autonomy Matrix | `core.sovereign.autonomy_matrix` | `core/sovereign/autonomy_matrix.py` |
| SNR Engine | `core.iaas` | `core/iaas/` |
| PCI Receipts | `core.pci.envelope` | `core/pci/envelope.py` |
| Ed25519 Crypto | `core.pci.crypto` | `core/pci/crypto.py` |
| Thresholds | `core.integration.constants` | `core/integration/constants.py` |
| PAT/SAT Teams | `core.sovereign.collective_intelligence` | `core/sovereign/collective_intelligence.py` |
| Opportunity Pipeline | `core.sovereign.opportunity_pipeline` | `core/sovereign/` |
| OODA Loop | `core.sovereign.autonomy` | `core/sovereign/autonomy.py` |
| Proactive Entity | `core.sovereign.ProactiveSovereignEntity` | `core/sovereign/` |
| Graph-of-Thoughts | `tools.sacred_wisdom_engine.GoTReasoningLayer` | `tools/sacred_wisdom_engine.py` |
| Event Bus | `core.sovereign.event_bus` | `core/sovereign/event_bus.py` |
| Checkpointing | State persistence | `sovereign_state/checkpoints/` |

## 0.6 TDD Anchors — Phase 00

```pseudocode
TEST "prime_directive_constraints_importable":
  FROM core.integration.constants IMPORT
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
    STRICT_IHSAN_THRESHOLD
  ASSERT UNIFIED_IHSAN_THRESHOLD == 0.95
  ASSERT UNIFIED_SNR_THRESHOLD == 0.85
  ASSERT STRICT_IHSAN_THRESHOLD == 0.99

TEST "constraint_lattice_complete":
  lattice = load_constraint_lattice()
  ASSERT "constitutional" IN lattice
  ASSERT "operational" IN lattice
  ASSERT "security" IN lattice
  ASSERT lattice.operational.INSTALL_TIME_MAX <= 600
  ASSERT lattice.operational.CLOUD_DEPENDENCY == "none"
  ASSERT lattice.security.SECRETS_HARDCODED == "never"

TEST "module_dependency_acyclic":
  graph = build_dependency_graph(phases=[1,2,3,4,5])
  ASSERT is_directed_acyclic(graph) == true
  ASSERT topological_sort(graph) == [1, 2, 3, 4, 5]

TEST "existing_code_reuse_paths_valid":
  FOR path IN reuse_map.values():
    ASSERT file_exists(path) == true
```
