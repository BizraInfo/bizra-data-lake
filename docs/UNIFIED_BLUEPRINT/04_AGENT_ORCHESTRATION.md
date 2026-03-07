# Module 04 — Agent Orchestration

> **Domain:** PAT-7, SAT-5, A2A protocol, swarm coordination, harness
> **Source Specs:** V3 Swarm, SAP v0, bizra-harness, Alpha-100
> **Key Paths:** `core/pat/`, `core/swarm/`, `core/a2a/`, `core/orchestration/`, `core/elite/`

## 4.1 PAT-7 (Personal Agent Types)

**Status:** [x] BUILT
**Path:** `core/pat/`

7 personal agent archetypes, each with distinct personality, capabilities,
and interaction patterns. Defined in spec, implemented as configurable agents.

**Spec:** `bizra-normalizers/specs/pat_collection_agent.md`

---

## 4.2 Swarm Engine (3 Strategies)

**Status:** [x] BUILT
**Path:** `core/swarm/engine.py` — `SwarmEngine`

Three coordination strategies (ADR-004):
- **Sequential** — agents execute in order, output feeds next
- **Parallel** — agents execute simultaneously, results merged
- **HierarchicalMesh** — tree structure with delegation and rollup

**Tests:** `tests/core/swarm/`

---

## 4.3 Swarm Event Bridge

**Status:** [x] BUILT
**Path:** `core/swarm/event_bridge.py` — `SwarmEventBridge`

Connects swarm events to sovereign EventBus. Enables monitoring
and constitutional oversight of swarm operations.

---

## 4.4 Agent-to-Agent Protocol (A2A)

**Status:** [x] BUILT
**Path:** `core/a2a/`

Direct agent communication protocol. Agents can discover, negotiate,
and collaborate without central coordinator.

---

## 4.5 Elite Reasoning Module

**Status:** [x] BUILT
**Path:** `core/elite/`

Enhanced reasoning for high-stakes decisions. Applies stronger verification,
multiple hypothesis evaluation, and constitutional double-checks.

Container: bizra-elite (port 8080), 11 agents, ihsan=0.95

---

## 4.6 Orchestration Event System

**Status:** [x] BUILT
**Path:** `core/orchestration/`

Event-driven agent lifecycle management. Handles agent spawn, monitor,
terminate, and health-check cycles.

---

## 4.7 Spearpoint Benchmark Campaigns

**Status:** [x] BUILT
**Path:** `core/spearpoint/`

Strict evaluation campaigns for agent quality. CLEAR-style harness with
ablation-driven architecture optimization.

**Results:** 3 targets, 12/12 gates PASSED, CLEAR 0.836-0.862

---

## 4.8 PersonaPlex Voice System

**Status:** [x] BUILT
**Path:** `core/personaplex/voices.py`

Agent personality and voice configuration. Enables distinct communication
styles per agent type.

---

## 4.9 SAT-5 (System Agent Types)

**Status:** [~] PARTIAL
**Path:** System agents exist implicitly (guardian, dispatcher, kernel) but no
formal SAT-5 taxonomy as specified in SAP v0.
**Gap:** No `SystemAgentCard` type, no formal registration of 5 system agent archetypes

### TDD Anchor
```
def test_sat5_registration():
    registry = SystemAgentRegistry()
    sat_types = registry.list_system_types()
    assert len(sat_types) == 5
    assert "guardian" in [s.name for s in sat_types]
    assert all(s.has_card() for s in sat_types)
```

---

## 4.10 SAP v0 (Sovereign Agent Protocol)

**Status:** [ ] NOT BUILT
**Spec:** `specs/sap-v0/README.md`
**9 canonical types:** SovereignAgentCard, PermitEnvelope, ConsentReceipt,
InteractionReceipt, TaskReceipt, ResourceAllocation, ReputationDelta,
GovernanceVote, AuditBundle
**Gap:** Zero code. Only spec and 2 validation scripts exist.

### Pseudocode
```
@dataclass
class SovereignAgentCard:
    """Identity card for any agent in the BIZRA ecosystem"""
    agent_id: str               # Ed25519 public key
    agent_type: str             # PAT-7 or SAT-5 type
    capabilities: List[str]     # Declared capabilities
    tier: SovereigntyTier       # SEED/SPROUT/TREE/FOREST
    constitutional_hash: str    # Hash of constitutional version
    created_at: datetime
    signature: bytes            # Self-signed with agent's private key

    def verify(self) -> bool:
        return ed25519_verify(self.agent_id, self.serialize(), self.signature)

@dataclass
class PermitEnvelope:
    """Authorization for an agent to perform an action"""
    permit_id: str
    agent_card: SovereignAgentCard
    action: str
    scope: str                  # Namespace/resource scope
    expires_at: datetime
    issuer_signature: bytes     # Signed by authorizing agent
    fate_gate_result: FATEResult  # Must pass FATE before issuance
```

### TDD Anchors
```
def test_agent_card_creation_and_verification():
    card = SovereignAgentCard.create(agent_type="PAT-researcher")
    assert card.verify()
    assert card.tier == SovereigntyTier.SEED  # Starts at SEED

def test_permit_requires_fate_gate():
    card = SovereignAgentCard.create(agent_type="PAT-researcher")
    with pytest.raises(ConstitutionalViolation):
        PermitEnvelope.issue(card, action="delete_data", scope="*")
        # delete_data on wildcard scope should fail FATE

def test_consent_receipt_chain():
    receipt = ConsentReceipt.create(
        subject="user_a", action="data_access",
        granted_to="agent_b", duration_hours=24
    )
    assert receipt.verify_chain()
    assert receipt.expires_at > datetime.now()
```

---

## 4.11 BIZRA-Harness (Unified QA Framework)

**Status:** [ ] NOT BUILT
**Spec:** `specs/bizra-harness/`
**Gap:** Zero implementation. Spec defines a unified test/benchmark/audit harness.

### Pseudocode
```
class BIZRAHarness:
    """Unified QA framework for agent evaluation"""

    def __init__(self):
        self.suites = {
            "constitutional": ConstitutionalSuite(),
            "performance": PerformanceSuite(),
            "security": SecuritySuite(),
            "integration": IntegrationSuite(),
        }

    def run_full_audit(self, target: Agent) -> AuditReport:
        results = {}
        for name, suite in self.suites.items():
            results[name] = suite.evaluate(target)
        return AuditReport(
            target=target.id,
            results=results,
            overall_score=self._compute_composite(results),
            passed=all(r.passed for r in results.values())
        )
```

---

## 4.12 Alpha-100 Go-Live Gate

**Status:** [~] PARTIAL
**Spec:** `specs/alpha100-sprint3/` — AHK Bridge + MCP Transport + Audit + Key Vault
**Built:** AHK bridge exists, MCP transport exists, some audit capability
**Gap:** No unified Alpha-100 checklist gate, no Key Vault implementation

### TDD Anchor
```
def test_alpha100_readiness_gate():
    gate = Alpha100ReadinessGate()
    result = gate.evaluate()
    assert result.ahk_bridge == "PASS"
    assert result.mcp_transport == "PASS"
    assert result.audit_trail == "PASS"
    assert result.key_vault == "PASS"  # NOT YET BUILT
    assert result.overall == "GO"
```

---

## 4.13 Swarm Consensus Protocol

**Status:** [~] PARTIAL
**Path:** Swarm engine coordinates, but no formal BFT consensus among agents
**Gap:** No Byzantine fault tolerance for agent disagreement resolution

### TDD Anchor
```
def test_swarm_consensus_with_byzantine_agent():
    swarm = SwarmEngine(strategy="HierarchicalMesh")
    agents = [honest_agent() for _ in range(4)] + [byzantine_agent()]
    result = swarm.reach_consensus(agents, query="evaluate_proposal")
    assert result.consensus_reached  # 4/5 honest = quorum
    assert result.excluded_agents == 1  # Byzantine detected
```

---

## Completion

| Feature | Status | Coverage |
|---------|--------|----------|
| 4.1 PAT-7 | BUILT | Full |
| 4.2 Swarm Engine | BUILT | 3 strategies |
| 4.3 Swarm Bridge | BUILT | EventBus |
| 4.4 A2A Protocol | BUILT | Full |
| 4.5 Elite Reasoning | BUILT | Container |
| 4.6 Orchestration | BUILT | Lifecycle |
| 4.7 Spearpoint | BUILT | 12/12 gates |
| 4.8 PersonaPlex | BUILT | Voices |
| 4.9 SAT-5 | PARTIAL | Implicit |
| 4.10 SAP v0 | NOT BUILT | Spec only |
| 4.11 BIZRA-Harness | NOT BUILT | Spec only |
| 4.12 Alpha-100 Gate | PARTIAL | No vault |
| 4.13 Swarm Consensus | PARTIAL | No BFT |
| **TOTAL** | **8/13 + 3P + 2N** | **69%** |
