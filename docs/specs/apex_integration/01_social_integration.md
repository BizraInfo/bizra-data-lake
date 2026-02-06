# Social Integration Specification

## Phase 1: SocialGraph ↔ Dual-Agentic Bridge

**Module**: `core/sovereign/social_integration.py`
**Dependencies**: `core/apex/social_graph.py`, `core/sovereign/dual_agentic_bridge.py`
**Lines**: ~200

---

## Functional Requirements

### FR-1: Trust-Based Agent Routing

When delegating tasks to PAT agents, use SocialGraph trust scores to:
- Prefer agents with higher trust for sensitive operations
- Avoid agents with low reliability scores
- Balance load across trusted agents

### FR-2: Collaboration Discovery

Use SocialGraph's collaboration finder to:
- Identify agent pairs that work well together
- Route related tasks to collaborating agents
- Avoid pairing agents with conflict history

### FR-3: Reputation Updates

After task completion:
- Update trust scores based on success/failure
- Record interaction in SocialGraph
- Propagate reputation changes via PageRank

---

## Pseudocode

```
MODULE social_integration

IMPORT SocialGraph FROM core.apex
IMPORT DualAgenticBridge FROM core.sovereign

CLASS SociallyAwareBridge EXTENDS DualAgenticBridge:
    """
    Enhanced bridge that uses social trust for routing.

    Standing on Giants:
    - Granovetter: Weak ties for diverse task routing
    - PageRank: Trust propagation across network
    """

    PROPERTY social_graph: SocialGraph
    PROPERTY trust_weight: FLOAT = 0.3  # How much trust influences routing
    PROPERTY min_trust_threshold: FLOAT = 0.1

    CONSTRUCTOR(self, node_id: STRING):
        SUPER().__init__()
        self.social_graph = SocialGraph(agent_id=node_id)
        self._register_pat_agents_as_peers()

    METHOD _register_pat_agents_as_peers(self):
        """Register all PAT agents in social graph."""
        FOR agent IN self.pat_agents:
            self.social_graph.add_relationship(
                peer_id=agent.id,
                relationship_type=RelationshipType.COLLABORATOR,
                trust_score=0.5  # Start neutral
            )

    METHOD select_agent_for_task(self, task: Task) -> Agent:
        """
        Select best agent considering both capability and trust.

        Algorithm:
        1. Filter agents by capability
        2. Score by: capability_match * (1 - trust_weight) + trust * trust_weight
        3. Select highest scoring agent
        """
        capable_agents = self._filter_by_capability(task)

        IF NOT capable_agents:
            RAISE NoCapableAgentError(task.required_capabilities)

        scored_agents = []
        FOR agent IN capable_agents:
            trust = self.social_graph.get_trust(agent.id)

            # Skip agents below trust threshold
            IF trust < self.min_trust_threshold:
                CONTINUE

            capability_score = self._calculate_capability_match(agent, task)
            combined_score = (
                capability_score * (1 - self.trust_weight) +
                trust * self.trust_weight
            )
            scored_agents.APPEND((agent, combined_score))

        # Sort by score descending
        scored_agents.SORT(KEY=lambda x: x[1], REVERSE=True)

        RETURN scored_agents[0][0]

    METHOD find_collaboration_partners(self, task: Task) -> LIST[Tuple[Agent, Agent]]:
        """
        Find agent pairs that collaborate well for multi-agent tasks.

        Uses Graph-of-Thoughts to explore collaboration possibilities.
        """
        opportunities = self.social_graph.find_collaborations(
            capability_requirements=task.required_capabilities,
            min_synergy=0.6
        )

        RETURN [
            (self._get_agent(opp.agent_id), self._get_agent(opp.partner_id))
            FOR opp IN opportunities
        ]

    ASYNC METHOD report_task_outcome(self, task: Task, agent: Agent, success: BOOL, value: FLOAT):
        """
        Update social graph after task completion.

        Trust Model:
        - Success: trust += 0.05 * (1 - current_trust)  # Diminishing returns
        - Failure: trust -= 0.1 * current_trust          # Proportional penalty
        """
        AWAIT SUPER().report_task_outcome(task, agent, success, value)

        # Record interaction
        self.social_graph.record_interaction(
            peer_id=agent.id,
            interaction_type=InteractionType.TASK_COMPLETION,
            success=success,
            value=value
        )

        # Trigger trust propagation if significant change
        IF ABS(value) > 100:  # High-value task
            self.social_graph.propagate_trust(iterations=5)

---

## TDD Anchors

### Test: Trust-Based Routing

```python
def test_trust_influences_agent_selection():
    """Higher trust agents should be preferred for sensitive tasks."""
    bridge = SociallyAwareBridge(node_id="test")

    # Setup: Two agents with different trust
    bridge.social_graph._relationships["agent-high"] = Relationship(
        trust_score=0.9, reliability_score=0.9
    )
    bridge.social_graph._relationships["agent-low"] = Relationship(
        trust_score=0.3, reliability_score=0.5
    )

    # Both agents have same capability
    task = Task(required_capabilities={"reasoning"})

    # Higher trust agent should be selected
    selected = bridge.select_agent_for_task(task)
    assert selected.id == "agent-high"
```

### Test: Trust Update After Success

```python
async def test_trust_increases_on_success():
    """Successful task completion should increase trust."""
    bridge = SociallyAwareBridge(node_id="test")

    initial_trust = bridge.social_graph.get_trust("agent-1")

    await bridge.report_task_outcome(
        task=mock_task,
        agent=mock_agent,
        success=True,
        value=50.0
    )

    new_trust = bridge.social_graph.get_trust("agent-1")
    assert new_trust > initial_trust
```

### Test: Collaboration Discovery

```python
def test_find_collaboration_partners():
    """Should find agent pairs with high synergy."""
    bridge = SociallyAwareBridge(node_id="test")

    # Setup: Add collaboration history
    for _ in range(10):
        bridge.social_graph.record_interaction(
            peer_id="agent-1",
            other_peer="agent-2",
            interaction_type=InteractionType.COLLABORATION,
            success=True,
            value=100.0
        )

    task = Task(requires_collaboration=True)
    partners = bridge.find_collaboration_partners(task)

    assert len(partners) > 0
    assert ("agent-1", "agent-2") in [(p[0].id, p[1].id) for p in partners]
```

---

## Edge Cases

| Case | Handling |
|------|----------|
| No trusted agents available | Fall back to capability-only routing |
| All agents below trust threshold | Raise warning, use least-bad option |
| Circular trust propagation | Limit iterations, use damping factor |
| New agent (no trust history) | Start at neutral 0.5 trust |

---

## Security Considerations

1. **Trust manipulation**: Rate-limit trust updates to prevent gaming
2. **Sybil attacks**: Verify agent identity before trust assignment
3. **Privacy**: Trust scores are internal, not exposed to agents

---

## Dependencies

```python
# requirements.txt additions
# (none - uses existing dependencies)
```

---

## File Output

**Target**: `core/sovereign/social_integration.py`
**Lines**: ~200
**Imports**:
```python
from core.apex import SocialGraph, RelationshipType, InteractionType
from core.sovereign.dual_agentic_bridge import DualAgenticBridge
```
