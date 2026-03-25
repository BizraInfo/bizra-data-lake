# Phase 1: Sovereignty Axiom — Identity Bifurcation

**Spec:** CMN-001
**Status:** Gap-fill (existing: PARTIAL)
**Formal Property:** For every node n, private workspace Omega_n intersect URP = empty set
**Existing Code:** `core/urp/membrane.py`, `bizra-core/src/lib.rs` (PAT/SAT split), `bizra-core/identity/`

---

## 1. Objective

Prove and enforce that user-local state (PAT_7 execution, private memory, local reflexes)
is **topologically disjoint** from URP-shared state (SAT_5 services, shared ledger, consensus).

Currently the membrane *filters crossings* but does not formally define or enforce the
workspace boundary. This spec fills that gap.

---

## 2. Definitions

```
Omega_n := {
    local_memory:    Path(node.data_dir / "memory/"),
    local_reflexes:  ReflexCache (in-process, not shared),
    pat_agents:      PAT_7 roster (7 private agents),
    local_receipts:  append-only JSONL on local disk,
    local_models:    Ollama/LMStudio VRAM state,
    local_keys:      Ed25519 signing keys (never leave node)
}

URP := {
    shared_ledger:   SEED token ledger (consensus-verified),
    sat_services:    SAT_5 agents (shared utility pool),
    knowledge_graph: Neo4j provenanced graph,
    consensus_state: BFT agreement records
}

INVARIANT: Omega_n.keys() & URP.keys() == empty_set
```

---

## 3. Pseudocode

### 3.1 WorkspaceBoundary (new: `core/sovereign/workspace_boundary.py`)

```python
class WorkspaceBoundary:
    """Enforces Omega_n disjoint URP at runtime."""

    def __init__(self, node_id: str, data_dir: Path):
        self._node_id = node_id
        self._omega = self._enumerate_local_state(data_dir)
        self._urp_namespaces = frozenset([
            "shared_ledger", "sat_pool", "knowledge_graph", "consensus"
        ])

    def check_disjoint(self) -> bool:
        """INVARIANT: local namespace keys never overlap URP keys."""
        overlap = self._omega.keys() & self._urp_namespaces
        return len(overlap) == 0

    def guard_outbound(self, payload: dict) -> dict:
        """Strip private fields before membrane crossing."""
        PRIVATE_FIELDS = {"signing_key", "local_memory", "reflex_cache"}
        return {k: v for k, v in payload.items() if k not in PRIVATE_FIELDS}

    def guard_inbound(self, payload: dict) -> dict:
        """Reject URP payloads that attempt to write local-only namespaces."""
        LOCAL_ONLY = {"pat_roster", "local_receipts", "local_models"}
        for key in payload:
            if key in LOCAL_ONLY:
                raise SovereigntyViolation(f"URP cannot write to {key}")
        return payload
```

### 3.2 Linear Scaling Verification

```python
def verify_linear_scaling(nodes: list[Node]) -> ScalingResult:
    """V(N) = sum(SAT_5_i for i in 1..N) — linear, not sublinear."""
    capacities = [node.sat_verification_capacity() for node in nodes]
    total = sum(capacities)
    expected_linear = len(nodes) * BASELINE_SAT_CAPACITY

    # Allow 5% variance for heterogeneous hardware
    ratio = total / expected_linear if expected_linear > 0 else 0.0
    is_linear = 0.95 <= ratio <= 1.05

    return ScalingResult(
        node_count=len(nodes),
        total_capacity=total,
        is_linear=is_linear,
        ratio=ratio,
    )
```

---

## 4. TDD Anchors

```python
# tests/core/test_sovereignty_axiom.py

def test_omega_urp_disjoint():
    """Axiom 1.1: Omega_n and URP share no namespace keys."""
    boundary = WorkspaceBoundary("node0", Path("/tmp/test-node"))
    assert boundary.check_disjoint() is True

def test_outbound_strips_private_fields():
    """Private keys, local memory never cross the membrane."""
    boundary = WorkspaceBoundary("node0", Path("/tmp/test-node"))
    payload = {"query": "hello", "signing_key": "PRIVATE", "reflex_cache": {}}
    clean = boundary.guard_outbound(payload)
    assert "signing_key" not in clean
    assert "reflex_cache" not in clean
    assert clean["query"] == "hello"

def test_inbound_rejects_local_namespace_writes():
    """URP cannot inject into PAT roster or local receipts."""
    boundary = WorkspaceBoundary("node0", Path("/tmp/test-node"))
    with pytest.raises(SovereigntyViolation):
        boundary.guard_inbound({"pat_roster": "hijack"})

def test_linear_scaling_capacity():
    """V(N) scales linearly with node count."""
    nodes = [MockNode(capacity=100) for _ in range(1000)]
    result = verify_linear_scaling(nodes)
    assert result.is_linear is True
    assert result.ratio == pytest.approx(1.0, abs=0.05)

def test_single_node_is_sovereign():
    """A lone node has full sovereignty — no URP dependency."""
    boundary = WorkspaceBoundary("node0", Path("/tmp/solo"))
    assert boundary.check_disjoint() is True
    # PAT_7 operates without URP connection
```

---

## 5. Integration Points

| Existing Module | Integration |
|----------------|-------------|
| `core/urp/membrane.py` | `guard_outbound()` called before `filter_outbound()` |
| `core/urp/service.py` | `guard_inbound()` called on `NodeRegistration` payloads |
| `bizra-core/identity/` | Ed25519 keys are in Omega_n, never serialized to URP |
| `bizra-agent/persistence.rs` | Reflex store is Omega_n — content-addressed, local only |

---

## 6. Formal Statement (Z3-ready)

```
; SMT-LIB2 sketch
(declare-sort Namespace)
(declare-fun omega (Namespace) Bool)
(declare-fun urp (Namespace) Bool)
(assert (forall ((ns Namespace))
    (not (and (omega ns) (urp ns)))))
; UNSAT if any namespace is in both => DISJOINT proven
```
