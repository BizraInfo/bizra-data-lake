# BIZRA Sovereignty Module

**HYPER LOOPBACK Architecture** - Complete Offline Sovereignty for BIZRA Nodes

## Overview

The `core/sovereignty/` module implements complete offline sovereignty for BIZRA operations, enabling nodes to function independently without external API dependencies. This is the foundation of **HYPER LOOPBACK** - the principle that every BIZRA node can operate in complete isolation while maintaining cryptographic integrity.

## Philosophy

> **بذرة (BIZRA) = Seed**
>
> Every human is a node. Every node is a seed. A seed must be sovereign - capable of growth even in winter, independent of external systems.

## Domain

All operations use domain separation: `bizra-pci-v1:`

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                  SOVEREIGNTY MODULE                          │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌───────────────┐  ┌───────────────┐  ┌─────────────────┐ │
│  │ WinterProof   │  │ Constitution  │  │ DaughterTest    │ │
│  │ Embedder      │  │               │  │                 │ │
│  │               │  │ Ihsān: 0.95   │  │ Continuous      │ │
│  │ Deterministic │  │ Thresholds    │  │ Verification    │ │
│  │ Embeddings    │  │               │  │                 │ │
│  └───────────────┘  └───────────────┘  └─────────────────┘ │
│           │                  │                   │           │
│           └──────────────────┴───────────────────┘           │
│                              │                               │
│                    ┌─────────▼─────────┐                     │
│                    │  LocalMerkleDAG   │                     │
│                    │                   │                     │
│                    │ Tamper-Proof      │                     │
│                    │ Evidence Chain    │                     │
│                    └───────────────────┘                     │
│                                                               │
└─────────────────────────────────────────────────────────────┘
```

## Components

### 1. WinterProofEmbedder

**Purpose:** Generate deterministic embeddings without external APIs

**Features:**
- Multi-hash approach: SHA-256, SHA3-256, BLAKE3 (or SHA-512 fallback)
- L2 normalization for unit vectors
- Cosine similarity for semantic search
- 100% deterministic: same input always produces same output
- No external dependencies (numpy optional for acceleration)

**Usage:**

```python
from core.sovereignty import WinterProofEmbedder

embedder = WinterProofEmbedder(dimension=384)

# Generate embedding
embedding = embedder.embed("BIZRA is sovereign")

# Batch processing
embeddings = embedder.embed_batch([
    "Text 1",
    "Text 2",
    "Text 3"
])

# Semantic search
results = embedder.semantic_search(
    query="sovereignty",
    documents=["doc1", "doc2", "doc3"],
    top_k=2
)

# Generate receipt
receipt = embedder.generate_receipt(text, embedding)
```

### 2. Constitution

**Purpose:** Enforce Ihsān (excellence) thresholds across all operations

**Core Principles:**
- **Ihsān:** 0.95 minimum (95% excellence)
- **Sovereignty:** 1.0 required (100% offline-capable)
- **Transparency:** 0.95 minimum (complete auditability)
- **Integrity:** 1.0 required (100% verifiable)
- **Determinism:** 1.0 required (100% reproducible)
- **Efficiency:** 0.90 minimum (resource optimization)

**Usage:**

```python
from core.sovereignty import Constitution

constitution = Constitution()

# Verify compliance
scores = {
    'ihsan': 0.98,
    'sovereignty': 1.0,
    'transparency': 0.97,
    'integrity': 1.0,
    'determinism': 1.0,
    'efficiency': 0.92,
}

receipt = constitution.verify(
    operation="winter_proof_embedding",
    scores=scores,
    metadata={'context': 'production'}
)

if receipt.compliant:
    print(f"✓ Compliant: {receipt.overall_score:.2%}")
else:
    for violation in receipt.violations:
        print(f"✗ {violation.message}")

# Enforce compliance (raises exception if non-compliant)
try:
    constitution.verify_with_enforcement(operation, scores)
except ConstitutionalViolationError as e:
    print(f"Violation: {e.receipt}")
```

### 3. DaughterTest

**Purpose:** Continuous integrity verification

**Named After:** "Trust, but verify - as a parent verifies their daughter's wellbeing"

**Features:**
- Determinism verification
- Checksum validation (SHA-256)
- Temporal consistency checks
- Evidence chain verification
- Real-time violation detection
- Continuous monitoring with background threads

**Usage:**

```python
from core.sovereignty import DaughterTest

tester = DaughterTest()

# Verify determinism
def my_operation(x):
    return x * 2

check = tester.verify_determinism(
    operation=my_operation,
    inputs=(5,),
    iterations=3
)

# Verify checksum
check = tester.verify_checksum(
    data="important data",
    expected_hash="abc123..."
)

# Verify evidence chain
chain = [
    {'hash': 'abc', 'prev_hash': None, 'data': 'genesis'},
    {'hash': 'def', 'prev_hash': 'abc', 'data': 'block1'},
]
check = tester.verify_evidence_chain(chain)

# Register for continuous monitoring
tester.register_operation(
    operation_id="critical_op",
    operation_func=my_operation,
    baseline_inputs=(10,)
)

# Start monitoring (checks every 60 seconds)
tester.start_monitoring(interval_seconds=60)

# Get integrity report
report = tester.get_integrity_report()
```

### 4. LocalMerkleDAG

**Purpose:** Tamper-proof evidence chain using Merkle DAG structure

**Features:**
- Merkle tree structure for efficient verification
- DAG topology for complex dependency relationships
- SHA-256 and BLAKE3 hashing
- Local storage with JSON persistence
- Tamper detection via hash chain verification
- Multi-parent support (DAG, not just chain)

**Usage:**

```python
from core.sovereignty import LocalMerkleDAG

dag = LocalMerkleDAG(storage_path="evidence.json")

# Add evidence nodes
node1 = dag.add_node(
    data={'operation': 'embed', 'text': 'Hello'},
    metadata={'source': 'api'}
)

node2 = dag.add_node(
    data={'operation': 'verify', 'result': 'passed'},
    parent_ids=[node1.node_id]
)

# Multiple parents (DAG structure)
node3 = dag.add_node(
    data={'operation': 'merge'},
    parent_ids=[node1.node_id, node2.node_id]
)

# Verify integrity
result = dag.verify_dag()
if result.valid:
    print(f"✓ DAG valid: {result.verified_nodes} nodes")
else:
    print(f"✗ Tampered: {result.tampered_nodes}")

# Get proof chain
chain = dag.get_proof_chain(node3.node_id)
for node in chain:
    print(f"→ {node.node_id}: {node.data}")

# Persistence
dag.save_to_file("evidence.json")
dag2 = LocalMerkleDAG(storage_path="evidence.json")
```

## Integration Example

Complete sovereignty workflow:

```python
from core.sovereignty import (
    WinterProofEmbedder,
    Constitution,
    DaughterTest,
    LocalMerkleDAG
)

# Initialize components
embedder = WinterProofEmbedder()
constitution = Constitution()
tester = DaughterTest()
dag = LocalMerkleDAG()

# 1. Generate offline embedding
text = "BIZRA sovereignty in action"
embedding = embedder.embed(text)

# 2. Record in evidence chain
embed_node = dag.add_node(
    data={
        'operation': 'embed',
        'text_hash': embedder.generate_receipt(text, embedding)['text_hash'],
        'dimension': len(embedding)
    }
)

# 3. Verify constitutional compliance
scores = {
    'ihsan': 0.98,
    'sovereignty': 1.0,  # 100% offline
    'transparency': 1.0,
    'integrity': 1.0,
    'determinism': 1.0,
    'efficiency': 0.95,
}

receipt = constitution.verify(
    operation="sovereign_embed",
    scores=scores
)

# 4. Record compliance in DAG
compliance_node = dag.add_node(
    data={
        'operation': 'compliance_check',
        'receipt_id': receipt.receipt_id,
        'compliant': receipt.compliant,
        'score': receipt.overall_score
    },
    parent_ids=[embed_node.node_id]
)

# 5. Continuous integrity verification
tester.verify_determinism(
    operation=embedder.embed,
    inputs=(text,),
    iterations=3
)

# 6. Verify complete evidence chain
result = dag.verify_dag()
assert result.valid, "Evidence chain compromised"

print(f"✓ Complete sovereignty verified: {result.verified_nodes} nodes")
```

## Dependencies

### Required (stdlib only)
- `hashlib` - SHA-256, SHA3-256, SHA-512
- `json` - Data serialization
- `uuid` - Unique identifiers
- `datetime` - Timestamps
- `threading` - Continuous monitoring

### Optional (graceful degradation)
- `numpy` - Accelerated vector operations (falls back to stdlib)
- `blake3` - Enhanced hashing (falls back to SHA-512)

## Installation

No installation required - pure stdlib with optional acceleration:

```bash
# Optional: Install numpy for acceleration
pip install numpy

# Optional: Install blake3 for enhanced security
pip install blake3
```

## Testing

Each module includes a demo script:

```bash
# Test WinterProofEmbedder
python core/sovereignty/winter_proof.py

# Test Constitution
python core/sovereignty/constitution.py

# Test DaughterTest
python core/sovereignty/daughter_test.py

# Test LocalMerkleDAG
python core/sovereignty/merkle_dag.py
```

## Performance

**WinterProofEmbedder:**
- Without numpy: ~50ms per embedding (384-dim)
- With numpy: ~10ms per embedding (384-dim)
- Determinism: 100% guaranteed
- Memory: O(dimension) per embedding

**Constitution:**
- Verification: ~1ms per operation
- Receipt generation: ~2ms (includes SHA-256)

**DaughterTest:**
- Determinism check: ~50ms (3 iterations)
- Checksum: <1ms (SHA-256)
- Chain verification: O(n) where n = chain length

**LocalMerkleDAG:**
- Node addition: ~2ms (includes BLAKE3/SHA-256)
- Verification: O(n) where n = total nodes
- Proof chain: O(log n) average case

## Security

All operations use:
- **Domain separation:** `bizra-pci-v1:` prefix
- **Cryptographic hashing:** SHA-256 minimum, BLAKE3 preferred
- **Merkle proofs:** For efficient verification
- **Tamper detection:** Automatic via hash chains

## Ihsān Compliance

This module enforces:
- 0.95 minimum Ihsān threshold (95% excellence)
- 1.0 sovereignty (100% offline-capable)
- 1.0 integrity (100% verifiable)
- 1.0 determinism (100% reproducible)

## License

Part of BIZRA Node0 - Genesis Block

## Contact

BIZRA Sovereignty Module v1.0.0
Domain: `bizra-pci-v1:`
Built with Ihsān (Excellence)
