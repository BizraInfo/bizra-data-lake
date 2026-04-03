# BIZRA Sovereignty Module - Quick Start Guide

**HYPER LOOPBACK Architecture** - Complete Offline Sovereignty

## Installation

No installation required - pure stdlib with optional acceleration:

```bash
# Optional: Install numpy for 5x faster embeddings
pip install numpy

# Optional: Install blake3 for enhanced security hashing
pip install blake3
```

## 5-Minute Tutorial

### 1. Generate Offline Embeddings

```python
from core.sovereignty import WinterProofEmbedder

# Initialize embedder (384-dimensional by default)
embedder = WinterProofEmbedder()

# Generate embedding (100% deterministic)
text = "BIZRA is a sovereign agentic system"
embedding = embedder.embed(text)

print(f"Embedding dimension: {len(embedding)}")
print(f"First 5 values: {embedding[:5]}")

# Verify determinism
emb1 = embedder.embed(text)
emb2 = embedder.embed(text)
assert emb1 == emb2, "Not deterministic!"
print("✓ Determinism verified")
```

### 2. Check Constitutional Compliance

```python
from core.sovereignty import Constitution

# Initialize constitution with 0.95 Ihsān threshold
constitution = Constitution()

# Define operation scores
scores = {
    'ihsan': 0.98,           # Excellence: 98%
    'sovereignty': 1.0,       # Offline: 100%
    'transparency': 1.0,      # Auditable: 100%
    'integrity': 1.0,         # Verifiable: 100%
    'determinism': 1.0,       # Reproducible: 100%
    'efficiency': 0.95,       # Optimized: 95%
}

# Verify compliance
receipt = constitution.verify(
    operation="sovereign_operation",
    scores=scores
)

if receipt.compliant:
    print(f"✓ Compliant: {receipt.overall_score:.2%}")
else:
    print("✗ Non-compliant")
    for violation in receipt.violations:
        print(f"  - {violation.message}")
```

### 3. Continuous Integrity Verification

```python
from core.sovereignty import DaughterTest

tester = DaughterTest()

# Test function determinism
def my_function(x):
    return x * 2

check = tester.verify_determinism(
    operation=my_function,
    inputs=(5,),
    iterations=3
)

print(f"Deterministic: {check.passed}")

# Register for continuous monitoring
tester.register_operation(
    operation_id="critical_op",
    operation_func=my_function,
    baseline_inputs=(10,)
)

# Start monitoring (checks every 60 seconds)
tester.start_monitoring(interval_seconds=60)
print("✓ Monitoring started")

# ... later ...
tester.stop_monitoring()
```

### 4. Build Evidence Chain

```python
from core.sovereignty import LocalMerkleDAG

# Initialize DAG
dag = LocalMerkleDAG()

# Add evidence nodes
node1 = dag.add_node(
    data={'operation': 'embed', 'text': 'Hello BIZRA'},
    metadata={'source': 'api'}
)

node2 = dag.add_node(
    data={'operation': 'verify', 'result': 'passed'},
    parent_ids=[node1.node_id]
)

# Verify integrity
result = dag.verify_dag()
print(f"DAG valid: {result.valid}")
print(f"Nodes: {result.verified_nodes}/{result.total_nodes}")

# Get proof chain
chain = dag.get_proof_chain(node2.node_id)
for node in chain:
    print(f"→ {node.node_id[:8]}... - {node.data}")
```

## Complete Example: Sovereign Workflow

```python
from core.sovereignty import (
    WinterProofEmbedder,
    Constitution,
    DaughterTest,
    LocalMerkleDAG
)

# 1. Initialize components
embedder = WinterProofEmbedder()
constitution = Constitution()
tester = DaughterTest()
dag = LocalMerkleDAG()

# 2. Generate offline embedding
text = "BIZRA sovereignty in action"
embedding = embedder.embed(text)

# 3. Record in evidence chain
receipt = embedder.generate_receipt(text, embedding)
embed_node = dag.add_node(
    data={
        'operation': 'embed',
        'text_hash': receipt['text_hash'],
        'embedding_hash': receipt['embedding_hash'],
    }
)

# 4. Verify constitutional compliance
scores = {
    'ihsan': 0.98,
    'sovereignty': 1.0,
    'transparency': 1.0,
    'integrity': 1.0,
    'determinism': 1.0,
    'efficiency': 0.95,
}

compliance = constitution.verify("sovereign_embed", scores)
compliance_node = dag.add_node(
    data={
        'operation': 'compliance',
        'compliant': compliance.compliant,
        'score': compliance.overall_score,
    },
    parent_ids=[embed_node.node_id]
)

# 5. Verify determinism
check = tester.verify_determinism(
    operation=embedder.embed,
    inputs=(text,),
    iterations=3
)

# 6. Verify complete evidence chain
result = dag.verify_dag()

# 7. Generate report
print("\n" + "=" * 60)
print("SOVEREIGNTY WORKFLOW COMPLETE")
print("=" * 60)
print(f"✓ Embedding: {len(embedding)}-dimensional")
print(f"✓ Compliance: {compliance.overall_score:.2%}")
print(f"✓ Determinism: {check.passed}")
print(f"✓ Evidence: {result.verified_nodes} nodes verified")
print(f"\nDomain: {embedder.DOMAIN_PREFIX}")
print(f"Status: SOVEREIGN\n")
```

## Quick Reference

### WinterProofEmbedder

```python
embedder = WinterProofEmbedder(dimension=384)
embedding = embedder.embed(text)                    # Single
embeddings = embedder.embed_batch(texts)            # Batch
similarity = embedder.cosine_similarity(emb1, emb2) # Compare
results = embedder.semantic_search(query, docs)     # Search
```

### Constitution

```python
constitution = Constitution()
receipt = constitution.verify(operation, scores)    # Verify
constitution.verify_with_enforcement(op, scores)    # Enforce (raises exception)
is_valid = constitution.verify_receipt_integrity(r) # Check receipt
```

### DaughterTest

```python
tester = DaughterTest()
tester.verify_determinism(func, inputs, iterations) # Test determinism
tester.verify_checksum(data, expected_hash)         # Test checksum
tester.verify_evidence_chain(chain)                 # Test chain
tester.start_monitoring()                           # Start continuous checks
```

### LocalMerkleDAG

```python
dag = LocalMerkleDAG()
node = dag.add_node(data, parent_ids)              # Add node
result = dag.verify_dag()                          # Verify integrity
chain = dag.get_proof_chain(node_id)               # Get proof
dag.save_to_file(path)                             # Persist
```

## Demo Scripts

Run individual component demos:

```bash
# Test WinterProofEmbedder
python core/sovereignty/winter_proof.py

# Test Constitution
python core/sovereignty/constitution.py

# Test DaughterTest
python core/sovereignty/daughter_test.py

# Test LocalMerkleDAG
python core/sovereignty/merkle_dag.py

# Run integration test
python core/sovereignty/test_integration.py
```

## Key Concepts

### HYPER LOOPBACK
- 100% offline operation
- No external API dependencies
- Pure stdlib with optional acceleration
- Cryptographic integrity (SHA-256, BLAKE3)

### Ihsān (Excellence)
- 0.95 minimum threshold (95% excellence)
- 6 core principles enforced
- Constitutional compliance required
- Automatic violation detection

### Evidence Chain
- Merkle DAG structure
- Tamper-proof via hash chains
- Parent-child relationships
- Complete proof chains

### Determinism
- Same input → same output
- Cryptographic guarantees
- Continuous verification
- Real-time monitoring

## Performance

| Operation | Time | Notes |
|-----------|------|-------|
| Generate embedding | 10-50ms | 10ms with numpy, 50ms without |
| Verify compliance | 1-2ms | Constitution check |
| Add DAG node | 2ms | Includes hash calculation |
| Verify DAG | O(n) | n = total nodes |
| Determinism check | 150ms | 3 iterations @ 50ms each |

## Requirements

**Stdlib Only:**
- hashlib (SHA-256, SHA3-256, SHA-512)
- json (data serialization)
- uuid (unique identifiers)
- threading (continuous monitoring)

**Optional:**
- numpy (5x faster embeddings)
- blake3 (enhanced security hashing)

## Troubleshooting

**"Dimension must be divisible by 3"**
```python
# Use 384, 768, 1536, etc.
embedder = WinterProofEmbedder(dimension=384)
```

**"Constitutional violation detected"**
```python
# Check which principle failed
for v in receipt.violations:
    print(f"{v.principle}: {v.actual} < {v.threshold}")
```

**"DAG verification failed"**
```python
# Check for tampering
result = dag.verify_dag()
print(f"Tampered nodes: {result.tampered_nodes}")
```

## Next Steps

- Read full documentation: `/mnt/c/BIZRA-Dual-Agentic-system--main/core/sovereignty/README.md`
- Explore source code for advanced usage
- Check integration test for complete examples
- Review constitutional principles in `constitution.py`

## Support

For issues or questions:
1. Check README.md for detailed documentation
2. Review test_integration.py for examples
3. Run demo scripts to verify installation
4. Check module docstrings for API details

---

**BIZRA Sovereignty Module v1.0.0**
Domain: `bizra-pci-v1:`
Status: OPERATIONAL
Built with Ihsān (Excellence)
