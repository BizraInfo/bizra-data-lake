# SAT Consensus Test Suite Guide

## Overview

The SAT (System Agentic Team) consensus test suite (`test_sat_consensus.py`) provides comprehensive testing for Byzantine fault-tolerant consensus validation with 5 guardian validators requiring 3/5 consensus.

## SAT Validator Architecture

### The 5 Guardians

| Validator | Role | Rejection Code | Primary Function |
|-----------|------|----------------|------------------|
| **PoiVerifier** | Proof of Impact | `POI_INSUFFICIENT` | Validates impact attestations and merit scores |
| **RiskGuardian** | Security | `SECURITY_THREAT` | Detects threats, vulnerabilities, dangerous patterns |
| **GovernanceEngine** | Policy | `POLICY_VIOLATION` | Enforces governance rules and compliance |
| **ResourceAllocator** | Efficiency | `RESOURCE_CONSTRAINT` | Validates resource allocation and efficiency |
| **EvidenceEngine** | Audit | `EVIDENCE_MISSING` | Ensures audit trail completeness and integrity |

### Consensus Rules

**Byzantine Fault Tolerance**:
- System tolerates up to 2 malicious validators (f < n/3 where n=5)
- Requires 3/5 = 60% supermajority (> 2f+1)
- Fail-closed: ambiguous state → rejection

**Vote Types**:
- `APPROVE` - Validator approves the request
- `REJECT` - Validator rejects with specific rejection code
- `ABSTAIN` - Validator abstains (insufficient information)
- `TIMEOUT` - Validator did not respond in time
- `MALICIOUS` - Byzantine behavior (for testing only)

**Critical Rejections (VETO Power)**:
The following rejection codes override majority approval:
- `SECURITY_THREAT` - Security vulnerability detected
- `ETHICS_VIOLATION` - Ethical violation detected
- `QUARANTINE` - Uncertain, requires human review

## Test Coverage

### Test Classes

#### 1. TestUnanimousApproval (5/5 PASS)
- All validators approve → consensus reached
- Receipt generation with unanimous approval
- Zero rejection codes, NONE escalation level

#### 2. TestSupermajorityApproval (4/5 PASS)
- 4 approve, 1 rejects → consensus reached
- 4 approve, 1 abstains → consensus reached
- Non-critical rejection codes allowed

#### 3. TestMinimumConsensus (3/5 PASS)
- 3 approve, 2 reject → consensus reached
- 3 approve, 1 reject, 1 abstain → consensus reached
- Minimum threshold for Byzantine fault tolerance

#### 4. TestSplitDecisions
- 2 approve, 2 reject, 1 abstain → FAIL (below threshold)
- 3 approve, 2 abstain → PASS (minimum consensus)
- Tests boundary conditions

#### 5. TestConsensusFailure (2/5 FAIL)
- 2 approve, 3 reject → FAIL with MEDIUM escalation
- 1 approve, 4 reject → FAIL with HIGH escalation
- 0 approve, 5 reject → FAIL with HIGH escalation
- FATE escalation triggered correctly

#### 6. TestVetoPower
- Security threat rejection vetoes 4 approvals → FAIL
- Quarantine rejection triggers HIGH escalation → FAIL
- Critical rejections override majority

#### 7. TestByzantineFaultTolerance
- 2 malicious, 3 honest approve → PASS (at tolerance threshold)
- 3 malicious validators → DETECTABLE via evidence mismatch
- Evidence hash inconsistencies flag Byzantine behavior

#### 8. TestTimeoutScenarios
- 1 timeout, 4 approve → PASS (timeout as abstention)
- 2 timeout, 3 approve → PASS (minimum consensus)
- 3 timeout, 2 approve → FAIL (not enough votes)

#### 9. TestConflictingVerdicts
- Same evidence, different conclusions → normal behavior
- Evidence hash consistency verification
- Validator independence validation

#### 10. TestQuorumEdgeCases
- Exactly 3 approve → PASS (boundary test)
- Exactly 2 approve → FAIL (below threshold)
- All abstain → FAIL (fail-closed behavior)

#### 11. TestReceiptGeneration
- Receipt contains all validator votes
- Deterministic integrity hash for same input
- Rejection codes included in receipt
- Escalation level recorded

#### 12. TestEvidenceChainIntegrity
- Evidence hash consistency across honest validators
- Evidence includes request context
- Tamper detection via hash mismatch

#### 13. TestFailClosedBehavior
- Ambiguous state (all abstentions) → rejection
- VETO overrides majority approval
- Safety-first architecture validation

#### 14. TestSATIntegration
- End-to-end consensus flow
- Multiple sequential validations
- Concurrent validations (stress test)

#### 15. TestPerformance
- Validation latency benchmarks
- Throughput measurements (100 validations)

## Running the Tests

### Run All SAT Consensus Tests

```bash
pytest tests/test_sat_consensus.py -v
```

### Run Specific Test Class

```bash
pytest tests/test_sat_consensus.py::TestByzantineFaultTolerance -v
```

### Run Specific Test

```bash
pytest tests/test_sat_consensus.py::TestVetoPower::test_security_threat_veto -v
```

### Run with Coverage

```bash
pytest tests/test_sat_consensus.py --cov=core.sat --cov-report=html
```

### Skip Slow Tests

```bash
pytest tests/test_sat_consensus.py -m "not slow"
```

### Run Only Async Tests

```bash
pytest tests/test_sat_consensus.py -m asyncio
```

## Test Architecture

### Mock SAT Consensus Engine

The test suite uses `MockSATConsensusEngine` which simulates:
- All 5 SAT validators
- Byzantine fault tolerance logic
- Consensus calculation (3/5 threshold)
- VETO power for critical rejections
- Evidence hash computation
- Receipt generation
- FATE escalation levels

### Key Features

**Pre-configured Responses**:
```python
engine = MockSATConsensusEngine(
    validator_responses={
        ValidatorRole.POI_VERIFIER: VoteDecision.APPROVE,
        ValidatorRole.RISK_GUARDIAN: VoteDecision.REJECT,
        # ... configure all 5 validators
    }
)
```

**Byzantine Testing**:
```python
ValidatorRole.EVIDENCE_ENGINE: VoteDecision.MALICIOUS
# Produces conflicting evidence hash for Byzantine detection
```

**Evidence Integrity**:
- SHA-256 hashing of request context
- Deterministic hash generation
- Tamper detection via hash mismatch

## Consensus Logic

### Approval Calculation

```
consensus_reached = (approvals >= 3) AND (no critical rejections)

Critical Rejections (VETO):
- SECURITY_THREAT
- ETHICS_VIOLATION
- QUARANTINE

Timeout Behavior:
- Timeouts treated as abstentions
- Do not count toward approve/reject
- Fail-closed: not enough approvals → rejection
```

### FATE Escalation Levels

| Condition | Escalation Level |
|-----------|------------------|
| Consensus reached | `NONE` |
| SECURITY_THREAT or ETHICS_VIOLATION | `CRITICAL` |
| QUARANTINE | `HIGH` |
| 4+ rejections | `HIGH` |
| 3 rejections | `MEDIUM` |
| <3 rejections | `LOW` |

### Receipt Structure

```json
{
  "receipt_id": "SAT-TASK001-a1b2c3d4",
  "task_id": "TASK-001",
  "timestamp": "2026-01-27T18:00:00Z",
  "consensus_reached": true,
  "votes": [
    {
      "validator": "poi_verifier",
      "decision": "approve",
      "confidence": 0.95
    },
    // ... all 5 votes
  ],
  "rejection_codes": [],
  "escalation_level": "none",
  "integrity_hash": "sha256_hash_here"
}
```

## Byzantine Fault Tolerance

### Theory

For a system with `n` validators and `f` Byzantine (malicious) validators:
- System is Byzantine fault-tolerant if `f < n/3`
- Requires `2f + 1` honest validators for consensus

**BIZRA SAT**:
- `n = 5` validators
- Tolerates `f = 1` Byzantine validator (f < 5/3 = 1.67)
- At the threshold: `f = 2` is detectable but system may fail
- Requires `3/5` consensus (> 2f+1 = 2×1+1 = 3)

### Detection Mechanisms

1. **Evidence Hash Mismatch**: Malicious validators produce different evidence hashes
2. **Conflicting Verdicts**: Same evidence, different conclusions flags review
3. **Timing Analysis**: Unusual response patterns (not in mock)
4. **Reputation Scoring**: Historical behavior tracking (not in mock)

### Test Coverage

- **2 malicious, 3 honest**: System passes (at tolerance threshold)
- **3 malicious**: Detectable via evidence hash mismatches
- **Evidence tampering**: Caught by integrity hash verification

## Integration with BIZRA System

### Connection to Rust Core

The Python test suite mirrors the Rust implementation in `src/sat.rs`:

```rust
// src/sat.rs
pub struct SATOrchestrator {
    agents: Vec<SATAgent>,
    max_task_tokens: usize,
    max_execution_ms: u64,
}

// Consensus rules match Python tests
let consensus_reached = approvals >= 3 && !has_veto;
```

### FATE Escalation Integration

Failed consensus triggers FATE (Fail-Safe Agentic Trust Escalation):

```python
if not result.consensus_reached:
    fate.escalate(
        level=result.escalation_level,
        reason=result.rejection_codes,
        evidence=result.receipt_id
    )
```

### Receipt Pipeline

All consensus results generate receipts stored in:
```
docs/evidence/receipts/sat_consensus_*.jsonl
```

Receipt format follows BIZRA receipt schema defined in `src/receipts.rs`.

## Best Practices

### Writing New SAT Tests

1. **Use descriptive test names**: `test_security_threat_veto`, not `test_scenario_1`
2. **Test one concept per test**: Don't combine multiple assertions
3. **Use fixtures for common setups**: `@pytest.fixture` for reusable components
4. **Mark async tests**: `@pytest.mark.asyncio` for async functions
5. **Add docstrings**: Explain what the test validates

### Testing Byzantine Scenarios

```python
@pytest.mark.asyncio
async def test_byzantine_detection(self, sample_request):
    """Test Byzantine fault detection via evidence mismatch"""
    engine = MockSATConsensusEngine(
        validator_responses={
            # Configure malicious validators
            ValidatorRole.EVIDENCE_ENGINE: VoteDecision.MALICIOUS,
        }
    )

    result = await engine.validate(sample_request)

    # Verify detection
    malicious_hashes = [
        v.evidence_hash for v in result.votes
        if v.evidence_hash == "malicious_hash_mismatch"
    ]
    assert len(malicious_hashes) > 0
```

### Testing FATE Escalation

```python
@pytest.mark.asyncio
async def test_fate_escalation_critical(self, sample_request):
    """Test CRITICAL escalation on security threat"""
    engine = MockSATConsensusEngine()

    # Mock critical rejection
    async def mock_vote(role, request):
        if role == ValidatorRole.RISK_GUARDIAN:
            return ValidatorVote(
                validator_role=role,
                decision=VoteDecision.REJECT,
                rejection_code=RejectionCode.SECURITY_THREAT,
                confidence=0.99,
                reasoning="Critical threat detected"
            )
        return await engine._get_validator_vote(role, request)

    engine._get_validator_vote = mock_vote
    result = await engine.validate(sample_request)

    assert result.escalation_level == EscalationLevel.CRITICAL
```

## Troubleshooting

### Common Issues

**Import Errors**:
```bash
# Ensure you're in project root
cd /mnt/c/BIZRA-Dual-Agentic-system--main

# Check pytest is installed
pip install pytest pytest-asyncio
```

**Async Test Failures**:
```bash
# Verify pytest.ini has asyncio_mode = auto
cat pytest.ini | grep asyncio_mode
```

**Fixture Not Found**:
```bash
# Ensure fixture is defined in same file or conftest.py
# Check fixture scope (function, class, module, session)
```

### Debug Mode

Run with verbose output and debug logs:
```bash
pytest tests/test_sat_consensus.py -vv -s --log-cli-level=DEBUG
```

## Performance Benchmarks

Expected performance (mock implementation):
- **Single validation**: < 100ms
- **Throughput**: > 10 validations/second
- **Concurrent validations**: No degradation up to 10 concurrent

Real implementation (Rust + LLM):
- **Single validation**: 200-2000ms (depends on LLM latency)
- **Throughput**: 5-50 validations/second
- **Concurrent validations**: Limited by LLM backend

## Future Enhancements

### Planned Test Additions

1. **Validator Timeout Recovery**: Test retry mechanisms
2. **Dynamic Validator Addition**: Test adding 6th validator
3. **Reputation-Based Weighting**: Test confidence-weighted consensus
4. **Historical Pattern Analysis**: Test learning from past validations
5. **Cross-Chain Validation**: Test multi-node consensus

### Integration Tests

Connect to real components:
- Rust SAT orchestrator (`src/sat.rs`)
- FATE engine (`core/fate.py`)
- Receipt storage (`docs/evidence/receipts/`)
- Neo4j graph evidence (`wisdom` service)

## References

- **BIZRA Constitution**: `constitution/ihsan_v1.yaml`
- **SAT Implementation**: `src/sat.rs`
- **FATE Engine**: `core/fate.py`
- **Receipt Schema**: `src/receipts.rs`
- **Byzantine Fault Tolerance**: [Wikipedia](https://en.wikipedia.org/wiki/Byzantine_fault)
- **CLAUDE.md**: Project documentation at root

## Contributing

When adding new tests:
1. Follow existing naming conventions
2. Add docstrings explaining the test
3. Update this guide with new test classes
4. Ensure all tests pass: `pytest tests/test_sat_consensus.py -v`
5. Check coverage: `pytest --cov=core.sat --cov-report=term-missing`

## License

Part of the BIZRA ecosystem. See project LICENSE for details.

---

**Last Updated**: 2026-01-27
**Test Count**: 40+ test cases across 15 test classes
**Coverage**: Consensus logic, Byzantine faults, FATE escalation, receipt generation
