# PAT Enforcement Pipeline — Integration Checklist

**Status**: Ready for Integration
**Target**: PAT Unified Orchestrator & Production Systems
**Priority**: HIGH — Constitution-mandated enforcement

## Pre-Integration Verification

### ☐ 1. Syntax Validation

```bash
# Check all Python files compile
python -m py_compile bizra_kernel/pat_enforcement_pipeline.py
python -m py_compile bizra_kernel/pat_domain_validator.py
python -m py_compile bizra_kernel/pat_novelty_probe.py
python -m py_compile bizra_kernel/pat_citation_validator.py
python -m py_compile bizra_kernel/test_pat_enforcement.py
python -m py_compile bizra_kernel/pat_enforcement_example.py
```

### ☐ 2. Run Test Suite

```bash
# Run all tests
pytest bizra_kernel/test_pat_enforcement.py -v --asyncio-mode=auto

# Expected: All tests pass
# If failures: Review logs and fix issues
```

### ☐ 3. Run Examples

```bash
# Run integration examples
python bizra_kernel/pat_enforcement_example.py

# Expected: All 5 examples execute successfully
# Check: docs/evidence/receipts/pat/ directory populated
```

### ☐ 4. Verify Dependencies

```bash
# Required packages (should already be installed)
pip list | grep -E "(pytest|pydantic|numpy|asyncio)"

# If missing:
pip install pytest pytest-asyncio pydantic numpy
```

## Integration Steps

### Phase 1: Component Integration (Low Risk)

#### ☐ 1.1 Import Validation

Add to your integration test file:

```python
# Test imports
from bizra_kernel.pat_enforcement_pipeline import (
    PATEnforcementPipeline,
    PATRequest,
    PATEnforcementResult,
    PATTelemetry,
)
from bizra_kernel.pat_domain_validator import PATDomainValidator
from bizra_kernel.pat_novelty_probe import PATNoveltyProbe
from bizra_kernel.pat_citation_validator import PATCitationValidator

print("✓ All PAT components imported successfully")
```

**Acceptance**: No import errors

#### ☐ 1.2 Initialize Pipeline

```python
# Initialize with default thresholds
pipeline = PATEnforcementPipeline()

# Or with custom thresholds
pipeline = PATEnforcementPipeline(
    snr_minimum=0.98,
    novelty_minimum=0.75,
    ihsan_minimum=0.95,
)

print(f"✓ Pipeline initialized: SNR≥{pipeline.snr_minimum}")
```

**Acceptance**: Pipeline initializes without errors

#### ☐ 1.3 Create Test Request

```python
# Create minimal valid request
request = PATRequest(
    session_id="integration_test",
    task_id="test_001",
    query="Test query",
    context={},
    synthesis_nodes=[
        {"id": "n1", "content": "Test", "snr": 0.98, "claim_tag": "DERIVED"}
    ],
    domains=[
        {"name": f"Domain{i}", "cluster_id": f"c{i}"}
        for i in range(3)
    ],
    practitioners=[
        {
            "name": f"Expert{i}",
            "tier": "top_1%",
            "domains": [f"Domain{i}"],
            "relevance_score": 0.75,
        }
        for i in range(3)
    ],
    response_sections=[
        {"id": sid, "claims": []}
        for sid in [
            "executive_synthesis",
            "domain_cross_pollination_map",
            "elite_practitioner_anchoring",
            "novel_insight_synthesis",
            "validation_evidence_trail",
            "actionable_recommendations",
        ]
    ],
)

print("✓ Test request created")
```

**Acceptance**: Request created successfully

#### ☐ 1.4 Execute Test Enforcement

```python
import asyncio

async def test_enforcement():
    result = await pipeline.enforce(request)
    assert result.passed, f"Enforcement failed: {result.gate_results[-1].evidence}"
    print(f"✓ Test enforcement passed (Receipt: {result.receipt_id})")
    return result

result = asyncio.run(test_enforcement())
```

**Acceptance**: Enforcement passes and receipt is generated

### Phase 2: PAT Orchestrator Integration (Medium Risk)

#### ☐ 2.1 Add to Orchestrator Imports

In `bizra_kernel/pat_unified_orchestrator.py` (or equivalent):

```python
from bizra_kernel.pat_enforcement_pipeline import (
    PATEnforcementPipeline,
    PATRequest,
)
```

#### ☐ 2.2 Initialize in Orchestrator

In orchestrator `__init__`:

```python
class PATUnifiedOrchestrator:
    def __init__(self, ...):
        # Existing initialization
        ...

        # Add PAT enforcement
        self.enforcement_pipeline = PATEnforcementPipeline(
            snr_minimum=0.98,
            novelty_minimum=0.75,
            ihsan_minimum=0.95,
        )

        logger.info("PAT Enforcement Pipeline initialized")
```

#### ☐ 2.3 Add Enforcement Step

In orchestrator response flow:

```python
async def process_request(self, query: str, context: dict) -> dict:
    # 1. Existing: Domain analysis, synthesis, etc.
    domains = await self.analyze_domains(query)
    synthesis_nodes = await self.synthesize(query, domains)
    practitioners = await self.fetch_practitioners(domains)
    response = await self.format_response(synthesis_nodes)

    # 2. NEW: PAT Enforcement
    pat_request = PATRequest(
        session_id=context.get("session_id"),
        task_id=context.get("task_id"),
        query=query,
        context=context,
        synthesis_nodes=synthesis_nodes,
        domains=domains,
        practitioners=practitioners,
        response_sections=response.get("sections", []),
    )

    enforcement_result = await self.enforcement_pipeline.enforce(pat_request)

    # 3. Handle enforcement result
    if not enforcement_result.passed:
        # Log failure
        logger.error(
            f"PAT enforcement failed: {enforcement_result.gate_results[-1].gate_id}"
        )

        # Escalate via FATE
        await self.fate_engine.escalate(
            level="high",
            reason=f"PAT gate failure: {enforcement_result.gate_results[-1].evidence}",
            context=enforcement_result.to_dict(),
        )

        # Return error response
        return {
            "status": "rejected",
            "reason": "PAT enforcement failed",
            "receipt_id": enforcement_result.receipt_id,
            "gate_failures": [
                g.gate_id.value for g in enforcement_result.gate_results if not g.passed
            ],
        }

    # 4. Add enforcement metadata to response
    response["pat_enforcement"] = {
        "receipt_id": enforcement_result.receipt_id,
        "final_snr": enforcement_result.final_snr,
        "final_novelty": enforcement_result.final_novelty,
        "final_ihsan": enforcement_result.final_ihsan,
        "gate_latencies": {
            g.gate_id.value: g.latency_ms for g in enforcement_result.gate_results
        },
    }

    return response
```

**Acceptance**: Orchestrator enforces PAT gates before returning responses

#### ☐ 2.4 Test Orchestrator Integration

```python
async def test_orchestrator_with_pat():
    orchestrator = PATUnifiedOrchestrator()

    result = await orchestrator.process_request(
        query="Optimize database performance",
        context={"session_id": "test", "task_id": "test_001"},
    )

    assert "pat_enforcement" in result
    assert result["pat_enforcement"]["receipt_id"]
    print("✓ Orchestrator integration successful")

asyncio.run(test_orchestrator_with_pat())
```

**Acceptance**: Orchestrator returns responses with PAT enforcement metadata

### Phase 3: Database Integration (High Priority)

#### ☐ 3.1 Pattern Database for Novelty Probe

Connect `PATNoveltyProbe` to pattern database:

```python
# In pat_novelty_probe.py, update _load_known_patterns():
async def _load_known_patterns(self, domain: Optional[str] = None):
    # Query pattern database
    query = """
        SELECT pattern_id, content, embedding, domain, frequency
        FROM known_patterns
        WHERE domain = %s OR domain IS NULL
        ORDER BY frequency DESC
        LIMIT 100
    """

    patterns = await db.fetch_all(query, (domain,))

    for row in patterns:
        pattern = KnownPattern(
            pattern_id=row["pattern_id"],
            content=row["content"],
            embedding=row["embedding"],
            domain=row["domain"],
            frequency=row["frequency"],
            timestamp=row["created_at"],
        )
        self.known_patterns.append(pattern)
```

**Acceptance**: Novelty probe loads real patterns from database

#### ☐ 3.2 Practitioner Registry

Connect `PATCitationValidator` to practitioner database:

```python
# In pat_citation_validator.py, update _query_practitioner_registry():
async def _query_practitioner_registry(self, domain: str, query: str, limit: int):
    # Query practitioner database
    sql = """
        SELECT practitioner_id, name, tier, domains, relevance_score,
               credentials, publications, h_index
        FROM practitioners
        WHERE %s = ANY(domains)
        AND tier = 'top_1%'
        ORDER BY relevance_score DESC
        LIMIT %s
    """

    practitioners = await db.fetch_all(sql, (domain, limit))

    return [dict(row) for row in practitioners]
```

**Acceptance**: Citation validator fetches real practitioners from database

### Phase 4: REST API Endpoint (Production)

#### ☐ 4.1 Add FastAPI Route

In `core/main.py` (or equivalent):

```python
from bizra_kernel.pat_enforcement_pipeline import (
    PATEnforcementPipeline,
    PATRequest,
)

# Initialize pipeline at startup
pat_pipeline = PATEnforcementPipeline()

@app.post("/v1/pat/enforce")
async def enforce_pat(request: PATRequest):
    """
    Enforce PAT validation on a request.

    Returns:
        PATEnforcementResult with gate results and receipt
    """
    try:
        result = await pat_pipeline.enforce(request)
        return result.to_dict()
    except Exception as e:
        logger.error(f"PAT enforcement error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
```

**Acceptance**: API endpoint `/v1/pat/enforce` accepts requests and returns results

#### ☐ 4.2 Test API Endpoint

```bash
# Test with curl
curl -X POST http://localhost:8000/v1/pat/enforce \
  -H "Content-Type: application/json" \
  -d '{
    "session_id": "test",
    "task_id": "test_001",
    "query": "Test query",
    "domains": [...],
    "synthesis_nodes": [...],
    "practitioners": [...],
    "response_sections": [...]
  }'

# Expected: JSON response with enforcement result
```

**Acceptance**: API returns valid PAT enforcement results

### Phase 5: Claude Code Integration (Optional)

#### ☐ 5.1 Create PAT Skill

In `.claude/skills/pat-enforcement.md`:

```markdown
---
name: PAT Enforcement
description: Enforce PAT validation gates for peak performance mode
activates_on:
  - "peak mode"
  - "PAT enforcement"
  - "maximum validation"
---

When the user requests "peak mode" or PAT enforcement, you should:

1. Extract the query and context
2. Identify domains and synthesis nodes
3. Call the PAT enforcement pipeline
4. Display gate results in formatted output

Use this tool: bizra_kernel/pat_enforcement_pipeline.py
```

**Acceptance**: `/peak` command triggers PAT enforcement

## Post-Integration Validation

### ☐ 1. Receipt Verification

```bash
# Check receipts are being generated
ls -lh docs/evidence/receipts/pat/

# Verify receipt format
cat docs/evidence/receipts/pat/<receipt_id>.json | jq .

# Expected: Valid JSON receipts with all required fields
```

### ☐ 2. Telemetry Monitoring

```python
# Track enforcement metrics
telemetry = PATTelemetry()

# After 100 enforcements
stats = telemetry.get_stats()
print(f"Pass rate: {stats['pass_rate']:.1%}")
print(f"Average latency: {stats['average_latency_ms']}ms")

# Expected: Pass rate > 80%, latency < 5000ms
```

### ☐ 3. Load Testing

```bash
# Run load test (if available)
pytest tests/load/test_pat_enforcement_load.py -v

# Or manual load test
python -c "
import asyncio
from bizra_kernel.pat_enforcement_pipeline import PATEnforcementPipeline

async def load_test():
    pipeline = PATEnforcementPipeline()
    tasks = [pipeline.enforce(create_test_request()) for _ in range(100)]
    results = await asyncio.gather(*tasks)
    print(f'Completed: {len(results)}, Passed: {sum(r.passed for r in results)}')

asyncio.run(load_test())
"

# Expected: All requests complete, pass rate > 80%
```

## Rollback Plan

If integration issues occur:

### ☐ 1. Disable PAT Enforcement

```python
# In orchestrator, add feature flag
class PATUnifiedOrchestrator:
    def __init__(self, enable_enforcement: bool = True):
        self.enable_enforcement = enable_enforcement

        if self.enable_enforcement:
            self.enforcement_pipeline = PATEnforcementPipeline()

    async def process_request(self, ...):
        # Skip enforcement if disabled
        if not self.enable_enforcement:
            logger.warning("PAT enforcement disabled")
            return await self._process_without_enforcement(...)

        # Normal flow with enforcement
        ...
```

### ☐ 2. Remove API Endpoint

```python
# Comment out or remove endpoint
# @app.post("/v1/pat/enforce")
# async def enforce_pat(request: PATRequest):
#     ...
```

### ☐ 3. Revert Code Changes

```bash
# If using git
git revert <commit_hash>

# Or restore from backup
cp orchestrator.py.backup orchestrator.py
```

## Success Criteria

Integration is successful when:

- ✅ All tests pass
- ✅ Orchestrator enforces PAT gates on all requests
- ✅ Receipts are generated and stored correctly
- ✅ Telemetry shows pass rate > 80%
- ✅ Average latency < 5000ms
- ✅ No errors in production logs
- ✅ API endpoint responds correctly
- ✅ Database integrations working

## Support

If issues arise:

1. **Check logs**: `tail -f logs/pat_enforcement.log`
2. **Review receipts**: `cat docs/evidence/receipts/pat/<receipt_id>.json`
3. **Run diagnostics**: `python bizra_kernel/pat_enforcement_example.py`
4. **Review documentation**: `bizra_kernel/PAT_ENFORCEMENT_README.md`

---

**Integration Owner**: Development Team
**Reviewer**: System Architect
**Approver**: Technical Lead
**Date**: 2026-01-27
