---
paths:
  - "core/**/*.py"
  - "bizra_kernel/**/*.py"
  - "constellation/**/*.py"
  - "scripts/**/*.py"
  - "tests/**/*.py"
---

# Python Code Style Rules

Rules for Python code in BIZRA's kernel and tooling.

## Formatting

### Tools
- Use `black` for formatting (line length: 88)
- Use `isort` for import sorting
- Use `pyright` or `mypy` for type checking

### Naming Conventions
- Classes: `PascalCase` (e.g., `SapeEngine`, `FateEscalation`)
- Functions/methods: `snake_case` (e.g., `validate_receipt`, `emit_evidence`)
- Constants: `SCREAMING_SNAKE_CASE` (e.g., `IHSAN_THRESHOLD`)
- Private: `_leading_underscore`

### Imports
- Group: stdlib, third-party, local
- Use absolute imports
- Avoid `from x import *`

```python
# Good
import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException

from core.sape import SapeEngine
from core.fate import FateHandler
```

## Type Annotations

### Required Types
- All function signatures must have type hints
- Use `Optional[T]` for nullable types
- Use `TypedDict` for structured dictionaries

```python
from typing import Optional, TypedDict

class ReceiptData(TypedDict):
    receipt_id: str
    timestamp: str
    task_summary: str
    rejection_codes: List[str]
    escalation_level: str
    integrity_hash: str

def create_receipt(
    task_summary: str,
    rejection_codes: Optional[List[str]] = None,
    escalation_level: str = "None"
) -> ReceiptData:
    ...
```

### Generic Types
- Use `list[T]` over `List[T]` in Python 3.9+
- Use `dict[K, V]` over `Dict[K, V]`
- Use `X | None` over `Optional[X]` in Python 3.10+

## Error Handling

### Exceptions
- Define domain-specific exceptions
- Never use bare `except:` clauses
- Always log exceptions with context

```python
class BizraError(Exception):
    """Base exception for BIZRA errors."""
    pass

class IhsanGateError(BizraError):
    """Raised when Ihsān gate check fails."""
    def __init__(self, score: float, threshold: float):
        self.score = score
        self.threshold = threshold
        super().__init__(
            f"Ihsān score {score:.3f} below threshold {threshold}"
        )

# Usage - fail-closed
try:
    result = await validate_request(request)
except IhsanGateError as e:
    logger.error("Ihsān gate failure", extra={"score": e.score})
    await fate.escalate(EscalationLevel.HIGH, str(e))
    raise HTTPException(status_code=403, detail="Ihsān validation failed")
```

### Assertions
- Use `assert` only for invariants in development
- Use explicit validation for production checks

## Async Patterns

### FastAPI
- Use `async def` for I/O-bound endpoints
- Use dependency injection for shared resources
- Handle startup/shutdown properly

```python
@app.on_event("startup")
async def startup():
    app.state.sape = SapeEngine()
    app.state.fate = FateHandler()

@app.get("/health")
async def health_check(
    sape: SapeEngine = Depends(get_sape)
) -> dict:
    return {"status": "healthy", "sape": sape.status()}
```

### Concurrency
- Use `asyncio.gather()` for parallel operations
- Use `asyncio.wait_for()` with timeouts
- Avoid blocking calls in async functions

## BIZRA-Specific Patterns

### Receipt Generation
```python
import hashlib
from datetime import datetime, timezone

def generate_receipt(
    task_summary: str,
    rejection_codes: list[str] | None = None,
    escalation_level: str = "None"
) -> ReceiptData:
    receipt_id = f"receipt-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    timestamp = datetime.now(timezone.utc).isoformat()

    hash_input = f"{receipt_id}{timestamp}{task_summary}"
    integrity_hash = hashlib.sha256(hash_input.encode()).hexdigest()

    return {
        "receipt_id": receipt_id,
        "timestamp": timestamp,
        "task_summary": task_summary,
        "rejection_codes": rejection_codes or [],
        "escalation_level": escalation_level,
        "integrity_hash": integrity_hash
    }
```

### SAPE Integration
```python
async def run_sape_probes(context: ProbeContext) -> list[ProbeResult]:
    probes = [
        probe_threat_scan,
        probe_compliance,
        probe_bias,
        probe_user_benefit,
        probe_correctness,
        probe_safety,
        probe_groundedness,
        probe_relevance,
        probe_fluency,
    ]

    results = await asyncio.gather(
        *[probe(context) for probe in probes],
        return_exceptions=True
    )

    # Log failures, don't proceed silently
    for result in results:
        if isinstance(result, Exception):
            logger.error(f"Probe exception: {result}")
        elif not result.passed:
            logger.warning(f"Probe failed: {result.name} = {result.score}")

    return [r for r in results if not isinstance(r, Exception)]
```

## Testing

- Use `pytest` for all tests
- Use `pytest-asyncio` for async tests
- Name tests: `test_<function>_<condition>_<expected>`
- Use fixtures for shared setup

```python
import pytest

@pytest.fixture
def sample_receipt():
    return generate_receipt("Test task")

@pytest.mark.asyncio
async def test_sape_probe_threat_scan_passes_safe_input(sample_receipt):
    context = ProbeContext(receipt=sample_receipt, content="Hello world")
    result = await probe_threat_scan(context)
    assert result.passed
    assert result.score >= 0.9
```
