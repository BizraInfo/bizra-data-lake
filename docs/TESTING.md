# BIZRA Integration Test Suite

**Created:** 2026-01-29  
**Purpose:** End-to-end validation of the thermodynamic flywheel

---

## Test Categories

### 1. Unit Tests (Offline)
- `test_flywheel_validation.py` — Core component tests
- Run without model, validate structure

### 2. Integration Tests (Model Required)
- `test_inference_integration.py` — Full inference pipeline
- Requires: llama-cpp-python + GGUF model

### 3. System Tests (Full Stack)
- `test_system_e2e.py` — Complete flywheel cycle
- Receipt → Inference → Receipt → Epigenome

---

## Running Tests

```bash
# Activate venv
cd /mnt/c/BIZRA-DATA-LAKE
source .venv/bin/activate

# Unit tests (always work)
python test_flywheel_validation.py

# Integration tests (need model)
BIZRA_MODEL_PATH=/mnt/c/BIZRA-DATA-LAKE/models/qwen2.5-1.5b-instruct-q4_k_m.gguf \
python test_inference_integration.py

# System tests (need everything)
python test_system_e2e.py
```

---

## Test Matrix

| Test | Model | GPU | Internet | Expected Time |
|------|-------|-----|----------|---------------|
| Unit | ❌ | ❌ | ❌ | <5s |
| Integration | ✅ | Optional | ❌ | ~30s |
| System | ✅ | Recommended | ❌ | ~2min |

---

## CI/CD Pipeline (Future)

```yaml
# .github/workflows/test.yml
name: BIZRA Tests

on: [push, pull_request]

jobs:
  unit:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.12'
      - run: pip install -e .
      - run: python test_flywheel_validation.py

  integration:
    runs-on: self-hosted  # Needs GPU
    steps:
      - uses: actions/checkout@v4
      - run: |
          source .venv/bin/activate
          python test_inference_integration.py
```

---

## Coverage Goals

| Component | Target | Current |
|-----------|--------|---------|
| core/inference | 80% | ~60% |
| core/pci | 90% | ~70% |
| core/vault | 80% | 0% |
| core/federation | 60% | 0% |

---

## Ihsan Quality Gate

All tests must pass before merge:
1. ✅ Unit tests pass
2. ✅ No new linting errors
3. ✅ Integration tests pass (if model available)
4. ✅ No secrets in code
5. ✅ Documentation updated

*"Excellence is not optional. It is the minimum."*
