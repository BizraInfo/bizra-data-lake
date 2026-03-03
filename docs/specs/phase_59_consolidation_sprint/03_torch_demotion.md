# Step 3: PyTorch Dependency Demotion

## Standing on Giants: Amdahl — minimize the critical path; lightweight by default

## Problem Statement

`torch>=2.0.0,<3.0` is listed in the primary dependencies of `pyproject.toml`.
PyTorch installed size is ~2GB. For a system that targets universal deployment
("grandmother's $200 phone"), this creates an unnecessary barrier. The
`minimal` optional group exists but `torch` isn't properly gated behind the
`full` optional group.

**Accessibility Principle Violation:** A lightweight node should install with
`pip install bizra-data-lake` and work. Torch should only be pulled in with
`pip install bizra-data-lake[full]`.

## Target Files

| File | Action |
|------|--------|
| `pyproject.toml` | Move torch from `[project.dependencies]` to `[project.optional-dependencies.full]` |
| `core/inference/*.py` | Add torch-optional guards with fallback |
| `core/living_memory/*.py` | Add torch-optional guards for embedding operations |
| `core/resonance/*.py` | Add torch-optional guards |
| `tests/conftest.py` | Add `requires_torch` marker |

## Pseudocode

### pyproject.toml Changes

```pseudocode
# BEFORE:
[project.dependencies]
    ...
    "torch>=2.0.0,<3.0",
    "transformers>=4.51.0,<6.0",
    "sentence-transformers>=4.1,<6.0",
    ...

# AFTER:
[project.dependencies]
    ...
    # torch, transformers, sentence-transformers moved to [full]
    ...

[project.optional-dependencies]
full = [
    "torch>=2.0.0,<3.0",
    "transformers>=4.51.0,<6.0",
    "sentence-transformers>=4.1,<6.0",
    ...existing full deps...
]
```

### Import Guard Pattern

```pseudocode
# Standard guard pattern for all torch-dependent modules:

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# At function/method level:
FUNCTION embed_text(text: str) -> List[float]:
    IF NOT TORCH_AVAILABLE:
        # Fallback: use numpy-only TF-IDF or return zero vector
        RETURN _tfidf_fallback(text)
    # Normal torch path
    ...
```

### Module-Specific Fallback Strategy

```pseudocode
# core/inference/ — LLM inference does NOT need torch
#   Ollama uses HTTP API, LM Studio uses HTTP API
#   llama.cpp uses ctypes bindings
#   STATUS: No changes needed — inference is torch-free

# core/living_memory/ — Embedding generation needs torch
#   FALLBACK: sentence-transformers → ONNX Runtime → TF-IDF
#   The FAISS index itself is numpy-based (faiss-cpu), no torch needed
#   Only the embedding model (nomic-embed-text) needs torch

FUNCTION get_embedder():
    IF TORCH_AVAILABLE:
        from sentence_transformers import SentenceTransformer
        RETURN SentenceTransformer("nomic-embed-text")
    ELIF ONNXRUNTIME_AVAILABLE:
        RETURN OnnxEmbedder("nomic-embed-text.onnx")
    ELSE:
        RETURN TFIDFEmbedder()  # Numpy-only, lower quality but functional

# core/resonance/ — Neural resonance scoring
#   FALLBACK: statistical scoring (cosine similarity on TF-IDF vectors)
#   Quality degrades but functionality preserved

# core/benchmark/ — CLEAR framework scoring
#   Uses numpy for statistics, NOT torch
#   STATUS: No changes needed
```

### Test Marker

```pseudocode
# conftest.py addition:
def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "requires_torch: mark test as requiring PyTorch",
    )

# Usage in tests:
@pytest.mark.requires_torch
def test_embedding_quality():
    """Tests that require torch for embedding model."""
    ...
```

## TDD Anchors

```pseudocode
TEST minimal_install_works_without_torch:
    """Core pipeline works without torch installed.

    This test runs in a venv WITHOUT torch and verifies:
    1. Mission orchestrator initializes
    2. Bridge starts
    3. SNR/Ihsan gates function
    4. Evidence ledger works
    5. Template synthesis works
    """
    # Run in subprocess with PYTHONPATH set but no torch
    result = subprocess.run([
        sys.executable, "-c",
        "from core.sovereign.mission import MissionOrchestrator; "
        "print('import ok')"
    ], env={**os.environ, "TORCH_DISABLED": "1"})
    ASSERT result.returncode == 0

TEST torch_available_flag_correct:
    """TORCH_AVAILABLE reflects actual import state."""
    from core.living_memory import TORCH_AVAILABLE
    try:
        import torch
        ASSERT TORCH_AVAILABLE is True
    except ImportError:
        ASSERT TORCH_AVAILABLE is False

TEST embedding_fallback_returns_valid_vector:
    """Without torch, embedder returns valid (lower quality) vectors."""
    embedder = get_embedder()  # Will use fallback if no torch
    vec = embedder.encode("test sentence")
    ASSERT isinstance(vec, (list, np.ndarray))
    ASSERT len(vec) > 0

TEST snr_engine_works_without_torch:
    """SNR scoring is numpy-only, never needs torch."""
    from core.proof_engine.snr import SNREngine
    engine = SNREngine()
    result = engine.score(...)
    ASSERT 0 <= result.normalized_score <= 1

TEST ihsan_gate_works_without_torch:
    """Ihsan gate is pure arithmetic, never needs torch."""
    from core.proof_engine.ihsan_gate import IhsanGate
    gate = IhsanGate()
    result = gate.evaluate(...)
    ASSERT result.passed or not result.passed  # Just verify it runs

TEST evidence_ledger_works_without_torch:
    """Evidence ledger is stdlib-only (json, hashlib, fcntl)."""
    from core.proof_engine.evidence_ledger import EvidenceLedger
    ledger = EvidenceLedger(path=tmp_path / "test.jsonl")
    ledger.append(receipt={...})
    ok, errors = ledger.verify_chain()
    ASSERT ok
```

## Risk Mitigation

**Risk:** Some module does `import torch` at module level without try/except,
causing ImportError at import time even when torch isn't needed.

**Mitigation:** Before making pyproject.toml changes:
1. Grep for all `import torch` across `core/`
2. Wrap each in try/except with `TORCH_AVAILABLE` flag
3. Verify with `python -c "import core"` in torch-free venv
4. Only then modify pyproject.toml

**Rollback:** If torch demotion breaks CI:
1. Revert pyproject.toml
2. Keep the import guards (they're harmless)
3. File issue for incremental torch decoupling

## Acceptance Criteria

1. `torch` is in `[project.optional-dependencies.full]`, not `[project.dependencies]`
2. `pip install .` works without torch installed
3. `import core` succeeds without torch
4. Mission pipeline runs to completion without torch (template synthesis)
5. All existing tests pass (torch is installed in CI dev env)
6. Tests that require torch are marked with `@pytest.mark.requires_torch`
