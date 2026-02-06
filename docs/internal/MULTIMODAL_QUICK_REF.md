# BIZRA Multi-Modal Router — Quick Reference

## One-Liner Setup

```python
from core.inference import get_multimodal_router
router = get_multimodal_router()
```

## Basic Routing

```python
# Text task
decision = router.route("What is 2+2?")
print(decision.model.name)  # llama2

# Reasoning task
decision = router.route("Prove the Goldbach conjecture")
print(decision.model.name)  # qwq

# Vision task
decision = router.route("Describe this image", explicit_type="image")
print(decision.model.name)  # moondream

# Voice task
decision = router.route("Transcribe the speech", explicit_type="audio")
print(decision.model.name)  # whisper
```

## Routing Result

```python
decision = router.route("task")

# Available fields
print(decision.model.name)           # Model name
print(decision.model.params_b)       # Size in billions
print(decision.model.backend)        # "lmstudio" or "ollama"
print(decision.model.endpoint)       # Server endpoint
print(decision.model.speed_tok_per_sec)  # Tokens/sec
print(decision.capability_match)     # REASONING|VISION|VOICE|GENERAL
print(decision.confidence)           # 0.95|0.75|0.50
print(decision.reason)               # Human-readable explanation
print(decision.alternatives)         # Fallback models
```

## Capability-Based Selection

```python
from core.inference import ModelCapability

# Select best reasoning model
decision = router.select_model(ModelCapability.REASONING)
# -> qwq (32B, largest reasoning model)

# Select best vision model
decision = router.select_model(ModelCapability.VISION)
# -> moondream (1.9B, fastest vision model)

# Select best voice model
decision = router.select_model(ModelCapability.VOICE)
# -> whisper (0.77B, speech specialist)

# Select best general model
decision = router.select_model(ModelCapability.GENERAL)
# -> llama2 (7B, balanced general model)
```

## Listing Models

```python
# All models
for model in router.list_models():
    print(f"{model.name}: {model.capabilities}")

# Models by capability
vision_models = router.list_by_capability(ModelCapability.VISION)
for model in vision_models:
    print(f"{model.name} - {model.speed_tok_per_sec} tok/s")
```

## Dict-Style Task Specification

```python
task = {
    "text": "Analyze the document",
    "has_image": True,
    "has_audio": False,
}

decision = router.route(task)
```

## Register Custom Model

```python
from core.inference import ModelInfo, ModelCapability

custom = ModelInfo(
    name="my-model",
    capabilities=[ModelCapability.REASONING, ModelCapability.GENERAL],
    primary_capability=ModelCapability.REASONING,
    backend="lmstudio",
    endpoint="192.168.1.100:5000",
    params_b=70.0,
    context_length=4096,
    speed_tok_per_sec=10.0,
    description="My custom reasoning model"
)

router.register_model(custom)
```

## Selection Logic

| Task Type | Detection | Primary Model | Confidence |
|-----------|-----------|---------------|-----------|
| "Prove that..." | REASONING | qwq (32B) | 95% |
| "Analyze..." | REASONING | qwq (32B) | 95% |
| "Describe image" | VISION | moondream (1.9B) | 95% |
| "Extract text" | VISION | moondream (1.9B) | 95% |
| "Transcribe" | VOICE | whisper (0.77B) | 95% |
| "What is..." | GENERAL | llama2 (7B) | 95% |
| "Write..." | GENERAL | llama2 (7B) | 95% |

## Detection Keywords

### Reasoning
`prove`, `analyze`, `synthesize`, `evaluate`, `compare`, `debug`, `research`, `design`, `create`, `diagnose`

### Vision
`image`, `photo`, `picture`, `screenshot`, `visual`, `see`, `look at`, `ocr`, `extract`, `chart`, `diagram`

### Voice
`audio`, `sound`, `voice`, `speech`, `transcribe`, `listen`, `acoustic`, `hearing`

## Default Endpoints

- **LM Studio**: `192.168.56.1:1234`
- **Ollama**: `localhost:11434`

## Use with Gateway

```python
from core.inference import get_multimodal_router, get_inference_system

router = get_multimodal_router()
gateway = get_inference_system()

decision = router.route("your task")

result = gateway.infer(
    prompt="...",
    model=decision.model.name,
    backend=decision.model.backend,
    endpoint=decision.model.endpoint,
)
```

## Model Quick Stats

| Model | Size | Speed | Type |
|-------|------|-------|------|
| whisper | 0.77B | 50 tok/s | Voice specialist |
| moondream | 1.9B | 30 tok/s | Fast vision |
| mistral | 7B | 22 tok/s | Fast general |
| llama2 | 7B | 18 tok/s | Balanced general |
| bakllava | 7B | 20 tok/s | Balanced vision |
| deepseek-r1 | 7B | 15 tok/s | Reasoning |
| moshi | 7B | 25 tok/s | Speech specialist |
| qwen | 7B | 20 tok/s | Large context general |
| llava | 13B | 12 tok/s | High-quality vision |
| qwq | 32B | 10 tok/s | Largest reasoning |

## Configuration

```python
from core.inference import MultiModalConfig, MultiModalRouter

config = MultiModalConfig(
    lmstudio_endpoint="custom:5000",
    ollama_endpoint="custom:11434",
    prefer_reasoning_models=True,
    enable_fallback=True,
    latency_aware=True,
)

router = MultiModalRouter(config)
```

## Debugging

```python
# See routing reason
decision = router.route("task")
print(decision.reason)  # "Routed to qwq (reasoning specialist)"

# See confidence
if decision.confidence >= 0.9:
    print("High confidence match")
else:
    print("Low confidence, verify routing")

# See alternatives
for alt in decision.alternatives:
    print(f"  Alternative: {alt.name}")
```

## Pattern: Real-Time vs Batch

```python
# Real-time (prefer fast models)
decision = router.select_model(ModelCapability.VISION)
# -> moondream (30 tok/s) ✓ Good for streaming

# Batch/Analysis (prefer large models)
decision = router.select_model(ModelCapability.REASONING)
# -> qwq (32B, 10 tok/s) ✓ Good for complex analysis
```

## Tests

```bash
# Run all tests
pytest tests/core/inference/test_multimodal.py -v

# Run examples
python3 core/inference/standalone_example.py

# Quick validation (no deps)
python3 core/inference/multimodal.py
```

## Files

| File | Purpose |
|------|---------|
| `core/inference/multimodal.py` | Core implementation |
| `core/inference/MULTIMODAL.md` | Full API docs |
| `core/inference/README_MULTIMODAL.md` | Getting started |
| `core/inference/standalone_example.py` | Runnable demo |
| `tests/core/inference/test_multimodal.py` | Tests |

---

**Key Concept**: Mixture of Experts pattern with pattern-based task detection
**Location**: `core/inference/multimodal.py`
**Import**: `from core.inference import get_multimodal_router`
