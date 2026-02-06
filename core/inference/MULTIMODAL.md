# BIZRA Multi-Modal Router

Intelligent task routing to the right local model based on input modality and task requirements.

## Architecture

The multi-modal router implements a **Mixture of Experts (MoE)** approach:

1. **Task Detection**: Analyzes input to determine required capability
2. **Model Selection**: Routes to specialized model or fallback
3. **Smart Fallback**: Cascades through general models if specialist unavailable

## Quick Start

```python
from core.inference.multimodal import get_multimodal_router, ModelCapability

# Get singleton router
router = get_multimodal_router()

# Route a task
decision = router.route("Prove the Riemann Hypothesis")
print(f"Model: {decision.model.name}")
print(f"Backend: {decision.model.backend}")
print(f"Endpoint: {decision.model.endpoint}")
```

## Task Types

### 1. Reasoning Tasks

Complex thinking, chain-of-thought, mathematical proofs, analysis.

**Detection patterns:**
- "prove", "derive", "analyze", "synthesize"
- "compare", "evaluate", "trade-offs"
- "debug", "diagnose", "root cause"
- "step by step", "reasoning"

**Recommended models:**
- `deepseek-r1` (7B) - Fast reasoning with CoT
- `qwq` (32B) - Mathematics & logic specialist
- `llama2` (7B) - General reasoning fallback

```python
decision = router.route("Analyze the trade-offs between monolithic vs microservices")
# -> Routed to qwq (reasoning specialist)
```

### 2. Vision Tasks

Image understanding, OCR, visual analysis, charts/diagrams.

**Detection patterns:**
- "image", "photo", "picture", "screenshot"
- "visual", "see", "look at", "describe image"
- "ocr", "read text", "extract image"
- "analyze figure", "chart", "graph"

**Recommended models:**
- `moondream` (1.9B) - Lightweight, fast edge deployment
- `bakllava` (7B) - Balanced vision performance
- `llava` (13B) - High-quality vision understanding

```python
decision = router.route("Describe what you see in this image", explicit_type="image")
# -> Routed to moondream (fastest vision model)

# Or with dict syntax
decision = router.route({
    "text": "Extract text from screenshot",
    "has_image": True
})
# -> Routed to moondream
```

### 3. Voice Tasks

Speech recognition, audio processing, transcription.

**Detection patterns:**
- "audio", "sound", "voice", "speech"
- "transcribe", "speech to text"
- "listen", "hearing", "acoustic"

**Recommended models:**
- `whisper` (0.77B) - Speech-to-text, 99% accuracy
- `moshi` (7B) - Real-time speech understanding & generation

```python
decision = router.route("Transcribe this audio clip", explicit_type="audio")
# -> Routed to whisper (speech specialist)
```

### 4. General Tasks

Default capability for text Q&A, summarization, creative writing.

**Recommended models:**
- `mistral` (7B) - Fast, efficient
- `llama2` (7B) - Well-balanced
- `qwen` (7B) - Large context, multilingual

```python
decision = router.route("What is the capital of France?")
# -> Routed to general model
```

## API Reference

### Router.route(task, explicit_type=None)

Route a task to appropriate model.

**Parameters:**
- `task` (str or dict): Task description or specification
  - If string: text query
  - If dict: `{"text": "...", "has_image": bool, "has_audio": bool}`
- `explicit_type` (str, optional): Override detection with explicit type
  - Valid: `"text"`, `"image"`, `"audio"`, `"reasoning"`

**Returns:** `RoutingDecision`
- `model`: Selected `ModelInfo`
- `capability_match`: The matched `ModelCapability`
- `confidence`: 0.0-1.0 confidence score
- `reason`: Human-readable routing explanation
- `alternatives`: Fallback models if available

**Example:**
```python
decision = router.route("Analyze this chart")

print(decision.model.name)           # "moondream"
print(decision.model.endpoint)       # "localhost:11434"
print(decision.confidence)           # 0.95
print(decision.reason)               # "Routed to moondream..."
print(decision.alternatives)         # [bakllava, llava]
```

### Router.select_model(capability)

Select best model for specific capability.

**Parameters:**
- `capability` (`ModelCapability`): Required capability

**Returns:** `RoutingDecision`

```python
from core.inference.multimodal import ModelCapability

decision = router.select_model(ModelCapability.VISION)
# -> Returns fastest vision model
```

### Router.detect_task_type(text, has_image, has_audio, input_type)

Detect required capability from input.

**Parameters:**
- `text` (str, optional): Query text
- `has_image` (bool): Whether input contains image
- `has_audio` (bool): Whether input contains audio
- `input_type` (str, optional): Explicit type hint

**Returns:** `ModelCapability`

```python
capability = router.detect_task_type(
    text="Describe the image",
    has_image=True
)
# -> ModelCapability.VISION
```

### Router.list_models()

List all registered models.

**Returns:** List of `ModelInfo`

```python
for model in router.list_models():
    print(f"{model.name}: {model.capabilities}")
```

### Router.list_by_capability(capability)

List models supporting specific capability.

**Parameters:**
- `capability` (`ModelCapability`): Required capability

**Returns:** List of `ModelInfo`

```python
reasoning_models = router.list_by_capability(ModelCapability.REASONING)
for model in reasoning_models:
    print(f"  {model.name} ({model.params_b}B) - {model.description}")
```

### Router.register_model(model)

Register a custom model.

**Parameters:**
- `model` (`ModelInfo`): Model to register

```python
from core.inference.multimodal import ModelInfo, ModelCapability

custom = ModelInfo(
    name="local-custom",
    capabilities=[ModelCapability.REASONING, ModelCapability.GENERAL],
    primary_capability=ModelCapability.REASONING,
    backend="lmstudio",
    endpoint="192.168.1.100:1234",
    params_b=70.0,
    context_length=8192,
    speed_tok_per_sec=8.0,
    description="Custom model running on local machine"
)

router.register_model(custom)
```

## Default Models

### Reasoning
- **deepseek-r1** (7B, LM Studio)
  - Advanced reasoning with explicit chain-of-thought
  - Good general performance
- **qwq** (32B, Ollama)
  - Mathematics and logical reasoning specialist
  - Excellent for complex proofs and analysis

### Vision
- **moondream** (1.9B, Ollama)
  - Ultra-lightweight, suitable for edge/real-time
  - Fast vision understanding (~30 tok/s)
- **bakllava** (7B, Ollama)
  - Balanced performance and speed (~20 tok/s)
  - Good for most vision tasks
- **llava** (13B, Ollama)
  - Highest quality vision understanding (~12 tok/s)
  - For when accuracy matters most

### Voice
- **whisper** (0.77B, LM Studio)
  - Speech-to-text with 99% accuracy
  - Fast inference (~50 tok/s)
- **moshi** (7B, Ollama)
  - Real-time speech understanding and generation
  - Suitable for dialogue systems

### General
- **mistral** (7B, Ollama)
  - Fast, efficient (~22 tok/s)
  - Good for real-time chat
- **llama2** (7B, Ollama)
  - Well-balanced, reliable (~18 tok/s)
- **qwen** (7B, LM Studio)
  - Multilingual, large context (32K tokens) (~20 tok/s)

## Configuration

### Default Endpoints
- **LM Studio:** `192.168.56.1:1234`
- **Ollama:** `localhost:11434`

### Custom Configuration
```python
from core.inference.multimodal import MultiModalConfig, MultiModalRouter

config = MultiModalConfig(
    lmstudio_endpoint="custom.host:5000",
    ollama_endpoint="ollama.local:11434",
    prefer_reasoning_models=True,
    enable_fallback=True,
    latency_aware=True
)

router = MultiModalRouter(config)
```

## Selection Algorithm

The router uses a **greedy Mixture of Experts approach**:

1. **Capability Detection**: Analyze input to determine required `ModelCapability`
2. **Primary Capability Match**: Find models where capability is primary
3. **Optimization**:
   - For VISION/VOICE: Prefer fastest model
   - For REASONING: Prefer largest model
4. **Secondary Fallback**: If no primary match, use secondary capabilities
5. **General Fallback**: If still no match, use general-purpose model
6. **Confidence Scoring**:
   - Exact primary match: 0.95
   - Secondary capability: 0.75
   - General fallback: 0.5

## Integration with Gateway

Use with `core.inference.gateway.UnifiedGateway`:

```python
from core.inference.multimodal import get_multimodal_router
from core.inference.gateway import UnifiedGateway

router = get_multimodal_router()
gateway = UnifiedGateway()

# Route task
decision = router.route("Analyze the image")

# Use routed model with gateway
response = gateway.infer(
    prompt="Describe this image",
    model=decision.model.name,
    backend=decision.model.backend,
    endpoint=decision.model.endpoint,
)
```

## References

**Academic Papers:**
- Shazeer et al. (2017): "Outrageously Large Neural Networks for Efficient Conditional Computation"
  - Sparse Mixture of Experts architecture
- Graves (2016): "Adaptive Computation Time for Recurrent Neural Networks"
  - Dynamic routing based on task complexity
- Vaswani et al. (2017): "Attention Is All You Need"
  - Transformer architecture for multi-modal tasks

## Design Principles

1. **Modularity**: Easy to add new models without changing router logic
2. **Extensibility**: Custom models can be registered at runtime
3. **Fallback Safety**: Always has a model available, never fails
4. **Transparency**: Every decision includes reasoning and confidence
5. **Latency Awareness**: Optimizes for speed in real-time scenarios
6. **Signal Quality**: Routes to best-fit specialist for signal preservation

## Future Enhancements

- [ ] Performance profiling: Track actual model speeds vs estimates
- [ ] Adaptive learning: Adjust routing based on historical accuracy
- [ ] Cost optimization: Balance model quality with inference cost
- [ ] Hybrid routing: Combine multiple models for complex tasks
- [ ] Dynamic quantization: Adjust model precision based on task
- [ ] Model chaining: Route task through multiple specialists sequentially
