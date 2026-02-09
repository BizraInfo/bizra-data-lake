# bizra-inference

**BIZRA Inference** — Sovereign LLM Gateway with Tiered Backends.

## Overview

Unified inference gateway routing requests to the optimal backend:

1. **LM Studio** (primary) — 192.168.56.1:1234
   - Reasoning: DeepSeek-R1, Qwen-72B
   - Agentic: function calling, tool use
   - Vision: LLaVA, Qwen-VL
   - Voice: Whisper, Moshi
2. **Ollama** (fallback) — localhost:11434
3. **LlamaCpp** (embedded) — edge/offline

## Key Types

| Type | Purpose |
|------|---------|
| `InferenceGateway` | Top-level request router |
| `ModelSelector` | Selects backend based on `TaskComplexity` |
| `ModelTier` | Quality tiers (Reasoning, Agentic, Vision, Voice) |
| `Backend` | Trait for pluggable inference backends |

## Dependencies

- bizra-core — Identity and PCI protocol
- reqwest — HTTP client for backend communication

## License

MIT
