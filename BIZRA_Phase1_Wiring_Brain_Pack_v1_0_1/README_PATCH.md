# README Patch Notes (Phase 1: Wiring the Brain)

## Add: Prerequisites
- Ollama running on `http://localhost:11434`
- (Optional) LM Studio running on `http://localhost:1234` with OpenAI-compatible server enabled

## Add: Runtime Config
- `docs/runtime/slots.yaml` defines slot -> provider -> model routing
- `config/model-family-genesis-*.yaml` defines which models are allowed and how they are pinned (digest + modelfile hash)

## Add: Verification
- `scripts/setup_intelligence.ps1` pulls required models
- Integration test:
  - `set BIZRA_INT_TESTS=1`
  - `cargo test --test integration_intelligence`
