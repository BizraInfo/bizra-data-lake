# iceoryx-bridge

**Iceoryx2 Bridge** — Zero-Copy IPC for BIZRA Sovereign LLM.

## Overview

Ultra-low-latency inter-process communication (target: **250ns**) between:

- **TypeScript Orchestrator** (Elite Blueprint)
- **Python Inference Sandbox** (llama.cpp)
- **Rust Gate Chain** (FATE validation)

Uses Iceoryx2 for true zero-copy shared memory transport.

## Channel Topology

| Channel | Direction | Max Size |
|---------|-----------|----------|
| InferenceRequest | Orchestrator → Sandbox | 64KB |
| InferenceResponse | Sandbox → Orchestrator | 1MB |
| GateRequest | Orchestrator → FATE Gate | 1MB |
| GateResponse | FATE Gate → Orchestrator | 64KB |
| ModelRegistry | Broadcast | 256KB |
| Control | Bidirectional | 4KB |
| Metrics | Broadcast | 16KB |

## Wire Format

- **MessagePack** (rmp-serde) for compact binary serialization
- **MessageEnvelope** wrapping all IPC messages with type, timestamp, sender/target, and payload

## Benchmarks

```bash
cargo bench -p iceoryx-bridge
```

Covers: envelope creation/decode, roundtrip at various payload sizes, channel metadata, throughput.

## Build Requirements

- Rust stable toolchain
- Node.js (for NAPI bindings)

## License

MIT
