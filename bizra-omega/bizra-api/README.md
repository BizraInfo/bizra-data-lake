# bizra-api

**BIZRA Sovereign API Gateway** — Unified REST/WebSocket interface for all BIZRA services.

## Overview

Axum-based API gateway exposing:

- **Identity Management** — Ed25519 keypair generation and management
- **PCI Protocol** — Envelope creation, signing, and verification
- **Inference Gateway** — Tiered LLM access (LM Studio → Ollama → LlamaCpp)
- **Federation Status** — Gossip protocol and consensus monitoring
- **Autopoiesis** — Pattern memory and preference tracking
- **WebSocket** — Real-time event streaming

## Architecture

```
Client → Rate Limiter → Router → Handler → Service Layer → Response
```

### Modules

| Module | Purpose |
|--------|---------|
| `handlers/` | Request handlers for each service domain |
| `middleware/` | Rate limiting, CORS, tracing |
| `routes` | Axum router construction |
| `state` | Shared application state (`AppState`) |
| `websocket` | WebSocket upgrade and event streaming |
| `error` | Unified `ApiError` → HTTP status mapping |

## Dependencies

- [axum](https://github.com/tokio-rs/axum) — Web framework
- [tower-http](https://github.com/tower-rs/tower-http) — CORS, compression, tracing
- bizra-core, bizra-inference, bizra-federation, bizra-autopoiesis

## Standing on Giants

Fielding (REST), Kleppmann (DDIA), Axum team

## License

MIT
