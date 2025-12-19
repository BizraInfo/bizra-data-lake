# BIZRA META ALPHA ELITE - API Documentation

## OpenAPI Specification

The complete API documentation is available in OpenAPI 3.1.0 format:

- **File**: [openapi.yaml](openapi.yaml)
- **Format**: OpenAPI 3.1.0 / Swagger

### Viewing the Documentation

#### Option 1: Swagger Editor (Online)
1. Go to [editor.swagger.io](https://editor.swagger.io)
2. File → Import File → Select `openapi.yaml`

#### Option 2: Swagger UI (Local)
```bash
docker run -p 8081:8080 -e SWAGGER_JSON=/spec/openapi.yaml \
  -v $(pwd)/docs:/spec swaggerapi/swagger-ui
```
Then open http://localhost:8081

#### Option 3: VS Code Extension
Install "OpenAPI (Swagger) Editor" extension and open `openapi.yaml`

---

## Quick Reference

### Public Endpoints (No Auth)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | System status and capabilities |
| `/health` | GET | Health check |
| `/metrics` | GET | Prometheus metrics |
| `/stats` | GET | Aggregated statistics |
| `/dashboard` | GET | Dashboard redirect |
| `/static/*` | GET | Static files |

### Protected Endpoints (Bearer Auth Required)

#### Core Execution
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/dual/execute` | POST | Execute dual-agent task (PAT+SAT) |
| `/enhanced/execute` | POST | Enhanced execution with SAPE |

#### MCP Integration
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/mcp/rpc` | POST | JSON-RPC 2.0 for MCP tools |
| `/mcp/tools` | GET | List available MCP tools |

#### SAPE (Security)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/sape/probes` | POST | Execute security probes |
| `/sape/stats` | GET | SAPE statistics |

#### Ollama (LLM)
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ollama/generate` | POST | Text generation |
| `/ollama/chat` | POST | Multi-turn chat |
| `/ollama/status` | GET | Ollama status |

---

## Authentication

Protected endpoints require a Bearer token:

```bash
curl -X POST http://localhost:8080/dual/execute \
  -H "Authorization: Bearer YOUR_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{"task": "Write a factorial function"}'
```

The API token is set via the `BIZRA_API_TOKEN` environment variable.

---

## Rate Limiting

- **Algorithm**: Token bucket
- **Max Tokens**: 100
- **Refill Rate**: 2 tokens/second
- **Scope**: Per-IP address

When rate limited, you'll receive:
```json
{
  "error": "Rate Limited",
  "message": "Too many requests, please try again later"
}
```

---

## Example Requests

### Simple Task Execution
```bash
curl -X POST http://localhost:8080/dual/execute \
  -H "Authorization: Bearer $BIZRA_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Write a Python function to calculate factorial",
    "priority": "high"
  }'
```

### Enhanced Execution with Context
```bash
curl -X POST http://localhost:8080/enhanced/execute \
  -H "Authorization: Bearer $BIZRA_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "task": "Review this code for security issues",
    "context": "def login(user, pwd): return db.query(f\"SELECT * FROM users WHERE name={user}\")",
    "adapter_modes": ["security", "coding"],
    "enable_sape": true
  }'
```

### MCP Tool Call
```bash
curl -X POST http://localhost:8080/mcp/rpc \
  -H "Authorization: Bearer $BIZRA_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "jsonrpc": "2.0",
    "id": "1",
    "method": "tools/list",
    "params": {}
  }'
```

### SAPE Probe
```bash
curl -X POST http://localhost:8080/sape/probes \
  -H "Authorization: Bearer $BIZRA_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "input": "rm -rf /",
    "dimensions": ["threat_scan", "safety_check"]
  }'
```

---

## Response Codes

| Code | Description |
|------|-------------|
| 200 | Success |
| 400 | Bad request (validation failed) |
| 401 | Unauthorized (missing/invalid token) |
| 403 | Forbidden (SAT rejected request) |
| 429 | Rate limited |
| 500 | Internal server error |
| 503 | Service unavailable (Ollama down) |

---

## Metrics

Available at `/metrics` in Prometheus format:

- `bizra_sat_validations_total{outcome}` - SAT validation outcomes
- `bizra_ihsan_score_computed` - Ihsān scores histogram
- `bizra_a2a_delegations_total{agent,result}` - A2A delegation counts
- `bizra_http_requests_allowed_total` - Requests that passed rate limiting
- `bizra_http_requests_rate_limited_total` - Rate-limited requests
- `bizra_sape_probes_total{dimension}` - SAPE probe counts
