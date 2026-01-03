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

The API token is set via the `BIZRA_API_TOKEN` environment variable. The server fails to start if the token is missing.

## Request ID

All responses include an `x-request-id` header for correlation across logs, receipts, and metrics.

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

## Architecture: PAT + SAT Dual-Agent Pipeline

### PAT (Primary Agent Team) - 7 Agents

Specialized execution agents that handle the actual task work:

- Code, Security, Ethics, Reasoning, Memory, Integration, Synthesis

### SAT (Sentinel Agent Team) - 5 Agents (Veto-Only)

Validation sentinels that gate requests before PAT execution:

- Security, Ethics, Policy, Complexity, Resources

**⚠️ SAT Uses Veto-Only Consensus (Fail-Safe Design)**

Unlike democratic voting (e.g., 3/5 majority), SAT operates on a **fail-safe veto model**:

- **ANY single SAT agent can reject** the entire request
- All 5 agents must approve for execution to proceed
- This maximizes safety at the potential cost of false rejections

Rejection types trigger immediate abort:
- `RejectionType::Security` - Malicious patterns detected
- `RejectionType::Ethics` - Ihsān constitution violation
- `RejectionType::Policy` - Policy constraint breach
- `RejectionType::Complexity` - Task exceeds safe complexity
- `RejectionType::Resources` - Resource limits exceeded

---

## Example Requests

### Simple Task Execution
```bash
curl -X POST http://localhost:8080/dual/execute \
  -H "Authorization: Bearer $BIZRA_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "user-12345",
    "task": "Write a Python function to calculate factorial",
    "requirements": ["handle negative numbers", "include docstring"],
    "target": "python_function",
    "priority": "High"
  }'
```

### Enhanced Execution with Context
```bash
curl -X POST http://localhost:8080/enhanced/execute \
  -H "Authorization: Bearer $BIZRA_API_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "base": {
      "user_id": "user-12345",
      "task": "Review this code for security issues",
      "requirements": ["check SQL injection", "check XSS"],
      "target": "security_report",
      "context": {
        "code": "def login(user, pwd): return db.query(f\"SELECT * FROM users WHERE name={user}\")"
      }
    },
    "reasoning_preference": "ChainOfThought",
    "enable_sub_agents": true
  }'
```

Note: Use `mcp_tools_whitelist` to restrict MCP tool access for this request. If provided as an empty list, tool listing and calls are blocked.

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
    "content": "rm -rf /"
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
| 422 | Unprocessable entity (Ihsan gate failed) |
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
