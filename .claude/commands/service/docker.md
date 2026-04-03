---
allowed-tools: Bash(docker:*), Bash(docker-compose:*), Bash(docker compose:*)
description: Manage BIZRA Docker services
argument-hint: [up|down|restart|status|logs] [service-name]
---

# BIZRA Docker Service Management

## Current Service Status

!`docker compose ps --format json 2>/dev/null | jq -r '.[] | "\(.Service): \(.State)"' || echo "Services not running"`

## Command: **$1** | Target: **${2:-all}**

## Service Architecture

| Service | Port | Purpose |
|---------|------|---------|
| **elite** | 8080 | Rust PAT+SAT+SAPE engine |
| **kernel** | 8010 | Python FastAPI (SAPE/FATE/LLM) |
| **postgres** | 5432 | Knowledge graph + pgvector |
| **synapse** | 6379 | Redis (State/receipts/FATE) |
| **wisdom** | 7474/7687 | Neo4j (Graph evidence) |
| **vectors** | 8001 | ChromaDB (Embeddings) |
| **refinery** | 8081 | Python refinery daemon |

## Your Task

### For "up" command:
```bash
# Start services
docker compose up -d ${2:-}

# Wait for health checks
sleep 5

# Verify services are healthy
docker compose ps --all

# Test connectivity
curl -f http://localhost:8080/health || echo "Elite service not ready"
curl -f http://localhost:8010/health || echo "Kernel service not ready"
```

### For "down" command:
```bash
# Check for running tasks first
docker compose exec elite echo "Checking for active tasks..." || true

# Stop services gracefully
docker compose down ${2:-}

# Verify stopped
docker compose ps --all
```

### For "restart" command:
```bash
# Restart specific service or all
docker compose restart ${2:-}

# Wait and verify
sleep 3
docker compose ps ${2:-}
```

### For "status" command:
```bash
# Detailed status
docker compose ps --all --no-trunc

# Resource usage
docker stats --no-stream

# Health checks
for service in elite kernel postgres synapse wisdom vectors refinery; do
    echo "$service: $(docker compose ps $service --format json | jq -r '.[0].Health // "no health check"')"
done
```

### For "logs" command:
```bash
# Follow logs for specific service or all
docker compose logs -f --tail=100 ${2:-}
```

## Critical Checks

### TLS Certificates (for synapse)
- Verify `config/redis/ca-cert.pem` exists
- Check `config/redis/redis-server-cert.pem` is valid
- Ensure `REDIS_PASSWORD` is set in environment

### Health Checks
- Elite service responds on :8080/health
- Kernel service responds on :8010/health
- Postgres accepts connections
- Redis accepts TLS connections

### Fail-Closed Requirements
- If TLS certificates are missing, synapse service WILL FAIL
- If services don't become healthy within 60s, investigate before proceeding
- Never start services with invalid configuration

## Evidence Generation

After "up" command:
- Create service start receipt with timestamp
- Record service versions and health status
- Log any startup warnings or errors

Report:
- Services started/stopped
- Health check status
- Any errors or warnings
- Resource usage (for status command)
