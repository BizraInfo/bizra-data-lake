# ADR-005: Refinery Daemon Implementation

**Status:** Accepted  
**Date:** 2025-01-27  
**Author:** BIZRA Genesis System  
**Finding:** F-PERF-002 (Ingestion Latency)

## Context

The SAPE Multi-Lens Audit identified batch-only processing as a bottleneck:

> **F-PERF-002:** Refinery is batch-only; for continuous ingestion, a daemon-mode service with streaming writes is required.

The original `bizra_refinery.py` requires manual execution and processes all files in a single run, causing:
- Latency spikes during large ingestions
- No real-time file processing
- Manual intervention required for updates

## Decision

Implement a **Refinery Daemon** as a continuous background service:

1. **File Watching** - Detect new/modified files in real-time
2. **Priority Queue** - Process high-value files first
3. **Throughput Control** - Target 10MB/sec to prevent I/O saturation
4. **Health Endpoints** - Monitor service status via HTTP
5. **Docker Deployment** - Container-ready architecture

### Architecture

```
┌────────────────────────────────────────────────────────────────┐
│                     REFINERY DAEMON                             │
├────────────────────────────────────────────────────────────────┤
│                                                                 │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────┐     │
│   │   WATCHER    │───▶│    QUEUE     │───▶│   WORKER     │     │
│   │  (Polling)   │    │  (Priority)  │    │  (Refinery)  │     │
│   └──────────────┘    └──────────────┘    └──────────────┘     │
│          │                   │                   │             │
│          │                   │                   │             │
│          ▼                   ▼                   ▼             │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │              HEALTH SERVER (:8081)                       │  │
│   │                                                          │  │
│   │  GET /health   →  {"status": "healthy", "queue": 42}    │  │
│   │  GET /metrics  →  {"files_per_sec": 12.5, ...}          │  │
│   │  GET /queue    →  {"size": 42, "max_size": 10000}       │  │
│   │                                                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│   ┌─────────────────────────────────────────────────────────┐  │
│   │                 LEDGER OUTPUT (JSONL)                    │  │
│   │                                                          │  │
│   │  {"filename": "data.py", "hash": "abc...", ...}         │  │
│   │  {"filename": "config.json", "hash": "def...", ...}     │  │
│   │                                                          │  │
│   └─────────────────────────────────────────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Components

#### FileWatcher
- Polls directories every 2 seconds
- Tracks file modification times
- Skips system directories (.git, node_modules, etc.)
- Queues new/modified files for processing

#### FileQueue
- Priority-based queue (PriorityQueue)
- High priority: .py, .rs, .md, .json
- Medium priority: Other known extensions
- Low priority: Unknown extensions
- Deduplication via path tracking

#### RefineryWorker
- Batch processing (default: 50 files/batch)
- Throughput limiting (default: 10MB/sec)
- Content hashing (SHA256)
- Chain continuity from Genesis Block
- Atomic ledger writes

#### HealthServer
- Port 8081 (configurable)
- `/health` - Service status
- `/metrics` - Processing stats
- `/queue` - Queue status

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `BIZRA_REFINERY_THROUGHPUT` | 10 | MB/sec target |
| `BIZRA_REFINERY_PORT` | 8081 | Health server port |
| `BIZRA_REFINERY_BATCH_SIZE` | 50 | Files per batch |

### Metrics

```json
{
  "files_processed": 1234,
  "bytes_processed_mb": 856.42,
  "total_value": 42156.75,
  "errors": 3,
  "queue_size": 42,
  "files_per_sec": 12.5,
  "bytes_per_sec_mb": 8.2,
  "uptime_sec": 3600
}
```

### Docker Deployment

```yaml
# docker-compose.yml
refinery:
  build:
    context: .
    dockerfile: Dockerfile.refinery
  ports:
    - "8081:8081"
  volumes:
    - ./bizra_data_vault:/app/bizra_data_vault:ro
    - ./BIZRA_KNOWLEDGE_LEDGER.jsonl:/app/BIZRA_KNOWLEDGE_LEDGER.jsonl
```

### CLI Usage

```bash
# Direct execution
python -m core.refinery_daemon \
  --watch bizra_data_vault/roots \
  --ledger BIZRA_KNOWLEDGE_LEDGER.jsonl \
  --throughput 10 \
  --port 8081

# Docker
docker-compose up refinery
```

## Invariants

1. **I5: Throughput Bound**  
   Processing rate ≤ target_bytes_per_sec

2. **I6: Queue Bound**  
   Queue size ≤ max_queue_size (10000)

3. **I7: Chain Continuity**  
   All records linked to Genesis Block hash

## Consequences

### Positive
- Real-time file ingestion
- Configurable throughput prevents I/O saturation
- Docker-ready for production deployment
- Health monitoring for observability
- Priority processing for high-value files

### Negative
- Polling-based (no inotify) on Windows
- Single-writer ledger (no concurrent daemons)
- No backpressure to external systems

### Future Work
- inotify/FSEvents for native watching
- Distributed queue (Redis-backed)
- Webhook notifications
- S3/cloud storage support

## Files

- `core/refinery_daemon.py` - Daemon implementation
- `Dockerfile.refinery` - Container definition
- `docker-compose.yml` - Service configuration (refinery service added)

## References

- ADR-001: Genesis Block
- BIZRA Unified Execution Blueprint v2.0
- F-PERF-002: Ingestion Latency Finding
