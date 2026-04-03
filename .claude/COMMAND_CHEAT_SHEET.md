# BIZRA Slash Commands - Cheat Sheet

## Quick Command Reference

### 🏗️ Build & Setup
```bash
/rust [release|debug]              # Build Rust Elite engine
/python                            # Setup Python kernel
```

### 🐳 Services (Docker)
```bash
/docker up                         # Start all services
/docker down                       # Stop all services
/docker restart [service]          # Restart service(s)
/docker status                     # Service health check
/docker logs [service]             # View service logs
```

### 🧪 Testing
```bash
/rust [test] [--nocapture]         # Run Rust tests
/python [path] [-v|-vv]            # Run Python tests
```

### ✅ Validation
```bash
/ihsan                             # Validate Ihsān constitution
/sape                              # Validate SAPE probes
```

### 📋 Evidence
```bash
/receipts count                    # Count receipts by type
/receipts validate                 # Validate all receipts
/receipts recent                   # Last 10 receipts
/receipts stats                    # Statistical analysis
```

### 📝 General
```bash
/commit [message]                  # Create commit with receipt
/guide                             # Show full command guide
```

## Common Workflows

### Development Cycle
```bash
/docker up → [code] → /rust → /python → /commit "message"
```

### Pre-Deployment
```bash
/rust release → /python → /ihsan → /sape → /receipts validate
```

### Troubleshooting
```bash
/docker status → /docker logs elite → /receipts recent
```

## Services (Port Reference)

| Service | Port | Purpose |
|---------|------|---------|
| elite | 8080 | Rust PAT+SAT+SAPE |
| kernel | 8010 | Python FastAPI |
| postgres | 5432 | Knowledge graph |
| synapse | 6379 | Redis (TLS) |
| wisdom | 7474/7687 | Neo4j |
| vectors | 8001 | ChromaDB |
| refinery | 8081 | Refinery daemon |

## Critical Tests (Must Pass)

- **Rust**: PAT/SAT, Ihsān, SAPE, Receipts
- **Python**: Agent Factory, SAPE, FATE, Synapse Security

## Validation Thresholds

- **Ihsān**: ≥0.99 (production)
- **SAPE Probes**: <100ms per probe
- **Test Coverage**: ≥65% (Python), varies (Rust)

## Receipt Types

- `build-*.json` - Build operations
- `test-*.json` - Test executions
- `ihsan-*.json` - Ihsān validation
- `sape-*.json` - SAPE validation
- `commit-*.json` - Git commits
- `receipt-validation-*.json` - Meta-receipts

## Fail-Closed Triggers

Commands **BLOCK** on:
- `/rust` - Clippy warnings
- `/ihsan` - Weights ≠ 1.0 or threshold < 0.99
- `/sape` - Missing probes
- `/commit` - Syntax errors or secrets
- `/receipts validate` - Schema violations

## Help & Documentation

```bash
/guide                             # Full command guide
/help                              # Claude Code help
cat .claude/commands/README.md     # Complete docs
cat .claude/commands/guide.md      # User guide
```

## Environment Variables

```bash
IHSAN_THRESHOLD=0.99               # Ethics threshold
SAPE_CACHE_TTL=3600                # SAPE cache (seconds)
SYNAPSE_URL=rediss://...           # Redis (TLS)
RUST_LOG=info,bizra=debug          # Logging level
```

---

**Print this card for quick reference during development**

Full documentation: `.claude/commands/README.md` or type `/guide`
