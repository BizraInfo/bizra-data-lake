# BIZRA General Rules

Universal rules that apply to all BIZRA development activities.

## Core Principles

### Receipt-First Development
- Every significant operation MUST emit an evidence receipt
- Receipts are append-only - never delete or modify existing receipts
- Receipt schema: `receipt_id`, `timestamp`, `task_summary`, `rejection_codes`, `escalation_level`, `integrity_hash`
- Use SHA-256 for integrity hashes

### Fail-Closed Error Handling
- Critical errors MUST block execution, never proceed silently
- SAT consensus failures MUST escalate via FATE
- Ihsān gate failures (< 0.95) MUST reject the request
- SAPE probe failures MUST be logged with evidence

### Ihsān (إحسان) Excellence
- Production threshold: 0.95 (balanced for practical flexibility)
- 8 dimensions: correctness (0.22), safety (0.22), user_benefit (0.14), efficiency (0.12), auditability (0.12), anti_centralization (0.08), robustness (0.06), adl_fairness (0.04)
- Constitution is single source of truth: `constitution/ihsan_v1.yaml`

## Code Quality Standards

### Never Skip
- Always read files before editing
- Never guess file contents or structure
- Don't create new files when editing existing ones works
- Don't add features beyond what's requested

### Documentation
- Only add comments where logic isn't self-evident
- Don't add docstrings to code you didn't write
- Keep commit messages focused on "why" not "what"

### Dependencies
- Pin versions in requirements files
- Check for security vulnerabilities before adding dependencies
- Prefer stdlib over external packages when reasonable

## Git Workflow

### Commits
- Don't commit unless explicitly asked
- Never use `--force` unless explicitly requested
- Include receipt evidence for significant commits
- Use conventional commit format: `type(scope): description`

### Protected Files
These files require extra care:
- `constitution/ihsan_v1.yaml` - Constitution
- `src/receipts.rs` - Receipt schema (sync with core/fate.py)
- `docker-compose.yml` - Service configuration
- `config/redis/*.pem` - TLS certificates (never commit private keys)

## Environment

### Required Services
- PostgreSQL (5432) - Knowledge graph
- Redis/Synapse (6379) - State persistence with TLS
- Neo4j/Wisdom (7474/7687) - Graph evidence
- ChromaDB/Vectors (8001) - Embeddings

### Environment Variables
- `IHSAN_THRESHOLD=0.95` - Balanced for practical flexibility
- `SYNAPSE_URL=rediss://...` - Must use TLS (rediss://)
- `BIZRA_ADAPTER_MODE=real` - Use real LLMs in production

## When Uncertain

1. Ask for clarification via AskUserQuestion
2. Check CLAUDE.md for project-specific guidance
3. Review existing patterns in codebase
4. Prefer conservative, reversible changes
