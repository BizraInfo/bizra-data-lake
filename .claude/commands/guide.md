---
description: BIZRA slash commands guide and quick reference
---

# BIZRA Slash Commands Guide

Welcome to the BIZRA slash command system! This guide provides an overview of all available commands organized by category.

## Command Categories

### 🏗️ Build Commands (`.claude/commands/build/`)

**`/rust [release|debug]`**
- Build Rust Elite engine with validation
- Runs clippy linting
- Generates build receipt
- Default: release mode

**`/python`**
- Setup and validate Python kernel
- Install dependencies
- Validate imports and type hints
- Check configuration files

**Usage**:
```
> /rust release
> /python
```

### 🐳 Service Commands (`.claude/commands/service/`)

**`/docker [up|down|restart|status|logs] [service-name]`**
- Manage Docker Compose services
- Check health status
- View logs
- Monitor resources

**Services**:
- `elite` (8080) - Rust PAT+SAT+SAPE
- `kernel` (8010) - Python FastAPI
- `postgres` (5432) - Knowledge graph
- `synapse` (6379) - Redis (TLS)
- `wisdom` (7474/7687) - Neo4j
- `vectors` (8001) - ChromaDB
- `refinery` (8081) - Refinery daemon

**Usage**:
```
> /docker up
> /docker status
> /docker logs elite
> /docker restart kernel
```

### 🧪 Test Commands (`.claude/commands/test/`)

**`/rust [test-name] [--nocapture]`**
- Run Rust test suite
- Test PAT/SAT integration
- Validate Ihsān gates
- Check SAPE probes
- Generate test receipt

**`/python [test-path] [-v|-vv]`**
- Run pytest with coverage
- Test agent factory
- Validate SAPE logic
- Check TLS security
- Generate coverage report

**Usage**:
```
> /rust
> /rust ihsan --nocapture
> /python tests/test_sape.py -vv
> /python
```

### ✅ Validation Commands (`.claude/commands/validation/`)

**`/ihsan`**
- Validate Ihsān (إحسان) constitution
- Check ethical dimension weights
- Verify 0.99 threshold
- Cross-reference Rust/Python implementations
- Ensure weight sum = 1.0

**`/sape`**
- Validate SAPE 9-probe system
- Check pattern elevation (>3 repetitions)
- Test Neo4j graph evidence
- Measure probe performance
- Verify Redis connectivity

**Usage**:
```
> /ihsan
> /sape
```

### 📋 Evidence Commands (`.claude/commands/evidence/`)

**`/receipts [count|validate|recent|stats]`**
- Manage receipt evidence
- Validate schema compliance
- Analyze success rates
- Check integrity hashes
- Track FATE escalations

**Subcommands**:
- `count` - Receipt inventory by type
- `validate` - Schema validation
- `recent` - Show last 10 receipts
- `stats` - Statistical analysis

**Usage**:
```
> /receipts count
> /receipts validate
> /receipts recent
> /receipts stats
```

### 📝 General Commands

**`/commit [message]`**
- Create BIZRA-compliant git commit
- Run pre-commit validation
- Generate commit receipt
- Add Claude co-authorship
- Check for protected files

**Usage**:
```
> /commit "feat(rust): add new SAPE probe"
> /commit
```

## Quick Reference Matrix

| Command | Purpose | Critical? | Evidence |
|---------|---------|-----------|----------|
| `/rust` | Build Rust engine | Yes | Build receipt |
| `/python` | Setup Python kernel | Yes | - |
| `/docker` | Manage services | Yes | Service receipt |
| `/rust` (test) | Run Rust tests | Yes | Test receipt |
| `/python` (test) | Run Python tests | Yes | Coverage report |
| `/ihsan` | Validate constitution | Yes | Validation receipt |
| `/sape` | Validate probes | Yes | Validation receipt |
| `/receipts` | Manage evidence | Yes | Meta-receipt |
| `/commit` | Create commit | Yes | Commit receipt |

## Common Workflows

### Development Cycle
```bash
# 1. Start services
/docker up

# 2. Make code changes
# ... edit files ...

# 3. Run tests
/rust
/python

# 4. Validate gates
/ihsan
/sape

# 5. Commit changes
/commit "feat: add new feature"

# 6. Check evidence
/receipts recent
```

### Pre-deployment Validation
```bash
# 1. Build components
/rust release
/python

# 2. Run full test suite
/rust
/python

# 3. Validate all gates
/ihsan
/sape

# 4. Check receipts
/receipts validate
/receipts stats

# 5. Verify services
/docker status
```

### Troubleshooting
```bash
# Check service status
/docker status

# View service logs
/docker logs elite
/docker logs kernel

# Validate configuration
/ihsan
/sape

# Check recent evidence
/receipts recent

# Analyze test failures
/rust --nocapture
/python -vv
```

## Command Features

### Arguments
Commands support positional arguments:
```bash
/rust release          # $1 = "release"
/docker logs elite     # $1 = "logs", $2 = "elite"
/commit "fix bug"      # $ARGUMENTS = "fix bug"
```

### Bash Execution
Commands can execute bash with `!` prefix:
```markdown
Current branch: !`git branch --show-current`
```

### File References
Commands can reference files with `@`:
```markdown
Review @src/receipts.rs
```

### Hooks Integration
Commands trigger hooks:
- PreToolUse: Validate before execution
- PostToolUse: Check results after execution
- Stop: Validate completion

## BIZRA Principles in Commands

### 1. Receipt-First Development
Every command generates evidence:
- Build receipts for `/rust`, `/python`
- Test receipts for test commands
- Validation receipts for `/ihsan`, `/sape`
- Commit receipts for `/commit`

### 2. Fail-Closed Error Handling
Commands block on critical failures:
- `/rust` - Blocks on clippy errors
- `/ihsan` - Blocks if weights ≠ 1.0
- `/sape` - Blocks if probes missing
- `/commit` - Blocks on syntax errors

### 3. Ihsān Gate Enforcement
Commands validate 0.99 threshold:
- `/ihsan` checks constitution
- `/sape` validates probe scores
- `/rust` compiles gate logic
- `/python` tests gate implementation

### 4. Evidence-Driven Workflow
Commands produce auditable evidence:
- Receipts in `docs/evidence/receipts/`
- SHA-256 integrity hashing
- Append-only storage
- Schema compliance validation

## Advanced Usage

### Command Chaining
Execute multiple commands in sequence:
```bash
> /docker up && /rust && /python && /commit "ready for testing"
```

### Conditional Execution
Use command output for decisions:
```bash
> /receipts validate && echo "✓ All receipts valid"
```

### Custom Arguments
Pass specific arguments:
```bash
> /rust debug
> /python tests/test_specific.py -vvv
> /docker restart elite kernel
```

## Environment Variables

Commands respect BIZRA environment:
```bash
RUST_LOG=debug           # Rust logging level
IHSAN_THRESHOLD=0.99     # Ethics threshold
SAPE_CACHE_TTL=3600      # SAPE cache duration
SYNAPSE_URL=rediss://... # Redis connection
OLLAMA_HOST=...          # LLM endpoint
```

## Troubleshooting Commands

### Command Not Found
```bash
# List available commands
/help

# Check command file exists
ls .claude/commands/
```

### Command Fails
```bash
# Check command syntax
cat .claude/commands/<category>/<command>.md

# Test command manually
cat test-input.json | bash -c "$(grep -A100 '```bash' .claude/commands/<category>/<command>.md | head -n-1 | tail -n+2)"
```

### Permission Issues
```bash
# Make commands executable
chmod +x .claude/commands/**/*.md

# Check hooks permissions
ls -la .claude/hooks/
```

## Getting Help

### Built-in Help
```bash
/help                 # List all commands
/help <command>       # Get command details
```

### Documentation
- Command README: `.claude/commands/README.md`
- Hooks guide: `.claude/hooks/README.md`
- BIZRA guide: `CLAUDE.md`
- Quick reference: `.claude/HOOKS_QUICK_REFERENCE.md`

### Examples
All commands include:
- Usage examples
- Expected output
- Error handling
- Receipt generation

## See Also

- **CLAUDE.md** - Full BIZRA development guide
- **.claude/hooks/** - Hook system documentation
- **docs/evidence/** - Evidence and receipts
- **constitution/** - Ihsān and governance

---

**Command Philosophy**: "Receipt-first. Fail-closed. Evidence-driven. Every command leaves a trail."

Type `/help` for a list of all available commands.
