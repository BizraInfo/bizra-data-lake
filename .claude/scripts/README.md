# BIZRA CLI Scripts

Automation scripts for Claude Code CLI integration with BIZRA.

## Available Scripts

### bizra-dev.sh

Development session helper for starting and managing Claude Code sessions.

```bash
# Start new dev session
./bizra-dev.sh start

# Continue last session
./bizra-dev.sh continue

# Resume named session
./bizra-dev.sh resume -n "feature-name"

# Quick non-interactive query
./bizra-dev.sh quick "Build Rust in release mode"

# Run validation
./bizra-dev.sh validate

# Show project status
./bizra-dev.sh status

# Options
-m, --model     Model (sonnet/opus)
-p, --plan      Plan mode
-v, --verbose   Verbose logging
-n, --name      Session name
```

### bizra-validate.sh

Full validation pipeline for all BIZRA gates.

```bash
# Run all gates
./bizra-validate.sh

# Strict mode (exit on first failure)
./bizra-validate.sh --strict

# JSON output for CI parsing
./bizra-validate.sh --json
```

**Gates**:
1. Rust Build
2. Rust Clippy
3. Rust Tests
4. Python Imports
5. Python Tests
6. Ihsān Constitution
7. Receipt Schema

### ci-claude-gate.sh

CI/CD integration for running Claude Code validation gates.

```bash
# Run specific gate
./ci-claude-gate.sh build
./ci-claude-gate.sh test
./ci-claude-gate.sh ihsan
./ci-claude-gate.sh sape
./ci-claude-gate.sh receipts

# Run all gates
./ci-claude-gate.sh full

# Options
--json          JSON output
--max-turns N   Max agentic turns (default: 10)
--max-budget N  Max USD budget (default: 2.00)
--timeout N     Timeout in seconds (default: 300)
```

### generate-receipt.sh

Create BIZRA-compliant evidence receipts.

```bash
# Generate build receipt
./generate-receipt.sh build "Rust build completed successfully"

# Generate test receipt with failure
./generate-receipt.sh test "Tests failed" -s failure -e "TEST_FAILURE"

# Generate validation receipt with escalation
./generate-receipt.sh validation "Ihsān gate passed" -l None

# Generate chained receipt
./generate-receipt.sh deploy "Deployed to staging" -p "build-20260120-103000-abc123"

# Options
-s, --status   Status (success/failure)
-e, --error    Error/rejection code
-l, --level    Escalation level (None/Low/Medium/High/Critical)
-o, --output   Output directory
-p, --parent   Parent receipt ID (for chaining)
```

## Usage Examples

### Development Workflow

```bash
# Morning: Start fresh session
./bizra-dev.sh start -p  # Plan mode for exploration

# Work on feature
# ...

# Before commit: Validate
./bizra-validate.sh

# End of day: Session auto-saved
# Can continue tomorrow with: ./bizra-dev.sh continue
```

### CI/CD Pipeline

```bash
#!/bin/bash
# .github/workflows/bizra-validate.yml

set -e

# Run full validation with timeout
./bizra-validate.sh --strict --json > results.json

# Or use Claude Code for more detailed validation
./ci-claude-gate.sh full --json --timeout 600
```

### Receipt Generation

```bash
# After successful build
receipt_id=$(./generate-receipt.sh build "Rust release build")

# After tests pass (chain to build)
./generate-receipt.sh test "All tests passed" -p "$receipt_id"

# After deployment
./generate-receipt.sh deploy "Production deployment" -s success
```

### Quick Validation

```bash
# Check project status
./bizra-dev.sh status

# Quick Ihsān check
./bizra-dev.sh quick "Validate Ihsān constitution"

# Quick receipt count
./bizra-dev.sh quick "Count receipts in docs/evidence/receipts/"
```

## Environment Variables

These scripts respect the following environment variables:

```bash
# BIZRA Configuration
BIZRA_MODE=development      # development/production
IHSAN_THRESHOLD=0.99        # Ihsān gate threshold

# Redis (for receipt storage)
SYNAPSE_URL=rediss://:pass@synapse:6379

# Neo4j (for evidence graphs)
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=password
```

## Integration with Hooks

These scripts work with the hooks system in `.claude/hooks/`:

- `validate-bash.py` - Pre-validates bash commands
- `post-bash.py` - Post-processing for receipts
- `inject-context.py` - Injects BIZRA context

## Integration with Commands

These scripts complement slash commands in `.claude/commands/`:

- `/rust` - Build Rust components
- `/python` - Validate Python kernel
- `/ihsan` - Validate constitution
- `/sape` - Run SAPE probes
- `/receipts` - Manage receipts

## Troubleshooting

### Script Not Executable

```bash
chmod +x .claude/scripts/*.sh
```

### Line Ending Issues

```bash
# Convert CRLF to LF
sed -i 's/\r$//' .claude/scripts/*.sh
```

### Claude Code Not Found

```bash
# Check installation
which claude

# Install if missing
npm install -g @anthropic/claude-code
```

### Permission Denied

```bash
# Run with explicit bash
bash .claude/scripts/bizra-validate.sh
```

## See Also

- CLI Reference: `.claude/CLI_REFERENCE.md`
- CLI Cheat Sheet: `.claude/CLI_CHEAT_SHEET.md`
- Hooks: `.claude/hooks/README.md`
- Commands: `.claude/commands/README.md`
