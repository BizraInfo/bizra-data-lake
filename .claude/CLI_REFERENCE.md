# BIZRA CLI Reference

Complete reference for Claude Code command-line interface usage with BIZRA dual-agentic system.

## Table of Contents

1. [Quick Start](#quick-start)
2. [CLI Commands](#cli-commands)
3. [CLI Flags](#cli-flags)
4. [BIZRA-Specific Usage](#bizra-specific-usage)
5. [Session Management](#session-management)
6. [MCP Integration](#mcp-integration)
7. [Custom Agents](#custom-agents)
8. [Permission Modes](#permission-modes)
9. [Scripting and Automation](#scripting-and-automation)
10. [Debugging](#debugging)

---

## Quick Start

### Start Interactive Session

```bash
# Basic REPL
claude

# With initial prompt
claude "Explain the BIZRA architecture"

# Continue last session
claude -c

# Resume specific session
claude -r "ihsan-refactor"
```

### Non-Interactive (Print Mode)

```bash
# Query and exit
claude -p "List all PAT agents"

# Process piped content
cat src/ihsan.rs | claude -p "Explain this Ihsān implementation"

# JSON output for scripting
claude -p --output-format json "Show Ihsān weights"
```

---

## CLI Commands

| Command | Description | BIZRA Example |
|---------|-------------|---------------|
| `claude` | Start interactive REPL | `claude` |
| `claude "query"` | Start REPL with prompt | `claude "Explain SAT consensus"` |
| `claude -p "query"` | Query via SDK, then exit | `claude -p "List SAPE probes"` |
| `cat file \| claude -p "query"` | Process piped content | `cat src/sape.rs \| claude -p "Analyze"` |
| `claude -c` | Continue most recent conversation | `claude -c` |
| `claude -c -p "query"` | Continue via SDK | `claude -c -p "Run tests"` |
| `claude -r "session" "query"` | Resume by ID/name | `claude -r "receipts-fix" "Continue"` |
| `claude update` | Update to latest version | `claude update` |
| `claude mcp` | Configure MCP servers | `claude mcp add bizra-tools` |

---

## CLI Flags

### Essential Flags

| Flag | Description | BIZRA Usage |
|------|-------------|-------------|
| `--continue`, `-c` | Load most recent conversation | Resume BIZRA work session |
| `--resume`, `-r` | Resume specific session | `claude -r "sape-optimization"` |
| `--print`, `-p` | Non-interactive mode | CI/CD automation |
| `--model` | Set model (sonnet/opus) | `claude --model opus` for complex analysis |
| `--verbose` | Detailed logging | Debug BIZRA hooks/commands |
| `--debug` | Debug with filtering | `claude --debug "hooks,mcp"` |

### Working Directory Flags

| Flag | Description | BIZRA Usage |
|------|-------------|-------------|
| `--add-dir` | Add working directories | `claude --add-dir ../ace-framework ../HyperGraphRAG` |

### Tool Control Flags

| Flag | Description | BIZRA Usage |
|------|-------------|-------------|
| `--tools` | Restrict available tools | `claude --tools "Read,Grep,Bash"` |
| `--allowedTools` | Pre-approve tools | `"Bash(cargo:*)" "Bash(pytest:*)"` |
| `--disallowedTools` | Block specific tools | Block dangerous commands |

### System Prompt Flags

| Flag | Description | BIZRA Usage |
|------|-------------|-------------|
| `--system-prompt` | Replace system prompt | Custom BIZRA-only context |
| `--append-system-prompt` | Add to system prompt | Add BIZRA-specific rules |
| `--system-prompt-file` | Load from file | `claude -p --system-prompt-file ./bizra-rules.txt` |
| `--append-system-prompt-file` | Append from file | Load extra BIZRA context |

### Output Flags

| Flag | Description | BIZRA Usage |
|------|-------------|-------------|
| `--output-format` | Output format (text/json/stream-json) | `json` for CI parsing |
| `--json-schema` | Structured JSON output | Receipt validation |
| `--include-partial-messages` | Include streaming events | Real-time monitoring |

### Session Flags

| Flag | Description | BIZRA Usage |
|------|-------------|-------------|
| `--session-id` | Use specific session UUID | Reproducible sessions |
| `--fork-session` | Create new session from existing | Branch work |
| `--no-session-persistence` | Don't save session | Ephemeral CI runs |

### Permission Flags

| Flag | Description | BIZRA Usage |
|------|-------------|-------------|
| `--permission-mode` | Set permission mode (plan/default) | Plan before implementation |
| `--dangerously-skip-permissions` | Skip all prompts | Automated pipelines only |
| `--permission-prompt-tool` | MCP tool for permissions | Custom auth handling |

### Plugin Flags

| Flag | Description | BIZRA Usage |
|------|-------------|-------------|
| `--plugin-dir` | Load plugins from directories | `claude --plugin-dir ./custom-plugins` |
| `--disable-slash-commands` | Disable skills/commands | Minimal mode |

### MCP Flags

| Flag | Description | BIZRA Usage |
|------|-------------|-------------|
| `--mcp-config` | Load MCP config | `claude --mcp-config ./mcp-bizra.json` |
| `--strict-mcp-config` | Only use specified MCP | Isolated MCP environment |

### Resource Flags

| Flag | Description | BIZRA Usage |
|------|-------------|-------------|
| `--max-turns` | Limit agentic turns | `claude -p --max-turns 10` |
| `--max-budget-usd` | API spending limit | `claude -p --max-budget-usd 5.00` |

### Advanced Flags

| Flag | Description | BIZRA Usage |
|------|-------------|-------------|
| `--agents` | Define custom subagents | BIZRA validation agents |
| `--fallback-model` | Fallback when overloaded | `claude -p --fallback-model sonnet` |
| `--chrome` | Enable Chrome integration | Web testing |
| `--remote` | Create web session | `claude --remote "Fix SAPE bug"` |
| `--teleport` | Resume web session locally | Continue cloud work |

---

## BIZRA-Specific Usage

### Development Workflow

```bash
# Start BIZRA development session
claude --append-system-prompt "Focus on BIZRA dual-agentic architecture. Enforce receipt-first development and 0.99 Ihsān threshold."

# Quick check with allowed tools
claude --allowedTools "Bash(cargo:*)" "Bash(pytest:*)" "Read" "Grep"

# Multi-directory access
claude --add-dir ../ace-framework ../HyperGraphRAG
```

### Rust Development

```bash
# Build Rust with validation
claude -p "Build Rust in release mode, run clippy, verify no errors"

# Continue Rust session
claude -c -p "Fix the clippy warnings from last build"

# Rust-focused session
claude --allowedTools "Bash(cargo:*)" "Read" "Edit" "Grep" \
       --append-system-prompt "Focus on Rust code quality. Use clippy."
```

### Python Development

```bash
# Python kernel validation
claude -p "Validate Python imports and run pytest"

# Python-focused session
claude --allowedTools "Bash(python:*)" "Bash(pytest:*)" "Read" "Edit" \
       --append-system-prompt "Focus on Python type safety. Use pyright."
```

### Docker Services

```bash
# Service management
claude -p "Start all Docker services and show status"

# Debug services
claude --verbose -p "Check why synapse service is unhealthy"
```

### Receipt Operations

```bash
# Validate receipts with structured output
claude -p --output-format json "Validate all receipts in docs/evidence/receipts/"

# Receipt validation with schema
claude -p --json-schema '{
  "type": "object",
  "properties": {
    "total": {"type": "number"},
    "valid": {"type": "number"},
    "invalid": {"type": "number"},
    "violations": {"type": "array"}
  },
  "required": ["total", "valid", "invalid"]
}' "Analyze receipt validity"
```

### Gate Validation

```bash
# Ihsān validation
claude -p "Validate Ihsān constitution - check 8 dimensions, weights=1.0, threshold=0.99"

# SAPE validation
claude -p "Run SAPE 9-probe validation and report results"

# Full gate check
claude -p "Validate all BIZRA gates: Ihsān, SAPE, evidence chains"
```

---

## Session Management

### Creating Named Sessions

```bash
# Start named session (use descriptive names)
claude --session-id "ihsan-refactor-20260120"

# Continue named session
claude -r "ihsan-refactor-20260120"
```

### Session Workflows

```bash
# Feature development workflow
claude "Let's implement warm pool optimization"
# ... work on feature ...
# Later:
claude -c  # Continue where you left off

# Branch work into new session
claude -r "main-work" --fork-session "I want to try a different approach"
```

### Session Best Practices

1. **Use descriptive names**: `sape-probe-fix`, `receipts-schema-update`
2. **Fork for experiments**: `--fork-session` preserves original
3. **Continue vs Resume**: `-c` for last session, `-r` for specific
4. **CI/CD sessions**: `--no-session-persistence` for ephemeral runs

---

## MCP Integration

### BIZRA MCP Configuration

Create `mcp-bizra.json`:

```json
{
  "mcpServers": {
    "bizra-tools": {
      "command": "python",
      "args": ["-m", "bizra_mcp_server"],
      "env": {
        "BIZRA_MODE": "production",
        "SYNAPSE_URL": "rediss://:password@synapse:6379"
      }
    },
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "."]
    }
  }
}
```

### Using MCP

```bash
# Load BIZRA MCP config
claude --mcp-config ./mcp-bizra.json

# Strict MCP (only specified servers)
claude --strict-mcp-config --mcp-config ./mcp-bizra.json

# Add MCP server permanently
claude mcp add bizra-tools -- python -m bizra_mcp_server
```

### MCP Tool Access

```bash
# Call MCP tools
claude -p "Use the bizra-tools MCP to validate receipts"

# List available MCP tools
claude mcp list
```

---

## Custom Agents

### Define BIZRA Validation Agents

```bash
# Define multiple agents via CLI
claude --agents '{
  "receipt-auditor": {
    "description": "Audit receipt evidence. Use after generating receipts.",
    "prompt": "You are a receipt validation specialist. Check schema compliance, integrity hashes, and evidence chains.",
    "tools": ["Read", "Grep", "Glob", "Bash"],
    "model": "haiku"
  },
  "ihsan-validator": {
    "description": "Validate Ihsān constitution. Use before deployment.",
    "prompt": "You are an Ihsān expert. Validate 8 dimensions, weights sum to 1.0, and threshold is 0.99.",
    "tools": ["Read", "Bash"],
    "model": "haiku"
  },
  "sape-analyzer": {
    "description": "Analyze SAPE probe results. Use after probe failures.",
    "prompt": "You are a SAPE specialist. Analyze 9 probes, identify failures, provide root cause and fixes.",
    "tools": ["Read", "Grep", "Bash"],
    "model": "sonnet"
  }
}'
```

### Agent Use Cases

| Agent | Description | Model | Tools |
|-------|-------------|-------|-------|
| `receipt-auditor` | Receipt validation | haiku (fast) | Read, Grep, Glob, Bash |
| `ihsan-validator` | Constitution validation | haiku (fast) | Read, Bash |
| `sape-analyzer` | SAPE probe analysis | sonnet (thorough) | Read, Grep, Bash |
| `evidence-tracker` | Evidence chain tracking | haiku (fast) | Read, Grep, Glob |
| `code-reviewer` | Code quality review | sonnet (thorough) | Read, Grep, Glob, Bash |

### Creating Agent Config File

Create `bizra-agents.json`:

```json
{
  "receipt-auditor": {
    "description": "Audit receipt evidence. Use after generating receipts.",
    "prompt": "You are a receipt validation specialist for BIZRA.\n\nYour responsibilities:\n1. Validate JSON schema compliance\n2. Verify SHA-256 integrity hashes\n3. Check required fields: receipt_id, timestamp, task_summary, rejection_codes, escalation_level, integrity_hash\n4. Report any violations\n\nAlways fail-closed: if validation fails, report clearly.",
    "tools": ["Read", "Grep", "Glob", "Bash"],
    "model": "haiku"
  },
  "ihsan-validator": {
    "description": "Validate Ihsān constitution. Use before deployment.",
    "prompt": "You are an Ihsān (إحسان) constitution expert.\n\nValidate:\n1. 8 dimensions present: correctness, safety, user_benefit, efficiency, auditability, anti_centralization, robustness, adl_fairness\n2. Weights sum to exactly 1.0\n3. Production threshold is 0.99\n4. Cross-reference Rust (src/ihsan.rs) and Python implementations\n\nFail-closed: any violation is critical.",
    "tools": ["Read", "Bash"],
    "model": "haiku"
  },
  "sape-analyzer": {
    "description": "Analyze SAPE probe results. Use after probe failures.",
    "prompt": "You are a SAPE (Symbolic-Abstraction Probe Elevation) specialist.\n\n9 Probes: threat_scan, compliance, bias, user_benefit, correctness, safety, groundedness, relevance, fluency\n\nFor each failure:\n1. Identify specific probe and score\n2. Analyze root cause\n3. Provide actionable fix\n4. Estimate score improvement after fix\n\nPattern elevation: track >3 occurrences for optimization.",
    "tools": ["Read", "Grep", "Bash"],
    "model": "sonnet"
  }
}
```

Usage:

```bash
# Load agents from file (use --settings or environment)
claude --agents "$(cat bizra-agents.json)"
```

---

## Permission Modes

### Plan Mode (Recommended for Complex Tasks)

```bash
# Start in plan mode
claude --permission-mode plan "Implement warm pool optimization"

# Plan mode workflow:
# 1. Claude explores codebase (no edits)
# 2. Creates implementation plan
# 3. Requests approval before implementing
# 4. Executes with your permission
```

### Default Mode

```bash
# Standard interactive mode (default)
claude --permission-mode default

# Each tool use prompts for permission
```

### Automated Mode (CI/CD Only)

```bash
# Skip all permissions (use with caution!)
claude --dangerously-skip-permissions -p "Run tests"

# Only skip with specific allowed tools
claude --permission-mode plan --allow-dangerously-skip-permissions \
       --allowedTools "Bash(cargo test:*)" "Read"
```

### Permission Best Practices

1. **Development**: Use `default` or `plan` mode
2. **CI/CD**: Use `--dangerously-skip-permissions` with `--allowedTools`
3. **Code Review**: Use `plan` mode for exploration first
4. **Production**: Never skip permissions for deployment

---

## Scripting and Automation

### CI/CD Integration

```bash
#!/bin/bash
# ci-bizra-validate.sh

set -e

# Run Ihsān validation
ihsan_result=$(claude -p --output-format json --max-turns 5 \
  "Validate Ihsān constitution and return JSON with status")

# Parse result
if echo "$ihsan_result" | jq -e '.status == "VALID"' > /dev/null; then
  echo "✅ Ihsān validation passed"
else
  echo "❌ Ihsān validation failed"
  exit 1
fi

# Run SAPE validation
sape_result=$(claude -p --output-format json --max-turns 10 \
  "Run SAPE 9-probe validation and return JSON with all probe results")

# Check all probes passed
failed_probes=$(echo "$sape_result" | jq '.probes | map(select(.passed == false)) | length')
if [ "$failed_probes" -gt 0 ]; then
  echo "❌ SAPE validation failed: $failed_probes probes failed"
  exit 1
fi

echo "✅ All BIZRA gates passed"
```

### Batch Processing

```bash
#!/bin/bash
# batch-receipt-validation.sh

# Process all receipt files
find docs/evidence/receipts -name "*.json" | while read receipt; do
  echo "Validating: $receipt"
  cat "$receipt" | claude -p --output-format json \
    "Validate this receipt JSON. Return {valid: boolean, errors: string[]}"
done
```

### Structured Output

```bash
# Get structured analysis
claude -p --json-schema '{
  "type": "object",
  "properties": {
    "rust_files": {"type": "number"},
    "python_files": {"type": "number"},
    "test_coverage": {"type": "string"},
    "ihsan_status": {"type": "string"},
    "recommendations": {
      "type": "array",
      "items": {"type": "string"}
    }
  },
  "required": ["rust_files", "python_files", "ihsan_status"]
}' "Analyze BIZRA codebase health"
```

### Piping Content

```bash
# Analyze specific file
cat src/ihsan.rs | claude -p "Explain this Ihsān implementation"

# Analyze git diff
git diff | claude -p "Review these changes for BIZRA compliance"

# Analyze logs
docker compose logs elite --tail=100 | claude -p "Identify any errors or issues"

# Process test output
cargo test 2>&1 | claude -p "Summarize test results and failures"
```

### Environment Variables

```bash
# Set BIZRA environment
export BIZRA_MODE=development
export IHSAN_THRESHOLD=0.99
export SYNAPSE_URL=rediss://:password@synapse:6379

# Claude inherits environment
claude -p "Check current BIZRA configuration"
```

---

## Debugging

### Verbose Mode

```bash
# Enable verbose logging
claude --verbose

# Shows:
# - Full turn-by-turn output
# - Tool calls and results
# - Token usage
```

### Debug Mode

```bash
# Full debug
claude --debug

# Category filtering
claude --debug "api,hooks"       # Only API and hooks
claude --debug "mcp,tools"       # Only MCP and tools
claude --debug "!statsig,!file"  # Exclude statsig and file
```

### Debug BIZRA Hooks

```bash
# Debug hooks specifically
claude --debug "hooks"

# Test hook manually
cat test-input.json | python3 .claude/hooks/validate-bash.py

# Check hook configuration
cat .claude/settings.json | jq '.hooks'
```

### Debug MCP

```bash
# Debug MCP connections
claude --debug "mcp"

# List MCP servers
claude mcp list

# Test MCP server directly
claude mcp test bizra-tools
```

### Debug Sessions

```bash
# View session history
ls -la ~/.claude/sessions/

# Resume with verbose
claude -r "session-name" --verbose

# Check session file
cat ~/.claude/sessions/your-session-id.json | jq '.'
```

### Common Issues

**Hook Not Running**:
```bash
# Check permissions
chmod +x .claude/hooks/*.py .claude/hooks/*.sh

# Check line endings
file .claude/hooks/validate-bash.py
# Should be: Python script, ASCII text executable

# Fix line endings
sed -i 's/\r$//' .claude/hooks/*.py
```

**MCP Connection Failed**:
```bash
# Check server is running
pgrep -f bizra_mcp_server

# Test manually
python -m bizra_mcp_server --test

# Check MCP config
cat ~/.config/claude/mcp.json
```

**Session Not Found**:
```bash
# List available sessions
ls ~/.claude/sessions/ | grep -i "session-name"

# Session may have expired or been cleaned up
claude  # Start fresh session
```

---

## Examples

### Complete Development Session

```bash
# 1. Start with BIZRA context
claude --append-system-prompt "BIZRA development session. Enforce receipts, Ihsān 0.99, fail-closed."

# 2. In session, use slash commands
/rust release    # Build
/ihsan           # Validate
/sape full       # Run probes
/receipts stats  # Check evidence

# 3. Continue later
claude -c
```

### CI/CD Pipeline

```bash
#!/bin/bash
# .github/scripts/bizra-validate.sh

set -e

# Gate 1: Build
claude -p --max-turns 5 --dangerously-skip-permissions \
  --allowedTools "Bash(cargo:*)" \
  "Build Rust in release mode. Exit 1 on failure."

# Gate 2: Test
claude -p --max-turns 10 --dangerously-skip-permissions \
  --allowedTools "Bash(cargo test:*)" "Bash(pytest:*)" \
  "Run all tests. Exit 1 on any failure."

# Gate 3: Ihsān
claude -p --max-turns 5 --output-format json \
  "Validate Ihsān constitution"

# Gate 4: Receipts
claude -p --max-turns 5 --output-format json \
  "Validate all receipts in docs/evidence/receipts/"

echo "All gates passed!"
```

### Code Review Session

```bash
# Start review in plan mode
claude --permission-mode plan --agents '{
  "reviewer": {
    "description": "Expert code reviewer",
    "prompt": "Review code for BIZRA compliance, security, and quality.",
    "tools": ["Read", "Grep", "Glob"],
    "model": "sonnet"
  }
}'

# In session:
# "Review the changes in src/sape.rs for BIZRA compliance"
```

### Quick Validation Scripts

```bash
# Quick Ihsān check
alias bizra-ihsan='claude -p --max-turns 3 "Validate Ihsān constitution. Report: dimensions, weights, threshold."'

# Quick receipt count
alias bizra-receipts='claude -p --max-turns 2 "Count receipts in docs/evidence/receipts/"'

# Quick service status
alias bizra-status='claude -p --max-turns 2 "Show Docker service status"'
```

---

## See Also

- **Hooks Reference**: `.claude/hooks/HOOKS_QUICK_REFERENCE.md`
- **Commands Reference**: `.claude/commands/COMMAND_CHEAT_SHEET.md`
- **Plugin Reference**: `.claude-plugin/COMPONENT_REFERENCE.md`
- **Full System Guide**: `CLAUDE.md`
- **Official Docs**: [Claude Code Documentation](https://code.claude.com/docs)
