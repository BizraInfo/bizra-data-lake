# Claude Code CLI - BIZRA Cheat Sheet

Quick reference for Claude Code CLI commands with BIZRA.

## Basic Commands

```bash
claude                              # Start interactive REPL
claude "query"                      # Start with prompt
claude -p "query"                   # Print mode (non-interactive)
claude -c                           # Continue last session
claude -r "name"                    # Resume named session
claude update                       # Update Claude Code
```

## Essential Flags

```bash
# Session Management
-c, --continue                      # Continue last conversation
-r, --resume "name"                 # Resume specific session
--session-id "uuid"                 # Use specific session ID
--fork-session                      # Branch from existing session

# Output Control
-p, --print                         # Non-interactive mode
--output-format [text|json|stream-json]
--json-schema '{...}'               # Structured JSON output
--verbose                           # Detailed logging

# Model Selection
--model [sonnet|opus]               # Set model
--fallback-model sonnet             # Fallback when overloaded

# Working Directory
--add-dir ../path                   # Add directories

# Tools Control
--tools "Bash,Edit,Read"            # Restrict tools
--allowedTools "Bash(cargo:*)"      # Pre-approve tools
--disallowedTools "Bash(rm:*)"      # Block tools
```

## BIZRA Development

### Quick Start

```bash
# Standard dev session
claude

# With BIZRA context
claude --append-system-prompt "BIZRA session. Enforce receipts and 0.99 Ihsān."

# Multi-repo access
claude --add-dir ../ace-framework ../HyperGraphRAG
```

### Build & Test

```bash
# Rust build
claude -p "cargo build --release"

# Rust tests
claude -p --allowedTools "Bash(cargo test:*)" "Run all Rust tests"

# Python tests
claude -p --allowedTools "Bash(pytest:*)" "Run pytest"

# Full validation
claude -p "Build, test, and validate Ihsān"
```

### Gate Validation

```bash
# Ihsān check
claude -p "Validate Ihsān constitution"

# SAPE probes
claude -p "Run SAPE 9-probe validation"

# Receipt validation
claude -p "Validate all receipts"

# Evidence chains
claude -p "Check evidence chain integrity"
```

### Docker Services

```bash
# Start services
claude -p "docker compose up -d"

# Check status
claude -p "docker compose ps"

# View logs
claude -p "docker compose logs elite --tail=50"

# Stop services
claude -p "docker compose down"
```

## Session Patterns

### Development Workflow

```bash
# Start feature work
claude --session-id "feature-warm-pools"

# ... work on feature ...

# Continue next day
claude -r "feature-warm-pools"

# Branch for experiment
claude -r "feature-warm-pools" --fork-session
```

### CI/CD Automation

```bash
# Non-interactive with limits
claude -p --max-turns 10 --max-budget-usd 1.00 "Run validation"

# Skip permissions (CI only)
claude -p --dangerously-skip-permissions "Build and test"

# JSON output for parsing
claude -p --output-format json "Analyze health"
```

## Permission Modes

```bash
# Plan mode (explore first)
claude --permission-mode plan

# Default (prompt each tool)
claude --permission-mode default

# Automated (CI/CD only!)
claude --dangerously-skip-permissions
```

## Custom Agents

```bash
# Define inline
claude --agents '{
  "reviewer": {
    "description": "Code reviewer",
    "prompt": "Review for quality",
    "tools": ["Read", "Grep"],
    "model": "haiku"
  }
}'
```

## MCP Integration

```bash
# Load MCP config
claude --mcp-config ./mcp.json

# Strict MCP only
claude --strict-mcp-config --mcp-config ./mcp.json

# Manage MCP
claude mcp add tool-name -- command args
claude mcp list
```

## Debugging

```bash
# Verbose output
claude --verbose

# Debug specific categories
claude --debug "hooks,mcp"

# Test hooks manually
cat input.json | python3 .claude/hooks/validate-bash.py
```

## Piping Content

```bash
# File analysis
cat src/ihsan.rs | claude -p "Explain"

# Git diff review
git diff | claude -p "Review changes"

# Log analysis
docker compose logs | claude -p "Find errors"

# Test output
cargo test 2>&1 | claude -p "Summarize"
```

## Structured Output

```bash
# JSON schema validation
claude -p --json-schema '{
  "type": "object",
  "properties": {
    "status": {"type": "string"},
    "issues": {"type": "array"}
  }
}' "Validate receipts"
```

## Aliases (Add to .bashrc)

```bash
# BIZRA shortcuts
alias bc='claude'
alias bcc='claude -c'
alias bcr='claude -r'
alias bcp='claude -p'

# Validation shortcuts
alias bizra-ihsan='claude -p "Validate Ihsān"'
alias bizra-sape='claude -p "Run SAPE probes"'
alias bizra-receipts='claude -p "Validate receipts"'
alias bizra-status='claude -p "Docker status"'

# Build shortcuts
alias bizra-build='claude -p "cargo build --release"'
alias bizra-test='claude -p "cargo test && pytest"'
```

## Common Patterns

### Feature Development

```bash
claude --permission-mode plan "Implement feature X"
# 1. Claude explores codebase
# 2. Creates implementation plan
# 3. You approve
# 4. Claude implements
```

### Quick Fix

```bash
claude -p "Fix the clippy warning in src/sape.rs line 42"
```

### Code Review

```bash
git diff main..feature | claude -p "Review for BIZRA compliance"
```

### Deployment Validation

```bash
claude -p --output-format json "Run all gates: build, test, Ihsān, SAPE, receipts"
```

## Environment Variables

```bash
export BIZRA_MODE=development
export IHSAN_THRESHOLD=0.99
export SYNAPSE_URL=rediss://:pass@synapse:6379
```

## Quick Reference Card

| Action | Command |
|--------|---------|
| Start session | `claude` |
| With prompt | `claude "query"` |
| Non-interactive | `claude -p "query"` |
| Continue | `claude -c` |
| Resume | `claude -r "name"` |
| JSON output | `claude -p --output-format json` |
| Plan mode | `claude --permission-mode plan` |
| Verbose | `claude --verbose` |
| Debug | `claude --debug` |
| Model | `claude --model opus` |
| Add dirs | `claude --add-dir ../path` |
| Limit turns | `claude -p --max-turns 10` |

## See Also

- Full CLI Reference: `.claude/CLI_REFERENCE.md`
- Hooks: `.claude/hooks/HOOKS_QUICK_REFERENCE.md`
- Commands: `.claude/commands/COMMAND_CHEAT_SHEET.md`
- Plugin: `.claude-plugin/README.md`
- System: `CLAUDE.md`
