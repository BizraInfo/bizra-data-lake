# ✅ Claude Code CLI Reference - Setup Complete

## Installation Summary

The Claude Code CLI reference has been configured with comprehensive documentation and automation scripts for BIZRA development.

## 📦 Installed Components

### Documentation

- ✅ `.claude/CLI_REFERENCE.md` - Complete CLI reference guide
- ✅ `.claude/CLI_CHEAT_SHEET.md` - Quick reference cheat sheet
- ✅ `.claude/CLI_SETUP_COMPLETE.md` - This file

### Automation Scripts

- ✅ `.claude/scripts/bizra-dev.sh` - Development session helper
- ✅ `.claude/scripts/bizra-validate.sh` - Full validation pipeline
- ✅ `.claude/scripts/ci-claude-gate.sh` - CI/CD gate runner
- ✅ `.claude/scripts/generate-receipt.sh` - Receipt generator
- ✅ `.claude/scripts/README.md` - Scripts documentation

### CLAUDE.md Integration

- ✅ CLI Reference section added to `CLAUDE.md`
- ✅ Plugin System section added to `CLAUDE.md`

## 🚀 Quick Start

### Start Development Session

```bash
# Standard session
claude

# With BIZRA context
claude --append-system-prompt "BIZRA session. Enforce receipts and 0.99 Ihsān."

# Using helper script
.claude/scripts/bizra-dev.sh start
```

### Run Validation

```bash
# Quick validation
.claude/scripts/bizra-validate.sh

# Strict mode
.claude/scripts/bizra-validate.sh --strict

# Using Claude directly
claude -p "Validate Ihsān constitution"
```

### Session Management

```bash
# Continue last session
claude -c

# Resume named session
claude -r "feature-name"

# Project status
.claude/scripts/bizra-dev.sh status
```

## 📋 CLI Commands Reference

| Command | Description |
|---------|-------------|
| `claude` | Start interactive REPL |
| `claude "query"` | Start with prompt |
| `claude -p "query"` | Non-interactive mode |
| `claude -c` | Continue last session |
| `claude -r "name"` | Resume named session |
| `claude update` | Update Claude Code |
| `claude mcp` | Manage MCP servers |

## 🎯 Essential Flags

| Flag | Description |
|------|-------------|
| `-c`, `--continue` | Continue last conversation |
| `-r`, `--resume` | Resume specific session |
| `-p`, `--print` | Non-interactive mode |
| `--model` | Set model (sonnet/opus) |
| `--verbose` | Detailed logging |
| `--debug` | Debug with filtering |
| `--output-format` | Output format (text/json/stream-json) |
| `--permission-mode` | Permission mode (plan/default) |
| `--allowedTools` | Pre-approve tools |
| `--add-dir` | Add working directories |

## 🛠️ Automation Scripts

### bizra-dev.sh

Development session helper:

```bash
.claude/scripts/bizra-dev.sh start     # New session
.claude/scripts/bizra-dev.sh continue  # Continue
.claude/scripts/bizra-dev.sh resume -n "name"  # Resume
.claude/scripts/bizra-dev.sh quick "query"     # Quick query
.claude/scripts/bizra-dev.sh validate  # Run validation
.claude/scripts/bizra-dev.sh status    # Project status
```

### bizra-validate.sh

Full validation pipeline:

```bash
.claude/scripts/bizra-validate.sh          # All gates
.claude/scripts/bizra-validate.sh --strict # Exit on first failure
.claude/scripts/bizra-validate.sh --json   # JSON output
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

CI/CD integration:

```bash
.claude/scripts/ci-claude-gate.sh build    # Build gate
.claude/scripts/ci-claude-gate.sh test     # Test gate
.claude/scripts/ci-claude-gate.sh ihsan    # Ihsān gate
.claude/scripts/ci-claude-gate.sh sape     # SAPE gate
.claude/scripts/ci-claude-gate.sh receipts # Receipts gate
.claude/scripts/ci-claude-gate.sh full     # All gates
```

### generate-receipt.sh

Receipt generator:

```bash
.claude/scripts/generate-receipt.sh build "Build completed"
.claude/scripts/generate-receipt.sh test "Tests passed"
.claude/scripts/generate-receipt.sh validation "Gate passed"
.claude/scripts/generate-receipt.sh commit "Commit created"
.claude/scripts/generate-receipt.sh deploy "Deployed"
```

## 🔧 Usage Examples

### Development Workflow

```bash
# Morning: Start session
.claude/scripts/bizra-dev.sh start -p  # Plan mode

# Work on feature
# ...

# Before commit: Validate
.claude/scripts/bizra-validate.sh --strict

# End of day: Session auto-saved
```

### CI/CD Pipeline

```yaml
# .github/workflows/validate.yml
jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - name: Run BIZRA validation
        run: .claude/scripts/bizra-validate.sh --json > results.json
```

### Quick Checks

```bash
# Check status
.claude/scripts/bizra-dev.sh status

# Quick Ihsān validation
.claude/scripts/bizra-dev.sh quick "Validate Ihsān constitution"

# Count receipts
.claude/scripts/bizra-dev.sh quick "Count receipts"
```

## 📊 Permission Modes

| Mode | Use Case | Command |
|------|----------|---------|
| **Plan** | Complex tasks, exploration first | `--permission-mode plan` |
| **Default** | Normal development | (default) |
| **Skip** | CI/CD only | `--dangerously-skip-permissions` |

### Plan Mode Workflow

```bash
claude --permission-mode plan "Implement feature X"
```

1. Claude explores codebase (no edits)
2. Creates implementation plan
3. Requests your approval
4. Executes with permission

## 🎉 Testing the Setup

### Test CLI

```bash
# Basic test
claude -p "What is BIZRA?"

# Session test
claude "Start a test session"
# Then: claude -c

# Verbose test
claude --verbose "Hello"
```

### Test Scripts

```bash
# Dev helper
.claude/scripts/bizra-dev.sh status

# Validation
.claude/scripts/bizra-validate.sh

# Receipt generation
.claude/scripts/generate-receipt.sh test "CLI setup test"
```

## 📚 Documentation Index

| Document | Purpose |
|----------|---------|
| `.claude/CLI_REFERENCE.md` | Complete CLI documentation |
| `.claude/CLI_CHEAT_SHEET.md` | Quick reference card |
| `.claude/scripts/README.md` | Scripts documentation |
| `CLAUDE.md` (CLI section) | Overview in main guide |

## ✅ What's Now Available

1. ✅ Full CLI flag reference with BIZRA examples
2. ✅ Permission mode documentation
3. ✅ Custom agents configuration
4. ✅ Session management patterns
5. ✅ Piping and output formatting
6. ✅ MCP integration guide
7. ✅ Development session helper script
8. ✅ Full validation pipeline script
9. ✅ CI/CD gate runner script
10. ✅ Receipt generator script
11. ✅ Debugging guide
12. ✅ Bash aliases for common tasks

## 🔄 Integration Points

The CLI reference integrates with:

- **Hooks**: `.claude/hooks/` - Pre/post validation
- **Commands**: `.claude/commands/` - Slash commands
- **Plugin**: `.claude-plugin/` - Agents and skills
- **CLAUDE.md**: Main documentation

## 📖 See Also

- **Hooks**: `.claude/hooks/HOOKS_QUICK_REFERENCE.md`
- **Commands**: `.claude/commands/COMMAND_CHEAT_SHEET.md`
- **Plugin**: `.claude-plugin/README.md`
- **System Guide**: `CLAUDE.md`
- **Official Docs**: [code.claude.com/docs](https://code.claude.com/docs)

---

**🎊 CLI Reference setup complete!**

Use `claude --help` for built-in help, or refer to `.claude/CLI_REFERENCE.md` for BIZRA-specific usage.
