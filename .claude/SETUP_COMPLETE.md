# BIZRA Claude Code Hooks - Setup Complete ✅

## What Was Installed

A comprehensive hooks system has been configured for BIZRA to enforce architectural principles and safety policies during Claude Code development.

### Created Files

```
.claude/
├── settings.json                   # Hook configuration
├── HOOKS_QUICK_REFERENCE.md        # Quick reference guide
└── hooks/
    ├── README.md                   # Comprehensive documentation
    ├── session-start.sh            # SessionStart hook
    ├── validate-bash.py            # PreToolUse:Bash hook
    ├── validate-file-ops.py        # PreToolUse:Write|Edit hook
    ├── post-file-edit.sh           # PostToolUse:Write|Edit hook
    ├── post-bash.py                # PostToolUse:Bash hook
    └── inject-context.py           # UserPromptSubmit hook
```

### Hook Verification

All hooks have been tested and are working correctly:

✅ **Safe command (ls -la)**: Allowed (exit 0)
✅ **Dangerous command (rm -rf /)**: Blocked (exit 2)
✅ **Risky command (git push --force)**: Asks permission (JSON output)

## How It Works

### 1. Session Starts
When you start Claude Code:
- Loads BIZRA system context
- Sets environment variables (RUST_LOG, IHSAN_THRESHOLD, etc.)
- Shows Docker service status
- Reports recent receipt count

### 2. Bash Commands
Before executing bash commands:
- **Blocks**: `rm -rf /`, fork bombs, disk wipes
- **Asks permission**: `rm -rf dir`, `git push --force`, `curl | bash`
- **Allows**: Safe commands like `ls`, `cargo test`, `docker compose ps`

After execution:
- Detects errors and BIZRA policy violations
- Checks for Ihsān threshold violations (<0.99)
- Validates receipt emission

### 3. File Operations
Before writing/editing files:
- **Asks permission** for protected files:
  - `constitution/ihsan_v1.yaml`
  - `src/receipts.rs`
  - `core/fate.py`
  - `docker-compose.yml`
  - `.env`
  - `config/redis/*.pem`

After modifications:
- Runs appropriate linting (cargo check, python syntax, etc.)
- Shows Receipt Schema Guard for receipt-related changes

### 4. User Prompts
When you submit a prompt with keywords:
- **"architecture"** → Adds system architecture overview
- **"build"** → Adds build commands and service status
- **"test"** → Adds testing commands
- **"ihsan"** → Adds validation gate details
- **"service"** → Adds service architecture and ports

### 5. Stopping
Before Claude stops:
- LLM evaluates if work is complete
- Checks that receipts were emitted
- Validates SAPE probes passed
- Ensures Ihsān score maintained
- Prevents premature exit

## Quick Start

### View Hook Status
```bash
# In Claude Code session
/hooks
```

### Test Hooks Manually
```bash
# Create test input
cat > /tmp/test.json <<EOF
{
  "tool_name": "Bash",
  "tool_input": {"command": "ls -la"}
}
EOF

# Test bash validator
cat /tmp/test.json | .claude/hooks/validate-bash.py
```

### Debug Hooks
```bash
# Run Claude Code in debug mode
claude --debug

# View hook execution (during session)
# Press Ctrl+O for verbose mode
```

## Protected Files

These files trigger permission dialogs:

| File | Protection Reason |
|------|-------------------|
| `constitution/ihsan_v1.yaml` | Constitution - single source of truth |
| `src/receipts.rs` | Receipt schema - triggers schema guard |
| `core/fate.py` | FATE engine - must sync with receipts.rs |
| `docker-compose.yml` | Infrastructure configuration |
| `.env` | Environment secrets |
| `config/redis/*.pem` | TLS certificates |

## Receipt Schema Guard

When modifying `src/receipts.rs` or `core/fate.py`, hooks automatically display:

```
🔐 Receipt Schema Guard Activated
┌─────────────────────────────────────────────────────────────┐
│ Receipt schema modification detected!                       │
│                                                             │
│ Required updates:                                           │
│ ✓ src/receipts.rs (Rust struct)                           │
│ ✓ core/fate.py (Python equivalent)                        │
│ ○ Update tests in tests/                                   │
│ ○ Update evidence docs in docs/execution/                  │
│ ○ Update CLAUDE.md documentation                           │
│                                                             │
│ Verify backward compatibility for existing receipts!        │
└─────────────────────────────────────────────────────────────┘
```

This ensures schema changes are coordinated across the entire codebase.

## Architecture Integration

The hooks system enforces BIZRA's core principles:

### 1. Receipt-First Development
- Validates receipt emission after operations
- Protects receipt schema integrity
- Detects missing receipts in bash output

### 2. Fail-Closed Error Handling
- Dangerous commands are blocked, not warned
- Critical files require permission
- Policy violations stop execution

### 3. Ihsān Gate Enforcement
- Detects threshold violations (<0.99)
- Protects constitution modifications
- Validates ethical constraints

### 4. SAPE/FATE Integration
- Detects SAPE probe failures
- Catches FATE escalation violations
- Ensures evidence generation

### 5. Constitution as Code
- `constitution/ihsan_v1.yaml` is protected
- Changes trigger comprehensive validation
- Schema stability enforced

## Examples

### Automatic Approval
```bash
# Safe operations
Read: docs/README.md          → Allowed
Bash: cargo test              → Allowed
Write: scripts/helper.py      → Allowed
```

### Permission Required
```bash
# High-risk operations
Bash: rm -rf build/           → ⚠️ Permission dialog
Write: constitution/ihsan_v1.yaml → ⚠️ Protected file
Edit: src/receipts.rs         → 🔐 Receipt Schema Guard
Bash: git push --force        → ⚠️ Force push warning
```

### Blocked Operations
```bash
# Fail-closed violations
Bash: rm -rf /                → 🛑 BLOCKED
Bash: :(){ :|: & };:          → 🛑 BLOCKED (fork bomb)
Bash: dd if=/dev/zero of=/dev/sda → 🛑 BLOCKED
```

## Customization

To modify hook behavior, edit `.claude/settings.json`:

```json
{
  "hooks": {
    "PreToolUse": [
      {
        "matcher": "Bash",
        "hooks": [
          {
            "type": "command",
            "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/validate-bash.py",
            "timeout": 10
          }
        ]
      }
    ]
  }
}
```

⚠️ Changes require restarting Claude Code to take effect.

## Documentation

- **Quick Reference**: `.claude/HOOKS_QUICK_REFERENCE.md`
- **Full Documentation**: `.claude/hooks/README.md`
- **BIZRA Guide**: `CLAUDE.md` (includes hooks section)
- **Hook Scripts**: `.claude/hooks/`

## Troubleshooting

### Hook Not Running
1. Check `/hooks` menu in Claude Code
2. Verify scripts are executable: `ls -la .claude/hooks/`
3. Restart Claude Code session
4. Run `claude --debug` for detailed logs

### Hook Blocking Valid Operation
1. Ask user for explicit approval
2. Test hook manually with sample input
3. Report issue to development team
4. Adjust patterns if false positive

### Hook Timing Out
- Default timeout: 60 seconds (configurable)
- Increase timeout in settings.json if needed
- Optimize hook logic for performance

## Security Notes

**Hooks execute arbitrary shell commands on your system.**

Safety measures in place:
- Hooks review and test before deployment
- Fail-closed design (block by default)
- Protected files require permission
- Dangerous patterns explicitly blocked
- All hooks use absolute paths

**DO NOT**:
- Add untrusted hooks without review
- Disable fail-closed protections
- Skip permission dialogs without understanding impact
- Modify hooks without testing

## Next Steps

The hooks system is now active and will automatically enforce BIZRA policies:

1. **Start Claude Code** - Hooks load automatically
2. **Work normally** - Hooks run transparently
3. **Review permissions** - Approve/deny when asked
4. **Check verbose mode** - Press Ctrl+O to see hook execution
5. **Debug if needed** - Use `claude --debug`

## Support

For questions or issues:

1. Read `.claude/hooks/README.md` (comprehensive guide)
2. Check `.claude/HOOKS_QUICK_REFERENCE.md` (common scenarios)
3. Review CLAUDE.md hooks section
4. Test hooks manually (see Testing section above)
5. Enable debug mode: `claude --debug`

---

**Setup completed**: 2026-01-20

**Status**: ✅ All hooks active and tested

**Philosophy**: Fail-closed, receipt-first, evidence-driven development

الحمد لله - All praise belongs to Allah
