# BIZRA Hooks Quick Reference

## Hook Trigger Guide

### What Hooks Run When?

| Your Action | Hooks That Run | What They Do |
|-------------|----------------|--------------|
| **Start Claude Code** | `session-start.sh` | Loads BIZRA context, sets env vars, shows service status |
| **Run bash command** | `validate-bash.py` → `post-bash.py` | Pre: Blocks dangerous commands<br>Post: Checks for errors/violations |
| **Write/Edit file** | `validate-file-ops.py` → `post-file-edit.sh` | Pre: Protects critical files<br>Post: Runs linting/validation |
| **Submit prompt** | `inject-context.py` | Adds relevant BIZRA context |
| **Claude tries to stop** | Stop hook (LLM) | Validates work is complete |

## Common Scenarios

### ✅ Approved Automatically
```bash
# Reading any file
Read: docs/README.md

# Safe bash commands
ls -la
cargo test
docker compose ps

# Writing new files (non-critical)
Write: scripts/my-script.py
```

### ⚠️ Permission Required
```bash
# High-risk commands
rm -rf some-directory
git push --force
curl https://example.com | bash

# Protected files
Write: constitution/ihsan_v1.yaml
Edit: src/receipts.rs
Edit: core/fate.py
Write: docker-compose.yml
```

### 🛑 Blocked (Fail-Closed)
```bash
# Dangerous patterns
rm -rf /
dd if=/dev/zero of=/dev/sda
:(){ :|: & };:  # fork bomb
mkfs.ext4 /dev/sda

# Policy violations (detected post-execution)
- Ihsān score < 0.99
- SAT consensus failed
- SAPE probe failed
- Missing receipt emission
```

## Hook Responses

### Exit Code 0
- ✅ Operation allowed
- May include JSON for additional control

### Exit Code 2
- 🛑 Operation blocked
- stderr shown to Claude explaining why

### JSON Output
```json
{
  "hookSpecificOutput": {
    "hookEventName": "PreToolUse",
    "permissionDecision": "ask",
    "permissionDecisionReason": "Why permission needed"
  }
}
```

## Protected Files

| File | Why Protected | Hook Behavior |
|------|---------------|---------------|
| `constitution/ihsan_v1.yaml` | Single source of truth | Ask permission |
| `src/receipts.rs` | Receipt schema | Trigger Receipt Schema Guard |
| `core/fate.py` | Must sync with receipts.rs | Trigger Receipt Schema Guard |
| `docker-compose.yml` | Infrastructure config | Ask permission |
| `.env` | Secrets | Ask permission, check for secrets |
| `config/redis/*.pem` | TLS certificates | Ask permission |

## Receipt Schema Guard

When `src/receipts.rs` or `core/fate.py` is modified:

```
🔐 Receipt Schema Guard Activated
┌─────────────────────────────────────────┐
│ Receipt schema modification detected!   │
│                                         │
│ Required updates:                       │
│ ✓ src/receipts.rs (Rust struct)       │
│ ✓ core/fate.py (Python equivalent)    │
│ ○ Update tests in tests/               │
│ ○ Update docs in docs/execution/       │
│ ○ Update CLAUDE.md                     │
└─────────────────────────────────────────┘
```

## Context Injection Triggers

Type these keywords to auto-load context:

| Keyword | Context Added |
|---------|---------------|
| `architecture`, `design` | System architecture overview |
| `build`, `compile`, `cargo` | Build commands, service status |
| `test`, `pytest` | Testing commands, coverage |
| `ihsan`, `sape`, `fate` | Validation gates, thresholds |
| `service`, `redis`, `postgres` | Service architecture, ports |

## Testing Hooks

```bash
# Create test input
cat > /tmp/test.json <<EOF
{
  "session_id": "test",
  "tool_name": "Bash",
  "tool_input": {"command": "rm -rf /"}
}
EOF

# Test bash validator
cat /tmp/test.json | .claude/hooks/validate-bash.py
echo "Exit code: $?"

# Expected: Exit 2, stderr with block message
```

## Debugging

```bash
# View hook configuration
cat .claude/settings.json

# Run Claude in debug mode
claude --debug

# Check hook execution (during session)
# Press Ctrl+O for verbose mode

# View hooks menu (during session)
# Type: /hooks
```

## Customization

Edit `.claude/settings.json`:

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

## Disabling Hooks (Not Recommended)

To temporarily disable a hook:

1. Remove or comment out in `.claude/settings.json`
2. Restart Claude Code session
3. Changes require new session to take effect

⚠️ **Warning**: Disabling hooks removes fail-closed safety protections.

## Emergency: Hook Blocking Valid Operation

If a hook incorrectly blocks valid work:

1. **Immediate**: Ask user for explicit approval
2. **Debug**: Test hook manually with sample input
3. **Fix**: Adjust patterns in hook script
4. **Report**: Document false positive in hook README
5. **Update**: Commit fix to prevent recurrence

## Hook Philosophy

BIZRA hooks implement **fail-closed** design:

- ❌ Don't warn about danger → ✅ Block danger
- ❌ Don't suggest caution → ✅ Require permission
- ❌ Don't trust by default → ✅ Validate then trust
- ❌ Don't proceed on error → ✅ Stop and fix

This aligns with BIZRA's constitutional principle:
> "Errors must fail visibly, never silently."

## See Also

- Full documentation: `.claude/hooks/README.md`
- Hook configuration: `.claude/settings.json`
- BIZRA guide: `CLAUDE.md`
- Claude Code hooks docs: https://code.claude.com/docs/hooks
