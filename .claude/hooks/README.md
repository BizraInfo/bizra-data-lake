# BIZRA Claude Code Hooks

This directory contains hooks that enforce BIZRA's architectural principles and safety policies during Claude Code operations.

## Hook Architecture

BIZRA's hooks implement the **fail-closed** philosophy: operations are blocked or require approval when they might violate system invariants.

### Hooks Overview

| Hook | Event | Purpose | Exit Behavior |
|------|-------|---------|---------------|
| `session-start.sh` | SessionStart | Load BIZRA context and set environment | Adds context |
| `validate-bash.py` | PreToolUse:Bash | Block dangerous commands, warn on high-risk | Exit 2 = block, JSON = ask |
| `validate-file-ops.py` | PreToolUse:Write\|Edit | Protect critical files, validate changes | JSON = ask permission |
| `post-file-edit.sh` | PostToolUse:Write\|Edit | Run linting, validate syntax | Non-blocking warnings |
| `post-bash.py` | PostToolUse:Bash | Detect errors and BIZRA policy violations | JSON = block on violations |
| `inject-context.py` | UserPromptSubmit | Add relevant system context based on prompt | Adds context |
| Stop hook | Stop | Validate completion before stopping | Prompt-based LLM decision |

## Hook Configuration

Hooks are configured in `.claude/settings.json` and follow the pattern:

```json
{
  "hooks": {
    "EventName": [
      {
        "matcher": "ToolPattern",
        "hooks": [
          {
            "type": "command",
            "command": "\"$CLAUDE_PROJECT_DIR\"/.claude/hooks/script.py",
            "timeout": 30
          }
        ]
      }
    ]
  }
}
```

## Hook Details

### session-start.sh (SessionStart)

**Purpose**: Initialize BIZRA environment and provide system context

**Behavior**:
- Loads BIZRA architectural overview
- Sets environment variables (RUST_LOG, IHSAN_THRESHOLD, etc.)
- Reports Docker service status
- Shows recent receipt count
- Outputs context visible to Claude

**Exit codes**:
- Always exits 0 (success)
- Stdout is added to conversation context

### validate-bash.py (PreToolUse:Bash)

**Purpose**: Prevent dangerous bash commands and validate against BIZRA policies

**Blocked patterns**:
- `rm -rf /` - Root deletion
- `dd if=/dev/zero of=/dev/*` - Disk wipe
- `:(){ :|: & };:` - Fork bomb
- `mkfs.*` - Filesystem formatting

**High-risk patterns** (require permission):
- `rm -rf` - Recursive force delete
- `sudo rm` - Privileged deletion
- `curl ... | bash` - Pipe to shell
- `git push --force` - Force push

**BIZRA-critical paths** (require permission):
- `constitution/ihsan_v1.yaml` - Constitution
- `src/receipts.rs` - Receipt schema
- `core/fate.py` - FATE engine
- `docker-compose.yml` - Service config
- `.env` - Secrets
- `config/redis/*.pem` - TLS certificates

**Exit codes**:
- Exit 0: Command safe, allow
- Exit 0 + JSON: High-risk, ask permission
- Exit 2: Blocked, show stderr to Claude

### validate-file-ops.py (PreToolUse:Write|Edit)

**Purpose**: Protect critical BIZRA files and enforce schema guards

**Protected files**:
- `constitution/ihsan_v1.yaml` - Ethical weights (single source of truth)
- `src/receipts.rs` - Receipt schema (triggers schema guard)
- `core/fate.py` - FATE engine (must sync with receipts.rs)
- `docker-compose.yml` - Infrastructure config
- `model-family-genesis-v1-SEALED.yaml` - SEALED model config
- `.env` - Environment secrets

**Receipt Schema Guard**:
When modifying `src/receipts.rs` or `core/fate.py`:
1. Warns that schema is changing
2. Lists required updates:
   - `src/receipts.rs` (Rust struct)
   - `core/fate.py` (Python equivalent)
   - Tests in `tests/`
   - Evidence docs in `docs/execution/`
   - CLAUDE.md documentation
3. Asks for permission to proceed

**Critical extensions**:
- `.rs` - Rust source (suggest cargo clippy, cargo test)
- `.py` - Python source (suggest pytest)
- `.yaml`/`.yml` - Config (suggest yamllint)
- `.toml` - TOML config (validate Cargo.toml)

**Exit codes**:
- Exit 0: File safe, allow
- Exit 0 + JSON: Protected/critical, ask permission

### post-file-edit.sh (PostToolUse:Write|Edit)

**Purpose**: Validate file changes and run linting after modifications

**Per-file-type validation**:
- **Rust (.rs)**: `cargo check`, `cargo clippy`
- **Python (.py)**: `python -m py_compile`, `mypy` (for core/)
- **YAML (.yaml/.yml)**: YAML syntax validation
- **JSON (.json)**: `jq` validation
- **TOML (.toml)**: `cargo metadata` (for Cargo.toml)

**Receipt Schema Guard Display**:
When `src/receipts.rs` or `core/fate.py` is modified, shows a visual alert:
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

**Exit codes**:
- Always exits 0 (non-blocking warnings only)

### post-bash.py (PostToolUse:Bash)

**Purpose**: Analyze bash command output for errors and BIZRA policy violations

**Error patterns** (add context):
- `error:`, `fatal:` - Command errors
- `cannot find`, `No such file` - Missing resources
- `permission denied` - Permission issues
- `command not found` - Missing tools
- `Connection refused` - Service issues
- `FAIL` - Test failures

**Security patterns** (block):
- Password in plaintext
- Secret exposure
- Vulnerability found

**BIZRA validation patterns** (block):
- `Ihsan score < 0.99` - Below threshold
- `SAT consensus failed` - Validation failure
- `SAPE probe failed` - Probe failure
- `Receipt missing` - Receipt-first violation
- `FATE escalation critical` - Critical escalation

**Exit codes**:
- Exit 0: No issues
- Exit 0 + JSON (decision: block): Security/policy violation
- Exit 0 + JSON (additionalContext): Non-blocking warnings

### inject-context.py (UserPromptSubmit)

**Purpose**: Add relevant BIZRA context based on user prompt patterns

**Context triggers**:

1. **Architecture** keywords (`architecture`, `design`, `how does`, `explain`):
   - Dual implementation overview
   - PAT/SAT agent details
   - Request flow diagram

2. **Build** keywords (`build`, `compile`, `deploy`, `docker`, `cargo`):
   - Rust build commands
   - Python setup
   - Docker compose
   - Service status

3. **Testing** keywords (`test`, `pytest`, `cargo test`):
   - Test commands for Rust/Python
   - Integration testing
   - Coverage options

4. **Validation** keywords (`ihsan`, `sape`, `fate`, `validation`, `receipt`):
   - Ihsān threshold details
   - SAPE probe info
   - FATE escalation levels
   - Receipt requirements
   - Fail-closed policy

5. **Services** keywords (`service`, `redis`, `postgres`, `neo4j`, `port`):
   - Service architecture table
   - Port mappings
   - Current service status

**Exit codes**:
- Exit 0: Context added via stdout (visible to Claude)

### Stop Hook (Prompt-Based)

**Purpose**: Ensure Claude doesn't stop prematurely when BIZRA tasks are incomplete

**Decision criteria**:
- All user-requested tasks complete
- No errors requiring fixing
- Receipts emitted (check `docs/evidence/receipts/`)
- SAPE probes passed
- Ihsān score ≥ 0.99 maintained
- No fail-closed violations

**Response**:
- `{"ok": true}` - Allow stopping
- `{"ok": false, "reason": "..."}` - Continue working

**LLM model**: Claude Haiku (fast evaluation)

## Development Guidelines

### Adding New Hooks

1. **Identify the event**: Choose from PreToolUse, PostToolUse, UserPromptSubmit, Stop, etc.
2. **Create the script**: Add to `.claude/hooks/` directory
3. **Make executable**: `chmod +x .claude/hooks/your-hook.sh`
4. **Configure**: Add to `.claude/settings.json`
5. **Test**: Run manually with sample JSON input
6. **Document**: Update this README

### Hook Script Best Practices

1. **Always handle JSON input**:
   ```python
   import json, sys
   input_data = json.load(sys.stdin)
   ```

2. **Use correct exit codes**:
   - Exit 0: Success (with optional JSON output)
   - Exit 2: Block operation (stderr shown to Claude)
   - Other: Non-blocking error (stderr shown in verbose mode)

3. **Use `$CLAUDE_PROJECT_DIR`** for paths:
   ```bash
   PROJECT_DIR="${CLAUDE_PROJECT_DIR:-$(pwd)}"
   ```

4. **Provide clear feedback**:
   ```python
   # For blocking
   print("⚠️ Clear explanation of why blocked", file=sys.stderr)
   sys.exit(2)

   # For asking permission
   output = {
       "hookSpecificOutput": {
           "hookEventName": "PreToolUse",
           "permissionDecision": "ask",
           "permissionDecisionReason": "Clear explanation"
       }
   }
   print(json.dumps(output))
   sys.exit(0)
   ```

5. **Handle errors gracefully**:
   ```python
   try:
       # Hook logic
   except Exception as e:
       print(f"Hook error: {e}", file=sys.stderr)
       sys.exit(1)  # Non-blocking error
   ```

### Testing Hooks

**Manual testing**:
```bash
# Create test input
cat > /tmp/test-input.json <<EOF
{
  "session_id": "test",
  "tool_name": "Bash",
  "tool_input": {
    "command": "rm -rf /"
  }
}
EOF

# Test the hook
cat /tmp/test-input.json | .claude/hooks/validate-bash.py
echo "Exit code: $?"
```

**In Claude Code**:
1. Enable the hook in settings
2. Trigger the event (e.g., run a bash command)
3. Check verbose mode (Ctrl+O) for hook execution
4. Use `claude --debug` for detailed hook logs

## Security Considerations

### Hook Risks

**Hooks can**:
- Read any file your user can access
- Execute arbitrary commands
- Modify or delete files
- Access network resources

**Mitigation**:
- Review all hook scripts before use
- Use absolute paths with `$CLAUDE_PROJECT_DIR`
- Validate and sanitize all inputs
- Quote shell variables: `"$VAR"` not `$VAR`
- Test in safe environments first
- Never commit secrets in hooks

### Protected by Fail-Closed

BIZRA's hooks implement **fail-closed** error handling:
- Dangerous operations are blocked, not warned
- Protected files require explicit permission
- Schema changes trigger guard validation
- Policy violations stop execution

This aligns with BIZRA's constitutional principle: "Errors must fail visibly, never silently."

## Troubleshooting

### Hook not running

1. Check configuration: Run `/hooks` in Claude Code
2. Verify script is executable: `ls -la .claude/hooks/`
3. Test manually: `cat test-input.json | .claude/hooks/script.py`
4. Check syntax: Ensure JSON is valid in settings.json
5. Review logs: Use `claude --debug`

### Hook timing out

- Default timeout: 60 seconds
- Configure per-hook: `"timeout": 30`
- Optimize hook logic (avoid expensive operations)
- Use background processes for slow operations

### Hook failing

1. Check stderr output in verbose mode (Ctrl+O)
2. Verify input JSON matches expected schema
3. Test with sample input manually
4. Add debug logging to hook script
5. Ensure all dependencies are installed

### Hook blocking incorrectly

1. Review patterns in hook script
2. Test with edge cases
3. Adjust matchers in settings.json
4. Use `permissionDecision: "ask"` instead of blocking
5. Add context to help Claude understand

## Integration with BIZRA Architecture

The hooks system integrates with BIZRA's core principles:

1. **Receipt-First Development**:
   - Hooks validate receipt emission
   - Schema guards protect receipt integrity
   - Post-bash hook detects missing receipts

2. **Fail-Closed Error Handling**:
   - Dangerous commands are blocked (exit 2)
   - Critical files require permission
   - Policy violations stop execution

3. **Ihsān Gate Enforcement**:
   - Hooks validate 0.99 threshold compliance
   - Constitution modifications are protected
   - Post-bash detects threshold violations

4. **SAPE/FATE Integration**:
   - Hooks detect SAPE probe failures
   - FATE escalation violations trigger blocks
   - Evidence generation is validated

5. **Constitution as Code**:
   - constitution/ihsan_v1.yaml is protected
   - Changes trigger comprehensive validation
   - Schema stability is enforced

## Examples

See the hooks in action:

**Safe operation** (auto-approved):
```bash
# Reading documentation - no hook intervention
Read tool: docs/README.md
```

**High-risk operation** (permission requested):
```bash
# Modifying protected file
Write tool: constitution/ihsan_v1.yaml
→ Hook asks: "Protected File: Constitution - Single source of truth"
→ User approves/denies
```

**Blocked operation** (fail-closed):
```bash
# Dangerous command
Bash: rm -rf /
→ Hook blocks: "🛑 BLOCKED: Command matches dangerous pattern"
→ Execution prevented
```

**Context injection** (automatic):
```
User: "How do I build the Rust components?"
→ Hook adds: "Build Commands: cargo build --release && cargo test..."
→ Claude has relevant context
```

## See Also

- [Claude Code Hooks Documentation](https://docs.anthropic.com/claude/docs/hooks)
- BIZRA CLAUDE.md - Project development guide
- constitution/ihsan_v1.yaml - Ethical constraints
- src/receipts.rs - Receipt schema
- .github/copilot-instructions.md - AI agent instructions
