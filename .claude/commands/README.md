## BIZRA Slash Commands

This directory contains custom slash commands for BIZRA development workflows. Commands are organized by category and integrate with BIZRA's architectural principles.

## Directory Structure

```
.claude/commands/
├── README.md                    # This file
├── guide.md                     # Comprehensive command guide
├── commit.md                    # Git commit with evidence
├── build/
│   ├── rust.md                  # Build Rust Elite engine
│   └── python.md                # Setup Python kernel
├── service/
│   └── docker.md                # Manage Docker services
├── test/
│   ├── rust.md                  # Run Rust test suite
│   └── python.md                # Run Python tests with coverage
├── validation/
│   ├── ihsan.md                 # Validate Ihsān constitution
│   └── sape.md                  # Validate SAPE probes
└── evidence/
    └── receipts.md              # Manage receipt evidence
```

## Available Commands

### Build Commands

| Command | Description | Arguments | Evidence |
|---------|-------------|-----------|----------|
| `/rust` | Build Rust Elite engine | [release\|debug] | Build receipt |
| `/python` | Setup Python kernel | None | - |

### Service Commands

| Command | Description | Arguments | Evidence |
|---------|-------------|-----------|----------|
| `/docker` | Manage Docker services | [up\|down\|restart\|status\|logs] [service] | Service receipt |

### Test Commands

| Command | Description | Arguments | Evidence |
|---------|-------------|-----------|----------|
| `/rust` (test) | Run Rust test suite | [test-name] [--nocapture] | Test receipt |
| `/python` (test) | Run Python tests | [test-path] [-v\|-vv] | Coverage report |

### Validation Commands

| Command | Description | Arguments | Evidence |
|---------|-------------|-----------|----------|
| `/ihsan` | Validate Ihsān constitution | None | Validation receipt |
| `/sape` | Validate SAPE probes | None | Validation receipt |

### Evidence Commands

| Command | Description | Arguments | Evidence |
|---------|-------------|-----------|----------|
| `/receipts` | Manage receipts | [count\|validate\|recent\|stats] | Meta-receipt |

### General Commands

| Command | Description | Arguments | Evidence |
|---------|-------------|-----------|----------|
| `/commit` | Create git commit | [message] | Commit receipt |
| `/guide` | Show command guide | None | - |

## Command Features

### Frontmatter Support

Commands support YAML frontmatter for metadata:

```markdown
---
allowed-tools: Bash(cargo:*)
description: Build Rust components
argument-hint: [release|debug]
hooks:
  PostToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "$CLAUDE_PROJECT_DIR/.claude/hooks/post-bash.py"
---
```

**Supported frontmatter**:
- `allowed-tools`: Tools the command can use
- `description`: Brief command description
- `argument-hint`: Expected arguments
- `model`: Specific model to use
- `context`: `fork` for sub-agent context
- `agent`: Agent type (when `context: fork`)
- `hooks`: Command-scoped hooks

### Argument Passing

**All arguments**: `$ARGUMENTS`
```bash
/commit "feat: add feature"
# $ARGUMENTS = "feat: add feature"
```

**Positional arguments**: `$1`, `$2`, etc.
```bash
/docker logs elite
# $1 = "logs", $2 = "elite"
```

### Bash Execution

Execute bash commands with `!` prefix:

```markdown
Current branch: !`git branch --show-current`
Status: !`git status --short`
```

Output is included in command context.

### File References

Reference files with `@` prefix:

```markdown
Review @src/receipts.rs
Compare @src/sape.rs with @core/sape.py
```

### Hook Integration

Commands can define hooks that run during execution:

```yaml
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "./scripts/validate.sh"
          once: true
```

Hooks are scoped to the command's execution and automatically cleaned up.

## Creating Custom Commands

### Basic Command

1. Create file in appropriate directory:
```bash
touch .claude/commands/category/mycommand.md
```

2. Add frontmatter and content:
```markdown
---
description: Brief description
---

# My Command

Your command instructions here...
```

3. Use it:
```bash
/mycommand
```

### Command with Arguments

```markdown
---
argument-hint: [arg1] [arg2]
---

# My Command

Argument 1: $1
Argument 2: $2
All args: $ARGUMENTS
```

### Command with Bash Execution

```markdown
---
allowed-tools: Bash(git:*), Bash(cargo:*)
---

# My Command

Current branch: !`git branch --show-current`

Execute:
bash
cargo test
```

### Command with Hooks

```markdown
---
hooks:
  PostToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: ".claude/hooks/post-bash.py"
---

# My Command

[Instructions...]
```

## Command Development Guidelines

### 1. Follow BIZRA Principles

**Receipt-First**:
- Generate receipts for all operations
- Include integrity hashes (SHA-256)
- Store in `docs/evidence/receipts/`

**Fail-Closed**:
- Block on critical failures
- Use exit code 2 for blocking errors
- Validate before proceeding

**Evidence-Driven**:
- Produce auditable output
- Track success/failure rates
- Maintain append-only evidence

### 2. Structure and Organization

**Naming**:
- Use kebab-case: `my-command.md`
- Be descriptive: `build-rust.md` not `rust.md`
- Group by category in subdirectories

**Content**:
- Start with overview and current status
- Show `!` bash output for context
- Provide clear step-by-step tasks
- Include validation checklists
- Generate evidence/receipts
- Provide formatted report output

### 3. Error Handling

**Exit Codes**:
- `0` - Success
- `2` - Blocking failure (fail-closed)
- `1` - Non-blocking warning

**Error Messages**:
```bash
echo "❌ FAIL-CLOSED: Descriptive error message" >&2
exit 2
```

**Warnings**:
```bash
echo "⚠️ WARNING: Issue description"
# Continue execution
```

### 4. Evidence Generation

Every command should generate receipts:

```bash
cat > "docs/evidence/receipts/command-$(date +%Y%m%d-%H%M%S).json" <<EOF
{
  "receipt_id": "command-$(date +%s)",
  "timestamp": "$(date -Iseconds)",
  "command": "command-name",
  "status": "pass|fail",
  "summary": "What was done",
  "integrity_hash": "$(echo -n 'data' | sha256sum | cut -d' ' -f1)"
}
EOF
```

### 5. Testing Commands

**Test manually**:
```bash
# Source the command (if applicable)
bash .claude/commands/category/command.md

# Or invoke via Claude Code
/command
```

**Test with sample input**:
```bash
cat > /tmp/test-input.json <<EOF
{
  "tool_name": "Bash",
  "tool_input": {"command": "ls"}
}
EOF

cat /tmp/test-input.json | .claude/hooks/post-bash.py
```

## Command Integration with BIZRA

### Receipt Schema Compliance

Commands that modify receipts must follow the schema in `src/receipts.rs`:

```rust
pub struct Receipt {
    receipt_id: String,
    timestamp: String,
    task_summary: String,
    rejection_codes: Vec<String>,
    escalation_level: EscalationLevel,
    integrity_hash: String,
}
```

### Ihsān Integration

Commands should respect Ihsān thresholds:

```bash
# Check current threshold
echo "IHSAN_THRESHOLD=${IHSAN_THRESHOLD:-0.99}"

# Validate against threshold
python3 << 'EOF'
import os
threshold = float(os.getenv('IHSAN_THRESHOLD', '0.99'))
score = 0.97  # calculated score

if score < threshold:
    print(f'❌ FAIL-CLOSED: Ihsān score {score} < {threshold}')
    exit(2)
EOF
```

### SAPE Integration

Commands can trigger SAPE probes:

```bash
# Reference SAPE validation
echo "SAPE probes must pass before proceeding"
echo "Run /sape to validate probe system"
```

### FATE Integration

Commands handling errors should consider FATE escalation:

```bash
# Determine escalation level
if [ critical_error ]; then
    ESCALATION_LEVEL="Critical"
elif [ high_error ]; then
    ESCALATION_LEVEL="High"
else
    ESCALATION_LEVEL="Low"
fi

echo "FATE escalation: $ESCALATION_LEVEL"
```

## Common Patterns

### Pre-flight Checks

```markdown
## Current Status

- Git branch: !`git branch --show-current`
- Uncommitted changes: !`git status --short | wc -l`
- Last commit: !`git log -1 --oneline`
```

### Validation Checklist

```markdown
## Critical Checks (MUST PASS)

- [ ] Syntax is valid
- [ ] Tests pass
- [ ] No critical errors
- [ ] Receipt generated
```

### Conditional Logic

```bash
if [ "$1" = "debug" ]; then
    BUILD_FLAGS=""
else
    BUILD_FLAGS="--release"
fi

cargo build $BUILD_FLAGS
```

### Multiple Steps

```markdown
### 1. Pre-validation
[Steps...]

### 2. Main Execution
[Steps...]

### 3. Post-validation
[Steps...]

### 4. Evidence Generation
[Steps...]
```

### Report Format

```markdown
## Command Report

**Status**: ✅ PASS | ❌ FAIL
**Metric**: value
**Duration**: time

### Summary
[Details...]

### Evidence
- Receipt: path/to/receipt.json
```

## Troubleshooting

### Command Not Found

```bash
# List all commands
ls -la .claude/commands/**/*.md

# Check command is in settings
grep -r "command-name" .claude/settings.json
```

### Command Fails

```bash
# Check command syntax
cat .claude/commands/category/command.md

# Test bash sections manually
bash << 'EOF'
[bash code from command]
EOF
```

### Argument Issues

```bash
# Test argument passing
echo "Args: $1, $2, $ARGUMENTS"
```

### Hook Issues

```bash
# Test hooks
.claude/hooks/validate-bash.py < test-input.json

# Check hook configuration
cat .claude/settings.json | jq '.hooks'
```

## Best Practices

### DO

✅ Generate receipts for all operations
✅ Use fail-closed error handling
✅ Provide clear error messages
✅ Include validation checklists
✅ Show current status with `!` bash
✅ Format reports consistently
✅ Test commands thoroughly
✅ Document arguments clearly
✅ Follow BIZRA principles

### DON'T

❌ Silently fail on errors
❌ Skip evidence generation
❌ Modify receipts without schema guard
❌ Use destructive operations without warnings
❌ Ignore Ihsān thresholds
❌ Skip validation steps
❌ Omit frontmatter metadata
❌ Use unclear argument names

## Examples

### Minimal Command

```markdown
---
description: Echo a message
---

# Echo Command

Message: $ARGUMENTS

bash
echo "$ARGUMENTS"
```

### Full-Featured Command

```markdown
---
allowed-tools: Bash(cargo:*), Bash(git:*)
description: Build and validate Rust component
argument-hint: [component]
hooks:
  PostToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "$CLAUDE_PROJECT_DIR/.claude/hooks/post-bash.py"
---

# Build Component

## Current Status

- Component: $1
- Branch: !`git branch --show-current`
- Last build: !`ls -t target/release/$1 | head -1`

## Your Task

### 1. Pre-build Validation
[Steps...]

### 2. Build
bash
cargo build --release -p $1
```

### 3. Post-build Validation
[Steps...]

### 4. Evidence Generation
[Create receipt...]

## Report
[Formatted output...]
```

## See Also

- **guide.md** - Comprehensive command guide
- **CLAUDE.md** - Full BIZRA development guide
- **.claude/hooks/** - Hook system documentation
- **docs/evidence/** - Receipt evidence
- **constitution/** - Ihsān and governance

---

**Command Philosophy**: "Receipt-first. Fail-closed. Evidence-driven. Every command leaves a trail."

For command usage, type `/guide` in Claude Code.
