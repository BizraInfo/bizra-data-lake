# BIZRA Memory Management

Comprehensive guide for managing Claude Code's memory across sessions.

## Table of Contents

1. [Memory Hierarchy](#memory-hierarchy)
2. [Memory Locations](#memory-locations)
3. [Modular Rules](#modular-rules)
4. [CLAUDE.md Imports](#claudemd-imports)
5. [Best Practices](#best-practices)
6. [BIZRA Integration](#bizra-integration)

---

## Memory Hierarchy

Claude Code uses a hierarchical memory system. Higher levels take precedence:

```
                    ┌─────────────────────────────┐
                    │    Enterprise Policy        │ ← Highest priority
                    │  (System-wide, IT managed)  │
                    └─────────────────────────────┘
                                  ▼
                    ┌─────────────────────────────┐
                    │     Project Memory          │
                    │   (CLAUDE.md, shared)       │
                    └─────────────────────────────┘
                                  ▼
                    ┌─────────────────────────────┐
                    │     Project Rules           │
                    │   (.claude/rules/*.md)      │
                    └─────────────────────────────┘
                                  ▼
                    ┌─────────────────────────────┐
                    │      User Memory            │
                    │   (~/.claude/CLAUDE.md)     │
                    └─────────────────────────────┘
                                  ▼
                    ┌─────────────────────────────┐
                    │   Project Memory (Local)    │ ← Lowest priority
                    │     (CLAUDE.local.md)       │
                    └─────────────────────────────┘
```

---

## Memory Locations

### Enterprise Policy (IT-Managed)

Organization-wide instructions managed by IT/DevOps.

| OS | Location |
|----|----------|
| macOS | `/Library/Application Support/ClaudeCode/CLAUDE.md` |
| Linux | `/etc/claude-code/CLAUDE.md` |
| Windows | `C:\Program Files\ClaudeCode\CLAUDE.md` |

**Use for**:
- Company coding standards
- Security policies
- Compliance requirements

### Project Memory (Shared)

Team-shared instructions in the repository.

| Location | Purpose |
|----------|---------|
| `./CLAUDE.md` | Primary project instructions |
| `./.claude/CLAUDE.md` | Alternative location |

**Use for**:
- Project architecture
- Coding standards
- Common workflows
- Build/test commands

### Project Rules (Modular)

Topic-specific instructions in `.claude/rules/`.

```
.claude/rules/
├── general.md           # Universal rules
├── rust/
│   ├── code-style.md   # Rust formatting
│   ├── safety.md       # Rust safety
│   └── testing.md      # Rust tests
├── python/
│   └── code-style.md   # Python formatting
├── validation/
│   ├── ihsan.md        # Ihsān rules
│   └── sape.md         # SAPE rules
└── evidence/
    └── receipts.md     # Receipt rules
```

**Use for**:
- Language-specific guidelines
- Domain-specific rules
- Conditional rules (via `paths` frontmatter)

### User Memory (Personal, Global)

Personal preferences across all projects.

| Location | Purpose |
|----------|---------|
| `~/.claude/CLAUDE.md` | Your global preferences |
| `~/.claude/rules/*.md` | Your personal rules |

**Use for**:
- Personal coding style
- Tooling preferences
- Custom workflows

### Project Memory (Local, Personal)

Personal project-specific preferences.

| Location | Purpose |
|----------|---------|
| `./CLAUDE.local.md` | Your local project settings |

**Use for**:
- Local sandbox URLs
- Personal test data
- Machine-specific settings

**Note**: CLAUDE.local.md is automatically gitignored.

---

## Modular Rules

### Basic Structure

Create `.md` files in `.claude/rules/`:

```markdown
# Rule Title

Your instructions here.
```

### Path-Specific Rules

Add YAML frontmatter to scope rules to specific files:

```markdown
---
paths:
  - "src/**/*.rs"
  - "crates/**/*.rs"
---

# Rust-Specific Rules

These rules only apply when working with Rust files.
```

### Supported Glob Patterns

| Pattern | Matches |
|---------|---------|
| `**/*.rs` | All Rust files in any directory |
| `src/**/*` | All files under src/ |
| `*.md` | Markdown files in root |
| `{src,lib}/**/*.ts` | TypeScript in src/ or lib/ |
| `**/*.{py,pyi}` | Python files and stubs |

### Subdirectories

Organize rules into subdirectories:

```
.claude/rules/
├── frontend/
│   ├── react.md
│   └── styles.md
├── backend/
│   ├── api.md
│   └── database.md
└── general.md
```

All `.md` files are discovered recursively.

---

## CLAUDE.md Imports

### Import Syntax

Import additional files using `@path/to/import`:

```markdown
See @README for project overview.
See @package.json for npm commands.

# Additional Instructions
- Git workflow: @docs/git-instructions.md
```

### Import Features

- **Relative paths**: `@docs/guide.md`
- **Absolute paths**: `@/etc/company-rules.md`
- **Home directory**: `@~/.claude/my-notes.md`
- **Recursive imports**: Imported files can import others (max depth: 5)

### Import Exclusions

Imports are NOT evaluated in code blocks:

```markdown
This will NOT import: `@some-package`

```code
@this-is-not-imported
```
```

### View Loaded Memories

Run `/memory` command to see all loaded memory files.

---

## Best Practices

### Writing Effective Memories

**Be Specific**:
```markdown
# Good
Use 2-space indentation for TypeScript files.

# Bad
Format code properly.
```

**Use Structure**:
```markdown
# Build Commands

- Rust: `cargo build --release`
- Python: `pip install -r requirements.txt`
- Docker: `docker compose up -d`

# Testing

- Run all: `cargo test && pytest`
- Single: `cargo test ihsan`
```

### Organizing Rules

**Keep Rules Focused**:
- One topic per file
- Use descriptive filenames
- Group related rules in subdirectories

**Avoid Conflicts**:
- Don't contradict rules in other files
- Document exceptions explicitly
- Higher-level rules override lower-level

### Regular Review

- Update memories as project evolves
- Remove outdated instructions
- Add new patterns as they emerge

---

## BIZRA Integration

### Memory + Hooks

Memories inform hook validation:

```markdown
# .claude/rules/security/secrets.md
Never log or commit:
- API keys
- Passwords
- Private keys (*.pem)
```

The `validate-file-ops.py` hook references these rules when protecting files.

### Memory + Commands

Memories guide slash command behavior:

```markdown
# .claude/rules/validation/ihsan.md
When validating Ihsān:
1. Check 8 dimensions present
2. Verify weights sum to 1.0
3. Confirm threshold ≥ 0.99
```

The `/ihsan` command follows these instructions.

### Memory + Agents

Memories shape agent decision-making:

```markdown
# .claude/rules/orchestration/pat-sat.md
SAT Pre-validation:
- Requires 3/5 consensus
- Never proceed without approval
- Emit rejection receipts on failure
```

Agents use these rules when orchestrating.

### BIZRA Memory Structure

```
CLAUDE.md                          # Main BIZRA instructions
CLAUDE.local.md.template           # Template for personal settings
.claude/
├── CLAUDE.md                      # Alternative location (not used)
├── MEMORY_MANAGEMENT.md           # This file
└── rules/
    ├── README.md                  # Rules documentation
    ├── general.md                 # Universal BIZRA rules
    ├── rust/                      # Rust-specific rules
    │   ├── code-style.md
    │   ├── safety.md
    │   └── testing.md
    ├── python/                    # Python-specific rules
    │   ├── code-style.md
    │   └── typing.md
    ├── security/                  # Security rules
    │   ├── secrets.md
    │   └── authentication.md
    ├── validation/                # BIZRA validation rules
    │   ├── ihsan.md
    │   ├── sape.md
    │   └── fate.md
    ├── evidence/                  # Evidence rules
    │   ├── receipts.md
    │   └── chains.md
    └── orchestration/             # Multi-agent rules
        ├── pat-sat.md
        └── swarm.md
```

---

## Quick Reference

### Create Project Memory
```bash
# Option 1: Root level
touch CLAUDE.md

# Option 2: .claude directory
mkdir -p .claude && touch .claude/CLAUDE.md
```

### Create Modular Rules
```bash
mkdir -p .claude/rules/rust
touch .claude/rules/rust/code-style.md
```

### Create Personal Settings
```bash
cp CLAUDE.local.md.template CLAUDE.local.md
# Edit CLAUDE.local.md with your preferences
```

### View Loaded Memories
```
/memory
```

### Initialize CLAUDE.md
```
/init
```

---

## See Also

- **Modular Rules**: `.claude/rules/README.md`
- **Hooks Reference**: `.claude/hooks/HOOKS_QUICK_REFERENCE.md`
- **Commands Reference**: `.claude/commands/COMMAND_CHEAT_SHEET.md`
- **CLI Reference**: `.claude/CLI_REFERENCE.md`
- **Full System Guide**: `CLAUDE.md`

---

## Claude-Flow Integration

BIZRA's memory system integrates with [claude-flow](https://github.com/ruvnet/claude-flow) patterns:

### Neural Routing Memory
Store learned routing patterns in memory:

```markdown
# .claude/rules/orchestration/neural-routing.md
Learned patterns:
- Code review tasks → MasterReasoner + EthicsGuardian
- Data analysis → DataAnalyzer
- Creative writing → CreativeSynthesizer
```

### Pattern Elevation Memory
Track elevated patterns:

```markdown
# .claude/rules/orchestration/elevated-patterns.md
Elevated patterns (>3 occurrences):
- Simple transforms → WASM handler
- Code formatting → Direct execution
- Lint checks → Haiku model
```

### Swarm Configuration Memory
Define swarm behaviors:

```markdown
# .claude/rules/orchestration/swarm.md
Swarm modes:
- Independent: Parallelizable tasks
- Collaborative: Complex multi-perspective tasks
- HiveMind: High-stakes consensus decisions
```

Sources:
- [claude-flow GitHub](https://github.com/ruvnet/claude-flow)
- [claude-flow Official Site](https://claude-flow.ruv.io/)
