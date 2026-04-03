# BIZRA Modular Rules

Organized, topic-specific instructions for Claude Code that automatically load based on context.

## Directory Structure

```
.claude/rules/
├── README.md                   # This file
├── general.md                  # Universal BIZRA rules
├── rust/                       # Rust-specific rules
│   ├── code-style.md          # Rust formatting and style
│   ├── safety.md              # Rust safety patterns
│   └── testing.md             # Rust testing conventions
├── python/                     # Python-specific rules
│   ├── code-style.md          # Python formatting and style
│   ├── typing.md              # Type annotation rules
│   └── testing.md             # Python testing conventions
├── security/                   # Security rules
│   ├── secrets.md             # Secret handling
│   └── authentication.md      # Auth patterns
├── validation/                 # BIZRA validation rules
│   ├── ihsan.md               # Ihsān gate rules
│   ├── sape.md                # SAPE probe rules
│   └── fate.md                # FATE escalation rules
├── evidence/                   # Evidence rules
│   ├── receipts.md            # Receipt generation rules
│   └── chains.md              # Evidence chain rules
└── orchestration/              # Multi-agent rules
    ├── pat-sat.md             # PAT/SAT coordination
    └── swarm.md               # Swarm intelligence patterns
```

## How Rules Work

### Automatic Loading

All `.md` files in `.claude/rules/` are automatically loaded as project memory when Claude Code starts. Files without `paths` frontmatter apply globally.

### Path-Specific Rules

Rules can be scoped to specific files using YAML frontmatter:

```markdown
---
paths:
  - "src/**/*.rs"
---

# Rust-specific rules here
```

### Supported Glob Patterns

| Pattern | Matches |
|---------|---------|
| `**/*.rs` | All Rust files |
| `src/**/*` | All files under src/ |
| `*.md` | Markdown files in root |
| `{src,lib}/**/*.ts` | TypeScript in src/ or lib/ |
| `**/*.{py,pyi}` | Python files and stubs |

## BIZRA-Specific Patterns

### Conditional Validation Rules

Rules in `validation/` apply to validation-related files:

```markdown
---
paths:
  - "constitution/**/*.yaml"
  - "src/ihsan.rs"
  - "core/sape.py"
---

# These rules apply when editing validation code
```

### Evidence Rules

Rules in `evidence/` apply to receipt and evidence handling:

```markdown
---
paths:
  - "docs/evidence/**/*"
  - "src/receipts.rs"
  - "core/fate.py"
---

# Receipt generation and validation rules
```

## Creating New Rules

1. Create a `.md` file in the appropriate subdirectory
2. Add optional `paths` frontmatter for conditional loading
3. Write clear, specific instructions
4. Test by running `/memory` to see loaded rules

## Best Practices

- **Keep rules focused**: One topic per file
- **Use descriptive filenames**: `testing.md`, not `misc.md`
- **Scope appropriately**: Only add `paths` when rules truly apply to specific files
- **Avoid conflicts**: Don't contradict rules in other files
- **Document exceptions**: Note when rules have exceptions

## Integration with BIZRA Systems

These rules integrate with:

- **Hooks**: Rules inform hook validation
- **Commands**: Rules guide slash command behavior
- **Agents**: Rules shape agent decision-making
- **Skills**: Rules define skill activation context

## See Also

- Memory Management: `.claude/MEMORY_MANAGEMENT.md`
- Project Memory: `CLAUDE.md`
- Local Memory: `CLAUDE.local.md`
