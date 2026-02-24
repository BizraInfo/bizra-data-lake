---
name: "Verification"
description: "Quick alias for verification-quality. Comprehensive truth scoring, code quality verification, and automatic rollback system with 0.95 accuracy threshold."
version: "1.0.0"
category: "quality-assurance"
tags: ["verification", "truth-scoring", "quality", "alias"]
---

# Verification

Quick alias for the `verification-quality` skill.

## Usage

Run verification checks on the codebase using truth scoring (0.0-1.0 scale), automated code quality validation, and automatic rollback for changes below threshold.

## Quick Commands

```bash
# View current truth scores
npx claude-flow@alpha truth

# Run verification check
npx claude-flow@alpha verify check

# Verify specific file
npx claude-flow@alpha verify check --file <path> --threshold 0.95

# Rollback failed changes
npx claude-flow@alpha verify rollback --last-good
```

## Thresholds

| Level | Score | Use Case |
|-------|-------|----------|
| Production (Ihsan) | 0.95 | Standard quality gate |
| Strict (Ihsan) | 0.99 | Critical code paths |
| Warning | 0.85 | Needs attention |
| Critical | 0.75 | Requires immediate action |

## What Gets Checked

1. **Code Correctness** - Syntax, types, logic flow, error handling
2. **Security** - Vulnerability scanning, secret detection, input validation
3. **Best Practices** - SOLID principles, design patterns, modularity
4. **Performance** - Algorithmic complexity, memory usage, query optimization
5. **Documentation** - JSDoc completeness, README accuracy

## Integration

```bash
# CI/CD gate
npx claude-flow@alpha verify check --threshold 0.95 --json > verification.json

# Pre-commit hook
npx claude-flow@alpha verify install-hook --pre-commit

# Watch mode
npx claude-flow@alpha verify watch --directory src/ --auto-fix

# Swarm integration
npx claude-flow@alpha swarm --verify --threshold 0.98
```

## See Also

- `verification-quality` - Full detailed documentation
- `guardian-review` - Multi-perspective quality review
- `snr-check` - Signal-to-noise analysis
