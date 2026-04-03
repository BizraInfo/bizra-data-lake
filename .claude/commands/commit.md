---
allowed-tools: Bash(git:*)
description: Create BIZRA-compliant git commit with evidence
argument-hint: [message]
hooks:
  PreToolUse:
    - matcher: "Bash"
      hooks:
        - type: command
          command: "$CLAUDE_PROJECT_DIR/.claude/hooks/validate-bash.py"
---

# BIZRA Git Commit

## Current Repository Status

- Branch: !`git branch --show-current`
- Status: !`git status --short`
- Staged files: !`git diff --cached --name-only | wc -l`
- Unstaged changes: !`git diff --name-only | wc -l`
- Recent commits: !`git log --oneline -5`

## Commit Message

Message: **$ARGUMENTS**

## Your Task

### 1. Pre-commit Validation

**Check for protected files**:
```bash
# Verify no uncommitted changes to critical files without review
PROTECTED_FILES="constitution/ihsan_v1.yaml src/receipts.rs core/fate.py docker-compose.yml"

for file in $PROTECTED_FILES; do
    if git diff --name-only | grep -q "^$file$"; then
        echo "⚠️ WARNING: Uncommitted changes to protected file: $file"
        echo "   Receipt Schema Guard may be required"
    fi
done
```

**Run pre-commit checks**:
```bash
# Rust linting if Rust files changed
if git diff --cached --name-only | grep -q '\.rs$'; then
    echo "Running cargo clippy..."
    cargo clippy --all-targets -- -D warnings || {
        echo "❌ FAIL-CLOSED: Clippy errors must be fixed before commit"
        exit 2
    }
fi

# Python syntax if Python files changed
if git diff --cached --name-only | grep -q '\.py$'; then
    echo "Checking Python syntax..."
    git diff --cached --name-only | grep '\.py$' | while read file; do
        python3 -m py_compile "$file" || {
            echo "❌ FAIL-CLOSED: Python syntax errors in $file"
            exit 2
        }
    done
fi

# YAML validation if YAML files changed
if git diff --cached --name-only | grep -q '\.ya\?ml$'; then
    echo "Validating YAML files..."
    git diff --cached --name-only | grep '\.ya\?ml$' | while read file; do
        python3 -c "import yaml; yaml.safe_load(open('$file'))" || {
            echo "❌ FAIL-CLOSED: YAML syntax error in $file"
            exit 2
        }
    done
fi
```

**Check for secrets**:
```bash
# Scan for potential secrets in staged files
if git diff --cached | grep -iE "(password|secret|key|token)\s*[:=]\s*['\"]?\w+"; then
    echo "⚠️ WARNING: Potential secrets detected in commit"
    echo "   Review changes carefully before committing"
fi
```

### 2. Generate Commit Message

If no message provided (`$ARGUMENTS` is empty), analyze changes:

```bash
# Analyze git diff for commit message
CHANGED_FILES=$(git diff --cached --name-only | tr '\n' ' ')
FILE_COUNT=$(git diff --cached --name-only | wc -l)

# Categorize changes
RUST_CHANGES=$(git diff --cached --name-only | grep -c '\.rs$' || echo 0)
PYTHON_CHANGES=$(git diff --cached --name-only | grep -c '\.py$' || echo 0)
DOC_CHANGES=$(git diff --cached --name-only | grep -c '\.md$' || echo 0)
CONFIG_CHANGES=$(git diff --cached --name-only | grep -cE '\.(yaml|yml|toml|json)$' || echo 0)

# Determine commit type
if [ $RUST_CHANGES -gt 0 ]; then
    COMMIT_TYPE="feat(rust)"
elif [ $PYTHON_CHANGES -gt 0 ]; then
    COMMIT_TYPE="feat(python)"
elif [ $DOC_CHANGES -gt 0 ]; then
    COMMIT_TYPE="docs"
elif [ $CONFIG_CHANGES -gt 0 ]; then
    COMMIT_TYPE="config"
else
    COMMIT_TYPE="chore"
fi

echo "Suggested commit type: $COMMIT_TYPE"
echo "Files changed: $FILE_COUNT"
```

### 3. Create Commit

**Commit with BIZRA signature**:
```bash
# If message provided, use it; otherwise prompt for one
if [ -n "$ARGUMENTS" ]; then
    COMMIT_MSG="$ARGUMENTS"
else
    echo "Enter commit message (or press Ctrl+C to cancel):"
    read COMMIT_MSG
fi

# Add all staged changes
git add --update

# Create commit with Co-Authored-By
git commit -m "$(cat <<EOF
$COMMIT_MSG

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
EOF
)"

# Verify commit
git log -1 --pretty=format:"%h - %s (%an, %ar)"
```

### 4. Generate Commit Receipt

**Create evidence of commit**:
```bash
COMMIT_HASH=$(git rev-parse HEAD)
COMMIT_TIME=$(git log -1 --format=%ai)
FILES_CHANGED=$(git diff --name-only HEAD~1 HEAD | wc -l)
LINES_ADDED=$(git diff --shortstat HEAD~1 HEAD | grep -oE '[0-9]+ insertion' | cut -d' ' -f1 || echo 0)
LINES_DELETED=$(git diff --shortstat HEAD~1 HEAD | grep -oE '[0-9]+ deletion' | cut -d' ' -f1 || echo 0)

cat > "docs/evidence/receipts/commit-$(date +%Y%m%d-%H%M%S).json" <<EOF
{
  "receipt_id": "commit-${COMMIT_HASH:0:8}",
  "timestamp": "$(date -Iseconds)",
  "commit_hash": "$COMMIT_HASH",
  "commit_time": "$COMMIT_TIME",
  "commit_message": "$COMMIT_MSG",
  "files_changed": $FILES_CHANGED,
  "lines_added": $LINES_ADDED,
  "lines_deleted": $LINES_DELETED,
  "branch": "$(git branch --show-current)",
  "author": "$(git log -1 --format=%an) ($(git log -1 --format=%ae))",
  "co_authored": "Claude Opus 4.5 <noreply@anthropic.com>",
  "integrity_hash": "$(echo -n "$COMMIT_HASH$COMMIT_MSG" | sha256sum | cut -d' ' -f1)"
}
EOF

echo "✓ Commit receipt generated"
```

### 5. Post-commit Status

```bash
# Show post-commit status
git status --short

# Show commit details
git show --stat HEAD

# Check if ahead of remote
git status | grep "Your branch is ahead" || echo "Up to date with remote"
```

## Commit Message Guidelines

Follow conventional commits format:

```
<type>(<scope>): <description>

[optional body]

[optional footer]

Co-Authored-By: Claude Opus 4.5 <noreply@anthropic.com>
```

**Types**:
- `feat`: New feature
- `fix`: Bug fix
- `docs`: Documentation only
- `style`: Formatting, missing semicolons, etc.
- `refactor`: Code restructuring
- `test`: Adding tests
- `chore`: Maintenance

**Scopes**:
- `rust`: Rust code changes
- `python`: Python code changes
- `docker`: Docker/compose changes
- `config`: Configuration changes
- `hooks`: Hook system changes

## Fail-Closed Requirements

**BLOCK commit** if:
- Clippy errors in Rust code
- Python syntax errors
- YAML syntax errors
- Protected files changed without Receipt Schema Guard
- Potential secrets detected

**WARN** but allow:
- No commit message provided (will prompt)
- Large commit (>100 files)
- Merge commit

## Evidence Generation

Commit receipt includes:
- commit_hash: Git SHA-1
- commit_time: Timestamp
- commit_message: Full message
- files_changed: Count
- lines_added/deleted: Stats
- branch: Current branch
- author + co_author: Attribution
- integrity_hash: Receipt verification

## Report Format

```
## Commit Created

**Hash**: [short-hash]
**Branch**: [branch-name]
**Files**: X changed
**Lines**: +X -Y

### Message
[commit message]

### Files Changed
[list of files]

### Receipt
- Location: docs/evidence/receipts/commit-YYYYMMDD-HHMMSS.json
- Integrity: [SHA-256]

### Next Steps
- Review changes: git show HEAD
- Push to remote: git push
- Create PR: gh pr create
```

---

**Git Philosophy**: "Receipt-first commits. Co-authored by Claude. Evidence-backed. Fail-closed on errors."
