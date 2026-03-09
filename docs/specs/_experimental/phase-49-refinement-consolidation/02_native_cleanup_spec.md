# Phase 49 Spec — Part 2: Native Cleanup

> Standing on Giants: Brooks (no silver bullet — eliminate dead code) · Fowler (refactoring — safe incremental steps)

## Problem

The `native/` directory contains 8,472 LOC of Rust code that is now duplicated in `bizra-omega/`. Two parallel workspace definitions confuse builds, waste CI cycles, and risk divergent edits.

## Current Layout

```
native/
  Cargo.toml          # 4-member workspace (shadows bizra-omega/)
  Cargo.lock          # Separate lock file
  DEPRECATED.md       # Deprecation notice (written Phase 48)
  bizra-hooks/        # Identical to bizra-omega/bizra-hooks/
  bizra-memory/       # Identical to bizra-omega/bizra-memory/
  fate-binding/       # Identical to bizra-omega/fate-binding/
  iceoryx-bridge/     # Identical to bizra-omega/iceoryx-bridge/
  target/             # Build artifacts
```

## Verification Before Cleanup

```pseudocode
FUNCTION verify_native_removal_is_safe():
    # 1. Confirm no files differ
    FOR each crate IN [bizra-hooks, bizra-memory, fate-binding, iceoryx-bridge]:
        diff = shell("diff -rq native/{crate}/src bizra-omega/{crate}/src")
        ASSERT diff.is_empty(), "Source files differ — merge before deleting"

    # 2. Confirm bizra-omega workspace includes all 4 crates
    omega_toml = read("bizra-omega/Cargo.toml")
    ASSERT "bizra-hooks" IN omega_toml.members
    ASSERT "bizra-memory" IN omega_toml.members
    ASSERT "fate-binding" IN omega_toml.members
    ASSERT "iceoryx-bridge" IN omega_toml.members

    # 3. Confirm bizra-omega tests pass
    result = shell("cd bizra-omega && cargo test --workspace")
    ASSERT result.exit_code == 0, "Workspace tests must pass before native/ removal"

    # 4. Confirm no Python code imports from native/
    grep_result = shell("grep -r 'native/' core/ tools/ scripts/ --include='*.py'")
    ASSERT grep_result.is_empty(), "Python code references native/ — update paths first"

    # 5. Confirm CI doesn't reference native/ in required paths
    ci_yml = read(".github/workflows/ci.yml")
    # native-ci.yml is a separate workflow that should also be retired

    RETURN "safe_to_remove"
```

## Cleanup Steps

```pseudocode
FUNCTION cleanup_native():
    verify_native_removal_is_safe()

    # Step 1: Remove native CI workflow (or redirect to bizra-omega)
    IF exists(".github/workflows/native-ci.yml"):
        DELETE ".github/workflows/native-ci.yml"
        # Tests are already covered by test-rust job which builds bizra-omega/

    # Step 2: Update any remaining references
    FOR each file IN find_references_to("native/"):
        UPDATE file: replace "native/" with "bizra-omega/"

    # Step 3: Remove native/ directory
    # Keep DEPRECATED.md as a forwarding notice? No — git history preserves it.
    DELETE "native/"

    # Step 4: Update .gitignore if it references native/target
    UPDATE .gitignore: remove native/target entries

    # Step 5: Verify workspace still builds
    shell("cd bizra-omega && cargo test --workspace")
    ASSERT exit_code == 0
```

## TDD Anchors

```pseudocode
TEST "workspace_builds_after_native_removal":
    # Given: native/ has been deleted
    # When: cargo test --workspace in bizra-omega/
    # Then: 610+ tests pass

TEST "no_python_imports_reference_native":
    # Given: full codebase
    # When: grep for "native/" in *.py files
    # Then: zero matches

TEST "ci_does_not_reference_native":
    # Given: .github/workflows/ci.yml
    # When: grep for "native/" or "native-ci"
    # Then: zero matches in required jobs
```

## Risk

**Low.** `DEPRECATED.md` already documents the migration. The only risk is if someone made edits to `native/` since the merge — verify with `diff -rq` first.

## Estimated Effort

~15 minutes. This is a deletion with verification, not new code.
