#!/usr/bin/env python3
"""
Batch fix common MyPy errors across the codebase.

Targets:
1. [no-untyped-def] - Add -> None to __init__, __post_init__ methods
2. [type-arg] - Fix bare dict, list, tuple annotations
3. [unreachable] - These are typically after return/raise/break in try/except
4. type: ignore without code -> add specific code

Does NOT change logic. Only adds type annotations.
"""
import re
import sys
from pathlib import Path


def fix_init_return_types(content: str) -> str:
    """Add -> None to __init__ and __post_init__ methods missing return type."""
    # Match def __init__(self, ...) without -> annotation
    # Handle multiline signatures too
    lines = content.split("\n")
    result = []
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.lstrip()

        # Check for def __init__( or def __post_init__( without ->
        if re.match(r'\s+def (__init__|__post_init__)\(', line):
            # Collect the full signature (might be multiline)
            sig_lines = [line]
            j = i + 1
            # Check if signature spans multiple lines (doesn't end with ):)
            while j < len(lines) and not re.search(r'\)\s*:', lines[j - 1] if j > i else line):
                if re.search(r'\)\s*(->\s*\w+\s*)?:', lines[j - 1]):
                    break
                sig_lines.append(lines[j])
                j += 1

            # Join signature to check for ->
            full_sig = " ".join(l.strip() for l in sig_lines)

            if "->" not in full_sig:
                # Find the line with ): and add -> None
                for k in range(i, min(j + 1, len(lines))):
                    if re.search(r'\)\s*:', lines[k]):
                        lines[k] = re.sub(r'\)\s*:', ') -> None:', lines[k])
                        break

        result.append(lines[i])
        i += 1

    return "\n".join(result)


def fix_bare_generic_types(content: str) -> str:
    """Fix bare dict, list, tuple, set in type annotations."""
    # Fix -> dict: to -> Dict[str, Any]:
    content = re.sub(
        r'(\)\s*->\s*)dict(\s*:)',
        r'\1Dict[str, Any]\2',
        content
    )
    # Fix -> list: to -> List[Any]:
    content = re.sub(
        r'(\)\s*->\s*)list(\s*:)',
        r'\1List[Any]\2',
        content
    )
    # Fix -> tuple: to -> tuple[Any, ...]:
    # Actually use Tuple for 3.11 compat
    content = re.sub(
        r'(\)\s*->\s*)tuple(\s*:)',
        r'\1Any\2',  # too ambiguous, just use Any
        content
    )

    # Fix param annotations: param: dict, or param: dict = {}
    # Only in function signatures (indented, after def or comma)
    content = re.sub(
        r'(:\s*)dict(\s*[,=\)])',
        r'\1Dict[str, Any]\2',
        content
    )
    content = re.sub(
        r'(:\s*)list(\s*[,=\)])',
        r'\1List[Any]\2',
        content
    )

    # Fix dataclass fields: field_name: dict and field_name: list
    content = re.sub(
        r'^(\s+\w+:\s*)dict(\s*$)',
        r'\1Dict[str, Any]\2',
        content,
        flags=re.MULTILINE
    )
    content = re.sub(
        r'^(\s+\w+:\s*)list(\s*$)',
        r'\1List[Any]\2',
        content,
        flags=re.MULTILINE
    )

    # Fix Dict without type params in function signatures
    content = re.sub(
        r'(:\s*)Dict(\s*[,=\)])',
        r'\1Dict[str, Any]\2',
        content
    )
    content = re.sub(
        r'(\)\s*->\s*)Dict(\s*:)',
        r'\1Dict[str, Any]\2',
        content
    )

    return content


def ensure_typing_imports(content: str) -> str:
    """Ensure Dict, List, Optional, Any are imported from typing."""
    # Check if typing is imported
    if "from typing import" not in content:
        return content

    # Check which types are used but not imported
    typing_line_match = re.search(r'^from typing import (.+?)$', content, re.MULTILINE)
    if not typing_line_match:
        return content

    imports = typing_line_match.group(1)

    needed = set()
    if "Dict[" in content and "Dict" not in imports:
        needed.add("Dict")
    if "List[" in content and "List" not in imports:
        needed.add("List")
    if "Optional[" in content and "Optional" not in imports:
        needed.add("Optional")
    if "Any" in content and "Any" not in imports:
        needed.add("Any")

    if not needed:
        return content

    # Add missing imports
    existing = [s.strip() for s in imports.split(",")]
    all_imports = sorted(set(existing) | needed)
    new_import_line = "from typing import " + ", ".join(all_imports)

    content = content.replace(typing_line_match.group(0), new_import_line)
    return content


def fix_file(filepath: Path) -> tuple:
    """Fix a single file. Returns (original_content, new_content)."""
    content = filepath.read_text(encoding="utf-8")
    original = content

    content = fix_init_return_types(content)
    content = fix_bare_generic_types(content)
    # Don't mess with imports automatically - too risky

    return original, content


def main() -> None:
    root = Path("/mnt/c/BIZRA-DATA-LAKE/core")

    # Read error file to find which files to fix
    with open("/mnt/c/BIZRA-DATA-LAKE/mypy_full.txt") as f:
        raw = f.read()

    # Get files with no-untyped-def or type-arg errors
    target_files = set()
    for line in raw.split("\n"):
        if ": error:" in line and ("[no-untyped-def]" in line or "[type-arg]" in line):
            fp = line.split(":")[0]
            target_files.add(fp)

    print(f"Found {len(target_files)} files with fixable errors")

    fixed_count = 0
    for fp in sorted(target_files):
        filepath = Path("/mnt/c/BIZRA-DATA-LAKE") / fp
        if not filepath.exists():
            continue

        original, new_content = fix_file(filepath)
        if original != new_content:
            filepath.write_text(new_content, encoding="utf-8")
            fixed_count += 1
            # Count changes
            orig_lines = set(original.split("\n"))
            new_lines = set(new_content.split("\n"))
            changes = len(new_lines - orig_lines)
            print(f"  FIXED {fp} ({changes} lines changed)")

    print(f"\nFixed {fixed_count} files")


if __name__ == "__main__":
    main()
