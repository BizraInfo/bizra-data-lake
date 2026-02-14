#!/usr/bin/env python3
"""
Lightweight secret scan for YAML/scripts without external dependencies.

Fail criteria:
- Hardcoded token/secret/password-like assignments in tracked YAML/script files.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

YAML_EXT = {".yml", ".yaml"}
SCRIPT_EXT = {".py", ".sh", ".bash", ".zsh", ".ahk", ".ps1"}
SCRIPT_DIR_PREFIXES = ("scripts/", "bin/", "tools/")
PATH_SKIP_PARTS = {
    ".git/",
    "tests/",
    "docs/",
    "node_modules/",
    ".venv/",
    "venv/",
    ".mypy_cache/",
    ".pytest_cache/",
}

ASSIGNMENT_RE = re.compile(
    r"""(?ix)
    \b(api[_-]?token|api[_-]?key|access[_-]?token|secret|password)\b
    \s*[:=]\s*
    ["']([^"'\n]{8,})["']
    """
)

HIGH_ENTROPY_RE = re.compile(
    r"""(?ix)
    (sk-[a-z0-9:_-]{16,}|gh[pousr]_[a-z0-9]{20,}|xox[baprs]-[a-z0-9-]{20,})
    """
)


def tracked_files() -> list[Path]:
    try:
        out = subprocess.check_output(
            ["git", "ls-files"],
            cwd=ROOT,
            text=True,
        )
    except Exception:
        return []
    files: list[Path] = []
    for rel in out.splitlines():
        p = Path(rel)
        rel_posix = rel.replace("\\", "/")
        suffix = p.suffix.lower()
        is_yaml = suffix in YAML_EXT
        is_script = suffix in SCRIPT_EXT and rel_posix.startswith(SCRIPT_DIR_PREFIXES)
        if not (is_yaml or is_script):
            continue
        if any(skip in rel_posix for skip in PATH_SKIP_PARTS):
            continue
        files.append(ROOT / p)
    return files


def is_placeholder(value: str) -> bool:
    v = value.strip()
    return (
        v.startswith("${")
        or v.startswith("$(")
        or v.lower().startswith("env:")
        or v.lower() in {"lm-studio", "local", "localhost", "none", "null"}
        or "changeme" in v.lower()
        or "example" in v.lower()
        or "dummy" in v.lower()
        or "test" in v.lower()
    )


def scan_file(path: Path) -> list[str]:
    findings: list[str] = []
    try:
        text = path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return findings

    for i, line in enumerate(text.splitlines(), start=1):
        m = ASSIGNMENT_RE.search(line)
        if m:
            value = m.group(2)
            if not is_placeholder(value):
                findings.append(f"{path.relative_to(ROOT)}:{i}: hardcoded {m.group(1)}")
                continue
        hm = HIGH_ENTROPY_RE.search(line)
        if hm and not is_placeholder(hm.group(1)):
            findings.append(f"{path.relative_to(ROOT)}:{i}: high-entropy token pattern")
    return findings


def main() -> int:
    files = tracked_files()
    findings: list[str] = []
    for path in files:
        findings.extend(scan_file(path))

    if findings:
        print("Secret scan failed. Findings:")
        for item in findings:
            print(f"  - {item}")
        return 1

    print(f"Secret scan passed ({len(files)} files checked).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
