"""Detect URP acronym drift across documentation.

The audit treats acronym expansion as a truth-boundary issue: if the same
public architecture acronym expands to multiple concepts, downstream claims and
runbooks can silently diverge.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List


CANONICAL_EXPANSION = "Universal Resource Pool"
ALTERNATE_EXPANSIONS = {
    "Universal Receipt Plane": "historical proof-plane alias",
    "Universal Resource Protocol": "historical protocol alias",
    "Universal Rights Protocol": "historical rights-layer alias",
    "Universal Reasoning Protocol": "historical reasoning-layer alias",
}

_EXPANSION_RE = re.compile(
    r"\bURP\s*(?:\(|:|-|=)?\s*"
    r"(Universal\s+(?:Receipt\s+Plane|Resource\s+Pool|Resource\s+Protocol|"
    r"Rights\s+Protocol|Reasoning\s+Protocol))\)?"
    r"|"
    r"\b(Universal\s+(?:Receipt\s+Plane|Resource\s+Pool|Resource\s+Protocol|"
    r"Rights\s+Protocol|Reasoning\s+Protocol))\s*\(\s*URP\s*\)",
    flags=re.IGNORECASE,
)


def _is_excluded(rel_path: Path, exclude_dirs: List[str]) -> bool:
    parts = set(rel_path.parts)
    for directory in exclude_dirs:
        if directory in parts:
            return True
        rel = rel_path.as_posix()
        if rel == directory or rel.startswith(directory + "/"):
            return True
    return False


def _iter_docs(repo_root: Path, roots: List[str], exclude_dirs: List[str]) -> Iterable[Path]:
    for root in roots:
        abs_root = repo_root / root
        if abs_root.is_file():
            rel = abs_root.relative_to(repo_root)
            if not _is_excluded(rel, exclude_dirs):
                yield abs_root
            continue
        if not abs_root.is_dir():
            continue
        for dirpath, dirnames, filenames in os.walk(abs_root):
            dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
            for filename in filenames:
                if not filename.lower().endswith((".md", ".txt")):
                    continue
                path = Path(dirpath) / filename
                rel = path.relative_to(repo_root)
                if not _is_excluded(rel, exclude_dirs):
                    yield path


def _normalize_expansion(value: str) -> str:
    words = value.split()
    return " ".join(word[:1].upper() + word[1:].lower() for word in words)


def _excerpt(line: str, start: int, end: int, radius: int = 80) -> str:
    lo = max(0, start - radius)
    hi = min(len(line), end + radius)
    return line[lo:hi].strip()


def scan(
    repo_root: Path,
    roots: List[str],
    exclude_dirs: List[str],
    max_bytes: int,
    limit: int,
) -> List[dict]:
    """Return URP expansion observations from configured documentation roots."""

    observations: List[dict] = []
    seen: set[tuple[str, int, str]] = set()

    for path in _iter_docs(repo_root, roots, exclude_dirs):
        try:
            if path.stat().st_size > max_bytes:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue

        rel = path.relative_to(repo_root).as_posix()
        for line_no, line in enumerate(text.splitlines(), 1):
            for match in _EXPANSION_RE.finditer(line):
                raw = match.group(1) or match.group(2) or ""
                expansion = _normalize_expansion(raw)
                key = (rel, line_no, expansion)
                if key in seen:
                    continue
                seen.add(key)

                if expansion == CANONICAL_EXPANSION:
                    classification = "CANONICAL"
                elif expansion in ALTERNATE_EXPANSIONS:
                    classification = (
                        "DOCUMENTED_ALIAS"
                        if rel.endswith("URP_CANONICAL_DEFINITION.md")
                        else "ALTERNATE"
                    )
                else:
                    classification = "UNKNOWN"
                observations.append(
                    {
                        "path": rel,
                        "line": line_no,
                        "expansion": expansion,
                        "classification": classification,
                        "canonical_expansion": CANONICAL_EXPANSION,
                        "note": ALTERNATE_EXPANSIONS.get(expansion, ""),
                        "excerpt": _excerpt(line, match.start(), match.end()),
                    }
                )
                if len(observations) >= limit:
                    return observations

    observations.sort(key=lambda item: (item["classification"], item["path"], item["line"]))
    return observations


def write_outputs(observations: List[dict], out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "urp_canonicality.json"
    with path.open("w", encoding="utf-8") as f:
        json.dump(observations, f, indent=2, ensure_ascii=False)
    return {"urp_canonicality_json": str(path)}
