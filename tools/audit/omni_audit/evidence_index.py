"""Build a read-only evidence index over doctrine, manifests, and key configs."""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List

from .schemas import EvidenceItem


_TYPE_BY_SUFFIX = {
    ".md": "markdown",
    ".rs": "rust",
    ".py": "python",
    ".toml": "manifest",
    ".lock": "lock",
    ".json": "json",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".txt": "text",
    ".html": "html",
    ".svg": "svg",
    ".csv": "csv",
}


def _is_excluded(rel_path: Path, exclude_dirs: List[str]) -> bool:
    parts = set(rel_path.parts)
    for d in exclude_dirs:
        if d in parts:
            return True
        # also handle dotted forms
        if rel_path.as_posix().startswith(d + "/") or rel_path.as_posix() == d:
            return True
    return False


def _classify(rel_path: Path) -> tuple[str, str]:
    """Return (evidence_class, purpose_guess) via heuristics."""
    s = rel_path.as_posix().lower()
    name = rel_path.name.lower()

    if s.startswith("tools/cognitive_foundry/claude_lane/canon_packs"):
        return ("CANON_PACK", "Cognitive Foundry canon-pack staging (not runtime canon).")
    if "canonical_receipt" in s or "receipt" in s and "rs" in s:
        return ("CODE", "Receipt / canonical-receipt code.")
    if "constitution" in s or "constitutional" in s or "genesis_seal" in s:
        return ("DOCTRINE", "Constitutional / foundational doctrine.")
    if "manifesto" in s or "charter" in s or "covenant" in s:
        return ("DOCTRINE", "Manifesto / charter-level doctrine.")
    if name in ("claude.md", "readme.md") or name.startswith("readme"):
        return ("DOCTRINE", "Top-level / section README.")
    if name == "memory.md":
        return ("DOCTRINE", "Memory anchor / persistent doctrine.")
    if name == "pyproject.toml":
        return ("MANIFEST", "Python project manifest.")
    if name == "cargo.toml":
        return ("MANIFEST", "Rust crate / workspace manifest.")
    if name == "cargo.lock":
        return ("MANIFEST", "Rust dependency lockfile.")
    if name.startswith("requirements") and name.endswith(".txt"):
        return ("MANIFEST", "Python pip requirements.")
    if name == "package.json":
        return ("MANIFEST", "Node package manifest.")
    if name == "package-lock.json":
        return ("MANIFEST", "Node lockfile.")
    if s.startswith("docs/adr/"):
        return ("ADR", "Architecture decision record.")
    if s.startswith("docs/security/") or "security" in s:
        return ("SECURITY", "Security-related documentation.")
    if s.startswith("docs/strategy/"):
        return ("STRATEGY", "Strategic positioning doc.")
    if s.startswith("docs/brand/"):
        return ("BRAND", "Brand / public-surface documentation.")
    if s.startswith("docs/"):
        return ("DOC", "General documentation.")
    if name.endswith(".yaml") or name.endswith(".yml"):
        return ("CONFIG", "YAML config.")
    if name.endswith(".json"):
        return ("CONFIG", "JSON config or data.")
    if name.endswith(".rs"):
        return ("CODE", "Rust source.")
    if name.endswith(".py"):
        return ("CODE", "Python source.")
    return ("ARTIFACT", "Repository artifact.")


def iter_candidate_paths(
    repo_root: Path,
    include_suffixes: List[str],
    include_basenames: List[str],
    exclude_dirs: List[str],
    max_files: int,
) -> Iterable[Path]:
    """Walk repo and yield relative paths matching any suffix/basename rule."""

    count = 0
    suffixes = {s.lower() for s in include_suffixes}
    basenames_exact = {b for b in include_basenames if not b.startswith("README")}
    basename_prefixes = tuple(b for b in include_basenames if b.startswith("README"))

    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in exclude_dirs]
        rel_dir = Path(dirpath).relative_to(repo_root)

        for fn in filenames:
            rel = rel_dir / fn
            if _is_excluded(rel, exclude_dirs):
                continue
            low = fn.lower()
            ok = False
            # suffix match
            for suf in suffixes:
                if low.endswith(suf):
                    ok = True
                    break
            if not ok and fn in basenames_exact:
                ok = True
            if not ok and fn.startswith(basename_prefixes):
                ok = True
            if ok:
                yield rel
                count += 1
                if count >= max_files:
                    return


def _sha256_of(path: Path, chunk: int = 65536) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
    except OSError:
        return ""
    return h.hexdigest()


def build_evidence_index(
    repo_root: Path,
    include_suffixes: List[str],
    include_basenames: List[str],
    exclude_dirs: List[str],
    limit: int,
) -> List[EvidenceItem]:
    items: List[EvidenceItem] = []
    seq = 1
    for rel in iter_candidate_paths(repo_root, include_suffixes, include_basenames,
                                     exclude_dirs, max_files=limit * 3):
        abs_path = repo_root / rel
        if not abs_path.is_file():
            continue
        try:
            st = abs_path.stat()
        except OSError:
            continue
        suffix = abs_path.suffix.lower()
        typ = _TYPE_BY_SUFFIX.get(suffix, "other")
        cls, purpose = _classify(rel)
        sha = _sha256_of(abs_path)
        items.append(
            EvidenceItem(
                item_id=f"E{seq:05d}",
                path=rel.as_posix(),
                sha256=sha,
                size_bytes=st.st_size,
                modified_ts=datetime.fromtimestamp(st.st_mtime, tz=timezone.utc)
                    .replace(microsecond=0).isoformat(),
                type=typ,
                purpose_guess=purpose,
                evidence_class=cls,
            )
        )
        seq += 1
        if len(items) >= limit:
            break

    # Deterministic order: class-major, then path.
    items.sort(key=lambda it: (it.evidence_class, it.path))
    # Reassign sequential IDs after sort for stability.
    for i, it in enumerate(items, 1):
        it.item_id = f"E{i:05d}"
    return items


def write_outputs(items: List[EvidenceItem], out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "evidence_index.json"
    csv_path = out_dir / "evidence_index.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump([asdict(i) for i in items], f, indent=2, ensure_ascii=False)

    cols = ["item_id", "evidence_class", "type", "path", "size_bytes",
            "modified_ts", "sha256", "purpose_guess"]
    with csv_path.open("w", encoding="utf-8") as f:
        f.write(",".join(cols) + "\n")
        for it in items:
            row = [str(getattr(it, c, "")) for c in cols]
            row = [v.replace(",", ";").replace("\n", " ") for v in row]
            f.write(",".join(row) + "\n")

    return {"evidence_index_json": str(json_path), "evidence_index_csv": str(csv_path)}
