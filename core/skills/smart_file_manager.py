"""
Smart File Management Skill — Batch Rename, Organize, Classify, Merge
======================================================================

Registers a programmatic skill on the SkillRouter accessible via the
Desktop Bridge's ``invoke_skill`` JSON-RPC method (Ctrl+B, F).

Exposed operations:
  scan     -- Classify files in a directory by category
  organize -- Move/copy files into category sub-folders (dry-run default)
  rename   -- Batch rename with pattern + hash (dry-run default)
  merge    -- Concatenate text files into one

Zero new dependencies -- stdlib only + existing codebase imports.

Standing on Giants:
- Boyd (OODA): Scan before acting (dry-run default)
- Shannon (1948): Classify by signal type (extension -> category)
- Nygard (2007): Fail-safe defaults (dry_run=True)

Created: 2026-02-13 | BIZRA Smart File Manager v1.0
"""

from __future__ import annotations

import logging
import os
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.integration.constants import UNIFIED_IHSAN_THRESHOLD

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Reuse file-type classification from existing indexer
# ---------------------------------------------------------------------------

from tools.file_type_indexer import EXT_TO_CATEGORY

# ---------------------------------------------------------------------------
# Path safety
# ---------------------------------------------------------------------------

_DATA_LAKE_ROOT: Optional[Path] = None


def _get_data_lake_root() -> Path:
    """Resolve DATA_LAKE_ROOT lazily to avoid import-time side effects."""
    global _DATA_LAKE_ROOT
    if _DATA_LAKE_ROOT is None:
        try:
            from bizra_config import DATA_LAKE_ROOT

            _DATA_LAKE_ROOT = Path(DATA_LAKE_ROOT).resolve()
        except ImportError:
            _DATA_LAKE_ROOT = Path("/mnt/c/BIZRA-DATA-LAKE").resolve()
    return _DATA_LAKE_ROOT


def _validate_path(path: str, allow_outside_root: bool = False) -> Path:
    """Resolve and validate a path. Reject traversal outside DATA_LAKE_ROOT.

    Args:
        path: The path string to validate.
        allow_outside_root: If True, skip containment check (for testing).

    Returns:
        Resolved Path object.

    Raises:
        ValueError: If path is invalid or outside root.
    """
    resolved = Path(path).resolve()
    if not allow_outside_root:
        root = _get_data_lake_root()
        try:
            resolved.relative_to(root)
        except ValueError:
            raise ValueError(
                f"Path '{resolved}' is outside DATA_LAKE_ROOT '{root}'. "
                "Traversal blocked."
            )
    if not resolved.exists():
        raise ValueError(f"Path does not exist: {resolved}")
    return resolved


# ---------------------------------------------------------------------------
# Safe rename tokens
# ---------------------------------------------------------------------------

_SAFE_TOKENS = {"name", "ext", "n", "hash", "date", "cat"}


def _expand_pattern(
    pattern: str,
    *,
    name: str,
    ext: str,
    n: int,
    content_hash: str,
    date: str,
    cat: str,
) -> str:
    """Expand safe tokens in a rename pattern. No eval().

    Supported tokens: {name}, {ext}, {n}, {hash}, {date}, {cat}
    """
    result = pattern
    result = result.replace("{name}", name)
    result = result.replace("{ext}", ext)
    result = result.replace("{n}", str(n))
    result = result.replace("{hash}", content_hash[:16])
    result = result.replace("{date}", date)
    result = result.replace("{cat}", cat)
    return result


def _file_hash(filepath: Path) -> str:
    """Compute BLAKE3 hex digest of file contents."""
    try:
        from core.proof_engine.canonical import hex_digest

        data = filepath.read_bytes()
        return hex_digest(data)
    except (ImportError, Exception):
        import hashlib

        data = filepath.read_bytes()
        return hashlib.sha256(data).hexdigest()


def _file_mod_date(filepath: Path) -> str:
    """Return file modification date as YYYY-MM-DD."""
    mtime = filepath.stat().st_mtime
    return datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")


def _size_human(size_bytes: int) -> str:
    """Convert bytes to human-readable size."""
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024:
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} PB"


# ---------------------------------------------------------------------------
# SmartFileHandler
# ---------------------------------------------------------------------------


class SmartFileHandler:
    """
    Bridge adapter that registers file management as an invocable skill.

    Registration:
        handler = SmartFileHandler()
        handler.register(router)

    Invocation (via bridge):
        {"jsonrpc": "2.0", "method": "invoke_skill", "params": {
            "skill": "smart_files",
            "inputs": {"operation": "scan", "path": "/mnt/c/..."}
        }, "id": 1}
    """

    SKILL_NAME = "smart_files"
    AGENT_NAME = "file-manager"
    DESCRIPTION = (
        "Smart File Management -- batch rename, auto-organize, "
        "classify, and merge files"
    )
    TAGS = ["files", "organize", "rename", "merge", "cowork"]
    VERSION = "1.0.0"

    def __init__(self, allow_outside_root: bool = False):
        self._invocation_count = 0
        self._created_at = datetime.now(timezone.utc).isoformat()
        self._allow_outside_root = allow_outside_root

    def register(self, router: Any) -> None:
        """Register this skill on a SkillRouter."""
        from core.skills.registry import (
            RegisteredSkill,
            SkillContext,
            SkillManifest,
            SkillStatus,
        )

        manifest = SkillManifest(
            name=self.SKILL_NAME,
            description=self.DESCRIPTION,
            version=self.VERSION,
            author="BIZRA Node0",
            context=SkillContext.INLINE,
            agent=self.AGENT_NAME,
            tags=self.TAGS,
            required_inputs=["operation"],
            optional_inputs=[
                "path",
                "paths",
                "target_path",
                "pattern",
                "prefix",
                "suffix",
                "hash_suffix",
                "dry_run",
                "recursive",
                "copy_mode",
                "category_filter",
                "output_path",
                "separator",
                "add_headers",
            ],
            outputs=["result"],
            ihsan_floor=UNIFIED_IHSAN_THRESHOLD,
        )

        skill = RegisteredSkill(
            manifest=manifest,
            path="core/skills/smart_file_manager.py",
            status=SkillStatus.AVAILABLE,
        )

        # Register in registry
        router.registry._skills[self.SKILL_NAME] = skill

        # Index by tag
        for tag in self.TAGS:
            tag_list = router.registry._by_tag.setdefault(tag, [])
            if self.SKILL_NAME not in tag_list:
                tag_list.append(self.SKILL_NAME)

        # Index by agent
        agent_list = router.registry._by_agent.setdefault(self.AGENT_NAME, [])
        if self.SKILL_NAME not in agent_list:
            agent_list.append(self.SKILL_NAME)

        # Register handler
        router.register_handler(self.AGENT_NAME, self._handle)

        logger.info(f"Smart Files skill '{self.SKILL_NAME}' registered on SkillRouter")

    async def _handle(
        self,
        skill: Any,
        inputs: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Async handler invoked by SkillRouter.invoke()."""
        operation = inputs.get("operation", "")
        self._invocation_count += 1

        dispatch = {
            "scan": self._op_scan,
            "organize": self._op_organize,
            "rename": self._op_rename,
            "merge": self._op_merge,
        }

        handler = dispatch.get(operation)
        if handler is None:
            return {
                "error": f"Unknown operation: '{operation}'",
                "available_operations": list(dispatch.keys()),
            }

        return handler(inputs)

    # -- scan ----------------------------------------------------------------

    def _op_scan(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Classify files in a directory by category."""
        path_str = inputs.get("path", "")
        if not path_str:
            return {"error": "Missing required input: path"}

        recursive = bool(inputs.get("recursive", False))
        category_filter = inputs.get("category_filter", "")

        try:
            target = _validate_path(path_str, self._allow_outside_root)
        except ValueError as exc:
            return {"error": str(exc)}

        if not target.is_dir():
            return {"error": f"Not a directory: {target}"}

        iterator = target.rglob("*") if recursive else target.iterdir()

        categories: Dict[str, int] = {}
        sizes: Dict[str, int] = {}
        total_size = 0
        total_files = 0
        top_files: List[Dict[str, Any]] = []

        for item in iterator:
            if not item.is_file():
                continue

            ext = item.suffix.lower()
            cat = EXT_TO_CATEGORY.get(ext, "other")

            if category_filter and cat != category_filter:
                continue

            try:
                fsize = item.stat().st_size
            except OSError:
                continue

            categories[cat] = categories.get(cat, 0) + 1
            sizes[cat] = sizes.get(cat, 0) + fsize
            total_size += fsize
            total_files += 1

            if len(top_files) < 20:
                top_files.append(
                    {
                        "name": item.name,
                        "category": cat,
                        "size": _size_human(fsize),
                        "size_bytes": fsize,
                    }
                )

        return {
            "operation": "scan",
            "path": str(target),
            "total_files": total_files,
            "total_size": _size_human(total_size),
            "total_size_bytes": total_size,
            "categories": {
                cat: {"count": categories[cat], "size": _size_human(sizes[cat])}
                for cat in sorted(categories, key=lambda c: categories[c], reverse=True)
            },
            "top_files": sorted(top_files, key=lambda f: f["size_bytes"], reverse=True)[
                :10
            ],
        }

    # -- organize ------------------------------------------------------------

    def _op_organize(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Move/copy files into category sub-folders."""
        path_str = inputs.get("path", "")
        if not path_str:
            return {"error": "Missing required input: path"}

        dry_run = inputs.get("dry_run", True)
        if isinstance(dry_run, str):
            dry_run = dry_run.lower() not in ("false", "0", "no")
        copy_mode = bool(inputs.get("copy_mode", False))
        target_path_str = inputs.get("target_path", "")

        try:
            source = _validate_path(path_str, self._allow_outside_root)
        except ValueError as exc:
            return {"error": str(exc)}

        if not source.is_dir():
            return {"error": f"Not a directory: {source}"}

        # Determine target root
        if target_path_str:
            target_root = Path(target_path_str).resolve()
            if not dry_run:
                target_root.mkdir(parents=True, exist_ok=True)
        else:
            target_root = source

        planned_moves: List[Dict[str, str]] = []
        category_counts: Dict[str, int] = {}

        for item in source.iterdir():
            if not item.is_file():
                continue

            ext = item.suffix.lower()
            cat = EXT_TO_CATEGORY.get(ext, "other")

            dest_dir = target_root / cat
            dest_file = dest_dir / item.name

            planned_moves.append(
                {
                    "source": str(item),
                    "destination": str(dest_file),
                    "category": cat,
                }
            )
            category_counts[cat] = category_counts.get(cat, 0) + 1

        executed = 0
        errors: List[str] = []

        if not dry_run:
            for move in planned_moves:
                try:
                    dest = Path(move["destination"])
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    src = Path(move["source"])
                    if copy_mode:
                        shutil.copy2(str(src), str(dest))
                    else:
                        shutil.move(str(src), str(dest))
                    executed += 1
                except Exception as exc:
                    errors.append(f"{move['source']}: {exc}")

        action = "copied" if copy_mode else "moved"
        return {
            "operation": "organize",
            "dry_run": dry_run,
            "action": action,
            "planned": len(planned_moves),
            "executed": executed,
            "errors": errors,
            "category_breakdown": category_counts,
            "moves": planned_moves[:50],
        }

    # -- rename --------------------------------------------------------------

    def _op_rename(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Batch rename files with pattern + hash."""
        path_str = inputs.get("path", "")
        if not path_str:
            return {"error": "Missing required input: path"}

        dry_run = inputs.get("dry_run", True)
        if isinstance(dry_run, str):
            dry_run = dry_run.lower() not in ("false", "0", "no")

        pattern = inputs.get("pattern", "{name}{ext}")
        prefix = inputs.get("prefix", "")
        suffix = inputs.get("suffix", "")
        hash_suffix = bool(inputs.get("hash_suffix", False))

        try:
            target = _validate_path(path_str, self._allow_outside_root)
        except ValueError as exc:
            return {"error": str(exc)}

        if not target.is_dir():
            return {"error": f"Not a directory: {target}"}

        renames: List[Dict[str, str]] = []
        n = 1

        for item in sorted(target.iterdir()):
            if not item.is_file():
                continue

            ext = item.suffix.lower()
            stem = item.stem
            cat = EXT_TO_CATEGORY.get(ext, "other")
            mod_date = _file_mod_date(item)

            content_hash = ""
            if hash_suffix or "{hash}" in pattern:
                content_hash = _file_hash(item)

            new_name = _expand_pattern(
                pattern,
                name=stem,
                ext=ext,
                n=n,
                content_hash=content_hash,
                date=mod_date,
                cat=cat,
            )

            if prefix:
                new_name = prefix + new_name
            if suffix:
                base, fext = os.path.splitext(new_name)
                new_name = base + suffix + fext
            if hash_suffix and "{hash}" not in pattern:
                base, fext = os.path.splitext(new_name)
                new_name = f"{base}_{content_hash[:16]}{fext}"

            renames.append(
                {
                    "old": item.name,
                    "new": new_name,
                }
            )
            n += 1

        executed = 0
        errors: List[str] = []

        if not dry_run:
            for entry in renames:
                old_path = target / entry["old"]
                new_path = target / entry["new"]
                try:
                    old_path.rename(new_path)
                    executed += 1
                except Exception as exc:
                    errors.append(f"{entry['old']}: {exc}")

        return {
            "operation": "rename",
            "dry_run": dry_run,
            "planned": len(renames),
            "executed": executed,
            "errors": errors,
            "renames": renames[:100],
        }

    # -- merge ---------------------------------------------------------------

    def _op_merge(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Concatenate text files into one."""
        paths = inputs.get("paths", [])
        if not paths or not isinstance(paths, list):
            return {"error": "Missing required input: paths (list of file paths)"}

        output_path_str = inputs.get("output_path", "")
        separator = inputs.get("separator", "\n")
        add_headers = bool(inputs.get("add_headers", True))

        resolved_paths: List[Path] = []
        for p in paths:
            try:
                rp = _validate_path(str(p), self._allow_outside_root)
                if not rp.is_file():
                    return {"error": f"Not a file: {rp}"}
                resolved_paths.append(rp)
            except ValueError as exc:
                return {"error": str(exc)}

        if not output_path_str:
            first_parent = resolved_paths[0].parent
            output_path = first_parent / f"merged_{uuid.uuid4().hex[:8]}.txt"
        else:
            output_path = Path(output_path_str).resolve()

        parts: List[str] = []
        total_size = 0

        for rp in resolved_paths:
            try:
                content = rp.read_text(encoding="utf-8", errors="replace")
            except Exception as exc:
                return {"error": f"Cannot read {rp}: {exc}"}

            if add_headers:
                parts.append(f"# === {rp.name} ===")
            parts.append(content)
            total_size += rp.stat().st_size

        merged = separator.join(parts)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(merged, encoding="utf-8")

        return {
            "operation": "merge",
            "files_merged": len(resolved_paths),
            "output_path": str(output_path),
            "total_input_size": _size_human(total_size),
            "output_size": _size_human(len(merged.encode("utf-8"))),
        }


# ---------------------------------------------------------------------------
# Module-level convenience
# ---------------------------------------------------------------------------

_default_handler: Optional[SmartFileHandler] = None


def get_smart_file_handler() -> SmartFileHandler:
    """Get the singleton SmartFileHandler."""
    global _default_handler
    if _default_handler is None:
        _default_handler = SmartFileHandler()
    return _default_handler


def register_smart_files(router: Any) -> SmartFileHandler:
    """Register the smart_files skill on a SkillRouter. Returns the handler."""
    handler = get_smart_file_handler()
    handler.register(router)
    return handler


__all__ = [
    "SmartFileHandler",
    "get_smart_file_handler",
    "register_smart_files",
]
