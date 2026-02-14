"""
Tests for Smart File Management Skill
======================================

Covers all 4 operations (scan, organize, rename, merge), dry-run safety,
path traversal blocking, and skill registration.

Uses tmp_path fixture for isolated filesystem tests -- no real files touched.
"""

import asyncio
import os
import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.skills.smart_file_manager import (
    SmartFileHandler,
    _expand_pattern,
    _size_human,
    _validate_path,
    register_smart_files,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def handler():
    """Handler with allow_outside_root=True for tmp_path testing."""
    return SmartFileHandler(allow_outside_root=True)


@pytest.fixture
def populated_dir(tmp_path):
    """Directory with sample files of various types."""
    (tmp_path / "report.pdf").write_text("pdf content")
    (tmp_path / "photo.jpg").write_bytes(b"\xff\xd8sample image")
    (tmp_path / "script.py").write_text("print('hello')")
    (tmp_path / "data.csv").write_text("a,b,c\n1,2,3")
    (tmp_path / "notes.txt").write_text("some notes here")
    (tmp_path / "archive.zip").write_bytes(b"PK\x03\x04fake zip")
    (tmp_path / "unknown.xyz").write_text("mystery file")
    return tmp_path


@pytest.fixture
def text_files(tmp_path):
    """Directory with text files for merge testing."""
    (tmp_path / "part1.txt").write_text("First part content.")
    (tmp_path / "part2.txt").write_text("Second part content.")
    (tmp_path / "part3.txt").write_text("Third part content.")
    return tmp_path


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestExpandPattern:
    def test_all_tokens(self):
        result = _expand_pattern(
            "{name}_{date}_{n}_{hash}_{cat}{ext}",
            name="report",
            ext=".pdf",
            n=5,
            content_hash="abcdef1234567890abcdef",
            date="2026-02-13",
            cat="documents",
        )
        assert result == "report_2026-02-13_5_abcdef1234567890_documents.pdf"

    def test_no_tokens(self):
        result = _expand_pattern(
            "fixed_name.txt",
            name="x", ext=".y", n=1, content_hash="h", date="d", cat="c",
        )
        assert result == "fixed_name.txt"

    def test_name_ext_only(self):
        result = _expand_pattern(
            "{name}{ext}",
            name="file", ext=".md", n=0, content_hash="", date="", cat="",
        )
        assert result == "file.md"


class TestSizeHuman:
    def test_bytes(self):
        assert _size_human(500) == "500.0 B"

    def test_kilobytes(self):
        assert _size_human(2048) == "2.0 KB"

    def test_megabytes(self):
        assert _size_human(5 * 1024 * 1024) == "5.0 MB"

    def test_zero(self):
        assert _size_human(0) == "0.0 B"


# ═══════════════════════════════════════════════════════════════════════════════
# SCAN TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestScan:
    def test_scan_empty_dir(self, handler, tmp_path):
        result = handler._op_scan({"path": str(tmp_path)})
        assert result["total_files"] == 0
        assert result["categories"] == {}
        assert result["operation"] == "scan"

    def test_scan_categorizes_files(self, handler, populated_dir):
        result = handler._op_scan({"path": str(populated_dir)})
        assert result["total_files"] == 7
        cats = result["categories"]
        assert cats["documents"]["count"] == 2  # pdf + txt
        assert cats["images"]["count"] == 1     # jpg
        assert cats["code"]["count"] == 1       # py
        assert cats["data"]["count"] == 1       # csv
        assert cats["archives"]["count"] == 1   # zip
        assert cats["other"]["count"] == 1      # xyz

    def test_scan_with_filter(self, handler, populated_dir):
        result = handler._op_scan({
            "path": str(populated_dir),
            "category_filter": "code",
        })
        assert result["total_files"] == 1
        assert "code" in result["categories"]
        assert len(result["categories"]) == 1

    def test_scan_recursive(self, handler, tmp_path):
        sub = tmp_path / "subdir"
        sub.mkdir()
        (tmp_path / "root.py").write_text("root")
        (sub / "nested.py").write_text("nested")

        non_recursive = handler._op_scan({"path": str(tmp_path), "recursive": False})
        recursive = handler._op_scan({"path": str(tmp_path), "recursive": True})

        assert non_recursive["total_files"] == 1
        assert recursive["total_files"] == 2

    def test_scan_missing_path(self, handler):
        result = handler._op_scan({})
        assert "error" in result

    def test_scan_nonexistent_path(self, handler):
        result = handler._op_scan({"path": "/nonexistent/path/xyz"})
        assert "error" in result

    def test_scan_not_a_directory(self, handler, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        result = handler._op_scan({"path": str(f)})
        assert "error" in result
        assert "Not a directory" in result["error"]


# ═══════════════════════════════════════════════════════════════════════════════
# ORGANIZE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestOrganize:
    def test_organize_dry_run(self, handler, populated_dir):
        result = handler._op_organize({"path": str(populated_dir)})
        assert result["dry_run"] is True
        assert result["planned"] == 7
        assert result["executed"] == 0
        # Files should NOT have moved
        assert (populated_dir / "report.pdf").exists()

    def test_organize_execute(self, handler, populated_dir):
        result = handler._op_organize({
            "path": str(populated_dir),
            "dry_run": False,
        })
        assert result["dry_run"] is False
        assert result["executed"] == 7
        assert (populated_dir / "documents" / "report.pdf").exists()
        assert (populated_dir / "images" / "photo.jpg").exists()
        assert (populated_dir / "code" / "script.py").exists()
        assert not (populated_dir / "report.pdf").exists()

    def test_organize_copy_mode(self, handler, populated_dir):
        result = handler._op_organize({
            "path": str(populated_dir),
            "dry_run": False,
            "copy_mode": True,
        })
        assert result["action"] == "copied"
        assert result["executed"] == 7
        # Originals still exist
        assert (populated_dir / "report.pdf").exists()
        # Copies also exist
        assert (populated_dir / "documents" / "report.pdf").exists()

    def test_organize_with_target_path(self, handler, populated_dir, tmp_path):
        target = tmp_path / "organized_output"
        result = handler._op_organize({
            "path": str(populated_dir),
            "target_path": str(target),
            "dry_run": False,
        })
        assert result["executed"] == 7
        assert (target / "documents" / "report.pdf").exists()

    def test_organize_category_breakdown(self, handler, populated_dir):
        result = handler._op_organize({"path": str(populated_dir)})
        breakdown = result["category_breakdown"]
        assert breakdown["documents"] == 2
        assert breakdown["code"] == 1

    def test_organize_missing_path(self, handler):
        result = handler._op_organize({})
        assert "error" in result


# ═══════════════════════════════════════════════════════════════════════════════
# RENAME TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestRename:
    def test_rename_dry_run(self, handler, populated_dir):
        result = handler._op_rename({"path": str(populated_dir)})
        assert result["dry_run"] is True
        assert result["planned"] > 0
        assert result["executed"] == 0
        # Files unchanged
        assert (populated_dir / "report.pdf").exists()

    def test_rename_with_pattern(self, handler, tmp_path):
        (tmp_path / "doc.pdf").write_text("content")
        result = handler._op_rename({
            "path": str(tmp_path),
            "pattern": "{name}_{date}{ext}",
            "dry_run": True,
        })
        assert result["planned"] == 1
        rename = result["renames"][0]
        assert rename["old"] == "doc.pdf"
        # Should contain date pattern YYYY-MM-DD
        assert rename["new"].startswith("doc_")
        assert rename["new"].endswith(".pdf")

    def test_rename_hash_suffix(self, handler, tmp_path):
        (tmp_path / "test.txt").write_text("hello world")
        result = handler._op_rename({
            "path": str(tmp_path),
            "hash_suffix": True,
            "dry_run": True,
        })
        rename = result["renames"][0]
        # Should have hash suffix: test_<16chars>.txt
        assert "_" in rename["new"]
        assert len(rename["new"]) > len("test.txt")

    def test_rename_execute(self, handler, tmp_path):
        (tmp_path / "a.txt").write_text("aaa")
        (tmp_path / "b.txt").write_text("bbb")
        result = handler._op_rename({
            "path": str(tmp_path),
            "pattern": "file_{n}{ext}",
            "dry_run": False,
        })
        assert result["executed"] == 2
        assert (tmp_path / "file_1.txt").exists()
        assert (tmp_path / "file_2.txt").exists()

    def test_rename_prefix_suffix(self, handler, tmp_path):
        (tmp_path / "doc.pdf").write_text("content")
        result = handler._op_rename({
            "path": str(tmp_path),
            "prefix": "bizra_",
            "suffix": "_v2",
            "dry_run": True,
        })
        rename = result["renames"][0]
        assert rename["new"].startswith("bizra_")
        assert "_v2" in rename["new"]

    def test_rename_missing_path(self, handler):
        result = handler._op_rename({})
        assert "error" in result


# ═══════════════════════════════════════════════════════════════════════════════
# MERGE TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestMerge:
    def test_merge_files(self, handler, text_files):
        paths = [
            str(text_files / "part1.txt"),
            str(text_files / "part2.txt"),
            str(text_files / "part3.txt"),
        ]
        result = handler._op_merge({"paths": paths})
        assert result["files_merged"] == 3
        output = Path(result["output_path"])
        assert output.exists()
        content = output.read_text()
        assert "First part content." in content
        assert "Second part content." in content
        assert "Third part content." in content
        # Default add_headers=True
        assert "# === part1.txt ===" in content

    def test_merge_with_separator(self, handler, text_files):
        paths = [
            str(text_files / "part1.txt"),
            str(text_files / "part2.txt"),
        ]
        result = handler._op_merge({
            "paths": paths,
            "separator": "\n---\n",
        })
        output = Path(result["output_path"])
        content = output.read_text()
        assert "\n---\n" in content

    def test_merge_without_headers(self, handler, text_files):
        paths = [
            str(text_files / "part1.txt"),
            str(text_files / "part2.txt"),
        ]
        result = handler._op_merge({
            "paths": paths,
            "add_headers": False,
        })
        output = Path(result["output_path"])
        content = output.read_text()
        assert "# ===" not in content

    def test_merge_custom_output(self, handler, text_files, tmp_path):
        out_file = tmp_path / "custom_merged.md"
        paths = [str(text_files / "part1.txt"), str(text_files / "part2.txt")]
        result = handler._op_merge({
            "paths": paths,
            "output_path": str(out_file),
        })
        assert result["output_path"] == str(out_file)
        assert out_file.exists()

    def test_merge_missing_paths(self, handler):
        result = handler._op_merge({})
        assert "error" in result

    def test_merge_empty_list(self, handler):
        result = handler._op_merge({"paths": []})
        assert "error" in result

    def test_merge_nonexistent_file(self, handler):
        result = handler._op_merge({"paths": ["/nonexistent/file.txt"]})
        assert "error" in result


# ═══════════════════════════════════════════════════════════════════════════════
# PATH TRAVERSAL TESTS
# ═══════════════════════════════════════════════════════════════════════════════


class TestPathTraversal:
    def test_path_traversal_blocked(self, tmp_path):
        """Handler without allow_outside_root blocks paths outside DATA_LAKE_ROOT."""
        handler = SmartFileHandler(allow_outside_root=False)
        result = handler._op_scan({"path": str(tmp_path)})
        # tmp_path is outside DATA_LAKE_ROOT, should be blocked
        assert "error" in result
        assert "outside" in result["error"].lower() or "traversal" in result["error"].lower()


# ═══════════════════════════════════════════════════════════════════════════════
# UNKNOWN OPERATION TEST
# ═══════════════════════════════════════════════════════════════════════════════


class TestUnknownOperation:
    def test_unknown_operation(self, handler):
        loop = asyncio.new_event_loop()
        try:
            result = loop.run_until_complete(
                handler._handle(None, {"operation": "delete_everything"})
            )
        finally:
            loop.close()
        assert "error" in result
        assert "available_operations" in result
        assert "scan" in result["available_operations"]
        assert "organize" in result["available_operations"]
        assert "rename" in result["available_operations"]
        assert "merge" in result["available_operations"]


# ═══════════════════════════════════════════════════════════════════════════════
# REGISTRATION TEST
# ═══════════════════════════════════════════════════════════════════════════════


class TestRegistration:
    def test_registration(self):
        """Handler registers on a mock router correctly."""
        from core.skills.registry import SkillRegistry
        from core.skills.mcp_bridge import MCPBridge
        from core.skills.router import SkillRouter

        registry = SkillRegistry(skills_dir="/nonexistent")
        router = SkillRouter(registry=registry, mcp_bridge=MCPBridge())

        handler = SmartFileHandler()
        handler.register(router)

        # Skill in registry
        skill = router.registry.get("smart_files")
        assert skill is not None
        assert skill.manifest.name == "smart_files"
        assert skill.manifest.agent == "file-manager"
        assert "cowork" in skill.manifest.tags

        # Handler registered
        assert "file-manager" in router._handlers

    def test_register_convenience_function(self):
        """register_smart_files() works end-to-end."""
        from core.skills.registry import SkillRegistry
        from core.skills.mcp_bridge import MCPBridge
        from core.skills.router import SkillRouter

        registry = SkillRegistry(skills_dir="/nonexistent")
        router = SkillRouter(registry=registry, mcp_bridge=MCPBridge())

        with patch(
            "core.skills.smart_file_manager._default_handler", None
        ):
            handler = register_smart_files(router)

        assert isinstance(handler, SmartFileHandler)
        assert router.registry.get("smart_files") is not None
