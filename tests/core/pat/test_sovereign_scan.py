from core.pat import sovereign_scan
from core.pat.sovereign_scan import (
    ScanAudit,
    ScanLimits,
    run_discovery_scan,
    scan_directory,
)


def test_scan_directory_respects_max_depth(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    (nested / "kept_out.py").write_text("print('hidden')", encoding="utf-8")
    audit = ScanAudit()

    results = scan_directory(
        tmp_path,
        limits=ScanLimits(max_depth=0),
        audit=audit,
    )

    assert results == []
    assert audit.as_dict()["limit_hit"] is True
    assert any(skipped["reason"] == "max_depth" for skipped in audit.skipped)


def test_scan_directory_respects_total_byte_budget(tmp_path):
    (tmp_path / "large.py").write_text("x" * 32, encoding="utf-8")
    audit = ScanAudit()

    results = scan_directory(
        tmp_path,
        limits=ScanLimits(max_total_bytes=8),
        audit=audit,
    )

    assert results == []
    assert audit.as_dict()["limit_hit"] is True
    assert any(skipped["reason"] == "max_total_bytes" for skipped in audit.skipped)


def test_scan_directory_skips_symlinks_by_default(tmp_path):
    target = tmp_path / "target.py"
    target.write_text("print('target')", encoding="utf-8")
    link = tmp_path / "link.py"
    link.symlink_to(target)
    audit = ScanAudit()

    results = scan_directory(tmp_path, audit=audit)

    paths = {result["path"] for result in results}
    assert str(target) in paths
    assert str(link) not in paths
    assert any(skipped["reason"] == "symlink_file" for skipped in audit.skipped)


def test_run_discovery_scan_respects_aggregate_max_files(tmp_path, monkeypatch):
    root_one = tmp_path / "one"
    root_two = tmp_path / "two"
    root_one.mkdir()
    root_two.mkdir()
    (root_one / "first.py").write_text("print('one')", encoding="utf-8")
    (root_two / "second.py").write_text("print('two')", encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    monkeypatch.setattr(sovereign_scan, "SCAN_ROOTS", [root_one, root_two])

    manifest = run_discovery_scan(
        output_dir=output_dir,
        limits=ScanLimits(max_files=1),
    )

    assert manifest["total_files"] == 1
    assert manifest["scan_audit"]["files_checked"] == 1
    assert any(
        skipped["reason"] == "max_files"
        for skipped in manifest["scan_audit"]["skipped"]
    )


def test_run_discovery_scan_uses_global_timeout(tmp_path, monkeypatch):
    root = tmp_path / "root"
    root.mkdir()
    (root / "file.py").write_text("print('late')", encoding="utf-8")
    output_dir = tmp_path / "out"
    output_dir.mkdir()
    monkeypatch.setattr(sovereign_scan, "SCAN_ROOTS", [root])

    manifest = run_discovery_scan(
        output_dir=output_dir,
        limits=ScanLimits(timeout_seconds=0.0),
    )

    assert manifest["total_files"] == 0
    assert any(
        skipped["reason"] == "timeout" for skipped in manifest["scan_audit"]["skipped"]
    )
