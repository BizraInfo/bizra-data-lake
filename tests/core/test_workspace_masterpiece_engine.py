from pathlib import Path

from scripts.atlas.workspace_masterpiece_engine import (
    analyze_inventory,
    classify_path,
)


def test_classify_path_runtime_and_build_and_code():
    assert classify_path("./.pytest_cache/v/cache/nodeids") == "runtime_state"
    assert classify_path("./bizra-omega/target/debug/app") == "build_artifacts"
    assert classify_path("./core/federation/secure_transport.py") == "code"


def test_classify_path_governance_docs_and_data():
    assert classify_path("./.github/workflows/ci.yml") == "governance"
    assert classify_path("./README.md") == "governance"
    assert classify_path("./docs/specs/phase_56_security_hardening.md") == "governance"
    assert classify_path("./research_archive/paper.pdf") == "data_assets"


def test_analyze_inventory_small_manifest(tmp_path: Path):
    files = tmp_path / "files.txt"
    dirs = tmp_path / "dirs.txt"
    files.write_text(
        "\n".join(
            [
                "./core/a.py",
                "./README.md",
                "./.pytest_cache/v/cache/nodeids",
                "./data/sample.csv",
                "./mystery.weird",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    dirs.write_text("./core\n./data\n", encoding="utf-8")

    report = analyze_inventory(files_manifest=files, dirs_manifest=dirs, top_n=5)
    assert report["inventory"]["total_files"] == 5
    assert report["inventory"]["total_dirs"] == 2
    assert report["global"]["lens_counts"]["code"] >= 1
    assert report["global"]["lens_counts"]["governance"] >= 1
    assert report["global"]["lens_counts"]["runtime_state"] >= 1
    assert report["global"]["lens_counts"]["data_assets"] >= 1
    assert report["global"]["lens_counts"]["unknown"] >= 1
    assert "unknown_extensions" in report["rankings"]
