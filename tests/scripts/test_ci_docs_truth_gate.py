from __future__ import annotations

from pathlib import Path

from scripts.ci_docs_truth_gate import _check_blueprint_truth, _check_readme_thresholds


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _seed_minimal_repo(root: Path) -> None:
    _write(
        root / "README.md",
        "\n".join(
            [
                "| Metric | Value |",
                "|---|---|",
                "| ADL Gini | <= 0.35 |",
                "| Ihsan | >= 0.95 |",
                "| SNR | >= 0.85 |",
                "High-performance core (2 Rust crates)",
            ]
        ),
    )
    _write(
        root / "core" / "integration" / "constants.py",
        "\n".join(
            [
                "from typing import Final",
                "ADL_GINI_THRESHOLD: Final[float] = 0.35",
                "UNIFIED_IHSAN_THRESHOLD: Final[float] = 0.95",
                "UNIFIED_SNR_THRESHOLD: Final[float] = 0.85",
            ]
        ),
    )
    _write(
        root / "bizra-omega" / "Cargo.toml",
        "\n".join(
            [
                "[workspace]",
                "members = [",
                '  "crate-a",',
                '  "crate-b",',
                "]",
            ]
        ),
    )


def test_docs_truth_gate_passes_for_consistent_repo(tmp_path: Path) -> None:
    _seed_minimal_repo(tmp_path)
    _write(
        tmp_path / "docs" / "UNIFIED_BLUEPRINT" / "01_ALPHA.md",
        "\n".join(
            [
                "**Status:** [x] BUILT",
                "**Status:** [~] PARTIAL",
                "| **TOTAL** | **1/2 + 1P + 0N** | **75%** |",
            ]
        ),
    )
    _write(
        tmp_path / "docs" / "UNIFIED_BLUEPRINT" / "02_BETA.md",
        "\n".join(
            [
                "**Status:** [x] BUILT",
                "**Status:** [ ] NOT BUILT",
                "| **TOTAL** | **1/2 + 0P + 1N** | **50%** |",
            ]
        ),
    )
    _write(
        tmp_path / "docs" / "UNIFIED_BLUEPRINT" / "00_MASTER_INDEX.md",
        "\n".join(
            [
                "# Master Index",
                "## Completion Summary",
                "| Domain | Built | Partial | Not Built | Coverage |",
                "|---|---|---|---|---|",
                "| Alpha | 1/2 | 1 | 0 | 75% |",
                "| Beta | 1/2 | 0 | 1 | 50% |",
                "| **TOTAL** | **2/4** | **1** | **1** | **62%** |",
            ]
        ),
    )

    assert _check_readme_thresholds(tmp_path) == []
    assert _check_blueprint_truth(tmp_path, expected_module_count=2) == []


def test_docs_truth_gate_reports_module_total_drift(tmp_path: Path) -> None:
    _seed_minimal_repo(tmp_path)
    _write(
        tmp_path / "docs" / "UNIFIED_BLUEPRINT" / "01_ALPHA.md",
        "\n".join(
            [
                "**Status:** [x] BUILT",
                "**Status:** [~] PARTIAL",
                "| **TOTAL** | **2/2 + 0P + 0N** | **100%** |",
            ]
        ),
    )
    _write(
        tmp_path / "docs" / "UNIFIED_BLUEPRINT" / "00_MASTER_INDEX.md",
        "\n".join(
            [
                "# Master Index",
                "## Completion Summary",
                "| Domain | Built | Partial | Not Built | Coverage |",
                "|---|---|---|---|---|",
                "| Alpha | 1/2 | 1 | 0 | 75% |",
                "| **TOTAL** | **1/2** | **1** | **0** | **75%** |",
            ]
        ),
    )

    issues = _check_blueprint_truth(tmp_path, expected_module_count=1)

    assert any("01_ALPHA.md total drift" in issue for issue in issues)


def test_docs_truth_gate_reports_master_total_drift(tmp_path: Path) -> None:
    _seed_minimal_repo(tmp_path)
    _write(
        tmp_path / "docs" / "UNIFIED_BLUEPRINT" / "01_ALPHA.md",
        "\n".join(
            [
                "**Status:** [x] BUILT",
                "| **TOTAL** | **1/1** | **100%** |",
            ]
        ),
    )
    _write(
        tmp_path / "docs" / "UNIFIED_BLUEPRINT" / "02_BETA.md",
        "\n".join(
            [
                "**Status:** [ ] NOT BUILT",
                "| **TOTAL** | **0/1 + 0P + 1N** | **0%** |",
            ]
        ),
    )
    _write(
        tmp_path / "docs" / "UNIFIED_BLUEPRINT" / "00_MASTER_INDEX.md",
        "\n".join(
            [
                "# Master Index",
                "## Completion Summary",
                "| Domain | Built | Partial | Not Built | Coverage |",
                "|---|---|---|---|---|",
                "| Alpha | 1/1 | 0 | 0 | 100% |",
                "| Beta | 0/1 | 0 | 1 | 0% |",
                "| **TOTAL** | **2/2** | **0** | **0** | **100%** |",
            ]
        ),
    )

    issues = _check_blueprint_truth(tmp_path, expected_module_count=2)

    assert any("Master index TOTAL drift" in issue for issue in issues)
