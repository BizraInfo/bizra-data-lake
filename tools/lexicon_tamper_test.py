#!/usr/bin/env python3
from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path


def run_lint(lexicon: Path, baseline: Path) -> int:
    return subprocess.run(
        [
            sys.executable,
            str(Path(__file__).resolve().parents[0] / "lexicon_lint.py"),
            "--lexicon",
            str(lexicon),
            "--baseline",
            str(baseline),
        ],
        check=False,
    ).returncode


def main() -> int:
    repo_root = Path(__file__).resolve().parents[1]
    source = repo_root / "constitution" / "lexicon_v1.yaml"
    if not source.exists():
        print(f"lexicon tamper test skipped: missing {source}")
        return 0

    original = source.read_text(encoding="utf-8", errors="replace")

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        baseline = td_path / "baseline.yaml"
        baseline.write_text(original, encoding="utf-8")

        tampered_modify = td_path / "tampered_modify.yaml"
        needle = 'role: "Generation/execution team producing candidate outputs"'
        if needle not in original:
            print("lexicon tamper test failed: expected PAT role line not found in lexicon")
            return 1
        tampered_modify.write_text(
            original.replace(needle, 'role: "TAMPERED: should fail append-only"'),
            encoding="utf-8",
        )

        rc1 = run_lint(tampered_modify, baseline)
        if rc1 == 0:
            print("lexicon tamper test failed: modified term was not rejected")
            return 1

        tampered_remove = td_path / "tampered_remove.yaml"
        tampered_remove.write_text(
            original.replace("\n  PAT:\n", "\n  PATT:\n"),
            encoding="utf-8",
        )

        rc2 = run_lint(tampered_remove, baseline)
        if rc2 == 0:
            print("lexicon tamper test failed: removed/renamed term was not rejected")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

