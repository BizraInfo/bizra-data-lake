from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

from core.integration import constants as canonical
from scripts import ci_proof_pyramid_gate as gate


REPO = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO / "scripts" / "ci_proof_pyramid_gate.py"
THRESHOLD_NAMES = {
    "IHSAN_THRESHOLD",
    "SNR_THRESHOLD",
    "ADL_GINI_MAX",
    "MAX_HARM_SCORE",
    "MIN_CONFIDENCE",
}


def test_proof_pyramid_gate_threshold_values_match_canonical_constants() -> None:
    assert gate.IHSAN_THRESHOLD == canonical.IHSAN_THRESHOLD
    assert gate.SNR_THRESHOLD == canonical.SNR_THRESHOLD
    assert gate.ADL_GINI_MAX == canonical.ADL_GINI_THRESHOLD
    assert gate.MAX_HARM_SCORE == canonical.MAX_HARM_SCORE
    assert gate.MIN_CONFIDENCE == canonical.MIN_CONFIDENCE


def test_proof_pyramid_gate_does_not_hardcode_constitutional_thresholds() -> None:
    tree = ast.parse(SCRIPT_PATH.read_text(encoding="utf-8"))
    hardcoded: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in THRESHOLD_NAMES and _is_numeric_literal(node.value):
                hardcoded.append(node.target.id)
        elif isinstance(node, ast.Assign) and _is_numeric_literal(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in THRESHOLD_NAMES:
                    hardcoded.append(target.id)

    assert hardcoded == []


def test_proof_pyramid_gate_cli_resolves_canonical_constants_when_run_as_file(
    tmp_path: Path,
) -> None:
    output = tmp_path / "evidence.json"
    result = subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            "--evidence-dir",
            str(tmp_path / "missing-evidence"),
            "--gate-results",
            "pp001=success",
            "pp002=success",
            "pp003=success",
            "pp004=success",
            "pp005=success",
            "pp006=success",
            "--output",
            str(output),
        ],
        cwd=REPO,
        capture_output=True,
        check=False,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    evidence = json.loads(output.read_text(encoding="utf-8"))
    assert evidence["constitutional_thresholds"] == {
        "IHSAN_THRESHOLD": canonical.IHSAN_THRESHOLD,
        "SNR_THRESHOLD": canonical.SNR_THRESHOLD,
        "ADL_GINI_MAX": canonical.ADL_GINI_THRESHOLD,
        "MAX_HARM_SCORE": canonical.MAX_HARM_SCORE,
        "MIN_CONFIDENCE": canonical.MIN_CONFIDENCE,
    }


def _is_numeric_literal(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
