from __future__ import annotations

import ast
import importlib.util
import sys
from pathlib import Path
from types import ModuleType

from core.integration import constants as canonical


REPO = Path(__file__).resolve().parents[3]
MIRRORS: dict[str, dict[str, float]] = {
    "runtime/core/constants.py": {
        "IHSAN_THRESHOLD": canonical.IHSAN_THRESHOLD,
    },
    "bizra-node0/core/integration/constants.py": {
        "IHSAN_THRESHOLD": canonical.IHSAN_THRESHOLD,
        "SNR_THRESHOLD": canonical.SNR_THRESHOLD,
        "ADL_GINI_THRESHOLD": canonical.ADL_GINI_THRESHOLD,
        "ADL_HARBERGER_TAX_RATE": canonical.ADL_HARBERGER_TAX_RATE,
        "MIN_CONFIDENCE": canonical.MIN_CONFIDENCE,
        "MAX_HARM_SCORE": canonical.MAX_HARM_SCORE,
    },
}


def test_python_constant_mirrors_match_canonical_tier1_values() -> None:
    for relative_path, expected in MIRRORS.items():
        module = _load_module(REPO / relative_path)
        for name, canonical_value in expected.items():
            assert getattr(module, name) == canonical_value, f"{relative_path}:{name}"


def test_python_constant_mirrors_do_not_hardcode_tier1_literals() -> None:
    hardcoded: dict[str, list[str]] = {}

    for relative_path, expected in MIRRORS.items():
        path = REPO / relative_path
        names = set(expected)
        found = _numeric_literal_assignments(path, names)
        if found:
            hardcoded[relative_path] = found

    assert hardcoded == {}


def _load_module(path: Path) -> ModuleType:
    module_name = f"_cross_lang_python_mirror_{path.stem}_{abs(hash(path))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def _numeric_literal_assignments(path: Path, names: set[str]) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    hardcoded: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in names and _is_numeric_literal(node.value):
                hardcoded.append(node.target.id)
        elif isinstance(node, ast.Assign) and _is_numeric_literal(node.value):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    hardcoded.append(target.id)

    return hardcoded


def _is_numeric_literal(node: ast.AST | None) -> bool:
    return isinstance(node, ast.Constant) and isinstance(node.value, (int, float))
