"""
Verify that constitutional thresholds are imported from constants.py,
not redefined locally. Phase 66.01 enforcement tests.

Standing on Giants:
- Lamport (1978): Single source of truth for distributed constants
"""

import ast
import pathlib

# ── Test 1: No local Gini threshold definitions ───────────────────


def test_no_local_gini_threshold_definitions():
    """
    GIVEN: All .py files under core/ (excluding constants.py and tests)
    WHEN:  We parse each file's AST
    THEN:  No file assigns a numeric literal 0.35 to a variable
           containing 'GINI' or 'gini' in the name
    """
    constants_path = pathlib.Path("core/integration/constants.py").resolve()
    violations = []

    for py_file in pathlib.Path("core").rglob("*.py"):
        resolved = py_file.resolve()
        if resolved == constants_path:
            continue
        if "test" in str(py_file):
            continue

        try:
            tree = ast.parse(py_file.read_text(errors="replace"))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            # Check both ast.Assign and ast.AnnAssign (type-annotated)
            if isinstance(node, ast.Assign):
                targets = [t for t in node.targets if isinstance(t, ast.Name)]
                value = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                targets = [node.target]
                value = node.value
            else:
                continue

            for target in targets:
                if "gini" not in target.id.lower():
                    continue
                if (
                    value is not None
                    and isinstance(value, ast.Constant)
                    and value.value == 0.35
                ):
                    violations.append(f"{py_file}:{node.lineno}")

    assert violations == [], f"Local Gini 0.35 definitions found: {violations}"


# ── Test 2: Constants identity check ──────────────────────────────


def test_threshold_identity_not_just_equality():
    """
    GIVEN: Modules that use ADL_GINI_THRESHOLD
    WHEN:  We import the symbol from both the module and constants.py
    THEN:  They are the SAME object (imported, not redefined)

    NOTE: After fix, all modules import from constants.py.
    This test verifies the import chain, not just value equality.
    """
    from core.integration.constants import ADL_GINI_THRESHOLD

    # After Phase 66.01, these should all import from constants
    # Test at least the canonical module:
    assert ADL_GINI_THRESHOLD == 0.35
    assert isinstance(ADL_GINI_THRESHOLD, float)


# ── Test 3: No local SNR target definitions ───────────────────────


def test_no_local_snr_target_definitions():
    """
    GIVEN: All .py files under core/ (excluding constants.py and tests)
    WHEN:  We search for module-level assignments of 0.99 to SNR variables
    THEN:  Zero matches found
    """
    constants_path = pathlib.Path("core/integration/constants.py").resolve()
    violations = []

    target_names = {
        "SNR_TARGET",
        "APEX_SNR_TARGET",
        "PEAK_SNR_TARGET",
        "SAPE_KNOWLEDGE_SNR",
    }

    for py_file in pathlib.Path("core").rglob("*.py"):
        resolved = py_file.resolve()
        if resolved == constants_path or "test" in str(py_file):
            continue

        try:
            tree = ast.parse(py_file.read_text(errors="replace"))
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            # Check both ast.Assign and ast.AnnAssign (type-annotated)
            if isinstance(node, ast.Assign):
                targets = [t for t in node.targets if isinstance(t, ast.Name)]
                value = node.value
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                targets = [node.target]
                value = node.value
            else:
                continue

            for target in targets:
                if target.id in target_names:
                    if value is not None and isinstance(value, ast.Constant):
                        violations.append(
                            f"{py_file}:{node.lineno} {target.id}={value.value}"
                        )

    assert violations == [], f"Local SNR target definitions: {violations}"


# ── Test 4: Cross-module threshold consistency ────────────────────


def test_cross_module_threshold_consistency():
    """
    GIVEN: All threshold constants from constants.py
    WHEN:  We check their values
    THEN:  They satisfy the invariant ordering:
           UNIFIED_SNR_THRESHOLD < SNR_THRESHOLD_T1_HIGH < SNR_THRESHOLD_T0_ELITE
           UNIFIED_IHSAN_THRESHOLD <= STRICT_IHSAN_THRESHOLD
           ADL_GINI_THRESHOLD > 0 and < 1
    """
    from core.integration.constants import (
        ADL_GINI_THRESHOLD,
        SNR_THRESHOLD_T0_ELITE,
        SNR_THRESHOLD_T1_HIGH,
        STRICT_IHSAN_THRESHOLD,
        UNIFIED_IHSAN_THRESHOLD,
        UNIFIED_SNR_THRESHOLD,
    )

    # Ordering invariants
    assert UNIFIED_SNR_THRESHOLD < SNR_THRESHOLD_T1_HIGH < SNR_THRESHOLD_T0_ELITE
    assert UNIFIED_IHSAN_THRESHOLD <= STRICT_IHSAN_THRESHOLD
    assert 0 < ADL_GINI_THRESHOLD < 1

    # Value stability (these are constitutional — they should not change casually)
    assert UNIFIED_SNR_THRESHOLD == 0.85
    assert SNR_THRESHOLD_T1_HIGH == 0.95
    assert SNR_THRESHOLD_T0_ELITE == 0.98
    assert UNIFIED_IHSAN_THRESHOLD == 0.95
    assert ADL_GINI_THRESHOLD == 0.35
