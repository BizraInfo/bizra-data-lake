# Phase 66.01: Threshold Canonicalization

## Problem Statement

11 files define local constants that duplicate values from the canonical
`core/integration/constants.py`. This creates dual truth sources — if
constants.py changes, these files silently diverge.

> Axiom: "Ethics must be the compiler, not guidelines."
> A threshold defined in two places is a guideline. Defined in one and
> imported everywhere, it becomes a compiler constraint.

## Pseudocode

### Step 1: Gini Threshold Dedup (4 files — highest priority)

```
FOR EACH file IN [
    core/sovereign/adl_kernel.py:52,
    core/sovereign/adl_invariant.py:50,
    core/elite/compute_market.py:63,
    core/token/emission_decay.py:40,
]:
    REMOVE local constant definition (ADL_GINI_THRESHOLD or GINI_THRESHOLD)
    ADD import: from core.integration.constants import ADL_GINI_THRESHOLD
    REPLACE all local references with imported constant

    # emission_decay.py uses DEFAULT_GINI_TARGET — rename reference
    IF file == emission_decay.py:
        REPLACE DEFAULT_GINI_TARGET with ADL_GINI_THRESHOLD
```

### Step 2: SNR Threshold Dedup (4 files)

```
FOR EACH file IN [
    core/command/sovereign_command.py:51,       # SNR_TARGET = 0.99
    core/apex/snr_apex_engine.py:61-62,         # APEX_SNR_TARGET/FLOOR
    core/apex/peak_masterpiece.py:74-75,        # PEAK_SNR_TARGET/FLOOR
    core/sdpo/__init__.py:41-42,                # SAPE_KNOWLEDGE/INFO_SNR
]:
    REMOVE local constant definition
    ADD import: from core.integration.constants import (
        SNR_THRESHOLD_T0_ELITE,    # replaces 0.99 targets
        SNR_THRESHOLD_T1_HIGH,     # replaces 0.95 floors
    )
    MAP local names to canonical names:
        SNR_TARGET            → SNR_THRESHOLD_T0_ELITE
        APEX_SNR_TARGET       → SNR_THRESHOLD_T0_ELITE
        APEX_SNR_FLOOR        → SNR_THRESHOLD_T1_HIGH
        PEAK_SNR_TARGET       → SNR_THRESHOLD_T0_ELITE
        PEAK_SNR_FLOOR        → SNR_THRESHOLD_T1_HIGH
        SAPE_KNOWLEDGE_SNR    → SNR_THRESHOLD_T0_ELITE
        SAPE_INFORMATION_SNR  → SNR_THRESHOLD_T1_HIGH
```

### Step 3: Ihsan Threshold Dedup (3 files)

```
FOR EACH file IN [
    core/autopoiesis/shadow_deploy.py:85,       # IHSAN_KILL_THRESHOLD = 0.95
    core/benchmark/moe_router.py:335,           # IHSAN_CONFIDENCE = 0.95
    core/spearpoint/true_spearpoint_loop.py:48, # target_snr: float = 0.99
]:
    REMOVE local constant
    ADD import: from core.integration.constants import UNIFIED_IHSAN_THRESHOLD
    REPLACE local name with UNIFIED_IHSAN_THRESHOLD

    # shadow_deploy.py already imports UNIFIED_IHSAN_THRESHOLD but defines
    # IHSAN_KILL_THRESHOLD separately — remove the separate definition
    # and use the imported constant directly

    # true_spearpoint_loop.py: dataclass field default
    # CHANGE: target_snr: float = 0.99
    # TO:     target_snr: float = SNR_THRESHOLD_T0_ELITE
```

## Edge Cases

- `core/proof_engine/ihsan_gate.py:40-43`: Fallback literals in `except`
  block for missing `bizra_constitution` package. KEEP AS IS — this is
  a correct defensive pattern, not a duplicate truth source.

- `core/sovereign/autonomy_matrix.py`: Inline `>= 0.95` comparisons in
  method bodies. DEFER — these are lower priority than module-level
  constant definitions. Phase 67 cleanup.

## Invariants

```
ASSERT: grep -rn "= 0.35" core/ | grep -v constants.py | grep -v test → EMPTY
ASSERT: grep -rn "GINI_THRESHOLD" core/ | grep "= 0.35" | grep -v constants.py → EMPTY
ASSERT: all 11 files import from core.integration.constants
```

## TDD Anchor

```python
# test_threshold_canonicalization.py

def test_no_duplicate_gini_threshold():
    """No file outside constants.py defines ADL_GINI_THRESHOLD as a literal."""
    import ast, pathlib

    constants_path = pathlib.Path("core/integration/constants.py")
    for py_file in pathlib.Path("core").rglob("*.py"):
        if py_file == constants_path or "test" in str(py_file):
            continue
        tree = ast.parse(py_file.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Name) and "GINI" in target.id:
                        if isinstance(node.value, ast.Constant) and node.value.value == 0.35:
                            pytest.fail(f"{py_file}:{node.lineno} defines Gini 0.35 locally")


def test_constants_imported_not_redefined():
    """Core modules use imports, not local definitions, for thresholds."""
    from core.sovereign.adl_kernel import ADL_GINI_THRESHOLD as kernel_val
    from core.integration.constants import ADL_GINI_THRESHOLD as canonical_val
    assert kernel_val is canonical_val  # same object, not just same value
```

## Estimated Impact

- Lines changed: ~30 (11 deletions + 11 imports + 8 reference renames)
- Risk: LOW — pure refactoring, no behavior change
- SNR improvement: eliminates 11 dual-truth sources
