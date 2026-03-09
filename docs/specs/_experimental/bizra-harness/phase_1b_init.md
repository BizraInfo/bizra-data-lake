# Phase 1b — Package Init and Public API

## File: `core/harness/__init__.py`

```pseudocode
"""BIZRA Unified Harness — end-to-end quality validation.

Usage:
    from core.harness import HarnessRunner, HarnessConfig, Verdict
    result = await HarnessRunner().run(HarnessConfig())
    assert result.verdict == Verdict.PASS
"""
from core.harness.types import (
    HarnessConfig,
    HarnessResult,
    HarnessScenario,
    PillarName,
    PillarResult,
    RegressionReport,
    RunMode,
    Verdict,
)
from core.harness.runner import HarnessRunner, run_harness
from core.harness.scenarios import ScenarioLibrary
from core.harness.persistence import BaselineStore
from core.harness.report import generate_json_report, generate_html_report

__all__ = [
    # Types
    "HarnessConfig",
    "HarnessResult",
    "HarnessScenario",
    "PillarName",
    "PillarResult",
    "RegressionReport",
    "RunMode",
    "Verdict",
    # Runner
    "HarnessRunner",
    "run_harness",
    # Scenarios
    "ScenarioLibrary",
    # Persistence
    "BaselineStore",
    # Reports
    "generate_json_report",
    "generate_html_report",
]
```

## `__main__.py` (CLI entry point)

```pseudocode
# core/harness/__main__.py
"""Allow: python -m core.harness [--mode smoke|standard|full|benchmark]"""
from core.harness._cli import main
main()
```

The `_cli.py` module contains the argparse logic from Phase 5.
