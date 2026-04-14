#!/usr/bin/env python3
"""
BIZRA Node0 Integration Test
==============================
Boots SovereignRuntime.initialize() and verifies all Phase 80 components
(PAT, SAT, DEMA, FATE, ProactiveScheduler) are alive.

This is a heavier test than the smoke test — it actually boots the runtime.

Exit codes:
  0 — all checks pass
  1 — one or more checks failed
"""

from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT))

PASS = 0
FAIL = 0
RESULTS: list[dict] = []


def check(name: str, condition: bool, detail: str = ""):
    global PASS, FAIL
    if condition:
        PASS += 1
        RESULTS.append({"check": name, "status": "PASS"})
        print(f"  ✓ {name}")
    else:
        FAIL += 1
        RESULTS.append({"check": name, "status": "FAIL", "detail": detail})
        print(f"  ✗ {name}: {detail}")


async def run_integration():
    from core.sovereign.runtime_core import SovereignRuntime, RuntimeConfig

    config = RuntimeConfig()
    config.autonomous_enabled = False  # Don't start autonomous loop in test
    runtime = SovereignRuntime(config=config)

    print("\n  Booting SovereignRuntime.initialize()...")
    t0 = time.monotonic()

    try:
        await asyncio.wait_for(runtime.initialize(), timeout=60.0)
        elapsed = time.monotonic() - t0
        print(f"  Boot completed in {elapsed:.1f}s\n")
        check("Runtime initialized", runtime._initialized, "initialize() did not set _initialized")
        check("Runtime running", runtime._running, "initialize() did not set _running")
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - t0
        print(f"  ✗ Boot TIMED OUT after {elapsed:.1f}s\n")
        check("Runtime boot", False, f"Timed out after {elapsed:.1f}s")
        return runtime
    except Exception as e:
        elapsed = time.monotonic() - t0
        print(f"  ⚠ Boot completed with errors after {elapsed:.1f}s: {e}\n")
        # Runtime may still be partially initialized — check what we got
        check("Runtime initialized (partial)", runtime._initialized, str(e))

    # Phase 80 component checks
    check(
        "PAT Runtime wired",
        runtime._pat_runtime is not None,
        "runtime._pat_runtime is None"
    )
    check(
        "SAT Runtime wired",
        runtime._sat_runtime is not None,
        "runtime._sat_runtime is None"
    )
    check(
        "DEMA Router wired",
        runtime._dema_router is not None,
        "runtime._dema_router is None"
    )
    check(
        "FATE Boundary wired",
        runtime._fate_boundary is not None,
        "runtime._fate_boundary is None"
    )
    check(
        "ProactiveScheduler wired",
        runtime._proactive_scheduler is not None,
        "runtime._proactive_scheduler is None"
    )
    check(
        "URP Service wired",
        runtime._urp_service is not None,
        "runtime._urp_service is None"
    )

    # Check older core components
    check(
        "Event bus alive",
        runtime._event_bus is not None,
        "runtime._event_bus is None"
    )
    check(
        "Gate chain alive",
        runtime._gate_chain is not None,
        "runtime._gate_chain is None"
    )

    # Shutdown cleanly
    try:
        if hasattr(runtime, 'shutdown'):
            await asyncio.wait_for(runtime.shutdown(), timeout=10.0)
            print("\n  Runtime shutdown cleanly.")
    except Exception:
        pass  # Best effort shutdown

    return runtime


def main():
    print()
    print("═" * 60)
    print("  BIZRA NODE0 INTEGRATION TEST")
    print("═" * 60)

    runtime = asyncio.run(run_integration())

    print()
    print("═" * 60)
    total = PASS + FAIL
    print(f"  Results: {PASS}/{total} passed, {FAIL} failed")
    print("═" * 60)

    # Save results
    state_dir = REPO_ROOT / "sovereign_state" / "receipts"
    state_dir.mkdir(parents=True, exist_ok=True)
    results_path = state_dir / "integration_test_latest.json"
    results_path.write_text(json.dumps({
        "test_type": "integration",
        "total": total,
        "passed": PASS,
        "failed": FAIL,
        "results": RESULTS,
    }, indent=2))
    print(f"\n  Results saved: {results_path}")

    if FAIL > 0:
        print(f"\n  ✗ {FAIL} check(s) FAILED")
        sys.exit(1)
    else:
        print(f"\n  ✓ All {PASS} checks passed")
        sys.exit(0)


if __name__ == "__main__":
    main()
