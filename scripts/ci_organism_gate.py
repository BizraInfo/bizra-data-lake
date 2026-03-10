#!/usr/bin/env python3
"""
CI Organism Smoke Gate (§7 T0 — < 30 seconds)
==============================================

Boots the Sovereign Organism, runs 3 missions, ticks once,
checks health, verifies constitutional invariants, and exits
with code 0 (pass) or 1 (fail).

Usage:
    python scripts/ci_organism_gate.py          # normal
    python scripts/ci_organism_gate.py --json   # JSON output for CI

Evidence: DDAGI Pilot v2.0 §7 Test Tier T0 Smoke
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
from typing import Any, Dict

# Ensure core/ is importable when run from repo root
_repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


class _CIInference:
    """Deterministic inference for CI — no LLM dependency."""

    async def infer(self, prompt: str, **kwargs: Any) -> str:
        agent_id = kwargs.get("agent_id", "unknown")
        return (
            f"[{agent_id}] CI smoke response:\n"
            f"- Analysis: comprehensive review completed\n"
            f"- Evidence: verified against constitutional standards\n"
            f"- Recommendation: proceed with confidence\n"
            f"- Quality: exceeds Ihsān threshold"
        )


async def run_gate(verbose: bool = True) -> Dict[str, Any]:
    """Execute the organism smoke gate.

    Returns dict with pass/fail status and diagnostics.
    """
    from core.sovereign.organism import SovereignOrganism

    results: Dict[str, Any] = {
        "gate": "organism-smoke",
        "version": "1.0.0",
        "timestamp": time.time(),
        "checks": [],
        "passed": False,
    }

    def check(name: str, passed: bool, detail: str = "") -> bool:
        results["checks"].append({
            "name": name,
            "passed": passed,
            "detail": detail,
        })
        if verbose:
            icon = "✅" if passed else "❌"
            msg = f"  {icon} {name}"
            if detail:
                msg += f" — {detail}"
            print(msg)
        return passed

    all_ok = True
    start = time.monotonic()

    # ── Check 1: Boot ──────────────────────────────────────────
    if verbose:
        print("═══ ORGANISM SMOKE GATE (§7 T0) ═══\n")

    try:
        org = await SovereignOrganism.boot(inference=_CIInference())
        all_ok &= check("boot", True, "organism created")
    except Exception as exc:
        check("boot", False, str(exc))
        results["duration_ms"] = round((time.monotonic() - start) * 1000, 1)
        return results

    # ── Check 2: Health after boot ─────────────────────────────
    h = org.health
    all_ok &= check("health.alive", h.alive)
    all_ok &= check("health.agents", h.agents_registered == 12, f"{h.agents_registered} agents")

    # ── Check 3: Run 3 missions (§6 Mode 2) ───────────────────
    missions = [
        "analyze authentication security",
        "implement error handling improvements",
        "review test coverage gaps",
    ]
    receipts = []
    for mission_text in missions:
        try:
            receipt = await org.mission(mission_text)
            receipts.append(receipt)
            all_ok &= check(
                f"mission({mission_text[:30]}...)",
                receipt.ihsan_score > 0,
                f"ihsan={receipt.ihsan_score:.4f}, system={receipt.system}",
            )
        except Exception as exc:
            all_ok &= check(f"mission({mission_text[:30]}...)", False, str(exc))

    # ── Check 4: Evolutionary tick (§2 Helix 3) ───────────────
    try:
        tick_receipt = await org.tick()
        all_ok &= check(
            "tick",
            tick_receipt.tick_number >= 1,
            f"tick #{tick_receipt.tick_number}",
        )
    except Exception as exc:
        all_ok &= check("tick", False, str(exc))

    # ── Check 5: Health after missions ─────────────────────────
    h = org.health
    all_ok &= check(
        "missions_completed",
        h.missions_completed >= 3,
        f"{h.missions_completed} completed",
    )
    all_ok &= check(
        "ihsan_avg",
        h.current_ihsan_avg > 0,
        f"avg={h.current_ihsan_avg:.4f}",
    )

    # ── Check 6: Constitutional invariants (§4) ───────────────
    violations = org.check_invariants()
    all_ok &= check(
        "invariants",
        len(violations) == 0,
        f"{len(violations)} violations" if violations else "clean",
    )

    # ── Check 7: Shutdown ──────────────────────────────────────
    try:
        await org.shutdown()
        all_ok &= check("shutdown", not org.health.alive, "graceful")
    except Exception as exc:
        all_ok &= check("shutdown", False, str(exc))

    duration_ms = round((time.monotonic() - start) * 1000, 1)
    results["duration_ms"] = duration_ms
    results["passed"] = all_ok

    checks_passed = sum(1 for c in results["checks"] if c["passed"])
    checks_total = len(results["checks"])

    if verbose:
        print(f"\n{'═' * 40}")
        icon = "✅" if all_ok else "❌"
        print(f"{icon} ORGANISM GATE: {checks_passed}/{checks_total} checks in {duration_ms}ms")
        if not all_ok:
            failed = [c["name"] for c in results["checks"] if not c["passed"]]
            print(f"   FAILED: {', '.join(failed)}")

    return results


def main() -> None:
    use_json = "--json" in sys.argv
    result = asyncio.run(run_gate(verbose=not use_json))

    if use_json:
        print(json.dumps(result, indent=2))

    sys.exit(0 if result["passed"] else 1)


if __name__ == "__main__":
    main()
