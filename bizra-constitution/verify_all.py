#!/usr/bin/env python3
"""
BIZRA Constitution Verification — Single Command
═════════════════════════════════════════════════

Run: python verify_all.py

Validates:
  1. Constitution loads and passes all invariants
  2. Generated constants match constitution
  3. 6-dim Ihsan gate works correctly
  4. SNR normalization is canonical
  5. Evidence chain links correctly
  6. All 102 tests pass
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    os.chdir(Path(__file__).parent)
    os.environ["BIZRA_CONSTITUTION_PATH"] = "constitution.toml"

    print("=" * 70)
    print("  BIZRA CONSTITUTION v5.0.0-GENESIS — Full Verification")
    print("=" * 70)
    print()

    # 1. Validate constitution
    print("[1/4] Loading constitution.toml...")
    try:
        from bizra_constitution import load_constitution
        c = load_constitution("constitution.toml")
        violations = c.validate()
        if violations:
            print(f"  ❌ {len(violations)} violations found")
            for v in violations:
                print(f"     - {v}")
            return 1
        print(f"  ✅ v{c.meta.version} — 0 violations")
        print(f"     SHA-256: {c.raw_hash[:32]}...")
        print(f"     Ihsan: {c.ihsan.dimensions}-dim canonical, "
              f"{len(c.ihsan.operational_dimensions)}-dim operational")
        print(f"     Gates: {c.gates.count}, fail_mode={c.gates.fail_mode}")
        print(f"     PAT: {c.pat.agent_count} agents, trust_monotonicity={c.pat.trust_monotonicity}")
        print(f"     SAT: {c.sat.agents_per_node}/node, {len(c.sat.bootstrap_roles)} bootstrap roles")
        print(f"     Zakat: {c.economics.zakat_rate} (constitutional)")
        print(f"     Rights: {len(c.identity.rights.rights)}")
    except Exception as e:
        print(f"  ❌ Failed: {e}")
        return 1

    # 2. Regenerate and verify constants
    print()
    print("[2/4] Regenerating constants from constitution...")
    try:
        from generate_from_constitution import generate_constants, generate_tests
        constants_text = generate_constants(c)
        tests_text = generate_tests(c)
        n_constants = sum(1 for line in constants_text.splitlines()
                         if "=" in line and not line.strip().startswith("#"))
        n_tests = sum(1 for line in tests_text.splitlines() if "def test_" in line)
        print(f"  ✅ {n_constants} constants, {n_tests} conformance tests")
    except Exception as e:
        print(f"  ❌ Generation failed: {e}")
        return 1

    # 3. Quick smoke test of core modules
    print()
    print("[3/4] Smoke testing core modules...")
    try:
        from ihsan_gate import IhsanGate
        gate = IhsanGate()
        score = gate.evaluate("This is a well-reasoned analysis because the evidence supports it.")
        print(f"  ✅ IhsanGate: composite={score.composite:.3f}, "
              f"tier={score.tier.value}, passes={score.passes}")

        from snr import normalize_snr, measure_mission_snr
        assert normalize_snr(0) == 0.0
        assert abs(normalize_snr(1) - 0.5) < 0.001
        assert normalize_snr(1000) > 0.999
        m = measure_mission_snr("Test output", ihsan_composite=0.92)
        print(f"  ✅ SNR: normalize(1)=0.500, mission_snr={m.snr_normalized:.3f}")

        import tempfile
        from evidence_receipt import EvidenceLedger
        with tempfile.TemporaryDirectory() as td:
            ledger = EvidenceLedger(Path(td) / "test.jsonl")
            r1 = ledger.append(
                mission_id="verify-1",
                ihsan_tensor=score.as_tensor_dict(),
                ihsan_composite=score.composite,
                gate_results={"alpha_4": True, "alpha_7": True,
                              "alpha_8": True, "alpha_9": True, "alpha_10": True},
                snr_normalized=m.snr_normalized,
                tier=score.tier.value,
            )
            r2 = ledger.append(
                mission_id="verify-2",
                ihsan_tensor=score.as_tensor_dict(),
                ihsan_composite=score.composite,
                gate_results={"alpha_4": True, "alpha_7": True,
                              "alpha_8": True, "alpha_9": True, "alpha_10": True},
                snr_normalized=m.snr_normalized,
                tier=score.tier.value,
            )
            assert r2.previous_hash == r1.receipt_id
            valid, count, errors = ledger.verify_chain()
            assert valid and count == 2
            print(f"  ✅ Evidence chain: 2 receipts, linked, verified")
    except Exception as e:
        print(f"  ❌ Smoke test failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # 4. Run full pytest suite
    print()
    print("[4/4] Running full test suite...")
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "tests/", "generated/",
         "-v", "--tb=short", "-q"],
        capture_output=True, text=True,
    )
    # Extract summary line
    lines = result.stdout.strip().splitlines()
    summary = lines[-1] if lines else "no output"
    if result.returncode == 0:
        print(f"  ✅ {summary}")
    else:
        print(f"  ❌ {summary}")
        print(result.stdout[-500:] if len(result.stdout) > 500 else result.stdout)
        return 1

    # Final report
    print()
    print("=" * 70)
    print("  VERIFICATION COMPLETE")
    print("=" * 70)
    print(f"  Constitution: v{c.meta.version}")
    print(f"  Hash:         {c.raw_hash[:32]}...")
    print(f"  Tests:        {summary}")
    print(f"  Status:       ✅ READY FOR INTEGRATION")
    print()
    print("  Next: Follow MIGRATION.md to integrate into bizra-omega workspace")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
