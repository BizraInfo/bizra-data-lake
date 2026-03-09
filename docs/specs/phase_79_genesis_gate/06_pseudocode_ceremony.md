# Phase 79: Genesis-100 Ceremony — Pseudocode

---

## Module: `core/sat/ceremony.py`

```pseudocode
IMPORT datetime, json, hashlib
IMPORT GateResult FROM core.sat.gate_result
IMPORT sentinel_verify FROM core.sat.sentinel_gate
IMPORT oracle_s_verify FROM core.sat.oracle_s_gate
IMPORT ledger_verify FROM core.sat.ledger_gate
IMPORT conductor_verify FROM core.sat.conductor_gate
IMPORT ambassador_verify FROM core.sat.ambassador_gate
IMPORT EvidenceLedger FROM core.proof_engine.evidence_ledger
IMPORT load_or_create_signer FROM core.proof_engine.genesis_ceremony


DATACLASS GenesisReceipt:
    ceremony: str = "GENESIS_100"
    timestamp: str
    agents: List[Dict]           # Each GateResult.to_dict()
    all_passed: bool
    total_checks: int
    passed_checks: int
    failed_checks: List[Tuple[str, str]]  # (agent_name, check_name)
    signature: Optional[str] = None
    hash: Optional[str] = None

    METHOD sign(signer):
        payload = json.dumps({
            "ceremony": ceremony,
            "timestamp": timestamp,
            "all_passed": all_passed,
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_count": len(failed_checks),
        }, sort_keys=True).encode()
        hash = hashlib.blake2b(payload).hexdigest()
        signature = signer.sign(payload).hex()
        self.hash = hash
        self.signature = signature

    METHOD to_dict() -> Dict:
        RETURN {
            "ceremony": ceremony,
            "timestamp": timestamp,
            "all_passed": all_passed,
            "total_checks": total_checks,
            "passed_checks": passed_checks,
            "failed_checks": failed_checks,
            "hash": hash,
            "signature": signature,
            "agents": agents,
        }


FUNCTION genesis_100_ceremony(
    skip_manual: bool = False,
    skip_slow: bool = False,
    sign: bool = True,
    store: bool = True,
) -> Tuple[bool, GenesisReceipt]:
    """The final gate. ALL 5 agents must approve."""

    # Run all 5 gates
    results = [
        sentinel_verify(skip_slow=skip_slow),
        oracle_s_verify(skip_manual=skip_manual),
        ledger_verify(),
        conductor_verify(),
        ambassador_verify(skip_manual=skip_manual),
    ]

    all_passed = all(r.passed FOR r IN results)
    total_checks = sum(len(r.checks) FOR r IN results)
    passed_checks = sum(
        1 FOR r IN results FOR c IN r.checks IF c.passed
    )
    failed_list = [
        (r.agent, c.name)
        FOR r IN results
        FOR c IN r.checks
        IF NOT c.passed AND c.status != SKIPPED
    ]

    receipt = GenesisReceipt(
        timestamp=datetime.utcnow().isoformat(),
        all_passed=all_passed,
        total_checks=total_checks,
        passed_checks=passed_checks,
        failed_checks=failed_list,
        agents=[r.to_dict() FOR r IN results],
    )

    # Sign with Node0 Ed25519 key
    IF sign:
        TRY:
            signer = load_or_create_signer()
            receipt.sign(signer)
        EXCEPT Exception AS e:
            print(f"WARNING: Could not sign receipt: {e}")

    # Store as evidence block
    IF store:
        TRY:
            ledger = EvidenceLedger()
            ledger.append(receipt=receipt.to_dict())
        EXCEPT Exception AS e:
            print(f"WARNING: Could not store receipt: {e}")

    # Print result
    _print_ceremony_result(receipt, results)

    RETURN (all_passed, receipt)


FUNCTION _print_ceremony_result(receipt, results):
    """Formatted ceremony output."""
    IF receipt.all_passed:
        print("╔═══════════════════════════════════════╗")
        print("║  GENESIS-100: ALL GATES PASSED        ║")
        print("║  100 invitations authorized.           ║")
        print("║  The forest begins.                    ║")
        print("╚═══════════════════════════════════════╝")
    ELSE:
        print(f"╔═══════════════════════════════════════╗")
        print(f"║  GENESIS-100: BLOCKED                 ║")
        print(f"║  {len(receipt.failed_checks)} checks failed.{' ' * 20}║")
        print(f"╚═══════════════════════════════════════╝")

    print()
    FOR result IN results:
        icon = "✓" IF result.passed ELSE "✗"
        print(f"  {icon} [{result.agent}] {result.layer}: {result.verdict.value}")
        stats = result.stats
        print(f"    {stats['pass']} pass, {stats['fail']} fail, "
              f"{stats['partial']} partial, {stats['not_impl']} not-impl, "
              f"{stats['skipped']} skipped")
        IF result.failed:
            FOR check IN result.failed:
                print(f"      ✗ {check.name}: {check.evidence[:80]}")

    print()
    print(f"  Total: {receipt.passed_checks}/{receipt.total_checks} checks passed")
    IF receipt.hash:
        print(f"  Receipt hash: {receipt.hash[:32]}...")
    IF receipt.signature:
        print(f"  Signed: Ed25519 ({receipt.signature[:16]}...)")
```

---

## Module: `core/sat/__init__.py`

```pseudocode
"""SAT-5 Genesis Gate Module.

Five agents, 68 checks, zero overrides on constitutional gates.
When ALL pass, the forest begins.
"""

FROM core.sat.gate_result IMPORT GateResult, CheckResult, CheckStatus
FROM core.sat.sentinel_gate IMPORT sentinel_verify
FROM core.sat.oracle_s_gate IMPORT oracle_s_verify
FROM core.sat.ledger_gate IMPORT ledger_verify
FROM core.sat.conductor_gate IMPORT conductor_verify
FROM core.sat.ambassador_gate IMPORT ambassador_verify
FROM core.sat.ceremony IMPORT genesis_100_ceremony, GenesisReceipt

__all__ = [
    "GateResult", "CheckResult", "CheckStatus",
    "sentinel_verify", "oracle_s_verify", "ledger_verify",
    "conductor_verify", "ambassador_verify",
    "genesis_100_ceremony", "GenesisReceipt",
]
```

---

## CLI Integration: `core/sovereign/__main__.py` additions

```pseudocode
# Add 'gate' subcommand group

gate_parser = subparsers.add_parser("gate", help="Genesis-100 release gates")
gate_sub = gate_parser.add_subparsers(dest="gate_command")

# bizra gate sentinel
gate_sub.add_parser("sentinel")
gate_sub.add_parser("oracle-s")
gate_sub.add_parser("ledger")
gate_sub.add_parser("conductor")
gate_sub.add_parser("ambassador")

# bizra gate all
all_parser = gate_sub.add_parser("all")
all_parser.add_argument("--skip-manual", action="store_true")
all_parser.add_argument("--json", action="store_true")

# bizra gate ceremony
ceremony_parser = gate_sub.add_parser("ceremony")
ceremony_parser.add_argument("--skip-manual", action="store_true")
ceremony_parser.add_argument("--skip-slow", action="store_true")
ceremony_parser.add_argument("--no-sign", action="store_true")
ceremony_parser.add_argument("--no-store", action="store_true")

# bizra gate scorecard
scorecard_parser = gate_sub.add_parser("scorecard")
scorecard_parser.add_argument("--json", action="store_true")


FUNCTION handle_gate(args):
    IF args.gate_command == "ceremony":
        passed, receipt = genesis_100_ceremony(
            skip_manual=args.skip_manual,
            skip_slow=args.skip_slow,
            sign=NOT args.no_sign,
            store=NOT args.no_store,
        )
        sys.exit(0 IF passed ELSE 1)

    ELIF args.gate_command == "all":
        results = run_all_gates(skip_manual=args.skip_manual)
        IF args.json:
            print(json.dumps([r.to_dict() FOR r IN results], indent=2))
        ELSE:
            print_scorecard(results)

    ELIF args.gate_command == "scorecard":
        results = run_all_gates(skip_manual=True)
        IF args.json:
            print(json.dumps(build_scorecard(results), indent=2))
        ELSE:
            print_scorecard(results)

    ELIF args.gate_command IN ("sentinel", "oracle-s", "ledger", "conductor", "ambassador"):
        gate_fn = GATE_MAP[args.gate_command]
        result = gate_fn()
        print_single_gate(result)
        sys.exit(0 IF result.passed ELSE 1)
```

---

## TDD Anchors

```pseudocode
TEST test_ceremony_runs_all_5_gates:
    passed, receipt = genesis_100_ceremony(skip_manual=True, skip_slow=True, sign=False, store=False)
    ASSERT receipt.ceremony == "GENESIS_100"
    ASSERT len(receipt.agents) == 5
    ASSERT receipt.total_checks >= 49  # 49 automated minimum

TEST test_ceremony_blocks_on_failure:
    # Mock sentinel to fail
    passed, receipt = genesis_100_ceremony(skip_manual=True, sign=False, store=False)
    IF NOT passed:
        ASSERT len(receipt.failed_checks) > 0
        ASSERT receipt.all_passed == False

TEST test_ceremony_receipt_signed:
    passed, receipt = genesis_100_ceremony(skip_manual=True, skip_slow=True, store=False)
    ASSERT receipt.hash IS NOT None
    ASSERT receipt.signature IS NOT None

TEST test_ceremony_receipt_stored:
    passed, receipt = genesis_100_ceremony(skip_manual=True, skip_slow=True, sign=False)
    # Verify evidence ledger has new entry
    ledger = EvidenceLedger()
    last = ledger.last()
    ASSERT last["ceremony"] == "GENESIS_100"

TEST test_scorecard_json_valid:
    result = subprocess.run(["python", "-m", "core.sovereign", "gate", "scorecard", "--json"],
                             capture_output=True)
    data = json.loads(result.stdout)
    ASSERT "layers" IN data
    ASSERT len(data["layers"]) == 5
```
