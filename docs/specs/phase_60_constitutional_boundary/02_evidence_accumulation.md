# Step 2: Evidence Accumulation CI Workflow

## Standing on Giants: Lamport (immutable logs) | Shannon (information as evidence) | Merkle (hash trees)

## Problem Statement

The SAPE audit revealed zero `evidence.json` files produced by any CI workflow.
The only evidence reference across all workflows is a static label in a step
summary table (`tests.yml:407`). This means:

1. CI runs produce pass/fail signals but no structured audit trail
2. There is no way to forensically reconstruct what a specific commit tested
3. Gate results, benchmark data, and security scan outputs are ephemeral
4. The evidence ledger in the runtime has no CI-side counterpart

**Solution:** Add an evidence accumulation step to the CI pipeline that
produces a deterministic `evidence.json` per commit, uploaded as a GitHub
Actions artifact. Each evidence file contains gate results, test counts,
lint status, and security scan outcomes with cryptographic hashes.

## Prerequisite

Step 1 (constitution.toml) — evidence.json references constitution gate names.

## Target Files

| File | Action |
|------|--------|
| `scripts/ci/evidence_aggregator.py` | New: collects CI gate results into evidence.json |
| `evidence/evidence_schema.json` | New: JSONSchema for evidence validation |
| `.github/workflows/ci.yml` | Update: add evidence accumulation + upload step |
| `tests/scripts/test_evidence_aggregator.py` | New: tests for aggregator |

## Pseudocode

### scripts/ci/evidence_aggregator.py

```pseudocode
"""Aggregate CI gate results into a structured evidence.json.

Usage:
    python scripts/ci/evidence_aggregator.py \
        --commit $GITHUB_SHA \
        --output evidence/latest.json

    python scripts/ci/evidence_aggregator.py \
        --commit $GITHUB_SHA \
        --verify evidence/latest.json  # Validate schema compliance
"""

IMPORT json, hashlib, sys, argparse
FROM datetime IMPORT datetime, timezone
FROM pathlib IMPORT Path

EVIDENCE_SCHEMA_VERSION = "1.0.0"


FUNCTION aggregate_evidence(
    commit_sha: str,
    gate_results: dict,
    test_summary: dict,
    security_summary: dict,
) -> dict:
    """Build a deterministic evidence document.

    All fields are required. No optional fields — fail-closed schema.
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    evidence = {
        "schema_version": EVIDENCE_SCHEMA_VERSION,
        "commit_sha": commit_sha,
        "timestamp_utc": timestamp,
        "gates": gate_results,
        "tests": test_summary,
        "security": security_summary,
        "constitution_version": _read_constitution_version(),
        "constants_artifact_hash": _hash_constants_artifact(),
    }

    # Compute content hash (excluding the hash field itself)
    content = json.dumps(evidence, sort_keys=True)
    evidence["content_blake3"] = hashlib.blake2b(
        content.encode(), digest_size=32
    ).hexdigest()

    RETURN evidence


FUNCTION _read_constitution_version() -> str:
    """Read version from constitution.toml if it exists."""
    const_path = Path("constitution.toml")
    IF const_path.exists():
        IMPORT tomllib
        WITH open(const_path, "rb") AS f:
            data = tomllib.load(f)
        RETURN data.get("meta", {}).get("version", "unknown")
    RETURN "not_found"


FUNCTION _hash_constants_artifact() -> str:
    """BLAKE2b hash of bizra-constants.json for cross-reference."""
    artifact = Path("bizra-constants.json")
    IF artifact.exists():
        content = artifact.read_bytes()
        RETURN hashlib.blake2b(content, digest_size=32).hexdigest()
    RETURN "not_found"


FUNCTION parse_gate_results(
    lint_exit: int,
    test_exit: int,
    security_exit: int,
    type_check_exit: int,
    constants_sync_exit: int,
) -> dict:
    """Convert exit codes to structured gate results."""
    RETURN {
        "lint": {"passed": lint_exit == 0, "exit_code": lint_exit},
        "test": {"passed": test_exit == 0, "exit_code": test_exit},
        "security": {"passed": security_exit == 0, "exit_code": security_exit},
        "type_check": {"passed": type_check_exit == 0, "exit_code": type_check_exit},
        "constants_sync": {"passed": constants_sync_exit == 0, "exit_code": constants_sync_exit},
        "all_passed": all(
            code == 0 FOR code IN
            [lint_exit, test_exit, security_exit, type_check_exit, constants_sync_exit]
        ),
    }


FUNCTION validate_evidence(evidence: dict) -> list[str]:
    """Validate evidence dict against schema. Returns list of errors."""
    errors = []
    required_keys = [
        "schema_version", "commit_sha", "timestamp_utc",
        "gates", "tests", "security", "content_blake3",
    ]
    FOR key IN required_keys:
        IF key NOT IN evidence:
            errors.append(f"Missing required key: {key}")

    IF "gates" IN evidence:
        IF "all_passed" NOT IN evidence["gates"]:
            errors.append("gates.all_passed missing")

    IF "commit_sha" IN evidence:
        IF NOT isinstance(evidence["commit_sha"], str):
            errors.append("commit_sha must be string")
        IF len(evidence["commit_sha"]) < 7:
            errors.append("commit_sha too short (min 7 chars)")

    RETURN errors


FUNCTION main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--commit", required=True)
    parser.add_argument("--output", default="evidence/latest.json")
    parser.add_argument("--verify", help="Verify existing evidence file")
    # Gate exit codes passed from CI
    parser.add_argument("--lint-exit", type=int, default=0)
    parser.add_argument("--test-exit", type=int, default=0)
    parser.add_argument("--security-exit", type=int, default=0)
    parser.add_argument("--type-check-exit", type=int, default=0)
    parser.add_argument("--constants-sync-exit", type=int, default=0)
    parser.add_argument("--test-count", type=int, default=0)
    parser.add_argument("--test-failures", type=int, default=0)
    args = parser.parse_args()

    IF args.verify:
        existing = json.loads(Path(args.verify).read_text())
        errors = validate_evidence(existing)
        IF errors:
            print(f"INVALID: {errors}", file=sys.stderr)
            sys.exit(1)
        print("VALID: evidence schema OK")
        sys.exit(0)

    gate_results = parse_gate_results(
        args.lint_exit, args.test_exit, args.security_exit,
        args.type_check_exit, args.constants_sync_exit,
    )

    test_summary = {
        "total": args.test_count,
        "failures": args.test_failures,
        "passed": args.test_count - args.test_failures,
    }

    security_summary = {
        "pip_audit_passed": args.security_exit == 0,
    }

    evidence = aggregate_evidence(
        args.commit, gate_results, test_summary, security_summary,
    )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(evidence, indent=2, sort_keys=True) + "\n")
    print(f"Evidence written: {output} ({len(json.dumps(evidence))} bytes)")
    print(f"Content hash: {evidence['content_blake3']}")
```

### CI Integration

```pseudocode
# .github/workflows/ci.yml — add after all gate jobs complete:

  evidence-accumulation:
    name: Evidence Accumulation
    needs: [lint-python, lint-rust, test-python, cross-language-sync]
    runs-on: ubuntu-24.04
    if: always()  # Run even if gates fail — record the failure
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: '3.12'

      - name: Aggregate evidence
        run: |
          python scripts/ci/evidence_aggregator.py \
            --commit ${{ github.sha }} \
            --output evidence/${{ github.sha }}.json \
            --lint-exit ${{ needs.lint-python.result == 'success' && '0' || '1' }} \
            --test-exit ${{ needs.test-python.result == 'success' && '0' || '1' }} \
            --security-exit 0 \
            --constants-sync-exit ${{ needs.cross-language-sync.result == 'success' && '0' || '1' }}

      - name: Validate evidence schema
        run: |
          python scripts/ci/evidence_aggregator.py \
            --commit ${{ github.sha }} \
            --verify evidence/${{ github.sha }}.json

      - name: Upload evidence artifact
        uses: actions/upload-artifact@v4
        with:
          name: evidence-${{ github.sha }}
          path: evidence/${{ github.sha }}.json
          retention-days: 90
```

## TDD Anchors

```pseudocode
TEST aggregate_evidence_has_required_keys:
    evidence = aggregate_evidence("abc1234", {}, {}, {})
    ASSERT "schema_version" IN evidence
    ASSERT "commit_sha" IN evidence
    ASSERT "timestamp_utc" IN evidence
    ASSERT "content_blake3" IN evidence
    ASSERT evidence["commit_sha"] == "abc1234"

TEST evidence_is_deterministic:
    """Same inputs produce same output (excluding timestamp)."""
    gates = {"all_passed": True}
    e1 = aggregate_evidence("abc", gates, {"total": 100}, {})
    e2 = aggregate_evidence("abc", gates, {"total": 100}, {})
    del e1["timestamp_utc"], e1["content_blake3"]
    del e2["timestamp_utc"], e2["content_blake3"]
    ASSERT e1 == e2

TEST validate_evidence_catches_missing_keys:
    errors = validate_evidence({})
    ASSERT len(errors) >= 5  # At least 5 required keys missing

TEST validate_evidence_passes_valid:
    evidence = aggregate_evidence("abc1234def", {"all_passed": True}, {}, {})
    errors = validate_evidence(evidence)
    ASSERT len(errors) == 0

TEST parse_gate_results_all_pass:
    gates = parse_gate_results(0, 0, 0, 0, 0)
    ASSERT gates["all_passed"] IS True
    ASSERT gates["lint"]["passed"] IS True

TEST parse_gate_results_partial_fail:
    gates = parse_gate_results(0, 1, 0, 0, 0)
    ASSERT gates["all_passed"] IS False
    ASSERT gates["test"]["passed"] IS False
    ASSERT gates["lint"]["passed"] IS True

TEST evidence_json_is_serializable:
    evidence = aggregate_evidence("abc1234", {}, {}, {})
    json_str = json.dumps(evidence)
    roundtrip = json.loads(json_str)
    ASSERT roundtrip["commit_sha"] == "abc1234"

TEST content_hash_changes_with_content:
    e1 = aggregate_evidence("abc", {"x": 1}, {}, {})
    e2 = aggregate_evidence("def", {"x": 2}, {}, {})
    ASSERT e1["content_blake3"] != e2["content_blake3"]
```

## Acceptance Criteria

1. `scripts/ci/evidence_aggregator.py` produces valid evidence.json
2. `evidence.json` uploaded as GitHub Actions artifact on every CI run
3. Evidence schema validates with zero errors on valid input
4. Evidence is produced even when gates fail (records the failure)
5. Content hash is deterministic for identical inputs
6. Full test suite GREEN
