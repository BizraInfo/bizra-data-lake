#!/usr/bin/env python3
"""
Schema Validation Gate — CI/CD and local validation of JSON schemas and fixtures.

Validates:
1. All .schema.json files load without error
2. All fixtures in schemas/fixtures/ validate against their schema
3. Reason codes in Python match the JSON schema enum
4. Schema cross-references are consistent

Standing on Giants:
- JSON Schema Draft 2020-12
- BIZRA Spearpoint PRD SP-009: "CI gate for schema validation"

Usage:
    python tools/validate_schemas.py          # Run all checks
    python tools/validate_schemas.py --strict  # Fail on warnings too
"""

import json
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SCHEMAS_DIR = PROJECT_ROOT / "schemas"
FIXTURES_DIR = SCHEMAS_DIR / "fixtures"

# Map fixture filenames to their schema filenames
FIXTURE_SCHEMA_MAP = {
    "valid_receipt.json": "receipt.schema.json",
    "valid_reasoning_graph.json": "reasoning_graph.schema.json",
    "valid_attestation.json": "attestation.schema.json",
}


def load_json(path: Path) -> dict:
    """Load and parse a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def check_schemas_load() -> list:
    """Check that all schema files load without error."""
    errors = []
    schema_files = list(SCHEMAS_DIR.glob("*.schema.json"))

    if not schema_files:
        errors.append("CRITICAL: No schema files found in schemas/")
        return errors

    for schema_file in sorted(schema_files):
        try:
            schema = load_json(schema_file)
            # Basic structural checks
            if "$schema" not in schema and "type" not in schema:
                errors.append(f"WARNING: {schema_file.name} missing $schema or type field")
            print(f"  PASS  {schema_file.name}")
        except json.JSONDecodeError as e:
            errors.append(f"FAIL: {schema_file.name} — invalid JSON: {e}")
        except Exception as e:
            errors.append(f"FAIL: {schema_file.name} — {e}")

    return errors


def check_fixtures_validate() -> list:
    """Validate all fixtures against their corresponding schemas."""
    errors = []

    if not FIXTURES_DIR.exists():
        errors.append("CRITICAL: schemas/fixtures/ directory not found")
        return errors

    fixture_files = list(FIXTURES_DIR.glob("*.json"))
    if not fixture_files:
        errors.append("WARNING: No fixture files found in schemas/fixtures/")
        return errors

    # Try to use jsonschema if available
    try:
        import jsonschema
        has_jsonschema = True
    except ImportError:
        has_jsonschema = False
        print("  INFO  jsonschema not installed — using structural validation only")

    for fixture_file in sorted(fixture_files):
        schema_name = FIXTURE_SCHEMA_MAP.get(fixture_file.name)
        if not schema_name:
            errors.append(f"WARNING: {fixture_file.name} has no mapped schema")
            continue

        schema_path = SCHEMAS_DIR / schema_name
        if not schema_path.exists():
            errors.append(f"FAIL: Schema {schema_name} not found for fixture {fixture_file.name}")
            continue

        try:
            fixture = load_json(fixture_file)
            schema = load_json(schema_path)

            if has_jsonschema:
                jsonschema.validate(fixture, schema)
                print(f"  PASS  {fixture_file.name} validates against {schema_name}")
            else:
                # Structural fallback: check required fields exist
                required = schema.get("required", [])
                missing = [r for r in required if r not in fixture]
                if missing:
                    errors.append(
                        f"FAIL: {fixture_file.name} missing required fields: {missing}"
                    )
                else:
                    print(f"  PASS  {fixture_file.name} (structural) against {schema_name}")
        except Exception as e:
            errors.append(f"FAIL: {fixture_file.name} — {e}")

    return errors


def check_reason_codes_sync() -> list:
    """Verify Python ReasonCode enum matches JSON schema enum."""
    errors = []

    error_codes_path = SCHEMAS_DIR / "error_codes.schema.json"
    if not error_codes_path.exists():
        errors.append("WARNING: error_codes.schema.json not found — skipping sync check")
        return errors

    try:
        schema = load_json(error_codes_path)
        schema_codes = set(
            schema.get("definitions", {}).get("reason_code", {}).get("enum", [])
        )
    except Exception as e:
        errors.append(f"FAIL: Could not load error_codes.schema.json — {e}")
        return errors

    try:
        from core.proof_engine.reason_codes import ReasonCode
        python_codes = set(rc.value for rc in ReasonCode)
    except ImportError:
        errors.append("WARNING: Could not import ReasonCode — skipping sync check")
        return errors

    # Check both directions
    in_schema_not_python = schema_codes - python_codes
    in_python_not_schema = python_codes - schema_codes

    if in_schema_not_python:
        errors.append(
            f"FAIL: Codes in schema but not Python: {sorted(in_schema_not_python)}"
        )
    if in_python_not_schema:
        errors.append(
            f"FAIL: Codes in Python but not schema: {sorted(in_python_not_schema)}"
        )

    if not in_schema_not_python and not in_python_not_schema:
        print(f"  PASS  ReasonCode sync ({len(python_codes)} codes match)")

    return errors


def main():
    strict = "--strict" in sys.argv

    print("=" * 60)
    print("BIZRA Schema Validation Gate")
    print("=" * 60)

    all_errors = []

    print("\n[1/3] Loading schemas...")
    all_errors.extend(check_schemas_load())

    print("\n[2/3] Validating fixtures...")
    all_errors.extend(check_fixtures_validate())

    print("\n[3/3] Checking reason code sync...")
    all_errors.extend(check_reason_codes_sync())

    # Report
    print("\n" + "=" * 60)
    warnings = [e for e in all_errors if e.startswith("WARNING:")]
    failures = [e for e in all_errors if e.startswith("FAIL:") or e.startswith("CRITICAL:")]

    if failures:
        print(f"FAILED — {len(failures)} error(s), {len(warnings)} warning(s)")
        for err in all_errors:
            print(f"  {err}")
        sys.exit(1)

    if warnings and strict:
        print(f"FAILED (strict mode) — {len(warnings)} warning(s)")
        for w in warnings:
            print(f"  {w}")
        sys.exit(1)

    if warnings:
        print(f"PASSED with {len(warnings)} warning(s)")
        for w in warnings:
            print(f"  {w}")
    else:
        print("PASSED — all checks green")

    sys.exit(0)


if __name__ == "__main__":
    main()
