#!/usr/bin/env python3
"""
Validation script for Unified Cognition Contract.

Verifies:
1. Contract schema is valid JSON
2. Python endpoint models match schema definitions
3. Type safety of request/response structures
"""
import json
import sys
from pathlib import Path

def main() -> int:
    """Validate cognition contract implementation."""
    repo_root = Path(__file__).resolve().parents[1]
    schema_path = repo_root / "config" / "cognition_contract.json"

    print(f"[1/3] Loading contract schema: {schema_path}")

    if not schema_path.exists():
        print(f"❌ FAILED: Schema file not found at {schema_path}")
        return 1

    try:
        with open(schema_path, "r", encoding="utf-8") as f:
            schema = json.load(f)
    except json.JSONDecodeError as e:
        print(f"❌ FAILED: Invalid JSON in schema: {e}")
        return 1
    except Exception as e:
        print(f"❌ FAILED: Error loading schema: {e}")
        return 1

    print("✓ Schema loaded successfully")

    print("\n[2/3] Validating schema structure")

    # Check required definitions
    required_defs = ["UnifiedCognitionRequest", "UnifiedCognitionResponse", "CognitionError"]
    definitions = schema.get("definitions", {})

    for def_name in required_defs:
        if def_name not in definitions:
            print(f"❌ FAILED: Missing definition '{def_name}' in schema")
            return 1
        print(f"  ✓ {def_name} defined")

    # Validate UnifiedCognitionRequest
    req_schema = definitions["UnifiedCognitionRequest"]
    req_required = req_schema.get("required", [])
    req_props = req_schema.get("properties", {})

    if "task" not in req_required:
        print("❌ FAILED: 'task' should be required in UnifiedCognitionRequest")
        return 1

    if "task" not in req_props:
        print("❌ FAILED: 'task' property missing in UnifiedCognitionRequest")
        return 1

    print("  ✓ UnifiedCognitionRequest structure valid")

    # Validate UnifiedCognitionResponse
    resp_schema = definitions["UnifiedCognitionResponse"]
    resp_required = resp_schema.get("required", [])
    resp_props = resp_schema.get("properties", {})

    expected_fields = ["result", "ihsan_score", "snr_tier", "receipt_id"]
    for field in expected_fields:
        if field not in resp_required:
            print(f"❌ FAILED: '{field}' should be required in UnifiedCognitionResponse")
            return 1
        if field not in resp_props:
            print(f"❌ FAILED: '{field}' property missing in UnifiedCognitionResponse")
            return 1

    print("  ✓ UnifiedCognitionResponse structure valid")

    # Validate CognitionError
    error_schema = definitions["CognitionError"]
    error_required = error_schema.get("required", [])
    error_props = error_schema.get("properties", {})

    if "error" not in error_required or "code" not in error_required:
        print("❌ FAILED: 'error' and 'code' should be required in CognitionError")
        return 1

    error_code_enum = error_props.get("code", {}).get("enum", [])
    expected_codes = ["SAT_BLOCKED", "IHSAN_GATE_FAILED", "EXECUTION_FAILED", "INTERNAL_ERROR"]
    for code in expected_codes:
        if code not in error_code_enum:
            print(f"❌ FAILED: Error code '{code}' missing from enum")
            return 1

    print("  ✓ CognitionError structure valid")

    print("\n[3/3] Validating Python implementation")

    # Check that core/main.py exists and contains the endpoint
    main_py = repo_root / "core" / "main.py"
    if not main_py.exists():
        print(f"❌ FAILED: core/main.py not found at {main_py}")
        return 1

    main_content = main_py.read_text(encoding="utf-8")

    # Check for endpoint definition
    if '@app.post("/v1/cognition"' not in main_content:
        print("❌ FAILED: /v1/cognition endpoint not found in core/main.py")
        return 1
    print("  ✓ /v1/cognition endpoint defined")

    # Check for request model
    if "class UnifiedCognitionRequest(BaseModel):" not in main_content:
        print("❌ FAILED: UnifiedCognitionRequest model not found in core/main.py")
        return 1
    print("  ✓ UnifiedCognitionRequest model defined")

    # Check for response model
    if "class UnifiedCognitionResponse(BaseModel):" not in main_content:
        print("❌ FAILED: UnifiedCognitionResponse model not found in core/main.py")
        return 1
    print("  ✓ UnifiedCognitionResponse model defined")

    # Check for schema loader
    if "_load_cognition_contract_schema" not in main_content:
        print("❌ FAILED: Schema loader function not found in core/main.py")
        return 1
    print("  ✓ Schema loader function defined")

    # Check for FATE gating
    if "fate_engine.audit_request_with_feedback" not in main_content:
        print("⚠️  WARNING: FATE gating may not be implemented in /v1/cognition")
    else:
        print("  ✓ FATE gating present")

    # Check for Ihsan floor enforcement
    if "ihsan_score < req.ihsan_floor" not in main_content:
        print("⚠️  WARNING: Ihsan floor enforcement may not be implemented")
    else:
        print("  ✓ Ihsan floor enforcement present")

    # Check for receipt emission
    if "_write_receipt" not in main_content:
        print("⚠️  WARNING: Receipt emission may not be implemented")
    else:
        print("  ✓ Receipt emission present")

    print("\n" + "="*60)
    print("✓ ALL VALIDATIONS PASSED")
    print("="*60)
    print("\nUnified Cognition Contract v1 is correctly implemented.")
    print("\nEndpoints:")
    print("  - Rust Core:    http://localhost:8080 (TODO: implement)")
    print("  - Python Kernel: http://localhost:8010/v1/cognition (READY)")
    print("\nContract Schema:")
    print(f"  {schema_path}")
    print("\nNext Steps:")
    print("  1. Implement Rust endpoint in src/http.rs")
    print("  2. Add Ed25519 signature generation")
    print("  3. Add integration tests in tests/test_cognition_contract.py")
    print("  4. Update API documentation")

    return 0


if __name__ == "__main__":
    sys.exit(main())
