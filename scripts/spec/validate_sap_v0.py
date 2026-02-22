#!/usr/bin/env python3
"""SAP v0 conformance validator.

Deterministic fixture validation for:
- 9 canonical schemas
- strict v0 session-limit ceilings (50/300/65536)
- 12 required negative scenarios
- wire mapping over existing verbs only

Exit 0: all positives pass and all negatives fail.
Exit 1: any mismatch.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

try:
    from jsonschema import Draft202012Validator
except ImportError as exc:  # pragma: no cover
    raise SystemExit("FATAL: install jsonschema>=4.20") from exc

REPO = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO / "schemas" / "sap" / "v0"
POS_DIR = REPO / "tests" / "conformance" / "sap_v0" / "positive"
NEG_DIR = REPO / "tests" / "conformance" / "sap_v0" / "negative"

INTENT_SCOPE: dict[str, set[str]] = {
    "discover": {"product_discovery"},
    "compare": {"price_comparison"},
    "offer": {"purchase_negotiation", "price_comparison"},
    "counter_offer": {"purchase_negotiation"},
    "accept": {"purchase_negotiation"},
    "reject": {"purchase_negotiation"},
    "disclose": {"preference_sharing"},
    "consent": {"preference_sharing", "identity_verification"},
    "close": set(),
}

ALLOWED_VERBS = {
    "PLAN_ACTION",
    "RUN_ACTION",
    "ACTION_STATUS",
    "ACTION_HISTORY",
    "EXPLAIN",
}

ALLOWED_SAP_TYPES = {
    "MeetOpen",
    "MeetMessage",
    "Offer",
    "Disclosure",
    "ConsentReceipt",
    "OutcomeReceipt",
    "RedlineViolation",
}

REQUIRED_POSITIVE = {
    "01_agent_card_valid.json",
    "02_permit_envelope_valid.json",
    "03_meet_open_valid.json",
    "04_meet_message_valid.json",
    "05_offer_valid.json",
    "06_disclosure_valid.json",
    "07_consent_receipt_valid.json",
    "08_outcome_receipt_valid.json",
    "09_redline_violation_valid.json",
    "10_end_to_end_wire_mapping_valid.json",
}

REQUIRED_NEGATIVE = {
    "01_agent_card_missing_compilation.json",
    "02_permit_capability_escalation.json",
    "03_out_of_scope_request.json",
    "04_offer_without_provenance.json",
    "05_tampered_receipt_chain.json",
    "06_expired_meet_open.json",
    "07_invalid_role_pairing.json",
    "08_meet_open_exceeds_limits.json",
    "09_data_sharing_without_consent.json",
    "10_missing_revocation_endpoint.json",
    "11_invalid_ihsan_threshold.json",
    "12_noncanonical_shape_rejected.json",
}


def load_schemas(schema_dir: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for path in sorted(schema_dir.glob("*.schema.json")):
        out[path.name] = json.loads(path.read_text(encoding="utf-8"))
    return out


def _check_required_fixture_set(fixtures_dir: Path, required: set[str]) -> list[str]:
    existing = {p.name for p in fixtures_dir.glob("*.json")}
    errors: list[str] = []

    missing = sorted(required - existing)
    extra = sorted(existing - required)

    if missing:
        errors.append(f"missing fixtures in {fixtures_dir}: {missing}")
    if extra:
        errors.append(f"non-canonical extra fixtures in {fixtures_dir}: {extra}")

    return errors


def _validate_wire_mapping(instance: dict[str, Any]) -> list[str]:
    errs: list[str] = []

    profile = instance.get("profile")
    if profile != "sap-ads-retail-v0":
        errs.append("WIRE_PROFILE_INVALID")

    steps = instance.get("steps", [])
    if not isinstance(steps, list) or not steps:
        errs.append("WIRE_STEPS_INVALID")
        return errs

    for idx, step in enumerate(steps):
        if not isinstance(step, dict):
            errs.append(f"WIRE_STEP_{idx}_NOT_OBJECT")
            continue
        verb = step.get("verb")
        if verb not in ALLOWED_VERBS:
            errs.append(f"WIRE_STEP_{idx}_INVALID_VERB:{verb}")
        sap_type = step.get("sap_type")
        if sap_type is not None and sap_type not in ALLOWED_SAP_TYPES:
            errs.append(f"WIRE_STEP_{idx}_INVALID_SAP_TYPE:{sap_type}")

    return errs


def cross_validate(
    schema_name: str,
    instance: dict[str, Any],
    meta: dict[str, Any],
) -> list[str]:
    errs: list[str] = []

    if schema_name == "wire_mapping":
        return _validate_wire_mapping(instance)

    if schema_name == "permit_envelope.schema.json":
        parent_caps = set(meta.get("parent_capabilities", []))
        if parent_caps:
            child_caps = set(instance.get("capabilities", []))
            if not child_caps.issubset(parent_caps):
                errs.append("CAPABILITY_ESCALATION")
            if meta.get("require_strict_subset") and child_caps == parent_caps:
                errs.append("NOT_STRICT_SUBSET")

        parent_limits = meta.get("parent_limit_caps")
        if isinstance(parent_limits, dict):
            child_limits = instance.get("limits", {})
            for k, v in parent_limits.items():
                cv = child_limits.get(k)
                if isinstance(v, (int, float)) and isinstance(cv, (int, float)) and cv > v:
                    errs.append(f"LIMIT_ESCALATION:{k}")

    if schema_name == "meet_open.schema.json":
        limits = instance.get("session_limits", {})
        if isinstance(limits, dict):
            mm = limits.get("max_messages")
            md = limits.get("max_duration_seconds")
            mp = limits.get("max_payload_bytes")
            if isinstance(mm, int) and mm > 50:
                errs.append("SESSION_LIMIT_MAX_MESSAGES_EXCEEDED")
            if isinstance(md, int) and md > 300:
                errs.append("SESSION_LIMIT_MAX_DURATION_EXCEEDED")
            if isinstance(mp, int) and mp > 65536:
                errs.append("SESSION_LIMIT_MAX_PAYLOAD_EXCEEDED")

        now = meta.get("now")
        if isinstance(now, (int, float)) and instance.get("expires_at", 0) <= now:
            errs.append("EXPIRED_MEET_OPEN")

        roles = meta.get("roles", [])
        if roles and "user_proxy" not in roles:
            errs.append("INVALID_ROLE_PAIRING")

    if schema_name == "meet_message.schema.json":
        scope = set(meta.get("session_consent_scope", []))
        if scope:
            intent = instance.get("intent", "")
            required = INTENT_SCOPE.get(intent, set())
            if required and not scope.intersection(required):
                errs.append("OUT_OF_CONSENT_SCOPE")

    if schema_name == "offer.schema.json":
        provenance = instance.get("provenance_hashes", [])
        if isinstance(provenance, list) and len(provenance) == 0:
            errs.append("MISSING_PROVENANCE")

        requires_data_shared = bool(meta.get("requires_data_shared", False))
        consent_present = bool(meta.get("consent_present", False))
        if requires_data_shared and not consent_present:
            errs.append("MISSING_CONSENT_RECEIPT")

    if schema_name == "consent_receipt.schema.json":
        if instance.get("data_shared") and not instance.get("consent_hash"):
            errs.append("MISSING_CONSENT_HASH")
        if not instance.get("revocation_endpoint"):
            errs.append("MISSING_REVOCATION_ENDPOINT")

    if schema_name == "outcome_receipt.schema.json":
        expected_prev = meta.get("expected_prev_receipt_hash")
        if expected_prev and instance.get("prev_receipt_hash") != expected_prev:
            errs.append("RECEIPT_CHAIN_TAMPERED")

    return errs


def validate_fixture(path: Path, schemas: dict[str, dict[str, Any]]) -> tuple[bool, list[str], str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []

    schema_name = raw.get("schema")
    instance = raw.get("instance")
    meta = raw.get("meta", {})

    if not isinstance(schema_name, str):
        return False, ["fixture missing string field: schema"], ""
    if not isinstance(instance, dict):
        return False, ["fixture missing object field: instance"], schema_name

    if schema_name == "wire_mapping":
        errors.extend(cross_validate(schema_name, instance, meta if isinstance(meta, dict) else {}))
        return len(errors) == 0, errors, schema_name

    schema = schemas.get(schema_name)
    if schema is None:
        return False, [f"unknown schema: {schema_name}"], schema_name

    for e in Draft202012Validator(schema).iter_errors(instance):
        errors.append(f"schema: {e.message}")

    if isinstance(meta, dict):
        errors.extend(cross_validate(schema_name, instance, meta))
    else:
        errors.append("meta must be object when present")

    return len(errors) == 0, errors, schema_name


def run(schemas_dir: Path) -> int:
    schemas = load_schemas(schemas_dir)
    if len(schemas) != 9:
        print(f"ERROR: expected 9 schemas, found {len(schemas)}")
        return 1

    structural_errors = []
    structural_errors.extend(_check_required_fixture_set(POS_DIR, REQUIRED_POSITIVE))
    structural_errors.extend(_check_required_fixture_set(NEG_DIR, REQUIRED_NEGATIVE))
    if structural_errors:
        print("Fixture set errors:")
        for err in structural_errors:
            print(f"  - {err}")
        return 1

    print(f"Loaded {len(schemas)} schemas from {schemas_dir}")

    passed = 0
    failed = 0
    total = 0

    print("\n=== POSITIVE FIXTURES (expect PASS) ===")
    for path in sorted(POS_DIR.glob("*.json")):
        total += 1
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("expect_valid") is not True:
            print(f"  FAIL  {path.name} (expect_valid must be true)")
            failed += 1
            continue

        is_valid, errs, _ = validate_fixture(path, schemas)
        if is_valid:
            print(f"  PASS  {path.name}")
            passed += 1
        else:
            print(f"  FAIL  {path.name}")
            for err in errs:
                print(f"        -> {err}")
            failed += 1

    print("\n=== NEGATIVE FIXTURES (expect FAIL) ===")
    for path in sorted(NEG_DIR.glob("*.json")):
        total += 1
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw.get("expect_valid") is not False:
            print(f"  FAIL  {path.name} (expect_valid must be false)")
            failed += 1
            continue

        is_valid, errs, _ = validate_fixture(path, schemas)
        if not is_valid:
            print(f"  PASS  {path.name} (correctly rejected)")
            for err in errs[:3]:
                print(f"        -> {err}")
            passed += 1
        else:
            print(f"  FAIL  {path.name} (should have failed)")
            failed += 1

    print("\n========================================================")
    print(f"  Total: {total}  |  Passed: {passed}  |  Failed: {failed}")
    print("========================================================")

    if failed:
        print("\nRESULT: CONFORMANCE FAILED")
        return 1

    print("\nRESULT: ALL CONFORMANCE CHECKS PASSED")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="SAP v0 conformance validator")
    parser.add_argument("--schemas-dir", type=Path, default=SCHEMA_DIR)
    args = parser.parse_args()
    raise SystemExit(run(args.schemas_dir))


if __name__ == "__main__":
    main()
