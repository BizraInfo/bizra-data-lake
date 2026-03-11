"""
Schema Validator — JSON Schema validation for Spearpoint artifacts.

Validates receipts, reasoning graphs, attestation envelopes, and other
artifacts against their canonical JSON schemas. Prevents truth drift
by enforcing structural contracts at runtime.

Standing on Giants:
- JSON Schema (draft 2020-12)
- OWASP input validation
- BIZRA Spearpoint PRD SP-001: "schemas first, so verified cannot drift"
"""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)

# Schema directory (relative to project root)
_SCHEMA_DIR = Path(__file__).parent.parent.parent / "schemas"

# Cache loaded schemas
_schema_cache: Dict[str, Dict[str, Any]] = {}


def _load_schema(schema_name: str) -> Dict[str, Any]:
    """Load a JSON schema by name (cached)."""
    if schema_name in _schema_cache:
        return _schema_cache[schema_name]

    schema_file = _SCHEMA_DIR / f"{schema_name}.schema.json"
    if not schema_file.exists():
        raise FileNotFoundError(f"Schema not found: {schema_file}")

    schema = json.loads(schema_file.read_text(encoding="utf-8"))
    _schema_cache[schema_name] = schema
    return schema


def validate(
    data: Dict[str, Any],
    schema_name: str,
) -> Tuple[bool, List[str]]:
    """
    Validate data against a named JSON schema.

    Args:
        data: The dict to validate
        schema_name: Schema name (e.g., "receipt", "reasoning_graph", "attestation")

    Returns:
        (is_valid, errors) — errors is empty list on success
    """
    schema = _load_schema(schema_name)
    errors: List[str] = []

    # Try jsonschema if available (full validation)
    try:
        import jsonschema  # type: ignore[import-untyped]

        try:
            jsonschema.validate(instance=data, schema=schema)
            return True, []
        except jsonschema.ValidationError as e:
            errors.append(f"{e.json_path}: {e.message}")
            return False, errors
        except jsonschema.SchemaError as e:
            errors.append(f"Schema error: {e.message}")
            return False, errors
    except ImportError:
        pass

    # Fallback: lightweight structural validation (no jsonschema dependency)
    return _validate_structural(data, schema, errors)


def _validate_value(
    value: Any,
    prop_schema: Dict[str, Any],
    errors: List[str],
    path: str,
) -> None:
    """Validate a single value against its schema definition (recursive)."""
    expected_type = prop_schema.get("type")

    # Type checking
    if expected_type == "string" and not isinstance(value, str):
        errors.append(f"{path}: expected string, got {type(value).__name__}")
    elif expected_type == "number" and not isinstance(value, (int, float)):
        errors.append(f"{path}: expected number, got {type(value).__name__}")
    elif expected_type == "integer" and not isinstance(value, int):
        errors.append(f"{path}: expected integer, got {type(value).__name__}")
    elif expected_type == "boolean" and not isinstance(value, bool):
        errors.append(f"{path}: expected boolean, got {type(value).__name__}")
    elif expected_type == "array" and not isinstance(value, list):
        errors.append(f"{path}: expected array, got {type(value).__name__}")
    elif expected_type == "object" and not isinstance(value, dict):
        errors.append(f"{path}: expected object, got {type(value).__name__}")

    # const constraint (exact value match)
    if "const" in prop_schema and value != prop_schema["const"]:
        errors.append(
            f"{path}: value {value!r} does not match const {prop_schema['const']!r}"
        )

    # enum constraint
    enum_values = prop_schema.get("enum")
    if enum_values is not None and value not in enum_values:
        errors.append(f"{path}: value {value!r} not in enum {enum_values}")

    # pattern constraint (strings only)
    pattern = prop_schema.get("pattern")
    if pattern and isinstance(value, str):
        if not re.fullmatch(pattern, value):
            errors.append(f"{path}: value {value!r} does not match pattern {pattern!r}")

    # minimum / maximum (numbers)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in prop_schema and value < prop_schema["minimum"]:
            errors.append(f"{path}: {value} < minimum {prop_schema['minimum']}")
        if "maximum" in prop_schema and value > prop_schema["maximum"]:
            errors.append(f"{path}: {value} > maximum {prop_schema['maximum']}")

    # minItems / maxItems (arrays)
    if isinstance(value, list):
        if "minItems" in prop_schema and len(value) < prop_schema["minItems"]:
            errors.append(
                f"{path}: array has {len(value)} items, minimum {prop_schema['minItems']}"
            )
        if "maxItems" in prop_schema and len(value) > prop_schema["maxItems"]:
            errors.append(
                f"{path}: array has {len(value)} items, maximum {prop_schema['maxItems']}"
            )
        # Validate each array item against items schema
        items_schema = prop_schema.get("items")
        if items_schema and isinstance(items_schema, dict):
            for i, item in enumerate(value):
                item_path = f"{path}[{i}]"
                _validate_value(item, items_schema, errors, item_path)
                # Recurse into object items
                if isinstance(item, dict) and items_schema.get("type") == "object":
                    _validate_object(item, items_schema, errors, item_path)

    # Recurse into nested objects
    if isinstance(value, dict) and expected_type == "object":
        _validate_object(value, prop_schema, errors, path)


def _validate_object(
    data: Dict[str, Any],
    schema: Dict[str, Any],
    errors: List[str],
    path: str,
) -> None:
    """Validate an object against its schema (required fields, properties, additionalProperties)."""
    # Check required fields
    for field in schema.get("required", []):
        if field not in data:
            errors.append(f"{path}: missing required field '{field}'")

    # Check additionalProperties constraint
    if schema.get("additionalProperties") is False:
        allowed = set(schema.get("properties", {}).keys())
        extra = set(data.keys()) - allowed
        for field in sorted(extra):
            errors.append(f"{path}: additional property '{field}' not allowed")

    # Validate each property against its sub-schema
    properties = schema.get("properties", {})
    for field, prop_schema in properties.items():
        if field not in data:
            continue
        _validate_value(data[field], prop_schema, errors, f"{path}.{field}")


def _check_conditional(
    data: Dict[str, Any],
    schema: Dict[str, Any],
    errors: List[str],
    path: str,
) -> None:
    """Evaluate if/then/else conditional schema rules."""
    if_schema = schema.get("if")
    then_schema = schema.get("then")
    else_schema = schema.get("else")

    if if_schema is None:
        return

    # Evaluate the "if" condition: check if all property constraints match
    condition_met = True
    if_props = if_schema.get("properties", {})
    for field, constraint in if_props.items():
        if field not in data:
            condition_met = False
            break
        value = data[field]
        if "enum" in constraint and value not in constraint["enum"]:
            condition_met = False
            break
        if "const" in constraint and value != constraint["const"]:
            condition_met = False
            break

    if condition_met and then_schema:
        # Apply "then" constraints
        for field in then_schema.get("required", []):
            if field not in data:
                errors.append(f"{path}: missing required field '{field}' (conditional)")
        then_props = then_schema.get("properties", {})
        for field, prop_constraint in then_props.items():
            if field in data:
                _validate_value(data[field], prop_constraint, errors, f"{path}.{field}")
    elif not condition_met and else_schema:
        # Apply "else" constraints
        for field in else_schema.get("required", []):
            if field not in data:
                errors.append(f"{path}: missing required field '{field}' (conditional)")


def _validate_structural(
    data: Dict[str, Any],
    schema: Dict[str, Any],
    errors: List[str],
    path: str = "$",
) -> Tuple[bool, List[str]]:
    """Lightweight structural validation without jsonschema dependency.

    Supports: required, properties (recursive), type, const, enum, pattern,
    minimum/maximum, minItems/maxItems, additionalProperties, if/then/else.
    """
    _validate_object(data, schema, errors, path)
    _check_conditional(data, schema, errors, path)

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_receipt(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a receipt against the receipt schema."""
    return validate(data, "receipt")


def validate_reasoning_graph(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate a reasoning graph against the reasoning_graph schema."""
    return validate(data, "reasoning_graph")


def validate_attestation(data: Dict[str, Any]) -> Tuple[bool, List[str]]:
    """Validate an attestation envelope against the attestation schema."""
    return validate(data, "attestation")


def list_schemas() -> List[str]:
    """List all available schema names."""
    if not _SCHEMA_DIR.exists():
        return []
    return [
        f.stem.replace(".schema", "") for f in sorted(_SCHEMA_DIR.glob("*.schema.json"))
    ]


__all__ = [
    "validate",
    "validate_receipt",
    "validate_reasoning_graph",
    "validate_attestation",
    "list_schemas",
]
