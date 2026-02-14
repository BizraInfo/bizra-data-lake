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


def _validate_structural(
    data: Dict[str, Any],
    schema: Dict[str, Any],
    errors: List[str],
    path: str = "$",
) -> Tuple[bool, List[str]]:
    """Lightweight structural validation without jsonschema dependency."""
    # Check required fields
    required = schema.get("required", [])
    for field in required:
        if field not in data:
            errors.append(f"{path}: missing required field '{field}'")

    # Check property types
    properties = schema.get("properties", {})
    for field, prop_schema in properties.items():
        if field not in data:
            continue

        value = data[field]
        expected_type = prop_schema.get("type")

        if expected_type == "string" and not isinstance(value, str):
            errors.append(
                f"{path}.{field}: expected string, got {type(value).__name__}"
            )
        elif expected_type == "number" and not isinstance(value, (int, float)):
            errors.append(
                f"{path}.{field}: expected number, got {type(value).__name__}"
            )
        elif expected_type == "integer" and not isinstance(value, int):
            errors.append(
                f"{path}.{field}: expected integer, got {type(value).__name__}"
            )
        elif expected_type == "boolean" and not isinstance(value, bool):
            errors.append(
                f"{path}.{field}: expected boolean, got {type(value).__name__}"
            )
        elif expected_type == "array" and not isinstance(value, list):
            errors.append(f"{path}.{field}: expected array, got {type(value).__name__}")
        elif expected_type == "object" and not isinstance(value, dict):
            errors.append(
                f"{path}.{field}: expected object, got {type(value).__name__}"
            )

        # Check enum constraints
        enum_values = prop_schema.get("enum")
        if enum_values and value not in enum_values:
            errors.append(f"{path}.{field}: value '{value}' not in enum {enum_values}")

        # Check minimum/maximum
        if isinstance(value, (int, float)):
            if "minimum" in prop_schema and value < prop_schema["minimum"]:
                errors.append(
                    f"{path}.{field}: {value} < minimum {prop_schema['minimum']}"
                )
            if "maximum" in prop_schema and value > prop_schema["maximum"]:
                errors.append(
                    f"{path}.{field}: {value} > maximum {prop_schema['maximum']}"
                )

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
