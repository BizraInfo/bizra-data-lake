"""Generate TypeScript types from the OpenAPI schema.

Usage:
    python scripts/generate_frontend_types.py

Reads docs/openapi.json and writes frontend/api-types.ts.
Eliminates hand-maintained frontend types — single source of truth.
"""

from __future__ import annotations

import json
from pathlib import Path


def ts_type(prop_schema: object) -> str:
    """Convert a JSON Schema property to a TypeScript type."""
    if not isinstance(prop_schema, dict):
        return "unknown"
    ref = prop_schema.get("$ref", "")
    if ref:
        return ref.split("/")[-1]
    all_of = prop_schema.get("allOf", [])
    if all_of:
        return ts_type(all_of[0])
    any_of = prop_schema.get("anyOf", [])
    if any_of:
        types = [
            ts_type(s)
            for s in any_of
            if isinstance(s, dict) and s.get("type") != "null"
        ]
        if len(types) == 1:
            return types[0]
        return " | ".join(types) if types else "unknown"
    t = prop_schema.get("type", "any")
    if t == "string":
        return "string"
    if t in ("integer", "number"):
        return "number"
    if t == "boolean":
        return "boolean"
    if t == "null":
        return "null"
    if t == "array":
        items = prop_schema.get("items", {})
        return f"{ts_type(items)}[]"
    if t == "object":
        add = prop_schema.get("additionalProperties")
        if isinstance(add, dict):
            return f"Record<string, {ts_type(add)}>"
        return "Record<string, unknown>"
    return "unknown"


def generate(schema: dict) -> str:
    """Generate TypeScript source from an OpenAPI schema dict."""
    schemas = schema.get("components", {}).get("schemas", {})
    version = schema["info"]["version"]
    path_count = len(schema["paths"])
    model_count = len(schemas)
    tag_count = len(schema.get("tags", []))

    lines: list[str] = [
        "/**",
        f" * BIZRA Sovereign API — TypeScript Client Types (v{version})",
        " *",
        " * Auto-generated from docs/openapi.json",
        " * DO NOT EDIT — regenerate with:",
        " *   python scripts/generate_frontend_types.py",
        " *",
        f" * {path_count} routes | {model_count} models | {tag_count} domains",
        " */",
        "",
        "// ═══════════════════════════════════════════════════════════",
        "// Request / Response Models",
        "// ═══════════════════════════════════════════════════════════",
        "",
    ]

    for name, model in sorted(schemas.items()):
        if name in ("HTTPValidationError", "ValidationError"):
            continue

        props = model.get("properties", {})
        required = set(model.get("required", []))
        desc = model.get("description", "")

        if desc:
            lines.append(f"/** {desc} */")
        lines.append(f"export interface {name} {{")

        for pname, pschema in props.items():
            optional = "" if pname in required else "?"
            pdesc = pschema.get("description", "") if isinstance(pschema, dict) else ""
            ptype = ts_type(pschema)
            if pdesc:
                lines.append(f"  /** {pdesc} */")
            lines.append(f"  {pname}{optional}: {ptype};")

        lines.append("}")
        lines.append("")

    # API endpoint constants
    lines.extend(
        [
            "// ═══════════════════════════════════════════════════════════",
            "// API Endpoint Paths",
            "// ═══════════════════════════════════════════════════════════",
            "",
            'export const API_BASE = "/v1";',
            "",
            "export const API_ENDPOINTS = {",
        ]
    )

    for path, methods in sorted(schema["paths"].items()):
        for method in sorted(methods.keys()):
            if method in ("options", "head"):
                continue
            clean = (
                path.replace("/v1/", "")
                .replace("/", "_")
                .replace("{", "")
                .replace("}", "")
                .replace("-", "_")
            )
            const_name = f"{method.upper()}_{clean}"
            lines.append(
                f'  {const_name}: {{ method: "{method.upper()}", path: "{path}" }},'
            )

    lines.extend(
        [
            "} as const;",
            "",
            "// ═══════════════════════════════════════════════════════════",
            "// Constitutional Thresholds (synced from constants.py)",
            "// ═══════════════════════════════════════════════════════════",
            "",
            "export const THRESHOLDS = {",
            "  IHSAN_PRODUCTION: 0.95,",
            "  SNR_MINIMUM: 0.85,",
            "  SNR_T1_HIGH: 0.95,",
            "  SNR_T0_ELITE: 0.98,",
            "  ADL_GINI_MAX: 0.35,",
            "  API_P99_LATENCY_MS: 200,",
            "} as const;",
            "",
            'export type MissionStatus = "COMPLETE" | "PARTIAL" | "FAILED";',
            "",
            'export type RouteExposure = "public" | "bootstrap_public" | "authenticated";',
            "",
        ]
    )

    return "\n".join(lines)


def main() -> int:
    schema_path = Path("docs/openapi.json")
    out_path = Path("frontend/api-types.ts")

    if not schema_path.exists():
        print(f"ERROR: {schema_path} not found. Run export_openapi_schema.py first.")
        return 1

    schema = json.loads(schema_path.read_text())
    output = generate(schema)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(output)

    iface_count = output.count("export interface ")
    endpoint_count = output.count("method: ")
    print(f"Generated {out_path} ({len(output.splitlines())} lines)")
    print(f"  Interfaces: {iface_count}")
    print(f"  Endpoints: {endpoint_count}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
