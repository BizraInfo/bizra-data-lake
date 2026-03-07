"""Export the OpenAPI schema as a static JSON artifact.

Usage:
    python scripts/export_openapi_schema.py [--out path/to/openapi.json]

Elite engineering pattern: the API schema is a CI-enforced artifact.
Frontend codegens against it. Drift is caught before merge.

Standing on Giants:
- Stripe: API-first development with schema-as-contract
- Google: API Design Guide (AIP-121: schema versioning)
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock


def _make_runtime(state_dir: Path) -> MagicMock:
    """Create a minimal mock runtime for schema extraction."""
    runtime = MagicMock()
    runtime.config = SimpleNamespace(state_dir=state_dir)
    runtime.metrics = MagicMock(to_prometheus=lambda include_help=False: "")
    runtime.status.return_value = {
        "health": {
            "status": "healthy",
            "strict_gate": {"enabled": False, "passed": True},
        },
        "identity": {"version": "schema-export"},
        "state": {"running": True},
        "autonomous": {"running": False},
        "pat_sat": {
            "negotiation_receipt_chain": {
                "verified_end_to_end": False,
                "chain_valid": None,
                "total_negotiation_receipts": 0,
                "latest_sequence": None,
                "latest_entry_hash": None,
                "latest_receipt_id": None,
            }
        },
    }
    runtime.query = MagicMock()
    runtime._orchestrator = None
    runtime._node_signer = None
    runtime._evidence_ledger = None
    return runtime


def export_schema(out_path: str | None = None) -> dict:
    """Export the OpenAPI schema from the live FastAPI app."""
    os.environ.setdefault(
        "BIZRA_USERSTORE_MASTER_SECRET", "schema-export-ephemeral"
    )

    with tempfile.TemporaryDirectory() as td:
        runtime = _make_runtime(Path(td))

        from core.sovereign.api import create_fastapi_app

        app = create_fastapi_app(runtime)
        schema = app.openapi()

    if out_path:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text(json.dumps(schema, indent=2) + "\n")
        print(f"OpenAPI schema exported to {out_path}")
        print(f"  Version: {schema['info']['version']}")
        print(f"  Paths: {len(schema['paths'])}")
        schemas = schema.get("components", {}).get("schemas", {})
        print(f"  Models: {len(schemas)}")
        tags = [t["name"] for t in schema.get("tags", [])]
        print(f"  Tags: {len(tags)} ({', '.join(tags)})")

    return schema


def main() -> int:
    out = "docs/openapi.json"
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--out" and i + 1 < len(sys.argv) - 1:
            out = sys.argv[i + 2]

    try:
        export_schema(out)
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
