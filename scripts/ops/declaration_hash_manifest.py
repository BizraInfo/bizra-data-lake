from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from core.proof_engine.canonical import blake3_digest

DEFAULT_PATHS = (
    Path("00_CONSTITUTION/DECLARATION.md"),
    Path("docs/DECLARATION_OF_DIGITAL_SOVEREIGNTY.md"),
)


def build_declaration_manifest(paths: list[Path]) -> dict[str, Any]:
    documents: list[dict[str, Any]] = []

    for path in paths:
        content = path.read_bytes()
        documents.append(
            {
                "path": path.as_posix(),
                "bytes": len(content),
                "blake2b_256": hashlib.blake2b(content, digest_size=32).hexdigest(),
                "blake3": blake3_digest(content).hex(),
            }
        )

    return {"documents": documents}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Print BLAKE2b/BLAKE3 publication hashes for declaration artifacts."
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="Optional declaration paths. Defaults to canonical and public declaration docs.",
    )
    args = parser.parse_args()

    paths = [Path(item) for item in args.paths] if args.paths else list(DEFAULT_PATHS)
    manifest = build_declaration_manifest(paths)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
