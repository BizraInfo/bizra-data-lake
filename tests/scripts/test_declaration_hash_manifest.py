from __future__ import annotations

import hashlib
from pathlib import Path

from core.proof_engine.canonical import blake3_digest
from scripts.ops.declaration_hash_manifest import build_declaration_manifest


def test_build_declaration_manifest_reports_expected_hashes(tmp_path: Path) -> None:
    first = tmp_path / "DECLARATION.md"
    second = tmp_path / "DECLARATION_OF_DIGITAL_SOVEREIGNTY.md"
    first.write_text("alpha\n", encoding="utf-8")
    second.write_text("beta\n", encoding="utf-8")

    manifest = build_declaration_manifest([first, second])

    assert manifest == {
        "documents": [
            {
                "path": first.as_posix(),
                "bytes": 6,
                "blake2b_256": hashlib.blake2b(
                    b"alpha\n",
                    digest_size=32,
                ).hexdigest(),
                "blake3": blake3_digest(b"alpha\n").hex(),
            },
            {
                "path": second.as_posix(),
                "bytes": 5,
                "blake2b_256": hashlib.blake2b(
                    b"beta\n",
                    digest_size=32,
                ).hexdigest(),
                "blake3": blake3_digest(b"beta\n").hex(),
            },
        ]
    }
