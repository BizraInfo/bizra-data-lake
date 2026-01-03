import json
import re
import tempfile
import unittest
from pathlib import Path

from tools.ecosystem.hashing import sha256_canonical_json
from tools.ecosystem.sealer import seal_manifest, write_manifest


class TestEcosystemSeal(unittest.TestCase):
    def test_ecosystem_seal_receipt_integrity_hash_format(self):
        with tempfile.TemporaryDirectory() as d:
            tmp_path = Path(d)

            manifest_no_sha = {
                "schema": "bizra_ecosystem_manifest_v1",
                "metadata": {
                    "bismillah": "بِسْمِ ٱللَّٰهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ",
                    "generated_at": "2025-01-01T00:00:00Z",
                },
                "authority": {
                    "system": "BIZRA",
                    "mode": "LOCAL",
                },
                "canonical_invariants": {
                    "hash": "sha256",
                    "json": "canonical_sorted_keys",
                },
                "architecture_hashes": {},
                "roots": [],
                "projects": [],
            }

            manifest = dict(manifest_no_sha)
            manifest["manifest_sha256"] = sha256_canonical_json(manifest_no_sha)

            out_path = tmp_path / "manifest.json"
            write_manifest(manifest, out_path=out_path)

            receipt = seal_manifest(manifest_path=out_path, seal_note="test")

            self.assertEqual(receipt["schema"], "bizra_ecosystem_receipt_v1")
            self.assertEqual(receipt["manifest_sha256"], manifest["manifest_sha256"])

            integrity_hash = receipt["integrity_hash"]
            self.assertRegex(integrity_hash, r"^sha256:[0-9a-f]{64}$")

    def test_ecosystem_manifest_sha256_deterministic(self):
        # Same logical JSON should hash the same regardless of ordering/whitespace.
        a = {"b": 2, "a": 1, "nested": {"y": 9, "x": 8}}
        b = {"nested": {"x": 8, "y": 9}, "a": 1, "b": 2}

        ha = sha256_canonical_json(a)
        hb = sha256_canonical_json(b)
        self.assertEqual(ha, hb)

        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "a.json"
            p.write_text(json.dumps(a, indent=2), encoding="utf-8")
            loaded = json.loads(p.read_text(encoding="utf-8"))
            self.assertEqual(sha256_canonical_json(loaded), ha)


if __name__ == "__main__":
    unittest.main()
