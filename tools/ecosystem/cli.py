from __future__ import annotations

import argparse
from pathlib import Path

from tools.ecosystem.config import load_ecosystem_config
from tools.ecosystem.indexer import build_manifest
from tools.ecosystem.sealer import seal_manifest, write_manifest, write_receipt


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="bizra-ecosystem", description="BIZRA Ecosystem Indexer + Sealer")
    parser.add_argument("--config", default=str(Path("tools/ecosystem/ecosystem_config.yaml")), help="Path to ecosystem_config.yaml")
    parser.add_argument("--out", default="BIZRA_ECOSYSTEM_MANIFEST.json", help="Output manifest path")
    parser.add_argument("--max-projects", type=int, default=500, help="Max projects to index")
    parser.add_argument("--seal", action="store_true", help="Also emit a seal receipt")
    parser.add_argument("--receipt-out", default="docs/evidence/receipts/ecosystem_receipt.json", help="Receipt output path")
    parser.add_argument("--seal-note", default="BIZRA Ecosystem Manifest sealed", help="Seal note")

    args = parser.parse_args(argv)

    repo_root = Path(__file__).resolve().parents[2]
    cfg = load_ecosystem_config(Path(args.config).expanduser().resolve())

    manifest = build_manifest(repo_root=repo_root, cfg=cfg, max_projects=int(args.max_projects))
    manifest_path = write_manifest(manifest, out_path=Path(args.out).expanduser().resolve())

    if args.seal:
        receipt = seal_manifest(manifest_path=manifest_path, seal_note=str(args.seal_note))
        write_receipt(receipt, out_path=Path(args.receipt_out).expanduser().resolve())

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
