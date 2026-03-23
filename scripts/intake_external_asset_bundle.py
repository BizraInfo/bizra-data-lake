#!/usr/bin/env python3
"""Govern external bundle intake into a reproducible manifest + triage report.

This turns ad hoc drops like ``B:\\all files`` into a governed intake surface:
every file is hashed, classified, and assigned a recommended action.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]


def _blake3_digest(path: Path) -> str:
    try:
        import blake3  # type: ignore

        hasher = blake3.blake3()
    except ImportError:
        hasher = hashlib.sha256()

    with path.open("rb") as handle:
        while chunk := handle.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()[:32]


@dataclass(frozen=True)
class BundleRecord:
    path: str
    relative_path: str
    name: str
    extension: str
    size_bytes: int
    content_hash: str
    category: str
    recommended_action: str
    rationale: str


def _categorize(path: Path) -> tuple[str, str, str]:
    name = path.name.lower()
    ext = path.suffix.lower()

    if name in {
        "cmn_evidence_redraft.md",
        "proof_kernel.py",
        "proof_kernel_receipt.json",
        "spearpoint_v2.py",
        "spearpoint_v2_receipt.json",
        "adversarial_how.py",
        "adversarial_how_results.json",
        "corpus_governance.py",
        "error_taxonomy.py",
        "release_discipline_patches.md",
        "membrane-tax-gate.yml",
    }:
        return (
            "research_governance_candidate",
            "merge_candidate",
            "Evidence-first paper/proof/governance artifact with direct repo relevance.",
        )

    if name in {
        "bizra_sovereigncockpit.jsx",
        "bizra_sovereignworld.jsx",
        "bizra_sovereignworld_v2.jsx",
        "bizra_frontdoor.jsx",
        "bizra_jarvis_v1.jsx",
        "bizra_jarvis_v2.jsx",
        "bridge.js",
        "index.html",
        "main.jsx",
        "package.json",
        "vite.config.js",
        "phase1_gate.py",
        "readme.md",
    }:
        return (
            "frontend_node0_cockpit",
            "prototype_merge_candidate",
            "Node0 cockpit/front-door prototype; useful for selective merge, not direct drop-in.",
        )

    if name in {
        "bizra_v1_product_spec.docx",
        "gtm_90day_launch_plan.docx",
    }:
        return (
            "business_product_reference",
            "reference_only",
            "Business/product planning material; valuable context but not runtime source of truth.",
        )

    if name in {
        "bizra_brand_identity.html",
        "bizra_brand_identity_fixed.html",
    }:
        return (
            "brand_mockup",
            "archive_reference",
            "Brand/visual artifact; keep as reference unless design system work resumes.",
        )

    if ext in {".json"}:
        return (
            "artifact_receipt",
            "reference_only",
            "Generated artifact/receipt; keep for evidence comparison rather than direct merge.",
        )

    if ext in {".jsx", ".js", ".html", ".md", ".py", ".yml", ".yaml"}:
        return (
            "unclassified_source_bundle",
            "review_manually",
            "Source-like artifact without a bundle-specific rule; requires manual review.",
        )

    return (
        "unclassified_asset",
        "review_manually",
        "Unknown artifact type; inspect manually before acting.",
    )


def build_manifest(bundle_root: Path) -> list[BundleRecord]:
    records: list[BundleRecord] = []
    for path in sorted(p for p in bundle_root.iterdir() if p.is_file()):
        category, action, rationale = _categorize(path)
        records.append(
            BundleRecord(
                path=str(path),
                relative_path=path.relative_to(bundle_root).as_posix(),
                name=path.name,
                extension=path.suffix.lower(),
                size_bytes=path.stat().st_size,
                content_hash=_blake3_digest(path),
                category=category,
                recommended_action=action,
                rationale=rationale,
            )
        )
    return records


def _render_markdown(bundle_root: Path, records: Iterable[BundleRecord]) -> str:
    records = list(records)
    category_counts = Counter(record.category for record in records)
    action_counts = Counter(record.recommended_action for record in records)

    lines = [
        f"# External Bundle Intake — {bundle_root.name}",
        "",
        f"- Root: `{bundle_root}`",
        f"- Files scanned: `{len(records)}`",
        "",
        "## Category Summary",
        "",
        "| Category | Count |",
        "|---|---:|",
    ]
    for category, count in sorted(category_counts.items()):
        lines.append(f"| {category} | {count} |")

    lines.extend(
        [
            "",
            "## Action Summary",
            "",
            "| Recommended Action | Count |",
            "|---|---:|",
        ]
    )
    for action, count in sorted(action_counts.items()):
        lines.append(f"| {action} | {count} |")

    lines.extend(
        [
            "",
            "## File Triage",
            "",
            "| File | Category | Action | Size (bytes) | Rationale |",
            "|---|---|---|---:|---|",
        ]
    )
    for record in records:
        lines.append(
            f"| `{record.relative_path}` | {record.category} | {record.recommended_action} | {record.size_bytes} | {record.rationale} |"
        )

    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description="Intake an external asset bundle.")
    parser.add_argument("--root", type=Path, required=True, help="Bundle directory")
    parser.add_argument(
        "--output-json", type=Path, required=True, help="Manifest JSON output"
    )
    parser.add_argument(
        "--output-md", type=Path, required=True, help="Markdown triage output"
    )
    args = parser.parse_args()

    records = build_manifest(args.root)
    manifest = {
        "bundle_root": str(args.root),
        "file_count": len(records),
        "records": [asdict(record) for record in records],
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    args.output_md.write_text(_render_markdown(args.root, records), encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
