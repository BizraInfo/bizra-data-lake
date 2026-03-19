#!/usr/bin/env python3
"""Provider readiness preflight: fast scan before expensive compile."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from normalizers import (  # noqa: E402
    COLLECTION_GAP,
    CORE8,
    EXPORTABLE_NOW,
    _detect_schema_provider,
    detect_provider,
)


def _iter_json_files(paths: list[Path], limit: int = 0) -> list[Path]:
    allowed_suffixes = {".json", ".jsonl"}
    out: list[Path] = []
    for base in paths:
        if base.is_file() and base.suffix.lower() in allowed_suffixes:
            out.append(base)
        elif base.is_dir():
            out.extend(
                sorted(
                    path
                    for path in base.rglob("*")
                    if path.is_file() and path.suffix.lower() in allowed_suffixes
                )
            )
        if limit and len(out) >= limit:
            break
    return out


def _read_payload(path: Path) -> Any:
    raw = path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() != ".jsonl":
        return json.loads(raw)
    rows: list[dict[str, Any]] = []
    for line in raw.splitlines():
        text = line.strip()
        if not text:
            continue
        try:
            item = json.loads(text)
        except json.JSONDecodeError:
            continue
        if isinstance(item, dict):
            rows.append(item)
    return rows


def scan_provider_readiness(
    paths: list[Path],
    required: set[str] | None = None,
) -> dict[str, Any]:
    """Scan JSON files and return provider readiness report."""
    if required is None:
        required = set(CORE8)

    t0 = time.monotonic()
    files = _iter_json_files(paths)
    found: dict[str, list[str]] = {p: [] for p in sorted(required)}
    schema_hits: dict[str, int] = {}
    signal_hits: dict[str, int] = {}
    unknown_count = 0
    error_count = 0

    for fp in files:
        try:
            payload = _read_payload(fp)
        except Exception:
            error_count += 1
            continue

        schema = _detect_schema_provider(payload)
        if schema:
            schema_hits[schema] = schema_hits.get(schema, 0) + 1
            if schema in found:
                if len(found[schema]) < 3:
                    found[schema].append(str(fp))
            continue

        detected = detect_provider(payload, source_path=str(fp))
        if detected and detected != "unknown":
            signal_hits[detected] = signal_hits.get(detected, 0) + 1
            if detected in found:
                if len(found[detected]) < 3:
                    found[detected].append(str(fp))
        else:
            unknown_count += 1

    present = {p for p in required if schema_hits.get(p, 0) + signal_hits.get(p, 0) > 0}
    missing = sorted(required - present)
    elapsed = round(time.monotonic() - t0, 3)

    return {
        "files_scanned": len(files),
        "errors": error_count,
        "unknown": unknown_count,
        "elapsed_sec": elapsed,
        "required_providers": sorted(required),
        "present_providers": sorted(present),
        "missing_providers": missing,
        "schema_verified": dict(sorted(schema_hits.items())),
        "signal_only": dict(sorted(signal_hits.items())),
        "sample_paths": {p: paths for p, paths in found.items() if paths},
        "ready": len(missing) == 0,
        "cv_achievable": round(len(present) / max(len(required), 1), 4),
    }


def _format_readiness(result: dict[str, Any]) -> str:
    lines = [
        "╔══════════════════════════════════════╗",
        "║   BIZRA Provider Readiness Preflight ║",
        "╚══════════════════════════════════════╝",
        "",
        f"Files scanned: {result['files_scanned']}  ({result['elapsed_sec']}s)",
        f"Errors: {result['errors']}  Unknown: {result['unknown']}",
        f"CV achievable: {result['cv_achievable']}",
        "",
    ]

    for provider in result["required_providers"]:
        schema = result["schema_verified"].get(provider, 0)
        signal = result["signal_only"].get(provider, 0)
        total = schema + signal
        if total > 0:
            tag = f"✅ {provider:12s}  schema={schema:>4d}  signal={signal:>4d}"
        else:
            is_gap = provider in COLLECTION_GAP
            if is_gap:
                tag = f"⏳ {provider:12s}  COLLECTION GAP — no native export (PAT required)"
            else:
                tag = f"❌ {provider:12s}  MISSING — export exists but not found in intake"
        lines.append(tag)

    lines.append("")
    if result["ready"]:
        lines.append("STATUS: READY — all required providers present")
    else:
        missing = result["missing_providers"]
        gaps = [p for p in missing if p in COLLECTION_GAP]
        real_missing = [p for p in missing if p not in COLLECTION_GAP]
        if real_missing:
            lines.append(
                f"STATUS: NOT READY — missing exports: {', '.join(real_missing)}"
            )
        if gaps:
            lines.append(
                f"COLLECTION GAPS: {', '.join(gaps)} — requires PAT-based scraping"
            )
        if not real_missing and gaps:
            lines.append(
                "TIP: Use --available-only to gate on exportable providers only"
            )

    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description="Provider readiness preflight check")
    parser.add_argument("paths", nargs="+", help="Corpus directories/files to scan")
    parser.add_argument("--json", action="store_true", help="Machine-readable output")
    parser.add_argument(
        "--required",
        type=str,
        default="",
        help="Comma-separated required providers (default: CORE8)",
    )
    parser.add_argument(
        "--available-only",
        action="store_true",
        help="Gate only on exportable providers (skip collection-gap platforms)",
    )
    args = parser.parse_args()

    scan_paths = [Path(p).expanduser().resolve() for p in args.paths]
    required = None
    if args.required:
        required = {p.strip().lower() for p in args.required.split(",") if p.strip()}
    elif args.available_only:
        required = set(EXPORTABLE_NOW)

    result = scan_provider_readiness(scan_paths, required=required)

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print(_format_readiness(result))

    return 0 if result["ready"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
