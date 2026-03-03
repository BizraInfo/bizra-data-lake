#!/usr/bin/env python3
"""Workspace Masterpiece Engine.

Interdisciplinary, multi-lens inventory analysis over full-workspace manifests.
Produces graph-shaped outputs and SNR-ranked domain priorities.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


LENSES = (
    "code",
    "operations",
    "governance",
    "research_docs",
    "data_assets",
    "runtime_state",
    "build_artifacts",
    "unknown",
)

SIGNAL_WEIGHTS = {
    "code": 1.00,
    "operations": 0.95,
    "governance": 1.10,
    "research_docs": 0.85,
    "data_assets": 0.80,
}

NOISE_WEIGHTS = {
    "runtime_state": 1.00,
    "build_artifacts": 0.90,
    "unknown": 0.60,
}

RUNTIME_MARKERS = (
    "/.git/",
    "/.venv/",
    "/.venv-",
    "/.pytest_cache/",
    "/.ruff_cache/",
    "/.mypy_cache/",
    "/__pycache__/",
    "/tmp_state/",
    "/sovereign_state/",
    "/logs/",
    "/.claude-flow/logs/",
    "/.swarm/",
)

BUILD_MARKERS = (
    "/target/",
    "/dist/",
    "/build/",
    "/node_modules/",
    "/artifacts/",
    "/out/",
    "/checkpoints/",
)

GOVERNANCE_MARKERS = (
    "/.github/workflows/",
    "/docs/specs/",
    "/docs/plans/",
    "/schemas/",
    "/deploy/",
)

RESEARCH_MARKERS = (
    "/research_archive/",
    "/00_intake/",
    "/01_raw/",
    "/02_processed/",
    "/03_indexed/",
    "/04_gold/",
    "/06_index/",
    "/99_quarantine/",
)

OPERATIONS_MARKERS = (
    "/scripts/",
    "/tools/",
    "/agents/",
    "/installers/",
)

CODE_DIR_MARKERS = (
    "/core/",
    "/bizra-omega/",
    "/src/",
    "/tests/",
    "/filedfs/",
    "/benchmark/",
    "/benchmark_suite/",
)

CODE_EXTS = {
    ".py",
    ".rs",
    ".js",
    ".jsx",
    ".ts",
    ".tsx",
    ".mjs",
    ".c",
    ".cc",
    ".cpp",
    ".h",
    ".hpp",
    ".go",
    ".java",
    ".kt",
    ".swift",
}

OPS_EXTS = {".sh", ".ps1", ".bat"}
GOV_EXTS = {".yml", ".yaml", ".toml", ".json"}
DOC_EXTS = {".md", ".pdf", ".html", ".rst", ".txt"}
DATA_EXTS = {
    ".csv",
    ".tsv",
    ".parquet",
    ".jsonl",
    ".zip",
    ".gz",
    ".tgz",
    ".tar",
    ".sqlite",
    ".db",
    ".npy",
    ".npz",
}
BUILD_EXTS = {".o", ".so", ".dll", ".exe", ".bin", ".whl", ".dylib", ".a"}

ROOT_GOV_FILES = {
    "readme.md",
    "security.md",
    "contributing.md",
    "code_of_conduct.md",
    "license",
    "pyproject.toml",
    "cargo.toml",
    "docker-compose.yml",
}


@dataclass(frozen=True)
class DomainScore:
    domain: str
    total_files: int
    signal: float
    noise: float
    snr: float


def _normalize_rel_path(rel_path: str) -> str:
    rel = rel_path.strip()
    if rel.startswith("./"):
        rel = rel[2:]
    return rel.replace("\\", "/")


def _top_domain(rel_path: str) -> str:
    if "/" not in rel_path:
        return "<root>"
    return rel_path.split("/", 1)[0]


def classify_path(rel_path: str) -> str:
    rel = _normalize_rel_path(rel_path)
    lower = "/" + rel.lower()
    ext = Path(rel).suffix.lower()
    base = Path(rel).name.lower()

    if any(marker in lower for marker in RUNTIME_MARKERS):
        return "runtime_state"
    if any(marker in lower for marker in BUILD_MARKERS):
        return "build_artifacts"
    if any(marker in lower for marker in RESEARCH_MARKERS):
        return "data_assets"
    if any(marker in lower for marker in GOVERNANCE_MARKERS):
        return "governance"
    if any(marker in lower for marker in CODE_DIR_MARKERS):
        if ext in OPS_EXTS:
            return "operations"
        if ext in GOV_EXTS:
            return "governance"
        if ext in DOC_EXTS:
            return "research_docs"
        if ext in DATA_EXTS:
            return "data_assets"
        return "code"
    if any(marker in lower for marker in OPERATIONS_MARKERS):
        return "operations"

    if base in ROOT_GOV_FILES:
        return "governance"
    if ext in CODE_EXTS:
        return "code"
    if ext in OPS_EXTS:
        return "operations"
    if ext in GOV_EXTS:
        return "governance"
    if ext in DOC_EXTS:
        return "research_docs"
    if ext in DATA_EXTS:
        return "data_assets"
    if ext in BUILD_EXTS:
        return "build_artifacts"

    return "unknown"


def _snr_from_counts(counts: Counter[str]) -> tuple[float, float, float]:
    signal = sum(counts[lens] * weight for lens, weight in SIGNAL_WEIGHTS.items())
    noise = sum(counts[lens] * weight for lens, weight in NOISE_WEIGHTS.items())
    denom = signal + noise
    snr = signal / denom if denom > 0 else 1.0
    return signal, noise, snr


def analyze_inventory(
    files_manifest: Path,
    dirs_manifest: Path | None = None,
    top_n: int = 20,
) -> dict[str, Any]:
    if not files_manifest.exists():
        raise FileNotFoundError(f"Missing files manifest: {files_manifest}")

    global_counts: Counter[str] = Counter()
    domain_counts: dict[str, Counter[str]] = defaultdict(Counter)
    unknown_samples: list[str] = []
    unknown_ext_counts: Counter[str] = Counter()
    total_files = 0

    with files_manifest.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            rel = _normalize_rel_path(line)
            if not rel:
                continue
            total_files += 1
            lens = classify_path(rel)
            domain = _top_domain(rel)
            global_counts[lens] += 1
            domain_counts[domain][lens] += 1
            if lens == "unknown" and len(unknown_samples) < 200:
                unknown_samples.append(rel)
            if lens == "unknown":
                suffix = Path(rel).suffix.lower() or "<none>"
                unknown_ext_counts[suffix] += 1

    total_dirs = 0
    if dirs_manifest and dirs_manifest.exists():
        with dirs_manifest.open("r", encoding="utf-8", errors="replace") as handle:
            for line in handle:
                if _normalize_rel_path(line):
                    total_dirs += 1

    signal, noise, snr = _snr_from_counts(global_counts)

    domain_scores: list[DomainScore] = []
    for domain, counts in domain_counts.items():
        d_signal, d_noise, d_snr = _snr_from_counts(counts)
        domain_scores.append(
            DomainScore(
                domain=domain,
                total_files=sum(counts.values()),
                signal=d_signal,
                noise=d_noise,
                snr=d_snr,
            )
        )

    by_low_snr = sorted(domain_scores, key=lambda row: (row.snr, -row.total_files))
    by_high_signal = sorted(
        domain_scores, key=lambda row: (-row.signal, -row.total_files, row.domain)
    )
    unknown_hotspots = sorted(
        (
            {
                "domain": domain,
                "unknown_files": counts.get("unknown", 0),
                "total_files": sum(counts.values()),
            }
            for domain, counts in domain_counts.items()
            if counts.get("unknown", 0) > 0
        ),
        key=lambda row: (-row["unknown_files"], -row["total_files"], row["domain"]),
    )

    action_plan = []
    runtime_share = (
        (global_counts.get("runtime_state", 0) + global_counts.get("build_artifacts", 0))
        / total_files
        if total_files
        else 0.0
    )
    if runtime_share > 0.20:
        action_plan.append(
            {
                "priority": "P0",
                "action": "segregate_runtime_noise",
                "detail": "Maintain runtime/build paths outside active reasoning loops via separate manifests and excludes.",
                "metric": {"runtime_plus_build_share": round(runtime_share, 6)},
            }
        )

    for row in by_low_snr:
        if row.signal >= 75 and row.total_files >= 500:
            action_plan.append(
                {
                    "priority": "P1",
                    "action": "split_signal_from_noise",
                    "detail": f"Domain '{row.domain}' has high signal but low SNR; split source/docs paths from generated/runtime paths.",
                    "metric": {
                        "domain": row.domain,
                        "snr": round(row.snr, 6),
                        "signal": round(row.signal, 3),
                        "files": row.total_files,
                    },
                }
            )
            if len(action_plan) >= 8:
                break

    unknown_ratio = (
        global_counts.get("unknown", 0) / total_files if total_files else 0.0
    )
    if unknown_ratio > 0.002:
        action_plan.append(
            {
                "priority": "P2",
                "action": "expand_unknown_classification",
                "detail": "Expand classifier rules for unknown extensions and edge-paths.",
                "metric": {"unknown_ratio": round(unknown_ratio, 6)},
            }
        )

    action_plan = action_plan[:10]

    domain_table: list[dict[str, Any]] = []
    for row in sorted(domain_scores, key=lambda r: (-r.total_files, r.domain)):
        counts = domain_counts[row.domain]
        domain_table.append(
            {
                "domain": row.domain,
                "total_files": row.total_files,
                "snr": round(row.snr, 6),
                "signal": round(row.signal, 3),
                "noise": round(row.noise, 3),
                **{lens: int(counts.get(lens, 0)) for lens in LENSES},
            }
        )

    graph_nodes = [
        {"id": lens, "type": "lens", "label": lens}
        for lens in LENSES
    ]
    graph_nodes.extend(
        {
            "id": f"domain:{row.domain}",
            "type": "domain",
            "label": row.domain,
            "snr": round(row.snr, 6),
            "total_files": row.total_files,
        }
        for row in domain_scores
    )
    graph_edges = []
    for domain, counts in domain_counts.items():
        for lens in LENSES:
            value = int(counts.get(lens, 0))
            if value == 0:
                continue
            graph_edges.append(
                {
                    "source": f"domain:{domain}",
                    "target": lens,
                    "weight": value,
                }
            )

    return {
        "protocol": "standing-on-the-shoulders-of-the-giants",
        "engine": "workspace-masterpiece-autonomous-atlas",
        "inventory": {
            "files_manifest": str(files_manifest),
            "dirs_manifest": str(dirs_manifest) if dirs_manifest else None,
            "total_files": total_files,
            "total_dirs": total_dirs,
        },
        "global": {
            "snr": round(snr, 6),
            "signal": round(signal, 3),
            "noise": round(noise, 3),
            "unknown_files": int(global_counts.get("unknown", 0)),
            "unknown_ratio": (
                round(global_counts.get("unknown", 0) / total_files, 6)
                if total_files
                else 0.0
            ),
            "lens_counts": {lens: int(global_counts.get(lens, 0)) for lens in LENSES},
        },
        "rankings": {
            "lowest_snr_domains": [
                {
                    "domain": row.domain,
                    "snr": round(row.snr, 6),
                    "total_files": row.total_files,
                    "signal": round(row.signal, 3),
                    "noise": round(row.noise, 3),
                }
                for row in by_low_snr[:top_n]
            ],
            "highest_signal_domains": [
                {
                    "domain": row.domain,
                    "signal": round(row.signal, 3),
                    "snr": round(row.snr, 6),
                    "total_files": row.total_files,
                }
                for row in by_high_signal[:top_n]
            ],
            "unknown_hotspots": unknown_hotspots[:top_n],
            "unknown_extensions": unknown_ext_counts.most_common(top_n),
            "unknown_samples": unknown_samples,
        },
        "action_plan": action_plan,
        "domain_table": domain_table,
        "graph": {
            "nodes": graph_nodes,
            "edges": graph_edges,
        },
    }


def _render_markdown(report: dict[str, Any], top_n: int) -> str:
    inv = report["inventory"]
    glob = report["global"]
    low = report["rankings"]["lowest_snr_domains"]
    signal = report["rankings"]["highest_signal_domains"]
    unknown = report["rankings"]["unknown_hotspots"]
    unknown_ext = report["rankings"]["unknown_extensions"]
    action_plan = report.get("action_plan", [])

    lines = [
        "# Workspace Masterpiece Atlas Report",
        "",
        f"- Engine: `{report['engine']}`",
        f"- Protocol: `{report['protocol']}`",
        f"- Files indexed: `{inv['total_files']}`",
        f"- Dirs indexed: `{inv['total_dirs']}`",
        f"- Global SNR: `{glob['snr']:.6f}`",
        f"- Unknown ratio: `{glob['unknown_ratio']:.6f}`",
        "",
        "## Lens Distribution",
        "",
        "| Lens | Count |",
        "|------|-------|",
    ]
    for lens, count in glob["lens_counts"].items():
        lines.append(f"| `{lens}` | {count} |")

    lines.extend(
        [
            "",
            f"## Lowest SNR Domains (Top {top_n})",
            "",
            "| Domain | SNR | Files | Signal | Noise |",
            "|--------|-----|-------|--------|-------|",
        ]
    )
    for row in low:
        lines.append(
            f"| `{row['domain']}` | {row['snr']:.6f} | {row['total_files']} | {row['signal']:.3f} | {row['noise']:.3f} |"
        )

    lines.extend(
        [
            "",
            f"## Highest Signal Domains (Top {top_n})",
            "",
            "| Domain | Signal | SNR | Files |",
            "|--------|--------|-----|-------|",
        ]
    )
    for row in signal:
        lines.append(
            f"| `{row['domain']}` | {row['signal']:.3f} | {row['snr']:.6f} | {row['total_files']} |"
        )

    lines.extend(
        [
            "",
            f"## Unknown Hotspots (Top {top_n})",
            "",
            "| Domain | Unknown Files | Total Files |",
            "|--------|---------------|-------------|",
        ]
    )
    if unknown:
        for row in unknown:
            lines.append(
                f"| `{row['domain']}` | {row['unknown_files']} | {row['total_files']} |"
            )
    else:
        lines.append("| `none` | 0 | 0 |")

    lines.extend(
        [
            "",
            f"## Unknown Extensions (Top {top_n})",
            "",
            "| Extension | Count |",
            "|-----------|-------|",
        ]
    )
    if unknown_ext:
        for ext, count in unknown_ext:
            lines.append(f"| `{ext}` | {count} |")
    else:
        lines.append("| `none` | 0 |")

    lines.extend(
        [
            "",
            "## Action Plan",
            "",
            "| Priority | Action | Detail |",
            "|----------|--------|--------|",
        ]
    )
    if action_plan:
        for item in action_plan:
            lines.append(
                f"| `{item['priority']}` | `{item['action']}` | {item['detail']} |"
            )
    else:
        lines.append("| `none` | `none` | No action required. |")

    lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run interdisciplinary multi-lens workspace atlas analysis."
    )
    parser.add_argument(
        "--files-manifest",
        type=Path,
        default=Path("artifacts/inventory/files_all.txt"),
        help="Path to full file inventory list.",
    )
    parser.add_argument(
        "--dirs-manifest",
        type=Path,
        default=Path("artifacts/inventory/dirs_all.txt"),
        help="Path to full directory inventory list.",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("artifacts/atlas/workspace_masterpiece_report.json"),
        help="Output JSON report path.",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=Path("artifacts/atlas/workspace_masterpiece_report.md"),
        help="Output markdown report path.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Number of ranked domains to include.",
    )
    parser.add_argument(
        "--max-unknown-ratio",
        type=float,
        default=0.35,
        help="Fail threshold when --strict is enabled.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if unknown ratio exceeds --max-unknown-ratio.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = analyze_inventory(
        files_manifest=args.files_manifest,
        dirs_manifest=args.dirs_manifest,
        top_n=args.top_n,
    )

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_md.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.out_md.write_text(_render_markdown(report, args.top_n), encoding="utf-8")

    unknown_ratio = float(report["global"]["unknown_ratio"])
    print(
        json.dumps(
            {
                "files": report["inventory"]["total_files"],
                "dirs": report["inventory"]["total_dirs"],
                "snr": report["global"]["snr"],
                "unknown_ratio": unknown_ratio,
                "json_report": str(args.out_json),
                "md_report": str(args.out_md),
            }
        )
    )

    if args.strict and unknown_ratio > args.max_unknown_ratio:
        print(
            f"unknown ratio {unknown_ratio:.6f} exceeds threshold {args.max_unknown_ratio:.6f}",
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
