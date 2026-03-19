#!/usr/bin/env python3
"""
CI Docs-Truth Gate — Prevents documentation drift against code and blueprint truth.

Checks:
1. README constitutional thresholds match authoritative constants.
2. README Rust crate count matches bizra-omega/Cargo.toml workspace members.
3. Each unified blueprint module's declared TOTAL matches its status markers.
4. The unified blueprint master index completion summary matches module reality.

Standing on Giants: Deming (PDCA, 1950) — verify the document, not just the code.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EXPECTED_BLUEPRINT_MODULE_COUNT = 12
STATUS_LINE_PATTERN = re.compile(
    r"^\*\*Status:\*\*\s*\[(?P<marker>[x~ ])\]\s*(?P<label>BUILT|PARTIAL|NOT BUILT)\s*$",
    re.MULTILINE,
)
MODULE_TOTAL_PATTERN = re.compile(
    r"^\|\s*\*\*TOTAL\*\*\s*\|\s*\*\*"
    r"(?P<built>\d+)/(?P<total>\d+)"
    r"(?:\s*\+\s*(?P<partial>\d+)P\s*\+\s*(?P<not_built>\d+)N)?"
    r"\*\*\s*\|",
    re.MULTILINE,
)


@dataclass(frozen=True)
class StatusCounts:
    built: int
    total: int
    partial: int = 0
    not_built: int = 0

    @property
    def accounted(self) -> int:
        return self.built + self.partial + self.not_built

    def format(self) -> str:
        return (
            f"{self.built}/{self.total} built, "
            f"{self.partial} partial, {self.not_built} not built"
        )


@dataclass(frozen=True)
class MasterSummaryRow:
    domain: str
    counts: StatusCounts


def _strip_markdown(text: str) -> str:
    return text.replace("**", "").replace("`", "").strip()


def _extract_constant(name: str, root: Path = ROOT) -> float | None:
    """Extract a Final[float] constant from core/integration/constants.py."""
    constants_path = root / "core" / "integration" / "constants.py"
    pattern = re.compile(
        rf"^{re.escape(name)}\s*:\s*Final\[float\]\s*=\s*([\d.]+)", re.MULTILINE
    )
    match = pattern.search(constants_path.read_text(encoding="utf-8"))
    return float(match.group(1)) if match else None


def _count_cargo_members(root: Path = ROOT) -> int:
    """Count workspace members in bizra-omega/Cargo.toml."""
    cargo_path = root / "bizra-omega" / "Cargo.toml"
    text = cargo_path.read_text(encoding="utf-8")
    block = re.search(r"members\s*=\s*\[(.*?)\]", text, re.S)
    if not block:
        return 0
    lines = block.group(1).splitlines()
    return sum(1 for line in lines if line.strip() and not line.strip().startswith("#"))


def _check_readme_thresholds(root: Path = ROOT) -> list[str]:
    """Verify README.md constitutional thresholds match authoritative sources."""
    issues: list[str] = []
    readme = (root / "README.md").read_text(encoding="utf-8")

    # --- ADL Gini ---
    adl_const = _extract_constant("ADL_GINI_THRESHOLD", root)
    adl_readme_match = re.search(r"ADL.*Gini\s*\|\s*<=?\s*([\d.]+)", readme)
    if adl_const is not None and adl_readme_match:
        readme_val = float(adl_readme_match.group(1))
        if abs(readme_val - adl_const) > 1e-6:
            issues.append(
                f"README ADL Gini says {readme_val} but constants.py says {adl_const}"
            )

    # --- Ihsan ---
    ihsan_const = _extract_constant("UNIFIED_IHSAN_THRESHOLD", root)
    ihsan_readme_match = re.search(r"Ihsan.*\|\s*>=?\s*([\d.]+)", readme)
    if ihsan_const is not None and ihsan_readme_match:
        readme_val = float(ihsan_readme_match.group(1))
        if abs(readme_val - ihsan_const) > 1e-6:
            issues.append(
                f"README Ihsan says {readme_val} but constants.py says {ihsan_const}"
            )

    # --- SNR ---
    snr_const = _extract_constant("UNIFIED_SNR_THRESHOLD", root)
    snr_readme_match = re.search(r"SNR.*\|\s*>=?\s*([\d.]+)", readme)
    if snr_const is not None and snr_readme_match:
        readme_val = float(snr_readme_match.group(1))
        if abs(readme_val - snr_const) > 1e-6:
            issues.append(
                f"README SNR says {readme_val} but constants.py says {snr_const}"
            )

    # --- Rust crate count ---
    cargo_count = _count_cargo_members(root)
    crate_match = re.search(r"High-performance core \((\d+) Rust crates?\)", readme)
    if crate_match:
        readme_count = int(crate_match.group(1))
        if readme_count != cargo_count:
            issues.append(
                f"README says {readme_count} Rust crates but Cargo.toml has {cargo_count}"
            )

    return issues


def _blueprint_module_paths(root: Path = ROOT) -> list[Path]:
    return sorted(
        path
        for path in (root / "docs" / "UNIFIED_BLUEPRINT").glob("[0-9][0-9]_*.md")
        if path.name != "00_MASTER_INDEX.md"
    )


def _count_status_markers(path: Path) -> StatusCounts:
    text = path.read_text(encoding="utf-8")
    built = partial = not_built = 0

    for match in STATUS_LINE_PATTERN.finditer(text):
        label = match.group("label")
        if label == "BUILT":
            built += 1
        elif label == "PARTIAL":
            partial += 1
        elif label == "NOT BUILT":
            not_built += 1

    total = built + partial + not_built
    return StatusCounts(
        built=built,
        partial=partial,
        not_built=not_built,
        total=total,
    )


def _parse_module_declared_totals(path: Path) -> StatusCounts | None:
    text = path.read_text(encoding="utf-8")
    match = MODULE_TOTAL_PATTERN.search(text)
    if not match:
        return None

    return StatusCounts(
        built=int(match.group("built")),
        total=int(match.group("total")),
        partial=int(match.group("partial") or 0),
        not_built=int(match.group("not_built") or 0),
    )


def _extract_section(text: str, heading: str) -> str:
    heading_pattern = re.compile(rf"^##\s+{re.escape(heading)}\s*$", re.MULTILINE)
    match = heading_pattern.search(text)
    if not match:
        return ""

    start = match.end()
    next_heading = re.search(r"^##\s+", text[start:], re.MULTILINE)
    if next_heading:
        return text[start : start + next_heading.start()]
    return text[start:]


def _parse_master_completion_summary(
    root: Path = ROOT,
) -> tuple[list[MasterSummaryRow], StatusCounts | None]:
    text = (root / "docs" / "UNIFIED_BLUEPRINT" / "00_MASTER_INDEX.md").read_text(
        encoding="utf-8"
    )
    section = _extract_section(text, "Completion Summary")
    if not section:
        return [], None

    rows: list[MasterSummaryRow] = []
    total_row: StatusCounts | None = None

    for raw_line in section.splitlines():
        line = raw_line.strip()
        if not line.startswith("|"):
            continue

        cells = [_strip_markdown(cell) for cell in line.strip("|").split("|")]
        if len(cells) != 5:
            continue

        domain, built_total, partial, not_built, _coverage = cells
        if domain == "Domain" or set(domain) == {"-"}:
            continue
        if not re.fullmatch(r"\d+/\d+", built_total):
            continue

        built_str, total_str = built_total.split("/", maxsplit=1)
        counts = StatusCounts(
            built=int(built_str),
            total=int(total_str),
            partial=int(partial),
            not_built=int(not_built),
        )

        if domain == "TOTAL":
            total_row = counts
        else:
            rows.append(MasterSummaryRow(domain=domain, counts=counts))

    return rows, total_row


def _check_blueprint_truth(
    root: Path = ROOT,
    *,
    expected_module_count: int | None = EXPECTED_BLUEPRINT_MODULE_COUNT,
) -> list[str]:
    issues: list[str] = []
    module_paths = _blueprint_module_paths(root)

    if expected_module_count is not None and len(module_paths) != expected_module_count:
        issues.append(
            "Unified blueprint module count mismatch: "
            f"expected {expected_module_count}, found {len(module_paths)}"
        )

    aggregate = StatusCounts(built=0, total=0, partial=0, not_built=0)
    module_actuals: list[tuple[Path, StatusCounts]] = []

    for path in module_paths:
        actual = _count_status_markers(path)
        declared = _parse_module_declared_totals(path)
        module_actuals.append((path, actual))

        if declared is None:
            issues.append(f"{path.relative_to(root)} is missing a TOTAL row")
            continue

        if declared.accounted != declared.total:
            issues.append(
                f"{path.relative_to(root)} declares inconsistent totals: "
                f"{declared.format()}"
            )

        if actual != declared:
            issues.append(
                f"{path.relative_to(root)} total drift: markers show "
                f"{actual.format()} but TOTAL row says {declared.format()}"
            )

        aggregate = StatusCounts(
            built=aggregate.built + actual.built,
            total=aggregate.total + actual.total,
            partial=aggregate.partial + actual.partial,
            not_built=aggregate.not_built + actual.not_built,
        )

    master_rows, master_total = _parse_master_completion_summary(root)
    if not master_rows:
        issues.append(
            "docs/UNIFIED_BLUEPRINT/00_MASTER_INDEX.md is missing a parsable "
            "Completion Summary table"
        )
        return issues

    if len(master_rows) != len(module_actuals):
        issues.append(
            "Master index domain row count mismatch: "
            f"{len(master_rows)} summary rows for {len(module_actuals)} module files"
        )

    for (path, actual), row in zip(module_actuals, master_rows, strict=False):
        if row.counts != actual:
            issues.append(
                "Master index row drift for "
                f"'{row.domain}' vs {path.name}: summary says {row.counts.format()} "
                f"but module markers show {actual.format()}"
            )

    if master_total is None:
        issues.append(
            "docs/UNIFIED_BLUEPRINT/00_MASTER_INDEX.md is missing a TOTAL summary row"
        )
    elif master_total != aggregate:
        issues.append(
            "Master index TOTAL drift: summary says "
            f"{master_total.format()} but module markers sum to {aggregate.format()}"
        )

    return issues


TRUTH_LABEL_PATTERN = re.compile(
    r"\[(?:ENFORCEMENT|OPTIMIZATION):\s*(?:PROVEN|WIRED|PARTIAL|PLANNED)\]"
)

# Minimum truth labels required in STATUS.md (ratchet-only)
MIN_TRUTH_LABELS_STATUS = 8


def _check_truth_labels(root: Path) -> list[str]:
    """Check that STATUS.md contains truth labels (honest labeling gate).

    Standing on Giants: Al-Ghazali (honest labeling, 1096) — no doc claims
    'proven' for what is 'wired but partial'.
    """
    issues: list[str] = []
    status_path = root / "STATUS.md"
    if not status_path.exists():
        issues.append("STATUS.md not found")
        return issues

    content = status_path.read_text(encoding="utf-8")
    labels = TRUTH_LABEL_PATTERN.findall(content)

    if len(labels) < MIN_TRUTH_LABELS_STATUS:
        issues.append(
            f"STATUS.md has {len(labels)} truth labels, "
            f"minimum required is {MIN_TRUTH_LABELS_STATUS}. "
            "Each subsystem row must carry an [ENFORCEMENT: X] or [OPTIMIZATION: X] label."
        )

    # Check vocabulary section exists
    if "Truth-Label Vocabulary" not in content:
        issues.append(
            "STATUS.md is missing the 'Truth-Label Vocabulary' section. "
            "This section defines the meaning of each label."
        )

    return issues


def main() -> int:
    issues = [
        *_check_readme_thresholds(ROOT),
        *_check_blueprint_truth(ROOT),
        *_check_truth_labels(ROOT),
    ]
    if issues:
        print("[DOCS-TRUTH-GATE] FAILED")
        for issue in issues:
            print(f"  - {issue}")
        return 1

    print("[DOCS-TRUTH-GATE] PASS")
    print("README thresholds, unified blueprint modules, and master rollups agree.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
