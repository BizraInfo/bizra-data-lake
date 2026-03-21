#!/usr/bin/env python3
"""
B3: Label AI-Assisted Documents
Adds a standard disclaimer header to documents identified as AI-generated
architectural explorations (not engineering test output).
"""
import os

DISCLAIMER = """---
> **DOCUMENT CLASSIFICATION: Architectural Exploration — AI-Assisted**
>
> This document was produced through AI-assisted collaborative analysis.
> It represents architectural thinking and design exploration, NOT verified
> test output or empirical measurement. Claims within should be validated
> against the canonical codebase (`cargo test`, `pytest`, STATUS.md).
>
> For verified evidence, see: `artifacts/CANONICAL_SPEARPOINT_V1/`
---

"""

# Files flagged by audit as containing AI-generation artifacts
FLAGGED_FILES = [
    r"docs\constitutional\BIZRA-Peak-Synthesis-Omega-Infinity.md",
    r"docs\internal\FINAL_OMNI_BLUEPRINT.md",
    r"docs\internal\PEAK_MASTERPIECE_NEXT.md",
    r"docs\internal\PEAK_MASTERPIECE_README.md",
    r"docs\internal\SAPE_ULTIMATE_ANALYSIS_v2.2.2.md",
    r"docs\internal\SAPE_SOVEREIGN_REVIEW_2026.md",
    r"docs\internal\REPOS_ANALYSIS.md",
    r"docs\specs\phase_28_northstar_flagship.md",
    r"docs\specs\phase_29_primordial_activation.md",
    r"docs\specs\phase_30_ddagi_os_definition.md",
    r"docs\specs\_experimental\phase-49-refinement-consolidation\05_phase49_roadmap.md",
    r"docs\reviews\BIZRA_EXECUTIVE_VERDICT.md",
    r"docs\internal\AUDIT_REPORT.md",
    r"docs\internal\USER.md",
    r"docs\BIZRA_STRATEGY_DECK_2026.md",
    r"docs\knowledge\MCP_SERVERS.md",
]

BASE = r"C:\BIZRA-DATA-LAKE"
labeled = 0
skipped = 0
missing = 0

for rel_path in FLAGGED_FILES:
    full_path = os.path.join(BASE, rel_path)
    if not os.path.exists(full_path):
        print(f"  SKIP (not found): {rel_path}")
        missing += 1
        continue

    with open(full_path, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    if "DOCUMENT CLASSIFICATION: Architectural Exploration" in content:
        print(f"  SKIP (already labeled): {rel_path}")
        skipped += 1
        continue

    with open(full_path, "w", encoding="utf-8") as f:
        f.write(DISCLAIMER + content)

    print(f"  LABELED: {rel_path}")
    labeled += 1

print(f"\nDone: {labeled} labeled, {skipped} already done, {missing} not found")
