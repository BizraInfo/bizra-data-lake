#!/usr/bin/env python3
"""
Extract the simulation section from agent-learning-lifecycle.md to a separate appendix.

This script:
1. Locates the simulation section using markers (marker-first behavior):
   - Primary marker: SIMULATION_START_MARKER
   - Alternate marker: ALT_SIMULATION_MARKER
   - Structural fallback: H2/H3 heading scan if markers are missing
2. Extracts it to a new file: agent-learning-lifecycle-simulations.md
3. Replaces the extracted content with a navigational summary and link
"""
# NOTE: This script mutates the input file; do not re-run blindly.
import argparse
import difflib
import re
import sys
from pathlib import Path

# Section marker that starts the simulation content
SIMULATION_START_MARKER = r'^\s*#{0,6}\s*Simulating End-to-End Agent Lifecycle'
# Alternative marker (contextual heading)
ALT_SIMULATION_MARKER = r'^\s*#{0,6}\s*Complete Agent Expert System\b'

# Frontmatter for the extracted simulations file
SIMULATIONS_FRONTMATTER = '''---
title: "Agent Learning Lifecycle - Simulations Appendix"
status: archived-research
source: Kimi conversation (Dec 2024)
parent: agent-learning-lifecycle.md
disclaimer: |
  This appendix contains extended simulation exercises from the
  agent learning research. These are conceptual explorations.
proof_status: exploratory
created: 2025-12-21
---

# Agent Learning Lifecycle - Simulation Exercises

> **Note**: This appendix was extracted from the main research document
> for readability. The simulations below explore various aspects of
> agent expert systems through detailed walkthroughs.

---

'''

# Navigational summary to replace the extracted content
REPLACEMENT_SUMMARY = '''
---

## Extended Simulations (Appendix)

> The following simulation exercises have been extracted to a separate document
> for improved readability and document organization.

See: [Agent Learning Lifecycle - Simulations Appendix](./agent-learning-lifecycle-simulations.md)

**Appendix Contents:**
- Complete Agent Expert System End-to-End Lifecycle Emulation
- Performance Evaluation Metrics and Benchmarks
- Comprehensive System Analysis & Validation
- Professional Elite Practitioner Implementation Plans
- Cognitive Symphony Architecture Roadmaps

---

'''
REPLACEMENT_SENTINEL = "## Extended Simulations (Appendix)"


def find_simulation_start(content: str) -> int:
    """Find the line number where simulation content starts."""
    lines = content.split('\n')
    
    # Look for the simulation start marker
    for i, line in enumerate(lines):
        if re.match(SIMULATION_START_MARKER, line, re.IGNORECASE):
            return i
        if re.match(ALT_SIMULATION_MARKER, line, re.IGNORECASE):
            return i

    # Structural fallback: use the last H2/H3 heading with content following it
    heading_re = re.compile(r'^\s*#{2,3}\s+\S')
    separator_re = re.compile(r'^\s*---\s*$')
    for i in range(len(lines) - 1, -1, -1):
        if heading_re.match(lines[i]):
            for j in range(i + 1, min(i + 6, len(lines))):
                if separator_re.match(lines[j]) or lines[j].strip():
                    print(f"Fallback: using heading at line {i + 1}")
                    return i

    return -1


def _print_diff(label: str, before: str, after: str) -> None:
    """Print a unified diff to stdout."""
    diff = difflib.unified_diff(
        before.splitlines(),
        after.splitlines(),
        fromfile=f"{label} (before)",
        tofile=f"{label} (after)",
        lineterm=""
    )
    for line in diff:
        print(line)


def extract_simulations(input_path: Path, output_path: Path, dry_run: bool) -> bool:
    """Extract simulation section to separate file."""
    content = input_path.read_text(encoding="utf-8", errors="replace")
    if REPLACEMENT_SENTINEL in content or REPLACEMENT_SUMMARY.strip() in content:
        print("ERROR: Replacement summary already present. Aborting to avoid duplication.")
        return False

    lines = content.split('\n')
    
    # Find where simulations start
    sim_start = find_simulation_start(content)
    if sim_start < 0:
        print("ERROR: Could not find simulation section marker")
        return False
    
    print(f"Found simulation section starting at line {sim_start + 1}")
    
    # Always split at sim_start for deterministic behavior.
    main_content_lines = lines[:sim_start]
    simulation_lines = lines[sim_start:]
    summary_heading_re = re.compile(r'^\s*#{1,6}\s*Summary\b', re.IGNORECASE)
    if any(summary_heading_re.match(line) for line in lines[:sim_start]):
        print("Note: Summary heading found before simulation start; split still begins at marker.")
    
    # Clean up simulation content
    simulation_content = '\n'.join(simulation_lines).strip()
    
    # Skip any meta-commentary at the start of simulations
    meta_patterns = [
        r'^This is a powerful paradigm shift.*?explore further\?.*?\n+',
        r'^The user wants me to.*?\n+',
        r'^Let me (?:analyze|break down).*?\n+',
    ]
    for pattern in meta_patterns:
        simulation_content = re.sub(pattern, '', simulation_content, flags=re.IGNORECASE | re.DOTALL)
    
    # Write the simulations appendix
    appendix_content = SIMULATIONS_FRONTMATTER + simulation_content.strip() + '\n'
    
    # Update the main document with the navigational summary
    main_content = '\n'.join(main_content_lines).strip()
    main_content = main_content + REPLACEMENT_SUMMARY

    if dry_run:
        print("Dry run: showing diffs only.")
        _print_diff(input_path.name, content, main_content)
        existing_appendix = ""
        if output_path.exists():
            existing_appendix = output_path.read_text(encoding="utf-8", errors="replace")
        _print_diff(output_path.name, existing_appendix, appendix_content)
        return True

    output_path.write_text(appendix_content, encoding="utf-8")
    print(f"Wrote {len(appendix_content)} bytes to {output_path.name}")
    
    input_path.write_text(main_content, encoding="utf-8")
    print(f"Updated {input_path.name} with navigational summary")
    
    # Stats
    print(f"Main document: {len(main_content_lines)} lines")
    print(f"Appendix: {len(simulation_lines)} lines")
    
    return True


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Extract simulation section to a separate appendix."
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print diffs to stdout instead of writing files."
    )
    args = parser.parse_args()

    research_dir = Path(__file__).parent.parent / "docs/research"
    input_file = research_dir / "agent-learning-lifecycle.md"
    output_file = research_dir / "agent-learning-lifecycle-simulations.md"
    
    if not input_file.exists():
        print(f"ERROR: File not found: {input_file}")
        sys.exit(1)
    
    print(f"Processing: {input_file}")
    print(f"Extracting to: {output_file}")
    
    success = extract_simulations(input_file, output_file, args.dry_run)
    
    if not success:
        sys.exit(1)
    
    print("\n✅ Extraction complete!")


if __name__ == "__main__":
    main()
