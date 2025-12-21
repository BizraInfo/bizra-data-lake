#!/usr/bin/env python3
r"""
Clean the Kimi Agent Learning Lifecycle research document.

Fixes:
- Escaped markdown (\# \*\*text\*\*)
- Chat UI artifacts (Ask Anything..., K2)
- Adds proper YAML frontmatter with disclaimer
- Adds trailing newline
"""
import re
import sys
from pathlib import Path

FRONTMATTER = '''---
title: "Agent Learning Lifecycle - Kimi Agent Expert Research"
status: archived-research
source: Kimi conversation (Dec 2024)
disclaimer: |
  This document is an archived research transcript exploring agent
  expertise architectures. Concepts are EXPLORATORY and require
  adaptation for BIZRA's specific Ihsān-constrained environment.
proof_status: exploratory
created: 2025-12-21
archived: 2025-12-21
---

# Document Status & Disclaimer

> **⚠️ EXPLORATORY RESEARCH**: This document captures a research
> conversation about agent learning and expertise accumulation. The
> architectural patterns described are conceptual explorations that
> inform BIZRA's design but are not directly implemented.
>
> **Key Concepts Applied to BIZRA:**
> - Mental models → expertise YAML schemas
> - Self-improvement loops → SAPE pattern elevation
> - Expertise files → PAT agent specialization

'''


def clean_kimi_document(input_path: Path) -> str:
    """Clean the Kimi research document."""
    content = input_path.read_text(encoding="utf-8", errors="replace")
    
    # Remove original YAML frontmatter - anchored to file start, tolerant of CRLF
    # Uses \A to anchor to absolute start of string
    content = re.sub(r'\A---\s*\r?\n.*?\r?\n---\s*\r?\n', '', content, flags=re.DOTALL)
    
    # Fix escaped markdown characters
    content = content.replace("\\#", "#")
    content = content.replace("\\*", "*")
    content = content.replace("\\[", "[")
    content = content.replace("\\]", "]")
    content = content.replace("\\_", "_")
    
    # Remove chat UI artifacts at the end
    content = re.sub(r'\n*Ask Anything\.\.\.\s*\n*K2\s*$', '', content, flags=re.MULTILINE)
    content = re.sub(r'\n*Ask Anything\.\.\.\s*$', '', content, flags=re.MULTILINE)
    
    # Remove assistant-style meta-commentary (narration to user)
    meta_commentary_patterns = [
        r'^This is a powerful paradigm shift.*?What aspect of this would you like to explore further\?.*?$',
        r'^Let me (?:analyze|break down|explain).*?(?:for you|step by step)\.?\s*$',
        r'^I (?:understand|see) that you.*?\.\s*$',
        r'^The user has provided.*?\.\s*$',
        r'^I should (?:focus on|note that|point out).*?\.\s*$',
        r'^(?:Acknowledging|Understanding) (?:Receipt|the request).*?$',
    ]
    for pattern in meta_commentary_patterns:
        content = re.sub(pattern, '', content, flags=re.MULTILINE | re.IGNORECASE)
    
    # Replace tabs with 2 spaces
    content = content.replace("\t", "  ")
    
    # Clean up excessive blank lines (more than 2 in a row)
    content = re.sub(r'\n{4,}', '\n\n\n', content)
    
    # Combine frontmatter + cleaned content
    result = FRONTMATTER + content.strip() + "\n"
    
    return result


def main():
    """Main entry point."""
    input_file = Path(__file__).parent.parent / "docs/research/agent-learning-lifecycle.md"
    
    if not input_file.exists():
        print(f"ERROR: File not found: {input_file}")
        sys.exit(1)
    
    print(f"Processing: {input_file}")
    cleaned = clean_kimi_document(input_file)
    
    # Write back
    input_file.write_text(cleaned, encoding="utf-8")
    print(f"Wrote {len(cleaned)} bytes to {input_file.name}")
    
    # Stats
    lines = cleaned.split("\n")
    print(f"Lines: {len(lines)}")


if __name__ == "__main__":
    main()
