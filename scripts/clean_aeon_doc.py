#!/usr/bin/env python3
"""
Clean the AEON research document.

Removes:
- Sidebar URLs from DeepSeek interface (lines 12-445)
- Chat artifacts (MeMeLee timestamps)
- Final UI artifacts (DeepThink, Search footer)
- Escaped characters in headings

Adds:
- Proper YAML frontmatter with disclaimer
- Trailing newline
"""
import re
import sys
from pathlib import Path

FRONTMATTER = '''---
title: "BIZRA AEON-HIVEMIND FATE Integration Analysis"
status: archived-research
source: DeepSeek conversation (Dec 2024)
disclaimer: |
  This document is an archived research transcript from an AI-assisted
  conversation. Claims and proofs are ASPIRATIONAL and require empirical
  validation before implementation. See "Empirical Validation Plan" section.
proof_status: aspirational
created: 2025-07-10
archived: 2025-12-21
---

# Document Status & Disclaimer

> **⚠️ ASPIRATIONAL ARCHITECTURE**: This document describes a conceptual
> framework for trustworthy AI systems. All theorems, proofs, and formal
> claims herein are *design goals* pending rigorous mathematical verification
> and empirical testing. The content represents research exploration, not
> production-ready specifications.
>
> **Proof Status Legend:**
> - **[PROVEN]**: Formally verified with machine-checked proofs
> - **[VALIDATED]**: Empirically tested with reproducible experiments
> - **[ASPIRATIONAL]**: Design goal requiring future verification

'''


def clean_aeon_document(input_path: Path) -> str:
    """Clean the AEON research document."""
    content = input_path.read_text(encoding="utf-8", errors="replace")
    lines = content.split("\n")
    
    # Find where the real content starts (after sidebar URLs)
    start_idx = 0
    for i, line in enumerate(lines):
        if line.strip().startswith("Mahmoud Hassan"):
            start_idx = i
            break
    
    # Extract content after sidebar
    real_lines = lines[start_idx:]
    content = "\n".join(real_lines)
    
    # Remove Telegram/chat artifacts: "MeMeLee, \[07/10/2025 11:04 PM\]"
    content = re.sub(r'MeMeLee,\s*\\\[[\d/]+\s+[\d:]+\s*[AP]M\\\]\s*', '', content)
    
    # Remove final DeepSeek UI artifacts
    content = re.sub(r'\*\*The Cognitive Symphony begins now\.\*\*.*$', '', content, flags=re.DOTALL)
    content = re.sub(r'DeepThink\s*\n\s*Search\s*\n.*AI-generated.*$', '', content, flags=re.DOTALL)
    
    # Clean escaped characters
    content = content.replace("\\.", ".")
    content = content.replace("\\*", "*")
    content = content.replace("\\[", "[")
    content = content.replace("\\]", "]")
    
    # Replace tabs with 2 spaces
    content = content.replace("\t", "  ")
    
    # Remove the "Mahmoud Hassan" name line and empty lines before title
    content = re.sub(r'^Mahmoud Hassan\s*\n+', '', content)
    
    # Remove informal user prompts (standalone lines)
    # Patterns: "can you proof/prove...", "please create/generate...", "eval system..."
    informal_prompts = [
        r'^can you (?:proof|prove|explain|show|help|check|verify)\b.*$',
        r'^please (?:create|generate|provide|make|write|build|add)\b.*$',
        r'^eval(?:uate)?\s+(?:system|performance|output)\b.*$',
        r'^can craft\b.*$',
        r'^map concepts\b.*$',
        r'^lets?\s+(?:emulate|simulate|test|run|build)\b.*$',
    ]
    for pattern in informal_prompts:
        content = re.sub(pattern, '', content, flags=re.MULTILINE | re.IGNORECASE)
    
    # Annotate bare code fences with language specifier
    # Simply replace bare ``` with ```text (idempotent - won't match already-annotated)
    # Match ``` at start of line followed immediately by newline (bare fence only)
    content = re.sub(
        r'^```\n',
        '```text\n',
        content,
        flags=re.MULTILINE
    )
    
    # Add proof status markers to key theorems/claims (idempotent)
    # Use negative lookbehind to avoid double-marking
    content = re.sub(
        r'(?<!\*\*\[ASPIRATIONAL\]\*\* )(Theorem \d+|Lemma \d+|Proof:|Claim:)',
        r'**[ASPIRATIONAL]** \1',
        content
    )
    
    # Combine frontmatter + cleaned content
    result = FRONTMATTER + content.strip() + "\n"
    
    return result


def main():
    """Main entry point."""
    input_file = Path(__file__).parent.parent / "docs/research/aeon-hivemind-fate-architecture.md"
    
    if not input_file.exists():
        print(f"ERROR: File not found: {input_file}")
        sys.exit(1)
    
    print(f"Processing: {input_file}")
    cleaned = clean_aeon_document(input_file)
    
    # Write back
    input_file.write_text(cleaned, encoding="utf-8")
    print(f"Wrote {len(cleaned)} bytes to {input_file.name}")
    
    # Stats
    lines = cleaned.split("\n")
    print(f"Lines: {len(lines)}")


if __name__ == "__main__":
    main()
