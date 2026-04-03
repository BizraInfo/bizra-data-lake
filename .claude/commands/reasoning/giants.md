---
allowed-tools: Bash(python*:*), Read, Grep, Glob, WebSearch, WebFetch
description: Standing on Shoulders Protocol - Build on existing patterns and prior art
argument-hint: [task-or-problem]
---

# Giants - Standing on Shoulders Protocol

## Overview

The Giants protocol ensures we **never reinvent the wheel**. Before implementing anything, we systematically search for:

1. **Internal Prior Art** - Existing patterns in this codebase
2. **Ecosystem Prior Art** - Patterns in BIZRA ecosystem repos
3. **Industry Prior Art** - Established solutions in the broader ecosystem
4. **Academic Prior Art** - Research papers and theoretical foundations

> "If I have seen further, it is by standing on the shoulders of giants." - Isaac Newton

## Current Codebase Context

- Total Rust Files: !`find src crates -name "*.rs" 2>/dev/null | wc -l || echo "N/A"`
- Total Python Files: !`find core bizra_kernel -name "*.py" 2>/dev/null | wc -l || echo "N/A"`
- Expertise Knowledge: !`wc -l bizra_memory/expertise.yaml 2>/dev/null | awk '{print $1}' || echo "N/A"`
- Constitution Rules: !`wc -l constitution/*.yaml 2>/dev/null | awk '{print $1}' || echo "N/A"`

## Your Task

### Phase 1: Internal Prior Art Search

**Search this codebase for existing patterns**:

```bash
# Search for related functions/structs
echo "=== Searching Rust codebase ==="
rg -i "{keyword}" src/ crates/ --type rust -l 2>/dev/null | head -10

echo "=== Searching Python codebase ==="
rg -i "{keyword}" core/ bizra_kernel/ --type python -l 2>/dev/null | head -10

echo "=== Searching documentation ==="
rg -i "{keyword}" docs/ *.md --type md -l 2>/dev/null | head -10
```

**Check expertise knowledge base**:
```bash
# Search semantic knowledge
grep -i "{keyword}" bizra_memory/expertise.yaml

# Search procedural patterns
grep -A5 "procedure:" bizra_memory/expertise.yaml | grep -i "{keyword}"
```

### Phase 2: Ecosystem Prior Art

**Search BIZRA ecosystem**:

| Repository | Purpose | Check For |
|------------|---------|-----------|
| `bizra-genesis-node` | Blockchain substrate | Consensus patterns |
| `HyperGraphRAG` | Knowledge retrieval | Graph patterns |
| `ace-framework` | Agent orchestration | Multi-agent patterns |
| `BIZRA-DATA-LAKE` | Unified memory | Data patterns |

```bash
# If submodules available
for repo in bizra-genesis-node HyperGraphRAG ace-framework; do
    if [ -d "$repo" ]; then
        echo "=== Searching $repo ==="
        rg -i "{keyword}" "$repo" -l 2>/dev/null | head -5
    fi
done
```

### Phase 3: Industry Prior Art

**Search for established solutions**:

Categories to check:
- [ ] Open source implementations
- [ ] Framework documentation
- [ ] Best practice guides
- [ ] Security advisories (if security-related)

**Common pattern sources**:
- Rust: crates.io, rust-lang/rust, tokio-rs
- Python: PyPI, FastAPI docs, asyncio patterns
- Blockchain: substrate, polkadot
- AI/ML: HuggingFace, LangChain

### Phase 4: Academic Prior Art

**For complex algorithms or novel approaches**:

Search areas:
- ArXiv (cs.AI, cs.DC, cs.CR)
- ACM Digital Library
- IEEE Xplore
- Google Scholar

Key terms to search:
- "[problem domain] + algorithm"
- "[approach] + distributed systems"
- "[technique] + formal verification"

## Giants Protocol Template

### Task: [User's Task]

---

#### Internal Prior Art

**Codebase Search Results**:

| File | Relevance | Pattern Found |
|------|-----------|---------------|
| `src/[file].rs` | High | [description] |
| `core/[file].py` | Medium | [description] |
| ... | ... | ... |

**Expertise Knowledge**:
- Semantic: [relevant facts]
- Procedural: [relevant procedures]

**Reuse Recommendation**:
- [ ] Can reuse: [file/function]
- [ ] Can extend: [file/function]
- [ ] Need to create new (no prior art)

---

#### Ecosystem Prior Art

**BIZRA Ecosystem**:

| Repo | Pattern | Applicability |
|------|---------|---------------|
| ... | ... | ... |

---

#### Industry Prior Art

**Established Solutions**:

| Source | Pattern/Library | License | Applicability |
|--------|-----------------|---------|---------------|
| ... | ... | ... | ... |

**Best Practices Found**:
1. [Practice 1]
2. [Practice 2]

---

#### Academic Prior Art

**Relevant Research**:

| Paper/Source | Key Insight | Applicability |
|--------------|-------------|---------------|
| ... | ... | ... |

---

#### Synthesis

**Prior Art Summary**:
- Internal patterns available: [count]
- Ecosystem patterns: [count]
- Industry solutions: [count]
- Academic foundations: [count]

**Recommended Approach**:
- Primary: [Reuse/Extend/Create]
- Foundation: [What to build on]
- Novelty Required: [What's actually new]

---

## Validation Checks

### Prior Art Diligence

- [ ] Internal codebase searched
- [ ] BIZRA ecosystem checked
- [ ] Industry solutions researched
- [ ] Academic foundations considered
- [ ] Licenses verified (if external code)

### Reuse Quality

- [ ] Existing pattern is production-quality
- [ ] Pattern matches use case (not forced fit)
- [ ] Extension points are clear
- [ ] No security vulnerabilities in prior art

## Anti-Patterns to Avoid

**NIH Syndrome (Not Invented Here)**:
- Rejecting good prior art for ego reasons
- Reimplementing well-tested solutions
- Ignoring ecosystem patterns

**Cargo Culting**:
- Copying without understanding
- Using patterns that don't fit
- Ignoring context differences

**Over-Generalization**:
- Making everything a reusable pattern
- Abstracting prematurely
- Adding complexity for future flexibility

## Evidence Generation

Generate Giants protocol receipt:

```json
{
  "receipt_id": "giants-$(date +%s)",
  "timestamp": "$(date -Iseconds)",
  "task_summary": "[task description]",
  "prior_art_search": {
    "internal": {
      "files_searched": 0,
      "patterns_found": 0,
      "reusable": []
    },
    "ecosystem": {
      "repos_checked": [],
      "patterns_found": 0
    },
    "industry": {
      "sources_checked": [],
      "solutions_found": 0
    },
    "academic": {
      "papers_reviewed": 0,
      "insights_extracted": []
    }
  },
  "recommendation": {
    "approach": "reuse|extend|create",
    "foundation": "",
    "novelty_required": ""
  },
  "integrity_hash": ""
}
```

## Report Format

```
## Giants Protocol Report

**Task**: [task description]
**Timestamp**: [ISO timestamp]

### Prior Art Summary

| Category | Searched | Found | Applicable |
|----------|----------|-------|------------|
| Internal | X files | Y patterns | Z usable |
| Ecosystem | X repos | Y patterns | Z usable |
| Industry | X sources | Y solutions | Z usable |
| Academic | X papers | Y insights | Z usable |

### Key Findings

**Best Prior Art Found**:
- Source: [where]
- Pattern: [what]
- Why suitable: [reasoning]

**Gaps Identified**:
- [What's not covered by prior art]

### Recommendation

**Approach**: [Reuse/Extend/Create]
**Foundation**: [What to build on]
**Novelty Required**: [What's actually new]

### Implementation Path

1. [Step based on prior art]
2. [Step based on prior art]
3. [Novel implementation if needed]

### Receipt
- ID: giants-[timestamp]
- Location: docs/evidence/receipts/
```

---

**Giants Philosophy**: "The best code is code we don't have to write. Search first, implement second. Build on proven foundations."
