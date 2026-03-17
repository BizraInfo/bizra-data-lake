---
name: smart-file-management
description: >
  Smart File Management (AI Cowork) — Batch renaming, automatic organization, smart
  content-aware classification, file merging, and intelligent folder structuring.
  Triggers on: organize, classify, rename, merge, tidy, sort, clean up, consolidate,
  batch, restructure, declutter, auto-organize.
license: MIT
metadata:
  author: m.beshr
  version: '1.0'
  category: productivity
  bizra_skill_tree: filesystem_skill_tree
  default_capability: true
  mcp_server: bizra-fs
---

# Smart File Management (AI Cowork)

Batch renaming, automatic organization, smart classification, file merging — all driven by content-aware AI analysis.

## BIZRA Skill Tree Mapping

This skill maps to `filesystem_skill_tree()` in `bizra-agent/src/skills/skill_tree.rs`.
Every operation below corresponds to a skill node that agents must master progressively:

| Skill Node | Mastery Required | Maps to Phase |
|---|---|---|
| `fs_classify` | Novice (boot) | Phase 1: Discovery & Analysis |
| `fs_organize` | Competent (3+ successes) | Phase 2: Auto Organize |
| `fs_rename` | Novice (boot) | Phase 3: Batch Rename |
| `fs_merge` | Competent classify + organize | Phase 4: Smart File Merge |
| `fs_dedup` | Competent classify | Phase 1 duplicate detection |
| `fs_snr_score` | Competent classify | Phase 1 SNR scoring for Mint Court |
| `fs_delete` | SAT + HITL required | Phase 5: Safe deletion only |
| `fs_sanitize` | Competent rename | Phase 3 name cleanup |
| `fs_archive` | Competent classify | Phase 4E archive packaging |

## Constitutional Requirements (from Enforceable Spine v1.1)

- Every operation produces a manifest (rollback-capable)
- SAT validates before destructive operations execute
- BLAKE3 hash of every moved/renamed file for integrity
- No silent data loss — every delete requires HITL approval
- Receipt emitted for every batch operation
- Ihsan floor applies: degraded quality must be declared, not hidden

## When to Use This Skill

Use this skill when the user asks to:

- **Organize or tidy** files in a directory or workspace
- **Classify or sort** files by type, content, or project context
- **Batch rename** files using patterns, templates, or conventions
- **Merge or consolidate** text files, CSVs, JSONs, PDFs, logs, or code modules
- **Restructure** a project directory or clean up a messy folder
- **Package** files into organized archives
- **Analyze** a directory to report what's inside and suggest organization

## Core Capabilities

### 1. Auto Organize (Content-Aware Classification)
Intelligently analyze file contents and metadata to auto-classify into a structured folder hierarchy.

### 2. Efficient Batch Rename
Rename files in bulk using flexible templates, naming conventions, and pattern matching.

### 3. Smart File Merge
Merge text, data, documents, code modules, and archives into consolidated outputs.

### 4. Directory Analysis & Reporting
Scan and report on directory contents, duplicates, anomalies, and suggested improvements.


## Phase 1: Discovery & Analysis (fs_classify — Novice from boot)

Before taking any action, always analyze the target directory first.

**Step 1 — Scan the target directory**

```bash
find <TARGET_DIR> -type f -printf '%s %p\n' | sort -rn > /home/user/workspace/_file_inventory.txt
find <TARGET_DIR> -type f | sed 's/.*\.//' | sort | uniq -c | sort -rn
find <TARGET_DIR> -type f -exec md5sum {} + | sort | uniq -w32 -dD
```

**Step 2 — Content-aware classification scan**

For each file, determine category using this priority:

| Priority | Signal | Method |
|----------|--------|--------|
| 1 | File extension | Direct mapping (see Extension Map) |
| 2 | Magic bytes | `file <path>` or read first 64 bytes |
| 3 | Content keywords | Read first 500 lines, detect patterns |
| 4 | Filename patterns | Regex against common conventions |
| 5 | Parent directory | Infer from neighboring files |

**Extension Map:**

| Category | Extensions |
|----------|-----------|
| `source-code/rust` | `.rs`, `.toml` (with `[package]`) |
| `source-code/javascript` | `.js`, `.jsx`, `.ts`, `.tsx`, `.mjs`, `.cjs` |
| `source-code/python` | `.py`, `.pyi`, `.pyx` |
| `config` | `.toml`, `.yaml`, `.yml`, `.json`, `.ini`, `.env`, `.cfg` |
| `docs/markdown` | `.md`, `.mdx` |
| `docs/formatted` | `.pdf`, `.docx`, `.pptx`, `.xlsx` |
| `data/tabular` | `.csv`, `.tsv`, `.parquet` |
| `data/structured` | `.json`, `.jsonl`, `.xml`, `.ndjson` |
| `assets/images` | `.png`, `.jpg`, `.jpeg`, `.gif`, `.svg`, `.webp` |
| `assets/media` | `.mp4`, `.mp3`, `.wav`, `.webm` |
| `build-artifacts` | `.wasm`, `.o`, `.so`, `.dll`, `.exe` |
| `archives` | `.zip`, `.tar`, `.gz`, `.7z`, `.rar` |
| `ai-models` | `.onnx`, `.safetensors`, `.gguf`, `.pt` |


**Disambiguation rules:**
- `.toml` with `[package]` → `source-code/rust`; with `[tool.` → `config`
- `.json` with `"dependencies"` + `"name"` → `config`; array of objects → `data/structured`

**Step 3 — Present analysis before any action**

```
Directory Analysis: <TARGET_DIR>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total files: <count>  |  Total size: <size>

Classification:
  source-code/    — <count> files (<size>)
  config/         — <count> files (<size>)
  docs/           — <count> files (<size>)
  data/           — <count> files (<size>)
  assets/         — <count> files (<size>)
  other/          — <count> files (<size>)

Duplicates: <count>  |  Unclassified: <count>
Proposed action: [organize | rename | merge | custom]
```

**Always ask user confirmation before modifying any files.**

## Phase 2: Auto Organize (fs_organize — requires fs_classify at Competent)

**Step 1 — Create target structure**
```bash
mkdir -p <OUTPUT>/{source-code/{rust,javascript,python},config,docs/{markdown,text,formatted},data/{tabular,structured},assets/{images,media},build-artifacts,archives,ai-models,_unsorted}
```

**Step 2 — Safe move with collision handling**
```python
def safe_move(src, dest_dir, dry_run=False):
    dest_path = Path(dest_dir) / Path(src).name
    if dest_path.exists():
        stem, suffix = Path(src).stem, Path(src).suffix
        counter = 1
        while dest_path.exists():
            dest_path = Path(dest_dir) / f"{stem}({counter}){suffix}"
            counter += 1
    if not dry_run:
        os.makedirs(dest_dir, exist_ok=True)
        shutil.move(str(src), str(dest_path))
    return str(dest_path)
```

**Step 3 — Generate manifest** (constitutional requirement)
```
# Move Manifest — <timestamp>
# Format: <original_path> -> <new_path>
/old/file.rs -> /organized/source-code/rust/file.rs
```

**Step 4 — Dry-run first, then execute on confirmation**


## Phase 3: Batch Rename (fs_rename — Novice from boot)

Three naming strategies:

**Strategy A: Kebab-case + Timestamps**
`{name-in-kebab-case}-{YYYY-MM-DD}.{ext}` → `bizra-agent-config-2026-03-17.toml`

**Strategy B: Semantic Prefixes**
`[SRC] main.rs`, `[AI] model.bin`, `[CONFIG] docker-compose.yaml`, `[DOC] overview.md`

**Strategy C: Flexible Templates**
Variables: `{name}`, `{ext}`, `{date}`, `{counter}`, `{category}`, `{project}`, `{parent}`, `{hash:N}`
Example: `"{project}-{category}-{name}-{counter}.{ext}"` → `"bizra-config-database-001.toml"`

```python
def to_kebab_case(name):
    name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1-\2', name)
    name = re.sub(r'([a-z\d])([A-Z])', r'\1-\2', name)
    name = re.sub(r'[_\s.]+', '-', name)
    name = re.sub(r'[^a-zA-Z0-9-]', '', name)
    return re.sub(r'-+', '-', name).strip('-').lower()
```

Always generate rename manifest for undo.

## Phase 4: Smart File Merge (fs_merge — requires classify + organize at Competent)

**4A — Text merge** (`.txt`, `.md`, `.log`): Concatenate with source headers and separators.
**4B — CSV merge**: Union columns, optional dedup, DictReader/DictWriter.
**4C — JSON merge**: Array collect or deep-merge strategies.
**4D — PDF merge**: pypdf PdfReader/PdfWriter concatenation.
**4E — Code consolidation**: Extract + dedup imports, combine bodies with source-origin comments.
**4F — Archive packaging**: `tar czf` or `zip -r` excluding `.git`, `node_modules`, `target`.


## Phase 5: Safety & Undo (Constitutional — applies to ALL phases)

1. **Dry-run first**: Always simulate and show user what will change
2. **Manifest logging**: Every move/rename/merge generates timestamped manifest
3. **No deletes without explicit request**: Organize = move, never delete
4. **Backup option**: Offer archive of original state before batch ops
5. **Undo script**: Generate reversible bash script alongside every manifest

```python
def generate_undo_script(manifest_path, undo_script_path):
    lines = Path(manifest_path).read_text().splitlines()
    undo_commands = []
    for line in lines:
        if ' -> ' in line and not line.startswith('#'):
            original, new = line.split(' -> ', 1)
            undo_commands.append(f'mv "{new.strip()}" "{original.strip()}"')
    script = "#!/bin/bash\nset -e\n" + '\n'.join(reversed(undo_commands))
    Path(undo_script_path).write_text(script)
    os.chmod(undo_script_path, 0o755)
```

## Decision Flowchart

```
User request
├─ "organize" / "tidy" / "clean up" / "classify"
│   → Phase 1 (Discovery) → Phase 2 (Auto Organize)
├─ "rename" / "batch rename" / "fix names"
│   → Phase 1 → Phase 3 (ask: Strategy A/B/C?)
├─ "merge" / "combine" / "consolidate"
│   → Phase 1 → Phase 4 (auto-detect type from extensions)
├─ "restructure" / "reorganize project"
│   → Phase 1 → Phase 2 with custom structure
└─ "analyze" / "what's in this folder" / "scan"
    → Phase 1 only — present report, suggest actions
```
