---
name: smart-file-management
description: >
  Smart File Management (AI Cowork) — Batch renaming, automatic organization, smart
  content-aware classification, file merging, and intelligent folder structuring.
  Use when the user asks to organize files, rename files in bulk, classify or sort files,
  merge documents, clean up a directory, tidy a folder, auto-organize a workspace,
  consolidate files, batch rename, combine PDFs, merge CSVs, merge logs, package archives,
  restructure a project, or declutter a file system. Triggers on keywords: organize, classify,
  rename, merge, tidy, sort, clean up, consolidate, batch, restructure, declutter, auto-organize.
license: MIT
metadata:
  author: m.beshr
  version: '1.0'
  category: productivity
---

# Smart File Management (AI Cowork)

Batch renaming, automatic organization, smart classification, file merging — all driven by content-aware AI analysis.

## When to Use This Skill

Use this skill when the user asks you to:

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

---

## Instructions

### Phase 1: Discovery & Analysis

Before taking any action, always analyze the target directory first.

**Step 1 — Scan the target directory**

```bash
# Get full directory tree (files, sizes, types)
find <TARGET_DIR> -type f -printf '%s %p\n' | sort -rn > /home/user/workspace/_file_inventory.txt

# Count files by extension
find <TARGET_DIR> -type f | sed 's/.*\.//' | sort | uniq -c | sort -rn

# Detect potential duplicates by size + partial hash
find <TARGET_DIR> -type f -exec md5sum {} + | sort | uniq -w32 -dD
```

**Step 2 — Content-aware classification scan**

For each file, determine its category using this classification hierarchy:

| Priority | Signal | Method |
|----------|--------|--------|
| 1 | File extension | Direct mapping (see Extension Map below) |
| 2 | Shebang / magic bytes | `file <path>` command or read first 64 bytes |
| 3 | Content keywords | Read first 500 lines, detect patterns (imports, headers, schemas) |
| 4 | Filename patterns | Regex match against common conventions |
| 5 | Parent directory context | Infer from neighboring files |

**Extension Map (Code & Docs focus):**

| Category | Extensions |
|----------|-----------|
| `source-code/rust` | `.rs`, `.toml` (with `[package]`) |
| `source-code/go` | `.go`, `go.mod`, `go.sum` |
| `source-code/javascript` | `.js`, `.jsx`, `.ts`, `.tsx`, `.mjs`, `.cjs` |
| `source-code/python` | `.py`, `.pyi`, `.pyx` |
| `config` | `.toml`, `.yaml`, `.yml`, `.json`, `.ini`, `.env`, `.cfg` |
| `docs/markdown` | `.md`, `.mdx` |
| `docs/text` | `.txt`, `.rst`, `.log` |
| `docs/formatted` | `.pdf`, `.docx`, `.pptx`, `.xlsx`, `.odt` |
| `data/tabular` | `.csv`, `.tsv`, `.xlsx`, `.parquet` |
| `data/structured` | `.json`, `.jsonl`, `.xml`, `.ndjson` |
| `data/database` | `.sql`, `.db`, `.sqlite` |
| `assets/images` | `.png`, `.jpg`, `.jpeg`, `.gif`, `.svg`, `.webp`, `.ico` |
| `assets/media` | `.mp4`, `.mp3`, `.wav`, `.webm`, `.ogg` |
| `build-artifacts` | `.wasm`, `.o`, `.so`, `.dll`, `.exe`, `.bin` |
| `archives` | `.zip`, `.tar`, `.gz`, `.7z`, `.rar`, `.tar.gz` |
| `ai-models` | `.onnx`, `.safetensors`, `.gguf`, `.pt`, `.bin` (with model indicators) |
| `blockchain` | `.sol`, `.move`, files with chain-related imports |

**Disambiguation rules for `.toml`:**
- Contains `[package]` or `[dependencies]` → `source-code/rust` (Cargo.toml)
- Contains `[tool.` → `config` (pyproject.toml)
- Otherwise → `config`

**Disambiguation rules for `.json`:**
- Contains `"dependencies"` + `"name"` → `config` (package.json)
- Contains array of objects with uniform keys → `data/structured`
- Otherwise → `config`

**Step 3 — Present analysis to user**

Before making changes, present a summary:

```
Directory Analysis: <TARGET_DIR>
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Total files: <count>
Total size: <size>

Classification Breakdown:
  source-code/    — <count> files (<size>)
  config/         — <count> files (<size>)
  docs/           — <count> files (<size>)
  data/           — <count> files (<size>)
  assets/         — <count> files (<size>)
  other/          — <count> files (<size>)

Duplicates found: <count> potential duplicates
Unclassified: <count> files

Proposed action: [organize | rename | merge | custom]
```

**Always ask for user confirmation before modifying any files.** Never auto-execute destructive operations (moves, renames, deletes) without explicit approval.

---

### Phase 2: Auto Organize

When the user confirms organization, execute the following:

**Step 1 — Create target folder structure**

```bash
# Standard organized structure
mkdir -p <OUTPUT_DIR>/{source-code/{rust,go,javascript,python},config,docs/{markdown,text,formatted},data/{tabular,structured,database},assets/{images,media},build-artifacts,archives,ai-models,blockchain,_unsorted}
```

**Step 2 — Move files with collision handling**

Use this Python script pattern for safe file moves:

```python
import os
import shutil
from pathlib import Path
from collections import defaultdict

def safe_move(src: str, dest_dir: str, dry_run: bool = False) -> str:
    """Move file to dest_dir with collision handling.
    If a file with the same name exists, append a counter: file(1).ext, file(2).ext, etc.
    Returns the final destination path.
    """
    src_path = Path(src)
    dest_path = Path(dest_dir) / src_path.name

    if dest_path.exists():
        stem = src_path.stem
        suffix = src_path.suffix
        counter = 1
        while dest_path.exists():
            dest_path = Path(dest_dir) / f"{stem}({counter}){suffix}"
            counter += 1

    if not dry_run:
        os.makedirs(dest_dir, exist_ok=True)
        shutil.move(str(src_path), str(dest_path))

    return str(dest_path)
```

**Step 3 — Generate move manifest**

Always generate a manifest file for undo capability:

```
# Move Manifest — <timestamp>
# Format: <original_path> -> <new_path>
/path/to/old/file.rs -> /path/to/organized/source-code/rust/file.rs
/path/to/old/config.yaml -> /path/to/organized/config/config.yaml
...
```

Save manifest to: `<OUTPUT_DIR>/_manifest_<timestamp>.txt`

**Step 4 — Execute with dry-run first**

1. Run the full operation in dry-run mode
2. Show the user what will change
3. On confirmation, execute the real moves
4. Verify all files are accessible post-move
5. Report results

---

### Phase 3: Efficient Batch Rename

Support three naming strategies the user can choose from:

#### Strategy A: Kebab-case + Timestamps
```
Pattern: {name-in-kebab-case}-{YYYY-MM-DD}.{ext}
Example: bizra-agent-config-2026-03-17.toml
```
- Convert CamelCase, snake_case, spaces → kebab-case
- Append date (file modified date or current date)
- Preserve extension

#### Strategy B: Semantic Prefixes
```
Pattern: [{CATEGORY}] {original-name}.{ext}
Example: [AI] model-weights.bin
         [CHAIN] validator-node.rs
         [CONFIG] docker-compose.yaml
         [DOC] architecture-overview.md
```

Prefix map:
| Prefix | Categories |
|--------|-----------|
| `[SRC]` | Source code files |
| `[AI]` | AI models, training data, prompts |
| `[CHAIN]` | Blockchain, smart contracts, validators |
| `[CONFIG]` | Configuration files |
| `[DOC]` | Documentation, markdown, text |
| `[DATA]` | Data files, CSVs, databases |
| `[ASSET]` | Images, media, fonts |
| `[BUILD]` | Build artifacts, binaries |
| `[TEST]` | Test files and fixtures |

#### Strategy C: Flexible Templates
```
Template variables:
  {name}      — Original filename (without extension)
  {ext}       — File extension
  {date}      — File modified date (YYYY-MM-DD)
  {time}      — File modified time (HH-MM-SS)
  {counter}   — Auto-incrementing counter (001, 002, ...)
  {category}  — Auto-detected category
  {project}   — Project name (from nearest Cargo.toml, package.json, etc.)
  {parent}    — Parent directory name
  {hash:N}    — First N chars of file content hash
  {size}      — Human-readable file size

Example template: "{project}-{category}-{name}-{counter}.{ext}"
Result: "bizra-config-database-001.toml"
```

**Batch rename execution:**

```python
import os
import re
from pathlib import Path
from datetime import datetime

def apply_rename_template(file_path: str, template: str, counter: int,
                          category: str, project: str) -> str:
    """Apply a rename template to a file path. Returns the new filename."""
    p = Path(file_path)
    stat = p.stat()
    mod_time = datetime.fromtimestamp(stat.st_mtime)

    variables = {
        'name': p.stem,
        'ext': p.suffix.lstrip('.'),
        'date': mod_time.strftime('%Y-%m-%d'),
        'time': mod_time.strftime('%H-%M-%S'),
        'counter': f'{counter:03d}',
        'category': category,
        'project': project,
        'parent': p.parent.name,
        'size': format_size(stat.st_size),
    }

    result = template
    for key, value in variables.items():
        result = result.replace(f'{{{key}}}', str(value))

    return result

def to_kebab_case(name: str) -> str:
    """Convert any naming convention to kebab-case."""
    # Handle CamelCase
    name = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1-\2', name)
    name = re.sub(r'([a-z\d])([A-Z])', r'\1-\2', name)
    # Replace underscores, spaces, dots (not extension) with hyphens
    name = re.sub(r'[_\s.]+', '-', name)
    # Remove non-alphanumeric except hyphens
    name = re.sub(r'[^a-zA-Z0-9-]', '', name)
    # Collapse multiple hyphens
    name = re.sub(r'-+', '-', name).strip('-')
    return name.lower()
```

**Always generate a rename manifest** for undo capability, same format as the move manifest.

---

### Phase 4: Smart File Merge

#### 4A — Text File Merge
Merge `.txt`, `.md`, `.log` files:

```python
def merge_text_files(file_paths: list, output_path: str,
                     separator: str = "\n\n---\n\n",
                     include_headers: bool = True) -> str:
    """Merge multiple text files into one with optional headers."""
    merged = []
    for fp in sorted(file_paths):
        content = Path(fp).read_text(encoding='utf-8', errors='replace')
        if include_headers:
            header = f"## Source: {Path(fp).name}\n"
            merged.append(header + content)
        else:
            merged.append(content)

    output = separator.join(merged)
    Path(output_path).write_text(output, encoding='utf-8')
    return output_path
```

#### 4B — CSV / Data Merge
Merge tabular data files:

```python
import csv
import json

def merge_csv_files(file_paths: list, output_path: str,
                    deduplicate: bool = False) -> str:
    """Merge multiple CSV files. Handles differing column sets via union."""
    all_rows = []
    all_headers = []

    for fp in file_paths:
        with open(fp, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for header in reader.fieldnames or []:
                if header not in all_headers:
                    all_headers.append(header)
            for row in reader:
                all_rows.append(row)

    if deduplicate:
        seen = set()
        unique_rows = []
        for row in all_rows:
            key = tuple(sorted(row.items()))
            if key not in seen:
                seen.add(key)
                unique_rows.append(row)
        all_rows = unique_rows

    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=all_headers, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(all_rows)

    return output_path

def merge_json_files(file_paths: list, output_path: str,
                     merge_strategy: str = "array") -> str:
    """Merge JSON files.
    Strategies:
      - 'array': Collect all top-level values into a JSON array
      - 'deep': Deep-merge objects (later files override earlier on key conflicts)
    """
    if merge_strategy == "array":
        merged = []
        for fp in file_paths:
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, list):
                    merged.extend(data)
                else:
                    merged.append(data)
    else:  # deep merge
        merged = {}
        for fp in file_paths:
            with open(fp, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if isinstance(data, dict):
                    deep_merge(merged, data)

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(merged, f, indent=2, ensure_ascii=False)

    return output_path
```

#### 4C — PDF Concatenation

```python
# Requires: pip install pypdf
from pypdf import PdfReader, PdfWriter

def merge_pdfs(file_paths: list, output_path: str) -> str:
    """Concatenate multiple PDFs into one."""
    writer = PdfWriter()
    for fp in sorted(file_paths):
        reader = PdfReader(fp)
        for page in reader.pages:
            writer.add_page(page)

    with open(output_path, 'wb') as f:
        writer.write(f)

    return output_path
```

#### 4D — Code Module Consolidation

For merging related source code files into a single module or documentation:

```python
def consolidate_code_module(file_paths: list, output_path: str,
                            language: str = "rust") -> str:
    """Consolidate related code files into a single documented module.
    Adds file-origin comments and preserves imports."""
    sections = []
    imports_seen = set()

    for fp in sorted(file_paths):
        content = Path(fp).read_text(encoding='utf-8')
        lines = content.splitlines()

        # Extract and deduplicate imports
        imports = []
        body = []
        for line in lines:
            if is_import_line(line, language):
                if line.strip() not in imports_seen:
                    imports_seen.add(line.strip())
                    imports.append(line)
            else:
                body.append(line)

        comment_prefix = get_comment_prefix(language)
        header = f"{comment_prefix} === Source: {Path(fp).name} ==="
        sections.append({
            'imports': imports,
            'header': header,
            'body': '\n'.join(body)
        })

    # Write: imports first, then all sections
    output_lines = []
    all_imports = []
    for s in sections:
        all_imports.extend(s['imports'])

    if all_imports:
        output_lines.extend(all_imports)
        output_lines.append('')

    for s in sections:
        output_lines.append(s['header'])
        output_lines.append(s['body'])
        output_lines.append('')

    Path(output_path).write_text('\n'.join(output_lines), encoding='utf-8')
    return output_path

def is_import_line(line: str, language: str) -> bool:
    """Detect import/use statements by language."""
    line = line.strip()
    patterns = {
        'rust': line.startswith('use ') or line.startswith('extern crate'),
        'python': line.startswith('import ') or line.startswith('from '),
        'javascript': line.startswith('import ') or line.startswith('const ') and 'require(' in line,
        'go': line.startswith('import '),
    }
    return patterns.get(language, False)

def get_comment_prefix(language: str) -> str:
    return {'rust': '//', 'python': '#', 'javascript': '//', 'go': '//'}.get(language, '//')
```

#### 4E — Archive Packaging

```bash
# Create organized archive from a directory
zip -r <output>.zip <source_dir> -x "*.DS_Store" "__pycache__/*" "node_modules/*" ".git/*" "target/*"

# Or tar.gz for Linux-native workflows
tar czf <output>.tar.gz --exclude='.git' --exclude='node_modules' --exclude='target' <source_dir>
```

---

### Phase 5: Safety & Undo

**Every operation MUST follow these safety rules:**

1. **Dry-run first**: Always simulate the operation and show the user what will change before executing
2. **Manifest logging**: Every move, rename, or merge generates a timestamped manifest file
3. **No deletes without explicit request**: Never delete files unless the user explicitly says "delete". Organize means move, not delete.
4. **Backup option**: Offer to create a backup archive of the original state before any batch operation
5. **Undo script**: Generate an undo script alongside every manifest

```python
def generate_undo_script(manifest_path: str, undo_script_path: str):
    """Generate a bash script that reverses all operations in a manifest."""
    lines = Path(manifest_path).read_text().splitlines()
    undo_commands = []

    for line in lines:
        if ' -> ' in line and not line.startswith('#'):
            original, new = line.split(' -> ', 1)
            original = original.strip()
            new = new.strip()
            undo_commands.append(f'mv "{new}" "{original}"')

    script = "#!/bin/bash\n# Undo script — reverses file operations\n"
    script += f"# Generated from: {manifest_path}\n\n"
    script += "set -e\n\n"
    # Reverse order for undo
    script += '\n'.join(reversed(undo_commands))

    Path(undo_script_path).write_text(script)
    os.chmod(undo_script_path, 0o755)
```

---

## Decision Flowchart

Use this to determine what the user needs:

```
User request
│
├─ "organize" / "tidy" / "clean up" / "sort" / "classify"
│   → Phase 1 (Discovery) → Phase 2 (Auto Organize)
│
├─ "rename" / "batch rename" / "fix names"
│   → Phase 1 (Discovery) → Phase 3 (Batch Rename)
│   → Ask user: Strategy A (kebab), B (prefix), or C (template)?
│
├─ "merge" / "combine" / "consolidate" / "join"
│   → Phase 1 (Discovery) → Phase 4 (Merge)
│   → Auto-detect merge type from file extensions
│
├─ "restructure" / "refactor folder" / "reorganize project"
│   → Phase 1 (Discovery) → Phase 2 (Auto Organize) with custom structure
│
└─ "analyze" / "what's in this folder" / "scan"
    → Phase 1 (Discovery) only — present report, suggest actions
```

---

## Examples

### Example 1: Organize a messy downloads folder

**User**: "Organize my ~/Downloads folder"

**Agent workflow**:
1. Scan `~/Downloads` — find 247 files across 15 types
2. Present classification breakdown to user
3. User confirms → execute dry-run
4. Show proposed moves → user approves
5. Execute moves, generate manifest + undo script
6. Report: "Organized 247 files into 8 categories. Manifest saved to `_manifest_2026-03-17.txt`"

### Example 2: Batch rename project files

**User**: "Rename all files in ./docs to kebab-case with dates"

**Agent workflow**:
1. Scan `./docs` — find 34 markdown files with mixed naming (CamelCase, spaces, underscores)
2. Apply Strategy A (kebab-case + timestamps)
3. Show dry-run: `ProjectOverview.md → project-overview-2026-03-17.md`
4. User confirms → execute renames
5. Generate rename manifest for undo

### Example 3: Merge CSV reports

**User**: "Merge all CSVs in ./reports into one file"

**Agent workflow**:
1. Scan `./reports` — find 12 CSV files with varying column sets
2. Detect column union: 18 unique columns
3. Ask: "Deduplicate rows? (3 potential duplicates found)"
4. Merge → output `./reports/merged_report_2026-03-17.csv`
5. Report: "Merged 12 files (4,821 rows) into one CSV with 18 columns"

### Example 4: Consolidate Rust modules

**User**: "Consolidate all the utility files in src/utils/ into one module"

**Agent workflow**:
1. Scan `src/utils/` — find 7 `.rs` files
2. Extract and deduplicate `use` statements
3. Combine bodies with source-origin comments
4. Output `src/utils/consolidated.rs`
5. Report: "Consolidated 7 files (342 lines) with 12 unique imports"

### Example 5: Full project restructure

**User**: "This project is a mess, help me restructure it"

**Agent workflow**:
1. Deep scan — analyze all files, detect project type (Rust + JS monorepo)
2. Propose new structure based on detected content
3. Present side-by-side: current vs proposed
4. User refines → finalize plan
5. Execute with full manifest + undo script
6. Update any import paths or references if requested
