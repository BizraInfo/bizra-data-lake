# Smart File Management Skill

Last updated: 2026-02-14

The Smart File Manager (`core/skills/smart_file_manager.py`) is a Desktop Bridge skill that provides batch renaming, auto-organization, classification, and file merging — one hotkey away from file operations on the Windows desktop.

---

## Quick Start

### Via Desktop Bridge (AHK Hotkey)

Press **Ctrl+B, F** to open the file management dialog, then select an operation.

### Via JSON-RPC

```json
{
  "jsonrpc": "2.0",
  "method": "invoke_skill",
  "params": {
    "skill": "smart_files",
    "inputs": {
      "operation": "scan",
      "path": "/mnt/c/Users/BIZRA-OS/Downloads"
    }
  },
  "id": 1
}
```

Send to `127.0.0.1:9742` (Desktop Bridge TCP port).

### Via Python

```python
from core.skills.smart_file_manager import SmartFileHandler

handler = SmartFileHandler(allow_outside_root=True)
result = handler._op_scan({"path": "/path/to/directory"})
```

---

## Operations

### 1. Scan

Classify files in a directory by category using the 114-extension taxonomy from `tools/file_type_indexer.py`.

| Input | Type | Required | Default |
|-------|------|----------|---------|
| `path` | string | Yes | - |
| `recursive` | bool | No | `false` |
| `category_filter` | string | No | - |

**Response:**

```json
{
  "operation": "scan",
  "path": "/mnt/c/Users/BIZRA-OS/Downloads",
  "total_files": 247,
  "total_size": "1.8 GB",
  "categories": {
    "document": {"count": 89, "size": "245.3 MB"},
    "image": {"count": 72, "size": "890.1 MB"},
    "code": {"count": 41, "size": "12.4 MB"}
  },
  "top_files": [
    {"name": "video.mp4", "category": "media", "size": "450.2 MB"}
  ]
}
```

### 2. Organize

Move or copy files into category sub-folders. **Dry-run by default** — returns a plan without executing.

| Input | Type | Required | Default |
|-------|------|----------|---------|
| `path` | string | Yes | - |
| `target_path` | string | No | Same as `path` |
| `dry_run` | bool | No | `true` |
| `copy_mode` | bool | No | `false` |

**Dry-run response:**

```json
{
  "operation": "organize",
  "dry_run": true,
  "planned": 247,
  "executed": 0,
  "category_breakdown": {"document": 89, "image": 72, "code": 41},
  "moves": [
    {"source": "/path/report.pdf", "destination": "/path/document/report.pdf", "category": "document"}
  ]
}
```

To execute: pass `"dry_run": false`.

### 3. Rename

Batch rename files with pattern expansion + optional BLAKE3 hash suffix. **Dry-run by default.**

| Input | Type | Required | Default |
|-------|------|----------|---------|
| `path` | string | Yes | - |
| `pattern` | string | No | `{name}{ext}` |
| `prefix` | string | No | - |
| `suffix` | string | No | - |
| `hash_suffix` | bool | No | `false` |
| `dry_run` | bool | No | `true` |

**Safe tokens** (no `eval()` — safe string substitution only):

| Token | Expands To | Example |
|-------|-----------|---------|
| `{name}` | Original filename stem | `report` |
| `{ext}` | File extension (with dot) | `.pdf` |
| `{n}` | Sequential number | `1`, `2`, `3` |
| `{hash}` | BLAKE3 content hash (16 chars) | `a1b2c3d4e5f6g7h8` |
| `{date}` | File modification date | `2026-02-14` |
| `{cat}` | File category | `document` |

**Example:**

```json
{
  "operation": "rename",
  "path": "/mnt/c/BIZRA-DATA-LAKE/00_INTAKE",
  "pattern": "{date}_{name}_{hash}{ext}",
  "dry_run": true
}
```

### 4. Merge

Concatenate text files into a single output file.

| Input | Type | Required | Default |
|-------|------|----------|---------|
| `paths` | list[string] | Yes | - |
| `output_path` | string | No | Auto-generated in first file's directory |
| `separator` | string | No | `"\n"` |
| `add_headers` | bool | No | `true` |

Headers are added as `# === filename.txt ===` before each file's content.

---

## File Categories

The skill reuses the 114-extension taxonomy from `tools/file_type_indexer.py` (`EXT_TO_CATEGORY`):

| Category | Extensions (sample) |
|----------|-------------------|
| `document` | `.pdf`, `.docx`, `.txt`, `.md`, `.rtf` |
| `spreadsheet` | `.xlsx`, `.csv`, `.tsv`, `.xls` |
| `presentation` | `.pptx`, `.ppt`, `.key` |
| `image` | `.png`, `.jpg`, `.svg`, `.gif`, `.webp` |
| `media` | `.mp4`, `.mp3`, `.wav`, `.avi` |
| `code` | `.py`, `.rs`, `.ts`, `.js`, `.java` |
| `data` | `.json`, `.yaml`, `.xml`, `.toml` |
| `archive` | `.zip`, `.tar`, `.gz`, `.7z` |
| `database` | `.db`, `.sqlite`, `.sql` |
| `config` | `.env`, `.ini`, `.cfg` |
| `notebook` | `.ipynb` |
| `executable` | `.exe`, `.sh`, `.bat` |
| `font` | `.ttf`, `.otf`, `.woff` |

Unknown extensions map to `"other"`.

---

## Security

### Path Traversal Protection

All paths are resolved and validated against `DATA_LAKE_ROOT` (from `bizra_config.py`). Attempts to access paths outside the root are rejected:

```
ValueError: Path '/etc/passwd' is outside DATA_LAKE_ROOT '/mnt/c/BIZRA-DATA-LAKE'. Traversal blocked.
```

### Dry-Run Default

Both `organize` and `rename` operations default to `dry_run=True`. Users must explicitly pass `dry_run: false` to execute file system mutations. This prevents accidental bulk operations.

### No `eval()`

Pattern expansion uses safe `str.replace()` with a fixed set of tokens (`_SAFE_TOKENS`). No dynamic code execution.

### BLAKE3 Hashing

File content hashing uses BLAKE3 via `core.proof_engine.canonical.hex_digest()` (SEC-001 compliant). Falls back to SHA-256 if BLAKE3 is unavailable.

### FATE Gate

All skill invocations pass through the SkillRouter's FATE gate and Ihsan threshold check (`>= 0.95`) before execution.

---

## Registration

The skill auto-registers on the Desktop Bridge SkillRouter at startup:

```python
# In core/bridges/desktop_bridge.py, _get_skill_router()
from core.skills.smart_file_manager import register_smart_files
register_smart_files(self._skill_router)
```

Registration creates a `RegisteredSkill` with:
- **Skill name:** `smart_files`
- **Agent name:** `file-manager`
- **Tags:** `files`, `organize`, `rename`, `merge`, `cowork`
- **Required inputs:** `operation`
- **Ihsan floor:** `UNIFIED_IHSAN_THRESHOLD` (from `core/integration/constants.py`)

---

## Testing

```bash
# Run skill tests
PYTHONPATH=/mnt/c/BIZRA-DATA-LAKE pytest tests/core/skills/test_smart_file_manager.py -v

# Run bridge tests (verify no regressions)
PYTHONPATH=/mnt/c/BIZRA-DATA-LAKE pytest tests/core/bridges/test_desktop_bridge.py -v
```

Test coverage includes:
- Empty directory scan
- Category classification and filtering
- Organize dry-run and execute modes
- Copy mode (copy vs move)
- Rename with patterns and hash suffix
- Merge with headers and custom separators
- Path traversal rejection
- Unknown operation error handling
- Registration on mock router

---

## Failure Modes

| Failure | Behavior | Recovery |
|---------|----------|----------|
| Missing `path` input | Returns `{"error": "Missing required input: path"}` | Provide path |
| Path outside root | Returns `{"error": "Path '...' is outside DATA_LAKE_ROOT..."}` | Use valid path |
| Path does not exist | Returns `{"error": "Path does not exist: ..."}` | Create directory first |
| Unknown operation | Returns `{"error": "Unknown operation: '...'", "available_operations": [...]}` | Use scan/organize/rename/merge |
| File read error (merge) | Returns error with file path and exception | Check file permissions |
| Move/copy error (organize) | Continues processing, collects errors in `errors` list | Review errors, retry |

---

*Source of truth: `core/skills/smart_file_manager.py`, `tests/core/skills/test_smart_file_manager.py`*
