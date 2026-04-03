import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

# Try to import fastmcp, guide user if missing
try:
    from fastmcp import FastMCP
except ImportError:
    print("❌ Error: 'fastmcp' is not installed.")
    print("Please run: pip install fastmcp")
    sys.exit(1)

# --- CONFIG ---
# Default to current workspace if not set
DEFAULT_DIR = r"C:\BIZRA-Dual-Agentic-system--main"
ENV_DIRS = os.getenv("ALLOWED_DIRS", "")

if ENV_DIRS:
    ALLOWED_DIRS = [p.strip() for p in ENV_DIRS.split(";") if p.strip()]
else:
    ALLOWED_DIRS = [DEFAULT_DIR]

MAX_READ_BYTES = int(os.getenv("MAX_READ_BYTES", "2000000"))  # 2MB
MAX_GREP_HITS = int(os.getenv("MAX_GREP_HITS", "100"))

ROOTS = [Path(p).expanduser().resolve() for p in ALLOWED_DIRS]

# Validate roots exist
VALID_ROOTS = []
for r in ROOTS:
    if r.exists() and r.is_dir():
        VALID_ROOTS.append(r)
    else:
        print(f"⚠️  Warning: Configured root does not exist: {r}")

if not VALID_ROOTS:
    print("❌ Error: No valid allowed directories found.")
    sys.exit(1)

mcp = FastMCP(
    name="BIZRA Sovereign Filesystem",
    instructions=f"Sovereign filesystem access. Allowed roots: {[str(r) for r in VALID_ROOTS]}",
)

def _is_within_roots(resolved: Path) -> bool:
    for r in VALID_ROOTS:
        try:
            if resolved.is_relative_to(r):
                return True
        except Exception:
            if str(resolved).startswith(str(r)):
                return True
    return False

def _safe_path(user_path: str) -> Path:
    p = Path(user_path).expanduser()
    resolved = p.resolve()
    if not _is_within_roots(resolved):
         # Try joining with first root if absolute path fails logic or if relative
        if not p.is_absolute():
             for r in VALID_ROOTS:
                 candidate = (r / p).resolve()
                 if _is_within_roots(candidate):
                     return candidate
        
        raise PermissionError(f"Path not allowed: {resolved}")
    return resolved

@mcp.tool
async def list_allowed_roots() -> Dict[str, Any]:
    """List the directories this server is allowed to access."""
    return {"roots": [str(r) for r in VALID_ROOTS]}

@mcp.tool
async def list_dir(path: str = ".") -> Dict[str, Any]:
    """List contents of a directory."""
    d = _safe_path(path)
    if not d.is_dir():
        raise FileNotFoundError(f"Not a directory: {d}")

    items = []
    for child in sorted(d.iterdir(), key=lambda x: x.name.lower()):
        try:
            stat = child.stat()
            items.append({
                "name": child.name,
                "path": str(child),
                "type": "dir" if child.is_dir() else "file",
                "size": stat.st_size,
                "mtime": stat.st_mtime,
            })
        except Exception as e:
            items.append({"name": child.name, "path": str(child), "error": str(e)})

    return {"path": str(d), "items": items}

@mcp.tool
async def read_text_file(path: str, head: Optional[int] = None, tail: Optional[int] = None) -> Dict[str, Any]:
    """Read contents of a text file."""
    f = _safe_path(path)
    if not f.is_file():
        raise FileNotFoundError(f"Not a file: {f}")

    if head is not None and tail is not None:
        raise ValueError("Use only one of head or tail")

    size = f.stat().st_size
    if size > MAX_READ_BYTES and head is None and tail is None:
        raise ValueError(f"File too large ({size} bytes). Use head or tail.")

    text = f.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()

    if head is not None:
        lines = lines[: max(0, head)]
    elif tail is not None:
        lines = lines[-max(0, tail):]

    return {"path": str(f), "text": "\n".join(lines)}

@mcp.tool
async def write_file(path: str, content: str, overwrite: bool = False) -> Dict[str, Any]:
    """Write text to a file. Requires overwrite=True to replace existing files."""
    f = _safe_path(path)
    if f.exists() and not overwrite:
        raise FileExistsError("File exists. Set overwrite=true to replace it.")
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(content, encoding="utf-8")
    return {"ok": True, "path": str(f), "bytes": len(content.encode("utf-8"))}

@mcp.tool
async def mkdir(path: str) -> Dict[str, Any]:
    """Create a directory (recursively)."""
    d = _safe_path(path)
    d.mkdir(parents=True, exist_ok=True)
    return {"ok": True, "path": str(d)}

@mcp.tool
async def move(src: str, dst: str, overwrite: bool = False) -> Dict[str, Any]:
    """Move or rename a file/directory."""
    s = _safe_path(src)
    d = _safe_path(dst)
    if d.exists() and not overwrite:
        raise FileExistsError("Destination exists. Set overwrite=true.")
    d.parent.mkdir(parents=True, exist_ok=True)
    if d.exists() and overwrite:
        if d.is_dir():
            shutil.rmtree(d)
        else:
            d.unlink()
    shutil.move(str(s), str(d))
    return {"ok": True, "src": str(s), "dst": str(d)}

@mcp.tool
async def delete(path: str, recursive: bool = False) -> Dict[str, Any]:
    """Delete a file or directory."""
    p = _safe_path(path)
    if p.is_dir():
        if not recursive:
            try:
                p.rmdir()
            except OSError:
                raise OSError("Directory not empty. Set recursive=true to force delete.")
        else:
            shutil.rmtree(p)
    else:
        p.unlink()
    return {"ok": True, "path": str(p)}

@mcp.tool
async def grep(query: str, path: str = ".", regex: bool = False, max_hits: Optional[int] = None) -> Dict[str, Any]:
    """Search for a string or regex pattern in files."""
    base = _safe_path(path)
    if not base.exists():
        raise FileNotFoundError(str(base))
    
    limit = max_hits if max_hits else MAX_GREP_HITS
    hits = []
    pattern = re.compile(query) if regex else None

    def match(line: str) -> bool:
        return bool(pattern.search(line)) if pattern else (query in line)

    paths = [base] if base.is_file() else [p for p in base.rglob("*") if p.is_file()]
    
    for fp in paths:
        try:
            if fp.stat().st_size > MAX_READ_BYTES:
                continue
            txt = fp.read_text(encoding="utf-8", errors="replace").splitlines()
            for i, line in enumerate(txt, start=1):
                if match(line):
                    hits.append({"path": str(fp), "line": i, "text": line.strip()[:200]})
                    if len(hits) >= limit:
                        return {"hits": hits, "truncated": True}
        except Exception:
            continue

    return {"hits": hits, "truncated": False}

if __name__ == "__main__":
    print(f"🌟 BIZRA Sovereign Filesystem MCP starting...")
    print(f"📂 Allowed Roots: {[str(r) for r in VALID_ROOTS]}")
    print(f"🚀 Exposing via SSE on 127.0.0.1:8000/sse")
    mcp.run(transport="sse", host="0.0.0.0", port=8000, path="/sse")
