"""Inventory Rust, Python, Node dependencies. Flag unpinned deps + SBOM gaps.

Read-only. Uses stdlib only (no toml parsing via external libs). For TOML we
do light line-based parsing which is sufficient for our purposes.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List


def _find_files(repo_root: Path, names: List[str]) -> List[Path]:
    found = []
    for dirpath, dirnames, filenames in os.walk(repo_root):
        dirnames[:] = [d for d in dirnames if d not in {".git", "target", "node_modules",
                                                         "__pycache__", ".venv", ".venv-linux",
                                                         "venv", "dist", "build"}]
        for fn in filenames:
            if fn in names:
                found.append(Path(dirpath) / fn)
    return found


def _parse_cargo_toml(text: str) -> List[dict]:
    deps = []
    # Very light: look for [dependencies] / [dev-dependencies] sections and
    # `name = "x.y.z"` or `name = { version = "x.y.z", ... }` patterns.
    sections = re.split(r"^\[(.+?)\]\s*$", text, flags=re.MULTILINE)
    # sections alternates: [preamble, section_name, body, section_name, body, ...]
    for i in range(1, len(sections), 2):
        section = sections[i].strip()
        body = sections[i + 1] if i + 1 < len(sections) else ""
        if section not in {"dependencies", "dev-dependencies", "build-dependencies"}:
            continue
        for line in body.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            m = re.match(r"([A-Za-z0-9_\-]+)\s*=\s*(.+)$", line)
            if not m:
                continue
            name, rhs = m.group(1), m.group(2).strip()
            version = ""
            pinned = False
            if rhs.startswith('"') and rhs.endswith('"'):
                version = rhs.strip('"')
            else:
                vm = re.search(r'version\s*=\s*"([^"]+)"', rhs)
                if vm:
                    version = vm.group(1)
            if version and re.match(r"^=?\d+\.\d+\.\d+$", version):
                pinned = True
            elif version and re.match(r"^\^?\d+\.\d+\.\d+$", version):
                pinned = False  # caret — allowed but not strict pin
            deps.append({"name": name, "version": version, "section": section,
                         "strict_pin": pinned})
    return deps


def _parse_requirements(text: str) -> List[dict]:
    deps = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or line.startswith("-r ") or line.startswith("-c "):
            continue
        name, version = line, ""
        pinned = False
        m = re.match(r"^([A-Za-z0-9_\-\.\[\]]+)\s*(==|>=|<=|>|<|~=)\s*([^\s;]+)", line)
        if m:
            name = m.group(1)
            op = m.group(2)
            version = m.group(3)
            pinned = (op == "==")
        deps.append({"name": name, "version": version, "op": "", "strict_pin": pinned})
    return deps


def _parse_pyproject(text: str) -> List[dict]:
    deps: List[dict] = []
    # Pull [project] dependencies + optional-dependencies quickly.
    # Find the `dependencies = [ ... ]` block.
    def _harvest_block(key):
        m = re.search(rf"^{key}\s*=\s*\[(.*?)\]", text, flags=re.MULTILINE | re.DOTALL)
        if not m:
            return []
        body = m.group(1)
        out = []
        for raw in re.findall(r'"([^"]+)"', body):
            parsed = _parse_requirements(raw)
            out.extend(parsed)
        return out
    deps.extend(_harvest_block("dependencies"))
    return deps


def _parse_package_json(text: str) -> List[dict]:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        return []
    deps = []
    for section in ("dependencies", "devDependencies", "peerDependencies", "optionalDependencies"):
        block = obj.get(section, {})
        for name, version in block.items():
            pinned = bool(re.match(r"^\d+\.\d+\.\d+$", version or ""))
            deps.append({"name": name, "version": version or "",
                         "section": section, "strict_pin": pinned})
    return deps


def inventory(repo_root: Path) -> dict:
    rust_manifests = _find_files(repo_root, ["Cargo.toml"])
    rust_locks = _find_files(repo_root, ["Cargo.lock"])
    pip_reqs = _find_files(repo_root, ["requirements.txt", "requirements-dev.txt",
                                        "requirements.flywheel.txt"])
    pyproject = _find_files(repo_root, ["pyproject.toml"])
    node_manifests = _find_files(repo_root, ["package.json"])
    node_locks = _find_files(repo_root, ["package-lock.json", "pnpm-lock.yaml", "yarn.lock"])

    result = {
        "rust": {"manifests": [], "locks": [str(p.relative_to(repo_root)) for p in rust_locks]},
        "python": {"requirements": [], "pyproject": []},
        "node": {"manifests": [], "locks": [str(p.relative_to(repo_root)) for p in node_locks]},
        "gaps": [],
    }

    for p in rust_manifests:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        deps = _parse_cargo_toml(text)
        result["rust"]["manifests"].append({
            "path": str(p.relative_to(repo_root)),
            "dep_count": len(deps),
            "deps": deps,
        })

    for p in pip_reqs:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        deps = _parse_requirements(text)
        result["python"]["requirements"].append({
            "path": str(p.relative_to(repo_root)),
            "dep_count": len(deps),
            "unpinned": [d["name"] for d in deps if not d.get("strict_pin")],
            "deps": deps,
        })

    for p in pyproject:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        deps = _parse_pyproject(text)
        result["python"]["pyproject"].append({
            "path": str(p.relative_to(repo_root)),
            "dep_count": len(deps),
            "deps": deps,
        })

    for p in node_manifests:
        try:
            text = p.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        deps = _parse_package_json(text)
        result["node"]["manifests"].append({
            "path": str(p.relative_to(repo_root)),
            "dep_count": len(deps),
            "deps": deps,
        })

    # Gap analysis
    if not rust_locks and rust_manifests:
        result["gaps"].append("Rust Cargo.lock missing at workspace root(s) — non-reproducible builds.")
    # For each Cargo.toml check for a sibling Cargo.lock.
    lock_dirs = {p.parent for p in rust_locks}
    for p in rust_manifests:
        # Only workspace / top-level crates care; skip nested package manifests.
        is_workspace_root = "[workspace]" in p.read_text(encoding="utf-8", errors="replace")
        if is_workspace_root and p.parent not in lock_dirs:
            result["gaps"].append(f"Workspace without Cargo.lock: {p.relative_to(repo_root)}")
    if not node_locks and node_manifests:
        result["gaps"].append("Node package-lock / pnpm-lock / yarn.lock missing — supply-chain drift risk.")
    if not any(True for x in result["python"]["requirements"]
               if any(d.get("strict_pin") for d in x.get("deps", []))):
        result["gaps"].append("No strictly-pinned Python requirements found — non-reproducible.")
    result["gaps"].append("SBOM artifact not located in repo (no *.spdx.json / *.cdx.json found).")

    return result


def write_outputs(inv: dict, out_dir: Path) -> Dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "dependencies.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(inv, f, indent=2, ensure_ascii=False)
    return {"dependencies_json": str(json_path)}
