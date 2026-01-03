#!/usr/bin/env python3
"""
OpenAPI Drift Detection Script
Validates that docs/openapi.yaml matches actual HTTP routes in src/http.rs

Part of BIZRA CI Integrity Gates
"""
import re
import sys
from pathlib import Path


def safe_read_file(file_path: Path, description: str) -> str:
    """Read file with proper error handling for CI environments.
    
    Returns file content or exits with non-zero status on error.
    Uses GitHub Actions ::error:: prefix for CI-friendly messages.
    """
    try:
        return file_path.read_text(encoding="utf-8")
    except FileNotFoundError:
        print(f"::error file={file_path}::File not found: {description} at {file_path}")
        sys.exit(1)
    except PermissionError:
        print(f"::error file={file_path}::Permission denied reading {description}: {file_path}")
        sys.exit(1)
    except UnicodeDecodeError as e:
        print(f"::error file={file_path}::Invalid UTF-8 in {description}: {file_path} ({e})")
        sys.exit(1)


def extract_routes_from_rust(http_rs_path: Path) -> set:
    """Extract route paths from src/http.rs
    
    Handles:
    - Direct string literals: .route("/path", ...)
    - Constant/variable references: .route(PATH_VAR, ...)
    - Nested router patterns: Router::new().route(...)
    """
    content = safe_read_file(http_rs_path, "Rust HTTP routes file")
    routes = set()
    
    # Step 1: Build a map of const/static/let string assignments
    # Match patterns like: const PATH: &str = "/some/path";
    #                  or: let path = "/some/path";
    #                  or: static PATH: &str = "/some/path";
    const_pattern = re.compile(
        r'(?:const|static|let)\s+(\w+)(?:\s*:\s*&?\s*str)?\s*=\s*"([^"]+)"'
    )
    identifier_to_path: dict = {}
    for match in const_pattern.finditer(content):
        ident = match.group(1)
        path_val = match.group(2)
        identifier_to_path[ident] = path_val
    
    # Step 2: Match .route("path", ...) with direct string literals
    literal_route_pattern = re.compile(r'\.route\(\s*"([^"]+)"')
    for match in literal_route_pattern.finditer(content):
        routes.add(match.group(1))
    
    # Step 3: Match .route(IDENT, ...) with identifier references
    # Match both uppercase constants (PATH_VAR) and lowercase variables (path_var)
    ident_route_pattern = re.compile(r'\.route\(\s*([A-Za-z_][A-Za-z0-9_]*)\s*,')
    for match in ident_route_pattern.finditer(content):
        ident = match.group(1)
        if ident in identifier_to_path:
            routes.add(identifier_to_path[ident])
        # If identifier not resolved, skip (don't add raw identifier)
    
    # Step 4: Handle nested Router::new().route(...) patterns (already captured by above)
    # The regex patterns above work on any .route() call regardless of context
    
    return routes

def extract_paths_from_openapi(openapi_path: Path) -> set:
    """Extract path patterns from openapi.yaml"""
    content = safe_read_file(openapi_path, "OpenAPI specification file")
    
    # Simple YAML path extraction (paths: section, lines starting with /)
    paths = set()
    in_paths_section = False
    
    for line in content.splitlines():
        stripped = line.strip()
        
        # Detect paths: section
        if line.startswith("paths:"):
            in_paths_section = True
            continue
        
        # Detect end of paths section (next top-level key)
        if in_paths_section and re.match(r'^[a-z]', line) and ":" in line:
            in_paths_section = False
            continue
        
        # Extract path (lines like "  /health:" with 2-space indent)
        if in_paths_section and line.startswith("  /"):
            path = stripped.rstrip(":")
            paths.add(path)
    
    return paths

def main():
    repo_root = Path(__file__).parent.parent
    http_rs = repo_root / "src" / "http.rs"
    openapi = repo_root / "docs" / "openapi.yaml"
    
    errors = []
    warnings = []
    
    if not http_rs.exists():
        print("::error::src/http.rs not found")
        sys.exit(1)
    
    if not openapi.exists():
        print("::error::docs/openapi.yaml not found")
        sys.exit(1)
    
    rust_routes = extract_routes_from_rust(http_rs)
    openapi_paths = extract_paths_from_openapi(openapi)
    
    print(f"📋 Routes in src/http.rs: {len(rust_routes)}")
    print(f"📋 Paths in docs/openapi.yaml: {len(openapi_paths)}")
    
    # Check for routes in Rust but missing from OpenAPI
    missing_from_openapi = rust_routes - openapi_paths
    if missing_from_openapi:
        for route in sorted(missing_from_openapi):
            # Some routes are internal/redirect - warn don't error
            if route in ("/dashboard",):
                warnings.append(f"Route '{route}' in Rust not in OpenAPI (internal)")
            else:
                errors.append(f"Route '{route}' exists in src/http.rs but missing from docs/openapi.yaml")
    
    # Check for paths in OpenAPI but missing from Rust
    missing_from_rust = openapi_paths - rust_routes
    if missing_from_rust:
        for path in sorted(missing_from_rust):
            errors.append(f"Path '{path}' in docs/openapi.yaml but not found in src/http.rs")
    
    # Report findings
    for w in warnings:
        print(f"::warning::{w}")
    
    for e in errors:
        print(f"::error::{e}")
    
    if errors:
        print(f"\n❌ OpenAPI drift detected: {len(errors)} error(s)")
        print("\n  Fix: Update docs/openapi.yaml to match src/http.rs routes")
        sys.exit(1)
    
    print("\n✅ OpenAPI spec in sync with HTTP routes")
    print(f"   Rust routes: {sorted(rust_routes)}")
    sys.exit(0)

if __name__ == "__main__":
    main()
