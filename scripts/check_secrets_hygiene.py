#!/usr/bin/env python3
"""
Secrets Hygiene Check Script
Ensures no .env files are committed and sensitive patterns are blocked.

Part of BIZRA CI Integrity Gates
"""
import re
import subprocess
import sys
from pathlib import Path

# Files that should NEVER be committed
BLOCKED_FILES = [
    ".env",
    ".env.local",
    ".env.production",
    ".env.development",
    "secrets.json",
    "credentials.json",
    "*.pem",
    "*.key",
]

# Patterns that indicate secrets (check in tracked files)
SECRET_PATTERNS = [
    (r'OPENAI_API_KEY\s*=\s*sk-[a-zA-Z0-9]+', "OpenAI API key"),
    (r'ANTHROPIC_API_KEY\s*=\s*[a-zA-Z0-9_-]+', "Anthropic API key"),
    (r'password\s*=\s*["\'][^"\']{8,}["\']', "Hardcoded password"),
    (r'secret\s*=\s*["\'][^"\']{16,}["\']', "Hardcoded secret"),
    (r'ghp_[a-zA-Z0-9]{36}', "GitHub personal access token"),
    (r'gho_[a-zA-Z0-9]{36}', "GitHub OAuth token"),
    (r'-----BEGIN RSA PRIVATE KEY-----', "RSA private key"),
    (r'-----BEGIN OPENSSH PRIVATE KEY-----', "SSH private key"),
]

# Files to exclude from secret scanning
EXCLUDED_PATHS = [
    ".env.example",
    "docs/",
    "*.md",
    "target/",
    ".git/",
]

def get_tracked_files() -> list:
    """Get list of git-tracked files"""
    try:
        result = subprocess.run(
            ["git", "ls-files"],
            capture_output=True,
            text=True,
            check=True
        )
        return result.stdout.strip().split("\n")
    except subprocess.CalledProcessError:
        return []

def should_exclude(path: str) -> bool:
    """Check if path should be excluded from scanning"""
    for pattern in EXCLUDED_PATHS:
        if pattern.endswith("/"):
            if path.startswith(pattern) or f"/{pattern}" in path:
                return True
        elif pattern.startswith("*."):
            ext = pattern[1:]
            if path.endswith(ext):
                return True
        elif path == pattern or path.endswith(f"/{pattern}"):
            return True
    return False

def main():
    repo_root = Path(__file__).parent.parent
    errors = []
    warnings = []
    
    # Check 1: Blocked files should not exist in git
    tracked_files = get_tracked_files()
    
    for blocked in BLOCKED_FILES:
        if blocked.startswith("*."):
            ext = blocked[1:]
            matches = [f for f in tracked_files if f.endswith(ext)]
            for match in matches:
                if not should_exclude(match):
                    errors.append(f"Blocked file pattern '{blocked}' matched: {match}")
        else:
            if blocked in tracked_files:
                errors.append(f"Blocked file '{blocked}' is tracked in git")
    
    # Check 2: .env file should not exist at all (even untracked)
    env_file = repo_root / ".env"
    if env_file.exists():
        # Check if it's in .gitignore
        gitignore = repo_root / ".gitignore"
        if gitignore.exists():
            gitignore_content = gitignore.read_text()
            if ".env" not in gitignore_content:
                errors.append(".env exists and is NOT in .gitignore")
            else:
                warnings.append(".env exists locally (OK - in .gitignore)")
        else:
            errors.append(".env exists but no .gitignore found")
    
    # Check 3: .env.example should exist
    env_example = repo_root / ".env.example"
    if not env_example.exists():
        warnings.append(".env.example not found - consider adding one")
    else:
        # Verify .env.example has no actual secrets
        content = env_example.read_text()
        for pattern, name in SECRET_PATTERNS:
            if re.search(pattern, content, re.IGNORECASE):
                errors.append(f".env.example contains actual secret: {name}")
    
    # Check 4: Scan tracked files for secret patterns
    print("🔍 Scanning tracked files for secrets...")
    files_scanned = 0
    
    for file_path in tracked_files:
        if should_exclude(file_path):
            continue
        
        full_path = repo_root / file_path
        if not full_path.exists() or not full_path.is_file():
            continue
        
        try:
            content = full_path.read_text(encoding="utf-8", errors="ignore")
            files_scanned += 1
            
            for pattern, name in SECRET_PATTERNS:
                if re.search(pattern, content, re.IGNORECASE):
                    errors.append(f"Potential secret ({name}) found in: {file_path}")
        except Exception:
            pass  # Skip binary/unreadable files
    
    print(f"   Scanned {files_scanned} files")
    
    # Report findings
    for w in warnings:
        print(f"::warning::{w}")
    
    for e in errors:
        print(f"::error::{e}")
    
    if errors:
        print(f"\n❌ Secrets hygiene check failed: {len(errors)} error(s)")
        print("\n  Fix: Remove secrets from tracked files, use .env.example with placeholders")
        sys.exit(1)
    
    print("\n✅ Secrets hygiene check passed")
    sys.exit(0)

if __name__ == "__main__":
    main()
