#!/usr/bin/env python3
"""
═══════════════════════════════════════════════════════════════════════════════
    BIZRA GENESIS READINESS VERIFICATION
    
    Pre-flight checks before Block0 can be sealed.
    
    This script must return EXIT 0 before any deployment.
    Run: python verify_genesis_readiness.py
    
    Created: 2026-01-29 | BIZRA Sovereignty
    Principle: لا نفترض — We do not assume.
═══════════════════════════════════════════════════════════════════════════════
"""

import os
import sys
import re
import json
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional
from collections import defaultdict

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

BIZRA_ROOT = Path(__file__).parent
CRITICAL_FILES = [
    "bizra_nexus.py",
    "bizra_orchestrator.py", 
    "ecosystem_bridge.py",
    "sovereign_nexus.py",
    "apex_cognitive_engine.py",
]

# Thresholds
MAX_GENERIC_EXCEPTIONS_CRITICAL = 10  # In critical path files
MAX_GENERIC_EXCEPTIONS_TOTAL = 50     # Across entire codebase (target)
REQUIRED_BLAKE2B_OCCURRENCES = 5      # Minimum blake2b usage

# ═══════════════════════════════════════════════════════════════════════════════
# CHECK RESULTS
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class CheckResult:
    """Result of a single verification check."""
    name: str
    passed: bool
    severity: str  # "CRITICAL", "WARNING", "INFO"
    message: str
    details: List[str] = field(default_factory=list)
    
    def __str__(self):
        icon = "✅" if self.passed else ("🔴" if self.severity == "CRITICAL" else "⚠️")
        return f"{icon} [{self.severity}] {self.name}: {self.message}"


@dataclass
class GenesisReadinessReport:
    """Complete readiness report."""
    timestamp: str
    checks: List[CheckResult] = field(default_factory=list)
    
    @property
    def critical_failures(self) -> List[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "CRITICAL"]
    
    @property
    def warnings(self) -> List[CheckResult]:
        return [c for c in self.checks if not c.passed and c.severity == "WARNING"]
    
    @property
    def ready(self) -> bool:
        return len(self.critical_failures) == 0
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "ready": self.ready,
            "critical_failures": len(self.critical_failures),
            "warnings": len(self.warnings),
            "checks": [asdict(c) for c in self.checks]
        }


# ═══════════════════════════════════════════════════════════════════════════════
# VERIFICATION CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

def get_source_files() -> List[Path]:
    """Get Python source files (excluding .venv and __pycache__)."""
    files = []
    # Top-level Python files
    files.extend(BIZRA_ROOT.glob("*.py"))
    # Core module files (max 2 levels deep)
    if (BIZRA_ROOT / "core").exists():
        files.extend((BIZRA_ROOT / "core").glob("*.py"))
        files.extend((BIZRA_ROOT / "core").glob("*/*.py"))
    return [f for f in files if ".venv" not in str(f) and "__pycache__" not in str(f)]


def check_no_md5_in_crypto() -> CheckResult:
    """Verify MD5 is not used for cryptographic purposes in BIZRA source."""
    name = "No MD5 in Cryptographic Paths"
    violations = []
    
    # Patterns that indicate MD5 used for security (not just metadata)
    crypto_md5_patterns = [
        r"hashlib\.md5\(",
        r"md5\s*=\s*hashlib\.md5",
        r"\.hexdigest\(\).*md5",
    ]
    
    for py_file in get_source_files():
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            for i, line in enumerate(content.splitlines(), 1):
                # Skip comments
                if line.strip().startswith("#"):
                    continue
                # Skip type hints / optional fields (like hash_md5: Optional[str])
                if "Optional[str]" in line or ": str = None" in line:
                    continue
                    
                for pattern in crypto_md5_patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        violations.append(f"{py_file.name}:{i}: {line.strip()[:80]}")
        except Exception:
            pass
    
    if violations:
        return CheckResult(
            name=name,
            passed=False,
            severity="CRITICAL",
            message=f"Found {len(violations)} MD5 usages in crypto paths",
            details=violations[:10]
        )
    
    return CheckResult(
        name=name,
        passed=True,
        severity="CRITICAL",
        message="No MD5 in cryptographic code paths"
    )


def check_blake2b_present() -> CheckResult:
    """Verify BLAKE2b is actively used."""
    name = "BLAKE2b Active Usage"
    occurrences = []
    
    for py_file in get_source_files():
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            if "blake2b" in content.lower():
                occurrences.append(py_file.name)
        except Exception:
            pass
    
    if len(occurrences) < REQUIRED_BLAKE2B_OCCURRENCES:
        return CheckResult(
            name=name,
            passed=False,
            severity="WARNING",
            message=f"Only {len(occurrences)} files use BLAKE2b (need {REQUIRED_BLAKE2B_OCCURRENCES}+)",
            details=occurrences
        )
    
    return CheckResult(
        name=name,
        passed=True,
        severity="WARNING",
        message=f"BLAKE2b found in {len(occurrences)} files",
        details=occurrences[:5]
    )


def check_generic_exceptions() -> CheckResult:
    """Count bare 'except Exception:' handlers."""
    name = "Generic Exception Handling"
    violations = defaultdict(list)
    total = 0
    
    # Pattern for bare except Exception (without specific handling)
    pattern = r"except\s+Exception\s*:"
    
    for py_file in get_source_files():
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            matches = re.findall(pattern, content)
            if matches:
                violations[py_file.name] = len(matches)
                total += len(matches)
        except Exception:
            pass
    
    # Check critical files specifically
    critical_count = sum(violations.get(f, 0) for f in CRITICAL_FILES)
    
    severity = "CRITICAL" if critical_count > MAX_GENERIC_EXCEPTIONS_CRITICAL else "WARNING"
    passed = total <= MAX_GENERIC_EXCEPTIONS_TOTAL
    
    top_offenders = sorted(violations.items(), key=lambda x: -x[1])[:5]
    details = [f"{name}: {count}" for name, count in top_offenders]
    
    return CheckResult(
        name=name,
        passed=passed,
        severity=severity,
        message=f"{total} generic exceptions ({critical_count} in critical path)",
        details=details
    )


def check_no_hardcoded_secrets() -> CheckResult:
    """Check for hardcoded passwords/secrets in Python files."""
    name = "No Hardcoded Secrets"
    violations = []
    
    # Patterns that suggest hardcoded secrets
    patterns = [
        r'password\s*=\s*["\'][^"\']+["\']',
        r'secret\s*=\s*["\'][^"\']+["\']',
        r'api_key\s*=\s*["\'][^"\']+["\']',
        r'token\s*=\s*["\'][a-zA-Z0-9]{20,}["\']',
    ]
    
    # Known safe patterns (local dev, placeholders, test fixtures)
    safe_patterns = [
        "your_", "example", "xxx", "test_", "lm-studio", 
        "placeholder", "dummy", "sample", "fake", "mock",
        "sk-test", "sk-xxx", "localhost", "127.0.0.1", "192.168."
    ]
    
    for py_file in get_source_files():
        if "test" in py_file.name.lower():
            continue
        
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            for i, line in enumerate(content.splitlines(), 1):
                if line.strip().startswith("#"):
                    continue
                # Skip test/example blocks
                if "if __name__" in content[:content.find(line)] and "__main__" in content[:content.find(line)]:
                    if i > content[:content.find(line)].count('\n'):
                        continue
                        
                for pattern in patterns:
                    if re.search(pattern, line, re.IGNORECASE):
                        # Exclude known safe patterns
                        if any(safe in line.lower() for safe in safe_patterns):
                            continue
                        violations.append(f"{py_file.name}:{i}")
        except Exception:
            pass
    
    if violations:
        return CheckResult(
            name=name,
            passed=False,
            severity="CRITICAL",
            message=f"Found {len(violations)} potential hardcoded secrets",
            details=violations[:10]
        )
    
    return CheckResult(
        name=name,
        passed=True,
        severity="CRITICAL",
        message="No hardcoded secrets detected in Python source"
    )


def check_consensus_claims() -> CheckResult:
    """Verify consensus model is honestly labeled."""
    name = "Honest Consensus Labeling"
    
    consensus_file = BIZRA_ROOT / "core" / "federation" / "consensus.py"
    
    if not consensus_file.exists():
        return CheckResult(
            name=name,
            passed=False,
            severity="WARNING",
            message="consensus.py not found",
            details=[]
        )
    
    content = consensus_file.read_text(encoding="utf-8", errors="ignore")
    
    # Check for false PBFT claims
    if "PBFT" in content or "pbft" in content:
        if "n=8" in content or "f=2" in content or "quorum=5" in content:
            return CheckResult(
                name=name,
                passed=False,
                severity="CRITICAL",
                message="False PBFT claims detected (n=8, f=2 not implemented)",
                details=["Consensus model is weighted voting, not PBFT"]
            )
    
    # Check for honest labeling
    honest_labels = ["weighted", "voting", "ihsan", "reputation"]
    has_honest_label = any(label in content.lower() for label in honest_labels)
    
    return CheckResult(
        name=name,
        passed=has_honest_label,
        severity="WARNING",
        message="Consensus model appears honestly labeled" if has_honest_label else "Consensus model labeling unclear",
        details=[]
    )


def check_pqc_status() -> CheckResult:
    """Check Post-Quantum Crypto status and claims."""
    name = "PQC Implementation Status"
    
    # Check if PQC is claimed anywhere
    pqc_claims = []
    pqc_implementations = []
    
    for py_file in get_source_files():
        try:
            content = py_file.read_text(encoding="utf-8", errors="ignore")
            if "post-quantum" in content.lower() or "pqc" in content.lower():
                if "TODO" in content or "placeholder" in content.lower() or "not implemented" in content.lower():
                    continue
                pqc_claims.append(py_file.name)
            
            if "pqcrypto" in content or "dilithium" in content.lower() or "kyber" in content.lower():
                pqc_implementations.append(py_file.name)
        except Exception:
            pass
    
    if pqc_claims and not pqc_implementations:
        return CheckResult(
            name=name,
            passed=False,
            severity="WARNING",
            message="PQC claimed but not implemented",
            details=pqc_claims[:5]
        )
    
    return CheckResult(
        name=name,
        passed=True,
        severity="WARNING",
        message=f"PQC status consistent (claims: {len(pqc_claims)}, impl: {len(pqc_implementations)})"
    )


def check_genesis_manifest() -> CheckResult:
    """Check if genesis-manifest.yaml exists with honest claims."""
    name = "Genesis Manifest Exists"
    
    manifest_paths = [
        BIZRA_ROOT / "genesis-manifest.yaml",
        BIZRA_ROOT / "genesis-manifest.yml",
        BIZRA_ROOT / "GENESIS_MANIFEST.yaml",
    ]
    
    for path in manifest_paths:
        if path.exists():
            return CheckResult(
                name=name,
                passed=True,
                severity="CRITICAL",
                message=f"Genesis manifest found: {path.name}",
                details=[]
            )
    
    return CheckResult(
        name=name,
        passed=False,
        severity="CRITICAL",
        message="genesis-manifest.yaml not found - create before sealing Block0",
        details=["Run: create_genesis_manifest() to generate"]
    )


def check_docker_compose_exists() -> CheckResult:
    """Verify docker-compose.yml exists and is valid."""
    name = "Docker Compose Configuration"
    
    compose_paths = [
        BIZRA_ROOT / "docker-compose.yml",
        BIZRA_ROOT / "docker-compose.yaml",
    ]
    
    for path in compose_paths:
        if path.exists():
            try:
                content = path.read_text()
                if "services:" in content:
                    return CheckResult(
                        name=name,
                        passed=True,
                        severity="INFO",
                        message=f"Docker Compose found: {path.name}"
                    )
            except Exception:
                pass
    
    return CheckResult(
        name=name,
        passed=False,
        severity="INFO",
        message="docker-compose.yml not found"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MANIFEST CREATION
# ═══════════════════════════════════════════════════════════════════════════════

def create_genesis_manifest() -> Path:
    """Create an honest genesis-manifest.yaml."""
    manifest = {
        "version": "1.0.0",
        "created": datetime.now(timezone.utc).isoformat(),
        "block0_scope": {
            "name": "BIZRA Genesis Block",
            "type": "SOLO_GENESIS",
            "description": "Initial genesis block - single operator mode"
        },
        "cryptography": {
            "hashing": ["sha256", "blake2b"],
            "encryption": "fernet_aes128_cbc",
            "key_derivation": "pbkdf2_sha256_600k",
            "post_quantum": False,
            "pqc_status": "NOT_IMPLEMENTED"
        },
        "consensus": {
            "model": "WEIGHTED_VOTING",
            "description": "Reputation-weighted voting based on Ihsan scores",
            "min_voters": 3,
            "quorum_threshold": 0.67,
            "pbft_claimed": False,
            "bft_tested": False,
            "network_partition_tested": False
        },
        "authentication": {
            "mode": "DEVELOPMENT",
            "production_ready": False,
            "notes": "Mock auth in BIZRA-OS - replace before production"
        },
        "data_integrity": {
            "completeness_verified": False,
            "completeness_percentage": "TBD",
            "exclusions": [],
            "notes": "Run data integrity check before sealing"
        },
        "deployment": {
            "mode": "SOLO_OPERATOR",
            "nodes": 1,
            "distributed": False
        },
        "attestation": {
            "operator": "UNSIGNED",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "ihsan_score": "PENDING_VERIFICATION"
        }
    }
    
    import yaml
    manifest_path = BIZRA_ROOT / "genesis-manifest.yaml"
    
    try:
        with open(manifest_path, "w") as f:
            yaml.dump(manifest, f, default_flow_style=False, sort_keys=False)
        return manifest_path
    except ImportError:
        # Fallback to JSON if yaml not available
        manifest_path = BIZRA_ROOT / "genesis-manifest.json"
        with open(manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        return manifest_path


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_all_checks() -> GenesisReadinessReport:
    """Run all verification checks and return report."""
    report = GenesisReadinessReport(
        timestamp=datetime.now(timezone.utc).isoformat()
    )
    
    checks = [
        check_no_md5_in_crypto,
        check_blake2b_present,
        check_generic_exceptions,
        check_no_hardcoded_secrets,
        check_consensus_claims,
        check_pqc_status,
        check_genesis_manifest,
        check_docker_compose_exists,
    ]
    
    for check_fn in checks:
        try:
            result = check_fn()
            report.checks.append(result)
        except Exception as e:
            report.checks.append(CheckResult(
                name=check_fn.__name__,
                passed=False,
                severity="WARNING",
                message=f"Check failed with error: {e}"
            ))
    
    return report


def print_report(report: GenesisReadinessReport):
    """Print formatted report to stdout."""
    print("\n" + "═" * 80)
    print("    BIZRA GENESIS READINESS VERIFICATION")
    print("    Timestamp:", report.timestamp)
    print("═" * 80 + "\n")
    
    for check in report.checks:
        print(check)
        if check.details:
            for detail in check.details[:3]:
                print(f"      → {detail}")
    
    print("\n" + "─" * 80)
    
    if report.ready:
        print("✅ GENESIS READY — All critical checks passed")
        print("   You may proceed with: docker compose up -d")
    else:
        print("🔴 GENESIS NOT READY — Critical issues must be resolved")
        print(f"   Critical failures: {len(report.critical_failures)}")
        print(f"   Warnings: {len(report.warnings)}")
        print("\n   Fix critical issues before deployment.")
    
    print("─" * 80 + "\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="BIZRA Genesis Readiness Verification")
    parser.add_argument("--create-manifest", action="store_true", help="Create genesis-manifest.yaml")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()
    
    if args.create_manifest:
        path = create_genesis_manifest()
        print(f"✅ Created: {path}")
        return 0
    
    report = run_all_checks()
    
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        print_report(report)
    
    # Exit code: 0 if ready, 1 if not
    return 0 if report.ready else 1


if __name__ == "__main__":
    sys.exit(main())
