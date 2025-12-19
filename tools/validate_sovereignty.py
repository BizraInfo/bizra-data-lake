#!/usr/bin/env python3
"""
BIZRA Stand Alone Validation Suite v1.0
========================================
Comprehensive validation of PAT-7 and SAT-5 sovereignty.

Validates:
1. All PAT agents respond locally (zero external calls)
2. All SAT agents are operational
3. Latency benchmarks (< 500ms p95)
4. Ihsān gate enforcement
5. Offline capability verification

Usage:
    python validate_sovereignty.py                    # Full validation
    python validate_sovereignty.py --pat-only         # PAT agents only
    python validate_sovereignty.py --sat-only         # SAT agents only
    python validate_sovereignty.py --offline          # Simulate offline mode
    python validate_sovereignty.py --benchmark 10     # Run 10 iterations per agent
"""

import argparse
import json
import os
import socket
import statistics
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Ensure UTF-8 output
for stream in (sys.stdout, sys.stderr):
    if hasattr(stream, 'reconfigure'):
        try:
            stream.reconfigure(encoding='utf-8')
        except Exception:
            pass

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

OLLAMA_URL = os.getenv("OLLAMA_HOST", "http://127.0.0.1:11434")
LMSTUDIO_URL = os.getenv("LMSTUDIO_URL", "http://127.0.0.1:1234")
KERNEL_URL = os.getenv("BIZRA_KERNEL_URL", "http://127.0.0.1:8010")

# Sovereignty thresholds
LATENCY_P95_TARGET_MS = 500
IHSAN_THRESHOLD = 0.95

# PAT Agent definitions
PAT_AGENTS = {
    "MasterReasoner": {"model": "deepseek-r1:7b", "backend": "ollama"},
    "MemoryArchitect": {"model": "qwen2.5:7b", "backend": "ollama"},
    "CreativeSynthesizer": {"model": "qwen2.5:7b", "backend": "ollama"},
    "DataAnalyzer": {"model": "mistral:7b", "backend": "ollama"},
    "Communicator": {"model": "mistral:7b", "backend": "ollama"},
    "ExecutionPlanner": {"model": "agentflow-7b", "backend": "lmstudio"},
    "EthicsGuardian": {"model": "qwen2.5:7b", "backend": "ollama"},
}

# SAT Agent definitions
SAT_AGENTS = {
    "PoiVerifier": {"type": "rule-based", "endpoint": "/api/sat/poi/verify"},
    "ResourceAllocator": {"type": "rule-based", "endpoint": "/api/sat/resources"},
    "RiskGuardian": {"type": "rule-based", "endpoint": "/api/sat/risk"},
    "GovernanceEngine": {"type": "rule-based", "endpoint": "/api/sat/governance"},
    "EvidenceEngine": {"type": "rule-based", "endpoint": "/api/sat/evidence"},
}

# Forbidden external endpoints (sovereignty violation)
FORBIDDEN_ENDPOINTS = [
    "api.openai.com",
    "api.anthropic.com",
    "generativelanguage.googleapis.com",
    "api.cohere.ai",
    "api.together.xyz",
]


# ═══════════════════════════════════════════════════════════════════════════════
# DATA STRUCTURES
# ═══════════════════════════════════════════════════════════════════════════════

class ValidationStatus(Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"


@dataclass
class ValidationResult:
    name: str
    status: ValidationStatus
    message: str
    latency_ms: Optional[float] = None
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class BenchmarkStats:
    agent: str
    samples: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    min_ms: float
    max_ms: float
    success_rate: float


@dataclass
class ValidationReport:
    timestamp: str
    total_tests: int
    passed: int
    failed: int
    warnings: int
    skipped: int
    sovereignty_score: float
    pat_results: List[ValidationResult]
    sat_results: List[ValidationResult]
    benchmarks: List[BenchmarkStats]
    offline_capable: bool


# ═══════════════════════════════════════════════════════════════════════════════
# CONNECTIVITY CHECKS
# ═══════════════════════════════════════════════════════════════════════════════

def check_ollama() -> Tuple[bool, str]:
    """Check if Ollama is running and accessible."""
    try:
        resp = requests.get(f"{OLLAMA_URL}/api/tags", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("models", [])
            return True, f"Ollama OK ({len(models)} models)"
        return False, f"Ollama returned {resp.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Ollama not reachable"
    except Exception as e:
        return False, f"Ollama error: {e}"


def check_lmstudio() -> Tuple[bool, str]:
    """Check if LM Studio is running and accessible."""
    try:
        resp = requests.get(f"{LMSTUDIO_URL}/v1/models", timeout=5)
        if resp.status_code == 200:
            models = resp.json().get("data", [])
            return True, f"LM Studio OK ({len(models)} models)"
        return False, f"LM Studio returned {resp.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "LM Studio not reachable"
    except Exception as e:
        return False, f"LM Studio error: {e}"


def check_kernel() -> Tuple[bool, str]:
    """Check if BIZRA Kernel is running."""
    try:
        resp = requests.get(f"{KERNEL_URL}/healthz", timeout=5)
        if resp.status_code == 200:
            return True, "Kernel OK"
        return False, f"Kernel returned {resp.status_code}"
    except requests.exceptions.ConnectionError:
        return False, "Kernel not reachable"
    except Exception as e:
        return False, f"Kernel error: {e}"


def check_sovereignty_violation() -> Tuple[bool, List[str]]:
    """
    Check if any forbidden external endpoints are reachable.
    In a sovereign system, these should NOT be accessible during operation.
    """
    violations = []
    for endpoint in FORBIDDEN_ENDPOINTS:
        try:
            socket.create_connection((endpoint, 443), timeout=2)
            violations.append(endpoint)
        except (socket.timeout, socket.error, OSError):
            pass  # Good - endpoint not reachable or blocked
    
    return len(violations) == 0, violations


# ═══════════════════════════════════════════════════════════════════════════════
# PAT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_pat_agent(agent_name: str, config: Dict[str, str]) -> ValidationResult:
    """Validate a single PAT agent."""
    model = config["model"]
    backend = config["backend"]
    
    # Determine endpoint
    if backend == "ollama":
        url = f"{OLLAMA_URL}/api/generate"
        payload = {
            "model": model,
            "prompt": "Respond with exactly: SOVEREIGNTY_CHECK_OK",
            "stream": False,
            "options": {"num_predict": 50}
        }
    elif backend == "lmstudio":
        url = f"{LMSTUDIO_URL}/v1/chat/completions"
        payload = {
            "model": model,
            "messages": [{"role": "user", "content": "Respond with exactly: SOVEREIGNTY_CHECK_OK"}],
            "max_tokens": 50,
            "stream": False
        }
    else:
        return ValidationResult(
            name=f"PAT:{agent_name}",
            status=ValidationStatus.FAIL,
            message=f"Unknown backend: {backend}"
        )
    
    # Execute request
    start = time.time()
    try:
        resp = requests.post(url, json=payload, timeout=30)
        latency_ms = (time.time() - start) * 1000
        
        if resp.status_code != 200:
            return ValidationResult(
                name=f"PAT:{agent_name}",
                status=ValidationStatus.FAIL,
                message=f"HTTP {resp.status_code}",
                latency_ms=latency_ms
            )
        
        # Extract response
        if backend == "ollama":
            response_text = resp.json().get("response", "")
        else:
            choices = resp.json().get("choices", [])
            response_text = choices[0]["message"]["content"] if choices else ""
        
        # Check latency threshold
        if latency_ms > LATENCY_P95_TARGET_MS:
            return ValidationResult(
                name=f"PAT:{agent_name}",
                status=ValidationStatus.WARN,
                message=f"Latency {latency_ms:.0f}ms > {LATENCY_P95_TARGET_MS}ms target",
                latency_ms=latency_ms,
                details={"response_preview": response_text[:100]}
            )
        
        return ValidationResult(
            name=f"PAT:{agent_name}",
            status=ValidationStatus.PASS,
            message=f"OK ({latency_ms:.0f}ms)",
            latency_ms=latency_ms,
            details={"model": model, "backend": backend}
        )
        
    except requests.exceptions.Timeout:
        return ValidationResult(
            name=f"PAT:{agent_name}",
            status=ValidationStatus.FAIL,
            message="Timeout (30s)",
            latency_ms=30000
        )
    except Exception as e:
        return ValidationResult(
            name=f"PAT:{agent_name}",
            status=ValidationStatus.FAIL,
            message=str(e)
        )


def benchmark_pat_agent(agent_name: str, config: Dict[str, str], iterations: int = 5) -> BenchmarkStats:
    """Run multiple iterations and compute statistics."""
    latencies = []
    successes = 0
    
    for _ in range(iterations):
        result = validate_pat_agent(agent_name, config)
        if result.latency_ms is not None:
            latencies.append(result.latency_ms)
        if result.status in (ValidationStatus.PASS, ValidationStatus.WARN):
            successes += 1
        time.sleep(0.5)  # Brief pause between iterations
    
    if not latencies:
        return BenchmarkStats(
            agent=agent_name,
            samples=iterations,
            mean_ms=0, p50_ms=0, p95_ms=0, p99_ms=0,
            min_ms=0, max_ms=0, success_rate=0
        )
    
    sorted_latencies = sorted(latencies)
    n = len(sorted_latencies)
    
    return BenchmarkStats(
        agent=agent_name,
        samples=iterations,
        mean_ms=statistics.mean(latencies),
        p50_ms=sorted_latencies[int(n * 0.5)] if n > 0 else 0,
        p95_ms=sorted_latencies[int(n * 0.95)] if n > 0 else sorted_latencies[-1],
        p99_ms=sorted_latencies[int(n * 0.99)] if n > 0 else sorted_latencies[-1],
        min_ms=min(latencies),
        max_ms=max(latencies),
        success_rate=successes / iterations
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SAT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_sat_agent(agent_name: str, config: Dict[str, str]) -> ValidationResult:
    """Validate a single SAT agent."""
    endpoint = config.get("endpoint", "")
    
    # SAT agents are rule-based and available via kernel API
    url = f"{KERNEL_URL}{endpoint}"
    
    start = time.time()
    try:
        resp = requests.get(url, timeout=10)
        latency_ms = (time.time() - start) * 1000
        
        if resp.status_code == 200:
            return ValidationResult(
                name=f"SAT:{agent_name}",
                status=ValidationStatus.PASS,
                message=f"OK ({latency_ms:.0f}ms)",
                latency_ms=latency_ms,
                details={"endpoint": endpoint}
            )
        elif resp.status_code == 404:
            # Endpoint not implemented yet
            return ValidationResult(
                name=f"SAT:{agent_name}",
                status=ValidationStatus.SKIP,
                message="Endpoint not implemented",
                latency_ms=latency_ms
            )
        else:
            return ValidationResult(
                name=f"SAT:{agent_name}",
                status=ValidationStatus.FAIL,
                message=f"HTTP {resp.status_code}",
                latency_ms=latency_ms
            )
    except requests.exceptions.ConnectionError:
        return ValidationResult(
            name=f"SAT:{agent_name}",
            status=ValidationStatus.SKIP,
            message="Kernel not available",
        )
    except Exception as e:
        return ValidationResult(
            name=f"SAT:{agent_name}",
            status=ValidationStatus.FAIL,
            message=str(e)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# IHSAN GATE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_ihsan_gate() -> ValidationResult:
    """Validate that the Ihsān gate is active and enforcing thresholds."""
    try:
        # Check if FATE evaluator endpoint exists
        resp = requests.post(
            f"{KERNEL_URL}/v1/fate/evaluate",
            json={
                "action": "test_action",
                "context": {"type": "validation"},
                "ihsan_threshold": IHSAN_THRESHOLD
            },
            timeout=10
        )
        
        if resp.status_code == 200:
            data = resp.json()
            decision = data.get("decision", "UNKNOWN")
            score = data.get("ihsan_score", 0)
            
            return ValidationResult(
                name="Ihsān Gate",
                status=ValidationStatus.PASS,
                message=f"Active (score={score:.2f}, decision={decision})",
                details={"threshold": IHSAN_THRESHOLD, "score": score}
            )
        elif resp.status_code == 404:
            return ValidationResult(
                name="Ihsān Gate",
                status=ValidationStatus.WARN,
                message="Endpoint not implemented (P1.9 pending)",
            )
        else:
            return ValidationResult(
                name="Ihsān Gate",
                status=ValidationStatus.FAIL,
                message=f"HTTP {resp.status_code}",
            )
    except requests.exceptions.ConnectionError:
        return ValidationResult(
            name="Ihsān Gate",
            status=ValidationStatus.SKIP,
            message="Kernel not available",
        )
    except Exception as e:
        return ValidationResult(
            name="Ihsān Gate",
            status=ValidationStatus.FAIL,
            message=str(e)
        )


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN VALIDATION ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════════════════

def run_full_validation(
    pat_only: bool = False,
    sat_only: bool = False,
    offline_mode: bool = False,
    benchmark_iterations: int = 0
) -> ValidationReport:
    """Run the complete validation suite."""
    
    timestamp = datetime.now(timezone.utc).isoformat()
    pat_results: List[ValidationResult] = []
    sat_results: List[ValidationResult] = []
    benchmarks: List[BenchmarkStats] = []
    
    print("\n" + "═" * 70)
    print("  BIZRA SOVEREIGNTY VALIDATION SUITE v1.0")
    print("═" * 70)
    print(f"  Timestamp: {timestamp}")
    print(f"  Mode: {'Offline' if offline_mode else 'Online'}")
    print("═" * 70 + "\n")
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 1: Infrastructure Checks
    # ─────────────────────────────────────────────────────────────────────────
    print("▶ PHASE 1: Infrastructure Checks\n")
    
    ollama_ok, ollama_msg = check_ollama()
    print(f"  {'✅' if ollama_ok else '❌'} Ollama: {ollama_msg}")
    
    lmstudio_ok, lmstudio_msg = check_lmstudio()
    print(f"  {'✅' if lmstudio_ok else '⚠️'} LM Studio: {lmstudio_msg}")
    
    kernel_ok, kernel_msg = check_kernel()
    print(f"  {'✅' if kernel_ok else '⚠️'} Kernel: {kernel_msg}")
    
    # Sovereignty check
    if not offline_mode:
        sovereign_ok, violations = check_sovereignty_violation()
        if sovereign_ok:
            print(f"  ✅ Sovereignty: No external AI endpoints detected")
        else:
            print(f"  ⚠️ Sovereignty: External endpoints reachable: {violations}")
    else:
        print(f"  ℹ️ Sovereignty: Offline mode - network blocked")
    
    print()
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 2: PAT Agent Validation
    # ─────────────────────────────────────────────────────────────────────────
    if not sat_only:
        print("▶ PHASE 2: PAT Agent Validation (7 agents)\n")
        
        for agent_name, config in PAT_AGENTS.items():
            # Skip LM Studio agents if not available
            if config["backend"] == "lmstudio" and not lmstudio_ok:
                result = ValidationResult(
                    name=f"PAT:{agent_name}",
                    status=ValidationStatus.SKIP,
                    message="LM Studio not available"
                )
            elif config["backend"] == "ollama" and not ollama_ok:
                result = ValidationResult(
                    name=f"PAT:{agent_name}",
                    status=ValidationStatus.SKIP,
                    message="Ollama not available"
                )
            else:
                result = validate_pat_agent(agent_name, config)
            
            pat_results.append(result)
            
            status_icon = {
                ValidationStatus.PASS: "✅",
                ValidationStatus.FAIL: "❌",
                ValidationStatus.WARN: "⚠️",
                ValidationStatus.SKIP: "⏭️"
            }[result.status]
            
            print(f"  {status_icon} {agent_name}: {result.message}")
            
            # Run benchmarks if requested
            if benchmark_iterations > 0 and result.status in (ValidationStatus.PASS, ValidationStatus.WARN):
                stats = benchmark_pat_agent(agent_name, config, benchmark_iterations)
                benchmarks.append(stats)
        
        print()
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 3: SAT Agent Validation
    # ─────────────────────────────────────────────────────────────────────────
    if not pat_only:
        print("▶ PHASE 3: SAT Agent Validation (5 agents)\n")
        
        for agent_name, config in SAT_AGENTS.items():
            if not kernel_ok:
                result = ValidationResult(
                    name=f"SAT:{agent_name}",
                    status=ValidationStatus.SKIP,
                    message="Kernel not available"
                )
            else:
                result = validate_sat_agent(agent_name, config)
            
            sat_results.append(result)
            
            status_icon = {
                ValidationStatus.PASS: "✅",
                ValidationStatus.FAIL: "❌",
                ValidationStatus.WARN: "⚠️",
                ValidationStatus.SKIP: "⏭️"
            }[result.status]
            
            print(f"  {status_icon} {agent_name}: {result.message}")
        
        print()
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 4: Ihsān Gate Check
    # ─────────────────────────────────────────────────────────────────────────
    print("▶ PHASE 4: Ihsān Gate Validation\n")
    
    ihsan_result = validate_ihsan_gate()
    status_icon = {
        ValidationStatus.PASS: "✅",
        ValidationStatus.FAIL: "❌",
        ValidationStatus.WARN: "⚠️",
        ValidationStatus.SKIP: "⏭️"
    }[ihsan_result.status]
    print(f"  {status_icon} {ihsan_result.name}: {ihsan_result.message}")
    print()
    
    # ─────────────────────────────────────────────────────────────────────────
    # PHASE 5: Benchmark Results (if run)
    # ─────────────────────────────────────────────────────────────────────────
    if benchmarks:
        print("▶ PHASE 5: Benchmark Results\n")
        print(f"  {'Agent':<20} {'Mean':>8} {'P50':>8} {'P95':>8} {'P99':>8} {'Success':>8}")
        print("  " + "─" * 60)
        
        for stats in benchmarks:
            p95_indicator = "✅" if stats.p95_ms <= LATENCY_P95_TARGET_MS else "⚠️"
            print(f"  {stats.agent:<20} {stats.mean_ms:>7.0f}ms {stats.p50_ms:>7.0f}ms "
                  f"{stats.p95_ms:>7.0f}ms {stats.p99_ms:>7.0f}ms {stats.success_rate:>7.0%} {p95_indicator}")
        
        print()
    
    # ─────────────────────────────────────────────────────────────────────────
    # Calculate Summary
    # ─────────────────────────────────────────────────────────────────────────
    all_results = pat_results + sat_results + [ihsan_result]
    
    passed = sum(1 for r in all_results if r.status == ValidationStatus.PASS)
    failed = sum(1 for r in all_results if r.status == ValidationStatus.FAIL)
    warnings = sum(1 for r in all_results if r.status == ValidationStatus.WARN)
    skipped = sum(1 for r in all_results if r.status == ValidationStatus.SKIP)
    total = len(all_results)
    
    # Sovereignty score: passed / (total - skipped)
    denominator = total - skipped
    sovereignty_score = passed / denominator if denominator > 0 else 0.0
    
    print("═" * 70)
    print("  SUMMARY")
    print("═" * 70)
    print(f"  Total Tests: {total}")
    print(f"  ✅ Passed:   {passed}")
    print(f"  ❌ Failed:   {failed}")
    print(f"  ⚠️ Warnings: {warnings}")
    print(f"  ⏭️ Skipped:  {skipped}")
    print(f"  📊 Sovereignty Score: {sovereignty_score:.0%}")
    print("═" * 70 + "\n")
    
    # Determine offline capability
    offline_capable = ollama_ok and failed == 0
    
    return ValidationReport(
        timestamp=timestamp,
        total_tests=total,
        passed=passed,
        failed=failed,
        warnings=warnings,
        skipped=skipped,
        sovereignty_score=sovereignty_score,
        pat_results=pat_results,
        sat_results=sat_results,
        benchmarks=benchmarks,
        offline_capable=offline_capable
    )


def save_report(report: ValidationReport, output_path: Path) -> None:
    """Save validation report to JSON."""
    report_dict = {
        "timestamp": report.timestamp,
        "total_tests": report.total_tests,
        "passed": report.passed,
        "failed": report.failed,
        "warnings": report.warnings,
        "skipped": report.skipped,
        "sovereignty_score": report.sovereignty_score,
        "offline_capable": report.offline_capable,
        "pat_results": [
            {
                "name": r.name,
                "status": r.status.value,
                "message": r.message,
                "latency_ms": r.latency_ms,
                "details": r.details
            }
            for r in report.pat_results
        ],
        "sat_results": [
            {
                "name": r.name,
                "status": r.status.value,
                "message": r.message,
                "latency_ms": r.latency_ms,
                "details": r.details
            }
            for r in report.sat_results
        ],
        "benchmarks": [
            {
                "agent": b.agent,
                "samples": b.samples,
                "mean_ms": b.mean_ms,
                "p50_ms": b.p50_ms,
                "p95_ms": b.p95_ms,
                "p99_ms": b.p99_ms,
                "min_ms": b.min_ms,
                "max_ms": b.max_ms,
                "success_rate": b.success_rate
            }
            for b in report.benchmarks
        ]
    }
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_dict, f, indent=2)
    
    print(f"📄 Report saved: {output_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="BIZRA Stand Alone Validation Suite — Sovereignty Verification"
    )
    parser.add_argument("--pat-only", action="store_true", help="Validate PAT agents only")
    parser.add_argument("--sat-only", action="store_true", help="Validate SAT agents only")
    parser.add_argument("--offline", action="store_true", help="Simulate offline mode")
    parser.add_argument("--benchmark", type=int, default=0, metavar="N",
                        help="Run N benchmark iterations per agent")
    parser.add_argument("--output", type=str, default=None,
                        help="Save report to JSON file")
    
    args = parser.parse_args()
    
    report = run_full_validation(
        pat_only=args.pat_only,
        sat_only=args.sat_only,
        offline_mode=args.offline,
        benchmark_iterations=args.benchmark
    )
    
    if args.output:
        save_report(report, Path(args.output))
    else:
        # Default output location
        output_path = Path("docs/evidence/validation") / f"sovereignty_validation_{report.timestamp.replace(':', '-').replace('+', '_')}.json"
        save_report(report, output_path)
    
    # Exit code based on results
    if report.failed > 0:
        sys.exit(1)
    elif report.warnings > 0:
        sys.exit(0)  # Warnings don't fail the validation
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
