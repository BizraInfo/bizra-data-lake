#!/usr/bin/env python3
"""
BIZRA Quality Radar - Real-Time Evidence-Based Evaluation

Generates a quality radar chart from ACTUAL system measurements:
- Test results (cargo test)
- Code quality (clippy warnings)
- SAPE probe scores (if server running)
- Receipt evidence (docs/evidence/)
- Ihsān constitution alignment

Part of BIZRA CI Integrity Gates
"""
import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

# Optional imports - graceful degradation
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False


# ============================================================================
# CONFIGURATION
# ============================================================================

WORKSPACE = Path(__file__).parent.parent
CONSTITUTION_PATH = WORKSPACE / "constitution" / "ihsan_v1.yaml"
EVIDENCE_PATH = WORKSPACE / "docs" / "evidence"
RECEIPTS_PATH = EVIDENCE_PATH / "receipts"
SERVER_URL = os.getenv("BIZRA_SERVER_URL", "http://127.0.0.1:8080")

# Ihsān dimension weights from constitution
IHSAN_WEIGHTS = {
    "correctness": 0.22,
    "safety": 0.22,
    "user_benefit": 0.14,
    "efficiency": 0.12,
    "auditability": 0.12,
    "anti_centralization": 0.08,
    "robustness": 0.06,
    "adl_fairness": 0.04,
}


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class MetricResult:
    """Result of a single metric measurement."""
    name: str
    score: float  # 0.0 - 10.0 scale
    raw_value: Any
    source: str  # "tests", "clippy", "sape", "evidence", "calculated"
    confidence: float = 1.0
    details: str = ""


@dataclass
class QualityReport:
    """Complete quality assessment report."""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    dimensions: dict = field(default_factory=dict)
    ihsan_score: float = 0.0
    overall_health: float = 0.0
    test_summary: dict = field(default_factory=dict)
    evidence_count: int = 0
    warnings: list = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "dimensions": {k: {"score": v.score, "source": v.source, "details": v.details} 
                          for k, v in self.dimensions.items()},
            "ihsan_score": self.ihsan_score,
            "overall_health": self.overall_health,
            "test_summary": self.test_summary,
            "evidence_count": self.evidence_count,
            "warnings": self.warnings,
        }


# ============================================================================
# METRIC COLLECTORS
# ============================================================================

def collect_test_metrics() -> tuple[MetricResult, dict]:
    """Run cargo test and extract pass/fail metrics."""
    print("🧪 Running cargo tests...")
    
    try:
        result = subprocess.run(
            ["cargo", "test", "--", "--test-threads=4"],
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )
        output = result.stdout + result.stderr
        
        # Parse test results
        # Pattern: "test result: ok. X passed; Y failed; Z ignored"
        match = re.search(
            r'test result: (\w+)\. (\d+) passed; (\d+) failed; (\d+) ignored',
            output
        )
        
        if match:
            status = match.group(1)
            passed = int(match.group(2))
            failed = int(match.group(3))
            ignored = int(match.group(4))
            total = passed + failed
            
            # Calculate score (10.0 = 100% pass rate)
            pass_rate = passed / max(total, 1)
            score = pass_rate * 10.0
            
            summary = {
                "passed": passed,
                "failed": failed,
                "ignored": ignored,
                "total": total,
                "pass_rate": pass_rate,
                "status": status,
            }
            
            # Count all test results in output
            all_passed = sum(1 for _ in re.finditer(r'test result: ok\.', output))
            all_failed = sum(1 for _ in re.finditer(r'test result: FAILED', output))
            
            # Aggregate totals from all test suites
            total_passed = sum(int(m.group(1)) for m in re.finditer(r'(\d+) passed;', output))
            total_failed = sum(int(m.group(1)) for m in re.finditer(r'(\d+) failed;', output))
            
            summary["total_passed"] = total_passed
            summary["total_failed"] = total_failed
            summary["suites_passed"] = all_passed
            summary["suites_failed"] = all_failed
            
            # Recalculate with totals
            if total_passed + total_failed > 0:
                score = (total_passed / (total_passed + total_failed)) * 10.0
                summary["pass_rate"] = total_passed / (total_passed + total_failed)
            
            return MetricResult(
                name="Test Coverage",
                score=score,
                raw_value=summary,
                source="cargo test",
                confidence=1.0,
                details=f"{total_passed}/{total_passed + total_failed} tests passed"
            ), summary
        else:
            return MetricResult(
                name="Test Coverage",
                score=0.0,
                raw_value={"error": "Could not parse test output"},
                source="cargo test",
                confidence=0.5,
                details="Failed to parse test results"
            ), {}
            
    except subprocess.TimeoutExpired:
        return MetricResult(
            name="Test Coverage",
            score=5.0,
            raw_value={"error": "timeout"},
            source="cargo test",
            confidence=0.3,
            details="Tests timed out after 10 minutes"
        ), {}
    except Exception as e:
        return MetricResult(
            name="Test Coverage",
            score=0.0,
            raw_value={"error": str(e)},
            source="cargo test",
            confidence=0.0,
            details=f"Error: {e}"
        ), {}


def collect_clippy_metrics() -> MetricResult:
    """Run clippy and count warnings/errors."""
    print("📎 Running clippy analysis...")
    
    try:
        result = subprocess.run(
            ["cargo", "clippy", "--all-targets", "--", "-D", "warnings"],
            cwd=WORKSPACE,
            capture_output=True,
            text=True,
            timeout=300
        )
        
        output = result.stdout + result.stderr
        
        # Count warnings and errors
        warnings = len(re.findall(r'warning:', output))
        errors = len(re.findall(r'error\[', output))
        
        # Score: 10.0 = no issues, deduct 0.5 per warning, 2.0 per error
        score = max(0.0, 10.0 - (warnings * 0.5) - (errors * 2.0))
        
        return MetricResult(
            name="Code Quality",
            score=score,
            raw_value={"warnings": warnings, "errors": errors},
            source="clippy",
            confidence=1.0,
            details=f"{warnings} warnings, {errors} errors"
        )
        
    except Exception as e:
        return MetricResult(
            name="Code Quality",
            score=5.0,
            raw_value={"error": str(e)},
            source="clippy",
            confidence=0.3,
            details=f"Clippy failed: {e}"
        )


def collect_sape_metrics() -> Optional[MetricResult]:
    """Query SAPE probe scores from running server."""
    if not HTTPX_AVAILABLE:
        return None
    
    print("🔬 Querying SAPE probes...")
    
    try:
        with httpx.Client(timeout=5.0) as client:
            # Check if server is running
            try:
                health = client.get(f"{SERVER_URL}/health/live")
                if health.status_code != 200:
                    return None
            except Exception:
                print("   ⚠️ Server not running, skipping SAPE metrics")
                return None
            
            # Get SAPE stats
            resp = client.get(f"{SERVER_URL}/sape/stats")
            if resp.status_code == 200:
                data = resp.json()
                
                # Calculate aggregate score from pattern activations
                total_patterns = data.get("total_patterns", 0)
                active_patterns = data.get("active_patterns", 0)
                snr_improvement = data.get("total_snr_improvement", 0)
                
                # Score based on SNR improvement and pattern activation
                base_score = 7.0
                if active_patterns > 0:
                    base_score += min(2.0, snr_improvement)
                if total_patterns >= 5:
                    base_score += 1.0
                
                return MetricResult(
                    name="SAPE Performance",
                    score=min(10.0, base_score),
                    raw_value=data,
                    source="sape/stats",
                    confidence=0.9,
                    details=f"{active_patterns}/{total_patterns} patterns active, SNR+{snr_improvement:.2f}"
                )
    except Exception as e:
        print(f"   ⚠️ SAPE query failed: {e}")
        return None


def collect_evidence_metrics() -> MetricResult:
    """Count and analyze evidence artifacts."""
    print("📁 Analyzing evidence artifacts...")
    
    evidence_count = 0
    receipt_count = 0
    
    try:
        if EVIDENCE_PATH.exists():
            for f in EVIDENCE_PATH.rglob("*"):
                if f.is_file():
                    evidence_count += 1
        
        if RECEIPTS_PATH.exists():
            for f in RECEIPTS_PATH.glob("*.json"):
                receipt_count += 1
        
        # Score: base 6.0, +1.0 per 10 evidence files, max 10.0
        score = min(10.0, 6.0 + (evidence_count / 10))
        
        return MetricResult(
            name="Auditability",
            score=score,
            raw_value={"evidence_files": evidence_count, "receipts": receipt_count},
            source="filesystem",
            confidence=1.0,
            details=f"{evidence_count} evidence files, {receipt_count} receipts"
        )
        
    except Exception as e:
        return MetricResult(
            name="Auditability",
            score=5.0,
            raw_value={"error": str(e)},
            source="filesystem",
            confidence=0.5,
            details=f"Error scanning: {e}"
        )


def collect_constitution_metrics() -> MetricResult:
    """Verify constitution file integrity."""
    print("📜 Checking constitution integrity...")
    
    try:
        if not CONSTITUTION_PATH.exists():
            return MetricResult(
                name="Ethics Framework",
                score=0.0,
                raw_value={"error": "constitution not found"},
                source="filesystem",
                confidence=1.0,
                details="constitution/ihsan_v1.yaml missing!"
            )
        
        import yaml
        content = CONSTITUTION_PATH.read_text(encoding="utf-8")
        data = yaml.safe_load(content)
        
        # Verify required fields
        required = ["dimensions", "threshold_policy", "units"]
        missing = [r for r in required if r not in data]
        
        if missing:
            score = 7.0 - len(missing)
        else:
            # Verify weights sum to 1.0
            dims = data.get("dimensions", {})
            weights = sum(d.get("weight", 0) for d in dims.values())
            
            if abs(weights - 1.0) < 0.001:
                score = 10.0
            else:
                score = 8.0
        
        return MetricResult(
            name="Ethics Framework",
            score=score,
            raw_value={"version": data.get("version", "unknown"), "id": data.get("id", "unknown")},
            source="constitution",
            confidence=1.0,
            details=f"Constitution v{data.get('version', '?')} loaded"
        )
        
    except Exception as e:
        return MetricResult(
            name="Ethics Framework",
            score=3.0,
            raw_value={"error": str(e)},
            source="constitution",
            confidence=0.5,
            details=f"Error loading: {e}"
        )


def collect_security_metrics() -> MetricResult:
    """Check for security issues."""
    print("🔒 Checking security posture...")
    
    issues = []
    
    # Check for .env files in repo
    env_files = list(WORKSPACE.glob("**/.env"))
    env_files = [f for f in env_files if ".git" not in str(f)]
    if env_files:
        issues.append(f"{len(env_files)} .env files found")
    
    # Check for hardcoded secrets patterns
    secret_patterns = [
        r'password\s*=\s*["\'][^"\']+["\']',
        r'api_key\s*=\s*["\'][^"\']+["\']',
        r'secret\s*=\s*["\'][^"\']+["\']',
    ]
    
    # Scan key config files
    for config_file in WORKSPACE.glob("**/*.yaml"):
        if ".git" in str(config_file) or "target" in str(config_file):
            continue
        try:
            content = config_file.read_text(encoding="utf-8", errors="ignore")
            for pattern in secret_patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    issues.append(f"Potential secret in {config_file.name}")
                    break
        except Exception:
            pass
    
    # Score: 10.0 = no issues, -2.0 per issue
    score = max(0.0, 10.0 - len(issues) * 2.0)
    
    return MetricResult(
        name="Security Posture",
        score=score,
        raw_value={"issues": issues},
        source="security scan",
        confidence=0.8,
        details=f"{len(issues)} potential issues" if issues else "No issues found"
    )


def collect_architecture_metrics() -> MetricResult:
    """Analyze architecture coherence from module structure."""
    print("🏗️ Analyzing architecture...")
    
    try:
        src_path = WORKSPACE / "src"
        core_path = WORKSPACE / "core"
        
        # Count Rust modules
        rust_modules = list(src_path.glob("*.rs")) if src_path.exists() else []
        
        # Count Python modules  
        python_modules = list(core_path.glob("*.py")) if core_path.exists() else []
        
        # Check for key architectural files
        key_files = [
            "src/lib.rs",
            "src/bridge.rs",
            "src/ihsan.rs",
            "src/sape.rs",
            "src/fate.rs",
            "core/sape.py",
            "core/fate.py",
        ]
        
        present = sum(1 for f in key_files if (WORKSPACE / f).exists())
        
        # Score based on architectural completeness
        score = 6.0 + (present / len(key_files)) * 4.0
        
        return MetricResult(
            name="Architecture",
            score=score,
            raw_value={
                "rust_modules": len(rust_modules),
                "python_modules": len(python_modules),
                "key_files_present": present,
                "key_files_expected": len(key_files),
            },
            source="filesystem",
            confidence=0.9,
            details=f"{len(rust_modules)} Rust + {len(python_modules)} Python modules"
        )
        
    except Exception as e:
        return MetricResult(
            name="Architecture",
            score=5.0,
            raw_value={"error": str(e)},
            source="filesystem",
            confidence=0.3,
            details=f"Error: {e}"
        )


def collect_documentation_metrics() -> MetricResult:
    """Analyze documentation coverage."""
    print("📚 Checking documentation...")
    
    try:
        docs_path = WORKSPACE / "docs"
        
        # Count markdown files
        md_files = list(WORKSPACE.glob("*.md"))
        md_files += list(docs_path.rglob("*.md")) if docs_path.exists() else []
        
        # Check for key docs
        key_docs = ["README.md", "ARCHITECTURE.md", "docs/openapi.yaml"]
        present = sum(1 for f in key_docs if (WORKSPACE / f).exists())
        
        # Score based on documentation presence
        base_score = 6.0
        base_score += min(2.0, len(md_files) / 10)  # +1 per 5 docs, max 2
        base_score += (present / len(key_docs)) * 2.0  # Up to 2 for key docs
        
        return MetricResult(
            name="Documentation",
            score=min(10.0, base_score),
            raw_value={
                "markdown_files": len(md_files),
                "key_docs_present": present,
            },
            source="filesystem",
            confidence=1.0,
            details=f"{len(md_files)} markdown files"
        )
        
    except Exception as e:
        return MetricResult(
            name="Documentation",
            score=5.0,
            raw_value={"error": str(e)},
            source="filesystem",
            confidence=0.3,
            details=f"Error: {e}"
        )


# ============================================================================
# IHSĀN SCORE CALCULATION
# ============================================================================

def calculate_ihsan_score(dimensions: dict[str, MetricResult]) -> float:
    """Calculate weighted Ihsān score from dimension metrics."""
    
    # Map our dimension names to Ihsān dimensions
    dimension_mapping = {
        "Test Coverage": "correctness",
        "Code Quality": "robustness", 
        "Security Posture": "safety",
        "Ethics Framework": "adl_fairness",
        "Auditability": "auditability",
        "Architecture": "efficiency",
        "Documentation": "user_benefit",
        "SAPE Performance": "anti_centralization",
    }
    
    # Normalize scores to 0-1 range and apply weights
    weighted_sum = 0.0
    total_weight = 0.0
    
    for dim_name, metric in dimensions.items():
        ihsan_dim = dimension_mapping.get(dim_name)
        if ihsan_dim and ihsan_dim in IHSAN_WEIGHTS:
            weight = IHSAN_WEIGHTS[ihsan_dim]
            normalized = metric.score / 10.0  # Convert 0-10 to 0-1
            weighted_sum += normalized * weight
            total_weight += weight
    
    # Fill missing dimensions with neutral 0.7
    for ihsan_dim, weight in IHSAN_WEIGHTS.items():
        if ihsan_dim not in [dimension_mapping.get(d) for d in dimensions]:
            weighted_sum += 0.7 * weight
            total_weight += weight
    
    return weighted_sum / total_weight if total_weight > 0 else 0.0


# ============================================================================
# REPORT GENERATION
# ============================================================================

def generate_report(skip_tests: bool = False) -> QualityReport:
    """Collect all metrics and generate quality report."""
    print("\n" + "=" * 60)
    print("🎯 BIZRA Quality Radar - Evidence-Based Evaluation")
    print("=" * 60 + "\n")
    
    report = QualityReport()
    
    # Collect metrics
    if skip_tests:
        print("⏭️ Skipping tests (--skip-tests flag)")
        test_metric = MetricResult(
            name="Test Coverage",
            score=8.0,
            raw_value={"skipped": True},
            source="skipped",
            confidence=0.5,
            details="Tests skipped"
        )
        report.test_summary = {"skipped": True}
    else:
        test_metric, test_summary = collect_test_metrics()
        report.test_summary = test_summary
    
    report.dimensions["Test Coverage"] = test_metric
    report.dimensions["Code Quality"] = collect_clippy_metrics()
    report.dimensions["Security Posture"] = collect_security_metrics()
    report.dimensions["Ethics Framework"] = collect_constitution_metrics()
    report.dimensions["Auditability"] = collect_evidence_metrics()
    report.dimensions["Architecture"] = collect_architecture_metrics()
    report.dimensions["Documentation"] = collect_documentation_metrics()
    
    # Try SAPE if server running
    sape_metric = collect_sape_metrics()
    if sape_metric:
        report.dimensions["SAPE Performance"] = sape_metric
    
    # Calculate aggregate scores
    report.ihsan_score = calculate_ihsan_score(report.dimensions)
    
    # Overall health = average of all dimension scores
    all_scores = [m.score for m in report.dimensions.values()]
    report.overall_health = sum(all_scores) / len(all_scores) if all_scores else 0.0
    
    # Evidence count
    report.evidence_count = report.dimensions.get("Auditability", MetricResult("", 0, {}, "")).raw_value.get("evidence_files", 0)
    
    return report


def print_report(report: QualityReport) -> None:
    """Print report to console."""
    print("\n" + "=" * 60)
    print("📊 QUALITY ASSESSMENT RESULTS")
    print("=" * 60)
    
    print(f"\n🎯 Overall Health: {report.overall_health:.1f}/10.0")
    print(f"⚖️  Ihsān Score: {report.ihsan_score:.4f} (target: 0.95)")
    
    print("\n📈 Dimension Scores:")
    print("-" * 50)
    
    for name, metric in sorted(report.dimensions.items(), key=lambda x: -x[1].score):
        bar = "█" * int(metric.score) + "░" * (10 - int(metric.score))
        status = "✅" if metric.score >= 8.0 else "⚠️" if metric.score >= 6.0 else "❌"
        print(f"  {status} {name:20} {bar} {metric.score:.1f}/10")
        print(f"     └─ {metric.details} ({metric.source})")
    
    if report.test_summary and not report.test_summary.get("skipped"):
        print(f"\n🧪 Test Summary:")
        print(f"   Passed: {report.test_summary.get('total_passed', 0)}")
        print(f"   Failed: {report.test_summary.get('total_failed', 0)}")
        print(f"   Pass Rate: {report.test_summary.get('pass_rate', 0)*100:.1f}%")
    
    print("\n" + "=" * 60)


def generate_radar_chart(report: QualityReport, output_path: Path) -> bool:
    """Generate radar chart visualization."""
    if not PLOTLY_AVAILABLE:
        print("⚠️ Plotly not installed, skipping chart generation")
        print("   Install with: pip install plotly kaleido")
        return False
    
    print("\n📊 Generating radar chart...")
    
    # Prepare data
    names = list(report.dimensions.keys())
    scores = [report.dimensions[n].score for n in names]
    
    # Abbreviate names for chart
    abbrev = {
        "Test Coverage": "Tests",
        "Code Quality": "Code",
        "Security Posture": "Security",
        "Ethics Framework": "Ethics",
        "Auditability": "Audit",
        "Architecture": "Arch",
        "Documentation": "Docs",
        "SAPE Performance": "SAPE",
    }
    names_short = [abbrev.get(n, n[:6]) for n in names]
    
    # Close the radar loop
    names_closed = names_short + [names_short[0]]
    scores_closed = scores + [scores[0]]
    
    # Colors
    color_exemplary = "#1FB8CD"
    color_excellent = "#5D878F"
    color_good = "#D2BA4C"
    color_needs = "#DB4545"
    
    fig = go.Figure()
    
    # Achievement bands
    fig.add_trace(go.Scatterpolar(
        r=[10] * len(names_closed),
        theta=names_closed,
        mode='lines',
        fill='toself',
        line=dict(color='rgba(0,0,0,0)'),
        fillcolor='rgba(31,184,205,0.08)',
        name='Exemplary (8.5+)',
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[8.5] * len(names_closed),
        theta=names_closed,
        mode='lines',
        fill='toself',
        line=dict(color='rgba(0,0,0,0)'),
        fillcolor='rgba(93,135,143,0.10)',
        name='Excellent (7.5+)',
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[7.5] * len(names_closed),
        theta=names_closed,
        mode='lines',
        fill='toself',
        line=dict(color='rgba(0,0,0,0)'),
        fillcolor='rgba(210,186,76,0.12)',
        name='Good (6.5+)',
        hoverinfo='skip'
    ))
    
    fig.add_trace(go.Scatterpolar(
        r=[6.5] * len(names_closed),
        theta=names_closed,
        mode='lines',
        fill='toself',
        line=dict(color='rgba(0,0,0,0)'),
        fillcolor='rgba(219,69,69,0.08)',
        name='Needs Attention',
        hoverinfo='skip'
    ))
    
    # Target threshold at 8.0
    fig.add_trace(go.Scatterpolar(
        r=[8.0] * len(names_closed),
        theta=names_closed,
        mode='lines',
        line=dict(color=color_excellent, width=2, dash='dash'),
        name='Target (8.0)',
        hovertemplate='Target: 8.00<extra></extra>'
    ))
    
    # Actual scores
    fig.add_trace(go.Scatterpolar(
        r=scores_closed,
        theta=names_closed,
        mode='lines+markers',
        line=dict(color=color_exemplary, width=3),
        marker=dict(size=9, color=color_exemplary),
        name='Actual Score',
        hovertemplate='%{theta}: %{r:.1f}/10<extra></extra>'
    ))
    
    # Highlight top 3 and bottom 3
    sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    
    # Top 3 strengths
    top3_r = [scores[i] for i in sorted_idx[:3]]
    top3_theta = [names_short[i] for i in sorted_idx[:3]]
    fig.add_trace(go.Scatterpolar(
        r=top3_r,
        theta=top3_theta,
        mode='markers',
        marker=dict(size=13, color='#2E8B57', symbol='star'),
        name='Top Strengths',
        hovertemplate='Strength %{theta}: %{r:.1f}<extra></extra>'
    ))
    
    # Bottom 3 (improvement areas)
    if len(sorted_idx) > 3:
        bot3_r = [scores[i] for i in sorted_idx[-3:]]
        bot3_theta = [names_short[i] for i in sorted_idx[-3:]]
        fig.add_trace(go.Scatterpolar(
            r=bot3_r,
            theta=bot3_theta,
            mode='markers',
            marker=dict(size=11, color=color_needs, symbol='triangle-up'),
            name='Improve Areas',
            hovertemplate='Improve %{theta}: %{r:.1f}<extra></extra>'
        ))
    
    # Layout
    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 10],
                tickvals=[6.0, 7.0, 8.0, 9.0, 10.0],
                ticktext=['6.0', '7.0', '8.0', '9.0', '10.0']
            ),
            angularaxis=dict(direction='clockwise')
        ),
        showlegend=True,
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.05,
            xanchor='center',
            x=0.5
        )
    )
    
    # Title with real metrics
    ihsan_pct = report.ihsan_score * 100
    health_pct = report.overall_health * 10
    main_title = f"BIZRA Quality Radar ({datetime.now().strftime('%Y-%m-%d %H:%M')})"
    subtitle = f"<span style='font-size: 16px;'>Overall: {report.overall_health:.1f}/10 | Ihsān: {report.ihsan_score:.3f}</span>"
    fig.update_layout(title={"text": main_title + "<br>" + subtitle})
    
    fig.update_traces(cliponaxis=False)
    
    # Save
    try:
        fig.write_image(str(output_path.with_suffix('.png')))
        fig.write_image(str(output_path.with_suffix('.svg')), format='svg')
        fig.write_html(str(output_path.with_suffix('.html')))
        print(f"   ✅ Saved: {output_path.with_suffix('.png')}")
        print(f"   ✅ Saved: {output_path.with_suffix('.svg')}")
        print(f"   ✅ Saved: {output_path.with_suffix('.html')}")
        return True
    except Exception as e:
        print(f"   ⚠️ Chart save failed: {e}")
        print("   Install kaleido for image export: pip install kaleido")
        # At least save HTML
        try:
            fig.write_html(str(output_path.with_suffix('.html')))
            print(f"   ✅ Saved HTML: {output_path.with_suffix('.html')}")
            return True
        except Exception:
            return False


# ============================================================================
# MAIN
# ============================================================================

def main():
    import argparse
    
    parser = argparse.ArgumentParser(
        description="BIZRA Quality Radar - Evidence-Based Evaluation"
    )
    parser.add_argument(
        "--output", "-o",
        default="quality_radar",
        help="Output file base name (default: quality_radar)"
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Skip running cargo tests (faster)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output JSON report"
    )
    parser.add_argument(
        "--ci",
        action="store_true",
        help="CI mode: exit non-zero if Ihsān < threshold"
    )
    
    args = parser.parse_args()
    
    # Generate report
    report = generate_report(skip_tests=args.skip_tests)
    
    # Print to console
    print_report(report)
    
    # Save JSON if requested
    if args.json:
        json_path = Path(args.output).with_suffix('.json')
        json_path.write_text(json.dumps(report.to_dict(), indent=2))
        print(f"\n📄 JSON report: {json_path}")
    
    # Generate chart
    output_path = Path(args.output)
    generate_radar_chart(report, output_path)
    
    # CI mode exit code
    if args.ci:
        # Get threshold from environment
        env = os.getenv("BIZRA_ENV", "development")
        thresholds = {"development": 0.80, "ci": 0.90, "production": 0.95}
        threshold = thresholds.get(env, 0.80)
        
        if report.ihsan_score < threshold:
            print(f"\n❌ CI FAILED: Ihsān {report.ihsan_score:.4f} < {threshold}")
            sys.exit(1)
        else:
            print(f"\n✅ CI PASSED: Ihsān {report.ihsan_score:.4f} >= {threshold}")
            sys.exit(0)
    
    sys.exit(0)


if __name__ == "__main__":
    main()
