# BIZRA End-to-End System Validation v1.0
# Comprehensive validation suite for all BIZRA components
# Part of SAPE Implementation Blueprint

import sys
import numpy as np

# Monkeypatch for libraries using deprecated np.object
if not hasattr(np, "object"):
    np.object = object
import json
import time
import hashlib
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import List, Dict, Tuple, Optional
from enum import Enum

# BIZRA paths
BIZRA_ROOT = Path("C:/BIZRA-DATA-LAKE")
sys.path.insert(0, str(BIZRA_ROOT))


class ValidationStatus(Enum):
    """Validation result status"""

    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"
    SKIP = "SKIP"


@dataclass
class ValidationResult:
    """Single validation check result"""

    name: str
    status: str
    message: str
    duration_ms: float
    details: Dict = None


@dataclass
class ValidationReport:
    """Complete validation report"""

    timestamp: str
    total_checks: int
    passed: int
    failed: int
    warnings: int
    skipped: int
    overall_status: str
    ihsan_compliant: bool
    results: List[ValidationResult]
    duration_seconds: float


class SystemValidator:
    """Comprehensive BIZRA system validator"""

    def __init__(self):
        self.results: List[ValidationResult] = []
        self.start_time = None

    def _record(
        self,
        name: str,
        status: ValidationStatus,
        message: str,
        duration_ms: float,
        details: Dict = None,
    ):
        """Record validation result"""
        result = ValidationResult(
            name=name,
            status=status.value,
            message=message,
            duration_ms=duration_ms,
            details=details,
        )
        self.results.append(result)

        # Print result
        symbol = {
            ValidationStatus.PASS: "✅",
            ValidationStatus.FAIL: "❌",
            ValidationStatus.WARN: "⚠️",
            ValidationStatus.SKIP: "⏭️",
        }[status]
        print(f"  {symbol} {name}: {message}")

    # ===== Directory Structure Validation =====

    def validate_directory_structure(self):
        """Validate BIZRA directory structure exists"""
        start = time.time()
        required_dirs = [
            "00_INTAKE",
            "01_RAW",
            "02_PROCESSED",
            "03_INDEXED",
            "04_GOLD",
            "99_QUARANTINE",
        ]

        missing = []
        for dir_name in required_dirs:
            dir_path = BIZRA_ROOT / dir_name
            if not dir_path.exists():
                missing.append(dir_name)

        duration = (time.time() - start) * 1000

        if not missing:
            self._record(
                "Directory Structure",
                ValidationStatus.PASS,
                f"All {len(required_dirs)} directories present",
                duration,
            )
        else:
            self._record(
                "Directory Structure",
                ValidationStatus.FAIL,
                f"Missing directories: {', '.join(missing)}",
                duration,
                {"missing": missing},
            )

    # ===== Configuration Validation =====

    def validate_config(self):
        """Validate bizra_config.py"""
        start = time.time()

        config_path = BIZRA_ROOT / "bizra_config.py"
        if not config_path.exists():
            self._record(
                "Configuration File",
                ValidationStatus.FAIL,
                "bizra_config.py not found",
                (time.time() - start) * 1000,
            )
            return

        try:
            import bizra_config as config

            # Check required constants
            required = [
                "BATCH_SIZE",
                "MAX_SEQ_LENGTH",
                "SNR_THRESHOLD",
                "IHSAN_CONSTRAINT",
            ]
            missing = [r for r in required if not hasattr(config, r)]

            duration = (time.time() - start) * 1000

            if not missing:
                ihsan = getattr(config, "IHSAN_CONSTRAINT", 0)
                self._record(
                    "Configuration File",
                    ValidationStatus.PASS,
                    f"Valid config (IHSAN_CONSTRAINT={ihsan})",
                    duration,
                    {"ihsan_constraint": ihsan},
                )
            else:
                self._record(
                    "Configuration File",
                    ValidationStatus.WARN,
                    f"Missing constants: {', '.join(missing)}",
                    duration,
                    {"missing": missing},
                )
        except Exception as e:
            self._record(
                "Configuration File",
                ValidationStatus.FAIL,
                f"Config import error: {str(e)[:50]}",
                (time.time() - start) * 1000,
            )

    # ===== Corpus Validation =====

    def validate_corpus(self):
        """Validate corpus (documents.parquet)"""
        start = time.time()

        corpus_path = BIZRA_ROOT / "04_GOLD" / "documents.parquet"
        if not corpus_path.exists():
            self._record(
                "Corpus Table",
                ValidationStatus.FAIL,
                "documents.parquet not found",
                (time.time() - start) * 1000,
            )
            return

        try:
            import pandas as pd

            df = pd.read_parquet(corpus_path)

            duration = (time.time() - start) * 1000

            if len(df) > 0:
                self._record(
                    "Corpus Table",
                    ValidationStatus.PASS,
                    f"{len(df):,} documents indexed",
                    duration,
                    {"document_count": len(df), "columns": list(df.columns)},
                )
            else:
                self._record(
                    "Corpus Table",
                    ValidationStatus.WARN,
                    "Corpus table is empty",
                    duration,
                )
        except Exception as e:
            self._record(
                "Corpus Table",
                ValidationStatus.FAIL,
                f"Read error: {str(e)[:50]}",
                (time.time() - start) * 1000,
            )

    # ===== Vector Embeddings Validation =====

    def validate_embeddings(self):
        """Validate vector embeddings"""
        start = time.time()

        chunks_path = BIZRA_ROOT / "04_GOLD" / "chunks.parquet"
        if not chunks_path.exists():
            self._record(
                "Vector Embeddings",
                ValidationStatus.FAIL,
                "chunks.parquet not found",
                (time.time() - start) * 1000,
            )
            return

        try:
            import pandas as pd

            df = pd.read_parquet(chunks_path)

            duration = (time.time() - start) * 1000

            if len(df) > 0:
                # Check for embedding column
                has_embeddings = "embedding" in df.columns or "embeddings" in df.columns
                self._record(
                    "Vector Embeddings",
                    ValidationStatus.PASS if has_embeddings else ValidationStatus.WARN,
                    f"{len(df):,} chunks"
                    + (
                        " with embeddings"
                        if has_embeddings
                        else " (no embedding column)"
                    ),
                    duration,
                    {"chunk_count": len(df), "has_embeddings": has_embeddings},
                )
            else:
                self._record(
                    "Vector Embeddings",
                    ValidationStatus.WARN,
                    "Chunks table is empty",
                    duration,
                )
        except Exception as e:
            self._record(
                "Vector Embeddings",
                ValidationStatus.FAIL,
                f"Read error: {str(e)[:50]}",
                (time.time() - start) * 1000,
            )

    # ===== Hypergraph Validation =====

    def validate_hypergraph(self):
        """Validate hypergraph structure"""
        start = time.time()

        stats_path = BIZRA_ROOT / "03_INDEXED" / "graph" / "statistics.json"
        if not stats_path.exists():
            self._record(
                "Hypergraph Structure",
                ValidationStatus.FAIL,
                "graph/statistics.json not found",
                (time.time() - start) * 1000,
            )
            return

        try:
            with open(stats_path) as f:
                stats = json.load(f)

            nodes = stats.get("total_nodes", 0)
            edges = stats.get("total_edges", 0)

            duration = (time.time() - start) * 1000

            if nodes > 0 and edges > 0:
                self._record(
                    "Hypergraph Structure",
                    ValidationStatus.PASS,
                    f"{nodes:,} nodes, {edges:,} edges",
                    duration,
                    stats,
                )
            else:
                self._record(
                    "Hypergraph Structure",
                    ValidationStatus.WARN,
                    f"Graph sparse: {nodes} nodes, {edges} edges",
                    duration,
                    stats,
                )
        except Exception as e:
            self._record(
                "Hypergraph Structure",
                ValidationStatus.FAIL,
                f"Read error: {str(e)[:50]}",
                (time.time() - start) * 1000,
            )

    # ===== POI Ledger Validation =====

    def validate_poi_ledger(self):
        """Validate Proof-of-Impact ledger"""
        start = time.time()

        ledger_path = BIZRA_ROOT / "04_GOLD" / "poi_ledger.jsonl"
        if not ledger_path.exists():
            self._record(
                "POI Ledger",
                ValidationStatus.SKIP,
                "poi_ledger.jsonl not found (optional)",
                (time.time() - start) * 1000,
            )
            return

        try:
            entries = []
            valid_hashes = 0
            with open(ledger_path) as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line)
                        entries.append(entry)
                        # Validate hash format
                        if entry.get("attestation_hash", "").startswith(
                            ("sha256:", "ed25519:")
                        ):
                            valid_hashes += 1

            duration = (time.time() - start) * 1000

            if entries:
                integrity = valid_hashes / len(entries) if entries else 0
                self._record(
                    "POI Ledger",
                    ValidationStatus.PASS
                    if integrity >= 0.95
                    else ValidationStatus.WARN,
                    f"{len(entries)} entries, {integrity * 100:.0f}% hash integrity",
                    duration,
                    {"entry_count": len(entries), "valid_hashes": valid_hashes},
                )
            else:
                self._record(
                    "POI Ledger", ValidationStatus.WARN, "Ledger is empty", duration
                )
        except Exception as e:
            self._record(
                "POI Ledger",
                ValidationStatus.FAIL,
                f"Read error: {str(e)[:50]}",
                (time.time() - start) * 1000,
            )

    # ===== DDAGI Consciousness Validation =====

    def validate_ddagi(self):
        """Validate DDAGI consciousness log"""
        start = time.time()

        ddagi_path = BIZRA_ROOT / "03_INDEXED" / "ddagi_consciousness.jsonl"
        if not ddagi_path.exists():
            self._record(
                "DDAGI Consciousness",
                ValidationStatus.SKIP,
                "ddagi_consciousness.jsonl not found (optional)",
                (time.time() - start) * 1000,
            )
            return

        try:
            events = []
            with open(ddagi_path) as f:
                for line in f:
                    if line.strip():
                        events.append(json.loads(line))

            duration = (time.time() - start) * 1000

            if events:
                avg_snr = sum(e.get("snr", 0) for e in events) / len(events)
                self._record(
                    "DDAGI Consciousness",
                    ValidationStatus.PASS,
                    f"{len(events)} events, avg SNR={avg_snr:.2f}",
                    duration,
                    {"event_count": len(events), "avg_snr": avg_snr},
                )
            else:
                self._record(
                    "DDAGI Consciousness",
                    ValidationStatus.WARN,
                    "No consciousness events recorded",
                    duration,
                )
        except Exception as e:
            self._record(
                "DDAGI Consciousness",
                ValidationStatus.FAIL,
                f"Read error: {str(e)[:50]}",
                (time.time() - start) * 1000,
            )

    # ===== Core Module Import Validation =====

    def validate_core_imports(self):
        """Validate core Python modules can be imported"""
        start = time.time()

        modules = [
            ("bizra_config", "Configuration"),
            ("corpus_manager", "Corpus Manager"),
            ("vector_engine", "Vector Engine"),
            ("arte_engine", "ARTE Engine"),
            ("hypergraph_engine", "Hypergraph Engine"),
            ("pat_engine", "PAT Engine"),
        ]

        importable = []
        failed = []

        for module_name, display_name in modules:
            try:
                __import__(module_name)
                importable.append(display_name)
            except Exception as e:
                failed.append((display_name, str(e)[:30]))

        duration = (time.time() - start) * 1000

        if not failed:
            self._record(
                "Core Module Imports",
                ValidationStatus.PASS,
                f"All {len(modules)} modules importable",
                duration,
                {"modules": importable},
            )
        elif len(failed) < len(modules) / 2:
            self._record(
                "Core Module Imports",
                ValidationStatus.WARN,
                f"{len(importable)}/{len(modules)} modules OK",
                duration,
                {"importable": importable, "failed": failed},
            )
        else:
            self._record(
                "Core Module Imports",
                ValidationStatus.FAIL,
                f"Only {len(importable)}/{len(modules)} modules importable",
                duration,
                {"importable": importable, "failed": failed},
            )

    # ===== SNR Engine Validation =====

    def validate_snr_engine(self):
        """Validate SNR calculation engine"""
        start = time.time()

        try:
            from arte_engine import SNREngine
            import numpy as np

            engine = SNREngine()

            # Test SNR calculation with synthetic data
            query_embedding = np.random.rand(384).astype(np.float32)
            context_embeddings = [
                np.random.rand(384).astype(np.float32) for _ in range(5)
            ]
            symbolic_facts = ["fact1", "fact2", "fact3"]
            neural_results = [{"text": "result1", "score": 0.9}]

            result = engine.calculate_snr(
                query_embedding=query_embedding,
                context_embeddings=context_embeddings,
                symbolic_facts=symbolic_facts,
                neural_results=neural_results,
            )

            duration = (time.time() - start) * 1000

            snr = result.get("snr", 0)
            if 0 <= snr <= 1:
                self._record(
                    "SNR Engine",
                    ValidationStatus.PASS,
                    f"Operational (test SNR={snr:.4f})",
                    duration,
                    {"test_snr": snr, "components": result.get("components", {})},
                )
            else:
                self._record(
                    "SNR Engine",
                    ValidationStatus.WARN,
                    f"SNR out of range: {snr}",
                    duration,
                )
        except ImportError:
            self._record(
                "SNR Engine",
                ValidationStatus.SKIP,
                "arte_engine not available",
                (time.time() - start) * 1000,
            )
        except Exception as e:
            self._record(
                "SNR Engine",
                ValidationStatus.FAIL,
                f"Calculation error: {str(e)[:50]}",
                (time.time() - start) * 1000,
            )

    # ===== Self-Healing Agent Validation =====

    def validate_self_healing(self):
        """Validate self-healing agent"""
        start = time.time()

        healing_path = BIZRA_ROOT / "self_healing.py"
        if not healing_path.exists():
            self._record(
                "Self-Healing Agent",
                ValidationStatus.SKIP,
                "self_healing.py not found",
                (time.time() - start) * 1000,
            )
            return

        try:
            from self_healing import HealingAgent

            agent = HealingAgent()
            critical_files = agent.critical_files

            duration = (time.time() - start) * 1000

            self._record(
                "Self-Healing Agent",
                ValidationStatus.PASS,
                f"Monitoring {len(critical_files)} critical files",
                duration,
                {"critical_files": [str(f) for f in critical_files]},
            )
        except Exception as e:
            self._record(
                "Self-Healing Agent",
                ValidationStatus.FAIL,
                f"Agent error: {str(e)[:50]}",
                (time.time() - start) * 1000,
            )

    # ===== Metrics Dashboard Validation =====

    def validate_metrics_dashboard(self):
        """Validate metrics dashboard"""
        start = time.time()

        dashboard_path = BIZRA_ROOT / "metrics_dashboard.py"
        if not dashboard_path.exists():
            self._record(
                "Metrics Dashboard",
                ValidationStatus.SKIP,
                "metrics_dashboard.py not found",
                (time.time() - start) * 1000,
            )
            return

        try:
            from metrics_dashboard import MetricsDashboard

            dashboard = MetricsDashboard()
            health = dashboard.get_system_health()

            duration = (time.time() - start) * 1000

            self._record(
                "Metrics Dashboard",
                ValidationStatus.PASS,
                f"Operational (status={health.status})",
                duration,
                asdict(health),
            )
        except Exception as e:
            self._record(
                "Metrics Dashboard",
                ValidationStatus.FAIL,
                f"Dashboard error: {str(e)[:50]}",
                (time.time() - start) * 1000,
            )

    # ===== Run All Validations =====

    def run_all(self) -> ValidationReport:
        """Run all validation checks"""
        self.start_time = time.time()
        self.results = []

        print("\n" + "=" * 60)
        print("        BIZRA END-TO-END SYSTEM VALIDATION")
        print("=" * 60)
        print(f"\n  Started: {datetime.now().isoformat()}")
        print(f"  Root: {BIZRA_ROOT}\n")

        print("  📁 STRUCTURE VALIDATION")
        print("  " + "-" * 40)
        self.validate_directory_structure()
        self.validate_config()

        print("\n  📊 DATA VALIDATION")
        print("  " + "-" * 40)
        self.validate_corpus()
        self.validate_embeddings()
        self.validate_hypergraph()

        print("\n  🔐 INTEGRITY VALIDATION")
        print("  " + "-" * 40)
        self.validate_poi_ledger()
        self.validate_ddagi()

        print("\n  ⚙️  ENGINE VALIDATION")
        print("  " + "-" * 40)
        self.validate_core_imports()
        self.validate_snr_engine()
        self.validate_self_healing()
        self.validate_metrics_dashboard()

        # Generate report
        total_duration = time.time() - self.start_time
        passed = sum(1 for r in self.results if r.status == ValidationStatus.PASS.value)
        failed = sum(1 for r in self.results if r.status == ValidationStatus.FAIL.value)
        warnings = sum(
            1 for r in self.results if r.status == ValidationStatus.WARN.value
        )
        skipped = sum(
            1 for r in self.results if r.status == ValidationStatus.SKIP.value
        )

        # Determine overall status
        if failed > 0:
            overall = "FAILED"
        elif warnings > 2:
            overall = "DEGRADED"
        else:
            overall = "HEALTHY"

        # Ihsan compliance check (>= 95% pass rate)
        pass_rate = passed / len(self.results) if self.results else 0
        ihsan_compliant = pass_rate >= 0.95

        report = ValidationReport(
            timestamp=datetime.now().isoformat(),
            total_checks=len(self.results),
            passed=passed,
            failed=failed,
            warnings=warnings,
            skipped=skipped,
            overall_status=overall,
            ihsan_compliant=ihsan_compliant,
            results=[asdict(r) for r in self.results],
            duration_seconds=total_duration,
        )

        # Print summary
        print("\n" + "=" * 60)
        print("                    SUMMARY")
        print("=" * 60)

        status_symbol = {"HEALTHY": "✅", "DEGRADED": "⚠️", "FAILED": "❌"}[overall]
        ihsan_symbol = "✅" if ihsan_compliant else "❌"

        print(f"\n  Overall Status:    {status_symbol} {overall}")
        print(f"  Ihsān Compliant:   {ihsan_symbol} {pass_rate * 100:.1f}% pass rate")
        print(f"\n  Total Checks:      {len(self.results)}")
        print(f"  ✅ Passed:         {passed}")
        print(f"  ❌ Failed:         {failed}")
        print(f"  ⚠️  Warnings:       {warnings}")
        print(f"  ⏭️  Skipped:        {skipped}")
        print(f"\n  Duration:          {total_duration:.2f}s")

        print("\n" + "=" * 60)

        return report

    def save_report(self, report: ValidationReport):
        """Save validation report to file"""
        report_dir = BIZRA_ROOT / "03_INDEXED" / "validation"
        report_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON report
        report_path = (
            report_dir / f"validation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, "w") as f:
            json.dump(asdict(report), f, indent=2)

        # Save latest symlink/copy
        latest_path = report_dir / "latest_validation.json"
        with open(latest_path, "w") as f:
            json.dump(asdict(report), f, indent=2)

        print(f"\n📄 Report saved: {report_path}")
        return report_path


# Main execution
if __name__ == "__main__":
    print("🔍 BIZRA System Validator v1.0")

    validator = SystemValidator()
    report = validator.run_all()

    # Save report
    report_path = validator.save_report(report)

    # Exit code based on status
    if report.overall_status == "FAILED":
        sys.exit(1)
    elif report.overall_status == "DEGRADED":
        sys.exit(0)  # Warning but not failure
    else:
        sys.exit(0)
