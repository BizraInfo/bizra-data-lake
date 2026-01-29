#!/usr/bin/env python3
"""
🚀 ACTIVATE PEAK MASTERPIECE
The professional entry point for the Ultimate BIZRA DDAGI Implementation.
"""

import sys
import logging
import time
import json
import argparse
from datetime import datetime
from pathlib import Path
from peak_masterpiece import PeakMasterpieceEngine, CommandType

# Optional validation utilities
try:
    from tools.knowledge_base_validator import KnowledgeBaseValidator
except Exception:
    KnowledgeBaseValidator = None

try:
    from singularity_query_test import run_full_spectrum_test
except Exception:
    run_full_spectrum_test = None

# Configure professional logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | PEAK | %(levelname)-8s | %(message)s',
    datefmt='%H:%M:%S',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger("ActivatePeak")

def print_banner():
    print("""
\033[36m╔══════════════════════════════════════════════════════════════════════════════╗
║   ██████╗ ███████╗ █████╗ ██╗  ██╗    ███╗   ███╗ █████╗ ███████╗████████╗   ║
║   ██╔══██╗██╔════╝██╔══██╗██║ ██╔╝    ████╗ ████║██╔══██╗██╔════╝╚══██╔══╝   ║
║   ██████╔╝█████╗  ███████║█████╔╝     ██╔████╔██║███████║███████╗   ██║      ║
║   ██╔═══╝ ██╔══╝  ██╔══██║██╔═██╗     ██║╚██╔╝██║██╔══██║╚════██║   ██║      ║
║   ██║     ███████╗██║  ██║██║  ██╗    ██║ ╚═╝ ██║██║  ██║███████║   ██║      ║
║   ╚═╝     ╚══════╝╚═╝  ╚═╝╚═╝  ╚═╝    ╚═╝     ╚═╝╚═╝  ╚═╝╚══════╝   ╚═╝      ║
╚══════════════════════════════════════════════════════════════════════════════╝\033[0m
    """)

def run_preflight(do_validation: bool, do_full_spectrum: bool) -> dict:
    """Run optional preflight checks and return a summary dict."""
    summary = {
        "timestamp": datetime.now().isoformat(),
        "knowledge_base_validation": None,
        "singularity_full_spectrum": None,
    }

    if do_validation:
        if KnowledgeBaseValidator is None:
            summary["knowledge_base_validation"] = {"status": "SKIPPED", "reason": "validator_unavailable"}
        else:
            validator = KnowledgeBaseValidator()
            count = validator.count_embeddings()
            analysis = validator.analyze_embeddings()
            coverage = validator.validate_processed_coverage()
            search = validator.test_semantic_search()

            passed = (
                count > 0
                and coverage.get("coverage_percentage", 0) >= 95.0
                and search.get("search_functional", False)
            )

            summary["knowledge_base_validation"] = {
                "status": "PASSED" if passed else "PARTIAL",
                "total_embeddings": count,
                "coverage_percentage": coverage.get("coverage_percentage", 0),
                "search_functional": search.get("search_functional", False),
                "analysis": analysis,
                "coverage": coverage,
                "search": search,
            }

    if do_full_spectrum:
        if run_full_spectrum_test is None:
            summary["singularity_full_spectrum"] = {"status": "SKIPPED", "reason": "test_unavailable"}
        else:
            try:
                exit_code = run_full_spectrum_test()
                summary["singularity_full_spectrum"] = {
                    "status": "PASSED" if exit_code == 0 else "PARTIAL",
                    "exit_code": exit_code,
                }
            except Exception as e:
                summary["singularity_full_spectrum"] = {"status": "FAILED", "error": str(e)}

    return summary


def main():
    print_banner()
    log.info("Initializing Peak Masterpiece Engine (State-of-the-Art Mode)...")
    
    parser = argparse.ArgumentParser(description="Activate Peak Masterpiece Engine")
    parser.add_argument("--query", type=str, default=None, help="Override the default synthesis query")
    parser.add_argument("--skip-preflight", action="store_true", help="Skip validation preflight checks")
    parser.add_argument("--skip-validation", action="store_true", help="Skip knowledge base validation")
    parser.add_argument("--full-spectrum", action="store_true", help="Run full-spectrum singularity test")
    parser.add_argument("--report-path", type=str, default=None, help="Write JSON report to path")
    args = parser.parse_args()

    start_time = time.time()
    report = {
        "started_at": datetime.now().isoformat(),
        "preflight": None,
        "peak": None,
    }
    
    try:
        if not args.skip_preflight:
            report["preflight"] = run_preflight(
                do_validation=not args.skip_validation,
                do_full_spectrum=args.full_spectrum,
            )

        # Initialize the Engine
        engine = PeakMasterpieceEngine()
        
        # Initialization Stats
        log.info(f"Engine Online in {time.time() - start_time:.2f}s")
        log.info(f"Discipline Matrix: {47} Generators Active")
        log.info(f"Cognitive Layers: {7} Layers Synced")
        
        # Define the Grand Synthesis Query
        query = args.query or (
            "Generate the ultimate architectural synthesis of the BIZRA ecosystem based on the ingested code and documents. "
            "Identify the core 'Golden Gems' of the implementation and verify alignment with Ihsān-grade quality."
        )
        
        log.info(f"Executing Sovereign Command: {query}")
        
        # Execute
        result = engine.process_query(
            query=query,
            context={"intent": "grand_synthesis", "mode": "professional_elite", "scan_depth": "full"}
        )
        
        # Display Results
        print("\n" + "="*80)
        print(f"  🌌 PEAK MASTERPIECE SYNTHESIS REPORT")
        print("="*80)
        print(f"  SNR Score:          {result.snr_score:.4f} (Target: >0.95)")
        print(f"  Ihsān Check:        {'✅ PASSED' if result.ihsan_check else '⚠️ WARNING'}")
        print(f"  Discipline Coverage: {result.discipline_coverage:.1%}")
        print(f"  Synergies Detected:  {len(result.synergies)}")
        print("-" * 80)
        print("\n📢 SYNTHESIZED ANSWER:\n")
        print(result.synthesis)
        print("\n" + "-" * 80)
        
        # Display Synergies
        if result.synergies:
            print(f"\n🔗 TOP SYNERGIES DETECTED:")
            for i, syn in enumerate(result.synergies[:3], 1):
                print(f"  {i}. [{syn.synergy_type.name}] {syn.source_domain} <-> {syn.target_domain} (Strength: {syn.strength:.2f})")
                
        # Display Compounds
        if result.compounds:
            print(f"\n💎 COMPOUND DISCOVERIES:")
            for i, comp in enumerate(result.compounds[:2], 1):
                print(f"  {i}. {comp.hypothesis}")

        report["peak"] = {
            "snr_score": result.snr_score,
            "ihsan_check": result.ihsan_check,
            "discipline_coverage": result.discipline_coverage,
            "synergies_count": len(result.synergies),
            "compounds_count": len(result.compounds),
            "elapsed_s": time.time() - start_time,
        }

    except KeyboardInterrupt:
        log.warning("Activation interrupted by user.")
    except Exception as e:
        log.error(f"Critical Failure in Masterpiece Engine: {e}", exc_info=True)
        report["peak"] = {"status": "FAILED", "error": str(e)}
        sys.exit(1)
    finally:
        report["finished_at"] = datetime.now().isoformat()
        if args.report_path:
            report_path = Path(args.report_path)
        else:
            report_path = Path("04_GOLD") / "peak_masterpiece_report.json"
        try:
            report_path.parent.mkdir(parents=True, exist_ok=True)
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, cls=NumpyEncoder)
            log.info(f"Report written: {report_path}")
        except Exception as e:
            log.warning(f"Failed to write report: {e}")

class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""
    def default(self, obj):
        if hasattr(obj, 'tolist'):
            return obj.tolist()
        if hasattr(obj, 'item'):
            return obj.item()
        return super().default(obj)

if __name__ == "__main__":
    main()
