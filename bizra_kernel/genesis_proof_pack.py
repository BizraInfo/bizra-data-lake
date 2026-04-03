"""
Genesis Proof Pack Generator
============================

Automated system to collect and package evidence for Genesis Proof Pack as specified in Phase 9 plan.

Collects:
- Boot logs with Ihsan scores ≥95%
- Z3 formal proofs for Code Law verification
- Ihsān compliance reports with audit trail
- Performance benchmarks (existing suite + federation)

Usage:
    python genesis_proof_pack.py [--output OUTPUT_FILE] [--federation-nodes N]
"""

import json
import os
import glob
from datetime import datetime
from typing import Dict, List, Any, Optional
import argparse

from .ihsan_gate import IhsanGate
from .benchmark_util import BIZRABenchmark


class GenesisProofPack:
    """
    Automated Genesis Proof Pack generator.
    """

    def __init__(self):
        self.ihsan_gate = IhsanGate()
        self.benchmark = BIZRABenchmark()
        self.proof_pack = {
            "metadata": {
                "version": "1.0",
                "generated_at": datetime.utcnow().isoformat(),
                "phase": "Phase 9 - Genesis Proof Pack",
                "description": "Automated collection of evidence for BIZRA Genesis validation"
            },
            "artifacts": {}
        }

    def collect_boot_logs(self, min_score: float = 0.95) -> List[Dict[str, Any]]:
        """
        Collect boot logs with Ihsan scores ≥ min_score.
        """
        audit_log_path = "bizra_memory/ihsan_audit.json"

        boot_logs = []
        if not os.path.exists(audit_log_path):
            raise RuntimeError(f"Audit log not found: {audit_log_path}")

        try:
            with open(audit_log_path, 'r') as f:
                for line in f:
                    if line.strip():
                        entry = json.loads(line.strip())
                        if entry.get('score', 0) >= min_score:
                            boot_logs.append(entry)
        except Exception as e:
            raise RuntimeError(f"Could not read audit log: {e}") from e

        if not boot_logs:
            raise RuntimeError("No boot logs meeting the minimum Ihsan score were found")

        return boot_logs

    def collect_z3_proofs(self) -> List[Dict[str, Any]]:
        """
        Collect Z3 formal proofs for Code Law verification.
        """
        proofs = []

        # Look for proof files in various locations
        proof_patterns = [
            "bizra_kernel/proofs/*.z3",
            "bizra_kernel/proofs/*.smt2",
            "proofs/*.z3",
            "proofs/*.smt2",
            "*.z3",
            "*.smt2"
        ]

        for pattern in proof_patterns:
            for proof_file in glob.glob(pattern):
                try:
                    with open(proof_file, 'r') as f:
                        content = f.read()
                        proofs.append({
                            "filename": os.path.basename(proof_file),
                            "path": proof_file,
                            "content": content,
                            "type": "Z3" if proof_file.endswith('.z3') else "SMT2",
                            "collected_at": datetime.utcnow().isoformat()
                        })
                except Exception as e:
                    print(f"Warning: Could not read proof file {proof_file}: {e}")

        if not proofs:
            raise RuntimeError("No Z3 proof files found for Code Law verification")

        return proofs

    def collect_ihsan_compliance_reports(self) -> Dict[str, Any]:
        """
        Collect Ihsān compliance reports with audit trail.
        """
        audit_log_path = "bizra_memory/ihsan_audit.json"
        assumptions_log_path = "bizra_memory/assumptions.json"

        compliance_report = {
            "audit_trail": [],
            "assumptions_log": [],
            "summary": {
                "total_audits": 0,
                "passed_audits": 0,
                "failed_audits": 0,
                "average_score": 0.0,
                "assumptions_made": 0
            }
        }

        # Read audit trail
        if os.path.exists(audit_log_path):
            try:
                with open(audit_log_path, 'r') as f:
                    scores = []
                    for line in f:
                        if line.strip():
                            entry = json.loads(line.strip())
                            compliance_report["audit_trail"].append(entry)
                            scores.append(entry.get('score', 0))
                            if entry.get('result') == 'PASS':
                                compliance_report["summary"]["passed_audits"] += 1
                            else:
                                compliance_report["summary"]["failed_audits"] += 1

                    compliance_report["summary"]["total_audits"] = len(scores)
                    if scores:
                        compliance_report["summary"]["average_score"] = sum(scores) / len(scores)
            except Exception as e:
                print(f"Warning: Could not read audit log: {e}")

        # Read assumptions log
        if os.path.exists(assumptions_log_path):
            try:
                with open(assumptions_log_path, 'r') as f:
                    for line in f:
                        if line.strip():
                            entry = json.loads(line.strip())
                            compliance_report["assumptions_log"].append(entry)
                    compliance_report["summary"]["assumptions_made"] = len(compliance_report["assumptions_log"])
            except Exception as e:
                print(f"Warning: Could not read assumptions log: {e}")

        return compliance_report

    def collect_performance_benchmarks(self, federation_nodes: int = 1) -> Dict[str, Any]:
        """
        Collect performance benchmarks (existing suite + federation).
        """
        benchmarks = {
            "logic_gate_benchmark": {},
            "blockgraph_benchmark": {},
            "federation_results": [],
        }

        print("Running Ihsan Gate benchmark...")
        try:
            logic_results = self.benchmark.benchmark_logic_gate()
            benchmarks["logic_gate_benchmark"] = {
                "iterations": 1000,
                "results": logic_results,
                "timestamp": datetime.utcnow().isoformat()
            }
        except Exception as e:
            print(f"Warning: Logic gate benchmark failed: {e}")
            benchmarks["logic_gate_benchmark"] = {"error": str(e)}

        print("Loading BlockGraph benchmark...")
        try:
            benchmarks["blockgraph_benchmark"] = self.benchmark.load_blockgraph_benchmark()
        except Exception as e:
            raise RuntimeError(f"BlockGraph benchmark load failed: {e}") from e

        federation_path = os.getenv("BIZRA_FEDERATION_BENCHMARK_PATH")
        if not federation_path or not os.path.exists(federation_path):
            raise RuntimeError("BIZRA_FEDERATION_BENCHMARK_PATH is required")
        with open(federation_path, "r", encoding="utf-8") as f:
            federation_results = json.load(f)
        if not isinstance(federation_results, list):
            raise RuntimeError("Federation benchmark must be a JSON array")
        benchmarks["federation_results"] = federation_results

        return benchmarks

    def generate_proof_pack(self, federation_nodes: int = 1) -> Dict[str, Any]:
        """
        Generate the complete Genesis Proof Pack.
        """
        print("Collecting boot logs with Ihsan scores >=95%...")
        self.proof_pack["artifacts"]["boot_logs"] = self.collect_boot_logs()

        print("Collecting Z3 formal proofs...")
        self.proof_pack["artifacts"]["z3_proofs"] = self.collect_z3_proofs()

        print("Collecting Ihsan compliance reports...")
        self.proof_pack["artifacts"]["ihsan_compliance"] = self.collect_ihsan_compliance_reports()

        print("Collecting performance benchmarks...")
        self.proof_pack["artifacts"]["performance_benchmarks"] = self.collect_performance_benchmarks(federation_nodes)

        # Add integrity hash
        import hashlib
        pack_content = json.dumps(self.proof_pack["artifacts"], sort_keys=True)
        self.proof_pack["metadata"]["integrity_hash"] = hashlib.sha256(pack_content.encode()).hexdigest()

        return self.proof_pack

    def save_proof_pack(self, output_file: str = "genesis_proof_pack.json") -> None:
        """
        Save the proof pack to a file.
        """
        with open(output_file, 'w') as f:
            json.dump(self.proof_pack, f, indent=2)
        print(f"Genesis Proof Pack saved to: {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate Genesis Proof Pack")
    parser.add_argument("--output", "-o", default="genesis_proof_pack.json",
                       help="Output file path (default: genesis_proof_pack.json)")
    parser.add_argument("--federation-nodes", "-n", type=int, default=3,
                       help="Number of federation nodes to simulate (default: 3)")

    args = parser.parse_args()

    generator = GenesisProofPack()
    proof_pack = generator.generate_proof_pack(federation_nodes=args.federation_nodes)
    generator.save_proof_pack(args.output)

    print("\nGenesis Proof Pack Summary:")
    print(f"- Boot logs collected: {len(proof_pack['artifacts']['boot_logs'])}")
    print(f"- Z3 proofs collected: {len(proof_pack['artifacts']['z3_proofs'])}")
    print(f"- Audit trail entries: {proof_pack['artifacts']['ihsan_compliance']['summary']['total_audits']}")
    print(f"- Federation nodes benchmarked: {len(proof_pack['artifacts']['performance_benchmarks']['federation_results'])}")
    print(f"- Integrity hash: {proof_pack['metadata']['integrity_hash'][:16]}...")


if __name__ == "__main__":
    main()
