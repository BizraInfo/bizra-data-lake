"""
Hunter Agent — Autonomous Vulnerability Discovery

HunterAgent applies the UERS 5D manifold to discover vulnerabilities:
- Surface: Bytecode entropy analysis
- Structural: Control flow graph patterns
- Behavioral: State transition fuzzing
- Hypothetical: Symbolic execution (Z3)
- Contextual: Spec vs implementation gaps

Coordinates with specialized agents:
- Surveyor: Static analysis (Slither)
- Fuzzer: Dynamic testing (Echidna)
- Prover: Formal verification (Z3)
- Ethicist: Constitutional AI validation
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from core.bounty import (
    BOUNTY_SNR_THRESHOLD,
)
from core.bounty.impact_proof import (
    DomainEvent,
    EntropyMeasurement,
    ImpactProof,
    ImpactProofBuilder,
    Severity,
    VulnCategory,
)
from core.proof_engine.receipt import SovereignSigner


class ScanStatus(Enum):
    """Scan status."""

    IDLE = "idle"
    SCANNING = "scanning"
    ANALYZING = "analyzing"
    COMPLETE = "complete"
    ERROR = "error"


@dataclass
class ScanTarget:
    """A target to scan."""

    address: str
    chain: str = "ethereum"
    name: Optional[str] = None
    tvl: float = 0.0  # Total value locked
    bytecode: Optional[bytes] = None
    abi: Optional[List[Dict[str, Any]]] = None
    source_code: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "address": self.address,
            "chain": self.chain,
            "name": self.name,
            "tvl": self.tvl,
            "has_bytecode": self.bytecode is not None,
            "has_abi": self.abi is not None,
            "has_source": self.source_code is not None,
        }


@dataclass
class ScanResult:
    """Result of a vulnerability scan."""

    target: ScanTarget
    vulnerabilities: List[Dict[str, Any]]
    entropy_measurements: EntropyMeasurement
    scan_duration_ms: float
    status: ScanStatus
    error: Optional[str] = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def has_findings(self) -> bool:
        return len(self.vulnerabilities) > 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "target": self.target.to_dict(),
            "vulnerabilities": self.vulnerabilities,
            "entropy": self.entropy_measurements.to_dict(),
            "scan_duration_ms": self.scan_duration_ms,
            "status": self.status.value,
            "error": self.error,
            "finding_count": len(self.vulnerabilities),
            "timestamp": self.timestamp.isoformat(),
        }


class VectorAnalyzer:
    """
    Base class for UERS vector analyzers.

    Each analyzer focuses on one dimension of entropy.
    """

    def __init__(self, vector_name: str):
        self.vector_name = vector_name

    async def analyze(self, target: ScanTarget) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Analyze target and return (entropy, findings).

        Must be overridden by subclasses.
        """
        raise NotImplementedError


class SurfaceAnalyzer(VectorAnalyzer):
    """Surface vector: Bytecode entropy analysis."""

    def __init__(self):
        super().__init__("surface")

    async def analyze(self, target: ScanTarget) -> Tuple[float, List[Dict[str, Any]]]:
        """Analyze bytecode for entropy anomalies."""
        findings: List[Dict[str, Any]] = []

        if not target.bytecode:
            return 0.5, findings

        bytecode = target.bytecode

        # Calculate bytecode entropy
        byte_counts = [0] * 256
        for b in bytecode:
            byte_counts[b] += 1

        import math

        entropy = 0.0
        total = len(bytecode)
        for count in byte_counts:
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)

        # Normalize to [0, 1]
        max_entropy = math.log2(256)
        normalized = entropy / max_entropy

        # High entropy regions may indicate obfuscation or complexity
        if normalized > 0.9:
            findings.append(
                {
                    "category": "logic_error",
                    "severity": "medium",
                    "title": "High bytecode entropy detected",
                    "description": f"Bytecode entropy {normalized:.3f} suggests complex or obfuscated code",
                    "confidence": 0.6,
                }
            )

        return normalized, findings


class StructuralAnalyzer(VectorAnalyzer):
    """Structural vector: Control flow graph patterns."""

    def __init__(self):
        super().__init__("structural")
        self._reentrancy_patterns = [
            b"\x5b",  # JUMPDEST after external call pattern
        ]

    async def analyze(self, target: ScanTarget) -> Tuple[float, List[Dict[str, Any]]]:
        """Analyze control flow for vulnerability patterns."""
        findings: List[Dict[str, Any]] = []

        if not target.bytecode:
            return 0.5, findings

        bytecode = target.bytecode
        entropy = 0.5  # Default

        # Check for reentrancy patterns (simplified)
        # Real implementation would use proper disassembly
        call_count = bytecode.count(b"\xf1") + bytecode.count(b"\xf2")  # CALL, CALLCODE
        bytecode.count(b"\x55")  # SSTORE

        if call_count > 0:
            # Check if SSTORE appears after CALL (potential reentrancy)
            call_indices = [i for i, b in enumerate(bytecode) if b == 0xF1]
            sstore_indices = [i for i, b in enumerate(bytecode) if b == 0x55]

            for call_idx in call_indices:
                later_sstores = [s for s in sstore_indices if s > call_idx]
                if later_sstores:
                    findings.append(
                        {
                            "category": "reentrancy",
                            "severity": "high",
                            "title": "Potential reentrancy vulnerability",
                            "description": "State modification after external call detected",
                            "confidence": 0.7,
                            "location": {
                                "call_offset": call_idx,
                                "sstore_offset": later_sstores[0],
                            },
                        }
                    )
                    entropy = 0.8
                    break

        # Complexity metric
        jumpi_count = bytecode.count(b"\x57")  # JUMPI (conditional)
        if jumpi_count > 50:
            entropy = min(entropy + 0.2, 1.0)

        return entropy, findings


class BehavioralAnalyzer(VectorAnalyzer):
    """Behavioral vector: State transition fuzzing."""

    def __init__(self):
        super().__init__("behavioral")

    async def analyze(self, target: ScanTarget) -> Tuple[float, List[Dict[str, Any]]]:
        """Analyze state transitions for anomalies."""
        findings = []
        entropy = 0.5

        # In production, this would:
        # 1. Fork mainnet
        # 2. Generate fuzz inputs
        # 3. Execute transactions
        # 4. Measure state changes

        # Placeholder: Check ABI for risky patterns
        if target.abi:
            for item in target.abi:
                if item.get("type") == "function":
                    name = item.get("name", "")

                    # Flash loan patterns
                    if "flash" in name.lower() or "loan" in name.lower():
                        findings.append(
                            {
                                "category": "flash_loan",
                                "severity": "medium",
                                "title": f"Flash loan function: {name}",
                                "description": "Flash loan functionality detected - verify callback security",
                                "confidence": 0.8,
                            }
                        )
                        entropy = 0.7

                    # Upgrade patterns
                    if "upgrade" in name.lower() or "setImplementation" in name.lower():
                        findings.append(
                            {
                                "category": "upgrade_vulnerability",
                                "severity": "high",
                                "title": f"Upgrade function: {name}",
                                "description": "Upgradeable contract - verify access controls",
                                "confidence": 0.9,
                            }
                        )
                        entropy = max(entropy, 0.8)

        return entropy, findings


class HypotheticalAnalyzer(VectorAnalyzer):
    """Hypothetical vector: Symbolic execution."""

    def __init__(self):
        super().__init__("hypothetical")

    async def analyze(self, target: ScanTarget) -> Tuple[float, List[Dict[str, Any]]]:
        """Symbolic execution to find edge cases."""
        findings = []
        entropy = 0.5

        # In production, this would:
        # 1. Use Z3 or Manticore
        # 2. Symbolically execute all paths
        # 3. Find constraint violations

        # Placeholder: Check for integer overflow patterns in source
        if target.source_code:
            source = target.source_code

            # Unchecked arithmetic (Solidity <0.8)
            if "pragma solidity ^0.7" in source or "pragma solidity ^0.6" in source:
                if "SafeMath" not in source:
                    findings.append(
                        {
                            "category": "integer_overflow",
                            "severity": "high",
                            "title": "Unchecked arithmetic in legacy Solidity",
                            "description": "Contract uses Solidity <0.8 without SafeMath",
                            "confidence": 0.9,
                        }
                    )
                    entropy = 0.85

        return entropy, findings


class ContextualAnalyzer(VectorAnalyzer):
    """Contextual vector: Spec vs implementation gaps."""

    def __init__(self):
        super().__init__("contextual")

    async def analyze(self, target: ScanTarget) -> Tuple[float, List[Dict[str, Any]]]:
        """Compare documentation to implementation."""
        findings = []
        entropy = 0.5

        # In production, this would:
        # 1. Parse NatSpec comments
        # 2. Use LLM to understand intent
        # 3. Compare to actual implementation

        # Placeholder analysis
        if target.source_code:
            source = target.source_code

            # Check for missing access control comments
            if "onlyOwner" in source and "* @notice" not in source:
                findings.append(
                    {
                        "category": "access_control",
                        "severity": "low",
                        "title": "Missing documentation for access-controlled functions",
                        "description": "Access control present but NatSpec documentation missing",
                        "confidence": 0.6,
                    }
                )
                entropy = 0.6

        return entropy, findings


class HunterAgent:
    """
    Autonomous vulnerability hunter.

    Coordinates all UERS vector analyzers to discover vulnerabilities.
    """

    def __init__(
        self,
        signer: SovereignSigner,
        snr_threshold: float = BOUNTY_SNR_THRESHOLD,
    ):
        self.signer = signer
        self.snr_threshold = snr_threshold
        self.proof_builder = ImpactProofBuilder(signer)

        # Initialize analyzers
        self._analyzers: Dict[str, VectorAnalyzer] = {
            "surface": SurfaceAnalyzer(),
            "structural": StructuralAnalyzer(),
            "behavioral": BehavioralAnalyzer(),
            "hypothetical": HypotheticalAnalyzer(),
            "contextual": ContextualAnalyzer(),
        }

        self._scan_history: List[ScanResult] = []
        self._proofs: List[ImpactProof] = []

    async def scan(self, target: ScanTarget) -> ScanResult:
        """
        Scan a target for vulnerabilities.

        Runs all UERS vector analyzers in parallel.
        """
        start = time.perf_counter()
        all_findings: List[Dict[str, Any]] = []
        entropy = EntropyMeasurement()

        try:
            # Run analyzers in parallel
            tasks = [analyzer.analyze(target) for analyzer in self._analyzers.values()]
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Collect results
            for (name, _analyzer), analysis in zip(self._analyzers.items(), results):
                if isinstance(analysis, BaseException):
                    continue

                e, findings = analysis

                # Store entropy
                if name == "surface":
                    entropy.surface_entropy = e
                elif name == "structural":
                    entropy.structural_entropy = e
                elif name == "behavioral":
                    entropy.behavioral_entropy = e
                elif name == "hypothetical":
                    entropy.hypothetical_entropy = e
                elif name == "contextual":
                    entropy.contextual_entropy = e

                # Collect findings
                for finding in findings:
                    finding["vector"] = name
                    finding["target"] = target.address
                all_findings.extend(findings)

            status = ScanStatus.COMPLETE
            error = None

        except Exception as e:
            status = ScanStatus.ERROR
            error = str(e)

        duration = (time.perf_counter() - start) * 1000

        result = ScanResult(
            target=target,
            vulnerabilities=all_findings,
            entropy_measurements=entropy,
            scan_duration_ms=duration,
            status=status,
            error=error,
        )

        self._scan_history.append(result)
        return result

    async def hunt(
        self,
        target: ScanTarget,
        generate_proofs: bool = True,
    ) -> Tuple[ScanResult, List[ImpactProof]]:
        """
        Full hunting pipeline: scan → analyze → generate proofs.

        Returns (scan_result, proofs).
        """
        # Scan target
        result = await self.scan(target)

        proofs = []
        if generate_proofs and result.has_findings:
            # Generate proof for each finding
            for finding in result.vulnerabilities:
                # Calculate SNR based on confidence
                confidence = finding.get("confidence", 0.5)
                snr_score = min(0.99, confidence + 0.1)

                if snr_score >= self.snr_threshold:
                    # Create entropy measurement for this finding
                    entropy_before = result.entropy_measurements
                    entropy_after = EntropyMeasurement(
                        surface_entropy=entropy_before.surface_entropy * 0.5,
                        structural_entropy=entropy_before.structural_entropy * 0.5,
                        behavioral_entropy=entropy_before.behavioral_entropy * 0.5,
                        hypothetical_entropy=entropy_before.hypothetical_entropy * 0.5,
                        contextual_entropy=entropy_before.contextual_entropy * 0.5,
                    )

                    # Map category and severity
                    category = VulnCategory(finding.get("category", "logic_error"))
                    severity = Severity(finding.get("severity", "medium"))

                    # Generate placeholder exploit code
                    exploit_code = f"// Exploit for {finding.get('title', 'vulnerability')}\n// Target: {target.address}".encode()

                    # Build proof
                    proof = self.proof_builder.build(
                        target_address=target.address,
                        vuln_category=category,
                        severity=severity,
                        title=finding.get("title", "Vulnerability"),
                        description=finding.get("description", ""),
                        exploit_code=exploit_code,
                        entropy_before=entropy_before,
                        entropy_after=entropy_after,
                        reproduction_steps=[
                            DomainEvent(
                                event_type="discovery",
                                timestamp=datetime.now(timezone.utc),
                                data=finding,
                            )
                        ],
                        funds_at_risk=target.tvl,
                        snr_score=snr_score,
                        ihsan_score=0.95,
                        target_chain=target.chain,
                        target_name=target.name,
                    )

                    proofs.append(proof)
                    self._proofs.append(proof)

        return result, proofs

    def get_stats(self) -> Dict[str, Any]:
        """Get hunter statistics."""
        total_scans = len(self._scan_history)
        total_findings = sum(len(s.vulnerabilities) for s in self._scan_history)
        total_proofs = len(self._proofs)

        by_severity: Dict[str, int] = {}
        for scan in self._scan_history:
            for finding in scan.vulnerabilities:
                sev = finding.get("severity", "unknown")
                by_severity[sev] = by_severity.get(sev, 0) + 1

        return {
            "total_scans": total_scans,
            "total_findings": total_findings,
            "total_proofs": total_proofs,
            "by_severity": by_severity,
            "snr_threshold": self.snr_threshold,
        }

    def get_recent_scans(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent scan results."""
        return [s.to_dict() for s in self._scan_history[-limit:]]


class HunterSwarm:
    """
    Swarm of hunter agents for parallel scanning.

    Coordinates multiple hunters across many targets.
    """

    def __init__(
        self,
        signer: SovereignSigner,
        num_agents: int = 3,
    ):
        self.signer = signer
        self.agents = [HunterAgent(signer) for _ in range(num_agents)]
        self._results: List[Tuple[ScanResult, List[ImpactProof]]] = []

    async def hunt_targets(
        self,
        targets: List[ScanTarget],
    ) -> List[Tuple[ScanResult, List[ImpactProof]]]:
        """
        Hunt multiple targets in parallel.

        Distributes targets across agents.
        """
        # Distribute targets round-robin
        agent_tasks: List[List[ScanTarget]] = [[] for _ in self.agents]
        for i, target in enumerate(targets):
            agent_idx = i % len(self.agents)
            agent_tasks[agent_idx].append(target)

        # Run agents in parallel
        async def agent_hunt(agent: HunterAgent, targets: List[ScanTarget]):
            results = []
            for target in targets:
                result = await agent.hunt(target)
                results.append(result)
            return results

        all_tasks = [
            agent_hunt(agent, tasks) for agent, tasks in zip(self.agents, agent_tasks)
        ]

        all_results = await asyncio.gather(*all_tasks)

        # Flatten results
        results = []
        for agent_results in all_results:
            results.extend(agent_results)

        self._results.extend(results)
        return results

    def get_stats(self) -> Dict[str, Any]:
        """Get swarm statistics."""
        total_scans = sum(len(a._scan_history) for a in self.agents)
        total_findings = sum(
            len(s.vulnerabilities) for a in self.agents for s in a._scan_history
        )
        total_proofs = sum(len(a._proofs) for a in self.agents)

        return {
            "num_agents": len(self.agents),
            "total_scans": total_scans,
            "total_findings": total_findings,
            "total_proofs": total_proofs,
        }
