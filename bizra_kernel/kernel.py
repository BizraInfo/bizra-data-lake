"""
SystemProtocolKernel v2.0 — The Ethical Microkernel
====================================================
From the Blueprint:
  The PAB's SystemProtocolKernel is not a middleware—it is the
  ethical microkernel that sits between Layer 3 (Execution) and
  Layer 4 (Cognitive), enforcing Ihsān thresholds, protocol hashing,
  and SAPE elevation.

This is the central orchestrator that:
1. Creates protocol-hashed sessions
2. Runs the 9-probe verification on all outputs
3. Computes Ihsān vectors and enforces thresholds
4. Tracks SNR and elevates patterns via SAPE
5. Escalates to FATE protocol when thresholds fail
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any
import json
import hashlib

from .ihsan_vector import IhsanVector, IhsanDimension, IHSAN_THRESHOLD
from .session_manager import SessionManager, Session, SessionState
from .verifier import MultiStageVerifier, VerificationResult
from .snr_tracker import SNRTracker, SNRMetrics, estimate_useful_tokens
from .sape_engine import SAPEEngine, ElevatedPattern
from .identity import get_identity
from .damage_control_engine import DamageControlEngine


@dataclass
class KernelConfig:
    """Configuration for the SystemProtocolKernel."""
    protocol_version: str = "2.0.0"
    ihsan_threshold: float = 0.95
    snr_target: float = 0.90
    sape_elevation_threshold: int = 3
    fate_circuit_breaker_ms: int = 100
    enable_verification: bool = True
    enable_sape: bool = True
    enable_snr_tracking: bool = True
    enable_damage_control: bool = True
    
    def to_dict(self) -> dict:
        return {
            "protocol_version": self.protocol_version,
            "ihsan_threshold": self.ihsan_threshold,
            "snr_target": self.snr_target,
            "sape_elevation_threshold": self.sape_elevation_threshold,
            "fate_circuit_breaker_ms": self.fate_circuit_breaker_ms,
            "enable_verification": self.enable_verification,
            "enable_sape": self.enable_sape,
            "enable_snr_tracking": self.enable_snr_tracking,
            "enable_damage_control": self.enable_damage_control,
        }
    
    def hash(self) -> str:
        """Compute SHA256 hash of kernel configuration."""
        canonical = json.dumps(self.to_dict(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()


@dataclass
class ExecutionResult:
    """Result of kernel-managed execution."""
    session_id: str
    response: str
    agent: str
    ihsan_vector: IhsanVector
    verification: VerificationResult
    snr_metrics: SNRMetrics
    elevated_pattern: Optional[ElevatedPattern]
    latency_ms: int
    passed: bool
    fate_escalated: bool
    protocol_hash: str


class SystemProtocolKernel:
    """
    The Ethical Microkernel — Layer 3.5 of APEX Architecture.
    
    Orchestrates all agent execution with:
    - Protocol hashing for traceability
    - Ihsān enforcement for ethics
    - 9-probe verification for correctness
    - SNR tracking for efficiency
    - SAPE for continuous optimization
    """
    
    VERSION = "2.0.0"
    
    def __init__(self, config: Optional[KernelConfig] = None):
        self.config = config or KernelConfig()
        self.session_manager = SessionManager()
        self.verifier = MultiStageVerifier()
        self.snr_tracker = SNRTracker()
        self.sape_engine = SAPEEngine()
        self.damage_control = DamageControlEngine() if self.config.enable_damage_control else None
        
        # FATE escalation callback (can be overridden)
        self.fate_callback: Optional[Callable[[Session, str], None]] = None
        
        # Protocol hash for this kernel instance
        self.protocol_hash = self.config.hash()
    
    def execute(
        self,
        agent: str,
        query: str,
        response: str,
        knowledge_context: str = "",
        token_count: int = 0,
        latency_ms: int = 0,
        user_id: str = "momo",  # Dedicated identity anchor
    ) -> ExecutionResult:
        """
        Execute a kernel-managed agent interaction.
        
        This is the main entry point that:
        1. Creates a protocol-hashed session
        2. Runs verification
        3. Computes Ihsān score
        4. Tracks SNR
        5. Checks for SAPE elevation
        6. Returns complete execution result
        """
        # 1. Create session
        session = self.session_manager.create_session(agent, query, user_id)
        
        # 2. Run 9-probe verification
        if self.config.enable_verification:
            verification = self.verifier.verify(
                query=query,
                response=response,
                agent_role=agent,
                knowledge_context=knowledge_context,
                token_count=token_count,
                latency_ms=latency_ms,
            )
        else:
            verification = VerificationResult(
                probe_results=[],
                overall_passed=True,
                composite_score=1.0,
                failing_probes=[],
            )
        
        # 3. Compute Ihsān vector
        ihsan_vec = IhsanVector.from_agent_response(
            response=response,
            latency_ms=latency_ms,
            token_count=token_count,
            rag_used=bool(knowledge_context),
            agent_role=agent,
        )
        
        # Incorporate verification results into Ihsān
        ihsan_vec.set_score(
            IhsanDimension.CORRECTNESS,
            verification.composite_score
        )
        
        # 4. Track SNR
        useful_tokens = estimate_useful_tokens(response)
        safety_compliance = 1.0
        if self.damage_control is not None:
            safety_report = self.damage_control.evaluate_text(query, response)
            safety_compliance = safety_report.get("safety_snr", 1.0)
        snr_metrics = SNRMetrics(
            total_tokens=token_count or len(response.split()),
            useful_tokens=useful_tokens,
            confidence_score=verification.composite_score,
            ethical_compliance=ihsan_vec.composite_score,
            safety_compliance=safety_compliance,
            tool_directness=0.9,  # Heuristic; could be computed
            latency_ms=latency_ms,
            agent_role=agent,
        )
        
        if self.config.enable_snr_tracking:
            self.snr_tracker.record(snr_metrics)
        
        # 5. Check for SAPE elevation
        elevated_pattern = None
        if self.config.enable_sape:
            # Build verification sequence from probe results
            sequence = [
                f"{probe.probe_type.value}:{'pass' if probe.passed else 'fail'}"
                for probe in verification.probe_results
            ]
            elevated_pattern = self.sape_engine.observe_sequence(sequence)
        
        # 6. Determine if execution passed
        passed = (
            ihsan_vec.passes_threshold
            and verification.overall_passed
        )
        
        # 7. Handle FATE escalation if needed
        fate_escalated = False
        if not passed:
            fate_escalated = True
            session.pause_for_fate(
                f"Ihsān: {ihsan_vec.composite_score:.3f}, "
                f"Verification: {verification.composite_score:.3f}"
            )
            if self.fate_callback:
                self.fate_callback(
                    session,
                    f"Execution failed: Ihsān={ihsan_vec.composite_score:.3f}"
                )
        else:
            session.complete(response, ihsan_vec)
        
        # 8. Build execution result
        return ExecutionResult(
            session_id=session.session_id,
            response=response,
            agent=agent,
            ihsan_vector=ihsan_vec,
            verification=verification,
            snr_metrics=snr_metrics,
            elevated_pattern=elevated_pattern,
            latency_ms=latency_ms,
            passed=passed,
            fate_escalated=fate_escalated,
            protocol_hash=self.protocol_hash,
        )
    
    def get_status(self) -> dict:
        """Get comprehensive kernel status."""
        session_stats = self.session_manager.get_statistics()
        snr_stats = self.snr_tracker.get_statistics()
        sape_stats = self.sape_engine.get_statistics()
        
        return {
            "kernel_version": self.VERSION,
            "protocol_hash": self.protocol_hash,
            "config": self.config.to_dict(),
            "sessions": session_stats,
            "snr": snr_stats,
            "sape": sape_stats,
            "health": {
                "ihsan_compliant": session_stats.get("avg_ihsan_score", 0) >= self.config.ihsan_threshold,
                "snr_target_met": snr_stats.get("average_snr", 0) >= self.config.snr_target,
                "fate_escalations": session_stats.get("paused_for_fate", 0),
            },
            "sovereignty": get_identity().to_dict(),
        }
    
    def register_fate_callback(
        self,
        callback: Callable[[Session, str], None]
    ) -> None:
        """Register a callback for FATE escalations."""
        self.fate_callback = callback
    
    def to_poi_receipt(self) -> dict:
        """Generate a PoI receipt for the kernel's current state."""
        status = self.get_status()
        return {
            "receipt_type": "kernel_state",
            "kernel_version": self.VERSION,
            "protocol_hash": self.protocol_hash,
            "timestamp": datetime.utcnow().isoformat(),
            "ihsan_average": status["sessions"].get("avg_ihsan_score", 0),
            "snr_average": status["snr"].get("average_snr", 0),
            "sape_elevations": status["sape"].get("elevated_patterns", 0),
            "health_status": status["health"],
        }


# Singleton instance for global access
_kernel_instance: Optional[SystemProtocolKernel] = None


def get_kernel() -> SystemProtocolKernel:
    """Get the global kernel instance."""
    global _kernel_instance
    if _kernel_instance is None:
        _kernel_instance = SystemProtocolKernel()
    return _kernel_instance


def reset_kernel(config: Optional[KernelConfig] = None) -> SystemProtocolKernel:
    """Reset the global kernel with optional new configuration."""
    global _kernel_instance
    _kernel_instance = SystemProtocolKernel(config)
    return _kernel_instance
