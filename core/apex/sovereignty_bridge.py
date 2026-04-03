"""
SovereigntyBridge - Unified Sovereignty Integration for BIZRA Apex Orchestrator
=================================================================================

Bridge to all sovereignty components for 100% offline operation capability.
Integrates WinterProofEmbedder, Constitution, DaughterTest, and LocalMerkleDAG.

Key Features:
- Complete offline operation (HYPER LOOPBACK)
- Determinism verification for all operations
- MerkleDAG evidence chaining
- Constitutional compliance verification
- Graceful fallbacks for all components

Domain: bizra-pci-v1:
Threshold: 0.95 Ihsan minimum
Hash: SHA-256 and BLAKE3 for integrity

NO external API dependencies - pure HYPER LOOPBACK operation.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeVar,
)

# Import sovereignty components
from core.sovereignty import (
    WinterProofEmbedder,
    Constitution,
    DaughterTest,
    LocalMerkleDAG,
)
from core.sovereignty.constitution import (
    ComplianceReceipt,
    ComplianceViolation,
    ConstitutionalViolationError,
)
from core.sovereignty.merkle_dag import MerkleNode, VerificationResult
from core.sovereignty.daughter_test import IntegrityCheck, ViolationAlert

# Type variable for generic function determinism verification
T = TypeVar('T')

# Configure logging
logger = logging.getLogger("sovereignty_bridge")

# Import constitutional thresholds
from core.constants import IHSAN_THRESHOLD

# Constants
DOMAIN_PREFIX = "bizra-pci-v1:"
DEFAULT_IHSAN_THRESHOLD = 0.95
DEFAULT_EMBEDDING_DIM = 384


# =============================================================================
# DATA CLASSES
# =============================================================================


@dataclass
class SovereigntyVerification:
    """
    Result of sovereignty verification combining all component checks.

    Attributes:
        passed: Overall verification passed
        constitution_score: Weighted compliance score from Constitution
        daughter_test_passed: Whether integrity checks passed
        merkle_node_id: ID of the evidence node in MerkleDAG
        receipt: Full compliance receipt from Constitution
        integrity_checks: List of integrity checks performed
        violations: List of any violations detected
        evidence_chain_valid: Whether MerkleDAG chain is intact
        metadata: Additional verification context
    """
    passed: bool
    constitution_score: float
    daughter_test_passed: bool
    merkle_node_id: str
    receipt: ComplianceReceipt
    integrity_checks: List[IntegrityCheck] = field(default_factory=list)
    violations: List[ViolationAlert] = field(default_factory=list)
    evidence_chain_valid: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "passed": self.passed,
            "constitution_score": self.constitution_score,
            "daughter_test_passed": self.daughter_test_passed,
            "merkle_node_id": self.merkle_node_id,
            "receipt_id": self.receipt.receipt_id,
            "receipt_compliant": self.receipt.compliant,
            "receipt_integrity_hash": self.receipt.integrity_hash,
            "integrity_check_count": len(self.integrity_checks),
            "violation_count": len(self.violations),
            "evidence_chain_valid": self.evidence_chain_valid,
            "metadata": self.metadata,
        }


@dataclass
class DeterminismReport:
    """
    Report from determinism verification.

    Attributes:
        is_deterministic: Whether function produced consistent outputs
        iterations: Number of test iterations performed
        unique_outputs: Count of unique outputs observed
        output_hashes: List of output hashes from each iteration
        function_name: Name of the function tested
        inputs_hash: Hash of the input arguments
        metadata: Additional context
    """
    is_deterministic: bool
    iterations: int
    unique_outputs: int
    output_hashes: List[str]
    function_name: str
    inputs_hash: str
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_deterministic": self.is_deterministic,
            "iterations": self.iterations,
            "unique_outputs": self.unique_outputs,
            "output_hashes": self.output_hashes,
            "function_name": self.function_name,
            "inputs_hash": self.inputs_hash,
            "metadata": self.metadata,
        }


@dataclass
class EvidenceNode:
    """
    Lightweight wrapper for evidence added to MerkleDAG.

    Attributes:
        node_id: Unique identifier in MerkleDAG
        timestamp: When evidence was recorded
        evidence_type: Category of evidence
        data_hash: Hash of the evidence data
        parent_ids: Parent node IDs for lineage
        merkle_root: Merkle root including ancestors
    """
    node_id: str
    timestamp: str
    evidence_type: str
    data_hash: str
    parent_ids: List[str]
    merkle_root: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# =============================================================================
# SOVEREIGNTY BRIDGE
# =============================================================================


class SovereigntyBridge:
    """
    Unified bridge to all BIZRA sovereignty components.

    Provides 100% offline operation capability by integrating:
    - WinterProofEmbedder: Deterministic embeddings without external APIs
    - Constitution: Compliance verification with Ihsan thresholds
    - DaughterTest: Continuous integrity verification
    - LocalMerkleDAG: Tamper-proof evidence chains

    All operations are designed to gracefully fallback and never
    require external API calls.
    """

    def __init__(
        self,
        embedding_dim: int = DEFAULT_EMBEDDING_DIM,
        ihsan_threshold: float = DEFAULT_IHSAN_THRESHOLD,
        merkle_storage_path: Optional[str] = None,
        auto_monitor: bool = False,
    ):
        """
        Initialize SovereigntyBridge with all sovereignty components.

        Args:
            embedding_dim: Dimension for WinterProofEmbedder (must be divisible by 3)
            ihsan_threshold: Minimum Ihsan compliance threshold
            merkle_storage_path: Optional path for persistent MerkleDAG storage
            auto_monitor: Start DaughterTest continuous monitoring
        """
        self.embedding_dim = embedding_dim
        self.ihsan_threshold = ihsan_threshold

        # Initialize WinterProofEmbedder
        try:
            self._embedder = WinterProofEmbedder(
                dimension=embedding_dim,
                use_numpy=True  # Use numpy if available for acceleration
            )
            self._embedder_available = True
            logger.info(f"WinterProofEmbedder initialized: dim={embedding_dim}")
        except Exception as e:
            logger.warning(f"WinterProofEmbedder initialization failed: {e}")
            self._embedder = None
            self._embedder_available = False

        # Initialize Constitution
        try:
            self._constitution = Constitution(global_threshold=ihsan_threshold)
            self._constitution_available = True
            logger.info(f"Constitution initialized: threshold={ihsan_threshold}")
        except Exception as e:
            logger.warning(f"Constitution initialization failed: {e}")
            self._constitution = None
            self._constitution_available = False

        # Initialize DaughterTest
        try:
            self._daughter_test = DaughterTest(auto_start=auto_monitor)
            self._daughter_test_available = True
            logger.info(f"DaughterTest initialized: auto_monitor={auto_monitor}")
        except Exception as e:
            logger.warning(f"DaughterTest initialization failed: {e}")
            self._daughter_test = None
            self._daughter_test_available = False

        # Initialize LocalMerkleDAG
        try:
            self._merkle_dag = LocalMerkleDAG(storage_path=merkle_storage_path)
            self._merkle_dag_available = True
            logger.info(f"LocalMerkleDAG initialized: storage={merkle_storage_path}")
        except Exception as e:
            logger.warning(f"LocalMerkleDAG initialization failed: {e}")
            self._merkle_dag = None
            self._merkle_dag_available = False

        # Track verification history
        self._verification_history: List[SovereigntyVerification] = []

    # =========================================================================
    # CORE VERIFICATION
    # =========================================================================

    async def verify_sovereignty(
        self,
        operation: str,
        scores: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
        add_evidence: bool = True,
    ) -> SovereigntyVerification:
        """
        Perform comprehensive sovereignty verification.

        Verifies constitutional compliance, runs integrity checks,
        and records evidence in the MerkleDAG.

        Args:
            operation: Name of the operation being verified
            scores: Dictionary mapping principle names to scores (0.0 to 1.0)
                   Expected keys: ihsan, sovereignty, transparency, integrity,
                                 determinism, efficiency
            metadata: Additional context for verification
            add_evidence: Whether to record evidence in MerkleDAG

        Returns:
            SovereigntyVerification with comprehensive verification results
        """
        verification_metadata = metadata or {}
        verification_metadata["operation"] = operation
        verification_metadata["timestamp"] = datetime.utcnow().isoformat()

        # Step 1: Constitutional compliance verification
        receipt = self._verify_constitution(operation, scores, verification_metadata)
        constitution_passed = receipt.compliant
        constitution_score = receipt.overall_score

        # Step 2: Integrity verification via DaughterTest
        integrity_checks: List[IntegrityCheck] = []
        violations: List[ViolationAlert] = []
        daughter_test_passed = True

        if self._daughter_test_available and self._daughter_test:
            # Verify the scores are internally consistent
            try:
                check = self._daughter_test.verify_checksum(
                    data=scores,
                    expected_hash=self._compute_scores_hash(scores),
                    metadata={"operation": operation}
                )
                integrity_checks.append(check)
                daughter_test_passed = check.passed
                violations.extend(self._daughter_test.violations[-5:])  # Last 5
            except Exception as e:
                logger.warning(f"DaughterTest integrity check failed: {e}")
                daughter_test_passed = False

        # Step 3: Add evidence to MerkleDAG
        merkle_node_id = ""
        evidence_chain_valid = True

        if add_evidence and self._merkle_dag_available and self._merkle_dag:
            try:
                evidence_data = {
                    "operation": operation,
                    "scores": scores,
                    "constitution_score": constitution_score,
                    "constitution_passed": constitution_passed,
                    "daughter_test_passed": daughter_test_passed,
                    "receipt_id": receipt.receipt_id,
                    "receipt_hash": receipt.integrity_hash,
                }

                node = self._merkle_dag.add_node(
                    data=evidence_data,
                    metadata=verification_metadata
                )
                merkle_node_id = node.node_id

                # Verify DAG integrity
                dag_result = self._merkle_dag.verify_dag()
                evidence_chain_valid = dag_result.valid
            except Exception as e:
                logger.warning(f"MerkleDAG evidence recording failed: {e}")
                evidence_chain_valid = False

        # Combine all verification results
        overall_passed = (
            constitution_passed
            and daughter_test_passed
            and evidence_chain_valid
        )

        verification = SovereigntyVerification(
            passed=overall_passed,
            constitution_score=constitution_score,
            daughter_test_passed=daughter_test_passed,
            merkle_node_id=merkle_node_id,
            receipt=receipt,
            integrity_checks=integrity_checks,
            violations=violations,
            evidence_chain_valid=evidence_chain_valid,
            metadata=verification_metadata,
        )

        # Store in history
        self._verification_history.append(verification)

        return verification

    # =========================================================================
    # EVIDENCE MANAGEMENT
    # =========================================================================

    async def add_evidence(
        self,
        evidence_data: Dict[str, Any],
        parent_ids: Optional[List[str]] = None,
        evidence_type: str = "operation",
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Add evidence to the MerkleDAG chain.

        Creates a new node in the evidence chain with cryptographic
        integrity verification.

        Args:
            evidence_data: Data to record as evidence
            parent_ids: Optional parent node IDs for lineage (None = attach to genesis)
            evidence_type: Category of evidence being recorded
            metadata: Additional context

        Returns:
            Node ID of the created evidence node

        Raises:
            RuntimeError: If MerkleDAG is not available
            ValueError: If parent_ids reference non-existent nodes
        """
        if not self._merkle_dag_available or not self._merkle_dag:
            raise RuntimeError("MerkleDAG not available - cannot add evidence")

        # Prepare evidence with type annotation
        full_evidence = {
            "evidence_type": evidence_type,
            "data": evidence_data,
            "timestamp": datetime.utcnow().isoformat(),
        }

        # Add to DAG
        node = self._merkle_dag.add_node(
            data=full_evidence,
            parent_ids=parent_ids,
            metadata=metadata or {}
        )

        logger.debug(f"Evidence added: node_id={node.node_id[:8]}..., type={evidence_type}")

        return node.node_id

    def get_evidence_chain(self, node_id: str) -> List[EvidenceNode]:
        """
        Get the proof chain from genesis to a specific node.

        Args:
            node_id: Target node ID

        Returns:
            List of EvidenceNode objects forming the proof chain
        """
        if not self._merkle_dag_available or not self._merkle_dag:
            return []

        chain = self._merkle_dag.get_proof_chain(node_id)

        return [
            EvidenceNode(
                node_id=node.node_id,
                timestamp=node.timestamp,
                evidence_type=node.data.get("data", {}).get("evidence_type", "unknown"),
                data_hash=node.hash,
                parent_ids=node.parents,
                merkle_root=node.merkle_root,
            )
            for node in chain
        ]

    def verify_evidence_chain(self, node_id: Optional[str] = None) -> VerificationResult:
        """
        Verify integrity of the evidence chain.

        Args:
            node_id: Optional specific node to verify (None = verify entire DAG)

        Returns:
            VerificationResult with detailed verification status
        """
        if not self._merkle_dag_available or not self._merkle_dag:
            return VerificationResult(
                valid=False,
                total_nodes=0,
                verified_nodes=0,
                tampered_nodes=[],
                orphaned_nodes=[],
                message="MerkleDAG not available"
            )

        if node_id:
            is_valid = self._merkle_dag.verify_node(node_id)
            return VerificationResult(
                valid=is_valid,
                total_nodes=1,
                verified_nodes=1 if is_valid else 0,
                tampered_nodes=[] if is_valid else [node_id],
                orphaned_nodes=[],
                message="Node verified" if is_valid else "Node verification failed"
            )

        return self._merkle_dag.verify_dag()

    # =========================================================================
    # OFFLINE EMBEDDINGS
    # =========================================================================

    def embed_offline(self, text: str) -> List[float]:
        """
        Generate deterministic embeddings without external API calls.

        Uses WinterProofEmbedder's multi-hash approach (SHA-256, SHA3-256, BLAKE3)
        to generate high-dimensional embeddings entirely offline.

        Args:
            text: Input text to embed

        Returns:
            List of floats representing the embedding vector (L2 normalized)

        Raises:
            RuntimeError: If WinterProofEmbedder is not available
        """
        if not self._embedder_available or not self._embedder:
            raise RuntimeError("WinterProofEmbedder not available")

        return self._embedder.embed(text)

    def embed_offline_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple texts offline.

        Args:
            texts: List of input texts

        Returns:
            List of embedding vectors
        """
        if not self._embedder_available or not self._embedder:
            raise RuntimeError("WinterProofEmbedder not available")

        return self._embedder.embed_batch(texts)

    def semantic_search_offline(
        self,
        query: str,
        documents: List[str],
        top_k: int = 5,
    ) -> List[Tuple[int, float, str]]:
        """
        Perform semantic search using offline embeddings.

        Args:
            query: Query text
            documents: List of document texts to search
            top_k: Number of top results to return

        Returns:
            List of (index, similarity_score, document) tuples
        """
        if not self._embedder_available or not self._embedder:
            raise RuntimeError("WinterProofEmbedder not available")

        return self._embedder.semantic_search(query, documents, top_k)

    # =========================================================================
    # DETERMINISM VERIFICATION
    # =========================================================================

    def verify_determinism(
        self,
        func: Callable[..., T],
        inputs: List[Any],
        iterations: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DeterminismReport:
        """
        Verify that a function produces deterministic outputs.

        Runs the function multiple times with the same inputs and verifies
        that all outputs are identical (by hash comparison).

        Args:
            func: Function to test for determinism
            inputs: List of input arguments for the function
            iterations: Number of test iterations (default: 3)
            metadata: Additional context

        Returns:
            DeterminismReport with detailed verification results

        Note:
            For async functions, use verify_determinism_async instead.
        """
        # Compute hash of inputs for tracking
        inputs_json = json.dumps(inputs, sort_keys=True, default=str)
        inputs_hash = hashlib.sha256(
            (DOMAIN_PREFIX + inputs_json).encode('utf-8')
        ).hexdigest()

        output_hashes: List[str] = []

        # Run function multiple times
        for _ in range(iterations):
            try:
                result = func(*inputs)
                result_json = json.dumps(result, sort_keys=True, default=str)
                result_hash = hashlib.sha256(
                    (DOMAIN_PREFIX + result_json).encode('utf-8')
                ).hexdigest()
                output_hashes.append(result_hash)
            except Exception as e:
                # Hash the error for consistency
                error_hash = hashlib.sha256(
                    (DOMAIN_PREFIX + f"error:{type(e).__name__}:{str(e)}").encode('utf-8')
                ).hexdigest()
                output_hashes.append(error_hash)

        # Check if all outputs are identical
        unique_outputs = len(set(output_hashes))
        is_deterministic = unique_outputs == 1

        # Also use DaughterTest if available
        if self._daughter_test_available and self._daughter_test:
            try:
                check = self._daughter_test.verify_determinism(
                    operation=func,
                    inputs=tuple(inputs),
                    iterations=iterations,
                    metadata=metadata
                )
                # DaughterTest might detect non-determinism we missed
                if not check.passed:
                    is_deterministic = False
            except Exception as e:
                logger.warning(f"DaughterTest determinism check failed: {e}")

        return DeterminismReport(
            is_deterministic=is_deterministic,
            iterations=iterations,
            unique_outputs=unique_outputs,
            output_hashes=output_hashes,
            function_name=func.__name__,
            inputs_hash=inputs_hash,
            metadata=metadata or {},
        )

    async def verify_determinism_async(
        self,
        func: Callable[..., T],
        inputs: List[Any],
        iterations: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DeterminismReport:
        """
        Verify determinism for async functions.

        Args:
            func: Async function to test
            inputs: Input arguments
            iterations: Test iterations
            metadata: Additional context

        Returns:
            DeterminismReport with verification results
        """
        import asyncio

        inputs_json = json.dumps(inputs, sort_keys=True, default=str)
        inputs_hash = hashlib.sha256(
            (DOMAIN_PREFIX + inputs_json).encode('utf-8')
        ).hexdigest()

        output_hashes: List[str] = []

        for _ in range(iterations):
            try:
                if asyncio.iscoroutinefunction(func):
                    result = await func(*inputs)
                else:
                    result = func(*inputs)
                result_json = json.dumps(result, sort_keys=True, default=str)
                result_hash = hashlib.sha256(
                    (DOMAIN_PREFIX + result_json).encode('utf-8')
                ).hexdigest()
                output_hashes.append(result_hash)
            except Exception as e:
                error_hash = hashlib.sha256(
                    (DOMAIN_PREFIX + f"error:{type(e).__name__}:{str(e)}").encode('utf-8')
                ).hexdigest()
                output_hashes.append(error_hash)

        unique_outputs = len(set(output_hashes))
        is_deterministic = unique_outputs == 1

        return DeterminismReport(
            is_deterministic=is_deterministic,
            iterations=iterations,
            unique_outputs=unique_outputs,
            output_hashes=output_hashes,
            function_name=func.__name__,
            inputs_hash=inputs_hash,
            metadata=metadata or {},
        )

    # =========================================================================
    # CONSTITUTION ACCESS
    # =========================================================================

    def verify_constitution(
        self,
        operation: str,
        scores: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ComplianceReceipt:
        """
        Verify constitutional compliance without full sovereignty check.

        Args:
            operation: Operation name
            scores: Principle scores
            metadata: Additional context

        Returns:
            ComplianceReceipt from Constitution
        """
        return self._verify_constitution(operation, scores, metadata)

    def verify_constitution_with_enforcement(
        self,
        operation: str,
        scores: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ComplianceReceipt:
        """
        Verify compliance and raise exception if non-compliant.

        Args:
            operation: Operation name
            scores: Principle scores
            metadata: Additional context

        Returns:
            ComplianceReceipt if compliant

        Raises:
            ConstitutionalViolationError: If operation is non-compliant
        """
        if not self._constitution_available or not self._constitution:
            raise RuntimeError("Constitution not available")

        return self._constitution.verify_with_enforcement(operation, scores, metadata)

    # =========================================================================
    # INTEGRITY MONITORING
    # =========================================================================

    def start_integrity_monitoring(self, interval_seconds: int = 60) -> None:
        """
        Start continuous integrity monitoring.

        Args:
            interval_seconds: Check interval
        """
        if self._daughter_test_available and self._daughter_test:
            self._daughter_test.start_monitoring(interval_seconds)
            logger.info(f"Integrity monitoring started: interval={interval_seconds}s")

    def stop_integrity_monitoring(self) -> None:
        """Stop continuous integrity monitoring."""
        if self._daughter_test_available and self._daughter_test:
            self._daughter_test.stop_monitoring()
            logger.info("Integrity monitoring stopped")

    def register_operation_for_monitoring(
        self,
        operation_id: str,
        operation_func: Callable,
        baseline_inputs: Tuple[Any, ...],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Register an operation for continuous monitoring.

        Args:
            operation_id: Unique identifier
            operation_func: Function to monitor
            baseline_inputs: Inputs for baseline determinism
            metadata: Additional context
        """
        if self._daughter_test_available and self._daughter_test:
            self._daughter_test.register_operation(
                operation_id=operation_id,
                operation_func=operation_func,
                baseline_inputs=baseline_inputs,
                metadata=metadata
            )
            logger.debug(f"Operation registered for monitoring: {operation_id}")

    # =========================================================================
    # STATUS & REPORTING
    # =========================================================================

    def get_status(self) -> Dict[str, Any]:
        """
        Get comprehensive status of all sovereignty components.

        Returns:
            Dictionary with component availability and health status
        """
        status = {
            "domain": DOMAIN_PREFIX,
            "ihsan_threshold": self.ihsan_threshold,
            "components": {
                "embedder": {
                    "available": self._embedder_available,
                    "dimension": self.embedding_dim if self._embedder_available else 0,
                },
                "constitution": {
                    "available": self._constitution_available,
                    "threshold": self.ihsan_threshold,
                    "verification_count": len(
                        self._constitution.verification_history
                    ) if self._constitution else 0,
                },
                "daughter_test": {
                    "available": self._daughter_test_available,
                    "monitoring_active": (
                        self._daughter_test.monitoring_active
                        if self._daughter_test else False
                    ),
                },
                "merkle_dag": {
                    "available": self._merkle_dag_available,
                    "node_count": len(
                        self._merkle_dag.nodes
                    ) if self._merkle_dag else 0,
                },
            },
            "verification_history_count": len(self._verification_history),
            "offline_capable": all([
                self._embedder_available,
                self._constitution_available,
                self._daughter_test_available,
                self._merkle_dag_available,
            ]),
        }

        return status

    def get_integrity_report(self) -> Dict[str, Any]:
        """
        Get comprehensive integrity report.

        Returns:
            Dictionary with integrity statistics
        """
        report = {
            "sovereignty_verifications": len(self._verification_history),
            "passed_verifications": sum(
                1 for v in self._verification_history if v.passed
            ),
            "constitution_available": self._constitution_available,
            "daughter_test_report": {},
            "merkle_dag_report": {},
        }

        if self._daughter_test_available and self._daughter_test:
            report["daughter_test_report"] = self._daughter_test.get_integrity_report()

        if self._merkle_dag_available and self._merkle_dag:
            dag_result = self._merkle_dag.verify_dag()
            report["merkle_dag_report"] = {
                "valid": dag_result.valid,
                "total_nodes": dag_result.total_nodes,
                "verified_nodes": dag_result.verified_nodes,
                "tampered_count": len(dag_result.tampered_nodes),
                "orphaned_count": len(dag_result.orphaned_nodes),
            }

        if self._constitution_available and self._constitution:
            report["constitution_summary"] = self._constitution.get_compliance_summary()

        return report

    def get_verification_history(
        self,
        limit: int = 100,
        passed_only: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Get recent verification history.

        Args:
            limit: Maximum number of entries to return
            passed_only: Filter to only passed verifications

        Returns:
            List of verification dictionaries
        """
        history = self._verification_history

        if passed_only:
            history = [v for v in history if v.passed]

        return [v.to_dict() for v in history[-limit:]]

    # =========================================================================
    # PRIVATE METHODS
    # =========================================================================

    def _verify_constitution(
        self,
        operation: str,
        scores: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ComplianceReceipt:
        """Internal constitution verification with fallback."""
        if self._constitution_available and self._constitution:
            return self._constitution.verify(operation, scores, metadata)

        # Fallback: Create minimal receipt
        return self._create_fallback_receipt(operation, scores, metadata)

    def _create_fallback_receipt(
        self,
        operation: str,
        scores: Dict[str, float],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ComplianceReceipt:
        """Create fallback receipt when Constitution is unavailable."""
        import uuid

        # Calculate simple average score
        overall_score = sum(scores.values()) / len(scores) if scores else 0.0
        compliant = overall_score >= self.ihsan_threshold

        receipt_id = str(uuid.uuid4())
        timestamp = datetime.utcnow().isoformat()

        # Create receipt data for hash
        receipt_data = {
            "receipt_id": receipt_id,
            "timestamp": timestamp,
            "domain": DOMAIN_PREFIX,
            "operation": operation,
            "overall_score": overall_score,
            "threshold": self.ihsan_threshold,
            "compliant": compliant,
            "principles_checked": len(scores),
            "principles_passed": sum(1 for s in scores.values() if s >= self.ihsan_threshold),
            "violations": [],
            "metadata": metadata or {},
        }

        integrity_hash = hashlib.sha256(
            (DOMAIN_PREFIX + json.dumps(receipt_data, sort_keys=True)).encode('utf-8')
        ).hexdigest()

        return ComplianceReceipt(
            receipt_id=receipt_id,
            timestamp=timestamp,
            domain=DOMAIN_PREFIX,
            operation=operation,
            overall_score=overall_score,
            threshold=self.ihsan_threshold,
            compliant=compliant,
            principles_checked=len(scores),
            principles_passed=sum(1 for s in scores.values() if s >= self.ihsan_threshold),
            violations=[],
            metadata=metadata or {},
            integrity_hash=integrity_hash,
        )

    def _compute_scores_hash(self, scores: Dict[str, float]) -> str:
        """Compute hash of scores dictionary."""
        scores_json = json.dumps(scores, sort_keys=True)
        return hashlib.sha256(scores_json.encode('utf-8')).hexdigest()


# =============================================================================
# FACTORY FUNCTION
# =============================================================================


def create_sovereignty_bridge(
    embedding_dim: int = DEFAULT_EMBEDDING_DIM,
    ihsan_threshold: float = DEFAULT_IHSAN_THRESHOLD,
    merkle_storage_path: Optional[str] = None,
    auto_monitor: bool = False,
) -> SovereigntyBridge:
    """
    Factory function to create a configured SovereigntyBridge.

    Args:
        embedding_dim: Embedding dimension (must be divisible by 3)
        ihsan_threshold: Minimum Ihsan compliance threshold
        merkle_storage_path: Path for persistent MerkleDAG storage
        auto_monitor: Start continuous integrity monitoring

    Returns:
        Configured SovereigntyBridge instance
    """
    return SovereigntyBridge(
        embedding_dim=embedding_dim,
        ihsan_threshold=ihsan_threshold,
        merkle_storage_path=merkle_storage_path,
        auto_monitor=auto_monitor,
    )


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    "SovereigntyBridge",
    "SovereigntyVerification",
    "DeterminismReport",
    "EvidenceNode",
    "create_sovereignty_bridge",
    "DOMAIN_PREFIX",
    "DEFAULT_IHSAN_THRESHOLD",
    "DEFAULT_EMBEDDING_DIM",
]


# =============================================================================
# DEMO / TESTING
# =============================================================================


async def main():
    """Demo SovereigntyBridge functionality."""
    print("SovereigntyBridge - Unified Sovereignty Integration")
    print("=" * 60)

    # Create bridge
    bridge = create_sovereignty_bridge(
        embedding_dim=384,
        ihsan_threshold=IHSAN_THRESHOLD,  # From core.constants
        auto_monitor=False,
    )

    # Check status
    print("\n1. Component Status:")
    status = bridge.get_status()
    print(f"Offline capable: {status['offline_capable']}")
    for component, info in status["components"].items():
        print(f"  {component}: {'OK' if info['available'] else 'UNAVAILABLE'}")

    # Test offline embedding
    print("\n2. Offline Embedding Test:")
    try:
        text = "BIZRA sovereignty enables offline operation"
        embedding = bridge.embed_offline(text)
        print(f"Text: {text}")
        print(f"Embedding dim: {len(embedding)}")
        print(f"First 5 values: {embedding[:5]}")
    except Exception as e:
        print(f"Embedding failed: {e}")

    # Test sovereignty verification
    print("\n3. Sovereignty Verification Test:")
    scores = {
        "ihsan": 0.98,
        "sovereignty": 1.0,
        "transparency": 0.97,
        "integrity": 1.0,
        "determinism": 1.0,
        "efficiency": 0.92,
    }

    verification = await bridge.verify_sovereignty(
        operation="test_embedding",
        scores=scores,
        metadata={"test": True},
    )

    print(f"Passed: {verification.passed}")
    print(f"Constitution score: {verification.constitution_score:.4f}")
    print(f"DaughterTest passed: {verification.daughter_test_passed}")
    print(f"MerkleDAG node: {verification.merkle_node_id[:8]}...")
    print(f"Receipt ID: {verification.receipt.receipt_id[:8]}...")

    # Test determinism verification
    print("\n4. Determinism Verification Test:")

    def deterministic_func(x: int) -> int:
        return x * 2

    report = bridge.verify_determinism(
        func=deterministic_func,
        inputs=[42],
        iterations=3,
    )

    print(f"Function: {report.function_name}")
    print(f"Deterministic: {report.is_deterministic}")
    print(f"Unique outputs: {report.unique_outputs}/{report.iterations}")

    # Test evidence chain
    print("\n5. Evidence Chain Test:")
    try:
        node_id = await bridge.add_evidence(
            evidence_data={"test": "evidence", "value": 123},
            evidence_type="test_operation",
        )
        print(f"Evidence added: {node_id[:8]}...")

        chain = bridge.get_evidence_chain(node_id)
        print(f"Proof chain length: {len(chain)}")
    except Exception as e:
        print(f"Evidence chain test failed: {e}")

    # Get integrity report
    print("\n6. Integrity Report:")
    report = bridge.get_integrity_report()
    print(f"Sovereignty verifications: {report['sovereignty_verifications']}")
    print(f"Passed: {report['passed_verifications']}")

    if report.get("merkle_dag_report"):
        print(f"MerkleDAG valid: {report['merkle_dag_report']['valid']}")
        print(f"MerkleDAG nodes: {report['merkle_dag_report']['total_nodes']}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
