# BIZRA Unified Error Hierarchy v1.0
# Professional-grade exception handling for all BIZRA components
# Implements: Error taxonomy, context preservation, recovery hints

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Any
import traceback
import json
import logging

logger = logging.getLogger("BIZRA.Errors")


class ErrorSeverity(Enum):
    """Error severity levels"""
    DEBUG = "debug"         # Development-only information
    INFO = "info"           # Informational, no action needed
    WARNING = "warning"     # Potential issue, operation continues
    ERROR = "error"         # Operation failed, but system stable
    CRITICAL = "critical"   # System stability at risk
    FATAL = "fatal"         # System cannot continue


class ErrorCategory(Enum):
    """Error categories for classification"""
    CONFIGURATION = "configuration"
    DATA = "data"
    NETWORK = "network"
    COMPUTATION = "computation"
    RESOURCE = "resource"
    VALIDATION = "validation"
    SECURITY = "security"
    INTEGRATION = "integration"


class RecoveryAction(Enum):
    """Suggested recovery actions"""
    RETRY = "retry"                 # Retry the operation
    RECONFIGURE = "reconfigure"     # Check configuration
    SCALE_DOWN = "scale_down"       # Reduce batch size / load
    FAILOVER = "failover"           # Switch to backup
    MANUAL = "manual"               # Requires manual intervention
    SKIP = "skip"                   # Skip and continue
    ABORT = "abort"                 # Abort current operation
    RESTART = "restart"             # Restart service


@dataclass
class ErrorContext:
    """Rich error context for debugging"""
    operation: str
    component: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    snr_at_failure: float = 0.0
    query_id: Optional[str] = None
    batch_id: Optional[str] = None
    user_context: Dict = field(default_factory=dict)
    stack_trace: str = ""
    related_errors: List[str] = field(default_factory=list)


class BIZRAError(Exception):
    """
    Base exception class for all BIZRA errors.

    Features:
    - Rich context preservation
    - SNR tracking at failure point
    - Recovery suggestions
    - Serialization support
    """

    def __init__(
        self,
        message: str,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        category: ErrorCategory = ErrorCategory.COMPUTATION,
        recoverable: bool = True,
        recovery_action: RecoveryAction = RecoveryAction.RETRY,
        context: Optional[ErrorContext] = None,
        snr: float = 0.0,
        cause: Optional[Exception] = None
    ):
        self.message = message
        self.severity = severity
        self.category = category
        self.recoverable = recoverable
        self.recovery_action = recovery_action
        self.context = context or ErrorContext(operation="unknown", component="unknown")
        self.snr = snr
        self.cause = cause
        self.timestamp = datetime.now().isoformat()

        # Capture stack trace
        if not self.context.stack_trace:
            self.context.stack_trace = traceback.format_exc()

        super().__init__(self.message)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize error to dictionary"""
        return {
            "error_type": self.__class__.__name__,
            "message": self.message,
            "severity": self.severity.value,
            "category": self.category.value,
            "recoverable": self.recoverable,
            "recovery_action": self.recovery_action.value,
            "snr": self.snr,
            "timestamp": self.timestamp,
            "context": {
                "operation": self.context.operation,
                "component": self.context.component,
                "timestamp": self.context.timestamp,
                "snr_at_failure": self.context.snr_at_failure,
                "query_id": self.context.query_id,
                "batch_id": self.context.batch_id,
            },
            "cause": str(self.cause) if self.cause else None
        }

    def to_json(self) -> str:
        """Serialize error to JSON"""
        return json.dumps(self.to_dict(), indent=2)

    def log(self, logger_instance: Optional[logging.Logger] = None):
        """Log the error with appropriate level"""
        log = logger_instance or logger

        level_map = {
            ErrorSeverity.DEBUG: logging.DEBUG,
            ErrorSeverity.INFO: logging.INFO,
            ErrorSeverity.WARNING: logging.WARNING,
            ErrorSeverity.ERROR: logging.ERROR,
            ErrorSeverity.CRITICAL: logging.CRITICAL,
            ErrorSeverity.FATAL: logging.CRITICAL
        }

        log.log(
            level_map[self.severity],
            f"[{self.category.value}] {self.message} "
            f"(SNR: {self.snr:.4f}, Recovery: {self.recovery_action.value})"
        )

    def __str__(self) -> str:
        return (
            f"{self.__class__.__name__}: {self.message} "
            f"[{self.severity.value}/{self.category.value}] "
            f"(SNR: {self.snr:.4f})"
        )


# ============================================================================
# SNR-Related Errors
# ============================================================================

class SNRError(BIZRAError):
    """Base class for SNR-related errors"""

    def __init__(self, message: str, snr: float, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.COMPUTATION,
            snr=snr,
            **kwargs
        )


class SNRBelowThresholdError(SNRError):
    """Raised when SNR falls below acceptable threshold"""

    def __init__(self, snr: float, threshold: float, **kwargs):
        super().__init__(
            f"SNR {snr:.4f} below threshold {threshold:.4f}",
            snr=snr,
            severity=ErrorSeverity.WARNING,
            recoverable=True,
            recovery_action=RecoveryAction.RETRY,
            **kwargs
        )
        self.threshold = threshold


class IhsanNotAchievedError(SNRError):
    """Raised when Ihsān threshold (0.99) not achieved"""

    def __init__(self, snr: float, **kwargs):
        super().__init__(
            f"Ihsān not achieved. SNR: {snr:.4f} < 0.99",
            snr=snr,
            severity=ErrorSeverity.INFO,
            recoverable=True,
            recovery_action=RecoveryAction.RETRY,
            **kwargs
        )


class SNRCalculationError(SNRError):
    """Raised when SNR calculation fails"""

    def __init__(self, message: str, cause: Optional[Exception] = None, **kwargs):
        super().__init__(
            f"SNR calculation failed: {message}",
            snr=0.0,
            severity=ErrorSeverity.ERROR,
            recoverable=False,
            recovery_action=RecoveryAction.MANUAL,
            cause=cause,
            **kwargs
        )


# ============================================================================
# Graph-Related Errors
# ============================================================================

class GraphError(BIZRAError):
    """Base class for graph-related errors"""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.DATA,
            **kwargs
        )


class GraphIntegrityError(GraphError):
    """Raised when graph integrity check fails"""

    def __init__(self, message: str, node_count: int = 0, edge_count: int = 0, **kwargs):
        super().__init__(
            f"Graph integrity violation: {message}",
            severity=ErrorSeverity.CRITICAL,
            recoverable=False,
            recovery_action=RecoveryAction.RESTART,
            **kwargs
        )
        self.node_count = node_count
        self.edge_count = edge_count


class GraphTraversalError(GraphError):
    """Raised when graph traversal fails"""

    def __init__(self, message: str, source: Optional[str] = None,
                 target: Optional[str] = None, **kwargs):
        super().__init__(
            f"Graph traversal failed: {message}",
            severity=ErrorSeverity.ERROR,
            recoverable=True,
            recovery_action=RecoveryAction.SKIP,
            **kwargs
        )
        self.source = source
        self.target = target


class NodeNotFoundError(GraphError):
    """Raised when a node is not found in the graph"""

    def __init__(self, node_id: str, **kwargs):
        super().__init__(
            f"Node not found: {node_id}",
            severity=ErrorSeverity.WARNING,
            recoverable=True,
            recovery_action=RecoveryAction.SKIP,
            **kwargs
        )
        self.node_id = node_id


# ============================================================================
# Retrieval-Related Errors
# ============================================================================

class RetrievalError(BIZRAError):
    """Base class for retrieval-related errors"""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.COMPUTATION,
            **kwargs
        )


class EmbeddingError(RetrievalError):
    """Raised when embedding generation fails"""

    def __init__(self, message: str, text_length: int = 0, **kwargs):
        super().__init__(
            f"Embedding generation failed: {message}",
            severity=ErrorSeverity.ERROR,
            recoverable=True,
            recovery_action=RecoveryAction.RETRY,
            **kwargs
        )
        self.text_length = text_length


class VectorSearchError(RetrievalError):
    """Raised when vector search fails"""

    def __init__(self, message: str, query_dim: int = 0, **kwargs):
        super().__init__(
            f"Vector search failed: {message}",
            severity=ErrorSeverity.ERROR,
            recoverable=True,
            recovery_action=RecoveryAction.FAILOVER,
            **kwargs
        )
        self.query_dim = query_dim


class BIZRAIndexError(RetrievalError):
    """Raised when index operation fails.

    Named BIZRAIndexError to avoid shadowing Python's builtin IndexError.
    Standing on Giants: Liskov (1987) — Substitution Principle preservation.
    """

    def __init__(self, message: str, index_type: str = "unknown", **kwargs):
        super().__init__(
            f"Index operation failed ({index_type}): {message}",
            severity=ErrorSeverity.ERROR,
            recoverable=False,
            recovery_action=RecoveryAction.RESTART,
            **kwargs
        )
        self.index_type = index_type


# Preserve backward compatibility without shadowing builtins
IndexError_ = BIZRAIndexError


# ============================================================================
# LLM-Related Errors
# ============================================================================

class LLMError(BIZRAError):
    """Base class for LLM-related errors"""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.INTEGRATION,
            **kwargs
        )


class LLMConnectionError(LLMError):
    """Raised when LLM backend connection fails"""

    def __init__(self, backend: str, endpoint: str, **kwargs):
        super().__init__(
            f"Cannot connect to {backend} at {endpoint}",
            severity=ErrorSeverity.ERROR,
            recoverable=True,
            recovery_action=RecoveryAction.FAILOVER,
            **kwargs
        )
        self.backend = backend
        self.endpoint = endpoint


class LLMTimeoutError(LLMError):
    """Raised when LLM request times out"""

    def __init__(self, backend: str, timeout_seconds: float, **kwargs):
        super().__init__(
            f"{backend} request timed out after {timeout_seconds}s",
            severity=ErrorSeverity.WARNING,
            recoverable=True,
            recovery_action=RecoveryAction.RETRY,
            **kwargs
        )
        self.backend = backend
        self.timeout_seconds = timeout_seconds


class LLMResponseError(LLMError):
    """Raised when LLM response is invalid"""

    def __init__(self, backend: str, status_code: int, **kwargs):
        super().__init__(
            f"{backend} returned error status: {status_code}",
            severity=ErrorSeverity.ERROR,
            recoverable=True,
            recovery_action=RecoveryAction.RETRY,
            **kwargs
        )
        self.backend = backend
        self.status_code = status_code


class CircuitBreakerOpenError(LLMError):
    """Raised when circuit breaker is open"""

    def __init__(self, service: str, **kwargs):
        super().__init__(
            f"Circuit breaker OPEN for {service}. Service unavailable.",
            severity=ErrorSeverity.WARNING,
            recoverable=True,
            recovery_action=RecoveryAction.FAILOVER,
            **kwargs
        )
        self.service = service


# ============================================================================
# Configuration Errors
# ============================================================================

class ConfigurationError(BIZRAError):
    """Base class for configuration errors"""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.CONFIGURATION,
            **kwargs
        )


class MissingConfigError(ConfigurationError):
    """Raised when required configuration is missing"""

    def __init__(self, config_key: str, **kwargs):
        super().__init__(
            f"Missing required configuration: {config_key}",
            severity=ErrorSeverity.ERROR,
            recoverable=False,
            recovery_action=RecoveryAction.RECONFIGURE,
            **kwargs
        )
        self.config_key = config_key


class InvalidConfigError(ConfigurationError):
    """Raised when configuration value is invalid"""

    def __init__(self, config_key: str, value: Any, expected: str, **kwargs):
        super().__init__(
            f"Invalid configuration for {config_key}: {value} (expected: {expected})",
            severity=ErrorSeverity.ERROR,
            recoverable=False,
            recovery_action=RecoveryAction.RECONFIGURE,
            **kwargs
        )
        self.config_key = config_key
        self.value = value
        self.expected = expected


# ============================================================================
# Resource Errors
# ============================================================================

class ResourceError(BIZRAError):
    """Base class for resource-related errors"""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.RESOURCE,
            **kwargs
        )


class BIZRAMemoryError(ResourceError):
    """Raised when memory is exhausted.

    Named BIZRAMemoryError to avoid shadowing Python's builtin MemoryError.
    Standing on Giants: Liskov (1987) — Substitution Principle preservation.
    """

    def __init__(self, required_mb: float, available_mb: float, **kwargs):
        super().__init__(
            f"Insufficient memory: {required_mb:.1f}MB required, {available_mb:.1f}MB available",
            severity=ErrorSeverity.CRITICAL,
            recoverable=True,
            recovery_action=RecoveryAction.SCALE_DOWN,
            **kwargs
        )
        self.required_mb = required_mb
        self.available_mb = available_mb


# Preserve backward compatibility without shadowing builtins
MemoryError_ = BIZRAMemoryError


class GPUMemoryError(ResourceError):
    """Raised when GPU memory is exhausted"""

    def __init__(self, required_mb: float, available_mb: float, **kwargs):
        super().__init__(
            f"Insufficient GPU memory: {required_mb:.1f}MB required, {available_mb:.1f}MB available",
            severity=ErrorSeverity.ERROR,
            recoverable=True,
            recovery_action=RecoveryAction.SCALE_DOWN,
            **kwargs
        )
        self.required_mb = required_mb
        self.available_mb = available_mb


class BIZRAFileNotFoundError(ResourceError):
    """Raised when required file is not found.

    Named BIZRAFileNotFoundError to avoid shadowing Python's builtin FileNotFoundError.
    Standing on Giants: Liskov (1987) — Substitution Principle preservation.
    """

    def __init__(self, file_path: str, **kwargs):
        super().__init__(
            f"File not found: {file_path}",
            severity=ErrorSeverity.ERROR,
            recoverable=False,
            recovery_action=RecoveryAction.MANUAL,
            **kwargs
        )
        self.file_path = file_path


# Preserve backward compatibility without shadowing builtins
FileNotFoundError_ = BIZRAFileNotFoundError


# ============================================================================
# Validation Errors
# ============================================================================

class ValidationError(BIZRAError):
    """Base class for validation errors"""

    def __init__(self, message: str, **kwargs):
        super().__init__(
            message,
            category=ErrorCategory.VALIDATION,
            **kwargs
        )


class POIVerificationError(ValidationError):
    """Raised when POI attestation verification fails"""

    def __init__(self, entry_id: str, reason: str, **kwargs):
        super().__init__(
            f"POI verification failed for {entry_id}: {reason}",
            severity=ErrorSeverity.CRITICAL,
            recoverable=False,
            recovery_action=RecoveryAction.MANUAL,
            **kwargs
        )
        self.entry_id = entry_id
        self.reason = reason


class MerkleVerificationError(ValidationError):
    """Raised when Merkle hash verification fails"""

    def __init__(self, expected_hash: str, actual_hash: str, **kwargs):
        super().__init__(
            f"Merkle verification failed: expected {expected_hash[:16]}..., got {actual_hash[:16]}...",
            severity=ErrorSeverity.CRITICAL,
            recoverable=False,
            recovery_action=RecoveryAction.MANUAL,
            **kwargs
        )
        self.expected_hash = expected_hash
        self.actual_hash = actual_hash


# ============================================================================
# Error Handler
# ============================================================================

class BIZRAErrorHandler:
    """Centralized error handler for BIZRA system"""

    def __init__(self):
        self.error_log: List[BIZRAError] = []
        self.max_log_size = 1000

    def handle(self, error: BIZRAError) -> Optional[Any]:
        """
        Handle an error with appropriate action.

        Returns recovery result if recoverable, None otherwise.
        """
        # Log error
        self.error_log.append(error)
        if len(self.error_log) > self.max_log_size:
            self.error_log = self.error_log[-500:]

        # Log to logging system
        error.log()

        # Handle based on severity
        if error.severity == ErrorSeverity.FATAL:
            logger.critical(f"FATAL ERROR: {error.message}")
            raise error

        if not error.recoverable:
            logger.error(f"Non-recoverable error: {error.message}")
            return None

        # Return recovery hint
        return error.recovery_action

    def get_error_statistics(self) -> Dict[str, Any]:
        """Get error statistics"""
        if not self.error_log:
            return {"total_errors": 0}

        by_severity = {}
        by_category = {}

        for error in self.error_log:
            sev = error.severity.value
            cat = error.category.value

            by_severity[sev] = by_severity.get(sev, 0) + 1
            by_category[cat] = by_category.get(cat, 0) + 1

        return {
            "total_errors": len(self.error_log),
            "by_severity": by_severity,
            "by_category": by_category,
            "recoverable_count": sum(1 for e in self.error_log if e.recoverable),
            "avg_snr_at_failure": sum(e.snr for e in self.error_log) / len(self.error_log)
        }


# Global error handler instance
error_handler = BIZRAErrorHandler()


# Convenience function
def handle_error(error: BIZRAError) -> Optional[RecoveryAction]:
    """Handle error using global handler"""
    return error_handler.handle(error)


# Example usage
if __name__ == "__main__":
    print("=" * 60)
    print("BIZRA Unified Error Hierarchy v1.0")
    print("=" * 60)

    # Demonstrate error creation and handling
    errors = [
        SNRBelowThresholdError(snr=0.87, threshold=0.95),
        IhsanNotAchievedError(snr=0.96),
        GraphIntegrityError("Missing edges in subgraph"),
        LLMConnectionError("Ollama", "http://localhost:11434"),
        MissingConfigError("OPENAI_API_KEY"),
    ]

    print("\n--- Error Demonstrations ---\n")

    for error in errors:
        print(f"Error: {error}")
        print(f"  Severity: {error.severity.value}")
        print(f"  Category: {error.category.value}")
        print(f"  Recoverable: {error.recoverable}")
        print(f"  Recovery: {error.recovery_action.value}")
        print()

        handle_error(error)

    print("--- Error Statistics ---")
    stats = error_handler.get_error_statistics()
    print(f"Total errors: {stats['total_errors']}")
    print(f"By severity: {stats['by_severity']}")
    print(f"By category: {stats['by_category']}")
    print(f"Recoverable: {stats['recoverable_count']}")
