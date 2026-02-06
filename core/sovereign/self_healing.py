"""
BIZRA Self-Healing Engine
═══════════════════════════════════════════════════════════════════════════════

Autonomous error detection and recovery with Guardian Council integration.

Recovery Types:
- RETRY: Retry with exponential backoff
- REINSTALL_DEP: pip install missing module
- ROLLBACK: Rollback to checkpoint
- RESTART_SERVICE: Restart failed daemon
- ESCALATE: Human intervention required

Security Violations (NEVER Auto-Recover):
- SANDBOX_VIOLATION
- REJECT_SIGNATURE
- REJECT_POLICY_MISMATCH
- IHSAN_BELOW_MIN
- SNR_BELOW_MIN

Created: 2026-02-01 | BIZRA Remediation v2.2.1
Principle: "Security violations escalate. Other errors self-heal."
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set

logger = logging.getLogger(__name__)


class RecoveryAction(str, Enum):
    """Recovery actions for self-healing."""
    RETRY = "retry"
    REINSTALL_DEP = "reinstall_dep"
    ROLLBACK = "rollback"
    RESTART_SERVICE = "restart_service"
    ESCALATE = "escalate"
    IGNORE = "ignore"


class ErrorSeverity(str, Enum):
    """Error severity levels."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"
    SECURITY = "security"


@dataclass
class ErrorContext:
    """Context about an error occurrence."""
    error_type: str
    message: str
    source: str  # file:line or module
    stack_trace: Optional[str] = None
    tool_name: Optional[str] = None
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RecoveryResult:
    """Result of a recovery attempt."""
    action: RecoveryAction
    success: bool
    message: str
    retries_remaining: int = 0
    next_action: Optional[RecoveryAction] = None


# Security violation codes that NEVER auto-recover
SECURITY_VIOLATIONS: Set[str] = {
    "SANDBOX_VIOLATION",
    "REJECT_SIGNATURE",
    "REJECT_POLICY_MISMATCH",
    "REJECT_IHSAN_BELOW_MIN",
    "REJECT_SNR_BELOW_MIN",
    "IHSAN_BELOW_MIN",
    "SNR_BELOW_MIN",
    "REJECT_FATE_VIOLATION",
    "REJECT_INVARIANT_FAILED",
}

# Error patterns and their recovery strategies
ERROR_RECOVERY_MAP: Dict[str, Dict[str, Any]] = {
    # Missing module errors
    r"ModuleNotFoundError.*No module named ['\"](\w+)['\"]": {
        "action": RecoveryAction.REINSTALL_DEP,
        "severity": ErrorSeverity.MEDIUM,
        "max_retries": 2,
    },
    r"ImportError.*cannot import name": {
        "action": RecoveryAction.REINSTALL_DEP,
        "severity": ErrorSeverity.MEDIUM,
        "max_retries": 2,
    },

    # Connection errors
    r"ConnectionRefusedError": {
        "action": RecoveryAction.RETRY,
        "severity": ErrorSeverity.MEDIUM,
        "max_retries": 5,
        "backoff_base": 2.0,
    },
    r"ConnectionResetError": {
        "action": RecoveryAction.RETRY,
        "severity": ErrorSeverity.MEDIUM,
        "max_retries": 3,
    },
    r"TimeoutError": {
        "action": RecoveryAction.RETRY,
        "severity": ErrorSeverity.LOW,
        "max_retries": 3,
    },

    # Service errors
    r"ServiceUnavailable": {
        "action": RecoveryAction.RESTART_SERVICE,
        "severity": ErrorSeverity.HIGH,
        "max_retries": 1,
    },

    # File system errors
    r"FileNotFoundError": {
        "action": RecoveryAction.ESCALATE,
        "severity": ErrorSeverity.HIGH,
        "max_retries": 0,
    },
    r"PermissionError": {
        "action": RecoveryAction.ESCALATE,
        "severity": ErrorSeverity.HIGH,
        "max_retries": 0,
    },

    # Transient errors
    r"OSError.*Temporary failure": {
        "action": RecoveryAction.RETRY,
        "severity": ErrorSeverity.LOW,
        "max_retries": 3,
    },

    # Out of resources
    r"MemoryError": {
        "action": RecoveryAction.RESTART_SERVICE,
        "severity": ErrorSeverity.CRITICAL,
        "max_retries": 1,
    },
}


class SelfHealingEngine:
    """
    Autonomous error detection and recovery engine.

    Integrates with Guardian Council for validation of recovery actions.
    Security violations are NEVER auto-recovered.
    """

    def __init__(
        self,
        guardian_validator: Optional[Callable[[RecoveryAction, ErrorContext], bool]] = None,
        max_consecutive_failures: int = 5,
    ):
        self.guardian_validator = guardian_validator
        self.max_consecutive_failures = max_consecutive_failures
        self._error_history: List[ErrorContext] = []
        self._retry_counts: Dict[str, int] = {}
        self._consecutive_failures = 0

    def is_security_violation(self, error: ErrorContext) -> bool:
        """Check if error is a security violation (never auto-recover)."""
        # Check error type directly
        if error.error_type in SECURITY_VIOLATIONS:
            return True

        # Check message for security codes
        for code in SECURITY_VIOLATIONS:
            if code in error.message:
                return True

        # Check metadata
        if error.metadata.get("code") in SECURITY_VIOLATIONS:
            return True

        return False

    def classify_error(self, error: ErrorContext) -> Dict[str, Any]:
        """Classify error and determine recovery strategy."""
        # Security violations are always escalated
        if self.is_security_violation(error):
            return {
                "action": RecoveryAction.ESCALATE,
                "severity": ErrorSeverity.SECURITY,
                "max_retries": 0,
                "reason": "Security violation - human intervention required",
            }

        # Match against known error patterns
        for pattern, strategy in ERROR_RECOVERY_MAP.items():
            if re.search(pattern, error.message) or re.search(pattern, error.error_type):
                return dict(strategy)

        # Unknown errors escalate by default
        return {
            "action": RecoveryAction.ESCALATE,
            "severity": ErrorSeverity.HIGH,
            "max_retries": 0,
            "reason": "Unknown error type - escalating for review",
        }

    async def handle_error(self, error: ErrorContext) -> RecoveryResult:
        """
        Handle an error and attempt recovery.

        Returns RecoveryResult with action taken and success status.
        """
        self._error_history.append(error)

        # Check for consecutive failure limit
        self._consecutive_failures += 1
        if self._consecutive_failures >= self.max_consecutive_failures:
            logger.error(f"Max consecutive failures ({self.max_consecutive_failures}) reached - escalating")
            return RecoveryResult(
                action=RecoveryAction.ESCALATE,
                success=False,
                message="Too many consecutive failures - human intervention required",
            )

        # Classify the error
        strategy = self.classify_error(error)
        action = strategy["action"]
        severity = strategy.get("severity", ErrorSeverity.MEDIUM)
        max_retries = strategy.get("max_retries", 0)

        # Log the error
        logger.warning(f"[{severity.value.upper()}] {error.error_type}: {error.message}")

        # Check retry count
        error_key = f"{error.error_type}:{error.source}"
        current_retries = self._retry_counts.get(error_key, 0)

        if current_retries >= max_retries:
            logger.info(f"Max retries ({max_retries}) exceeded for {error_key}")
            return RecoveryResult(
                action=RecoveryAction.ESCALATE,
                success=False,
                message=f"Max retries exceeded for {error.error_type}",
            )

        # Security violations never auto-recover
        if action == RecoveryAction.ESCALATE or severity == ErrorSeverity.SECURITY:
            return RecoveryResult(
                action=RecoveryAction.ESCALATE,
                success=False,
                message=strategy.get("reason", "Escalated for human review"),
            )

        # Guardian Council validation (if available)
        if self.guardian_validator:
            if not self.guardian_validator(action, error):
                logger.info("Guardian Council rejected recovery action")
                return RecoveryResult(
                    action=RecoveryAction.ESCALATE,
                    success=False,
                    message="Recovery action rejected by Guardian Council",
                )

        # Execute recovery action
        self._retry_counts[error_key] = current_retries + 1
        result = await self._execute_recovery(action, error, strategy)

        # Reset consecutive failures on success
        if result.success:
            self._consecutive_failures = 0

        return result

    async def _execute_recovery(
        self,
        action: RecoveryAction,
        error: ErrorContext,
        strategy: Dict[str, Any]
    ) -> RecoveryResult:
        """Execute a recovery action."""

        if action == RecoveryAction.RETRY:
            return await self._action_retry(error, strategy)

        elif action == RecoveryAction.REINSTALL_DEP:
            return await self._action_reinstall_dep(error, strategy)

        elif action == RecoveryAction.RESTART_SERVICE:
            return await self._action_restart_service(error, strategy)

        elif action == RecoveryAction.ROLLBACK:
            return await self._action_rollback(error, strategy)

        else:
            return RecoveryResult(
                action=RecoveryAction.ESCALATE,
                success=False,
                message=f"Unknown recovery action: {action}",
            )

    async def _action_retry(
        self, error: ErrorContext, strategy: Dict[str, Any]
    ) -> RecoveryResult:
        """Retry with exponential backoff."""
        backoff_base = strategy.get("backoff_base", 1.5)
        max_retries = strategy.get("max_retries", 3)
        error_key = f"{error.error_type}:{error.source}"
        current_retry = self._retry_counts.get(error_key, 1)

        wait_time = backoff_base ** current_retry
        logger.info(f"Retry {current_retry}/{max_retries} - waiting {wait_time:.1f}s")

        await asyncio.sleep(wait_time)

        return RecoveryResult(
            action=RecoveryAction.RETRY,
            success=True,
            message=f"Ready for retry (attempt {current_retry})",
            retries_remaining=max_retries - current_retry,
        )

    async def _action_reinstall_dep(
        self, error: ErrorContext, strategy: Dict[str, Any]
    ) -> RecoveryResult:
        """Attempt to reinstall a missing dependency."""
        # Extract module name from error message
        match = re.search(r"No module named ['\"](\w+)['\"]", error.message)
        if not match:
            return RecoveryResult(
                action=RecoveryAction.REINSTALL_DEP,
                success=False,
                message="Could not extract module name from error",
                next_action=RecoveryAction.ESCALATE,
            )

        module_name = match.group(1)
        logger.info(f"Attempting to install missing module: {module_name}")

        # Note: In production, this would actually run pip install
        # For safety, we just report what would be done
        return RecoveryResult(
            action=RecoveryAction.REINSTALL_DEP,
            success=True,
            message=f"Would run: pip install {module_name}",
            retries_remaining=strategy.get("max_retries", 2) - 1,
        )

    async def _action_restart_service(
        self, error: ErrorContext, strategy: Dict[str, Any]
    ) -> RecoveryResult:
        """Attempt to restart a failed service."""
        service_name = error.metadata.get("service", "unknown")
        logger.info(f"Attempting to restart service: {service_name}")

        # Note: In production, this would actually restart the service
        return RecoveryResult(
            action=RecoveryAction.RESTART_SERVICE,
            success=True,
            message=f"Would restart service: {service_name}",
            retries_remaining=0,
        )

    async def _action_rollback(
        self, error: ErrorContext, strategy: Dict[str, Any]
    ) -> RecoveryResult:
        """Rollback to a checkpoint."""
        checkpoint = error.metadata.get("checkpoint", "latest")
        logger.info(f"Rolling back to checkpoint: {checkpoint}")

        return RecoveryResult(
            action=RecoveryAction.ROLLBACK,
            success=True,
            message=f"Would rollback to: {checkpoint}",
            retries_remaining=0,
        )

    def get_error_history(self) -> List[ErrorContext]:
        """Get recent error history."""
        return self._error_history.copy()

    def clear_retry_counts(self):
        """Clear retry counts (e.g., after successful recovery)."""
        self._retry_counts.clear()
        self._consecutive_failures = 0

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        return {
            "total_errors": len(self._error_history),
            "consecutive_failures": self._consecutive_failures,
            "active_retry_counts": len(self._retry_counts),
            "security_violations": sum(
                1 for e in self._error_history if self.is_security_violation(e)
            ),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# SINGLETON
# ═══════════════════════════════════════════════════════════════════════════════

_engine_instance: Optional[SelfHealingEngine] = None


def get_self_healing_engine() -> SelfHealingEngine:
    """Get the singleton self-healing engine."""
    global _engine_instance
    if _engine_instance is None:
        _engine_instance = SelfHealingEngine()
    return _engine_instance


# ═══════════════════════════════════════════════════════════════════════════════
# DEMO
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import asyncio

    async def demo():
        print("=" * 70)
        print("BIZRA Self-Healing Engine — Demo")
        print("=" * 70)

        engine = SelfHealingEngine()

        # Test 1: Security violation (should escalate)
        print("\n[Test 1] Security Violation (SANDBOX_VIOLATION)")
        error1 = ErrorContext(
            error_type="SANDBOX_VIOLATION",
            message="Refusing execution: BIZRA_SANDBOX not set",
            source="sandbox/inference_worker.py:487",
            metadata={"code": "SANDBOX_VIOLATION", "fatal": True}
        )
        result1 = await engine.handle_error(error1)
        print(f"  Action: {result1.action.value}")
        print(f"  Success: {result1.success}")
        print(f"  Message: {result1.message}")

        # Test 2: Missing module (should attempt reinstall)
        print("\n[Test 2] Missing Module (ModuleNotFoundError)")
        error2 = ErrorContext(
            error_type="ModuleNotFoundError",
            message="No module named 'numpy'",
            source="core/inference/gateway.py:42",
        )
        result2 = await engine.handle_error(error2)
        print(f"  Action: {result2.action.value}")
        print(f"  Success: {result2.success}")
        print(f"  Message: {result2.message}")

        # Test 3: Connection error (should retry)
        print("\n[Test 3] Connection Error (should retry)")
        error3 = ErrorContext(
            error_type="ConnectionRefusedError",
            message="Connection refused to localhost:11434",
            source="core/inference/gateway.py:380",
        )
        result3 = await engine.handle_error(error3)
        print(f"  Action: {result3.action.value}")
        print(f"  Success: {result3.success}")
        print(f"  Message: {result3.message}")
        print(f"  Retries remaining: {result3.retries_remaining}")

        # Stats
        print("\n[Stats]")
        stats = engine.get_stats()
        print(f"  Total errors: {stats['total_errors']}")
        print(f"  Security violations: {stats['security_violations']}")
        print(f"  Consecutive failures: {stats['consecutive_failures']}")

        print("\n" + "=" * 70)
        print("Demo complete")

    asyncio.run(demo())
