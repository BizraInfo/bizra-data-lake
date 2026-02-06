"""
Secure Audit Mixin — Tamper-Evident Logging Integration
========================================================

Provides a mixin class that adds tamper-evident audit logging capabilities
to any class. Designed for integration with existing audit infrastructure
like ShadowDeployer.

Usage:
    from core.sovereign.secure_audit_mixin import SecureAuditMixin

    class MyService(SecureAuditMixin):
        def __init__(self):
            self.initialize_secure_audit()

        def perform_action(self):
            self.secure_log("action_performed", {"action": "something"})

Standing on Giants:
- Merkle (1979): Hash chains
- RFC 2104 (1997): HMAC
- GoF (1994): Mixin pattern

Genesis Strict Synthesis v2.2.2
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from core.sovereign.tamper_evident_log import (
    AuditKeyManager,
    TamperEvidentEntry,
    TamperEvidentLog,
    TamperingReport,
    VerificationStatus,
    create_audit_log,
)

logger = logging.getLogger(__name__)


class SecureAuditMixin:
    """
    Mixin class providing tamper-evident audit logging capabilities.

    Add this mixin to any class to get cryptographically secure audit logging
    with HMAC signatures and hash chain integrity.

    Example:
        class ShadowDeployer(SecureAuditMixin):
            def __init__(self, ...):
                self.initialize_secure_audit(persist_path=Path("audit.log"))

            def deploy_shadow(self, ...):
                self.secure_log("deploy_shadow_start", {
                    "hypothesis_id": hypothesis.id,
                    "deployment_id": deployment.deployment_id,
                })
                # ... deployment logic ...
                self.secure_log("deploy_shadow_complete", {...})

    Attributes:
        _secure_audit_log: TamperEvidentLog instance
        _secure_audit_key_manager: AuditKeyManager for key operations
        _secure_audit_enabled: Whether secure audit is active
    """

    _secure_audit_log: Optional[TamperEvidentLog] = None
    _secure_audit_key_manager: Optional[AuditKeyManager] = None
    _secure_audit_enabled: bool = False

    def initialize_secure_audit(
        self,
        persist_path: Optional[Path] = None,
        key_hex: Optional[str] = None,
        enable: bool = True,
    ) -> bool:
        """
        Initialize tamper-evident audit logging.

        Args:
            persist_path: Path for log persistence (optional)
            key_hex: Hex-encoded HMAC key (generates new if None)
            enable: Whether to enable secure audit logging

        Returns:
            True if initialization succeeded
        """
        if not enable:
            self._secure_audit_enabled = False
            logger.debug("Secure audit logging disabled")
            return True

        try:
            log, key_manager = create_audit_log(
                persist_path=persist_path,
                key_hex=key_hex,
            )

            self._secure_audit_log = log
            self._secure_audit_key_manager = key_manager
            self._secure_audit_enabled = True

            logger.info(
                f"Secure audit logging initialized "
                f"(key_id={key_manager.key_id}, persist={persist_path is not None})"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to initialize secure audit logging: {e}")
            self._secure_audit_enabled = False
            return False

    def secure_log(
        self,
        operation: str,
        details: Dict[str, Any],
        timestamp_ns: Optional[int] = None,
    ) -> Optional[TamperEvidentEntry]:
        """
        Log an operation with tamper-evident integrity.

        Args:
            operation: Name of the operation being logged
            details: Additional details about the operation
            timestamp_ns: Optional explicit timestamp (nanoseconds)

        Returns:
            TamperEvidentEntry if logged, None if audit disabled
        """
        if not self._secure_audit_enabled or self._secure_audit_log is None:
            return None

        try:
            content = {
                "operation": operation,
                "details": details,
                "source_class": self.__class__.__name__,
            }

            entry = self._secure_audit_log.append(content, timestamp_ns)

            logger.debug(
                f"Secure audit: {operation} (seq={entry.sequence})"
            )
            return entry

        except Exception as e:
            logger.error(f"Failed to create secure audit entry: {e}")
            return None

    def verify_audit_integrity(
        self,
        start_sequence: int = 0,
        end_sequence: Optional[int] = None,
    ) -> Tuple[bool, TamperingReport]:
        """
        Verify integrity of the audit log.

        Args:
            start_sequence: First sequence to verify
            end_sequence: Last sequence to verify (None = all)

        Returns:
            Tuple of (is_valid, TamperingReport)
        """
        if not self._secure_audit_enabled or self._secure_audit_log is None:
            return True, TamperingReport(
                is_tampered=False,
                tamper_type=None,
                affected_sequences=[],
                first_invalid_sequence=None,
                details="Secure audit not enabled",
                verified_count=0,
                total_count=0,
            )

        entries = self._secure_audit_log.get_entries(start_sequence, end_sequence)
        report = self._secure_audit_log.detect_tampering(entries)

        return not report.is_tampered, report

    def get_secure_audit_entries(
        self,
        operation: Optional[str] = None,
        start_time: Optional[datetime] = None,
        end_time: Optional[datetime] = None,
        limit: int = 100,
    ) -> List[Dict[str, Any]]:
        """
        Get audit log entries with optional filtering.

        Args:
            operation: Filter by operation name
            start_time: Filter entries after this time
            end_time: Filter entries before this time
            limit: Maximum entries to return

        Returns:
            List of entry dictionaries
        """
        if not self._secure_audit_enabled or self._secure_audit_log is None:
            return []

        entries = list(self._secure_audit_log)
        results = []

        for entry in entries:
            # Apply filters
            if operation and entry.content.get("operation") != operation:
                continue

            if start_time:
                entry_time = entry.timestamp_datetime
                if entry_time < start_time:
                    continue

            if end_time:
                entry_time = entry.timestamp_datetime
                if entry_time > end_time:
                    continue

            results.append({
                "sequence": entry.sequence,
                "timestamp": entry.timestamp_datetime.isoformat(),
                "operation": entry.content.get("operation"),
                "details": entry.content.get("details", {}),
                "content_hash": entry.content_hash[:16] + "...",
                "verified": entry.verify(
                    self._secure_audit_key_manager.get_signing_key(),
                    None if entry.sequence == 0 else self._secure_audit_log.get_entry(entry.sequence - 1),
                ) == VerificationStatus.VALID,
            })

            if len(results) >= limit:
                break

        return results

    def rotate_audit_key(
        self,
        reason: str = "scheduled",
    ) -> Optional[Dict[str, Any]]:
        """
        Rotate the audit log HMAC key.

        Args:
            reason: Reason for rotation

        Returns:
            Rotation event details or None if audit disabled
        """
        if not self._secure_audit_enabled or self._secure_audit_key_manager is None:
            return None

        if self._secure_audit_log is None:
            return None

        event = self._secure_audit_key_manager.rotate_key(
            reason=reason,
            current_sequence=len(self._secure_audit_log),
        )

        # Log the rotation event itself
        self.secure_log("key_rotation", {
            "old_key_id": event.old_key_id,
            "new_key_id": event.new_key_id,
            "reason": reason,
        })

        return event.to_dict()

    def export_audit_key(self) -> Optional[str]:
        """
        Export current audit key as hex (for backup).

        SECURITY: Store this securely. Required for verification
        after key rotation.

        Returns:
            Hex-encoded key or None if audit disabled
        """
        if not self._secure_audit_enabled or self._secure_audit_key_manager is None:
            return None

        return self._secure_audit_key_manager.export_key_hex()

    def get_audit_stats(self) -> Dict[str, Any]:
        """
        Get audit log statistics.

        Returns:
            Dictionary with audit log stats
        """
        if not self._secure_audit_enabled or self._secure_audit_log is None:
            return {
                "enabled": False,
                "entry_count": 0,
            }

        log = self._secure_audit_log
        key_manager = self._secure_audit_key_manager

        last_entry = log.last_entry
        rotation_count = len(key_manager.get_rotation_history()) if key_manager else 0

        return {
            "enabled": True,
            "entry_count": len(log),
            "last_sequence": last_entry.sequence if last_entry else None,
            "last_timestamp": last_entry.timestamp_datetime.isoformat() if last_entry else None,
            "current_key_id": key_manager.key_id if key_manager else None,
            "key_rotation_count": rotation_count,
            "chain_hash": log.last_hash[:16] + "..." if log.last_hash else None,
        }


class SecureAuditedShadowDeployerMixin(SecureAuditMixin):
    """
    Extended mixin specifically for ShadowDeployer integration.

    Provides shadow deployment-specific audit logging methods.

    Usage:
        class EnhancedShadowDeployer(ShadowDeployer, SecureAuditedShadowDeployerMixin):
            def __init__(self, *args, audit_path=None, **kwargs):
                super().__init__(*args, **kwargs)
                self.initialize_secure_audit(persist_path=audit_path)
    """

    def audit_deployment_start(
        self,
        deployment_id: str,
        hypothesis_id: str,
        hypothesis_name: str,
        duration_seconds: float,
    ) -> Optional[TamperEvidentEntry]:
        """Log deployment start with secure audit."""
        return self.secure_log("deploy_shadow_start", {
            "deployment_id": deployment_id,
            "hypothesis_id": hypothesis_id,
            "hypothesis_name": hypothesis_name,
            "duration_seconds": duration_seconds,
        })

    def audit_deployment_success(
        self,
        deployment_id: str,
        hypothesis_id: str,
    ) -> Optional[TamperEvidentEntry]:
        """Log successful deployment initialization."""
        return self.secure_log("deploy_shadow_success", {
            "deployment_id": deployment_id,
            "hypothesis_id": hypothesis_id,
        })

    def audit_deployment_failed(
        self,
        deployment_id: str,
        hypothesis_id: str,
        reason: str,
    ) -> Optional[TamperEvidentEntry]:
        """Log deployment failure."""
        return self.secure_log("deploy_shadow_failed", {
            "deployment_id": deployment_id,
            "hypothesis_id": hypothesis_id,
            "reason": reason,
        })

    def audit_kill_switch(
        self,
        deployment_id: str,
        reason: str,
        ihsan_score: float,
    ) -> Optional[TamperEvidentEntry]:
        """Log kill switch activation."""
        return self.secure_log("kill_switch_triggered", {
            "deployment_id": deployment_id,
            "reason": reason,
            "ihsan_score": ihsan_score,
        })

    def audit_evaluation(
        self,
        deployment_id: str,
        verdict: str,
        comparison_count: int,
        regression_detected: bool,
        improvement_detected: bool,
    ) -> Optional[TamperEvidentEntry]:
        """Log deployment evaluation result."""
        return self.secure_log("evaluate_complete", {
            "deployment_id": deployment_id,
            "verdict": verdict,
            "comparisons": comparison_count,
            "regression": regression_detected,
            "improvement": improvement_detected,
        })

    def audit_promotion(
        self,
        deployment_id: str,
        hypothesis_id: str,
        changes: Dict[str, Any],
    ) -> Optional[TamperEvidentEntry]:
        """Log successful promotion to production."""
        return self.secure_log("promote_success", {
            "deployment_id": deployment_id,
            "hypothesis_id": hypothesis_id,
            "changes": changes,
        })

    def audit_rollback(
        self,
        deployment_id: str,
        reason: str,
        verdict: str,
    ) -> Optional[TamperEvidentEntry]:
        """Log rollback/rejection."""
        return self.secure_log("rollback", {
            "deployment_id": deployment_id,
            "reason": reason,
            "verdict": verdict,
        })


# =============================================================================
# EXPORTS
# =============================================================================

__all__ = [
    "SecureAuditMixin",
    "SecureAuditedShadowDeployerMixin",
]
