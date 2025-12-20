# ═══════════════════════════════════════════════════════════════════════════════
# BIZRA Constellation - Auto-Audit Integration v1.0
# ═══════════════════════════════════════════════════════════════════════════════
"""
Integration with CodeRabbit for automated code auditing:
- Trigger reviews on agent output
- Apply AI suggestions
- Track review status
- Manage audit trails
"""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional, Any, Callable, Awaitable
from enum import Enum
from pathlib import Path


logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT TYPES
# ─────────────────────────────────────────────────────────────────────────────

class AuditType(str, Enum):
    """Types of audits."""
    CODE_REVIEW = "code_review"
    AGENT_OUTPUT = "agent_output"
    CLAIM_VERIFICATION = "claim_verification"
    SNR_CHECK = "snr_check"
    SECURITY_SCAN = "security_scan"
    QUALITY_CHECK = "quality_check"


class AuditStatus(str, Enum):
    """Status of an audit."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    NEEDS_ATTENTION = "needs_attention"


class SuggestionStatus(str, Enum):
    """Status of a suggestion."""
    PENDING = "pending"
    APPLIED = "applied"
    REJECTED = "rejected"
    DEFERRED = "deferred"


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT STRUCTURES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class AuditSuggestion:
    """A suggestion from the audit."""
    id: str
    type: str  # "improvement", "fix", "warning", "info"
    title: str
    description: str
    location: Optional[str] = None  # File path or agent output section
    severity: str = "info"  # "critical", "high", "medium", "low", "info"
    suggested_change: Optional[str] = None
    status: SuggestionStatus = SuggestionStatus.PENDING
    applied_at: Optional[str] = None
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "description": self.description,
            "location": self.location,
            "severity": self.severity,
            "suggested_change": self.suggested_change,
            "status": self.status.value,
            "applied_at": self.applied_at,
        }


@dataclass
class AuditReport:
    """Report from an audit."""
    id: str
    audit_type: AuditType
    target: str  # What was audited
    status: AuditStatus
    
    # Results
    suggestions: list[AuditSuggestion] = field(default_factory=list)
    score: Optional[float] = None
    summary: str = ""
    
    # Metadata
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: Optional[str] = None
    auditor: str = "coderabbit"  # or agent slug
    
    # Statistics
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    info_count: int = 0
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "audit_type": self.audit_type.value,
            "target": self.target,
            "status": self.status.value,
            "suggestions": [s.to_dict() for s in self.suggestions],
            "score": self.score,
            "summary": self.summary,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "auditor": self.auditor,
            "statistics": {
                "critical": self.critical_count,
                "high": self.high_count,
                "medium": self.medium_count,
                "low": self.low_count,
                "info": self.info_count,
            },
        }


# ─────────────────────────────────────────────────────────────────────────────
# CODERABBIT COMMANDS
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CodeRabbitCommand:
    """A CodeRabbit VS Code command."""
    id: str
    name: str
    description: str
    parameters: list[str] = field(default_factory=list)


class CodeRabbitCommands:
    """Available CodeRabbit VS Code commands."""
    
    # Review commands
    INITIATE_REVIEW = CodeRabbitCommand(
        id="coderabbit-vscode.initiateReview",
        name="Start Review",
        description="Initiate a new code review",
    )
    
    APPLY_SUGGESTION = CodeRabbitCommand(
        id="coderabbit-vscode.applySuggestion",
        name="Apply Suggestion",
        description="Apply a suggested change",
        parameters=["suggestionId"],
    )
    
    HANDOFF_TO_AGENT = CodeRabbitCommand(
        id="coderabbit-vscode.handoffToAgent",
        name="Fix with AI",
        description="Hand off to AI agent for fixing",
    )
    
    CLEANUP_REVIEWS = CodeRabbitCommand(
        id="coderabbit-vscode.cleanupReviews",
        name="Cleanup Reviews",
        description="Clean up previous review comments",
    )
    
    # Comment management
    COLLAPSE_ALL = CodeRabbitCommand(
        id="coderabbit-vscode.collapseAllComments",
        name="Collapse All Comments",
        description="Collapse all review comments",
    )
    
    EXPAND_ALL = CodeRabbitCommand(
        id="coderabbit-vscode.expandAllComments",
        name="Expand All Comments",
        description="Expand all review comments",
    )
    
    RESOLVE_COMMENT = CodeRabbitCommand(
        id="coderabbit-vscode.resolveComment",
        name="Resolve Comment",
        description="Mark comment as resolved/ignored",
        parameters=["commentId"],
    )
    
    # Feedback
    SUBMIT_FEEDBACK = CodeRabbitCommand(
        id="coderabbit-vscode.submitFeedback",
        name="Submit Feedback",
        description="Provide feedback on review",
        parameters=["feedback"],
    )


# ─────────────────────────────────────────────────────────────────────────────
# AUTO-AUDIT ENGINE
# ─────────────────────────────────────────────────────────────────────────────

class AutoAuditEngine:
    """
    Engine for automated auditing of code and agent outputs.
    
    Integrates with:
    - CodeRabbit for code review
    - Internal SNR verification
    - Claim validation
    """
    
    def __init__(
        self,
        storage_path: Optional[Path] = None,
        auto_apply_threshold: float = 0.95,  # Auto-apply suggestions above this confidence
    ):
        self.storage_path = storage_path or Path("bizra_data_vault/audits")
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.auto_apply_threshold = auto_apply_threshold
        
        self._reports: dict[str, AuditReport] = {}
        self._audit_counter = 0
        
    async def audit_code(
        self,
        file_path: str,
        content: Optional[str] = None,
        trigger_coderabbit: bool = True,
    ) -> AuditReport:
        """Audit a code file."""
        self._audit_counter += 1
        report_id = f"audit_{self._audit_counter:05d}"
        
        report = AuditReport(
            id=report_id,
            audit_type=AuditType.CODE_REVIEW,
            target=file_path,
            status=AuditStatus.IN_PROGRESS,
        )
        
        self._reports[report_id] = report
        
        # If CodeRabbit integration is enabled
        if trigger_coderabbit:
            await self._trigger_coderabbit_review(file_path)
            
        # Run internal checks
        await self._run_internal_checks(report, content or "")
        
        report.status = AuditStatus.COMPLETED
        report.completed_at = datetime.now(timezone.utc).isoformat()
        
        self._persist_report(report)
        return report
        
    async def audit_agent_output(
        self,
        agent_slug: str,
        output: str,
        claims: list[dict],
        snr_score: float,
    ) -> AuditReport:
        """Audit an agent's output."""
        self._audit_counter += 1
        report_id = f"audit_{self._audit_counter:05d}"
        
        report = AuditReport(
            id=report_id,
            audit_type=AuditType.AGENT_OUTPUT,
            target=f"agent:{agent_slug}",
            status=AuditStatus.IN_PROGRESS,
            auditor=agent_slug,
        )
        
        self._reports[report_id] = report
        
        # Check SNR
        if snr_score < 0.90:
            report.suggestions.append(AuditSuggestion(
                id=f"{report_id}_snr",
                type="warning",
                title="SNR Below Threshold",
                description=f"Output SNR ({snr_score:.2%}) is below the 90% threshold",
                severity="high" if snr_score < 0.85 else "medium",
            ))
            
        # Verify claims
        for i, claim in enumerate(claims):
            claim_text = claim.get("text", "")
            claim_tag = claim.get("tag", "HYPOTHESIS")
            
            # Flag unverified claims
            if claim_tag == "HYPOTHESIS" and not claim.get("verified"):
                report.suggestions.append(AuditSuggestion(
                    id=f"{report_id}_claim_{i}",
                    type="info",
                    title="Unverified Hypothesis",
                    description=f"Claim requires verification: {claim_text[:100]}",
                    severity="low",
                ))
                
        # Calculate statistics
        self._calculate_statistics(report)
        
        report.status = AuditStatus.COMPLETED
        report.completed_at = datetime.now(timezone.utc).isoformat()
        report.score = snr_score
        
        self._persist_report(report)
        return report
        
    async def verify_claims(
        self,
        claims: list[dict],
        context: str = "",
    ) -> AuditReport:
        """Verify a set of claims."""
        self._audit_counter += 1
        report_id = f"audit_{self._audit_counter:05d}"
        
        report = AuditReport(
            id=report_id,
            audit_type=AuditType.CLAIM_VERIFICATION,
            target="claims",
            status=AuditStatus.IN_PROGRESS,
        )
        
        verified_count = 0
        
        for i, claim in enumerate(claims):
            # Placeholder verification logic
            # Would integrate with knowledge graph
            is_verified = claim.get("tag") == "MEASURED"
            
            if is_verified:
                verified_count += 1
            else:
                report.suggestions.append(AuditSuggestion(
                    id=f"{report_id}_verify_{i}",
                    type="warning",
                    title="Unverified Claim",
                    description=claim.get("text", "")[:200],
                    severity="medium" if claim.get("tag") == "HYPOTHESIS" else "low",
                ))
                
        report.score = verified_count / len(claims) if claims else 1.0
        report.summary = f"Verified {verified_count}/{len(claims)} claims"
        
        self._calculate_statistics(report)
        report.status = AuditStatus.COMPLETED
        report.completed_at = datetime.now(timezone.utc).isoformat()
        
        self._persist_report(report)
        return report
        
    async def apply_suggestion(
        self,
        report_id: str,
        suggestion_id: str,
    ) -> bool:
        """Apply a suggestion from an audit."""
        report = self._reports.get(report_id)
        if not report:
            return False
            
        for suggestion in report.suggestions:
            if suggestion.id == suggestion_id:
                if suggestion.suggested_change:
                    # Would apply the change here
                    pass
                suggestion.status = SuggestionStatus.APPLIED
                suggestion.applied_at = datetime.now(timezone.utc).isoformat()
                return True
                
        return False
        
    async def reject_suggestion(
        self,
        report_id: str,
        suggestion_id: str,
        reason: Optional[str] = None,
    ) -> bool:
        """Reject a suggestion."""
        report = self._reports.get(report_id)
        if not report:
            return False
            
        for suggestion in report.suggestions:
            if suggestion.id == suggestion_id:
                suggestion.status = SuggestionStatus.REJECTED
                return True
                
        return False
        
    def get_report(self, report_id: str) -> Optional[AuditReport]:
        """Get an audit report by ID."""
        return self._reports.get(report_id)
        
    def get_reports(
        self,
        audit_type: Optional[AuditType] = None,
        status: Optional[AuditStatus] = None,
        limit: int = 50,
    ) -> list[AuditReport]:
        """Get audit reports with optional filtering."""
        reports = list(self._reports.values())
        
        if audit_type:
            reports = [r for r in reports if r.audit_type == audit_type]
        if status:
            reports = [r for r in reports if r.status == status]
            
        return sorted(reports, key=lambda r: r.started_at, reverse=True)[:limit]
        
    def get_pending_suggestions(self) -> list[tuple[AuditReport, AuditSuggestion]]:
        """Get all pending suggestions across reports."""
        pending = []
        for report in self._reports.values():
            for suggestion in report.suggestions:
                if suggestion.status == SuggestionStatus.PENDING:
                    pending.append((report, suggestion))
        return pending
        
    async def _trigger_coderabbit_review(self, file_path: str) -> None:
        """Trigger CodeRabbit review via VS Code command."""
        # This would be executed via VS Code extension API
        logger.info(f"Triggering CodeRabbit review for: {file_path}")
        # Command: coderabbit-vscode.initiateReview
        
    async def _run_internal_checks(
        self,
        report: AuditReport,
        content: str,
    ) -> None:
        """Run internal code quality checks."""
        # Check for common issues
        lines = content.split("\n")
        
        for i, line in enumerate(lines):
            # Long lines
            if len(line) > 120:
                report.suggestions.append(AuditSuggestion(
                    id=f"{report.id}_line_{i}",
                    type="info",
                    title="Long Line",
                    description=f"Line {i+1} exceeds 120 characters",
                    location=f"line {i+1}",
                    severity="low",
                ))
                
            # TODO comments
            if "TODO" in line or "FIXME" in line:
                report.suggestions.append(AuditSuggestion(
                    id=f"{report.id}_todo_{i}",
                    type="info",
                    title="TODO Found",
                    description=line.strip(),
                    location=f"line {i+1}",
                    severity="info",
                ))
                
    def _calculate_statistics(self, report: AuditReport) -> None:
        """Calculate suggestion statistics."""
        report.critical_count = sum(1 for s in report.suggestions if s.severity == "critical")
        report.high_count = sum(1 for s in report.suggestions if s.severity == "high")
        report.medium_count = sum(1 for s in report.suggestions if s.severity == "medium")
        report.low_count = sum(1 for s in report.suggestions if s.severity == "low")
        report.info_count = sum(1 for s in report.suggestions if s.severity == "info")
        
        # Determine if needs attention
        if report.critical_count > 0 or report.high_count > 2:
            report.status = AuditStatus.NEEDS_ATTENTION
            
    def _persist_report(self, report: AuditReport) -> None:
        """Persist report to storage."""
        report_file = self.storage_path / f"{report.id}.json"
        with open(report_file, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)


# ─────────────────────────────────────────────────────────────────────────────
# AUDIT HOOKS
# ─────────────────────────────────────────────────────────────────────────────

class AuditHooks:
    """Hook handlers for automatic auditing."""
    
    def __init__(self, engine: AutoAuditEngine):
        self.engine = engine
        
    async def on_agent_complete(self, event: dict) -> None:
        """Hook handler for agent completion."""
        agent_slug = event.get("agent_slug")
        output = event.get("result", {}).get("content", "")
        claims = event.get("result", {}).get("claims", [])
        snr = event.get("result", {}).get("snr_score", 0.0)
        
        # Audit the output
        await self.engine.audit_agent_output(
            agent_slug=agent_slug,
            output=output,
            claims=claims,
            snr_score=snr,
        )
        
    async def on_code_generated(self, event: dict) -> None:
        """Hook handler for code generation."""
        file_path = event.get("file_path")
        content = event.get("content", "")
        
        if file_path:
            await self.engine.audit_code(file_path, content)


# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL INSTANCE
# ─────────────────────────────────────────────────────────────────────────────

_engine: Optional[AutoAuditEngine] = None


def get_audit_engine() -> AutoAuditEngine:
    """Get the global audit engine."""
    global _engine
    if _engine is None:
        _engine = AutoAuditEngine()
    return _engine


def get_audit_hooks() -> AuditHooks:
    """Get audit hooks."""
    return AuditHooks(get_audit_engine())
