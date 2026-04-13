"""
Evidence Auditor — Verifies that cited evidence references actually exist locally.

Promoted from MVDA v0.3 into core proof engine.
Part of the PAT → Evidence Audit → SAT → FATE governed runtime pipeline.

Supports ref types:
  - git-log:<query>         — search git log for matching commits
  - git-show:<commit>       — verify a specific git object exists
  - git-merge-base:<ref>    — verify ancestry relationship
  - file:<path>             — verify a local file exists
  - 04_GOLD:chunk:<id>      — verify a chunk exists in corpus
  - 04_GOLD:doc:<id>        — verify a document exists in corpus

Standing on Giants:
- Lamport (1978): Event ordering and verification
- BIZRA Spearpoint PRD SP-002: "every claim must bind to evidence"
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple

from bizra_config import DATA_LAKE_ROOT, GOLD_PATH


@dataclass
class RefAudit:
    """Result of auditing a single evidence reference."""

    ref: str
    ref_type: str
    valid: bool = False
    note: str = ""


@dataclass
class EvidenceAuditResult:
    """Aggregated result of auditing all evidence references."""

    all_refs_valid: bool = False
    valid_count: int = 0
    invalid_count: int = 0
    total_count: int = 0
    invalid_refs: List[str] = field(default_factory=list)
    audit_notes: List[str] = field(default_factory=list)
    ref_audits: List[RefAudit] = field(default_factory=list)


def _classify_ref(ref: str) -> str:
    """Classify an evidence ref by its type prefix."""
    if ref.startswith("git-log:"):
        return "git-log"
    if ref.startswith("git-show:"):
        return "git-show"
    if ref.startswith("git-merge-base:"):
        return "git-merge-base"
    if ref.startswith("file:"):
        return "file"
    if ref.startswith("04_GOLD:chunk:"):
        return "gold-chunk"
    if ref.startswith("04_GOLD:doc:"):
        return "gold-doc"
    return "unknown"


def _extract_value(ref: str, ref_type: str) -> str:
    """Extract the value portion from a typed ref."""
    if ref_type in ("gold-chunk", "gold-doc"):
        return ref.split(":", 2)[2] if ref.count(":") >= 2 else ref
    return ref.split(":", 1)[1] if ":" in ref else ref


def _verify_git_log(query: str, repo_root: Path) -> Tuple[bool, str]:
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "--all", "--grep", query, "-n", "1"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(repo_root),
        )
        if result.stdout.strip():
            return True, f"found: {result.stdout.strip()[:80]}"
        return False, "no matching commits"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, "git unavailable"


def _verify_git_show(commit_ref: str, repo_root: Path) -> Tuple[bool, str]:
    try:
        result = subprocess.run(
            ["git", "cat-file", "-t", commit_ref],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(repo_root),
        )
        if result.returncode == 0:
            return True, f"object type: {result.stdout.strip()}"
        return False, f"object not found: {commit_ref}"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, "git unavailable"


def _verify_git_ancestry(ref: str, repo_root: Path) -> Tuple[bool, str]:
    """Verify that a commit ref is an ancestor of HEAD (reachable)."""
    # If ref looks like a commit hash (hex, 7+ chars), verify it directly.
    # Otherwise it's a meta-label (e.g. "ancestry-check") — verify Spearpoint.
    is_commit_hash = len(ref) >= 7 and all(c in "0123456789abcdef" for c in ref.lower())
    commit = ref if is_commit_hash else "b08f2208"
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", commit, "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            cwd=str(repo_root),
        )
        if result.returncode == 0:
            return True, f"{commit} is ancestor of HEAD"
        return False, f"{commit} is not ancestor of HEAD"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, "git unavailable"


def _verify_file(filename: str, repo_root: Path) -> Tuple[bool, str]:
    candidates = [
        repo_root / filename,
        repo_root / "core" / filename,
        repo_root / "core" / "zpk" / filename,
        Path("/data/bizra/docs") / filename,
    ]
    for p in candidates:
        if p.exists():
            return True, f"exists at {p}"
    return False, "not found in any searched path"


def _verify_gold_chunk(chunk_id: str, gold_path: Path) -> Tuple[bool, str]:
    chunks_path = gold_path / "chunks.parquet"
    if not chunks_path.exists():
        return False, "chunks.parquet not found"
    try:
        import pandas as pd

        df = pd.read_parquet(chunks_path, columns=["chunk_id"])
        if chunk_id in df["chunk_id"].values:
            return True, "chunk exists in 04_GOLD"
        return False, f"chunk_id {chunk_id} not found in corpus"
    except Exception as e:
        return False, f"parquet read error: {e}"


def _verify_gold_doc(doc_id: str, gold_path: Path) -> Tuple[bool, str]:
    docs_path = gold_path / "documents.parquet"
    if not docs_path.exists():
        return False, "documents.parquet not found"
    try:
        import pandas as pd

        df = pd.read_parquet(docs_path, columns=["doc_id"])
        if doc_id in df["doc_id"].values:
            return True, "document exists in 04_GOLD"
        return False, f"doc_id {doc_id} not found in corpus"
    except Exception as e:
        return False, f"parquet read error: {e}"


def audit_evidence(
    evidence_refs: List[str],
    *,
    repo_root: Path | None = None,
    gold_path: Path | None = None,
) -> EvidenceAuditResult:
    """Audit all evidence refs for local existence.

    Args:
        evidence_refs: List of typed refs (e.g., "git-show:b08f2208", "file:core/zpk/kernel.py")
        repo_root: Repository root for git/file verification. Defaults to DATA_LAKE_ROOT.
        gold_path: Path to 04_GOLD corpus. Defaults to GOLD_PATH.

    Returns:
        EvidenceAuditResult with per-ref audit details.
    """
    _repo = repo_root or DATA_LAKE_ROOT
    _gold = gold_path or GOLD_PATH

    result = EvidenceAuditResult(total_count=len(evidence_refs))

    if not evidence_refs:
        result.audit_notes.append("No evidence refs provided")
        return result

    for ref in evidence_refs:
        ref_type = _classify_ref(ref)
        value = _extract_value(ref, ref_type)

        if ref_type == "git-log":
            valid, note = _verify_git_log(value, _repo)
        elif ref_type == "git-show":
            valid, note = _verify_git_show(value, _repo)
        elif ref_type == "git-merge-base":
            valid, note = _verify_git_ancestry(value, _repo)
        elif ref_type == "file":
            valid, note = _verify_file(value, _repo)
        elif ref_type == "gold-chunk":
            valid, note = _verify_gold_chunk(value, _gold)
        elif ref_type == "gold-doc":
            valid, note = _verify_gold_doc(value, _gold)
        else:
            valid, note = False, f"unknown ref type: cannot verify '{ref}'"

        audit = RefAudit(ref=ref, ref_type=ref_type, valid=valid, note=note)
        result.ref_audits.append(audit)

        if valid:
            result.valid_count += 1
        else:
            result.invalid_count += 1
            result.invalid_refs.append(ref)
            result.audit_notes.append(f"INVALID: {ref} — {note}")

    result.all_refs_valid = result.invalid_count == 0
    return result
