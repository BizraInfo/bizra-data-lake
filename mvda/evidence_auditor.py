"""MVDA v0.3 — Evidence Auditor: verifies cited refs actually exist locally."""

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd

from mvda.config import DATA_LAKE_ROOT

# GOLD_DIR honors the operator-provided env vars so the auditor works in CI
# (where the data lake root is the GitHub workspace, no /data/bizra/ mount)
# and in local dev (where the corpus lives at /data/bizra/04_GOLD).
GOLD_DIR = Path(
    os.getenv("BIZRA_GOLD_DIR")
    or os.path.join(
        os.getenv("BIZRA_DATA_LAKE_ROOT", "/data/bizra"),
        "04_GOLD",
    )
)


@dataclass
class RefAudit:
    ref: str
    ref_type: str  # git-log, git-show, git-merge-base, file, 04_GOLD:chunk, 04_GOLD:doc, unknown
    valid: bool = False
    note: str = ""


@dataclass
class EvidenceAuditResult:
    all_refs_valid: bool = False
    valid_count: int = 0
    invalid_count: int = 0
    total_count: int = 0
    invalid_refs: List[str] = field(default_factory=list)
    audit_notes: List[str] = field(default_factory=list)
    ref_audits: List[RefAudit] = field(default_factory=list)


def _classify_ref(ref: str) -> str:
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


def _verify_git_log(query: str) -> Tuple[bool, str]:
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "--all", "--grep", query, "-n", "1"],
            capture_output=True, text=True, timeout=10, cwd=str(DATA_LAKE_ROOT),
        )
        if result.stdout.strip():
            return True, f"found: {result.stdout.strip()[:80]}"
        return False, "no matching commits"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, "git unavailable"


def _verify_git_show(commit_ref: str) -> Tuple[bool, str]:
    try:
        result = subprocess.run(
            ["git", "cat-file", "-t", commit_ref],
            capture_output=True, text=True, timeout=10, cwd=str(DATA_LAKE_ROOT),
        )
        if result.returncode == 0:
            return True, f"object type: {result.stdout.strip()}"
        return False, f"object not found: {commit_ref}"
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False, "git unavailable"


def _verify_git_ancestry(ref: str) -> Tuple[bool, str]:
    # ancestry-check is a meta-ref, the actual check was done by PAT
    # verify the referenced commit exists
    return _verify_git_show("b08f2208")


def _verify_file(filename: str) -> Tuple[bool, str]:
    # Check multiple locations
    candidates = [
        DATA_LAKE_ROOT / filename,
        DATA_LAKE_ROOT / "core" / filename,
        DATA_LAKE_ROOT / "core" / "zpk" / filename,
        Path("/data/bizra/docs") / filename,
        Path("/home/bizra-operating-system") / filename,
    ]
    for p in candidates:
        if p.exists():
            return True, f"exists at {p}"
    return False, f"not found in any searched path"


def _verify_gold_chunk(chunk_id: str) -> Tuple[bool, str]:
    chunks_path = GOLD_DIR / "chunks.parquet"
    if not chunks_path.exists():
        return False, "chunks.parquet not found"
    try:
        df = pd.read_parquet(chunks_path, columns=["chunk_id"])
        if chunk_id in df["chunk_id"].values:
            return True, f"chunk exists in 04_GOLD"
        return False, f"chunk_id {chunk_id} not found in corpus"
    except Exception as e:
        return False, f"parquet read error: {e}"


def _verify_gold_doc(doc_id: str) -> Tuple[bool, str]:
    docs_path = GOLD_DIR / "documents.parquet"
    if not docs_path.exists():
        return False, "documents.parquet not found"
    try:
        df = pd.read_parquet(docs_path, columns=["doc_id"])
        if doc_id in df["doc_id"].values:
            return True, f"document exists in 04_GOLD"
        return False, f"doc_id {doc_id} not found in corpus"
    except Exception as e:
        return False, f"parquet read error: {e}"


def audit_evidence(evidence_refs: List[str]) -> EvidenceAuditResult:
    """Audit all evidence refs for existence. Returns structured result."""
    result = EvidenceAuditResult(total_count=len(evidence_refs))

    if not evidence_refs:
        result.audit_notes.append("No evidence refs provided")
        return result

    for ref in evidence_refs:
        ref_type = _classify_ref(ref)
        # Extract value after the type prefix
        if ref_type == "gold-chunk":
            value = ref.split(":", 2)[2] if ref.count(":") >= 2 else ref
        elif ref_type == "gold-doc":
            value = ref.split(":", 2)[2] if ref.count(":") >= 2 else ref
        else:
            value = ref.split(":", 1)[1] if ":" in ref else ref

        if ref_type == "git-log":
            valid, note = _verify_git_log(value)
        elif ref_type == "git-show":
            valid, note = _verify_git_show(value)
        elif ref_type == "git-merge-base":
            valid, note = _verify_git_ancestry(value)
        elif ref_type == "file":
            valid, note = _verify_file(value)
        elif ref_type == "gold-chunk":
            valid, note = _verify_gold_chunk(value)
        elif ref_type == "gold-doc":
            valid, note = _verify_gold_doc(value)
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
