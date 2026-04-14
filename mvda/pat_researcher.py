"""PAT Researcher — local evidence-gathering execution role."""

import json
import subprocess
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from mvda.config import DATA_LAKE_ROOT, OLLAMA_URL, PAT_MODEL


@dataclass
class PatResult:
    answer: str = ""
    evidence_refs: List[str] = field(default_factory=list)
    confidence: str = ""
    model: str = ""
    raw_sources: str = ""


def _gather_local_evidence(question: str) -> tuple[str, list[str]]:
    """Search local files for evidence relevant to the question."""
    evidence_parts = []
    refs = []

    # Search git log for spearpoint-related commits
    try:
        result = subprocess.run(
            ["git", "log", "--oneline", "--all", "--grep=spearpoint", "-n", "5"],
            capture_output=True, text=True, timeout=10,
            cwd=str(DATA_LAKE_ROOT),
        )
        if result.stdout.strip():
            evidence_parts.append(f"Git commits matching 'spearpoint':\n{result.stdout.strip()}")
            refs.append("git-log:spearpoint")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Check for the specific spearpoint commit
    try:
        result = subprocess.run(
            ["git", "log", "--format=%H %s", "-1", "b08f2208"],
            capture_output=True, text=True, timeout=10,
            cwd=str(DATA_LAKE_ROOT),
        )
        if result.stdout.strip():
            evidence_parts.append(f"Spearpoint commit b08f2208:\n{result.stdout.strip()}")
            refs.append("git-show:b08f2208")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Check if spearpoint is reachable from HEAD
    try:
        result = subprocess.run(
            ["git", "merge-base", "--is-ancestor", "b08f2208", "HEAD"],
            capture_output=True, text=True, timeout=10,
            cwd=str(DATA_LAKE_ROOT),
        )
        reachable = result.returncode == 0
        evidence_parts.append(f"Spearpoint b08f2208 reachable from HEAD: {reachable}")
        refs.append("git-merge-base:ancestry-check")
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass

    # Search for relevant documentation
    for doc_path in [
        DATA_LAKE_ROOT / "BIZRA_CANONICAL.md",
        DATA_LAKE_ROOT / "ARCHITECTURE.md",
        Path("/data/bizra/docs/llm-stack.md"),
        Path("/home/bizra-operating-system/CLAUDE.md"),
    ]:
        if doc_path.exists():
            try:
                content = doc_path.read_text()[:2000]
                if "spearpoint" in content.lower() or "b08f2208" in content:
                    evidence_parts.append(f"From {doc_path.name}:\n{content[:500]}")
                    refs.append(f"file:{doc_path.name}")
            except OSError:
                pass

    # Search ZPK kernel for relevant code
    zpk_path = DATA_LAKE_ROOT / "core" / "zpk" / "kernel.py"
    if zpk_path.exists():
        try:
            content = zpk_path.read_text()[:500]
            evidence_parts.append(f"ZPK Kernel header:\n{content[:300]}")
            refs.append("file:core/zpk/kernel.py")
        except OSError:
            pass

    combined = "\n\n---\n\n".join(evidence_parts) if evidence_parts else ""
    return combined, refs


def _call_ollama(prompt: str, system: str, model: str) -> str:
    """Call Ollama chat API (generate endpoint has empty-response bug with gemma4)."""
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.3, "num_predict": 1024},
    }).encode()

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            msg = data.get("message", {})
            # gemma4 is a thinking model — content may be in "thinking" field
            content = msg.get("content", "")
            if not content:
                content = msg.get("thinking", "")
            return content
    except (urllib.error.URLError, TimeoutError) as e:
        return f"ERROR: Ollama unreachable — {e}"


def run_pat_researcher(question: str) -> PatResult:
    """Execute PAT Researcher: gather evidence, then reason with local model."""
    evidence_text, refs = _gather_local_evidence(question)

    if not evidence_text:
        return PatResult(
            answer="",
            evidence_refs=[],
            confidence="none — no local evidence found",
            model=PAT_MODEL,
        )

    system_prompt = (
        "You are a BIZRA Researcher agent. Answer the question using ONLY the "
        "evidence provided below. Do not invent facts. If the evidence is insufficient, "
        "say so. Be precise and cite specific commits, files, or documents."
    )

    user_prompt = f"QUESTION: {question}\n\nEVIDENCE:\n{evidence_text}\n\nANSWER:"

    answer = _call_ollama(user_prompt, system_prompt, PAT_MODEL)

    confidence = "high" if len(refs) >= 3 else "medium" if len(refs) >= 1 else "none"

    return PatResult(
        answer=answer,
        evidence_refs=refs,
        confidence=confidence,
        model=PAT_MODEL,
        raw_sources=evidence_text[:1000],
    )
