"""Researcher Agent — ADK reimplementation of mvda/pat_researcher.py.

Gathers local evidence (git, docs, corpus), calls Ollama,
produces a receipted answer through the FATE gate.
"""

from __future__ import annotations

import json
import os
import subprocess
import urllib.error
import urllib.request
from pathlib import Path
from typing import List

from core.adk.agent import Agent, charter
from core.adk.mission import Mission
from core.adk.tools import tool

DATA_LAKE_ROOT = Path(
    os.getenv("BIZRA_DATA_LAKE_ROOT", "/data/bizra/repos/bizra-data-lake")
)
OLLAMA_URL = os.getenv("BIZRA_OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.getenv("BIZRA_PAT_MODEL", "gemma4:26b-bizra-16k")


@charter("""
I am the Researcher. I find verified answers by searching the local
corpus and citing every source. I never make claims I cannot prove.
I bind to evidence. I refuse to fabricate.
""")
class ResearcherAgent(Agent):
    name = "Researcher"
    governance_class = "PAT"
    model = MODEL

    @tool(max_results=20)
    def search_local(self, question: str) -> List[str]:
        """Search git history, docs, and corpus for evidence."""
        evidence_parts: list[str] = []
        refs: list[str] = []

        # Git log for relevant commits
        for grep_term in ["spearpoint", "proof", "receipt", "fate"]:
            try:
                r = subprocess.run(
                    [
                        "git",
                        "log",
                        "--oneline",
                        "--all",
                        f"--grep={grep_term}",
                        "-n",
                        "3",
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                    cwd=str(DATA_LAKE_ROOT),
                )
                if r.stdout.strip():
                    evidence_parts.append(f"Git ({grep_term}):\n{r.stdout.strip()}")
                    refs.append(f"git-log:{grep_term}")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        # Spearpoint commit check
        try:
            r = subprocess.run(
                ["git", "log", "--format=%H %s", "-1", "b08f2208"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(DATA_LAKE_ROOT),
            )
            if r.stdout.strip():
                evidence_parts.append(f"Spearpoint: {r.stdout.strip()}")
                refs.append("git-show:b08f2208")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Documentation search
        for doc in [
            DATA_LAKE_ROOT / "BIZRA_CANONICAL.md",
            DATA_LAKE_ROOT / "ARCHITECTURE.md",
            Path("/data/bizra/docs/llm-stack.md"),
        ]:
            if doc.exists():
                try:
                    text = doc.read_text()[:2000]
                    q_lower = question.lower()
                    if any(kw in text.lower() for kw in q_lower.split()[:3]):
                        evidence_parts.append(f"{doc.name}:\n{text[:400]}")
                        refs.append(f"file:{doc.name}")
                except OSError:
                    pass

        self._evidence_text = "\n---\n".join(evidence_parts)
        return refs

    async def act(self, mission: Mission):
        refs = self.search_local(mission.question)

        if not refs:
            return self.refuse(reason="No local evidence found for this question.")

        # Call Ollama
        system = (
            "You are a BIZRA Researcher agent. Answer using ONLY the evidence "
            "provided. Do not invent facts. Cite specific commits, files, or "
            "documents. If evidence is insufficient, say so."
        )
        prompt = f"QUESTION: {mission.question}\n\nEVIDENCE:\n{self._evidence_text}\n\nANSWER:"

        answer = _call_ollama(prompt, system, self.model)
        if answer.startswith("ERROR:"):
            return self.refuse(reason=answer)

        return self.draft(content=answer, evidence=refs)


def _call_ollama(prompt: str, system: str, model: str) -> str:
    """Call Ollama chat API."""
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": 1024},
        }
    ).encode()

    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/chat",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read())
            msg = data.get("message", {})
            return msg.get("content", "") or msg.get("thinking", "")
    except (urllib.error.URLError, TimeoutError) as e:
        return f"ERROR: Ollama unreachable — {e}"
