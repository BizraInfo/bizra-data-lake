"""Creator Agent — generates content from evidence: documentation, summaries, briefs.

Produces structured output (markdown, JSON, reports) grounded in
evidence from the codebase and documentation.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import List

from bizra_config import DATA_LAKE_ROOT
from core.adk.agent import Agent, charter
from core.adk.mission import Mission
from core.adk.tools import tool

OLLAMA_URL = os.getenv("BIZRA_OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.getenv("BIZRA_CREATOR_MODEL", "gemma4:e4b")


@charter("""
I am the Creator. I generate documentation, summaries, briefs, and reports
from evidence. I never invent facts — every statement I produce is traceable
to a source document, code file, or metric. I produce structured output
(markdown with headers, tables, bullet points) optimized for human scanning.
""")
class CreatorAgent(Agent):
    name = "Creator"
    governance_class = "PAT"
    model = MODEL

    @tool(max_results=15)
    def gather_source_material(self, question: str) -> List[str]:
        """Collect source documents relevant to the creation task."""
        evidence_parts: list[str] = []
        refs: list[str] = []

        # Scan docs/ for relevant content
        docs_dir = DATA_LAKE_ROOT / "docs"
        if docs_dir.is_dir():
            for ext in ["*.md", "*.txt"]:
                for f in sorted(docs_dir.rglob(ext))[:20]:
                    try:
                        text = f.read_text()[:2000]
                        q_words = question.lower().split()[:3]
                        if any(w in text.lower() for w in q_words):
                            rel_path = f.relative_to(DATA_LAKE_ROOT)
                            evidence_parts.append(f"{rel_path}:\n{text[:500]}")
                            refs.append(f"file:{rel_path}")
                            if len(refs) >= 10:
                                break
                    except OSError:
                        pass

        # README and top-level docs
        for name in ["README.md", "ARCHITECTURE.md", "CONTRIBUTING.md", "CLAUDE.md"]:
            doc = DATA_LAKE_ROOT / name
            if doc.exists() and f"file:{name}" not in refs:
                try:
                    text = doc.read_text()[:2000]
                    evidence_parts.append(f"{name}:\n{text[:500]}")
                    refs.append(f"file:{name}")
                except OSError:
                    pass

        self._evidence_text = "\n---\n".join(evidence_parts)
        return refs

    async def act(self, mission: Mission):
        refs = self.gather_source_material(mission.question)

        if not refs:
            return self.refuse(reason="No source material found for this creation task")

        system = (
            "You are a BIZRA Creator agent. Generate structured content "
            "(markdown with headers, tables, bullet points) based ONLY on the "
            "source material provided. Every claim must be traceable to a source. "
            "Optimize for human scanning — use headers, short paragraphs, tables. "
            "If source material is insufficient, state what's missing."
        )
        prompt = f"CREATE: {mission.question}\n\nSOURCE MATERIAL:\n{self._evidence_text}\n\nOUTPUT:"

        answer = _call_ollama(prompt, system, self.model)
        if answer.startswith("ERROR:"):
            return self.refuse(reason=answer)

        return self.draft(content=answer, evidence=refs)


def _call_ollama(prompt: str, system: str, model: str) -> str:
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            "stream": False,
            "options": {"temperature": 0.5, "num_predict": 2048},
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
