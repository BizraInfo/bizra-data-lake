"""Strategist Agent — synthesizes multi-source evidence into strategic options.

Gathers evidence from local corpus, git history, and documentation,
then produces ranked strategic recommendations with evidence backing.
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

DATA_LAKE_ROOT = Path(os.getenv("BIZRA_DATA_LAKE_ROOT", "/data/bizra/repos/bizra-data-lake"))
OLLAMA_URL = os.getenv("BIZRA_OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.getenv("BIZRA_STRATEGIST_MODEL", "gemma4:26b-bizra-16k")


@charter("""
I am the Strategist. I synthesize multi-source evidence into ranked
strategic options. I never recommend an option I cannot defend with
at least 3 independent pieces of evidence. I bind to data, not to vibes.
I consider trade-offs honestly and flag risks I cannot mitigate.
""")
class StrategistAgent(Agent):
    name = "Strategist"
    governance_class = "PAT"
    model = MODEL

    @tool(max_results=20)
    def gather_strategic_context(self, question: str) -> List[str]:
        """Search architecture docs, plans, and git for strategic context."""
        evidence_parts: list[str] = []
        refs: list[str] = []

        # Architecture and planning docs
        for doc in [
            DATA_LAKE_ROOT / "ARCHITECTURE.md",
            DATA_LAKE_ROOT / "docs" / "BIZRA_STRATEGY_DECK_2026.md",
            DATA_LAKE_ROOT / "docs" / "business" / "investor_pitch.md",
            DATA_LAKE_ROOT / "docs" / "business" / "ONE_PAGE_PITCH.md",
            DATA_LAKE_ROOT / "docs" / "plans",
        ]:
            if doc.is_file():
                try:
                    text = doc.read_text()[:3000]
                    q_words = question.lower().split()[:4]
                    if any(w in text.lower() for w in q_words):
                        evidence_parts.append(f"{doc.name}:\n{text[:600]}")
                        refs.append(f"file:{doc.name}")
                except OSError:
                    pass
            elif doc.is_dir():
                try:
                    for f in sorted(doc.glob("*.md"))[:5]:
                        text = f.read_text()[:1500]
                        evidence_parts.append(f"{f.name}:\n{text[:400]}")
                        refs.append(f"file:{f.name}")
                except OSError:
                    pass

        # Git log for strategic decisions
        try:
            r = subprocess.run(
                ["git", "log", "--oneline", "--all", "-n", "10", "--format=%h %s"],
                capture_output=True, text=True, timeout=10,
                cwd=str(DATA_LAKE_ROOT),
            )
            if r.stdout.strip():
                evidence_parts.append(f"Recent commits:\n{r.stdout.strip()}")
                refs.append("git-log:recent-10")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # Constitutional constants
        constants_path = DATA_LAKE_ROOT / "core" / "integration" / "constants.py"
        if constants_path.exists():
            try:
                text = constants_path.read_text()[:2000]
                evidence_parts.append(f"Constitutional constants:\n{text[:500]}")
                refs.append("file:core/integration/constants.py")
            except OSError:
                pass

        self._evidence_text = "\n---\n".join(evidence_parts)
        return refs

    async def act(self, mission: Mission):
        refs = self.gather_strategic_context(mission.question)

        if len(refs) < 3:
            return self.refuse(reason=f"Insufficient strategic context ({len(refs)} sources, need >= 3)")

        system = (
            "You are a BIZRA Strategist agent. Produce ranked strategic options "
            "based ONLY on the evidence provided. For each option:\n"
            "1. State the option clearly\n"
            "2. List supporting evidence (cite specific docs/commits)\n"
            "3. Identify trade-offs and risks\n"
            "4. Give a confidence score (0-1)\n"
            "If evidence is insufficient for a recommendation, say so honestly."
        )
        prompt = f"QUESTION: {mission.question}\n\nEVIDENCE:\n{self._evidence_text}\n\nSTRATEGIC OPTIONS:"

        answer = _call_ollama(prompt, system, self.model)
        if answer.startswith("ERROR:"):
            return self.refuse(reason=answer)

        return self.draft(content=answer, evidence=refs)


def _call_ollama(prompt: str, system: str, model: str) -> str:
    payload = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
        "options": {"temperature": 0.4, "num_predict": 1536},
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
            return msg.get("content", "") or msg.get("thinking", "")
    except (urllib.error.URLError, TimeoutError) as e:
        return f"ERROR: Ollama unreachable — {e}"
