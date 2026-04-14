"""Coordinator Agent — orchestrates multi-agent missions across the PAT-7.

Decomposes complex questions into sub-missions, dispatches to specialist
agents, and synthesizes their results into a unified response.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import List

from core.adk.agent import Agent, charter
from core.adk.mission import Mission
from core.adk.tools import tool

OLLAMA_URL = os.getenv("BIZRA_OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.getenv("BIZRA_COORDINATOR_MODEL", "gemma4:26b-bizra-16k")


PAT_ROSTER = {
    "Researcher": "Find verified answers by searching local knowledge and citing every source",
    "Strategist": "Synthesize evidence into ranked strategic options with trade-offs",
    "Analyst": "Quantitative analysis — test counts, code metrics, compliance scores",
    "Creator": "Generate documentation, summaries, briefs from evidence",
    "Executor": "Run safe commands (tests, builds, health checks) and capture results",
    "Coordinator": "Orchestrate multi-agent missions (that's me)",
}


@charter("""
I am the Coordinator. I decompose complex questions into sub-missions
for specialist PAT agents. I never do the specialist work myself — I
route, synthesize, and ensure quality. I track which agents contributed
to the final answer and cite them. I refuse to answer questions that
should be routed to a specialist.
""")
class CoordinatorAgent(Agent):
    name = "Coordinator"
    governance_class = "PAT"
    model = MODEL

    @tool(max_results=5)
    def plan_delegation(self, question: str) -> List[str]:
        """Analyze question and determine which PAT agents should handle it."""
        refs: list[str] = []

        # Use LLM to decompose the question
        system = (
            "You are a mission coordinator. Given a question, determine which "
            "specialist agents should handle it. Available agents:\n"
            + "\n".join(f"- {name}: {desc}" for name, desc in PAT_ROSTER.items())
            + "\n\nRespond with JSON: {\"plan\": [{\"agent\": \"name\", \"sub_question\": \"...\"}]}"
        )
        prompt = f"QUESTION: {question}\n\nDELEGATION PLAN (JSON):"

        plan_text = _call_ollama(prompt, system, self.model)

        # Parse the plan
        try:
            # Extract JSON from response
            start = plan_text.find("{")
            end = plan_text.rfind("}") + 1
            if start >= 0 and end > start:
                plan = json.loads(plan_text[start:end])
                steps = plan.get("plan", [])
                for step in steps:
                    agent = step.get("agent", "Unknown")
                    sub_q = step.get("sub_question", "")
                    refs.append(f"delegate:{agent}:{sub_q[:50]}")
                self._delegation_plan = steps
            else:
                self._delegation_plan = []
                refs.append("delegate:parse_failed")
        except (json.JSONDecodeError, KeyError):
            self._delegation_plan = []
            refs.append("delegate:parse_failed")

        self._evidence_text = plan_text
        return refs

    async def act(self, mission: Mission):
        refs = self.plan_delegation(mission.question)

        if not self._delegation_plan:
            return self.refuse(reason="Could not decompose mission into sub-tasks")

        # At N=1, we can't actually dispatch to other agents in parallel yet.
        # Instead, synthesize the delegation plan as the output, showing
        # which agents would handle which parts.
        system = (
            "You are a BIZRA Coordinator. You've planned a multi-agent mission. "
            "Produce a coordination brief that:\n"
            "1. Lists each sub-mission and the assigned agent\n"
            "2. Explains why that agent was chosen\n"
            "3. Identifies dependencies between sub-missions\n"
            "4. Estimates the overall mission structure\n"
            "Be concise. Use markdown headers and bullet points."
        )
        plan_summary = json.dumps(self._delegation_plan, indent=2)
        prompt = (
            f"ORIGINAL MISSION: {mission.question}\n\n"
            f"DELEGATION PLAN:\n{plan_summary}\n\n"
            f"COORDINATION BRIEF:"
        )

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
            return msg.get("content", "") or msg.get("thinking", "")
    except (urllib.error.URLError, TimeoutError) as e:
        return f"ERROR: Ollama unreachable — {e}"
