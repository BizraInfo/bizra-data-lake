"""Analyst Agent — deep quantitative analysis of codebase and system metrics.

Examines test counts, code metrics, performance data, and constitutional
compliance numbers to produce grounded analytical reports.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import urllib.error
import urllib.request
from typing import List

from bizra_config import DATA_LAKE_ROOT
from core.adk.agent import Agent, charter
from core.adk.mission import Mission
from core.adk.tools import tool

OLLAMA_URL = os.getenv("BIZRA_OLLAMA_URL", "http://127.0.0.1:11434")
MODEL = os.getenv("BIZRA_ANALYST_MODEL", "qwen2.5-coder:14b")


@charter("""
I am the Analyst. I examine quantitative evidence — test counts, code metrics,
performance benchmarks, and constitutional compliance scores. I never round
numbers to sound better. I flag discrepancies between claimed and measured
values. I produce tables, not paragraphs.
""")
class AnalystAgent(Agent):
    name = "Analyst"
    governance_class = "PAT"
    model = MODEL

    @tool(max_results=30)
    def gather_metrics(self, question: str) -> List[str]:
        """Collect quantitative metrics from the codebase."""
        evidence_parts: list[str] = []
        refs: list[str] = []

        # Test counts by module
        try:
            r = subprocess.run(
                [sys.executable, "-m", "pytest", "--co", "-q"],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(DATA_LAKE_ROOT),
            )
            if r.stdout.strip():
                last_line = r.stdout.strip().split("\n")[-1]
                evidence_parts.append(f"Test suite: {last_line}")
                refs.append("pytest:collection-count")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        # LOC counts for key modules
        for module in [
            "proof_engine",
            "pat",
            "sat",
            "urp",
            "adk",
            "pci",
            "token",
            "zpk",
        ]:
            mod_path = DATA_LAKE_ROOT / "core" / module
            if mod_path.is_dir():
                try:
                    r = subprocess.run(
                        [
                            "find",
                            str(mod_path),
                            "-name",
                            "*.py",
                            "-not",
                            "-path",
                            "*__pycache__*",
                            "-exec",
                            "cat",
                            "{}",
                            "+",
                        ],
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    loc = len(r.stdout.split("\n"))
                    files = len(list(mod_path.glob("**/*.py")))
                    evidence_parts.append(f"core/{module}: {files} files, {loc} LOC")
                    refs.append(f"loc:core/{module}")
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass

        # Constitutional constants
        try:
            from core.integration.constants import IHSAN_THRESHOLD, SNR_THRESHOLD

            evidence_parts.append(
                f"IHSAN_THRESHOLD={IHSAN_THRESHOLD}, SNR_THRESHOLD={SNR_THRESHOLD}"
            )
            refs.append("const:ihsan+snr")
        except ImportError:
            pass

        # Rust workspace status (skip in CI: cargo check is slow and starves the
        # unit-test slice under pytest-xdist + pytest-timeout).
        if os.environ.get("CI") != "true":
            try:
                r = subprocess.run(
                    ["cargo", "check", "--workspace", "--message-format=short"],
                    capture_output=True,
                    text=True,
                    timeout=60,
                    cwd=str(DATA_LAKE_ROOT / "bizra-omega"),
                )
                status = "PASS" if r.returncode == 0 else "FAIL"
                evidence_parts.append(f"Rust workspace cargo check: {status}")
                refs.append("cargo:workspace-check")
            except (subprocess.TimeoutExpired, FileNotFoundError):
                pass

        # Git stats
        try:
            r = subprocess.run(
                ["git", "rev-list", "--count", "HEAD"],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=str(DATA_LAKE_ROOT),
            )
            commits = r.stdout.strip()
            evidence_parts.append(f"Total commits on main: {commits}")
            refs.append("git:commit-count")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            pass

        self._evidence_text = "\n---\n".join(evidence_parts)
        return refs

    async def act(self, mission: Mission):
        refs = self.gather_metrics(mission.question)

        if not refs:
            return self.refuse(reason="No quantitative metrics available")

        system = (
            "You are a BIZRA Analyst agent. Produce a quantitative analysis "
            "using ONLY the metrics provided. Format as tables where possible. "
            "Never round numbers to look better. Flag any discrepancy between "
            "different sources. If data is insufficient, state what's missing."
        )
        prompt = f"ANALYSIS REQUEST: {mission.question}\n\nMETRICS:\n{self._evidence_text}\n\nANALYSIS:"

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
            "options": {"temperature": 0.2, "num_predict": 1536},
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
