# Phase 50.2 — BIZRA-RLM Bridge: PAT Agent Integration

> Standing on Giants: Zhang et al. (RLM, 2026) · Boyd (OODA loop, 1976) · Besta (Graph-of-Thoughts, 2024) · Shannon (SNR gating, 1948)

## 1. Integration Point

The RLM bridge sits between the PAT mission pipeline (`scripts/node0_activate.py:_execute_mission`) and the LLM backends. It transforms the current single-shot agent call into a recursive REPL-mediated interaction.

### Current Flow (Single-Shot)
```
Mission → Agent Selection → LLM Call (prompt → response) → Evidence Chain
```

### RLM-Enhanced Flow
```
Mission → Agent Selection → RLM REPL Init → Iterative Code-Gen Loop →
    → Sub-LM Calls (recursive) → State Accumulation →
    → FINAL_ANSWER extraction → Ihsan Gate → Evidence Chain
```

## 2. Bridge Architecture

```python
# core/inference/rlm_bridge.py

"""
BIZRA-RLM Bridge — Recursive Language Model Integration
========================================================

Adapts the RLM paradigm (Zhang et al., 2026) for BIZRA's
PAT agent pipeline, enabling recursive prompt processing
over the data lake's 102,714 knowledge vectors.

Standing on Giants:
- Zhang, Kraska, Khattab (2026): RLM algorithm (Algorithm 1)
- Boyd (1976): OODA observe-orient loop maps to REPL iterate
- Shannon (1948): SNR gating on sub-call quality
"""

from __future__ import annotations

import ast
import logging
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)

# Sandbox whitelist — only these builtins are exposed to REPL
_SAFE_BUILTINS = {
    "len", "str", "int", "float", "bool", "list", "dict", "set",
    "tuple", "range", "enumerate", "zip", "map", "filter", "sorted",
    "reversed", "min", "max", "sum", "abs", "round", "print",
    "isinstance", "type", "hasattr", "getattr",
}


@dataclass
class REPLState:
    """Sandboxed REPL state — the RLM's persistent environment."""
    variables: Dict[str, Any] = field(default_factory=dict)
    functions: Dict[str, Callable] = field(default_factory=dict)
    stdout_buffer: List[str] = field(default_factory=list)
    iteration: int = 0
    total_sub_calls: int = 0
    total_tokens_used: int = 0


@dataclass
class RLMResult:
    """Result of an RLM execution session."""
    answer: str
    iterations: int
    sub_calls: int
    tokens_used: int
    trace: List[str]
    partial: bool = False
    ihsan_score: float = 0.0


class RLMSandbox:
    """
    Sandboxed Python REPL for RLM code execution.

    Provides a restricted execution environment where LLM-generated
    code can manipulate prompt variables and invoke sub-LM calls,
    but cannot access the filesystem, network, or system resources.

    Constitutional constraint: all code is AST-validated before execution.
    """

    BLOCKED_NODES = {
        ast.Import, ast.ImportFrom,      # No imports
        ast.Delete,                       # No deletion of critical vars
    }

    BLOCKED_NAMES = {
        "open", "exec", "eval", "compile", "__import__",
        "os", "sys", "subprocess", "socket", "requests",
        "pathlib", "shutil", "glob",
    }

    def __init__(self, state: REPLState, timeout: float = 10.0):
        self.state = state
        self.timeout = timeout

    def validate_code(self, code: str) -> tuple[bool, str]:
        """AST-validate code before execution."""
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return False, f"SyntaxError: {e}"

        for node in ast.walk(tree):
            if type(node) in self.BLOCKED_NODES:
                return False, f"Blocked: {type(node).__name__}"
            if isinstance(node, ast.Name) and node.id in self.BLOCKED_NAMES:
                return False, f"Blocked name: {node.id}"
            if isinstance(node, ast.Attribute):
                if isinstance(node.value, ast.Name):
                    if node.value.id in self.BLOCKED_NAMES:
                        return False, f"Blocked: {node.value.id}.{node.attr}"

        return True, "OK"

    def execute(self, code: str) -> tuple[REPLState, str]:
        """Execute validated code in sandboxed environment."""
        valid, msg = self.validate_code(code)
        if not valid:
            return self.state, f"[SANDBOX_BLOCKED: {msg}]"

        # Build restricted globals
        safe_globals = {
            "__builtins__": {k: __builtins__[k] if isinstance(__builtins__, dict)
                           else getattr(__builtins__, k)
                           for k in _SAFE_BUILTINS
                           if (isinstance(__builtins__, dict) and k in __builtins__)
                           or hasattr(__builtins__, k)},
            "re": re,  # Regex is allowed — core RLM pattern
        }

        # Merge state variables and functions into namespace
        namespace = {**safe_globals, **self.state.variables, **self.state.functions}

        # Capture stdout
        import io
        from contextlib import redirect_stdout

        stdout_capture = io.StringIO()
        try:
            with redirect_stdout(stdout_capture):
                exec(code, namespace)  # noqa: S102 — sandboxed
        except Exception as e:
            stdout_capture.write(f"[ERROR: {type(e).__name__}: {e}]")

        # Update state variables from namespace (exclude builtins/functions)
        for key, value in namespace.items():
            if (key not in safe_globals
                and key not in self.state.functions
                and not key.startswith("_")):
                self.state.variables[key] = value

        stdout_text = stdout_capture.getvalue()
        self.state.stdout_buffer.append(stdout_text)
        self.state.iteration += 1

        return self.state, stdout_text


class BizraRLMBridge:
    """
    Bridge between BIZRA PAT agents and the RLM execution model.

    Transforms single-shot LLM calls into recursive REPL sessions
    where the agent can programmatically probe, decompose, and
    synthesize over arbitrarily large inputs.
    """

    def __init__(
        self,
        llm_call: Callable,    # async (prompt, model, max_tokens) -> str
        max_iterations: int = 20,
        max_sub_calls: int = 60,
        max_recursion_depth: int = 3,
        ihsan_threshold: float = 0.95,
    ):
        self.llm_call = llm_call
        self.max_iterations = max_iterations
        self.max_sub_calls = max_sub_calls
        self.max_recursion_depth = max_recursion_depth
        self.ihsan_threshold = ihsan_threshold

    async def execute_rlm(
        self,
        prompt: str,
        task: str,
        agent_model: str,
        sub_model: str = "",
    ) -> RLMResult:
        """
        Execute an RLM session for a PAT agent.

        Args:
            prompt: The full input (may be 10M+ chars)
            task: The agent's mission objective
            agent_model: Model for root-level reasoning
            sub_model: Model for sub-RLM calls (default: same)
        """
        sub_model = sub_model or agent_model

        # Phase 1: Init REPL with prompt as external variable
        state = REPLState()
        state.variables["prompt"] = prompt
        state.variables["prompt_length"] = len(prompt)
        state.variables["prompt_preview"] = prompt[:500]
        state.variables["task"] = task
        state.variables["FINAL_ANSWER"] = None

        # Phase 2: Register sub-RLM function
        async def lm_query(sub_prompt: str) -> str:
            """Sub-RLM call — invoke LLM on a prompt slice."""
            if state.total_sub_calls >= self.max_sub_calls:
                return "[MAX_SUB_CALLS_REACHED]"
            state.total_sub_calls += 1
            response = await self.llm_call(sub_prompt, sub_model, 600)
            state.total_tokens_used += len(response.split())
            return response

        state.functions["lm_query"] = lm_query

        # Phase 3: Build initial metadata (NOT the prompt itself)
        metadata = (
            f"You are a BIZRA PAT agent operating in RLM mode.\n"
            f"Task: {task}\n\n"
            f"[REPL Environment]\n"
            f"Variable `prompt` contains the input ({len(prompt)} chars).\n"
            f"Preview: {prompt[:300]}...\n\n"
            f"Available:\n"
            f"  prompt[:n]          — peek at first n chars\n"
            f"  prompt[a:b]         — slice\n"
            f"  prompt.split(sep)   — decompose\n"
            f"  re.findall(pat, prompt) — search\n"
            f"  lm_query(sub_text)  — recursive LLM call on a slice\n"
            f"  FINAL_ANSWER = x    — set your final answer\n\n"
            f"Write Python code to analyze the prompt and solve the task.\n"
            f"Store your final answer in FINAL_ANSWER."
        )

        hist = [metadata]
        trace = [metadata]
        sandbox = RLMSandbox(state)

        # Phase 4: Iterative REPL loop
        for iteration in range(self.max_iterations):
            # LLM generates code
            code = await self.llm_call(
                "\n".join(hist), agent_model, 1200
            )
            state.total_tokens_used += len(code.split())
            trace.append(f"[Iteration {iteration + 1}] Code:\n{code}")

            # Execute in sandbox
            state, stdout = sandbox.execute(code)
            trace.append(f"[Stdout]: {stdout[:500]}")

            # Append to history (constant-size metadata only)
            hist.append(f"Code executed. stdout ({len(stdout)} chars): "
                       f"{stdout[:200]}")

            # Check termination
            if state.variables.get("FINAL_ANSWER") is not None:
                answer = str(state.variables["FINAL_ANSWER"])
                return RLMResult(
                    answer=answer,
                    iterations=iteration + 1,
                    sub_calls=state.total_sub_calls,
                    tokens_used=state.total_tokens_used,
                    trace=trace,
                )

        # Timeout — return partial
        return RLMResult(
            answer=str(state.variables.get("FINAL_ANSWER", "")),
            iterations=self.max_iterations,
            sub_calls=state.total_sub_calls,
            tokens_used=state.total_tokens_used,
            trace=trace,
            partial=True,
        )
```

## 3. PAT Agent Enhancement

Each PAT agent type gets an RLM mode selector:

| Agent | When to Use RLM | When to Use Single-Shot |
|-------|----------------|------------------------|
| **Coordinator** | Always single-shot — orchestrates, doesn't deep-analyze | Default |
| **Analyst** | When corpus > 32K tokens or multi-document reasoning needed | Small datasets |
| **Researcher** | When searching across 100K+ knowledge vectors | Known-answer lookups |
| **Strategist** | When reasoning over full system state | Quick decisions |
| **Creator** | When synthesizing from multiple sources | Short creative tasks |
| **Guardian** | When auditing large codebases or evidence chains | Single-file checks |
| **Executor** | Always single-shot — executes, doesn't analyze | Default |

### Decision Logic
```python
def should_use_rlm(agent_type: str, prompt_length: int, task_complexity: str) -> bool:
    """Decide whether to use RLM or single-shot for this agent call."""
    # Never RLM for coordinator/executor (they orchestrate/execute, not analyze)
    if agent_type in ("coordinator", "executor"):
        return False

    # Always RLM for very large inputs
    if prompt_length > 32_000:
        return True

    # RLM for complex tasks even on shorter inputs
    if task_complexity in ("multi_document", "deep_analysis", "synthesis"):
        return prompt_length > 8_000

    return False
```

## 4. Integration with `node0_activate.py`

The RLM bridge plugs into the existing `_call_agent()` function:

```python
# In _call_agent(), after model resolution:
if should_use_rlm(agent_type, len(prompt), task_meta.get("complexity", "")):
    bridge = BizraRLMBridge(
        llm_call=_raw_llm_call,
        max_iterations=10,
        max_sub_calls=20,
    )
    result = await bridge.execute_rlm(
        prompt=full_context,
        task=agent_task,
        agent_model=resolved_model,
    )
    return {
        "agent": agent_name,
        "model": resolved_model,
        "content": result.answer,
        "tokens": result.tokens_used,
        "success": not result.partial,
        "rlm_iterations": result.iterations,
        "rlm_sub_calls": result.sub_calls,
    }
```

## 5. GoT Synthesis Enhancement

The current GoT synthesis fallback (SNR 0.083) occurs because `core/sovereign/graph_reasoning` can't reach an LLM backend. With RLM, the GoT synthesis itself becomes an RLM session:

```
GoT Hypothesis Generation:
    state["hypotheses"] = []
    for agent_result in results:
        h = lm_query(f"Extract key hypotheses from: {agent_result[:2000]}")
        state["hypotheses"].append(h)

GoT Convergence:
    state["synthesis"] = lm_query(
        f"Synthesize these {len(state['hypotheses'])} hypotheses "
        f"into a coherent assessment: {state['hypotheses']}"
    )

GoT Scoring:
    FINAL_ANSWER = {
        "synthesis": state["synthesis"],
        "snr": compute_snr(state["synthesis"]),
        "ihsan": compute_ihsan(state["synthesis"]),
    }
```

This replaces the template-based fallback with an actual LLM-powered synthesis, directly addressing the SNR 0.083 issue from the last mission run.
