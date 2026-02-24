"""Recursive Language Model bridge with an AST-gated REPL sandbox."""

from __future__ import annotations

import ast
import builtins
import inspect
import io
import json
import logging
import math
import re
from contextlib import redirect_stdout
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_SAFE_BUILTINS: tuple[str, ...] = (
    "abs",
    "all",
    "any",
    "bool",
    "dict",
    "enumerate",
    "filter",
    "float",
    "int",
    "isinstance",
    "len",
    "list",
    "map",
    "max",
    "min",
    "print",
    "range",
    "reversed",
    "round",
    "set",
    "sorted",
    "str",
    "sum",
    "tuple",
    "zip",
)

_BLOCKED_NODES: tuple[type[ast.AST], ...] = (
    ast.Import,
    ast.ImportFrom,
    ast.Global,
    ast.Nonlocal,
)

_BLOCKED_NAMES: frozenset[str] = frozenset(
    {
        "__builtins__",
        "__import__",
        "breakpoint",
        "compile",
        "eval",
        "exec",
        "exit",
        "globals",
        "input",
        "locals",
        "open",
        "os",
        "pathlib",
        "quit",
        "requests",
        "shutil",
        "socket",
        "subprocess",
        "sys",
    }
)

_BLOCKED_ATTR_BASES: frozenset[str] = frozenset(
    {
        "os",
        "pathlib",
        "requests",
        "shutil",
        "socket",
        "subprocess",
        "sys",
    }
)

_RESERVED_GLOBALS: frozenset[str] = frozenset(
    {"__builtins__", "json", "lm_query", "math", "re"}
)


@dataclass
class REPLState:
    """Persistent state for sandboxed RLM execution."""

    variables: dict[str, Any] = field(default_factory=dict)
    stdout: list[str] = field(default_factory=list)
    iteration: int = 0
    sub_calls: int = 0
    trace: list[str] = field(default_factory=list)


@dataclass
class RLMResult:
    """Outcome of one recursive language model session."""

    final_answer: str
    state: REPLState
    iterations: int
    sub_calls: int
    halted_reason: str
    success: bool


class RLMSandboxError(RuntimeError):
    """Raised when code violates sandbox policy."""


class RLMSandbox:
    """AST-validated Python REPL with restricted builtins and helpers."""

    def __init__(
        self,
        state: Optional[REPLState] = None,
        *,
        lm_query_fn: Optional[Callable[[str], Any]] = None,
        max_sub_calls: int = 60,
    ) -> None:
        self.state = state or REPLState()
        self._lm_query_fn = lm_query_fn
        self._max_sub_calls = max(0, max_sub_calls)

    def validate_code(self, code: str) -> tuple[bool, str]:
        """Return `(allowed, reason)` after static validation."""
        stripped = _strip_code_fences(code)
        if not stripped.strip():
            return False, "empty code"

        try:
            tree = ast.parse(stripped)
        except SyntaxError as exc:
            return False, f"syntax error: {exc.msg}"

        for node in ast.walk(tree):
            if isinstance(node, _BLOCKED_NODES):
                return False, f"blocked node: {type(node).__name__}"

            if isinstance(node, ast.Name) and node.id in _BLOCKED_NAMES:
                return False, f"blocked name: {node.id}"

            if isinstance(node, ast.Attribute):
                is_blocked_attr = (
                    isinstance(node.value, ast.Name)
                    and node.value.id in _BLOCKED_ATTR_BASES
                )
                if is_blocked_attr:
                    return False, f"blocked attribute: {node.value.id}.{node.attr}"

            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id in _BLOCKED_NAMES:
                    return False, f"blocked call: {node.func.id}()"

        return True, "ok"

    def execute(self, code: str) -> tuple[REPLState, str]:
        """Execute validated code and return `(updated_state, output)`."""
        stripped = _strip_code_fences(code)
        allowed, reason = self.validate_code(stripped)
        if not allowed:
            self.state.iteration += 1
            msg = f"[SANDBOX_BLOCKED] {reason}"
            self.state.trace.append(msg)
            self.state.stdout.append(msg)
            return self.state, msg

        safe_builtins = {
            name: getattr(builtins, name)
            for name in _SAFE_BUILTINS
            if hasattr(builtins, name)
        }

        def lm_query(query: str) -> str:
            if self.state.sub_calls >= self._max_sub_calls:
                return "[MAX_SUB_CALLS_REACHED]"
            self.state.sub_calls += 1
            if self._lm_query_fn is None:
                return "[LM_QUERY_UNAVAILABLE]"
            try:
                value = self._lm_query_fn(query)
            except Exception as exc:
                return f"[LM_QUERY_ERROR] {type(exc).__name__}: {exc}"

            if inspect.isawaitable(value):
                return "[LM_QUERY_ASYNC_UNSUPPORTED]"
            return str(value)

        globals_ns: dict[str, Any] = {
            "__builtins__": safe_builtins,
            "math": math,
            "re": re,
            "json": json,
            "lm_query": lm_query,
        }
        locals_ns: dict[str, Any] = dict(self.state.variables)

        stream = io.StringIO()
        output = ""
        try:
            with redirect_stdout(stream):
                compiled = compile(stripped, "<rlm-sandbox>", "exec")  # noqa: S102
                exec(compiled, globals_ns, locals_ns)  # noqa: S102
        except Exception as exc:
            output = f"[SANDBOX_ERROR] {type(exc).__name__}: {exc}"

        stdout_text = stream.getvalue().strip()
        if stdout_text and output:
            output = f"{stdout_text}\n{output}"
        elif stdout_text:
            output = stdout_text

        if not output:
            output = "[NO_OUTPUT]"

        self.state.iteration += 1
        self.state.trace.append(output)
        self.state.stdout.append(output)

        for key, value in locals_ns.items():
            if key in _RESERVED_GLOBALS or key.startswith("__"):
                continue
            self.state.variables[key] = value

        return self.state, output


class BizraRLMBridge:
    """Iterative RLM executor for PAT agents."""

    def __init__(
        self,
        raw_llm_call: Optional[Callable[[str, str], Any]] = None,
        *,
        max_iterations: int = 20,
        max_sub_calls: int = 60,
    ) -> None:
        self._raw_llm_call = raw_llm_call
        self.max_iterations = max(1, int(max_iterations))
        self.max_sub_calls = max(1, int(max_sub_calls))

    async def execute_rlm(
        self,
        prompt: str,
        task: str,
        agent_model: Any,
        sub_model: Any = "",
    ) -> RLMResult:
        """Run iterative code-generation/REPL loop until `FINAL_ANSWER` is set."""
        if not prompt.strip():
            state = REPLState(variables={"prompt": "", "task": task})
            return RLMResult(
                final_answer="",
                state=state,
                iterations=0,
                sub_calls=0,
                halted_reason="empty_prompt",
                success=False,
            )

        state = REPLState(
            variables={
                "prompt": prompt,
                "task": task,
                "FINAL_ANSWER": "",
            }
        )

        def sync_sub_query(sub_prompt: str) -> str:
            if not sub_model:
                return "[SUB_MODEL_UNSET]"

            if callable(sub_model):
                value = sub_model(sub_prompt)
                if inspect.isawaitable(value):
                    return "[ASYNC_SUB_MODEL_UNSUPPORTED]"
                return str(value)

            if isinstance(sub_model, str):
                return "[STRING_SUB_MODEL_UNSUPPORTED_IN_SYNC_SANDBOX]"

            return "[SUB_MODEL_INVALID]"

        sandbox = RLMSandbox(
            state,
            lm_query_fn=sync_sub_query,
            max_sub_calls=self.max_sub_calls,
        )

        halted_reason = "max_iterations"
        success = False
        final_answer = ""
        transcript: list[str] = [
            (
                "You are in recursive REPL mode. Respond only with Python code. "
                "Set FINAL_ANSWER when complete. "
                "Available vars: prompt, task, FINAL_ANSWER. "
                "Available helper: lm_query(str)."
            )
        ]

        for _ in range(self.max_iterations):
            llm_prompt = self._build_repl_prompt(
                task=task,
                transcript=transcript,
                state=state,
            )
            candidate_code = await self._invoke_model(agent_model, llm_prompt)
            if not candidate_code.strip():
                halted_reason = "empty_model_response"
                break

            state, output = sandbox.execute(candidate_code)
            transcript.append(f"CODE:\n{candidate_code}")
            transcript.append(f"OUTPUT:\n{output}")

            maybe_answer = state.variables.get("FINAL_ANSWER")
            if isinstance(maybe_answer, str) and maybe_answer.strip():
                final_answer = maybe_answer.strip()
                halted_reason = "final_answer"
                success = True
                break

            if state.sub_calls >= self.max_sub_calls:
                halted_reason = "max_sub_calls"
                break

        if not final_answer and state.stdout:
            final_answer = state.stdout[-1]

        return RLMResult(
            final_answer=final_answer,
            state=state,
            iterations=state.iteration,
            sub_calls=state.sub_calls,
            halted_reason=halted_reason,
            success=success,
        )

    async def _invoke_model(self, model_ref: Any, prompt: str) -> str:
        if callable(model_ref):
            value = model_ref(prompt)
            if inspect.isawaitable(value):
                value = await value
            return "" if value is None else str(value)

        if isinstance(model_ref, str) and self._raw_llm_call is not None:
            value = self._raw_llm_call(model_ref, prompt)
            if inspect.isawaitable(value):
                value = await value
            return "" if value is None else str(value)

        logger.debug("RLM model reference is not callable/usable: %r", model_ref)
        return ""

    @staticmethod
    def _build_repl_prompt(task: str, transcript: list[str], state: REPLState) -> str:
        chunks = [
            f"TASK:\n{task}",
            f"ITERATION:{state.iteration}",
            f"SUB_CALLS:{state.sub_calls}",
            "LATEST_CONTEXT:",
            "\n".join(transcript[-6:]),
        ]
        return "\n\n".join(chunks)


def should_use_rlm(
    agent_type: str,
    prompt_length: int,
    task_complexity: float,
) -> bool:
    """Routing policy for deciding between single-shot and RLM flow."""
    normalized = (agent_type or "").strip().lower()
    if normalized in {"coordinator", "executor"}:
        return False

    if int(prompt_length) > 32_000:
        return True

    if int(prompt_length) >= 8_000 and float(task_complexity) >= 0.70:
        return True

    return False


def _strip_code_fences(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


__all__ = [
    "BizraRLMBridge",
    "REPLState",
    "RLMResult",
    "RLMSandbox",
    "RLMSandboxError",
    "should_use_rlm",
]
