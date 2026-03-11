"""Recursive Language Model bridge with an AST-gated REPL sandbox."""

from __future__ import annotations

import ast
import builtins
import concurrent.futures
import inspect
import io
import json
import logging
import math
import os
import re
import signal
import threading
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

# ── Allowlist-based AST validation (replaces former denylist) ──────────
# Standing on Giants: Saltzer & Schroeder (1975) — "fail-safe defaults"
# Only node types explicitly listed here are permitted.  Everything else
# (including ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal) is
# rejected BEFORE compile(), closing object-introspection escape surfaces.

_ALLOWED_AST_NODES: frozenset[type[ast.AST]] = frozenset(
    {
        # Structural
        ast.Module,
        ast.Expression,
        ast.Interactive,
        # Statements
        ast.Expr,
        ast.Assign,
        ast.AugAssign,
        ast.AnnAssign,
        ast.Return,
        ast.If,
        ast.For,
        ast.While,
        ast.Break,
        ast.Continue,
        ast.Pass,
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        # Expressions
        ast.Name,
        ast.Constant,
        ast.BinOp,
        ast.UnaryOp,
        ast.BoolOp,
        ast.Compare,
        ast.Call,
        ast.IfExp,
        ast.Lambda,
        ast.FormattedValue,
        ast.JoinedStr,
        ast.Starred,
        # Data structures
        ast.List,
        ast.Dict,
        ast.Tuple,
        ast.Set,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
        ast.comprehension,
        # Access
        ast.Subscript,
        ast.Attribute,
        ast.Slice,
        ast.Index,  # kept for Python 3.8 compat; no-op in 3.9+
        # Operators / context
        ast.Load,
        ast.Store,
        ast.Del,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.LShift,
        ast.RShift,
        ast.BitOr,
        ast.BitXor,
        ast.BitAnd,
        ast.MatMult,
        ast.And,
        ast.Or,
        ast.Not,
        ast.Invert,
        ast.UAdd,
        ast.USub,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.Is,
        ast.IsNot,
        ast.In,
        ast.NotIn,
        # Arguments
        ast.arguments,
        ast.arg,
        ast.keyword,
        # Other safe nodes
        ast.Await,
        ast.Yield,
    }
)

# Dunder attributes that enable sandbox escape via object introspection.
_BLOCKED_DUNDER_ATTRS: frozenset[str] = frozenset(
    {
        "__class__",
        "__bases__",
        "__subclasses__",
        "__globals__",
        "__builtins__",
        "__import__",
        "__loader__",
        "__spec__",
        "__code__",
        "__func__",
        "__self__",
        "__module__",
        "__dict__",
        "__mro__",
        "__init_subclass__",
        "__set_name__",
        "__reduce__",
        "__reduce_ex__",
        "__getattr__",
        "__setattr__",
        "__delattr__",
    }
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


class RLMSandboxTimeoutError(RuntimeError):
    """Raised when sandbox code exceeds execution time budget."""


class RLMSandbox:
    """AST-validated Python REPL with restricted builtins and helpers."""

    def __init__(
        self,
        state: Optional[REPLState] = None,
        *,
        lm_query_fn: Optional[Callable[[str], Any]] = None,
        max_sub_calls: int = 60,
        execution_timeout_seconds: Optional[float] = None,
    ) -> None:
        self.state = state or REPLState()
        self._lm_query_fn = lm_query_fn
        self._max_sub_calls = max(0, max_sub_calls)
        if execution_timeout_seconds is None:
            execution_timeout_seconds = float(os.environ.get("BIZRA_RLM_TIMEOUT", "10"))
        self._execution_timeout_seconds = max(0.1, float(execution_timeout_seconds))

    def validate_code(self, code: str) -> tuple[bool, str]:
        """Return `(allowed, reason)` after allowlist-based AST validation.

        Every AST node type must appear in ``_ALLOWED_AST_NODES``.
        Dunder attribute access and blocked built-in names are rejected
        regardless of node type.  This closes object-introspection escape
        surfaces that a denylist approach would miss.
        """
        stripped = _strip_code_fences(code)
        if not stripped.strip():
            return False, "empty code"

        try:
            tree = ast.parse(stripped)
        except SyntaxError as exc:
            return False, f"syntax error: {exc.msg}"

        for node in ast.walk(tree):
            # 1. Allowlist gate: reject any node type not explicitly permitted
            if type(node) not in _ALLOWED_AST_NODES:
                return False, f"blocked node: {type(node).__name__}"

            # 2. Block dangerous built-in / module names
            if isinstance(node, ast.Name) and node.id in _BLOCKED_NAMES:
                return False, f"blocked name: {node.id}"

            # 3. Block dunder attribute access (object-introspection escape)
            if isinstance(node, ast.Attribute) and node.attr in _BLOCKED_DUNDER_ATTRS:
                return False, f"blocked dunder attribute: {node.attr}"

            # 4. Block calls to blocked names
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
            except Exception as exc:  # noqa: BLE001 — boundary boundary
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
                self._exec_with_timeout(compiled, globals_ns, locals_ns)
        except RLMSandboxTimeoutError as exc:
            output = f"[SANDBOX_TIMEOUT] {exc}"
        except Exception as exc:  # noqa: BLE001 — boundary boundary
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

    def _exec_with_timeout(
        self,
        compiled: Any,
        globals_ns: dict[str, Any],
        locals_ns: dict[str, Any],
    ) -> None:
        timeout = self._execution_timeout_seconds
        message = f"sandbox execution exceeded {timeout:.2f}s"

        if (
            hasattr(signal, "SIGALRM")
            and threading.current_thread() is threading.main_thread()
        ):

            def _alarm_handler(signum: int, frame: Any) -> None:
                raise RLMSandboxTimeoutError(message)

            old_handler = signal.getsignal(signal.SIGALRM)
            signal.signal(signal.SIGALRM, _alarm_handler)
            signal.setitimer(signal.ITIMER_REAL, timeout)
            try:
                exec(compiled, globals_ns, locals_ns)  # noqa: S102
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, old_handler)
            return

        # Non-Unix or non-main-thread fallback.
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(exec, compiled, globals_ns, locals_ns)
            try:
                future.result(timeout=timeout)
            except concurrent.futures.TimeoutError as exc:
                raise RLMSandboxTimeoutError(message) from exc


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
