"""
Hook Condition Evaluator — Whitelist-Only Safe Expression Engine

Standing on Giants:
  Aho, Sethi, Ullman (1986) — Compilers: Principles, Techniques, and Tools
  Lamport (1978) — Time, Clocks, and the Ordering of Events

Evaluates conditions from hooks.yaml WITHOUT using eval() or exec().
Each condition is parsed into a safe AST of comparison/boolean nodes,
then evaluated against a runtime context dictionary.

Supported grammar (whitelist-only):
    expr     ::= comparison (('&&' | '||') comparison)*
    comparison ::= term (('<' | '<=' | '>' | '>=' | '==' | '!=') term)?
    term     ::= dotpath | STRING | NUMBER | BOOLEAN
    dotpath  ::= IDENT ('.' IDENT)*

Security Invariant (SEC-001):
    No dynamic code execution. No eval(). No builtins access.
    The evaluator is a pure function of (ast, context) → bool.

Created: 2026-02-23 | BIZRA Elite Integration v1.3.0
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional, Union

logger = logging.getLogger(__name__)

# ============================================================================
# AST NODES
# ============================================================================


class Op(str, Enum):
    """Comparison operators."""

    LT = "<"
    LE = "<="
    GT = ">"
    GE = ">="
    EQ = "=="
    NE = "!="


class BoolOp(str, Enum):
    """Boolean connectives."""

    AND = "&&"
    OR = "||"


@dataclass(frozen=True)
class Literal:
    """A literal value (string, number, or boolean)."""

    value: Union[str, int, float, bool]


@dataclass(frozen=True)
class DotPath:
    """A dotted path reference into the context (e.g., 'time.hour')."""

    segments: tuple[str, ...]


@dataclass(frozen=True)
class Comparison:
    """A binary comparison between two terms."""

    left: Union[Literal, DotPath]
    op: Op
    right: Union[Literal, DotPath]


@dataclass(frozen=True)
class BoolExpr:
    """A boolean combination of comparisons."""

    left: Union[Comparison, "BoolExpr", DotPath]
    op: BoolOp
    right: Union[Comparison, "BoolExpr", DotPath]


# Union of all expression node types
ExprNode = Union[Comparison, BoolExpr, DotPath, Literal]


# ============================================================================
# TOKENIZER
# ============================================================================

# Regex patterns for tokenization
_TOKEN_PATTERNS = [
    (r"&&", "AND"),
    (r"\|\|", "OR"),
    (r"<=", "LE"),
    (r">=", "GE"),
    (r"<", "LT"),
    (r">", "GT"),
    (r"==", "EQ"),
    (r"!=", "NE"),
    (r"-?\d+\.\d+", "FLOAT"),
    (r"-?\d+[hms]", "DURATION"),
    (r"-?\d+", "INT"),
    (r"true|false", "BOOL"),
    (r"'[^']*'", "STRING"),
    (r'"[^"]*"', "STRING"),
    (r"[a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*", "IDENT"),
    (r"\s+", "WS"),
]

_TOKEN_RE = re.compile(
    "|".join(f"(?P<{name}>{pattern})" for pattern, name in _TOKEN_PATTERNS)
)


@dataclass
class Token:
    """Lexer token."""

    kind: str
    value: str


def _tokenize(expr: str) -> list[Token]:
    """Tokenize a condition expression."""
    tokens: list[Token] = []
    for match in _TOKEN_RE.finditer(expr):
        kind = match.lastgroup
        if kind is None or kind == "WS":
            continue
        tokens.append(Token(kind=kind, value=match.group()))
    return tokens


# ============================================================================
# PARSER
# ============================================================================


class _Parser:
    """Recursive-descent parser for condition expressions."""

    def __init__(self, tokens: list[Token]) -> None:
        self._tokens = tokens
        self._pos = 0

    def _peek(self) -> Optional[Token]:
        if self._pos < len(self._tokens):
            return self._tokens[self._pos]
        return None

    def _advance(self) -> Token:
        token = self._tokens[self._pos]
        self._pos += 1
        return token

    def parse(self) -> ExprNode:
        """Parse the full expression."""
        node = self._parse_or()
        if self._pos < len(self._tokens):
            raise ConditionParseError(
                f"Unexpected token: {self._tokens[self._pos].value}"
            )
        return node

    def _parse_or(self) -> ExprNode:
        """Parse OR expressions (lowest precedence)."""
        left = self._parse_and()
        while self._peek() and self._peek().kind == "OR":  # type: ignore[union-attr]
            self._advance()
            right = self._parse_and()
            left = BoolExpr(left=left, op=BoolOp.OR, right=right)
        return left

    def _parse_and(self) -> ExprNode:
        """Parse AND expressions."""
        left = self._parse_comparison()
        while self._peek() and self._peek().kind == "AND":  # type: ignore[union-attr]
            self._advance()
            right = self._parse_comparison()
            left = BoolExpr(left=left, op=BoolOp.AND, right=right)
        return left

    def _parse_comparison(self) -> ExprNode:
        """Parse comparison expressions."""
        left = self._parse_term()
        tok = self._peek()
        if tok and tok.kind in {"LT", "LE", "GT", "GE", "EQ", "NE"}:
            self._advance()
            op_map = {
                "LT": Op.LT,
                "LE": Op.LE,
                "GT": Op.GT,
                "GE": Op.GE,
                "EQ": Op.EQ,
                "NE": Op.NE,
            }
            right = self._parse_term()
            return Comparison(left=left, op=op_map[tok.kind], right=right)
        return left

    def _parse_term(self) -> Union[Literal, DotPath]:
        """Parse a terminal value."""
        tok = self._peek()
        if tok is None:
            raise ConditionParseError("Unexpected end of expression")

        if tok.kind == "INT":
            self._advance()
            return Literal(value=int(tok.value))
        elif tok.kind == "FLOAT":
            self._advance()
            return Literal(value=float(tok.value))
        elif tok.kind == "BOOL":
            self._advance()
            return Literal(value=tok.value == "true")
        elif tok.kind == "STRING":
            self._advance()
            return Literal(value=tok.value.strip("'\""))
        elif tok.kind == "DURATION":
            self._advance()
            return Literal(value=_parse_duration(tok.value))
        elif tok.kind == "IDENT":
            self._advance()
            segments = tuple(tok.value.split("."))
            return DotPath(segments=segments)
        else:
            raise ConditionParseError(f"Unexpected token: {tok.value}")


def _parse_duration(value: str) -> int:
    """Parse a duration string like '24h', '5m', '30s' into seconds."""
    suffix = value[-1]
    num = int(value[:-1])
    if suffix == "h":
        return num * 3600
    elif suffix == "m":
        return num * 60
    elif suffix == "s":
        return num
    raise ConditionParseError(f"Unknown duration suffix: {suffix}")


# ============================================================================
# EVALUATOR
# ============================================================================


def _resolve(node: Union[Literal, DotPath], context: dict[str, Any]) -> Any:
    """Resolve a term to a concrete value against the context."""
    if isinstance(node, Literal):
        return node.value
    # DotPath: traverse context dict
    current: Any = context
    for segment in node.segments:
        if isinstance(current, dict):
            if segment not in current:
                return None
            current = current[segment]
        elif hasattr(current, segment):
            current = getattr(current, segment)
        else:
            return None
    return current


def _compare(left: Any, op: Op, right: Any) -> bool:
    """Perform a safe comparison."""
    if left is None or right is None:
        return False
    try:
        if op == Op.LT:
            return left < right
        elif op == Op.LE:
            return left <= right
        elif op == Op.GT:
            return left > right
        elif op == Op.GE:
            return left >= right
        elif op == Op.EQ:
            return left == right
        elif op == Op.NE:
            return left != right
    except TypeError:
        return False
    return False  # pragma: no cover


def _evaluate(node: ExprNode, context: dict[str, Any]) -> bool:
    """Evaluate an AST node against a context."""
    if isinstance(node, Comparison):
        left_val = _resolve(node.left, context)
        right_val = _resolve(node.right, context)
        return _compare(left_val, node.op, right_val)
    elif isinstance(node, BoolExpr):
        left_val = _evaluate(node.left, context)
        if node.op == BoolOp.AND:
            return left_val and _evaluate(node.right, context)
        else:
            return left_val or _evaluate(node.right, context)
    elif isinstance(node, DotPath):
        # Bare identifier is truthy check
        return bool(_resolve(node, context))
    elif isinstance(node, Literal):
        return bool(node.value)
    return False


# ============================================================================
# PUBLIC API
# ============================================================================


class ConditionParseError(ValueError):
    """Raised when a condition expression cannot be parsed."""


class HookConditionEvaluator:
    """
    Evaluates hook conditions from hooks.yaml safely.

    Thread-safe: stateless evaluation. Each call is a pure function.

    Usage:
        evaluator = HookConditionEvaluator()
        ctx = evaluator.build_context()  # auto-populates time, system state
        result = evaluator.evaluate("time.hour >= 6 && time.hour <= 10", ctx)
    """

    def __init__(self) -> None:
        self._cache: dict[str, ExprNode] = {}

    def parse(self, condition: str) -> ExprNode:
        """
        Parse a condition string into an AST.

        Results are cached for repeated evaluation.
        """
        if condition in self._cache:
            return self._cache[condition]

        tokens = _tokenize(condition)
        if not tokens:
            raise ConditionParseError(f"Empty condition: {condition!r}")

        parser = _Parser(tokens)
        ast = parser.parse()
        self._cache[condition] = ast
        return ast

    def evaluate(self, condition: str, context: dict[str, Any]) -> bool:
        """
        Evaluate a condition string against a context.

        Args:
            condition: The condition expression from hooks.yaml
            context: Runtime context dict with dotpath-accessible values

        Returns:
            True if condition is met, False otherwise
        """
        try:
            ast = self.parse(condition)
            return _evaluate(ast, context)
        except ConditionParseError:
            logger.warning(f"Failed to parse condition: {condition!r}")
            return False
        except Exception as e:  # noqa: BLE001 — boundary boundary
            logger.error(f"Condition evaluation error: {e}")
            return False

    def build_context(
        self,
        *,
        session_data: Optional[dict[str, Any]] = None,
        task_data: Optional[dict[str, Any]] = None,
        message_data: Optional[dict[str, Any]] = None,
        system_data: Optional[dict[str, Any]] = None,
    ) -> dict[str, Any]:
        """
        Build a runtime context dictionary for condition evaluation.

        Auto-populates temporal fields from current system time.
        """
        now = datetime.now(timezone.utc)
        context: dict[str, Any] = {
            "time": {
                "hour": now.hour,
                "minute": now.minute,
                "weekday": now.weekday(),
                "day": now.day,
                "month": now.month,
                "year": now.year,
                "is_morning": 6 <= now.hour <= 10,
                "is_evening": 18 <= now.hour <= 22,
                "is_weekend": now.weekday() >= 5,
            },
        }

        if session_data is not None:
            context["session"] = session_data
        if task_data is not None:
            context["task"] = task_data
        if message_data is not None:
            context["message"] = message_data
        if system_data is not None:
            context["system"] = system_data

        # Convenience boolean flags derived from context
        context["has_recent_topic"] = bool(
            session_data and session_data.get("recent_topic")
        )
        context["is_high_stakes"] = bool(
            message_data and message_data.get("high_stakes")
        )
        context["contains_important_info"] = bool(
            message_data and message_data.get("important")
        )

        return context
