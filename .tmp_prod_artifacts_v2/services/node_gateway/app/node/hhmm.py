from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class _Rule:
    macro_state: str
    keywords: tuple[str, ...]


class HHMM:
    """Lightweight deterministic macro-state classifier.

    The implementation is intentionally rule-based for predictable behavior in
    constrained deployment modes. It can be replaced by a learned HHMM later
    without changing the gateway API.
    """

    _RULES: tuple[_Rule, ...] = (
        _Rule("COMPOSE_EMAIL", ("email", "mail", "reply", "subject", "inbox")),
        _Rule("RESEARCH_WEB", ("research", "find", "search", "latest", "source")),
        _Rule("DESKTOP_AUTOMATION", ("desktop", "click", "window", "ahk", "open")),
        _Rule("CODE_IMPLEMENTATION", ("code", "implement", "refactor", "test", "fix")),
        _Rule("SYSTEM_OPERATIONS", ("deploy", "docker", "k8s", "cluster", "ci")),
    )

    def predict(self, text: str, context: dict[str, str] | None = None) -> str:
        haystack = self._normalize(text=text, context=context or {})
        for rule in self._RULES:
            if any(keyword in haystack for keyword in rule.keywords):
                return rule.macro_state
        return "GENERAL_REASONING"

    @staticmethod
    def _normalize(text: str, context: dict[str, str]) -> str:
        parts = [text or ""]
        if context:
            try:
                parts.append(json.dumps(context, ensure_ascii=True, sort_keys=True))
            except Exception:
                parts.append(str(context))
        return " ".join(parts).lower()
