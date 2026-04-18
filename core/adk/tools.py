"""Tool decorator and registry for BIZRA agents."""

from __future__ import annotations

import functools
from typing import Any, Callable


def tool(fn: Callable | None = None, *, max_results: int = 50) -> Callable:
    """Mark a method as an agent tool.

    Decorated methods are tracked in the agent's tool registry and each
    invocation is counted against the mission's tool-call budget.
    """

    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(self: Any, *args: Any, **kwargs: Any) -> Any:
            if hasattr(self, "_active_mission") and self._active_mission is not None:
                self._active_mission.consume_tool_call()
            result = func(self, *args, **kwargs)
            if isinstance(result, list) and len(result) > max_results:
                result = result[:max_results]
            return result

        wrapper._is_bizra_tool = True
        wrapper._max_results = max_results
        wrapper._tool_name = func.__name__
        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


def get_tools(agent: Any) -> list[str]:
    """Return names of all @tool-decorated methods on an agent."""
    results = []
    for name in type(agent).__mro__[0].__dict__:
        attr = getattr(type(agent), name, None)
        if callable(attr) and getattr(attr, "_is_bizra_tool", False):
            results.append(name)
    return results
