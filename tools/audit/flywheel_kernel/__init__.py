"""BIZRA Autonomous Flywheel Kernel v1."""

__all__ = [
    "FlywheelState",
    "GuardResult",
    "PriorityDecision",
    "build_report",
    "decide_priority",
    "evaluate_guards",
    "load_audit_state",
    "should_trigger_audit",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    from . import kernel

    return getattr(kernel, name)
