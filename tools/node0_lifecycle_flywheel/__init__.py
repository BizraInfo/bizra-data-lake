"""Node0 lifecycle flywheel harness.

Advisory by default. Use the CLI module for operator-run receipts:

    python -m tools.node0_lifecycle_flywheel.closed_loop
"""

__all__ = [
    "STATUS_GATES",
    "build_receipt",
    "decide_next_action",
]


def __getattr__(name: str):
    if name not in __all__:
        raise AttributeError(name)
    from . import closed_loop

    return getattr(closed_loop, name)
