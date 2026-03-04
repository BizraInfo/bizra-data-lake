"""BIZRA Constitution v5.0.0-GENESIS — typed parser and runtime gates."""

try:
    from .constitution import (
        Constitution,
        ConstitutionalViolation,
        load_constitution,
    )
except ImportError:
    # Hyphen dir can't be a Python package directly — imports work
    # through the bizra_constitution symlink instead.
    Constitution = None  # type: ignore[assignment,misc]
    ConstitutionalViolation = None  # type: ignore[assignment,misc]
    load_constitution = None  # type: ignore[assignment]

__all__ = [
    "Constitution",
    "ConstitutionalViolation",
    "load_constitution",
]
