"""
BIZRA CLI — Modular Command Registry
=====================================

Decomposes the monolithic bizra_cli.py into focused command modules
with performance monitoring and hooks integration.

Standing on Giants:
- Thompson & Ritchie (1973): Unix CLI conventions
- GNU (1987): Long-form argument standards
- Deming (1950): Measure everything, ratchet quality
"""

from .registry import CommandRegistry, CommandResult, BaseCommand

__all__ = ["CommandRegistry", "CommandResult", "BaseCommand"]
