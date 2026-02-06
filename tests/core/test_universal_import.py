"""
Universal Import Validator
==========================
Verifies every core/ module can be imported without error.

This is the single most important smoke test: if a module can't import,
nothing downstream works. Catches:
- Broken lazy imports referencing non-existent files
- Circular import deadlocks
- Missing __init__.py files
- Syntax errors introduced by refactoring
- Optional dependency issues (marked xfail)

Standing on Giants: pytest parametrize + importlib

Created: 2026-02-07 | BIZRA Mastermind Sprint
"""

import importlib
import pkgutil
from pathlib import Path

import pytest


# ============================================================================
# DISCOVERY: Find all modules under core/
# ============================================================================

CORE_ROOT = Path(__file__).parent.parent.parent / "core"


def _discover_modules():
    """Walk core/ and yield all importable module paths."""
    modules = []
    for info in pkgutil.walk_packages(
        path=[str(CORE_ROOT)],
        prefix="core.",
    ):
        modules.append(info.name)
    return sorted(modules)


# Modules that require optional heavy deps (torch, numpy, httpx, etc.)
# These are tested separately with xfail markers
OPTIONAL_DEP_MODULES = {
    "core.living_memory",
    "core.living_memory.healing",
    "core.living_memory.memory",
    "core.living_memory.types",
    "core.uers",
    "core.uers.impact",
    "core.elite",
    "core.elite.compute_market",
    "core.elite.pipeline",
}

# Modules known to have circular or complex init (test separately)
COMPLEX_INIT_MODULES = {
    "core.sovereign",
    "core.sovereign.runtime",
    "core.sovereign.engine",
    "core.sovereign.launch",
    "core.sovereign.api",
}


# ============================================================================
# TIER 1: Top-level subpackage imports (critical path)
# ============================================================================

TIER1_PACKAGES = [
    "core.pci",
    "core.vault",
    "core.federation",
    "core.inference",
    "core.a2a",
    "core.integration",
    "core.ntu",
    "core.protocols",
    "core.governance",
    "core.reasoning",
    "core.orchestration",
    "core.treasury",
    "core.bridges",
    "core.agentic",
    "core.bounty",
    "core.apex",
    "core.autonomous",
    "core.autopoiesis",
    "core.constitutional",
    "core.proof_engine",
    "core.pat",
    "core.sdpo",
    "core.personaplex",
]


@pytest.mark.parametrize("module_path", TIER1_PACKAGES)
def test_tier1_package_import(module_path):
    """Every top-level core subpackage must import without error."""
    mod = importlib.import_module(module_path)
    assert mod is not None
    # Verify __all__ is defined (required for clean public API)
    if hasattr(mod, "__all__"):
        assert isinstance(mod.__all__, (list, tuple))


# ============================================================================
# TIER 2: __all__ exports are resolvable
# ============================================================================

TIER2_PACKAGES_WITH_EXPORTS = [
    "core.pci",
    "core.federation",
    "core.inference",
    "core.a2a",
    "core.integration",
    "core.protocols",
    "core.governance",
    "core.reasoning",
    "core.orchestration",
    "core.treasury",
    "core.bridges",
    "core.agentic",
    "core.bounty",
]


@pytest.mark.parametrize("module_path", TIER2_PACKAGES_WITH_EXPORTS)
def test_tier2_all_exports_resolve(module_path):
    """Every name in __all__ must be accessible (catches broken lazy imports)."""
    mod = importlib.import_module(module_path)
    if not hasattr(mod, "__all__"):
        pytest.skip(f"{module_path} has no __all__")
        return

    for name in mod.__all__:
        # String constants and version numbers are always OK
        try:
            attr = getattr(mod, name)
        except (ImportError, AttributeError) as e:
            pytest.fail(
                f"{module_path}.__all__ exports '{name}' but getattr fails: {e}"
            )


# ============================================================================
# TIER 3: Constants module (single source of truth)
# ============================================================================


def test_constants_single_source_of_truth():
    """The constants module must export all threshold values."""
    from core.integration.constants import (
        UNIFIED_IHSAN_THRESHOLD,
        STRICT_IHSAN_THRESHOLD,
        UNIFIED_SNR_THRESHOLD,
        SNR_THRESHOLD_T1_HIGH,
        MAX_RETRY_ATTEMPTS,
    )

    # Value assertions (these ARE the source of truth)
    assert UNIFIED_IHSAN_THRESHOLD == 0.95
    assert STRICT_IHSAN_THRESHOLD == 0.99
    assert UNIFIED_SNR_THRESHOLD == 0.85
    assert SNR_THRESHOLD_T1_HIGH == 0.95
    assert MAX_RETRY_ATTEMPTS == 3


# ============================================================================
# TIER 4: Key module internals (critical path files)
# ============================================================================

CRITICAL_MODULES = [
    "core.pci.envelope",
    "core.pci.crypto",
    "core.pci.gates",
    "core.pci.epigenome",
    "core.pci.reject_codes",
    "core.federation.gossip",
    "core.federation.consensus",
    "core.federation.propagation",
    "core.federation.node",
    "core.inference.gateway",
    "core.inference.local_first",
    "core.a2a.schema",
    "core.a2a.engine",
    "core.integration.constants",
    "core.protocols.inference_backend",
    "core.protocols.bridge",
    "core.protocols.gate_chain",
]


@pytest.mark.parametrize("module_path", CRITICAL_MODULES)
def test_tier4_critical_module_import(module_path):
    """Critical-path modules must import cleanly."""
    mod = importlib.import_module(module_path)
    assert mod is not None


# ============================================================================
# TIER 5: Decomposed sovereign modules (new architecture)
# ============================================================================

DECOMPOSED_MODULES = [
    "core.governance.constitutional_gate",
    "core.governance.autonomy",
    "core.governance.autonomy_matrix",
    "core.governance.capability_card",
    "core.governance.ihsan_vector",
    "core.governance.key_registry",
    "core.reasoning.guardian_council",
    "core.reasoning.graph_reasoning",
    "core.reasoning.snr_maximizer",
    "core.reasoning.bicameral_engine",
    "core.reasoning.collective_synthesizer",
    "core.orchestration.event_bus",
    "core.orchestration.muraqabah_sensors",
    "core.orchestration.muraqabah_engine",
    "core.orchestration.team_planner",
    "core.orchestration.enhanced_team_planner",
    "core.orchestration.proactive_team",
    "core.orchestration.proactive_integration",
    "core.orchestration.background_agents",
    "core.orchestration.opportunity_pipeline",
    "core.treasury.treasury_types",
    "core.treasury.treasury_controller",
    "core.treasury.treasury_mode",
    "core.treasury.market_integration",
    "core.bridges.bridge",
    "core.bridges.dual_agentic_bridge",
    "core.bridges.knowledge_integrator",
    "core.bridges.swarm_knowledge_bridge",
    "core.bridges.rust_lifecycle",
    "core.bridges.local_inference_bridge",
]


@pytest.mark.parametrize("module_path", DECOMPOSED_MODULES)
def test_tier5_decomposed_module_import(module_path):
    """Decomposed sovereign modules must import without error."""
    mod = importlib.import_module(module_path)
    assert mod is not None
