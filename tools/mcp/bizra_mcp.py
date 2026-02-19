#!/usr/bin/env python3
"""
BIZRA MCP SERVER -- Unified Ecosystem Interface

Exposes the BIZRA DDAGI OS v3.0.0 via the Model Context Protocol (MCP).

CAPABILITIES:
  - Deep Cognitive Query (via UltimateEngine + Orchestrator + Peak)
  - System Health & Invariance Checks
  - MCP Server Health Monitoring

V3 Optimized: 2026-02-18 | Timeouts, health monitoring
Author: BIZRA Genesis NODE0
"""

import asyncio
import logging
import os
import sys
import time
from typing import Dict, Any

# Set up path to ensure tools/ sibling imports work
_mcp_dir = os.path.dirname(os.path.abspath(__file__))
_tools_dir = os.path.dirname(_mcp_dir)          # tools/
_project_root = os.path.dirname(_tools_dir)      # BIZRA-DATA-LAKE/
sys.path.insert(0, os.path.join(_tools_dir, "bridges"))
sys.path.insert(0, os.path.join(_tools_dir, "engines"))
sys.path.insert(0, _project_root)

# MCP Framework
from fastmcp import FastMCP

# BIZRA Ecosystem
from ecosystem_bridge import (
    initialize_ecosystem,
    UnifiedQuery,
    EcosystemBridge,
    UnifiedResponse
)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] MCP | %(message)s',
    handlers=[logging.StreamHandler(sys.stderr)]
)
logger = logging.getLogger("BIZRA-MCP")

# ===============================================================================
# CONSTANTS
# ===============================================================================

TOOL_TIMEOUT_SECONDS = 30.0

# ===============================================================================
# SERVER METRICS
# ===============================================================================

_server_start_time = time.monotonic()
_query_count = 0
_error_count = 0
_total_response_time = 0.0

# ===============================================================================
# SERVER DEFINITION
# ===============================================================================

mcp = FastMCP("BIZRA DDAGI OS")

# Global Ecosystem Instance
_ecosystem_instance: EcosystemBridge | None = None

async def get_instance() -> EcosystemBridge:
    """Singleton accessor for the ecosystem."""
    global _ecosystem_instance
    if _ecosystem_instance is None:
        logger.info("Initializing BIZRA Ecosystem...")
        _ecosystem_instance = await initialize_ecosystem()
        logger.info("Ecosystem initialized")
    return _ecosystem_instance


# ===============================================================================
# TOOLS
# ===============================================================================

@mcp.tool(description="Execute a cognitive query against the BIZRA DDAGI ecosystem")
async def query_bizra(
    query: str,
    require_deep_scan: bool = True,
    snr_threshold: float = 0.85
) -> str:
    """
    Process a natural language query through the BIZRA Unified Ecosystem.

    Args:
        query: The user's question or instruction.
        require_deep_scan: If True, engages the Orchestrator and Apex (slower, deeper).
        snr_threshold: Minimum Signal-to-Noise Ratio acceptable (0.0-1.0).

    Returns:
        The synthesized answer with provenance and confidence scores.
    """
    global _query_count, _error_count, _total_response_time
    _query_count += 1
    start = time.perf_counter()

    try:
        ecosystem = await asyncio.wait_for(get_instance(), timeout=TOOL_TIMEOUT_SECONDS)

        # Construct unified query context
        uq = UnifiedQuery(
            text=query,
            require_constitution_check=True,
            require_daughter_test=True,
            snr_threshold=snr_threshold,
            use_orchestrator=require_deep_scan,
            use_apex=require_deep_scan,
            metadata={"source": "mcp_client"}
        )

        # Execute with timeout
        result: UnifiedResponse = await asyncio.wait_for(
            ecosystem.query(uq),
            timeout=TOOL_TIMEOUT_SECONDS
        )
        elapsed = time.perf_counter() - start
        _total_response_time += elapsed * 1000

        return f"""# BIZRA System Response
-----------------------
{result.synthesis}

# Meta-Analysis
- SNR Score: {result.snr_score:.3f}
- Ihsan Score: {result.ihsan_score:.3f}
- Execution: {elapsed:.2f}s
- Constitution: {'PASSED' if result.constitution_check else 'FAILED'}
- Daughter Test: {'PASSED' if result.daughter_test_check else 'FAILED'}
- Component Trace: {', '.join(result.components_used)}
- Sources: {len(result.sources)}"""

    except asyncio.TimeoutError:
        _error_count += 1
        elapsed = time.perf_counter() - start
        _total_response_time += elapsed * 1000
        return f"Error: Query exceeded {TOOL_TIMEOUT_SECONDS}s timeout after {elapsed:.1f}s"
    except Exception as e:
        _error_count += 1
        elapsed = time.perf_counter() - start
        _total_response_time += elapsed * 1000
        logger.error(f"Query error: {e}")
        return f"Error: {e}"


@mcp.tool(description="Get the current health and status of the BIZRA OS Kernel")
async def get_system_health() -> Dict[str, Any]:
    """
    Returns diagnostic information about the ecosystem engines.
    Verifies Kernel Invariants (RIBA_ZERO, ZANN_ZERO, IHSAN_FLOOR).
    """
    global _query_count, _error_count, _total_response_time
    _query_count += 1
    start = time.perf_counter()

    try:
        ecosystem = await asyncio.wait_for(get_instance(), timeout=TOOL_TIMEOUT_SECONDS)
        health = ecosystem.get_health()
        status = ecosystem.get_status()

        elapsed = time.perf_counter() - start
        _total_response_time += elapsed * 1000

        return {
            "status": "operational" if health.overall_health > 0.8 else "degraded",
            "health_score": f"{health.overall_health * 100:.1f}%",
            "invariants_secure": health.kernel_invariants_ok,
            "uptime_hours": status["uptime_hours"],
            "components": health.to_dict()
        }

    except asyncio.TimeoutError:
        _error_count += 1
        elapsed = time.perf_counter() - start
        _total_response_time += elapsed * 1000
        return {"error": "timeout", "message": f"Health check exceeded {TOOL_TIMEOUT_SECONDS}s timeout"}
    except Exception as e:
        _error_count += 1
        elapsed = time.perf_counter() - start
        _total_response_time += elapsed * 1000
        logger.error(f"Health check error: {e}")
        return {"error": str(e)}


@mcp.tool(description="Get MCP server performance metrics: uptime, query count, cache hit rate, errors, avg response time.")
async def mcp_health() -> Dict[str, Any]:
    """Returns server-level health metrics."""
    uptime = time.monotonic() - _server_start_time
    return {
        "server": "bizra-ddagi-mcp",
        "version": "3.0.0",
        "uptime_seconds": round(uptime, 1),
        "query_count": _query_count,
        "error_count": _error_count,
        "avg_response_ms": round(_total_response_time / _query_count, 2) if _query_count > 0 else 0.0,
    }


# ===============================================================================
# ENTRY POINT
# ===============================================================================

if __name__ == "__main__":
    mcp.run()
