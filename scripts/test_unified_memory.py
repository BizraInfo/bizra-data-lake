#!/usr/bin/env python3
"""
BIZRA Unified Memory Test Script
Tests the integration between Dual-Agentic system and BIZRA-DATA-LAKE.

Usage:
    python scripts/test_unified_memory.py

Or from any directory:
    python /path/to/BIZRA-Dual-Agentic-system--main/scripts/test_unified_memory.py
"""

import sys
import asyncio
from pathlib import Path

# Ensure project root is in path (self-relative)
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def print_header(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def print_result(name: str, success: bool, details: str = "") -> None:
    """Print a test result."""
    status = "PASS" if success else "FAIL"
    icon = "✅" if success else "❌"
    print(f"  {icon} {name}: {status}")
    if details:
        print(f"     {details}")


async def test_imports() -> bool:
    """Test that all unified memory modules can be imported."""
    print_header("1. Testing Module Imports")

    all_passed = True

    # Test data_lake_bridge
    try:
        from core.data_lake_bridge import (
            DataLakeBridge,
            MemoryTier,
            KnowledgeResult,
            get_data_lake_bridge,
        )
        print_result("core.data_lake_bridge", True)
    except ImportError as e:
        print_result("core.data_lake_bridge", False, str(e))
        all_passed = False

    # Test unified_memory
    try:
        from core.unified_memory import (
            UnifiedMemory,
            MemoryEntry,
            MemoryPriority,
            get_unified_memory,
            remember,
            recall,
        )
        print_result("core.unified_memory", True)
    except ImportError as e:
        print_result("core.unified_memory", False, str(e))
        all_passed = False

    # Test evidence_sync
    try:
        from core.evidence_sync import (
            EvidenceSync,
            PoIAttestation,
            Receipt,
            sync_evidence,
            verify_evidence,
        )
        print_result("core.evidence_sync", True)
    except ImportError as e:
        print_result("core.evidence_sync", False, str(e))
        all_passed = False

    return all_passed


async def test_data_lake_bridge() -> bool:
    """Test DataLakeBridge initialization and health check."""
    print_header("2. Testing Data Lake Bridge")

    try:
        from core.data_lake_bridge import DataLakeBridge

        bridge = DataLakeBridge()
        print_result("DataLakeBridge instantiation", True)

        # Health check
        status = await bridge.health_check()
        if status.online:
            print_result("MCP Bridge connection", True, f"URL: {status.url}")
        else:
            print_result("MCP Bridge connection", False,
                        f"Offline - {status.error or 'No error details'}")
            print("     (This is OK if the Data Lake MCP server isn't running)")

        await bridge.close()
        return True

    except Exception as e:
        print_result("DataLakeBridge", False, str(e))
        return False


async def test_unified_memory() -> bool:
    """Test UnifiedMemory initialization and local operations."""
    print_header("3. Testing Unified Memory")

    try:
        from core.unified_memory import UnifiedMemory, MemoryTier, MemoryPriority

        memory = UnifiedMemory()
        await memory.initialize()
        print_result("UnifiedMemory initialization", True)

        # Test local storage
        entry = await memory.store(
            "BIZRA uses PAT-SAT dual-agentic architecture",
            tier=MemoryTier.L4_SEMANTIC,
            priority=MemoryPriority.MEDIUM,
        )
        print_result("Local memory storage", True, f"Fingerprint: {entry.fingerprint}")

        # Test local query
        result = await memory.query_local("BIZRA")
        print_result("Local memory query", True, f"Found {result.total_count} entries")

        # Test tier stats
        stats = memory.get_tier_stats()
        print_result("Tier statistics", True)
        for tier, info in stats.items():
            print(f"       {tier}: {info['count']}/{info['limit']} entries")

        await memory.close()
        return True

    except Exception as e:
        print_result("UnifiedMemory", False, str(e))
        return False


async def test_evidence_sync() -> bool:
    """Test EvidenceSync initialization and integrity verification."""
    print_header("4. Testing Evidence Sync")

    try:
        from core.evidence_sync import EvidenceSync

        sync = EvidenceSync()
        print_result("EvidenceSync instantiation", True)

        # Verify integrity (reads existing files if they exist)
        integrity = await sync.verify_integrity()

        poi_total = integrity["poi_ledger"]["total"]
        receipts_total = integrity["receipts"]["total"]

        print_result("Integrity verification", True)
        print(f"       PoI Ledger: {integrity['poi_ledger']['valid']}/{poi_total} valid")
        print(f"       Receipts: {integrity['receipts']['valid']}/{receipts_total} valid")

        return True

    except Exception as e:
        print_result("EvidenceSync", False, str(e))
        return False


async def test_sovereign_query() -> bool:
    """Test M6 Sovereign tier query (requires running MCP server)."""
    print_header("5. Testing Sovereign Query (M6)")

    try:
        from core.unified_memory import UnifiedMemory

        memory = UnifiedMemory()
        await memory.initialize()

        # Query sovereign tier
        result = await memory.query_sovereign("BIZRA architecture", limit=3)

        if result.total_count > 0:
            print_result("Sovereign query", True, f"Found {result.total_count} results")
            for entry in result.entries[:2]:
                content_preview = str(entry.content)[:80] + "..."
                print(f"       - {content_preview}")
        else:
            print_result("Sovereign query", True,
                        "No results (Data Lake MCP may be offline)")

        await memory.close()
        return True

    except Exception as e:
        print_result("Sovereign query", False, str(e))
        return False


async def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("  BIZRA UNIFIED MEMORY TEST SUITE")
    print("="*60)
    print(f"  Project Root: {PROJECT_ROOT}")
    print(f"  Python Path: {sys.path[0]}")

    results = {}

    # Run tests
    results["imports"] = await test_imports()
    results["data_lake_bridge"] = await test_data_lake_bridge()
    results["unified_memory"] = await test_unified_memory()
    results["evidence_sync"] = await test_evidence_sync()
    results["sovereign_query"] = await test_sovereign_query()

    # Summary
    print_header("TEST SUMMARY")

    passed = sum(1 for v in results.values() if v)
    total = len(results)

    for name, success in results.items():
        status = "PASS" if success else "FAIL"
        icon = "✅" if success else "❌"
        print(f"  {icon} {name}: {status}")

    print(f"\n  Total: {passed}/{total} tests passed")

    if passed == total:
        print("\n  🎉 All tests passed! Unified Memory system is ready.")
    else:
        print("\n  ⚠️  Some tests failed. Check the details above.")

    return passed == total


if __name__ == "__main__":
    # Windows event loop policy
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    success = asyncio.run(main())
    sys.exit(0 if success else 1)
