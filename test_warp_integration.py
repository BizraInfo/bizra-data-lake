# BIZRA WARP Integration Test
# Verifies XTR-WARP integration with Data Lake

import sys
import time
from pathlib import Path

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent))

def test_warp_config():
    """Test WARP configuration in bizra_config.py"""
    print("\n" + "="*60)
    print("TEST 1: WARP Configuration")
    print("="*60)
    
    from bizra_config import (
        WARP_INDEX_ROOT, WARP_EXPERIMENT_ROOT, WARP_CHECKPOINT,
        WARP_NBITS, WARP_ENABLED
    )
    
    print(f"  WARP_ENABLED: {WARP_ENABLED}")
    print(f"  WARP_INDEX_ROOT: {WARP_INDEX_ROOT}")
    print(f"  WARP_EXPERIMENT_ROOT: {WARP_EXPERIMENT_ROOT}")
    print(f"  WARP_CHECKPOINT: {WARP_CHECKPOINT}")
    print(f"  WARP_NBITS: {WARP_NBITS}")
    
    assert WARP_ENABLED, "WARP should be enabled"
    assert WARP_INDEX_ROOT.exists(), f"WARP_INDEX_ROOT should exist: {WARP_INDEX_ROOT}"
    print("  ✓ Configuration valid")
    return True


def test_warp_bridge_import():
    """Test WARP bridge module import"""
    print("\n" + "="*60)
    print("TEST 2: WARP Bridge Import")
    print("="*60)
    
    try:
        from warp_bridge import WARPBridge, WARPStatus, WARPResult, WARPQueryResponse
        print("  ✓ WARPBridge imported successfully")
        print("  ✓ WARPStatus imported successfully")
        print("  ✓ WARPResult imported successfully")
        print("  ✓ WARPQueryResponse imported successfully")
        return True
    except ImportError as e:
        print(f"  ✗ Import failed: {e}")
        return False


def test_warp_bridge_init():
    """Test WARP bridge initialization"""
    print("\n" + "="*60)
    print("TEST 3: WARP Bridge Initialization")
    print("="*60)
    
    from warp_bridge import WARPBridge, WARPStatus
    
    bridge = WARPBridge(lazy_init=True)
    
    print(f"  Status: {bridge.status.value}")
    print(f"  WARP Available: {bridge._warp_available}")
    
    stats = bridge.get_stats()
    print(f"  Engine: {stats['engine']}")
    print(f"  Checkpoint: {stats['checkpoint']}")
    
    health, msg = bridge.health_check()
    print(f"  Health Check: {health} - {msg}")
    
    print("  ✓ Bridge initialized (lazy mode)")
    return True


def test_nexus_warp_adapter():
    """Test WARP adapter in Nexus"""
    print("\n" + "="*60)
    print("TEST 4: Nexus WARP Adapter")
    print("="*60)
    
    try:
        from bizra_nexus import WARPAdapter, NexusConfig
        
        config = NexusConfig()
        adapter = WARPAdapter(config)
        
        print(f"  Adapter Name: {adapter.name}")
        
        # Initialize
        adapter.initialize()
        
        # Health check (method is health_check, not check_health)
        health = adapter.health_check()
        print(f"  Health Status: {health.status.name}")
        print(f"  Health Message: {health.message}")
        
        print("  ✓ WARP adapter registered in Nexus")
        return True
        
    except Exception as e:
        print(f"  ✗ Adapter test failed: {e}")
        return False


def test_hypergraph_warp_mode():
    """Test WARP retrieval mode in Hypergraph Engine"""
    print("\n" + "="*60)
    print("TEST 5: Hypergraph WARP Mode")
    print("="*60)
    
    try:
        from hypergraph_engine import RetrievalMode
        
        modes = [m.value for m in RetrievalMode]
        print(f"  Available modes: {modes}")
        
        assert "warp" in modes, "WARP mode should be available"
        print("  ✓ WARP retrieval mode available")
        return True
        
    except Exception as e:
        print(f"  ✗ Mode test failed: {e}")
        return False


def test_warp_fallback():
    """Test WARP fallback behavior when dependencies missing"""
    print("\n" + "="*60)
    print("TEST 6: WARP Fallback Behavior")
    print("="*60)
    
    from warp_bridge import WARPBridge
    
    bridge = WARPBridge(lazy_init=True)
    
    # Try a search (should gracefully handle missing WARP deps)
    response = bridge.search("test query", k=5)
    
    print(f"  Query: {response.query}")
    print(f"  Results: {response.total_results}")
    print(f"  Execution Time: {response.execution_time_ms:.2f}ms")
    print(f"  Fallback: {response.metadata.get('fallback', False)}")
    
    # Should return valid (possibly empty) response
    assert response.query == "test query", "Query should be preserved"
    print("  ✓ Fallback behavior works correctly")
    return True


def run_all_tests():
    """Run all integration tests"""
    print("\n" + "#"*60)
    print("#  BIZRA WARP INTEGRATION TEST SUITE")
    print("#"*60)
    
    start = time.time()
    
    tests = [
        ("Config", test_warp_config),
        ("Import", test_warp_bridge_import),
        ("Bridge Init", test_warp_bridge_init),
        ("Nexus Adapter", test_nexus_warp_adapter),
        ("Hypergraph Mode", test_hypergraph_warp_mode),
        ("Fallback", test_warp_fallback),
    ]
    
    results = []
    for name, test_fn in tests:
        try:
            passed = test_fn()
            results.append((name, passed))
        except Exception as e:
            print(f"  ✗ Exception: {e}")
            results.append((name, False))
    
    elapsed = time.time() - start
    
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, p in results if p)
    total = len(results)
    
    for name, p in results:
        status = "✓ PASS" if p else "✗ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  Total: {passed}/{total} passed in {elapsed:.2f}s")
    
    if passed == total:
        print("\n  🎉 ALL TESTS PASSED!")
    else:
        print(f"\n  ⚠️  {total - passed} test(s) failed")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
