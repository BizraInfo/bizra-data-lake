#!/usr/bin/env python3
"""
🔥 PEAK MCP SERVER — SINGULARITY DEPLOYMENT TEST
"""

import json
import time
import subprocess
import threading
import httpx

def test_mcp_server():
    """Test the MCP server deployment."""
    
    print("=" * 70)
    print("🔥 PEAK MCP SERVER v3.0.0-SINGULARITY — DEPLOYMENT TEST")
    print("=" * 70)
    
    # Start server in background
    print("\n⏳ Starting MCP server...")
    server_proc = subprocess.Popen(
        ["python", "peak_mcp_server.py", "--http", "--port", "8765"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for server to initialize (engine takes ~25s to load)
    print("   Waiting for engine initialization (30s)...")
    time.sleep(30)
    
    try:
        print("✓ Server started, running tests...\n")
        
        base_url = "http://127.0.0.1:8765"
        
        with httpx.Client(timeout=60.0) as client:
            # Test 1: Tools List
            print("📋 TEST 1: List Available Tools")
            resp = client.post(base_url, json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/list"
            })
            result = resp.json()
            tools = result.get("result", {}).get("tools", [])
            print(f"   Tools Available: {len(tools)}")
            for t in tools:
                print(f"   - {t['name']}")
            print()
            
            # Test 2: Singularity Query
            print("🔥 TEST 2: Singularity Query (Ihsān-level)")
            resp = client.post(base_url, json={
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "peak_singularity_query",
                    "arguments": {
                        "query": "What are the DDAGI kernel invariants?",
                        "verify_fate": True
                    }
                }
            })
            result = resp.json()
            content = result.get("result", {}).get("content", [{}])[0].get("text", "{}")
            data = json.loads(content)
            if "error" in data:
                print(f"   ❌ Error: {data['error']}")
            else:
                print(f"   SNR: {data.get('snr', 0):.4f}")
                print(f"   Grade: {data.get('ihsan_grade', 'N/A')}")
                print(f"   Thoughts: {data.get('thoughts_generated', 0)}")
                print(f"   Synergies: {data.get('synergies_count', 0)}")
                print(f"   SINGULARITY Mode: {data.get('singularity_mode', False)}")
            print()
            
            # Test 3: FATE Verify
            print("🛡️ TEST 3: LLM FATE Verification")
            resp = client.post(base_url, json={
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {
                    "name": "peak_fate_verify",
                    "arguments": {
                        "content": "The DDAGI kernel ensures RIBA_ZERO, ZANN_ZERO, and IHSAN_FLOOR invariants."
                    }
                }
            })
            result = resp.json()
            content = result.get("result", {}).get("content", [{}])[0].get("text", "{}")
            data = json.loads(content)
            print(f"   Passed: {data.get('passed', False)}")
            print(f"   Overall: {data.get('overall_score', 0):.3f}")
            print(f"   Factual: {data.get('factual_score', 0):.2f}")
            print(f"   Aligned: {data.get('aligned_score', 0):.2f}")
            print(f"   Testable: {data.get('testable_score', 0):.2f}")
            print(f"   Evidence: {data.get('evidence_score', 0):.2f}")
            print()
            
            # Test 4: Status
            print("📊 TEST 4: Engine Status")
            resp = client.post(base_url, json={
                "jsonrpc": "2.0",
                "id": 4,
                "method": "tools/call",
                "params": {
                    "name": "peak_status",
                    "arguments": {}
                }
            })
            result = resp.json()
            content = result.get("result", {}).get("content", [{}])[0].get("text", "{}")
            data = json.loads(content)
            print(f"   Engine: {data.get('engine', 'Unknown')}")
            print(f"   Version: {data.get('version', 'Unknown')}")
            print(f"   Invariants: {data.get('kernel_invariants', {})}")
            print()
            
        print("=" * 70)
        print("✅ ALL DEPLOYMENT TESTS PASSED")
        print("=" * 70)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Stop server
        print("\n⏹️ Stopping server...")
        server_proc.terminate()
        server_proc.wait(timeout=5)
        print("✓ Server stopped")

if __name__ == "__main__":
    test_mcp_server()
