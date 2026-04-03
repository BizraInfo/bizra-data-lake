#!/usr/bin/env python3
"""
BIZRA Chaos & Stress Test (Phase 2 Hardening)
Target: Rust Elite Service (Optionally Python Kernel)
Focus:
  1. Agent Fusion (MasterReasoner handling Strat+Creative)
  2. SAPE L1 Cache (Latency reduction on repeats)
  3. Deterministic Proofs (Math correctness)
"""

import requests
import time
import statistics
import json
import sys
import random

# Configuration
BASE_URL = "http://localhost:8080"
ENDPOINT = "/dual/execute"
AUTH_TOKEN = "NGbMcElfs1h--S8XfV7jmw7Waa9ElfUCAqH8GEzOIK6U4xLPZft4_C3JOz3EY3E9" # Retrieved from .env
HEADERS = {
    "Authorization": f"Bearer {AUTH_TOKEN}",
    "Content-Type": "application/json"
}

def banner(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")

def run_request(task, intent="chaos_test"):
    payload = {
        "user_id": "chaos_tester_v1",
        "target": "pat_sat_process",  # Based on src/types.rs
        "task": task,
        "requirements": [],
        "context": {
            "intent": intent,
            "chaos_mode": "true"
        }
    }
    start = time.time()
    try:
        # TIMEOUT increased to 180s for CPU inference compatibility
        resp = requests.post(f"{BASE_URL}{ENDPOINT}", json=payload, headers=HEADERS, timeout=180)
        duration = (time.time() - start) * 1000
        return resp.status_code, duration, resp.json() 
    except requests.exceptions.Timeout:
        return 408, (time.time() - start) * 1000, {"error": "Request Timed Out"}
    except Exception as e:
        return 0, (time.time() - start) * 1000, {"error": str(e)}

def run_request_with_id(task, request_id, intent="chaos_test"):
    payload = {
        "user_id": "chaos_tester_v1",
        "target": "pat_sat_process",
        "task": task,
        "requirements": [],
        "context": {
            "intent": intent,
            "chaos_mode": "true",
            "request_id": request_id
        }
    }
    start = time.time()
    try:
        resp = requests.post(f"{BASE_URL}{ENDPOINT}", json=payload, headers=HEADERS, timeout=180)
        duration = (time.time() - start) * 1000
        return resp.status_code, duration, resp.json()
    except requests.exceptions.Timeout:
        return 408, (time.time() - start) * 1000, {"error": "Request Timed Out"}
    except Exception as e:
        return 0, (time.time() - start) * 1000, {"error": str(e)}

def test_cache_velocity():
    banner("TEST 1: SAPE L1 Cache & Idempotency Velocity")
    prompt = "Verify the primality of 65537 and explain its significance in RSA."
    
    latencies = []
    print(f"Hammering endpoint with 5 identical requests (Shared Request ID)...")
    
    # Use fixed request_id to trigger Idempotency Cache (simulating instant replay)
    # AND to test SAPE cache on identical content
    fixed_id = "cache_test_" + str(random.randint(1000, 9999))
    
    for i in range(1, 6):
        # Manually construct payload with fixed ID
        payload = {
            "user_id": "chaos_tester_v1",
            "target": "pat_sat_process",
            "task": prompt,
            "requirements": [],
            "context": {
                "intent": "chaos_test",
                "chaos_mode": "true",
                "request_id": fixed_id
            }
        }
        
        start = time.time()
        try:
            resp = requests.post(f"{BASE_URL}{ENDPOINT}", json=payload, headers=HEADERS, timeout=180)
            ms = (time.time() - start) * 1000
            status = resp.status_code
            if status == 409:
                # Request still in progress (idempotency). Wait briefly and retry once.
                time.sleep(2)
                retry_start = time.time()
                resp = requests.post(f"{BASE_URL}{ENDPOINT}", json=payload, headers=HEADERS, timeout=180)
                ms = (time.time() - retry_start) * 1000
                status = resp.status_code
        except Exception as e:
            ms = (time.time() - start) * 1000
            status = 0
            
        latencies.append(ms)
        marker = "⚡ CACHE HIT" if ms < 2000 else "🐢 MISS" 
        if i == 1: marker = "🐢 COLD START"
        print(f"Req #{i:02}: {status} | {ms:6.1f}ms | {marker}")
        
    avg_speed = statistics.mean(latencies)
    hit_speed = statistics.mean(latencies[1:]) if len(latencies) > 1 else 0
    improvement = (latencies[0] - hit_speed) / latencies[0] * 100 if latencies[0] > 0 else 0
    
    print(f"\nCold Latency: {latencies[0]:.1f}ms")
    print(f"Warm Latency (Avg): {hit_speed:.1f}ms")
    print(f"Improvement : {improvement:.1f}%")
    
    if improvement > 50:
        print("✅ PASS: L1 Cache is effective (>50% reduction)")
    else:
        print("⚠️ WARN: L1 Cache ineffective or inconsistent")

def test_idempotency_regression():
    banner("TEST 1b: Idempotency Regression (Fixed Request ID)")
    prompt = "Idempotency regression check. Reply with: RESULT=OK."
    fixed_id = "idem_regression_" + str(random.randint(1000, 9999))

    status1, ms1, data1 = run_request_with_id(prompt, fixed_id)
    status2, ms2, data2 = run_request_with_id(prompt, fixed_id)

    if status1 != 200 or status2 != 200:
        print(f"❌ FAIL: Statuses {status1}/{status2}")
        return

    key1 = data1.get("meta", {}).get("idempotency_key")
    key2 = data2.get("meta", {}).get("idempotency_key")

    print(f"First: {ms1:.1f}ms | Second: {ms2:.1f}ms")
    if key1 and key1 == key2 and ms2 < 2000:
        print("✅ PASS: Idempotency replay is fast and stable")
    else:
        print("⚠️ WARN: Idempotency replay slower or key mismatch")

def test_agent_fusion():
    banner("TEST 2: Agent Fusion & Synergy")
    # Prompt requiring Strategy (MasterReasoner) AND Creative (Old CreativeSynthesizer)
    prompt = "Propose a strategic roadmap for expanding BIZRA into 2030, but write it as a cyberpunk manifesto poem."
    
    status, ms, data = run_request(prompt)
    
    if status != 200:
        print(f"❌ FAIL: Request failed with {status}")
        return

    pat_agents = data.get('meta', {}).get('pat_agents', 0)
    print(f"Active Agents: {pat_agents}")
    
    # We expect FEWER agents now (11/12 -> maybe fewer active per request, but total pool is fused)
    # The Fusion updated `agents` vec in pat.rs to remove CreativeSynthesizer
    # So we expect 'MasterReasoner' to likely handle this, or at least no error about missing agents.
    
    contributions = data.get('pat_contributions', [])
    print(f"Contributions: {len(contributions)}")
    for c in contributions:
        # Check snippet length as proxy for quality
        print(f" - {len(c)} chars")

    if pat_agents <= 6: # 7 original PATs, fused to 6.
        print("✅ PASS: Fusion detected (Agent count <= 6)")
    else:
        print(f"ℹ️ NOTE: Agent count {pat_agents} (Check if run-time fusion is active)")

def test_deterministic_math():
    banner("TEST 3: Deterministic Math Proofs (Chaos Edge Cases)")
    
    # F_5 = 2^32 + 1 = 4294967297 = 641 * 6700417 (Composite)
    # Many LLMs hallucinate this as prime because it's a Fermat number.
    prompt_f5 = (
        "Is 4294967297 a prime number? Verify deterministically. "
        "Reply with exactly one line: RESULT=COMPOSITE or RESULT=PRIME."
    )
    
    status, ms, data = run_request(prompt_f5)
    
    full_text = json.dumps(data)
    
    print(f"Response Status: {status}")
    if "RESULT=COMPOSITE" in full_text:
        print("✅ PASS: Correctly identified F5 (4294967297) as COMPOSITE")
    else:
        print(f"❌ FAIL: Failed to identify F5 composition. Resp: {full_text[:200]}...")

    # True Prime: 2^19 - 1 (Mersenne) = 524287
    prompt_m19 = (
        "Is 524287 a prime number? Verify deterministically. "
        "Reply with exactly one line: RESULT=PRIME or RESULT=COMPOSITE."
    )
    status2, ms2, data2 = run_request(prompt_m19)
    full_text2 = json.dumps(data2)
    
    if "RESULT=PRIME" in full_text2:
        print("✅ PASS: Correctly identified M19 (524287) as PRIME")
    else:
        print(f"❌ FAIL: Failed to verify M19. Resp: {full_text2[:200]}...")

if __name__ == "__main__":
    print("WARNING: This test runs against LOCALHOST:8080 (Rust Elite)")
    try:
        test_cache_velocity()
        test_idempotency_regression()
        test_agent_fusion()
        test_deterministic_math()
    except KeyboardInterrupt:
        print("\nTest interrupted.")
